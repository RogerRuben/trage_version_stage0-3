"""Patience-aware sparse rolling OR control with optional S4 decision layers."""

from __future__ import annotations

import time
from math import isfinite
from typing import Any

import pandas as pd

from stage4.fleetpy_adapter.native_fleet_control import _NativeFleetControlCore
from stage4.fleetpy_adapter.upstream import FleetPyBindings

from .acceptance import passenger_acceptance
from .candidate_graph import (
    SparseCandidateIndex,
    SparseValhallaMatrixAdapter,
    SpatialVehicle,
    search_radius_m,
)
from .exposure import (
    CumulativeExposureState,
    ExposureExcess,
    exposure_excess,
    parse_gammas,
)
from .gate_diagnostics import (
    empty_gate_counts,
    evidence_contract_complete,
    structural_reason,
    validate_gate_counts,
)
from .solver import AssignmentArc, solve_lexicographic


class RollingRuntimeGuardExceeded(RuntimeError):
    """Raised at a dispatch boundary so the runner can persist partial diagnostics."""


def patience_feasible(pickup_eta_s: float, remaining_patience_s: float) -> bool:
    return isfinite(pickup_eta_s) and pickup_eta_s <= remaining_patience_s


def patience_expired(simulation_time_s: int, pickup_deadline_s: int) -> bool:
    return int(simulation_time_s) >= int(pickup_deadline_s)


class _RollingORFleetControlCore(_NativeFleetControlCore):
    def __init__(
        self,
        bindings: FleetPyBindings,
        vehicles: list[Any],
        requests: list[Any],
        demand: Any,
        network: Any,
        eta_adapter: SparseValhallaMatrixAdapter,
        start: pd.Timestamp,
        end: pd.Timestamp,
        config: dict[str, Any],
        repositioning_manager: Any | None = None,
    ) -> None:
        super().__init__(
            bindings, vehicles, requests, demand, network, eta_adapter, start, end
        )
        self.config = config
        self.dispatch_interval_s = int(config["dispatch_interval_s"])
        self.max_pickup_wait_s = int(config["max_pickup_wait_s"])
        self.matching_end_s = int(config["matching_end_s"])
        self.runtime_guard_s = float(config["benchmark_runtime_guard_s"])
        self.run_started_perf = time.perf_counter()
        self.request_meta: dict[int, dict[str, Any]] = {}
        self.expired_rids: set[int] = set()
        self.epoch_rows: list[dict[str, Any]] = []
        self.exposure_rows: list[dict[str, Any]] = []
        self.candidate_generation_time_s = 0.0
        self.solver_time_s = 0.0
        self.acceptance_rate = float(config.get("passenger_acceptance_rate", 1.0))
        self.acceptance_seed = int(config.get("passenger_acceptance_seed", 0))
        self.gammas = parse_gammas(config)
        self.exposure_state = CumulativeExposureState()
        self.eta_cost_av_to_hv = float(config.get("eta_cost_av_to_hv", 1.0))
        if not isfinite(self.eta_cost_av_to_hv) or self.eta_cost_av_to_hv < 0.0:
            raise ValueError("eta_cost_av_to_hv must be finite and >= 0")
        self.cost_level_enabled = bool(config.get("cost_level_enabled", False))
        self.pickup_cost_epsilon = float(config.get("pickup_cost_epsilon", 0.0))
        self.solver_tolerance = float(config.get("solver_numerical_tolerance", 1e-7))
        self.av_candidates_pruned_by_acceptance = 0
        self.av_candidates_pruned_by_missing_exposure = 0
        self.cost_level_solve_count = 0
        self.prospective_gate_logging = bool(
            config.get("prospective_gate_logging", False)
        )
        self.repositioning_manager = repositioning_manager

    def record_tick(self, simulation_time: int) -> None:
        """Avoid retaining an O(ticks x fleet) trace; epoch aggregates are enough."""
        del simulation_time

    def user_request(self, rq: Any, simulation_time: int) -> None:
        rid = int(rq.get_rid_struct())
        request = self.request_by_rid[rid]
        self.rq_dict[rid] = rq
        self.activation_rows.append(
            {
                "native_request_id": rid,
                "order_id": request.order_id,
                "historical_request_time": request.request_time,
                "native_activation_time": self._timestamp(simulation_time),
                "activation_lag_s": float(simulation_time - request.sim_time_s),
            }
        )
        offer_time = (
            request.predicted_service_time_s
            if isfinite(request.predicted_service_time_s)
            else 0.0
        )
        self.offers[rid] = self.bindings.traveller_offer(
            rid, self.op_id, 0.0, offer_time, 0
        )
        acceptance = passenger_acceptance(
            request.order_id, self.acceptance_rate, self.acceptance_seed
        )
        request.passenger_accepts_av = acceptance.passenger_accepts_av
        request.acceptance_source = acceptance.acceptance_source
        exposure = exposure_excess(
            request.rho_static, request.rho_dynamic, request.rho_speed
        )
        self.request_meta[rid] = {
            "first_attempt_time": None,
            "attempt_count": 0,
            "failed_round_count": 0,
            "carry_over_flag": False,
            "pickup_deadline_s": int(request.sim_time_s + self.max_pickup_wait_s),
            "final_search_radius_m": float(self.config["search_radius_initial_m"]),
            "entered_critical": False,
            "passenger_accepts_av": acceptance.passenger_accepts_av,
            "acceptance_source": acceptance.acceptance_source,
            "exposure": exposure,
        }

    def _expire(self, rid: int, simulation_time: int) -> None:
        self.expired_rids.add(rid)
        self.cancelled_rids.add(rid)
        self.demand.waiting_rq.pop(rid, None)
        self.demand.rq_db.pop(rid, None)
        self.rq_dict.pop(rid, None)

    def _spatial_vehicle(self, runtime: Any) -> SpatialVehicle:
        lon, lat = self.routing_engine.return_position_coordinates(
            runtime.native_vehicle.pos
        )
        return SpatialVehicle(
            runtime.fixture.vehicle_id,
            int(runtime.fixture.native_id),
            runtime.fixture.vehicle_type,
            float(lon),
            float(lat),
        )

    def time_trigger(self, simulation_time: int) -> None:
        self.sim_time = int(simulation_time)
        if self.repositioning_manager is not None:
            self.repositioning_manager.before_normal_dispatch(simulation_time)
        if time.perf_counter() - self.run_started_perf > self.runtime_guard_s:
            raise RollingRuntimeGuardExceeded(
                "bounded benchmark runtime guard exceeded"
            )
        if simulation_time > self.matching_end_s:
            return
        epoch_started = time.perf_counter()
        waiting_ids = sorted(
            (
                int(rid)
                for rid in self.demand.waiting_rq
                if int(rid) not in self.rid_to_assigned_vid
            ),
            key=lambda rid: (self.request_by_rid[rid].request_time, rid),
        )
        for rid in list(waiting_ids):
            if patience_expired(
                simulation_time, self.request_meta[rid]["pickup_deadline_s"]
            ):
                self._expire(rid, simulation_time)
                waiting_ids.remove(rid)
        available = [
            runtime
            for runtime in self.runtime_by_vid.values()
            if self._available(runtime, simulation_time)
        ]
        spatial = [self._spatial_vehicle(runtime) for runtime in available]
        index = SparseCandidateIndex(spatial)
        by_vid = {int(runtime.fixture.native_id): runtime for runtime in available}
        timestamp = self._timestamp(simulation_time)
        arcs: list[AssignmentArc] = []
        spatial_pairs = 0
        topk_pairs = 0
        invalid_patience = 0
        invalid_hv_window = 0
        invalid_cost_evidence = 0
        critical_count = 0
        pruned_acceptance_epoch = 0
        pruned_exposure_epoch = 0
        gate_counts = empty_gate_counts() if self.prospective_gate_logging else None
        routing_before = self.eta_adapter.routing_time_s
        for rid in waiting_ids:
            request = self.request_by_rid[rid]
            meta = self.request_meta[rid]
            if meta["first_attempt_time"] is None:
                meta["first_attempt_time"] = timestamp
            meta["attempt_count"] += 1
            remaining = float(meta["pickup_deadline_s"] - simulation_time)
            critical = 0 < remaining <= self.dispatch_interval_s
            meta["entered_critical"] = bool(meta["entered_critical"] or critical)
            critical_count += int(critical)
            radius = search_radius_m(
                meta["failed_round_count"],
                self.config["search_radius_initial_m"],
                self.config["search_radius_step_m"],
                self.config["search_radius_cap_m"],
            )
            meta["final_search_radius_m"] = radius
            exposure: ExposureExcess | None = meta["exposure"]
            av_ready = bool(request.av_smoke_eligible)
            accepts = bool(meta["passenger_accepts_av"])
            if gate_counts is not None:
                nearby_av = index.count_vehicle_type_within(
                    request.pickup_lon_wgs84,
                    request.pickup_lat_wgs84,
                    radius,
                    "AV",
                )
                gate_counts["gate_av_n0_spatial"] += nearby_av
                if accepts:
                    gate_counts["gate_av_n1_passenger_compatible"] += nearby_av
                    reason = structural_reason(request)
                    if reason == "NO_SELECTED_ROUTE":
                        gate_counts["gate_av_loss_no_selected_route"] += nearby_av
                    elif reason == "HARD_INFEASIBLE":
                        gate_counts["gate_av_loss_hard_infeasible"] += nearby_av
                    elif reason == "HARD_UNKNOWN":
                        gate_counts["gate_av_loss_hard_unknown"] += nearby_av
                    else:
                        gate_counts["gate_av_n2_structurally_ready"] += nearby_av
                        if evidence_contract_complete(request, exposure):
                            gate_counts["gate_av_n3_evidence_complete"] += nearby_av
                        else:
                            gate_counts[
                                "gate_av_loss_evidence_incomplete"
                            ] += nearby_av
            av_eligible = av_ready and accepts and exposure is not None
            if av_ready and not accepts:
                pruned_acceptance_epoch += index.count_vehicle_type_within(
                    request.pickup_lon_wgs84,
                    request.pickup_lat_wgs84,
                    radius,
                    "AV",
                )
            elif av_ready and accepts and exposure is None:
                pruned_exposure_epoch += index.count_vehicle_type_within(
                    request.pickup_lon_wgs84,
                    request.pickup_lat_wgs84,
                    radius,
                    "AV",
                )
            candidates, raw_count = index.query(
                request.pickup_lon_wgs84,
                request.pickup_lat_wgs84,
                radius,
                int(self.config["candidate_top_k"]),
                av_eligible,
            )
            spatial_pairs += raw_count
            topk_pairs += len(candidates)
            estimates = self.eta_adapter.estimate_many(
                [item[0] for item in candidates],
                request.pickup_lon_wgs84,
                request.pickup_lat_wgs84,
                timestamp,
            )
            for vehicle, _distance in candidates:
                is_av = vehicle.vehicle_type == "AV"
                if gate_counts is not None and is_av:
                    gate_counts["gate_av_n3a_shared_topk"] += 1
                estimate = estimates.get(vehicle.native_vehicle_id)
                if estimate is None:
                    continue
                if gate_counts is not None and is_av:
                    gate_counts["gate_av_n3b_route_returned"] += 1
                if not patience_feasible(estimate.corrected_pickup_eta_s, remaining):
                    invalid_patience += 1
                    continue
                if gate_counts is not None and is_av:
                    gate_counts["gate_av_n4_pickup_within_patience"] += 1
                runtime = by_vid[vehicle.native_vehicle_id]
                predicted = float(request.predicted_service_time_s)
                if vehicle.vehicle_type == "HV":
                    if not isfinite(predicted):
                        invalid_hv_window += 1
                        continue
                    predicted_end = (
                        simulation_time + estimate.corrected_pickup_eta_s + predicted
                    )
                    if predicted_end > self._fixture_seconds(
                        runtime.fixture.availability_end_time
                    ):
                        invalid_hv_window += 1
                        self.hv_session_end_exclusions.add(
                            (simulation_time, vehicle.native_vehicle_id, rid)
                        )
                        continue
                elif self.cost_level_enabled and not isfinite(predicted):
                    invalid_cost_evidence += 1
                    continue
                factor = self.eta_cost_av_to_hv if vehicle.vehicle_type == "AV" else 1.0
                operating_cost = (
                    factor * (estimate.corrected_pickup_eta_s + predicted)
                    if isfinite(predicted)
                    else 0.0
                )
                arc_exposure = exposure if vehicle.vehicle_type == "AV" else None
                arcs.append(
                    AssignmentArc(
                        vehicle.native_vehicle_id,
                        rid,
                        estimate.corrected_pickup_eta_s,
                        critical,
                        bool(meta["carry_over_flag"]),
                        (runtime, request, estimate),
                        vehicle_type=vehicle.vehicle_type,
                        exposure_static=arc_exposure.static if arc_exposure else 0.0,
                        exposure_dynamic=arc_exposure.dynamic if arc_exposure else 0.0,
                        exposure_speed=arc_exposure.speed if arc_exposure else 0.0,
                        operating_cost=operating_cost,
                    )
                )
                if gate_counts is not None and is_av:
                    gate_counts["gate_av_n5_solver_eligible"] += 1
        candidate_elapsed = time.perf_counter() - epoch_started
        routing_elapsed = self.eta_adapter.routing_time_s - routing_before
        pure_candidate_elapsed = max(0.0, candidate_elapsed - routing_elapsed)
        self.candidate_generation_time_s += pure_candidate_elapsed
        self.av_candidates_pruned_by_acceptance += pruned_acceptance_epoch
        self.av_candidates_pruned_by_missing_exposure += pruned_exposure_epoch
        solver_started = time.perf_counter()
        result = solve_lexicographic(
            arcs,
            exposure_state=self.exposure_state,
            gammas=self.gammas,
            cost_level_enabled=self.cost_level_enabled,
            pickup_cost_epsilon=self.pickup_cost_epsilon,
            numerical_tolerance=self.solver_tolerance,
        )
        solver_elapsed = time.perf_counter() - solver_started
        self.solver_time_s += solver_elapsed
        self.cost_level_solve_count += int(result.cost_level_solved)
        selected_rids: set[int] = set()
        selected_av_exposures: list[ExposureExcess] = []
        for arc_index in result.selected_indices:
            arc = arcs[arc_index]
            runtime, request, estimate = arc.payload
            self._assign(runtime, request, estimate, simulation_time)
            self.assignment_rows[-1].update(
                {
                    "dispatch_policy": "GLOBAL_SPARSE_EXACT_LEXICOGRAPHIC_OR",
                    "passenger_accepts_av": self.request_meta[arc.request_id][
                        "passenger_accepts_av"
                    ],
                    "acceptance_source": self.request_meta[arc.request_id][
                        "acceptance_source"
                    ],
                    "exposure_static": arc.exposure_static,
                    "exposure_dynamic": arc.exposure_dynamic,
                    "exposure_speed": arc.exposure_speed,
                    "normalized_operating_cost": arc.operating_cost,
                }
            )
            if arc.vehicle_type == "AV":
                selected_av_exposures.append(
                    ExposureExcess(
                        arc.exposure_static, arc.exposure_dynamic, arc.exposure_speed
                    )
                )
            selected_rids.add(arc.request_id)
        if gate_counts is not None:
            gate_counts["gate_av_n6_selected"] = len(selected_av_exposures)
            gate_counts["gate_av_loss_acceptance"] = (
                gate_counts["gate_av_n0_spatial"]
                - gate_counts["gate_av_n1_passenger_compatible"]
            )
            gate_counts["gate_av_loss_shared_topk"] = (
                gate_counts["gate_av_n3_evidence_complete"]
                - gate_counts["gate_av_n3a_shared_topk"]
            )
            gate_counts["gate_av_loss_routing_failure"] = (
                gate_counts["gate_av_n3a_shared_topk"]
                - gate_counts["gate_av_n3b_route_returned"]
            )
            gate_counts["gate_av_loss_patience"] = (
                gate_counts["gate_av_n3b_route_returned"]
                - gate_counts["gate_av_n4_pickup_within_patience"]
            )
            gate_counts["gate_av_loss_other_arc_condition"] = (
                gate_counts["gate_av_n4_pickup_within_patience"]
                - gate_counts["gate_av_n5_solver_eligible"]
            )
            gate_counts["gate_av_loss_dispatch_competition"] = (
                gate_counts["gate_av_n5_solver_eligible"]
                - gate_counts["gate_av_n6_selected"]
            )
            validate_gate_counts(gate_counts)
        epoch_excess = self.exposure_state.update(selected_av_exposures)
        self.exposure_state.validate(self.gammas, self.solver_tolerance)
        for rid in waiting_ids:
            if rid not in selected_rids:
                meta = self.request_meta[rid]
                meta["carry_over_flag"] = True
                meta["failed_round_count"] += 1
        epoch_row = {
                "simulation_time_s": int(simulation_time),
                "timestamp": timestamp,
                "waiting_orders": len(waiting_ids),
                "critical_orders": critical_count,
                "available_vehicles": len(available),
                "candidate_spatial_pairs": spatial_pairs,
                "candidate_topk_pairs": topk_pairs,
                "valid_or_arcs": len(arcs),
                "patience_arc_exclusions": invalid_patience,
                "hv_window_arc_exclusions": invalid_hv_window,
                "cost_evidence_arc_exclusions": invalid_cost_evidence,
                "av_candidates_pruned_by_acceptance": pruned_acceptance_epoch,
                "av_candidates_pruned_by_missing_exposure": pruned_exposure_epoch,
                "enabled_gamma_constraint_count": result.enabled_gamma_constraint_count,
                "cost_level_solved": result.cost_level_solved,
                "matched": len(result.selected_indices),
                "critical_matched": result.critical_matched,
                "carry_over_matched": result.carry_over_matched,
                "candidate_generation_time_s": pure_candidate_elapsed,
                "routing_time_s": routing_elapsed,
                "solver_time_s": solver_elapsed,
                "solver_backend": result.backend,
            }
        if gate_counts is not None:
            epoch_row.update(gate_counts)
        self.epoch_rows.append(epoch_row)
        n_av = self.exposure_state.av_assignments
        self.exposure_rows.append(
            {
                "simulation_time_s": int(simulation_time),
                "av_assignments_this_epoch": len(selected_av_exposures),
                "cumulative_av_assignments": n_av,
                "static_excess_this_epoch": epoch_excess.static,
                "dynamic_excess_this_epoch": epoch_excess.dynamic,
                "speed_excess_this_epoch": epoch_excess.speed,
                "cumulative_static_excess": self.exposure_state.static,
                "cumulative_dynamic_excess": self.exposure_state.dynamic,
                "cumulative_speed_excess": self.exposure_state.speed,
                "cumulative_mean_static_excess": self.exposure_state.mean("static"),
                "cumulative_mean_dynamic_excess": self.exposure_state.mean("dynamic"),
                "cumulative_mean_speed_excess": self.exposure_state.mean("speed"),
                "gamma_static": self.gammas["static"],
                "gamma_dynamic": self.gammas["dynamic"],
                "gamma_speed": self.gammas["speed"],
            }
        )
        if self.repositioning_manager is not None:
            self.repositioning_manager.after_normal_dispatch(self, simulation_time)

    def reconcile(self) -> None:
        super().reconcile()
        if self.repositioning_manager is not None:
            self.repositioning_manager.finalize(self.sim_time)


def create_rolling_or_fleet_control(
    bindings: FleetPyBindings,
    vehicles: list[Any],
    requests: list[Any],
    demand: Any,
    network: Any,
    eta_adapter: SparseValhallaMatrixAdapter,
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: dict[str, Any],
    repositioning_manager: Any | None = None,
) -> Any:
    fleet_control_class = type(
        "Stage4RollingORFleetControl",
        (_RollingORFleetControlCore, bindings.fleet_control_base),
        {},
    )
    return fleet_control_class(
        bindings,
        vehicles,
        requests,
        demand,
        network,
        eta_adapter,
        start,
        end,
        config,
        repositioning_manager,
    )
