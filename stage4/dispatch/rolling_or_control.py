"""Patience-aware sparse rolling OR control on FleetPy-native state."""

from __future__ import annotations

import time
from math import isfinite
from typing import Any

import pandas as pd

from stage4.fleetpy_adapter.native_fleet_control import _NativeFleetControlCore
from stage4.fleetpy_adapter.upstream import FleetPyBindings

from .candidate_graph import (
    SparseCandidateIndex,
    SparseValhallaMatrixAdapter,
    SpatialVehicle,
    search_radius_m,
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
        self.candidate_generation_time_s = 0.0
        self.solver_time_s = 0.0

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
        self.request_meta[rid] = {
            "first_attempt_time": None,
            "attempt_count": 0,
            "failed_round_count": 0,
            "carry_over_flag": False,
            "pickup_deadline_s": int(request.sim_time_s + self.max_pickup_wait_s),
            "final_search_radius_m": float(self.config["search_radius_initial_m"]),
            "entered_critical": False,
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
        critical_count = 0
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
            candidates, raw_count = index.query(
                request.pickup_lon_wgs84,
                request.pickup_lat_wgs84,
                radius,
                int(self.config["candidate_top_k"]),
                request.av_smoke_eligible,
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
                estimate = estimates.get(vehicle.native_vehicle_id)
                if estimate is None:
                    continue
                if not patience_feasible(estimate.corrected_pickup_eta_s, remaining):
                    invalid_patience += 1
                    continue
                runtime = by_vid[vehicle.native_vehicle_id]
                if vehicle.vehicle_type == "HV":
                    if not isfinite(request.predicted_service_time_s):
                        invalid_hv_window += 1
                        continue
                    predicted_end = (
                        simulation_time
                        + estimate.corrected_pickup_eta_s
                        + request.predicted_service_time_s
                    )
                    if predicted_end > self._fixture_seconds(
                        runtime.fixture.availability_end_time
                    ):
                        invalid_hv_window += 1
                        self.hv_session_end_exclusions.add(
                            (simulation_time, vehicle.native_vehicle_id, rid)
                        )
                        continue
                arcs.append(
                    AssignmentArc(
                        vehicle.native_vehicle_id,
                        rid,
                        estimate.corrected_pickup_eta_s,
                        critical,
                        bool(meta["carry_over_flag"]),
                        (runtime, request, estimate),
                    )
                )
        candidate_elapsed = time.perf_counter() - epoch_started
        routing_elapsed = self.eta_adapter.routing_time_s - routing_before
        pure_candidate_elapsed = max(0.0, candidate_elapsed - routing_elapsed)
        self.candidate_generation_time_s += pure_candidate_elapsed
        solver_started = time.perf_counter()
        result = solve_lexicographic(arcs)
        solver_elapsed = time.perf_counter() - solver_started
        self.solver_time_s += solver_elapsed
        selected_rids: set[int] = set()
        for arc_index in result.selected_indices:
            arc = arcs[arc_index]
            runtime, request, estimate = arc.payload
            self._assign(runtime, request, estimate, simulation_time)
            self.assignment_rows[-1][
                "dispatch_policy"
            ] = "GLOBAL_SPARSE_EXACT_LEXICOGRAPHIC_OR"
            selected_rids.add(arc.request_id)
        for rid in waiting_ids:
            if rid not in selected_rids:
                meta = self.request_meta[rid]
                meta["carry_over_flag"] = True
                meta["failed_round_count"] += 1
        self.epoch_rows.append(
            {
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
                "matched": len(result.selected_indices),
                "critical_matched": result.critical_matched,
                "carry_over_matched": result.carry_over_matched,
                "candidate_generation_time_s": pure_candidate_elapsed,
                "routing_time_s": routing_elapsed,
                "solver_time_s": solver_elapsed,
                "solver_backend": result.backend,
            }
        )


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
) -> Any:
    fleet_control_class = type(
        "Stage4RollingORFleetControl",
        (_RollingORFleetControlCore, bindings.fleet_control_base),
        {},
    )
    return fleet_control_class(
        bindings, vehicles, requests, demand, network, eta_adapter, start, end, config
    )
