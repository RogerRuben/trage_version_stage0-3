"""Fleet controller that publishes plans but never mutates physical state."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from .behavior.driver_response import DriverResponseModel
from .economics import EconomicsModel
from .entities import RequestState, VehicleState
from .matching.candidate_generator import CandidateGenerator, CandidatePolicy
from .matching.sparse_matcher import CandidateEdge
from .matching import balanced_match, price_aware_match, safe_global_match
from .matching.balanced_match import BalancedEpochState
from .plan_validator import PlanValidator
from .preassignment.safe_release_buffer import SafeReleaseBufferResolver
from .routing_engine import RoutingEngine
from .system_state import SystemState
from .vehicle_plan import assignment_plan
from .vehicle_plan import insert_request_after_locked_stops


class FleetController:
    def __init__(
        self,
        routing: RoutingEngine,
        validator: PlanValidator,
        economics: EconomicsModel,
        driver_response: DriverResponseModel,
        candidate_policy: CandidatePolicy | None = None,
        passenger_patience_sec: float = 480.0,
        preassignment_enabled: bool = False,
        preassignment_horizon_sec: float = 480.0,
        safe_release_resolver: SafeReleaseBufferResolver | None = None,
        balanced_state: BalancedEpochState | None = None,
    ):
        self.routing = routing
        self.validator = validator
        self.economics = economics
        self.driver_response = driver_response
        self.candidates = CandidateGenerator(candidate_policy or CandidatePolicy())
        self.passenger_patience_sec = passenger_patience_sec
        self.preassignment_enabled = preassignment_enabled
        self.preassignment_horizon_sec = preassignment_horizon_sec
        self.safe_release_resolver = safe_release_resolver
        self.balanced_state = balanced_state or BalancedEpochState()

    def _radius_for_request(self, request: RequestState, now: pd.Timestamp) -> float:
        wait = max(0.0, (now - request.request_time).total_seconds())
        if wait < 120:
            return 2_000.0
        if wait < 240:
            return 3_000.0
        if wait < 360:
            return 4_500.0
        return 6_000.0

    def build_edges(self, state: SystemState, now: pd.Timestamp, strategy: str) -> tuple[list[CandidateEdge], dict]:
        pending = [state.requests[oid] for oid in sorted(state.pending_request_ids) if state.requests[oid].status.value == "PENDING"]
        vehicles = self._candidate_vehicles(state, now)
        radius_by_order = {r.order_id: self._radius_for_request(r, now) for r in pending}
        coarse, cand_stats = self.candidates.generate(pending, vehicles, radius_by_order)
        edges: list[CandidateEdge] = []
        validation_failures: dict[str, int] = {}
        stage_used_counts: dict[str, int] = {}
        for req in pending:
            candidates = coarse.get(req.order_id, [])
            selected_edges: list[CandidateEdge] = []
            last_limit = 0
            for label, limit in [
                ("initial_20", self.candidates.policy.initial_candidates),
                ("second_40", self.candidates.policy.second_stage_candidates),
                ("maximum_80", self.candidates.policy.maximum_candidates),
            ]:
                for vehicle in candidates[last_limit: min(limit, len(candidates))]:
                    edge, reason = self._edge_for_vehicle(req, vehicle, now, strategy)
                    if edge is None:
                        validation_failures[reason] = validation_failures.get(reason, 0) + 1
                    else:
                        edge.metadata["candidate_stage_used"] = label
                        selected_edges.append(edge)
                last_limit = min(limit, len(candidates))
                if len(selected_edges) >= self.candidates.policy.minimum_feasible_candidates or last_limit >= len(candidates):
                    stage_used_counts[label] = stage_used_counts.get(label, 0) + 1
                    break
            if selected_edges and req.first_candidate_time is None:
                req.first_candidate_time = now
            edges.extend(selected_edges)
        stats = dict(cand_stats)
        stats["validated_plans"] = len(edges)
        stats["validation_failures"] = validation_failures
        stats["candidate_stage_used_counts"] = stage_used_counts
        return edges, stats

    def _edge_for_vehicle(self, req: RequestState, vehicle: VehicleState, now: pd.Timestamp, strategy: str) -> tuple[CandidateEdge | None, str]:
        # A vehicle with a confirmed future reservation is not available to
        # the ordinary pending-order matcher, even if its current leg happens
        # to complete before the next decision epoch.  Without this guard an
        # idle assignment can overwrite the reserved stops before promotion.
        if vehicle.reserved_request_id is not None:
            return None, "vehicle_already_reserved"
        preassigned = False
        depart_time = now
        vehicle_location = (vehicle.current_lon, vehicle.current_lat, vehicle.current_zone)
        if vehicle.current_leg is not None:
            if not self.preassignment_enabled:
                return None, "vehicle_not_controllable"
            if vehicle.current_leg.planned_end > now + pd.Timedelta(seconds=self.preassignment_horizon_sec):
                return None, "preassignment_horizon"
            if self.safe_release_resolver is None:
                return None, "validation_release_residual_unavailable"
            resolution = self.safe_release_resolver.resolve(
                vehicle.current_leg.planned_end,
                int(req.metadata.get("time_bin", 0)),
                vehicle.current_zone,
                str(req.metadata.get("stress_bucket", "unknown")),
            )
            depart_time = max(now, resolution.safe_release_time)
            vehicle_location = (vehicle.current_leg.end_lon, vehicle.current_leg.end_lat, req.destination_zone if vehicle.current_request_id == req.order_id else vehicle.current_zone)
            preassigned = True
            release_metadata = resolution.to_metadata()
            release_metadata["raw_safe_release_time"] = str(resolution.safe_release_time)
            release_metadata["safe_release_time"] = str(depart_time)
            release_metadata["safe_release_clamped_to_now"] = bool(depart_time != resolution.safe_release_time)
        else:
            release_metadata = {}
        pickup_route = self.routing.query_pickup_route(
            vehicle_location,
            (req.origin_lon, req.origin_lat, req.origin_zone),
            depart_time,
            vehicle.vehicle_type,
            int(req.metadata.get("time_bin", 0)),
        )
        pickup_arrival = depart_time + pd.Timedelta(seconds=pickup_route.expected_travel_time_sec)
        pickup_departure = pickup_arrival
        service_end = pickup_departure + pd.Timedelta(seconds=req.predicted_service_time_sec)
        validation = self.validator.validate(vehicle, req, now, pickup_arrival, service_end, allow_preassignment=preassigned)
        if not validation.feasible:
            return None, validation.failure_reason
        capability = self.validator.service_odd.capability_rows.get(str(req.order_id), {}) if vehicle.vehicle_type == "AV" else {}
        capability_cost = float(capability.get("capability_cost", 0.0) or 0.0)
        remote_assistance_cost = (
            float(capability.get("remote_assistance_cost", 0.0) or 0.0)
            if bool(capability.get("feasible_with_extra_cost", False)) else 0.0
        )
        econ = self.economics.evaluate(
            req,
            vehicle,
            pickup_route.road_distance_m,
            pickup_route.expected_travel_time_sec,
            (now - req.request_time).total_seconds(),
            strategy,
            capability_cost=capability_cost,
            remote_assistance_cost=remote_assistance_cost,
        )
        if econ.passenger_gc > self.economics.passenger_gc_cap:
            return None, "passenger_gc_cap"
        edge = CandidateEdge(
            request_id=req.order_id,
            vehicle_id=vehicle.vehicle_id,
            pickup_eta_sec=pickup_route.expected_travel_time_sec,
            marginal_contribution=econ.marginal_operating_contribution,
            passenger_gc=econ.passenger_gc,
            driver_utility=econ.driver_utility,
            stress=econ.stress,
            objective=-pickup_route.expected_travel_time_sec if strategy == "Safe GlobalMatch-MinPickup" else econ.marginal_operating_contribution,
            metadata={
                "vehicle_type": vehicle.vehicle_type,
                "origin_zone": req.origin_zone,
                "pickup_distance_m": pickup_route.road_distance_m,
                "pickup_route_source": pickup_route.route_source,
                "pickup_odd_feasible": bool(validation.pickup_odd_feasible),
                "service_odd_feasible": bool(validation.service_odd_feasible),
                "combined_odd_feasible": bool(validation.pickup_odd_feasible and validation.service_odd_feasible),
                "capability_profile": validation.capability_profile,
                "capability_mapping_version": validation.capability_mapping_version,
                "fare_revenue": econ.fare_revenue,
                "driver_payout": econ.driver_payout,
                "pickup_variable_cost": econ.pickup_variable_cost,
                "service_variable_cost": econ.service_variable_cost,
                "capability_cost": econ.capability_cost,
                "remote_assistance_cost": econ.remote_assistance_cost,
                "platform_variable_cost": econ.platform_variable_cost,
                "preassigned": preassigned,
                "safe_release_time": str(depart_time) if preassigned else "",
                **release_metadata,
            },
        )
        return edge, ""

    def _candidate_vehicles(self, state: SystemState, now: pd.Timestamp) -> list[VehicleState]:
        return self.candidates.controllable_vehicles(state)

    def choose_edges(self, edges: list[CandidateEdge], strategy: str, state: SystemState | None = None, now: pd.Timestamp | None = None):
        if strategy == "Safe GlobalMatch-MinPickup":
            return safe_global_match.solve(edges)
        if strategy == "ODD-Gated Price-Aware":
            return price_aware_match.solve(edges)
        if strategy == "Three-Stakeholder Balanced":
            if state is None or now is None:
                raise ValueError("Balanced matching requires SystemState and decision time")
            time_bin = int(now.hour * 2 + now.minute // 30)
            pending = [
                state.requests[oid] for oid in sorted(state.pending_request_ids)
                if state.requests[oid].status.value == "PENDING"
            ]
            self.balanced_state.prepare_epoch(pending, time_bin)
            tables = self.balanced_state.tables_for_epoch(time_bin)
            table_hash = hashlib.sha256(
                json.dumps(tables, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            chosen, stats = balanced_match.solve(
                edges,
                constraint_tables=tables,
                passenger_gc_cap=self.economics.passenger_gc_cap,
                hv_utility_min=self.economics.driver_utility_min,
            )
            self.balanced_state.consume(chosen, time_bin)
            stats.update({
                "balanced_time_bin": time_bin,
                "remaining_stress_budget_zone_count": len(tables["remaining_stress_budget"]),
                "minimum_zone_service_target_count": len(tables["minimum_zone_service_target"]),
                "pending_zone_count": len(tables["pending_zone_count"]),
                "balanced_constraint_table_hash": table_hash,
                "remaining_stress_budget_total": float(sum(tables["remaining_stress_budget"].values())),
                "minimum_zone_service_target_total": int(sum(tables["minimum_zone_service_target"].values())),
                "served_zone_count_total": int(sum(tables["served_zone_count"].values())),
                "pending_order_count_in_constraint_table": int(sum(tables["pending_zone_count"].values())),
            })
            return chosen, stats
        raise ValueError(strategy)

    def publish_assignment_plans(self, state: SystemState, chosen: list[CandidateEdge], now: pd.Timestamp) -> list:
        plans = []
        for edge in chosen:
            req = state.requests[edge.request_id]
            vehicle = state.vehicles[edge.vehicle_id]
            # Persist the exact selected edge's ODD decisions and provenance.
            # Request logs are therefore independently auditable even after the
            # candidate graph has been discarded.
            req.metadata.update({
                "assigned_vehicle_type": vehicle.vehicle_type,
                "pickup_odd_feasible": bool(edge.metadata.get("pickup_odd_feasible", False)),
                "service_odd_feasible": bool(edge.metadata.get("service_odd_feasible", False)),
                "combined_odd_feasible": bool(edge.metadata.get("combined_odd_feasible", False)),
                "capability_profile": str(edge.metadata.get("capability_profile", "")),
                "capability_mapping_version": str(edge.metadata.get("capability_mapping_version", "")),
                "selected_edge_economics": {
                    key: float(edge.metadata.get(key, 0.0) or 0.0)
                    for key in [
                        "fare_revenue", "driver_payout", "pickup_variable_cost",
                        "service_variable_cost", "capability_cost",
                        "remote_assistance_cost", "platform_variable_cost",
                    ]
                },
            })
            pickup_arrival = now + pd.Timedelta(seconds=edge.pickup_eta_sec)
            pickup_departure = pickup_arrival
            dropoff_arrival = pickup_departure + pd.Timedelta(seconds=req.predicted_service_time_sec)
            if edge.metadata.get("preassigned", False):
                plan = insert_request_after_locked_stops(vehicle, req, now, pickup_arrival, pickup_departure, dropoff_arrival, edge.objective, "preassignment")
            else:
                plan = assignment_plan(vehicle, req, now, pickup_arrival, pickup_departure, dropoff_arrival, edge.objective, "matching_assignment")
            plans.append((vehicle, req, plan, edge))
        return plans
