"""Fleet controller that publishes plans but never mutates physical state."""

from __future__ import annotations

import pandas as pd

from .behavior.driver_response import DriverResponseModel
from .economics import EconomicsModel
from .entities import RequestState, VehicleState
from .matching.candidate_generator import CandidateGenerator, CandidatePolicy
from .matching.sparse_matcher import CandidateEdge
from .matching import balanced_match, price_aware_match, safe_global_match
from .plan_validator import PlanValidator
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
    ):
        self.routing = routing
        self.validator = validator
        self.economics = economics
        self.driver_response = driver_response
        self.candidates = CandidateGenerator(candidate_policy or CandidatePolicy())
        self.passenger_patience_sec = passenger_patience_sec
        self.preassignment_enabled = preassignment_enabled
        self.preassignment_horizon_sec = preassignment_horizon_sec

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
        preassigned = False
        depart_time = now
        vehicle_location = (vehicle.current_lon, vehicle.current_lat, vehicle.current_zone)
        if vehicle.current_leg is not None:
            if not self.preassignment_enabled:
                return None, "vehicle_not_controllable"
            if vehicle.current_leg.planned_end > now + pd.Timedelta(seconds=self.preassignment_horizon_sec):
                return None, "preassignment_horizon"
            if vehicle.vehicle_id in state_reserved_vehicle_ids_placeholder():
                return None, "vehicle_already_reserved"
            release_buffer_sec = 30.0
            depart_time = vehicle.current_leg.planned_end + pd.Timedelta(seconds=release_buffer_sec)
            vehicle_location = (vehicle.current_leg.end_lon, vehicle.current_leg.end_lat, req.destination_zone if vehicle.current_request_id == req.order_id else vehicle.current_zone)
            preassigned = True
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
        econ = self.economics.evaluate(req, vehicle, pickup_route.road_distance_m, pickup_route.expected_travel_time_sec, (now - req.request_time).total_seconds(), strategy)
        if econ.passenger_gc > self.economics.passenger_gc_cap:
            return None, "passenger_gc_cap"
        if vehicle.vehicle_type == "HV":
            response = self.driver_response.evaluate_offer(vehicle, req, econ, now, service_end)
            if response.response.value != "ACCEPT":
                return None, f"driver_{response.reason}"
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
                "fare_revenue": econ.fare_revenue,
                "driver_payout": econ.driver_payout,
                "pickup_variable_cost": econ.pickup_variable_cost,
                "service_variable_cost": econ.service_variable_cost,
                "preassigned": preassigned,
                "safe_release_time": str(depart_time) if preassigned else "",
                "release_buffer_sec": max(0.0, (depart_time - (vehicle.current_leg.planned_end if vehicle.current_leg else now)).total_seconds()) if preassigned else 0.0,
                "buffer_source": "global_q90_validation_residual_fallback" if preassigned else "",
            },
        )
        return edge, ""

    def _candidate_vehicles(self, state: SystemState, now: pd.Timestamp) -> list[VehicleState]:
        return self.candidates.controllable_vehicles(state)

    def choose_edges(self, edges: list[CandidateEdge], strategy: str):
        if strategy == "Safe GlobalMatch-MinPickup":
            return safe_global_match.solve(edges)
        if strategy == "ODD-Gated Price-Aware":
            return price_aware_match.solve(edges)
        if strategy == "Three-Stakeholder Balanced":
            return balanced_match.solve(edges)
        raise ValueError(strategy)

    def publish_assignment_plans(self, state: SystemState, chosen: list[CandidateEdge], now: pd.Timestamp) -> list:
        plans = []
        for edge in chosen:
            req = state.requests[edge.request_id]
            vehicle = state.vehicles[edge.vehicle_id]
            pickup_arrival = now + pd.Timedelta(seconds=edge.pickup_eta_sec)
            pickup_departure = pickup_arrival
            dropoff_arrival = pickup_departure + pd.Timedelta(seconds=req.predicted_service_time_sec)
            if edge.metadata.get("preassigned", False):
                plan = insert_request_after_locked_stops(vehicle, req, now, pickup_arrival, pickup_departure, dropoff_arrival, edge.objective, "preassignment")
            else:
                plan = assignment_plan(vehicle, req, now, pickup_arrival, pickup_departure, dropoff_arrival, edge.objective, "matching_assignment")
            plans.append((vehicle, req, plan, edge))
        return plans


def state_reserved_vehicle_ids_placeholder() -> set[str]:
    """Compatibility shim while ReservationManager lives in SimulationEngine."""
    return set()
