"""Event-driven simulation engine with fixed decision epochs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .economy_ledger import (
    ScenarioCostParameters,
    assignment_ledger_row,
    audit_ledger,
    build_scenario_ledger,
)
from .entities import VehiclePlan
from .enums import EventType, RequestStatus, VehicleExecutionStatus
from .event_queue import EventQueue
from .fleet_controller import FleetController
from .logging.plan_revision_logger import plan_revision_record
from .logging.request_logger import request_to_record
from .logging.system_epoch_logger import epoch_record
from .logging.vehicle_leg_logger import leg_to_record
from .request_manager import RequestManager
from .preassignment.reservation_manager import ReservationManager
from .idle_management import IdleMovementManager
from .system_state import SystemState
from .vehicle_executor import VehicleExecutor


class SimulationEngine:
    def __init__(
        self,
        state: SystemState,
        event_queue: EventQueue,
        controller: FleetController,
        executor: VehicleExecutor,
        request_manager: RequestManager,
        decision_epoch_sec: int = 30,
        idle_manager: IdleMovementManager | None = None,
        scenario_cost_parameters: ScenarioCostParameters | None = None,
    ):
        self.state = state
        self.events = event_queue
        self.controller = controller
        self.executor = executor
        self.requests = request_manager
        self.decision_epoch_sec = decision_epoch_sec
        self.request_log_records: list[dict] = []
        self.vehicle_leg_records: list[dict] = []
        self.plan_revision_records: list[dict] = []
        self.offer_records: list[dict] = []
        self.system_epoch_records: list[dict] = []
        self.event_execution_records: list[dict] = []
        self.position_mutation_entrypoints = {"VehicleExecutor.complete_current_leg"}
        self._event_counter = 0
        self.pending_offers: dict[str, dict] = {}
        self.reservations = ReservationManager()
        self.idle_manager = idle_manager
        self.scenario_cost_parameters = scenario_cost_parameters or ScenarioCostParameters()
        self.economy_ledger_records: list[dict] = []
        self.economy_audit: dict = {}

    def initialize_events(self, start: pd.Timestamp, end: pd.Timestamp) -> None:
        for req in self.state.requests.values():
            self.events.push(req.request_time, EventType.REQUEST_REVEALED, req.order_id)
        for vehicle in self.state.vehicles.values():
            self.events.push(vehicle.online_start, EventType.HV_SESSION_START if vehicle.vehicle_type == "HV" else EventType.HV_SESSION_START, vehicle.vehicle_id)
            self.events.push(vehicle.online_end, EventType.HV_SESSION_END if vehicle.vehicle_type == "HV" else EventType.HV_SESSION_END, vehicle.vehicle_id)
        t = start.floor(f"{self.decision_epoch_sec}s")
        while t <= end:
            self.events.push(t, EventType.DECISION_EPOCH, "fleet_controller")
            t += pd.Timedelta(seconds=self.decision_epoch_sec)

    def run(self, end_time: pd.Timestamp, finalize: bool = True) -> dict:
        while len(self.events) and self.events.peek_time() is not None and self.events.peek_time() <= end_time:
            event = self.events.pop()
            self.state.current_time = event.event_time
            self._event_counter += 1
            handled = True
            result = "OK"
            handler = ""
            try:
                if event.event_type == EventType.REQUEST_REVEALED:
                    handler = "RequestManager.reveal"
                    self.requests.reveal(self.state, event.entity_id, event.event_time)
                elif event.event_type == EventType.HV_SESSION_START:
                    handler = "SystemState.set_vehicle_status"
                    v = self.state.vehicles[event.entity_id]
                    if v.current_leg is None:
                        self.state.set_vehicle_status(v.vehicle_id, VehicleExecutionStatus.IDLE)
                elif event.event_type == EventType.HV_SESSION_END:
                    handler = "SystemState.set_vehicle_status"
                    v = self.state.vehicles[event.entity_id]
                    if v.current_leg is None:
                        self.state.set_vehicle_status(v.vehicle_id, VehicleExecutionStatus.OFFLINE)
                    else:
                        result = "ACTIVE_LOCKED_TASK"
                elif event.event_type == EventType.LEG_COMPLETED:
                    handler = "VehicleExecutor.complete_current_leg"
                    vehicle = self.state.vehicles[event.entity_id]
                    leg = self.executor.complete_current_leg(vehicle, self.state.requests, event.event_time)
                    if leg is not None:
                        self.vehicle_leg_records.append(leg_to_record(leg, vehicle))
                        if leg.request_id:
                            req = self.state.requests[leg.request_id]
                            if req.status == RequestStatus.COMPLETED:
                                self.state.completed_request_ids.add(req.order_id)
                                self.state.reserved_request_ids.discard(req.order_id)
                                self.reservations.complete(req.order_id, event.event_time)
                        if vehicle.current_leg is None and vehicle.reserved_request_id is not None:
                            reserved_request_id = vehicle.reserved_request_id
                            validation_result = self._revalidate_vehicle_reservation(vehicle.vehicle_id, event.event_time)
                            if validation_result == "PASS":
                                # The reservation has now been consumed into the
                                # executable plan.  Close it before the executor
                                # clears ``reserved_request_id`` so the scheduled
                                # expiry event becomes a harmless stale event.
                                self.reservations.complete(reserved_request_id, event.event_time)
                                self.executor.start_next_leg(vehicle, self.state.requests, event.event_time)
                            else:
                                result = validation_result
                    else:
                        result = "NO_ACTIVE_LEG"
                elif event.event_type == EventType.DECISION_EPOCH:
                    handler = "SimulationEngine._decision_epoch"
                    self._decision_epoch(event.event_time)
                elif event.event_type == EventType.DRIVER_RESPONSE:
                    handler = "SimulationEngine._driver_response"
                    result = self._driver_response(event.entity_id, event.event_time)
                elif event.event_type == EventType.RESERVATION_EXPIRED:
                    handler = "SimulationEngine._reservation_expired"
                    result = self._reservation_expired(event.entity_id, event.event_time)
                elif event.event_type == EventType.PLAN_INVALIDATED:
                    handler = "SimulationEngine._plan_invalidated"
                    result = self._plan_invalidated(event.entity_id, event.event_time, str(event.payload.get("reason", "PLAN_INVALIDATED")))
                else:
                    handled = False
                    result = "NO_HANDLER"
            except Exception as exc:
                handled = False
                result = f"ERROR:{type(exc).__name__}:{exc}"
                raise
            finally:
                self.event_execution_records.append({
                    "event_sequence": self._event_counter,
                    "event_time": str(event.event_time),
                    "event_priority": event.priority,
                    "event_type": event.event_type.value,
                    "entity_id": event.entity_id,
                    "event_payload": json.dumps(event.payload, sort_keys=True, default=str),
                    "handled": handled,
                    "handler": handler,
                    "result": result,
                })
        if finalize:
            terminal_candidates = set(self.state.pending_request_ids) | set(self.state.offered_request_ids) | set(self.state.reserved_request_ids)
            for oid in sorted(terminal_candidates):
                req = self.state.requests[oid]
                if req.status in {RequestStatus.COMPLETED, RequestStatus.CANCELLED}:
                    continue
                self.requests.transition(req, RequestStatus.CANCELLED, self.state.current_time, reason="END_OF_DAY", trigger="END_OF_DAY")
                self.state.cancelled_request_ids.add(oid)
                self.state.pending_request_ids.discard(oid)
                self.state.offered_request_ids.discard(oid)
                self.state.reserved_request_ids.discard(oid)
                if req.reserved_vehicle_id:
                    self.reservations.release(req.order_id, self.state.current_time, "END_OF_DAY")
            self.request_log_records = [request_to_record(r) for r in self.state.requests.values()]
        return self.summary()

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_checkpoint(path: Path) -> "SimulationEngine":
        with path.open("rb") as f:
            return pickle.load(f)

    def _decision_epoch(self, now: pd.Timestamp) -> None:
        expired = self.requests.cancel_expired(self.state, now)
        idle_moves = 0
        if self.idle_manager is not None:
            hv_decisions = self.idle_manager.build_plans(
                [self.state.vehicles[vid] for vid in sorted(self.state.idle_hv_ids)],
                now,
                "HV",
            )
            av_decisions = self.idle_manager.build_plans(
                [self.state.vehicles[vid] for vid in sorted(self.state.idle_av_ids)],
                now,
                "AV",
            )
            for decision in hv_decisions + av_decisions:
                old_plan = decision.vehicle.active_plan
                self.executor.publish_plan(decision.vehicle, decision.plan, self.state.requests, now)
                self.plan_revision_records.append(plan_revision_record(old_plan, decision.plan, now, "PASS"))
                idle_moves += 1
        edges, edge_stats = self.controller.build_edges(self.state, now, self.strategy)
        chosen, match_stats = self.controller.choose_edges(edges, self.strategy, self.state, now)
        plans = self.controller.publish_assignment_plans(self.state, chosen, now)
        for vehicle, req, plan, edge in plans:
            old_plan = vehicle.active_plan
            if edge.metadata.get("preassigned", False):
                if vehicle.vehicle_type == "HV":
                    self.requests.offer(self.state, req, vehicle.vehicle_id, now, trigger="PREASSIGNMENT_OFFER")
                    offer_id = f"{req.order_id}:{vehicle.vehicle_id}:{req.offer_round}"
                    response_time = now + pd.Timedelta(
                        seconds=self.controller.driver_response.response_delay_sec
                    )
                    self.pending_offers[offer_id] = {
                        "request_id": req.order_id,
                        "vehicle_id": vehicle.vehicle_id,
                        "plan": plan,
                        "old_plan": old_plan,
                        "edge": edge,
                        "offer_time": now,
                        "response_deadline": response_time,
                        "preassigned": True,
                    }
                    self.events.push(response_time, EventType.DRIVER_RESPONSE, offer_id)
                else:
                    self.requests.reserve(self.state, req, vehicle.vehicle_id, now)
                    vehicle.reserved_request_id = req.order_id
                    self._create_reservation(req, vehicle, edge, now)
                    vehicle.active_plan = plan
                    vehicle.plan_version = plan.plan_version
                    self.plan_revision_records.append(plan_revision_record(old_plan, plan, now, "PASS"))
                    self.offer_records.append({
                        "offer_id": f"{req.order_id}:{vehicle.vehicle_id}:{req.offer_round + 1}",
                        "request_id": req.order_id,
                        "vehicle_id": vehicle.vehicle_id,
                        "offer_time": str(now),
                        "response_deadline": str(now),
                        "response_time": str(now),
                        "driver_utility": edge.driver_utility,
                        "response": "RESERVED",
                        "rejection_reason": "",
                    })
            elif vehicle.vehicle_type == "HV":
                self.requests.offer(self.state, req, vehicle.vehicle_id, now)
                self.state.set_vehicle_status(vehicle.vehicle_id, VehicleExecutionStatus.WAITING)
                offer_id = f"{req.order_id}:{vehicle.vehicle_id}:{req.offer_round}"
                response_time = now + pd.Timedelta(
                    seconds=self.controller.driver_response.response_delay_sec
                )
                self.pending_offers[offer_id] = {
                    "request_id": req.order_id,
                    "vehicle_id": vehicle.vehicle_id,
                    "plan": plan,
                    "old_plan": old_plan,
                    "edge": edge,
                    "offer_time": now,
                    "response_deadline": response_time,
                }
                self.events.push(response_time, EventType.DRIVER_RESPONSE, offer_id)
            else:
                self.requests.assign(self.state, req, vehicle.vehicle_id, now)
                req.first_offer_time = req.first_offer_time or now
                req.offer_round += 1
                vehicle.cumulative_income += edge.metadata.get("driver_payout", 0.0)
                self.executor.publish_plan(vehicle, plan, self.state.requests, now)
                self.plan_revision_records.append(plan_revision_record(old_plan, plan, now, "PASS"))
                self.offer_records.append({
                    "offer_id": f"{req.order_id}:{vehicle.vehicle_id}:{req.offer_round}",
                    "request_id": req.order_id,
                    "vehicle_id": vehicle.vehicle_id,
                    "offer_time": str(now),
                    "response_deadline": str(now),
                    "response_time": str(now),
                    "driver_utility": edge.driver_utility,
                    "response": "ACCEPT",
                    "rejection_reason": "",
                    **self._offer_odd_fields(edge),
                })
        preassignment_offers = 0
        if self.controller.preassignment_enabled:
            preassignment_offers = self._preassignment_epoch(now, max_offers=50)
        self.state.routing_query_count = self.controller.routing.query_count
        self.state.routing_cache_hit_count = self.controller.routing.cache_hit_count
        self.system_epoch_records.append(epoch_record(self.state, now, {
            "expired_requests": expired,
            "candidate_plans": edge_stats.get("coarse_candidate_edges", 0),
            "validated_plans": edge_stats.get("validated_plans", 0),
            "matched_plans": len(chosen),
            "matching_runtime": match_stats.get("matching_runtime_sec", 0.0),
            "candidate_truncation_rate": edge_stats.get("candidate_truncation_rate", 0.0),
            "orders_hitting_candidate_cap": edge_stats.get("orders_hitting_candidate_cap", 0),
            "idle_movement_plans": idle_moves,
            "preassignment_offers": preassignment_offers,
            "stress_constraint_active": match_stats.get("stress_constraint_active", False),
            "zone_service_constraint_active": match_stats.get("zone_service_constraint_active", False),
            "stress_constraint_binding": match_stats.get("stress_constraint_binding", False),
            "zone_service_constraint_binding": match_stats.get("zone_service_constraint_binding", False),
            "zone_service_deficit_total": match_stats.get("zone_service_deficit_total", 0),
            "stress_budget_violation_total": match_stats.get("stress_budget_violation_total", 0.0),
            "balanced_constraint_source": match_stats.get("balanced_constraint_source", ""),
            "balanced_constraint_table_hash": match_stats.get("balanced_constraint_table_hash", ""),
            "remaining_stress_budget_total": match_stats.get("remaining_stress_budget_total", 0.0),
            "minimum_zone_service_target_total": match_stats.get("minimum_zone_service_target_total", 0),
            "served_zone_count_total": match_stats.get("served_zone_count_total", 0),
            "pending_order_count_in_constraint_table": match_stats.get("pending_order_count_in_constraint_table", 0),
            "maximum_cardinality_before_constraints": match_stats.get("maximum_cardinality_before_constraints", 0),
            "stress_replacement_count": match_stats.get("stress_replacement_count", 0),
            "stress_cardinality_reduction_count": match_stats.get("stress_cardinality_reduction_count", 0),
            "zone_fairness_swap_count": match_stats.get("zone_fairness_swap_count", 0),
            "constraint_relaxation_used": match_stats.get("constraint_relaxation_used", False),
            "balanced_price_aware_equivalent": match_stats.get("price_aware_equivalent", True),
            "balanced_edge_objective": match_stats.get("edge_objective", ""),
            "balanced_constraint_model": match_stats.get("constraint_model", ""),
        }))

    def _driver_response(self, offer_id: str, now: pd.Timestamp) -> str:
        offer = self.pending_offers.pop(offer_id, None)
        if offer is None:
            return "STALE_OFFER"
        req = self.state.requests[offer["request_id"]]
        vehicle = self.state.vehicles[offer["vehicle_id"]]
        edge = offer["edge"]
        if req.status != RequestStatus.OFFERED:
            self.state.set_vehicle_status(vehicle.vehicle_id, VehicleExecutionStatus.IDLE if vehicle.online_start <= now <= vehicle.online_end else VehicleExecutionStatus.OFFLINE)
            return f"REQUEST_NOT_OFFERED:{req.status.value}"
        if now >= req.latest_pickup_time:
            self.requests.reject_offer(self.state, req, vehicle.vehicle_id, now, "response_after_pickup_deadline")
            if not offer.get("preassigned", False):
                self.state.set_vehicle_status(
                    vehicle.vehicle_id,
                    VehicleExecutionStatus.IDLE if vehicle.online_start <= now <= vehicle.online_end else VehicleExecutionStatus.OFFLINE,
                )
            self.offer_records.append({
                "offer_id": offer_id,
                "request_id": req.order_id,
                "vehicle_id": vehicle.vehicle_id,
                "offer_time": str(offer["offer_time"]),
                "response_deadline": str(offer["response_deadline"]),
                "response_time": str(now),
                "driver_utility": edge.driver_utility,
                "response": "TIMEOUT",
                "rejection_reason": "response_after_pickup_deadline",
                **self._offer_odd_fields(edge),
            })
            return "TIMEOUT"
        if not offer.get("preassigned", False) and (vehicle.current_leg is not None or vehicle.execution_status != VehicleExecutionStatus.WAITING):
            self.requests.reject_offer(self.state, req, vehicle.vehicle_id, now, "vehicle_no_longer_waiting")
            return "VEHICLE_NOT_WAITING"
        service_end = offer["plan"].stops[-1].planned_arrival if offer["plan"].stops else now
        response = self.controller.driver_response.evaluate_delayed_response(
            vehicle,
            edge.driver_utility,
            service_end,
        )
        if response.response.value != "ACCEPT":
            self.requests.reject_offer(self.state, req, vehicle.vehicle_id, now, response.reason or response.response.value.lower())
            if not offer.get("preassigned", False):
                self.state.set_vehicle_status(
                    vehicle.vehicle_id,
                    VehicleExecutionStatus.IDLE if vehicle.online_start <= now <= vehicle.online_end else VehicleExecutionStatus.OFFLINE,
                )
            self.offer_records.append({
                "offer_id": offer_id,
                "request_id": req.order_id,
                "vehicle_id": vehicle.vehicle_id,
                "offer_time": str(offer["offer_time"]),
                "response_deadline": str(offer["response_deadline"]),
                "response_time": str(now),
                "driver_utility": edge.driver_utility,
                "response": response.response.value,
                "rejection_reason": response.reason,
                **self._offer_odd_fields(edge),
            })
            return response.response.value
        if offer.get("preassigned", False):
            # The vehicle may finish its locked task during the asynchronous
            # HV response delay.  The insertion plan was built against the old
            # locked leg and must not be installed after that leg disappears.
            # Return the request to pending; the now-idle vehicle can be paired
            # by the normal matcher at the next decision epoch.
            if vehicle.current_leg is None:
                self.requests.reject_offer(
                    self.state,
                    req,
                    vehicle.vehicle_id,
                    now,
                    "vehicle_released_before_preassignment_response",
                )
                self.offer_records.append({
                    "offer_id": offer_id,
                    "request_id": req.order_id,
                    "vehicle_id": vehicle.vehicle_id,
                    "offer_time": str(offer["offer_time"]),
                    "response_deadline": str(offer["response_deadline"]),
                    "response_time": str(now),
                    "driver_utility": edge.driver_utility,
                    "response": "REJECT",
                    "rejection_reason": "vehicle_released_before_preassignment_response",
                    **self._offer_odd_fields(edge),
                })
                return "VEHICLE_RELEASED_BEFORE_PREASSIGNMENT_RESPONSE"
            if vehicle.vehicle_id in self.reservations.vehicle_to_request:
                self.requests.reject_offer(self.state, req, vehicle.vehicle_id, now, "vehicle_already_reserved")
                return "VEHICLE_ALREADY_RESERVED"
            self.requests.reserve(self.state, req, vehicle.vehicle_id, now)
            vehicle.reserved_request_id = req.order_id
            self._create_reservation(req, vehicle, edge, now)
            vehicle.active_plan = offer["plan"]
            vehicle.plan_version = offer["plan"].plan_version
        else:
            self.requests.assign(self.state, req, vehicle.vehicle_id, now)
            vehicle.cumulative_income += edge.metadata.get("driver_payout", 0.0)
            self.executor.publish_plan(vehicle, offer["plan"], self.state.requests, now)
        self.plan_revision_records.append(plan_revision_record(offer["old_plan"], offer["plan"], now, "PASS"))
        self.offer_records.append({
            "offer_id": offer_id,
            "request_id": req.order_id,
            "vehicle_id": vehicle.vehicle_id,
            "offer_time": str(offer["offer_time"]),
            "response_deadline": str(offer["response_deadline"]),
            "response_time": str(now),
            "driver_utility": edge.driver_utility,
            "response": "ACCEPT",
            "rejection_reason": "",
            **self._offer_odd_fields(edge),
        })
        return "ACCEPT"

    def _preassignment_epoch(self, now: pd.Timestamp, max_offers: int = 50) -> int:
        if int(now.timestamp()) % 60 != 0:
            return 0
        active_offer_vehicles = {offer["vehicle_id"] for offer in self.pending_offers.values()}
        candidate_vehicles = [
            self.state.vehicles[vid]
            for vid in sorted(self.state.service_vehicle_ids)
            if self.state.vehicles[vid].current_leg is not None
            and self.state.vehicles[vid].reserved_request_id is None
            and vid not in self.reservations.vehicle_to_request
            and vid not in active_offer_vehicles
            and self.state.vehicles[vid].current_leg.planned_end <= now + pd.Timedelta(seconds=self.controller.preassignment_horizon_sec)
        ]
        if not candidate_vehicles:
            return 0
        pending = [self.state.requests[oid] for oid in sorted(self.state.pending_request_ids) if self.state.requests[oid].status == RequestStatus.PENDING]
        if not pending:
            return 0
        pending_coords = np.radians(np.asarray(
            [[float(req.origin_lat), float(req.origin_lon)] for req in pending], dtype=float
        ))
        pending_tree = BallTree(pending_coords, metric="haversine")
        offers = 0
        for vehicle in candidate_vehicles:
            if offers >= max_offers:
                break
            release_lon = float(vehicle.current_leg.end_lon)
            release_lat = float(vehicle.current_leg.end_lat)
            idx, _ = pending_tree.query_radius(
                np.radians([[release_lat, release_lon]]),
                r=6_000.0 / 6_371_000.0,
                return_distance=True,
                sort_results=True,
            )
            for pending_index in idx[0][:40]:
                req = pending[int(pending_index)]
                if req.status != RequestStatus.PENDING:
                    continue
                resolver = self.controller.safe_release_resolver
                if resolver is None:
                    raise RuntimeError("Preassignment requires a validation residual resolver")
                resolution = resolver.resolve(
                    vehicle.current_leg.planned_end,
                    int(req.metadata.get("time_bin", 0)),
                    vehicle.current_zone,
                    str(req.metadata.get("stress_bucket", "unknown")),
                )
                release_time = max(now, resolution.safe_release_time)
                pickup_route = self.controller.routing.query_pickup_route(
                    (vehicle.current_leg.end_lon, vehicle.current_leg.end_lat, vehicle.current_zone),
                    (req.origin_lon, req.origin_lat, req.origin_zone),
                    release_time,
                    vehicle.vehicle_type,
                    int(req.metadata.get("time_bin", 0)),
                )
                pickup_arrival = release_time + pd.Timedelta(seconds=pickup_route.expected_travel_time_sec)
                if pickup_arrival > req.latest_pickup_time:
                    continue
                service_end = pickup_arrival + pd.Timedelta(seconds=req.predicted_service_time_sec)
                validation = self.controller.validator.validate(vehicle, req, now, pickup_arrival, service_end, allow_preassignment=True)
                if not validation.feasible:
                    continue
                capability = (
                    self.controller.validator.service_odd.capability_rows.get(str(req.order_id), {})
                    if vehicle.vehicle_type == "AV" else {}
                )
                econ = self.controller.economics.evaluate(
                    req,
                    vehicle,
                    pickup_route.road_distance_m,
                    pickup_route.expected_travel_time_sec,
                    (now - req.request_time).total_seconds(),
                    self.strategy,
                    capability_cost=float(capability.get("capability_cost", 0.0) or 0.0),
                    remote_assistance_cost=float(capability.get("remote_assistance_cost", 0.0) or 0.0),
                )
                from .matching.sparse_matcher import CandidateEdge
                from .vehicle_plan import insert_request_after_locked_stops

                edge = CandidateEdge(
                    request_id=req.order_id,
                    vehicle_id=vehicle.vehicle_id,
                    pickup_eta_sec=pickup_route.expected_travel_time_sec,
                    marginal_contribution=econ.marginal_operating_contribution,
                    passenger_gc=econ.passenger_gc,
                    driver_utility=econ.driver_utility,
                    stress=econ.stress,
                    objective=econ.marginal_operating_contribution,
                    metadata={
                        "vehicle_type": vehicle.vehicle_type,
                        "origin_zone": req.origin_zone,
                        "preassigned": True,
                        "pickup_distance_m": pickup_route.road_distance_m,
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
                        **resolution.to_metadata(),
                        "raw_safe_release_time": str(resolution.safe_release_time),
                        "safe_release_time": str(release_time),
                        "safe_release_clamped_to_now": bool(release_time != resolution.safe_release_time),
                    },
                )
                plan = insert_request_after_locked_stops(
                    vehicle,
                    req,
                    now,
                    pickup_arrival,
                    pickup_arrival,
                    pickup_arrival + pd.Timedelta(seconds=req.predicted_service_time_sec),
                    edge.objective,
                    "preassignment",
                )
                old_plan = vehicle.active_plan
                req.metadata.update({
                    "assigned_vehicle_type": vehicle.vehicle_type,
                    "pickup_odd_feasible": bool(validation.pickup_odd_feasible),
                    "service_odd_feasible": bool(validation.service_odd_feasible),
                    "combined_odd_feasible": bool(validation.pickup_odd_feasible and validation.service_odd_feasible),
                    "capability_profile": validation.capability_profile,
                    "capability_mapping_version": validation.capability_mapping_version,
                    "selected_edge_economics": {
                        key: float(edge.metadata.get(key, 0.0) or 0.0)
                        for key in [
                            "fare_revenue", "driver_payout", "pickup_variable_cost",
                            "service_variable_cost", "capability_cost",
                            "remote_assistance_cost", "platform_variable_cost",
                        ]
                    },
                })
                if vehicle.vehicle_type == "AV":
                    # AV reservations are platform-controlled and therefore do
                    # not fabricate a driver response event.  Physical state is
                    # unchanged until the current locked service completes.
                    self.requests.reserve(self.state, req, vehicle.vehicle_id, now)
                    vehicle.reserved_request_id = req.order_id
                    self._create_reservation(req, vehicle, edge, now)
                    vehicle.active_plan = plan
                    vehicle.plan_version = plan.plan_version
                    self.plan_revision_records.append(plan_revision_record(old_plan, plan, now, "PASS"))
                    self.offer_records.append({
                        "offer_id": f"{req.order_id}:{vehicle.vehicle_id}:{req.offer_round}",
                        "request_id": req.order_id,
                        "vehicle_id": vehicle.vehicle_id,
                        "offer_time": str(now),
                        "response_deadline": str(now),
                        "response_time": str(now),
                        "driver_utility": edge.driver_utility,
                        "response": "RESERVED",
                        "rejection_reason": "",
                        **self._offer_odd_fields(edge),
                    })
                    active_offer_vehicles.add(vehicle.vehicle_id)
                    offers += 1
                    break

                self.requests.offer(self.state, req, vehicle.vehicle_id, now, trigger="PREASSIGNMENT_OFFER")
                offer_id = f"{req.order_id}:{vehicle.vehicle_id}:{req.offer_round}"
                response_time = now + pd.Timedelta(seconds=self.controller.driver_response.response_delay_sec)
                if response_time >= req.latest_pickup_time:
                    # The offer cannot be accepted before the passenger's
                    # deadline, so leave the request pending for another edge.
                    self.requests.reject_offer(self.state, req, vehicle.vehicle_id, now, "offer_response_after_deadline")
                    continue
                self.pending_offers[offer_id] = {
                    "request_id": req.order_id,
                    "vehicle_id": vehicle.vehicle_id,
                    "plan": plan,
                    "old_plan": old_plan,
                    "edge": edge,
                    "offer_time": now,
                    "response_deadline": response_time,
                    "preassigned": True,
                }
                self.events.push(response_time, EventType.DRIVER_RESPONSE, offer_id)
                active_offer_vehicles.add(vehicle.vehicle_id)
                offers += 1
                break
        return offers

    @staticmethod
    def _offer_odd_fields(edge) -> dict:
        return {
            "pickup_odd_feasible": bool(edge.metadata.get("pickup_odd_feasible", False)),
            "service_odd_feasible": bool(edge.metadata.get("service_odd_feasible", False)),
            "combined_odd_feasible": bool(edge.metadata.get("combined_odd_feasible", False)),
            "capability_profile": str(edge.metadata.get("capability_profile", "")),
            "capability_mapping_version": str(edge.metadata.get("capability_mapping_version", "")),
        }

    def _create_reservation(self, req, vehicle, edge, now: pd.Timestamp) -> None:
        expected_text = edge.metadata.get("expected_release_time") or edge.metadata.get("safe_release_time")
        expected = pd.Timestamp(expected_text) if expected_text else vehicle.current_leg.planned_end
        safe_text = edge.metadata.get("safe_release_time")
        safe = pd.Timestamp(safe_text) if safe_text else expected
        self.reservations.create(
            req.order_id,
            vehicle.vehicle_id,
            now,
            req.latest_pickup_time,
            expected_release_time=expected,
            safe_release_time=safe,
            buffer_source=str(edge.metadata.get("buffer_source", "")),
            buffer_sample_count=int(edge.metadata.get("buffer_sample_count", 0) or 0),
            buffer_quantile=float(edge.metadata.get("buffer_quantile", 0.9) or 0.9),
            residual_quantile_sec=float(edge.metadata.get("release_residual_quantile_sec", 0.0) or 0.0),
        )
        self.events.push(req.latest_pickup_time, EventType.RESERVATION_EXPIRED, req.order_id)

    def _release_reservation_state(self, request_id: str, now: pd.Timestamp, reason: str) -> str:
        req = self.state.requests[request_id]
        vehicle_id = req.reserved_vehicle_id or self.reservations.request_to_vehicle.get(request_id)
        if vehicle_id and vehicle_id in self.state.vehicles:
            vehicle = self.state.vehicles[vehicle_id]
            old_plan = vehicle.active_plan
            kept = [stop for stop in old_plan.stops if str(stop.request_id or "") != str(request_id)]
            new_plan = VehiclePlan(
                vehicle_id=vehicle.vehicle_id,
                plan_version=vehicle.plan_version + 1,
                stops=kept,
                created_time=now,
                trigger=reason,
                feasible=True,
                objective_value=old_plan.objective_value,
                assigned_request_ids=[oid for oid in old_plan.assigned_request_ids if oid != request_id],
                reserved_request_ids=[oid for oid in old_plan.reserved_request_ids if oid != request_id],
            )
            vehicle.active_plan = new_plan
            vehicle.plan_version = new_plan.plan_version
            vehicle.reserved_request_id = None
            self.plan_revision_records.append(plan_revision_record(old_plan, new_plan, now, "PASS"))
        req.reserved_vehicle_id = None
        if req.status == RequestStatus.RESERVED:
            self.requests.transition(req, RequestStatus.PENDING, now, reason=reason, trigger=reason, vehicle_id=vehicle_id)
            self.state.reserved_request_ids.discard(request_id)
            self.state.pending_request_ids.add(request_id)
        req.metadata["reservation_failure_count"] = int(req.metadata.get("reservation_failure_count", 0)) + 1
        req.last_failure_reason = reason
        return reason

    def _reservation_expired(self, request_id: str, now: pd.Timestamp) -> str:
        record = self.reservations.active_for_request(request_id)
        if record is None:
            return "STALE_RESERVATION_EXPIRY"
        if now < record.expiry_time:
            return "EARLY_RESERVATION_EXPIRY_IGNORED"
        self.reservations.expire(request_id, now, "RESERVATION_EXPIRED")
        return self._release_reservation_state(request_id, now, "RESERVATION_EXPIRED")

    def _plan_invalidated(self, request_id: str, now: pd.Timestamp, reason: str) -> str:
        if self.reservations.active_for_request(request_id) is not None:
            self.reservations.invalidate(request_id, now, reason)
        return self._release_reservation_state(request_id, now, reason)

    def _revalidate_vehicle_reservation(self, vehicle_id: str, now: pd.Timestamp) -> str:
        vehicle = self.state.vehicles[vehicle_id]
        request_id = vehicle.reserved_request_id
        if not request_id:
            return "NO_RESERVATION"
        req = self.state.requests[request_id]
        if self.reservations.active_for_request(request_id) is None:
            # Expiry/invalidation may close the authoritative reservation at
            # the same timestamp as the locked current leg completes.  Clear
            # the stale plan/vehicle pointer before any further validation.
            return self._release_reservation_state(
                request_id, now, "STALE_RESERVATION_STATE_RELEASED"
            )
        resolver = self.controller.safe_release_resolver
        if resolver is None:
            self.events.push(now, EventType.PLAN_INVALIDATED, request_id, {"reason": "PLAN_INVALIDATED:NO_VALIDATION_RESIDUAL"})
            return "PLAN_INVALIDATED:NO_VALIDATION_RESIDUAL"
        resolution = resolver.resolve(
            now,
            int(req.metadata.get("time_bin", 0)),
            vehicle.current_zone,
            str(req.metadata.get("stress_bucket", "unknown")),
        )
        safe_release = max(now, resolution.safe_release_time)
        pickup = self.controller.routing.query_pickup_route(
            (vehicle.current_lon, vehicle.current_lat, vehicle.current_zone),
            (req.origin_lon, req.origin_lat, req.origin_zone),
            safe_release,
            vehicle.vehicle_type,
            int(req.metadata.get("time_bin", 0)),
        )
        validation = self.reservations.revalidate(
            request_id,
            now,
            safe_release,
            pickup.expected_travel_time_sec,
            req.latest_pickup_time,
            vehicle_online_end=vehicle.online_end,
            projected_service_time_sec=req.predicted_service_time_sec,
            release_metadata=resolution.to_metadata(),
        )
        if not validation.feasible:
            self.events.push(now, EventType.PLAN_INVALIDATED, request_id, {"reason": validation.reason})
            return validation.reason
        return "PASS"

    def configure_strategy(self, strategy: str) -> None:
        self.strategy = strategy

    def write_outputs(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.request_log_records).to_parquet(out_dir / "request_log.parquet", index=False, compression="zstd")
        pd.DataFrame(self.vehicle_leg_records).to_parquet(out_dir / "vehicle_leg_log.parquet", index=False, compression="zstd")
        pd.DataFrame(self.plan_revision_records).to_parquet(out_dir / "plan_revision_log.parquet", index=False, compression="zstd")
        pd.DataFrame(self.offer_records).to_parquet(out_dir / "offer_log.parquet", index=False, compression="zstd")
        pd.DataFrame(self.requests.transition_records).to_parquet(out_dir / "request_transition_log.parquet", index=False, compression="zstd")
        pd.DataFrame(self.event_execution_records).to_parquet(out_dir / "event_execution_log.parquet", index=False, compression="zstd")
        pd.DataFrame(self.system_epoch_records).to_parquet(out_dir / "system_epoch_log.parquet", index=False, compression="zstd")
        pd.DataFrame([{
            "vehicle_id": vehicle.vehicle_id,
            "vehicle_type": vehicle.vehicle_type,
            "online_start": vehicle.online_start,
            "online_end": vehicle.online_end,
            "final_execution_status": vehicle.execution_status.value,
            "final_lon": vehicle.current_lon,
            "final_lat": vehicle.current_lat,
            "final_zone": vehicle.current_zone,
            "cumulative_pickup_distance_m": vehicle.cumulative_pickup_distance_m,
            "cumulative_service_distance_m": vehicle.cumulative_service_distance_m,
            "cumulative_reposition_distance_m": vehicle.cumulative_reposition_distance_m,
            "cumulative_rebalancing_distance_m": vehicle.cumulative_rebalancing_distance_m,
        } for vehicle in self.state.vehicles.values()]).to_parquet(
            out_dir / "vehicle_state_log.parquet", index=False, compression="zstd"
        )
        if self.idle_manager is not None:
            pd.DataFrame(self.idle_manager.records).to_parquet(out_dir / "idle_movement_log.parquet", index=False, compression="zstd")
        served_rows = []
        for req in self.state.requests.values():
            if req.status != RequestStatus.COMPLETED or not req.assigned_vehicle_id:
                continue
            vehicle = self.state.vehicles[req.assigned_vehicle_id]
            served_rows.append(assignment_ledger_row(
                req.order_id,
                vehicle.vehicle_id,
                vehicle.vehicle_type,
                req.metadata.get("selected_edge_economics", {}),
            ))
        legs = pd.DataFrame(self.vehicle_leg_records)
        av_count = sum(v.vehicle_type == "AV" for v in self.state.vehicles.values())
        depot_count = len({v.depot_id for v in self.state.vehicles.values() if v.vehicle_type == "AV" and v.depot_id})
        ledger = build_scenario_ledger(
            served_rows,
            [r.order_id for r in self.state.requests.values() if r.status == RequestStatus.CANCELLED],
            legs,
            av_count,
            depot_count,
            len(self.reservations.failure_records),
            self.scenario_cost_parameters,
        )
        self.economy_ledger_records = ledger.to_dict("records")
        self.economy_audit = audit_ledger(ledger)
        ledger.to_csv(out_dir / "economy_ledger.csv", index=False)
        pd.DataFrame([record.__dict__ for record in self.reservations.records]).to_parquet(out_dir / "reservation_log.parquet", index=False, compression="zstd")
        pd.DataFrame(self.reservations.failure_records).to_parquet(out_dir / "preassignment_failure_log.parquet", index=False, compression="zstd")
        (out_dir / "summary.json").write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")

    def summary(self) -> dict:
        total = len(self.state.requests)
        completed = sum(r.status == RequestStatus.COMPLETED for r in self.state.requests.values())
        cancelled = sum(r.status == RequestStatus.CANCELLED for r in self.state.requests.values())
        av_completed = sum(r.status == RequestStatus.COMPLETED and r.assigned_vehicle_id and self.state.vehicles[r.assigned_vehicle_id].vehicle_type == "AV" for r in self.state.requests.values())
        summary = {
            "orders": total,
            "completed_orders": completed,
            "cancelled_orders": cancelled,
            "match_rate": completed / total if total else 0.0,
            "av_assignment_share": av_completed / completed if completed else 0.0,
            "routing_query_count": self.controller.routing.query_count,
            "routing_cache_hit_count": self.controller.routing.cache_hit_count,
            "routing_cache_hit_rate": self.controller.routing.cache_hit_rate,
            "vehicle_leg_count": len(self.vehicle_leg_records),
            "plan_revision_count": len(self.plan_revision_records),
            "offer_count": len(self.offer_records),
        }
        if self.economy_audit:
            summary.update({
                "operating_contribution": self.economy_audit.get("operating_contribution", 0.0),
                "scenario_net_profit": self.economy_audit.get("scenario_net_profit", 0.0),
                "economy_audit_status": self.economy_audit.get("economy_audit_status", "FAIL"),
            })
        return summary
