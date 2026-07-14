"""Event-driven simulation engine with fixed decision epochs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

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
                                self.reservations.complete(req.order_id)
                    else:
                        result = "NO_ACTIVE_LEG"
                elif event.event_type == EventType.DECISION_EPOCH:
                    handler = "SimulationEngine._decision_epoch"
                    self._decision_epoch(event.event_time)
                elif event.event_type == EventType.DRIVER_RESPONSE:
                    handler = "SimulationEngine._driver_response"
                    result = self._driver_response(event.entity_id, event.event_time)
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
        chosen, match_stats = self.controller.choose_edges(edges, self.strategy)
        plans = self.controller.publish_assignment_plans(self.state, chosen, now)
        for vehicle, req, plan, edge in plans:
            old_plan = vehicle.active_plan
            if edge.metadata.get("preassigned", False):
                if vehicle.vehicle_type == "HV":
                    self.requests.offer(self.state, req, vehicle.vehicle_id, now, trigger="PREASSIGNMENT_OFFER")
                    offer_id = f"{req.order_id}:{vehicle.vehicle_id}:{req.offer_round}"
                    response_time = now + pd.Timedelta(seconds=10)
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
                    self.reservations.create(req.order_id, vehicle.vehicle_id, now, req.latest_pickup_time)
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
                response_time = now + pd.Timedelta(seconds=10)
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
        if not offer.get("preassigned", False) and (vehicle.current_leg is not None or vehicle.execution_status != VehicleExecutionStatus.WAITING):
            self.requests.reject_offer(self.state, req, vehicle.vehicle_id, now, "vehicle_no_longer_waiting")
            return "VEHICLE_NOT_WAITING"
        if offer.get("preassigned", False):
            if vehicle.vehicle_id in self.reservations.vehicle_to_request:
                self.requests.reject_offer(self.state, req, vehicle.vehicle_id, now, "vehicle_already_reserved")
                return "VEHICLE_ALREADY_RESERVED"
            self.requests.reserve(self.state, req, vehicle.vehicle_id, now)
            vehicle.reserved_request_id = req.order_id
            self.reservations.create(req.order_id, vehicle.vehicle_id, now, req.latest_pickup_time)
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
        offers = 0
        for vehicle in candidate_vehicles:
            if offers >= max_offers:
                break
            release_lon = float(vehicle.current_leg.end_lon)
            release_lat = float(vehicle.current_leg.end_lat)
            coarse: list[tuple[float, object]] = []
            for req in pending:
                dx = (float(req.origin_lon) - release_lon) * 91_000.0
                dy = (float(req.origin_lat) - release_lat) * 111_000.0
                dist = float(np.hypot(dx, dy))
                if dist <= 6_000.0:
                    coarse.append((dist, req))
            coarse.sort(key=lambda item: item[0])
            for _, req in coarse[:40]:
                if req.status != RequestStatus.PENDING:
                    continue
                release_time = vehicle.current_leg.planned_end + pd.Timedelta(seconds=30)
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
                econ = self.controller.economics.evaluate(
                    req,
                    vehicle,
                    pickup_route.road_distance_m,
                    pickup_route.expected_travel_time_sec,
                    (now - req.request_time).total_seconds(),
                    self.strategy,
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
                        "release_buffer_sec": 30.0,
                        "buffer_source": "global_q90_validation_residual_fallback",
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
                self.requests.offer(self.state, req, vehicle.vehicle_id, now, trigger="PREASSIGNMENT_OFFER")
                offer_id = f"{req.order_id}:{vehicle.vehicle_id}:{req.offer_round}"
                response_time = now + pd.Timedelta(seconds=10)
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
        if self.idle_manager is not None:
            pd.DataFrame(self.idle_manager.records).to_parquet(out_dir / "idle_movement_log.parquet", index=False, compression="zstd")
        (out_dir / "summary.json").write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")

    def summary(self) -> dict:
        total = len(self.state.requests)
        completed = sum(r.status == RequestStatus.COMPLETED for r in self.state.requests.values())
        cancelled = sum(r.status == RequestStatus.CANCELLED for r in self.state.requests.values())
        av_completed = sum(r.status == RequestStatus.COMPLETED and r.assigned_vehicle_id and self.state.vehicles[r.assigned_vehicle_id].vehicle_type == "AV" for r in self.state.requests.values())
        return {
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
