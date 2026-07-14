"""Event-driven simulation engine with fixed decision epochs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .enums import EventType, RequestStatus, VehicleExecutionStatus
from .event_queue import EventQueue
from .fleet_controller import FleetController
from .logging.plan_revision_logger import plan_revision_record
from .logging.request_logger import request_to_record
from .logging.system_epoch_logger import epoch_record
from .logging.vehicle_leg_logger import leg_to_record
from .request_manager import RequestManager
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
        self.position_mutation_entrypoints = {"VehicleExecutor.complete_current_leg"}

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

    def run(self, end_time: pd.Timestamp) -> dict:
        while len(self.events):
            event = self.events.pop()
            if event.event_time > end_time:
                break
            self.state.current_time = event.event_time
            if event.event_type == EventType.REQUEST_REVEALED:
                self.requests.reveal(self.state, event.entity_id, event.event_time)
            elif event.event_type == EventType.HV_SESSION_START:
                v = self.state.vehicles[event.entity_id]
                if v.current_leg is None:
                    v.execution_status = VehicleExecutionStatus.IDLE
            elif event.event_type == EventType.HV_SESSION_END:
                v = self.state.vehicles[event.entity_id]
                if v.current_leg is None:
                    v.execution_status = VehicleExecutionStatus.OFFLINE
            elif event.event_type == EventType.LEG_COMPLETED:
                vehicle = self.state.vehicles[event.entity_id]
                leg = self.executor.complete_current_leg(vehicle, self.state.requests, event.event_time)
                if leg is not None:
                    self.vehicle_leg_records.append(leg_to_record(leg, vehicle))
                    if leg.request_id:
                        req = self.state.requests[leg.request_id]
                        if req.status == RequestStatus.COMPLETED:
                            self.state.completed_request_ids.add(req.order_id)
            elif event.event_type == EventType.DECISION_EPOCH:
                self._decision_epoch(event.event_time)
        for oid in list(self.state.pending_request_ids):
            req = self.state.requests[oid]
            req.status = RequestStatus.CANCELLED
            req.cancellation_time = self.state.current_time
            req.cancellation_reason = "END_OF_DAY"
            self.state.cancelled_request_ids.add(oid)
            self.state.pending_request_ids.discard(oid)
        self.request_log_records = [request_to_record(r) for r in self.state.requests.values()]
        return self.summary()

    def _decision_epoch(self, now: pd.Timestamp) -> None:
        expired = self.requests.cancel_expired(self.state, now)
        edges, edge_stats = self.controller.build_edges(self.state, now, self.strategy)
        chosen, match_stats = self.controller.choose_edges(edges, self.strategy)
        plans = self.controller.publish_assignment_plans(self.state, chosen, now)
        for vehicle, req, plan, edge in plans:
            old_version = vehicle.plan_version
            self.requests.assign(self.state, req, vehicle.vehicle_id, now)
            req.first_offer_time = req.first_offer_time or now
            req.offer_round += 1
            vehicle.cumulative_income += edge.metadata.get("driver_payout", 0.0)
            leg = self.executor.publish_plan(vehicle, plan, self.state.requests, now)
            self.plan_revision_records.append(plan_revision_record(old_version, plan, now, "PASS"))
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
        }))

    def configure_strategy(self, strategy: str) -> None:
        self.strategy = strategy

    def write_outputs(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.request_log_records).to_parquet(out_dir / "request_log.parquet", index=False, compression="zstd")
        pd.DataFrame(self.vehicle_leg_records).to_parquet(out_dir / "vehicle_leg_log.parquet", index=False, compression="zstd")
        pd.DataFrame(self.plan_revision_records).to_parquet(out_dir / "plan_revision_log.parquet", index=False, compression="zstd")
        pd.DataFrame(self.offer_records).to_parquet(out_dir / "offer_log.parquet", index=False, compression="zstd")
        pd.DataFrame(self.system_epoch_records).to_parquet(out_dir / "system_epoch_log.parquet", index=False, compression="zstd")
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

