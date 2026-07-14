"""Request lifecycle management."""

from __future__ import annotations

import pandas as pd

from .entities import RequestState
from .enums import RequestStatus
from .system_state import SystemState


LEGAL_TRANSITIONS = {
    RequestStatus.UNREVEALED: {RequestStatus.PENDING, RequestStatus.CANCELLED},
    RequestStatus.PENDING: {RequestStatus.CANDIDATE_SELECTED, RequestStatus.OFFERED, RequestStatus.RESERVED, RequestStatus.ASSIGNED, RequestStatus.CANCELLED, RequestStatus.PICKUP_STARTED},
    RequestStatus.CANDIDATE_SELECTED: {RequestStatus.OFFERED, RequestStatus.ASSIGNED, RequestStatus.PENDING, RequestStatus.CANCELLED},
    RequestStatus.OFFERED: {RequestStatus.ASSIGNED, RequestStatus.RESERVED, RequestStatus.PENDING, RequestStatus.CANCELLED},
    RequestStatus.RESERVED: {RequestStatus.ASSIGNED, RequestStatus.PENDING, RequestStatus.CANCELLED},
    RequestStatus.ASSIGNED: {RequestStatus.PICKUP_PENDING, RequestStatus.PICKUP_STARTED, RequestStatus.CANCELLED},
    RequestStatus.PICKUP_PENDING: {RequestStatus.PICKUP_STARTED, RequestStatus.CANCELLED},
    RequestStatus.PICKUP_STARTED: {RequestStatus.BOARDED, RequestStatus.CANCELLED},
    RequestStatus.BOARDED: {RequestStatus.IN_SERVICE},
    RequestStatus.IN_SERVICE: {RequestStatus.COMPLETED},
    RequestStatus.COMPLETED: set(),
    RequestStatus.CANCELLED: set(),
}


class RequestManager:
    def __init__(self) -> None:
        self.transition_records: list[dict] = []

    def transition(
        self,
        request: RequestState,
        new_status: RequestStatus,
        now: pd.Timestamp,
        reason: str = "",
        trigger: str = "",
        vehicle_id: str | None = None,
        plan_version: int | None = None,
    ) -> None:
        if new_status not in LEGAL_TRANSITIONS[request.status]:
            raise ValueError(f"Illegal request transition {request.order_id}: {request.status} -> {new_status}")
        old = request.status
        request.status = new_status
        self.transition_records.append({
            "order_id": request.order_id,
            "old_status": old.value,
            "new_status": new_status.value,
            "transition_time": str(now),
            "trigger": trigger,
            "vehicle_id": vehicle_id or "",
            "plan_version": str(plan_version) if plan_version is not None else "",
            "reason": reason,
        })
        if new_status == RequestStatus.CANCELLED:
            request.cancellation_time = now
            request.cancellation_reason = reason

    def reveal(self, state: SystemState, order_id: str, now: pd.Timestamp) -> None:
        req = state.requests[order_id]
        self.transition(req, RequestStatus.PENDING, now, trigger="REQUEST_REVEALED")
        state.pending_request_ids.add(order_id)

    def assign(self, state: SystemState, request: RequestState, vehicle_id: str, now: pd.Timestamp) -> None:
        if request.status == RequestStatus.PENDING:
            self.transition(request, RequestStatus.OFFERED, now, trigger="OFFER_CREATED", vehicle_id=vehicle_id)
        if request.status in {RequestStatus.OFFERED, RequestStatus.RESERVED, RequestStatus.CANDIDATE_SELECTED}:
            self.transition(request, RequestStatus.ASSIGNED, now, trigger="ASSIGNMENT_CONFIRMED", vehicle_id=vehicle_id)
        elif request.status != RequestStatus.ASSIGNED:
            self.transition(request, RequestStatus.ASSIGNED, now, trigger="ASSIGNMENT_CONFIRMED", vehicle_id=vehicle_id)
        request.assigned_vehicle_id = vehicle_id
        request.assignment_time = now
        state.pending_request_ids.discard(request.order_id)
        state.offered_request_ids.discard(request.order_id)

    def offer(self, state: SystemState, request: RequestState, vehicle_id: str, now: pd.Timestamp, trigger: str = "OFFER_CREATED") -> None:
        self.transition(request, RequestStatus.OFFERED, now, trigger=trigger, vehicle_id=vehicle_id)
        request.last_offer_time = now
        request.first_offer_time = request.first_offer_time or now
        request.offer_round += 1
        state.pending_request_ids.discard(request.order_id)
        state.offered_request_ids.add(request.order_id)

    def reject_offer(self, state: SystemState, request: RequestState, vehicle_id: str, now: pd.Timestamp, reason: str) -> None:
        self.transition(request, RequestStatus.PENDING, now, reason=reason, trigger="DRIVER_RESPONSE", vehicle_id=vehicle_id)
        request.last_failure_reason = reason
        state.offered_request_ids.discard(request.order_id)
        state.pending_request_ids.add(request.order_id)

    def reserve(self, state: SystemState, request: RequestState, vehicle_id: str, now: pd.Timestamp) -> None:
        if request.status == RequestStatus.PENDING:
            self.transition(request, RequestStatus.OFFERED, now, trigger="PREASSIGNMENT_OFFER", vehicle_id=vehicle_id)
        if request.status == RequestStatus.OFFERED:
            self.transition(request, RequestStatus.RESERVED, now, trigger="RESERVATION_CONFIRMED", vehicle_id=vehicle_id)
        request.reserved_vehicle_id = vehicle_id
        state.pending_request_ids.discard(request.order_id)
        state.offered_request_ids.discard(request.order_id)
        state.reserved_request_ids.add(request.order_id)

    def cancel_expired(self, state: SystemState, now: pd.Timestamp) -> int:
        cancelled = 0
        for oid in sorted(state.pending_request_ids):
            req = state.requests[oid]
            if now > req.latest_pickup_time:
                self.transition(req, RequestStatus.CANCELLED, now, "PATIENCE_TIMEOUT", trigger="PATIENCE_TIMEOUT")
                state.pending_request_ids.discard(oid)
                state.cancelled_request_ids.add(oid)
                cancelled += 1
        return cancelled
