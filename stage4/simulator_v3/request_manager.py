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
    def transition(self, request: RequestState, new_status: RequestStatus, now: pd.Timestamp, reason: str = "") -> None:
        if new_status not in LEGAL_TRANSITIONS[request.status]:
            raise ValueError(f"Illegal request transition {request.order_id}: {request.status} -> {new_status}")
        request.status = new_status
        if new_status == RequestStatus.CANCELLED:
            request.cancellation_time = now
            request.cancellation_reason = reason

    def reveal(self, state: SystemState, order_id: str, now: pd.Timestamp) -> None:
        req = state.requests[order_id]
        self.transition(req, RequestStatus.PENDING, now)
        state.pending_request_ids.add(order_id)

    def assign(self, state: SystemState, request: RequestState, vehicle_id: str, now: pd.Timestamp) -> None:
        if request.status in {RequestStatus.PENDING, RequestStatus.OFFERED, RequestStatus.RESERVED, RequestStatus.CANDIDATE_SELECTED}:
            request.status = RequestStatus.ASSIGNED
        else:
            self.transition(request, RequestStatus.ASSIGNED, now)
        request.assigned_vehicle_id = vehicle_id
        request.assignment_time = now
        state.pending_request_ids.discard(request.order_id)

    def cancel_expired(self, state: SystemState, now: pd.Timestamp) -> int:
        cancelled = 0
        for oid in list(state.pending_request_ids):
            req = state.requests[oid]
            if now > req.latest_pickup_time:
                self.transition(req, RequestStatus.CANCELLED, now, "PATIENCE_TIMEOUT")
                state.pending_request_ids.discard(oid)
                state.cancelled_request_ids.add(oid)
                cancelled += 1
        return cancelled

