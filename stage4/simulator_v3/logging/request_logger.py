"""Request log builder."""

from __future__ import annotations

from ..entities import RequestState


def request_to_record(req: RequestState) -> dict:
    return {
        "order_id": req.order_id,
        "request_time": str(req.request_time),
        "first_candidate_time": str(req.first_candidate_time) if req.first_candidate_time is not None else "",
        "first_offer_time": str(req.first_offer_time) if req.first_offer_time is not None else "",
        "assignment_time": str(req.assignment_time) if req.assignment_time is not None else "",
        "pickup_start_time": str(req.pickup_start_time) if req.pickup_start_time is not None else "",
        "boarding_time": str(req.boarding_time) if req.boarding_time is not None else "",
        "service_start_time": str(req.service_start_time) if req.service_start_time is not None else "",
        "dropoff_time": str(req.dropoff_time) if req.dropoff_time is not None else "",
        "cancellation_time": str(req.cancellation_time) if req.cancellation_time is not None else "",
        "final_status": req.status.value,
        "cancellation_reason": req.cancellation_reason or "",
        "dispatch_rounds": req.dispatch_round,
        "offer_rounds": req.offer_round,
        "reserved_vehicle_id": req.reserved_vehicle_id or "",
        "assigned_vehicle_id": req.assigned_vehicle_id or "",
        "condition_available": req.condition_available,
        "last_failure_reason": req.last_failure_reason or "",
    }

