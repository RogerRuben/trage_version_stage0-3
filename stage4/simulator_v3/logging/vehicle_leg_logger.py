"""Vehicle leg log helpers."""

from __future__ import annotations

from ..entities import VehicleLeg, VehicleState


def leg_to_record(leg: VehicleLeg, vehicle: VehicleState) -> dict:
    return {
        "leg_id": leg.leg_id,
        "vehicle_id": leg.vehicle_id,
        "vehicle_type": vehicle.vehicle_type,
        "leg_type": leg.leg_type.value,
        "request_id": leg.request_id or "",
        "plan_version": leg.plan_version,
        "start_time": str(leg.actual_start) if leg.actual_start is not None else str(leg.planned_start),
        "end_time": str(leg.actual_end) if leg.actual_end is not None else "",
        "start_lon": leg.start_lon,
        "start_lat": leg.start_lat,
        "end_lon": leg.end_lon,
        "end_lat": leg.end_lat,
        "distance_m": leg.distance_m,
        "expected_time_sec": leg.expected_travel_time_sec,
        "realized_time_sec": leg.realized_travel_time_sec,
        "route_source": leg.route_source,
        "odd_feasible": leg.odd_feasible,
        "termination_reason": leg.termination_reason or "completed",
    }

