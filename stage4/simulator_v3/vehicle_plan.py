"""Helpers for building vehicle plans."""

from __future__ import annotations

import pandas as pd

from .entities import PlanStop, RequestState, VehiclePlan, VehicleState
from .enums import StopType


def assignment_plan(
    vehicle: VehicleState,
    request: RequestState,
    current_time: pd.Timestamp,
    pickup_arrival: pd.Timestamp,
    pickup_departure: pd.Timestamp,
    dropoff_arrival: pd.Timestamp,
    objective_value: float,
    trigger: str,
) -> VehiclePlan:
    version = vehicle.plan_version + 1
    stops = [
        PlanStop(
            stop_id=f"{vehicle.vehicle_id}:{version}:pickup:{request.order_id}",
            stop_type=StopType.PICKUP,
            request_id=request.order_id,
            lon=request.origin_lon,
            lat=request.origin_lat,
            zone=request.origin_zone,
            earliest_start=None,
            latest_start=request.latest_pickup_time,
            planned_arrival=pickup_arrival,
            planned_departure=pickup_departure,
            locked=False,
        ),
        PlanStop(
            stop_id=f"{vehicle.vehicle_id}:{version}:dropoff:{request.order_id}",
            stop_type=StopType.DROP_OFF,
            request_id=request.order_id,
            lon=request.destination_lon,
            lat=request.destination_lat,
            zone=request.destination_zone,
            earliest_start=None,
            latest_start=None,
            planned_arrival=dropoff_arrival,
            planned_departure=dropoff_arrival,
            locked=False,
        ),
    ]
    return VehiclePlan(
        vehicle_id=vehicle.vehicle_id,
        plan_version=version,
        stops=stops,
        created_time=current_time,
        trigger=trigger,
        feasible=True,
        objective_value=objective_value,
        assigned_request_ids=[request.order_id],
        reserved_request_ids=[],
    )


def insert_request_after_locked_stops(
    vehicle: VehicleState,
    request: RequestState,
    current_time: pd.Timestamp,
    pickup_arrival: pd.Timestamp,
    pickup_departure: pd.Timestamp,
    dropoff_arrival: pd.Timestamp,
    objective_value: float,
    trigger: str,
) -> VehiclePlan:
    """Append a reserved pickup/service pair after the current locked work.

    The executor owns any currently active leg.  The plan only carries future
    stops, so copying ``vehicle.active_plan.stops`` preserves the unfinished
    current service stop if the vehicle is still before drop-off.
    """

    version = vehicle.plan_version + 1
    preserved = list(vehicle.active_plan.stops)
    reserved = [
        PlanStop(
            stop_id=f"{vehicle.vehicle_id}:{version}:reserved_pickup:{request.order_id}",
            stop_type=StopType.PICKUP,
            request_id=request.order_id,
            lon=request.origin_lon,
            lat=request.origin_lat,
            zone=request.origin_zone,
            earliest_start=None,
            latest_start=request.latest_pickup_time,
            planned_arrival=pickup_arrival,
            planned_departure=pickup_departure,
            locked=False,
        ),
        PlanStop(
            stop_id=f"{vehicle.vehicle_id}:{version}:reserved_dropoff:{request.order_id}",
            stop_type=StopType.DROP_OFF,
            request_id=request.order_id,
            lon=request.destination_lon,
            lat=request.destination_lat,
            zone=request.destination_zone,
            earliest_start=None,
            latest_start=None,
            planned_arrival=dropoff_arrival,
            planned_departure=dropoff_arrival,
            locked=False,
        ),
    ]
    assigned = list(vehicle.active_plan.assigned_request_ids)
    reserved_ids = list(vehicle.active_plan.reserved_request_ids)
    if request.order_id not in reserved_ids:
        reserved_ids.append(request.order_id)
    return VehiclePlan(
        vehicle_id=vehicle.vehicle_id,
        plan_version=version,
        stops=preserved + reserved,
        created_time=current_time,
        trigger=trigger,
        feasible=True,
        objective_value=objective_value,
        assigned_request_ids=assigned,
        reserved_request_ids=reserved_ids,
    )
