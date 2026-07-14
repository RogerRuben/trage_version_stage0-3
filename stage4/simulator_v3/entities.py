"""Core data objects for Simulator v3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .enums import LegType, RequestStatus, StopType, VehicleExecutionStatus


@dataclass
class RequestState:
    order_id: str
    request_time: pd.Timestamp
    observed_boarding_time: pd.Timestamp
    origin_lon: float
    origin_lat: float
    origin_zone: str
    destination_lon: float
    destination_lat: float
    destination_zone: str
    latest_pickup_time: pd.Timestamp
    condition_available: bool
    predicted_service_time_sec: float
    realized_service_time_sec: float
    route_length_m: float
    stress_value: float
    status: RequestStatus = RequestStatus.UNREVEALED
    assigned_vehicle_id: str | None = None
    reserved_vehicle_id: str | None = None
    offer_round: int = 0
    dispatch_round: int = 0
    last_offer_time: pd.Timestamp | None = None
    last_failure_reason: str | None = None
    first_candidate_time: pd.Timestamp | None = None
    first_offer_time: pd.Timestamp | None = None
    assignment_time: pd.Timestamp | None = None
    pickup_start_time: pd.Timestamp | None = None
    boarding_time: pd.Timestamp | None = None
    service_start_time: pd.Timestamp | None = None
    dropoff_time: pd.Timestamp | None = None
    cancellation_time: pd.Timestamp | None = None
    cancellation_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanStop:
    stop_id: str
    stop_type: StopType
    request_id: str | None
    lon: float
    lat: float
    zone: str
    earliest_start: pd.Timestamp | None
    latest_start: pd.Timestamp | None
    planned_arrival: pd.Timestamp
    planned_departure: pd.Timestamp
    locked: bool = False


@dataclass
class VehiclePlan:
    vehicle_id: str
    plan_version: int
    stops: list[PlanStop]
    created_time: pd.Timestamp
    trigger: str
    feasible: bool
    objective_value: float
    assigned_request_ids: list[str] = field(default_factory=list)
    reserved_request_ids: list[str] = field(default_factory=list)


@dataclass
class VehicleLeg:
    leg_id: str
    vehicle_id: str
    leg_type: LegType
    request_id: str | None
    start_lon: float
    start_lat: float
    end_lon: float
    end_lat: float
    planned_start: pd.Timestamp
    planned_end: pd.Timestamp
    actual_start: pd.Timestamp | None
    actual_end: pd.Timestamp | None
    distance_m: float
    expected_travel_time_sec: float
    realized_travel_time_sec: float | None
    route_source: str
    odd_feasible: bool | None
    plan_version: int = 0
    termination_reason: str | None = None


@dataclass
class VehicleState:
    vehicle_id: str
    vehicle_type: str
    current_lon: float
    current_lat: float
    current_zone: str
    online_start: pd.Timestamp
    online_end: pd.Timestamp
    execution_status: VehicleExecutionStatus
    current_leg: VehicleLeg | None
    active_plan: VehiclePlan
    plan_version: int
    current_request_id: str | None = None
    reserved_request_id: str | None = None
    driver_id: str | None = None
    session_id: str | None = None
    depot_id: str | None = None
    profile: str = "moderate_av"
    cumulative_busy_time_sec: float = 0.0
    cumulative_pickup_distance_m: float = 0.0
    cumulative_service_distance_m: float = 0.0
    cumulative_reposition_distance_m: float = 0.0
    cumulative_rebalancing_distance_m: float = 0.0
    cumulative_income: float = 0.0
    cumulative_stress_burden: float = 0.0
    completed_legs: int = 0

