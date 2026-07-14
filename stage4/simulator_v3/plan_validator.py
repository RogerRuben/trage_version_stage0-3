"""Plan validation before a plan can be published."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .entities import RequestState, VehicleState
from .enums import VehicleExecutionStatus
from .odd.pickup_odd import PickupODDChecker
from .odd.service_odd import ServiceODDChecker


@dataclass
class ValidationResult:
    feasible: bool
    failure_reason: str = ""
    pickup_odd_feasible: bool = True
    service_odd_feasible: bool = True


class PlanValidator:
    def __init__(self, pickup_odd: PickupODDChecker, service_odd: ServiceODDChecker):
        self.pickup_odd = pickup_odd
        self.service_odd = service_odd

    def validate(
        self,
        vehicle: VehicleState,
        request: RequestState,
        current_time: pd.Timestamp,
        pickup_arrival: pd.Timestamp,
        service_end: pd.Timestamp,
    ) -> ValidationResult:
        if current_time < vehicle.online_start or current_time > vehicle.online_end:
            return ValidationResult(False, "vehicle_offline")
        if vehicle.execution_status not in {VehicleExecutionStatus.IDLE, VehicleExecutionStatus.WAITING}:
            return ValidationResult(False, "vehicle_not_controllable")
        if pickup_arrival > request.latest_pickup_time:
            return ValidationResult(False, "pickup_deadline")
        if service_end > vehicle.online_end:
            return ValidationResult(False, "session_constraint")
        pickup_ok, pickup_reason = self.pickup_odd.check(vehicle.vehicle_type, vehicle.current_zone, request.origin_zone, request.condition_available)
        service_ok, service_reason = self.service_odd.check(request.order_id, vehicle.vehicle_type, request.condition_available)
        if not pickup_ok:
            return ValidationResult(False, pickup_reason, pickup_odd_feasible=False, service_odd_feasible=service_ok)
        if not service_ok:
            return ValidationResult(False, service_reason, pickup_odd_feasible=pickup_ok, service_odd_feasible=False)
        return ValidationResult(True, "", pickup_odd_feasible=pickup_ok, service_odd_feasible=service_ok)

