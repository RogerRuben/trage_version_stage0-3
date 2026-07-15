"""Deterministic HV offer response model."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..economics import EdgeEconomics
from ..entities import RequestState, VehicleState
from ..enums import DriverResponse


@dataclass
class DriverOfferResult:
    response: DriverResponse
    response_delay_sec: float
    reason: str = ""


class DriverResponseModel:
    def __init__(self, utility_threshold: float = -2.0, response_delay_sec: float = 10.0):
        self.utility_threshold = utility_threshold
        self.response_delay_sec = response_delay_sec

    def evaluate_offer(
        self,
        vehicle: VehicleState,
        request: RequestState,
        economics: EdgeEconomics,
        current_time: pd.Timestamp,
        service_end_time: pd.Timestamp,
    ) -> DriverOfferResult:
        if service_end_time > vehicle.online_end:
            return DriverOfferResult(DriverResponse.REJECT, self.response_delay_sec, "session_constraint")
        if economics.driver_utility < self.utility_threshold:
            return DriverOfferResult(DriverResponse.REJECT, self.response_delay_sec, "utility_below_threshold")
        return DriverOfferResult(DriverResponse.ACCEPT, self.response_delay_sec, "")

    def evaluate_delayed_response(
        self,
        vehicle: VehicleState,
        driver_utility: float,
        service_end_time: pd.Timestamp,
    ) -> DriverOfferResult:
        """Evaluate only when the scheduled DRIVER_RESPONSE event fires."""
        if pd.Timestamp(service_end_time) > vehicle.online_end:
            return DriverOfferResult(DriverResponse.REJECT, self.response_delay_sec, "session_constraint")
        if float(driver_utility) < self.utility_threshold:
            return DriverOfferResult(DriverResponse.REJECT, self.response_delay_sec, "utility_below_threshold")
        return DriverOfferResult(DriverResponse.ACCEPT, self.response_delay_sec, "")
