"""Edge-level economics for simulator v3."""

from __future__ import annotations

from dataclasses import dataclass

from .entities import RequestState, VehicleState


@dataclass
class EdgeEconomics:
    fare_revenue: float
    driver_payout: float
    pickup_variable_cost: float
    service_variable_cost: float
    capability_cost: float
    remote_assistance_cost: float
    platform_variable_cost: float
    marginal_operating_contribution: float
    passenger_gc: float
    driver_utility: float
    stress: float


class EconomicsModel:
    def __init__(self, passenger_gc_cap: float = 120.0, driver_utility_min: float = -2.0):
        self.passenger_gc_cap = passenger_gc_cap
        self.driver_utility_min = driver_utility_min

    def evaluate(
        self,
        request: RequestState,
        vehicle: VehicleState,
        pickup_distance_m: float,
        pickup_time_sec: float,
        wait_sec: float,
        strategy: str,
        capability_cost: float = 0.0,
        remote_assistance_cost: float = 0.0,
    ) -> EdgeEconomics:
        route_km = request.route_length_m / 1000.0
        service_min = request.predicted_service_time_sec / 60.0
        stress = request.stress_value if request.condition_available else 0.0
        fare = 8.0 + 2.0 * route_km + 0.4 * service_min
        if strategy != "Safe GlobalMatch-MinPickup" and request.condition_available:
            fare += min(8.0, 4.0 * stress)
        passenger_gc = fare + wait_sec / 60.0 * 0.35 + pickup_time_sec / 60.0 * 0.25
        pickup_cost = 0.30 * pickup_distance_m / 1000.0
        service_cost = 0.45 * route_km
        if vehicle.vehicle_type == "AV":
            capability_cost = max(0.0, float(capability_cost))
            remote_assistance_cost = max(0.0, float(remote_assistance_cost))
            contribution = fare - pickup_cost - service_cost - capability_cost - remote_assistance_cost
            return EdgeEconomics(
                fare, 0.0, pickup_cost, service_cost, capability_cost,
                remote_assistance_cost, 0.0,
                contribution, passenger_gc, 0.0, stress,
            )
        gross_comp = 0.0 if strategy == "Safe GlobalMatch-MinPickup" or not request.condition_available else min(8.0, 5.0 * stress)
        base_payout = 5.0 + 1.15 * route_km + 0.2 * service_min
        pickup_comp = 0.25 * pickup_distance_m / 1000.0
        driver_payout = base_payout + pickup_comp + gross_comp
        driver_cost = 0.25 * pickup_distance_m / 1000.0 + 0.22 * route_km
        driver_utility = driver_payout - driver_cost - 2.5 * stress
        # The platform ledger treats the full driver payout as a cash outflow.
        # Vehicle operating costs are borne by the driver for HV service, so
        # they affect driver utility but are not subtracted a second time from
        # platform contribution.  A small platform transaction cost is kept as
        # a distinct, auditable component.
        platform_variable_cost = 0.1 * route_km
        contribution = fare - driver_payout - platform_variable_cost
        return EdgeEconomics(
            fare, driver_payout, 0.0, 0.0, 0.0, 0.0,
            platform_variable_cost, contribution, passenger_gc,
            driver_utility, stress,
        )
