"""Drive FleetPy's native external-vehicle lifecycle with frozen replay durations."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt

from .mixed_fleet_adapter import VehicleRuntime
from .test31_demand_adapter import SpikeRequest
from .upstream import CoordinateRegistry, FleetPyBindings, FleetPyCompatibilityError
from .valhalla_time_adapter import PickupEstimate

EARTH_RADIUS_M = 6_371_008.8


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1r, lat1r, lon2r, lat2r = map(radians, (lon1, lat1, lon2, lat2))
    dlon, dlat = lon2r - lon1r, lat2r - lat1r
    value = sin(dlat / 2) ** 2 + cos(lat1r) * cos(lat2r) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * asin(sqrt(min(max(value, 0.0), 1.0)))


@dataclass(frozen=True)
class _PlanRequestHandle:
    native_id: int

    def get_rid(self) -> int:
        return self.native_id


class FleetPyLifecycleAdapter:
    """Submit four native legs and trigger external arrivals at adapter times."""

    def __init__(self, bindings: FleetPyBindings, registry: CoordinateRegistry) -> None:
        self.bindings = bindings
        self.registry = registry

    def assign(
        self,
        vehicle: VehicleRuntime,
        request: SpikeRequest,
        pickup: PickupEstimate,
        sim_time_s: float,
    ) -> None:
        if vehicle.state != "AVAILABLE":
            raise FleetPyCompatibilityError("assignment requires AVAILABLE vehicle")
        handle = _PlanRequestHandle(request.native_id)
        states = self.bindings.states
        leg = self.bindings.vehicle_route_leg
        service_distance = _haversine_m(
            request.pickup_lon_wgs84,
            request.pickup_lat_wgs84,
            request.dropoff_lon_wgs84,
            request.dropoff_lat_wgs84,
        )
        self.registry.set_leg_metrics(
            vehicle.native_vehicle.pos,
            request.pickup_position,
            pickup.corrected_pickup_eta_s,
            pickup.route_distance_m,
        )
        self.registry.set_leg_metrics(
            request.pickup_position,
            request.dropoff_position,
            request.realized_service_time_s,
            service_distance,
        )
        route_legs = [
            leg(states.REPOSITION, request.pickup_position, {}),
            leg(
                states.BOARDING,
                request.pickup_position,
                {1: [handle]},
                duration=0.0,
                locked=True,
            ),
            leg(states.ROUTE, request.dropoff_position, {}),
            leg(
                states.BOARDING,
                request.dropoff_position,
                {-1: [handle]},
                duration=0.0,
                locked=True,
            ),
        ]
        vehicle.native_vehicle.assign_vehicle_plan(route_legs, sim_time_s)
        vehicle.native_vehicle.update_veh_state(sim_time_s, sim_time_s)
        vehicle.state = "TO_PICKUP"
        vehicle.active_order_id = request.order_id
        vehicle.pickup_eta_s = pickup.corrected_pickup_eta_s

    def arrive_pickup(
        self, vehicle: VehicleRuntime, request: SpikeRequest, sim_time_s: float
    ) -> None:
        if vehicle.state != "TO_PICKUP":
            raise FleetPyCompatibilityError("pickup event requires TO_PICKUP state")
        native = vehicle.native_vehicle
        native.update_vehicle_position(request.pickup_position, sim_time_s)
        native.reached_destination(sim_time_s)
        boarding, _, _, _ = native.update_veh_state(sim_time_s, sim_time_s)
        if list(boarding) != [request.native_id]:
            raise FleetPyCompatibilityError("FleetPy boarding identity mismatch")
        request.native_request.user_boards_vehicle(
            sim_time_s, 0, vehicle.fixture.native_id, request.pickup_position, None
        )
        native.end_current_leg(sim_time_s)
        native.start_next_leg(sim_time_s)
        vehicle.state = "IN_SERVICE"
        vehicle.current_lon_wgs84 = request.pickup_lon_wgs84
        vehicle.current_lat_wgs84 = request.pickup_lat_wgs84

    def arrive_dropoff(
        self, vehicle: VehicleRuntime, request: SpikeRequest, sim_time_s: float
    ) -> None:
        if vehicle.state != "IN_SERVICE":
            raise FleetPyCompatibilityError("dropoff event requires IN_SERVICE state")
        native = vehicle.native_vehicle
        native.update_vehicle_position(request.dropoff_position, sim_time_s)
        native.reached_destination(sim_time_s)
        _, _, _, start_alighting = native.update_veh_state(sim_time_s, sim_time_s)
        if list(start_alighting) != [request.native_id]:
            raise FleetPyCompatibilityError("FleetPy alighting-start identity mismatch")
        request.native_request.user_leaves_vehicle(
            sim_time_s, request.dropoff_position, None
        )
        alighting, _ = native.end_current_leg(sim_time_s)
        if alighting != [request.native_id] or native.assigned_route:
            raise FleetPyCompatibilityError("FleetPy completion lifecycle mismatch")
        vehicle.current_lon_wgs84 = request.dropoff_lon_wgs84
        vehicle.current_lat_wgs84 = request.dropoff_lat_wgs84
        vehicle.active_order_id = None
        vehicle.next_event_time = None
        vehicle.pickup_eta_s = None
