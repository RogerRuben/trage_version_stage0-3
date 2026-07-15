"""Vehicle physical execution layer.

Only this module mutates vehicle physical state: position, current leg,
execution status, and cumulative movement counters.
"""

from __future__ import annotations

import os
import uuid

import pandas as pd

from .entities import RequestState, VehicleLeg, VehiclePlan, VehicleState
from .enums import EventType, LegType, RequestStatus, StopType, VehicleExecutionStatus
from .event_queue import EventQueue
from .request_manager import RequestManager
from .routing_engine import RoutingEngine
from .system_state import SystemState


class VehicleExecutor:
    def __init__(self, routing_engine: RoutingEngine, event_queue: EventQueue, request_manager: RequestManager, state: SystemState):
        self.routing = routing_engine
        self.event_queue = event_queue
        self.request_manager = request_manager
        self.state = state

    def publish_plan(self, vehicle: VehicleState, plan: VehiclePlan, requests: dict[str, RequestState], current_time: pd.Timestamp) -> VehicleLeg | None:
        vehicle.active_plan = plan
        vehicle.plan_version = plan.plan_version
        return self._start_next_leg(vehicle, requests, current_time)

    def start_next_leg(self, vehicle: VehicleState, requests: dict[str, RequestState], current_time: pd.Timestamp) -> VehicleLeg | None:
        """Public execution entrypoint used after reservation revalidation."""
        return self._start_next_leg(vehicle, requests, current_time)

    def _start_next_leg(self, vehicle: VehicleState, requests: dict[str, RequestState], current_time: pd.Timestamp) -> VehicleLeg | None:
        if vehicle.current_leg is not None:
            return None
        if not vehicle.active_plan.stops:
            self.state.set_vehicle_status(vehicle.vehicle_id, VehicleExecutionStatus.IDLE if vehicle.online_start <= current_time <= vehicle.online_end else VehicleExecutionStatus.OFFLINE)
            return None
        stop = vehicle.active_plan.stops.pop(0)
        if stop.stop_type == StopType.PICKUP:
            req = requests[str(stop.request_id)]
            route = self.routing.query_pickup_route(
                (vehicle.current_lon, vehicle.current_lat, vehicle.current_zone),
                (req.origin_lon, req.origin_lat, req.origin_zone),
                current_time,
                vehicle.vehicle_type,
                int(req.metadata.get("time_bin", 0)),
            )
            leg_type = LegType.PICKUP
            status = VehicleExecutionStatus.PICKUP
            if req.status == RequestStatus.RESERVED:
                self.request_manager.transition(req, RequestStatus.ASSIGNED, current_time, trigger="RESERVED_PICKUP_RELEASED", vehicle_id=vehicle.vehicle_id, plan_version=vehicle.plan_version)
                req.assigned_vehicle_id = vehicle.vehicle_id
                req.reserved_vehicle_id = None
                req.assignment_time = req.assignment_time or current_time
                vehicle.reserved_request_id = None
                self.state.reserved_request_ids.discard(req.order_id)
            self.request_manager.transition(req, RequestStatus.PICKUP_STARTED, current_time, trigger="PICKUP_LEG_STARTED", vehicle_id=vehicle.vehicle_id, plan_version=vehicle.plan_version)
            req.pickup_start_time = current_time
            vehicle.current_request_id = req.order_id
        elif stop.stop_type == StopType.DROP_OFF:
            req = requests[str(stop.request_id)]
            route = self.routing.query_service_route(req, current_time, vehicle.vehicle_type)
            leg_type = LegType.SERVICE
            status = VehicleExecutionStatus.SERVICE
            if req.status == RequestStatus.BOARDED:
                self.request_manager.transition(req, RequestStatus.IN_SERVICE, current_time, trigger="SERVICE_LEG_STARTED", vehicle_id=vehicle.vehicle_id, plan_version=vehicle.plan_version)
            req.service_start_time = current_time
        elif stop.stop_type == StopType.HV_REPOSITION:
            route = self.routing.query((vehicle.current_lon, vehicle.current_lat, vehicle.current_zone), (stop.lon, stop.lat, stop.zone), current_time, vehicle.vehicle_type)
            leg_type = LegType.HV_REPOSITION
            status = VehicleExecutionStatus.REPOSITIONING
        elif stop.stop_type == StopType.AV_REBALANCE:
            route = self.routing.query((vehicle.current_lon, vehicle.current_lat, vehicle.current_zone), (stop.lon, stop.lat, stop.zone), current_time, vehicle.vehicle_type)
            leg_type = LegType.AV_REBALANCE
            status = VehicleExecutionStatus.REBALANCING
        else:
            return self._start_next_leg(vehicle, requests, current_time)
        duration = route.realized_travel_time_sec if route.realized_travel_time_sec is not None else route.expected_travel_time_sec
        # Formal runs keep a one-second minimum so a decision epoch cannot
        # causally create a higher-priority completion at the same timestamp.
        # The FleetPy co-located kernel control explicitly sets this to zero.
        minimum_duration = float(os.environ.get("SIMULATOR_V3_MIN_LEG_DURATION_SEC", "1"))
        duration = max(minimum_duration, float(duration))
        planned_end = current_time + pd.Timedelta(seconds=float(duration))
        leg = VehicleLeg(
            leg_id=f"LEG_{uuid.uuid4().hex[:16]}",
            vehicle_id=vehicle.vehicle_id,
            leg_type=leg_type,
            request_id=stop.request_id,
            start_lon=vehicle.current_lon,
            start_lat=vehicle.current_lat,
            end_lon=stop.lon,
            end_lat=stop.lat,
            planned_start=current_time,
            planned_end=planned_end,
            actual_start=current_time,
            actual_end=None,
            distance_m=route.road_distance_m,
            expected_travel_time_sec=route.expected_travel_time_sec,
            realized_travel_time_sec=float(duration),
            route_source=route.route_source,
            odd_feasible=None,
            plan_version=vehicle.plan_version,
        )
        vehicle.current_leg = leg
        self.state.set_vehicle_status(vehicle.vehicle_id, status)
        self.event_queue.push(planned_end, EventType.LEG_COMPLETED, vehicle.vehicle_id, {"leg_id": leg.leg_id})
        return leg

    def complete_current_leg(self, vehicle: VehicleState, requests: dict[str, RequestState], current_time: pd.Timestamp) -> VehicleLeg | None:
        leg = vehicle.current_leg
        if leg is None:
            return None
        leg.actual_end = current_time
        vehicle.current_lon = leg.end_lon
        vehicle.current_lat = leg.end_lat
        if leg.request_id and leg.request_id in requests:
            req = requests[leg.request_id]
            if leg.leg_type == LegType.PICKUP:
                vehicle.current_zone = req.origin_zone
                self.request_manager.transition(req, RequestStatus.BOARDED, current_time, trigger="PICKUP_LEG_COMPLETED", vehicle_id=vehicle.vehicle_id, plan_version=vehicle.plan_version)
                req.boarding_time = current_time
                vehicle.cumulative_pickup_distance_m += leg.distance_m
            elif leg.leg_type == LegType.SERVICE:
                vehicle.current_zone = req.destination_zone
                self.request_manager.transition(req, RequestStatus.COMPLETED, current_time, trigger="SERVICE_LEG_COMPLETED", vehicle_id=vehicle.vehicle_id, plan_version=vehicle.plan_version)
                req.dropoff_time = current_time
                req.assigned_vehicle_id = vehicle.vehicle_id
                vehicle.current_request_id = None
                vehicle.cumulative_service_distance_m += leg.distance_m
                vehicle.cumulative_stress_burden += req.stress_value
        elif leg.leg_type == LegType.HV_REPOSITION:
            vehicle.cumulative_reposition_distance_m += leg.distance_m
        elif leg.leg_type == LegType.AV_REBALANCE:
            vehicle.cumulative_rebalancing_distance_m += leg.distance_m
        vehicle.cumulative_busy_time_sec += max(0.0, (current_time - leg.actual_start).total_seconds()) if leg.actual_start is not None else 0.0
        vehicle.completed_legs += 1
        vehicle.current_leg = None
        self.state.set_vehicle_status(vehicle.vehicle_id, VehicleExecutionStatus.IDLE if vehicle.online_start <= current_time <= vehicle.online_end else VehicleExecutionStatus.OFFLINE)
        # A reserved next request must be revalidated against the realized
        # release time before its pickup leg can begin.  The event engine owns
        # that check and calls ``start_next_leg`` explicitly on success.
        if not (leg.leg_type == LegType.SERVICE and vehicle.reserved_request_id is not None):
            self._start_next_leg(vehicle, requests, current_time)
        return leg
