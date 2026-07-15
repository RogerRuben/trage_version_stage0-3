from types import SimpleNamespace

import pandas as pd

from stage4.simulator_v3.behavior.driver_response import DriverResponseModel
from stage4.simulator_v3.entities import PlanStop, RequestState, VehiclePlan, VehicleState
from stage4.simulator_v3.enums import EventType, RequestStatus, StopType, VehicleExecutionStatus
from stage4.simulator_v3.event_queue import EventQueue
from stage4.simulator_v3.matching.sparse_matcher import CandidateEdge
from stage4.simulator_v3.request_manager import RequestManager
from stage4.simulator_v3.simulation_engine import SimulationEngine
from stage4.simulator_v3.system_state import SystemState


class RecordingExecutor:
    def __init__(self):
        self.published = []

    def publish_plan(self, vehicle, plan, requests, current_time):
        self.published.append((vehicle.vehicle_id, plan.plan_version, current_time))


def test_hv_offer_remains_offered_until_driver_response_event_fires():
    now = pd.Timestamp("2016-10-23T10:00:00Z")
    response_time = now + pd.Timedelta(seconds=10)
    request = RequestState(
        order_id="order-1",
        request_time=now,
        observed_boarding_time=now + pd.Timedelta(minutes=2),
        origin_lon=108.91,
        origin_lat=34.21,
        origin_zone="z0_0",
        destination_lon=108.92,
        destination_lat=34.22,
        destination_zone="z1_1",
        latest_pickup_time=now + pd.Timedelta(minutes=8),
        condition_available=True,
        predicted_service_time_sec=300,
        realized_service_time_sec=300,
        route_length_m=2_000,
        stress_value=0.2,
        status=RequestStatus.OFFERED,
        last_offer_time=now,
        first_offer_time=now,
        offer_round=1,
    )
    old_plan = VehiclePlan("vehicle-1", 0, [], now, "init", True, 0.0)
    vehicle = VehicleState(
        vehicle_id="vehicle-1",
        vehicle_type="HV",
        current_lon=108.90,
        current_lat=34.20,
        current_zone="z0_0",
        online_start=now - pd.Timedelta(hours=1),
        online_end=now + pd.Timedelta(hours=2),
        execution_status=VehicleExecutionStatus.WAITING,
        current_leg=None,
        active_plan=old_plan,
        plan_version=0,
    )
    plan = VehiclePlan(
        vehicle_id=vehicle.vehicle_id,
        plan_version=1,
        stops=[
            PlanStop(
                "pickup", StopType.PICKUP, request.order_id,
                request.origin_lon, request.origin_lat, request.origin_zone,
                None, request.latest_pickup_time,
                now + pd.Timedelta(minutes=1), now + pd.Timedelta(minutes=1),
            ),
            PlanStop(
                "dropoff", StopType.DROP_OFF, request.order_id,
                request.destination_lon, request.destination_lat, request.destination_zone,
                None, None,
                now + pd.Timedelta(minutes=6), now + pd.Timedelta(minutes=6),
            ),
        ],
        created_time=now,
        trigger="unit-test",
        feasible=True,
        objective_value=1.0,
        assigned_request_ids=[request.order_id],
    )
    edge = CandidateEdge(
        request_id=request.order_id,
        vehicle_id=vehicle.vehicle_id,
        pickup_eta_sec=60,
        marginal_contribution=5.0,
        passenger_gc=10.0,
        driver_utility=1.0,
        stress=0.2,
        objective=5.0,
        metadata={"vehicle_type": "HV"},
    )
    state = SystemState(
        current_time=now,
        requests={request.order_id: request},
        vehicles={vehicle.vehicle_id: vehicle},
        offered_request_ids={request.order_id},
    )
    state.initialize_vehicle_indexes()
    state.set_vehicle_status(vehicle.vehicle_id, VehicleExecutionStatus.WAITING)
    events = EventQueue()
    offer_id = "order-1:vehicle-1:1"
    events.push(response_time, EventType.DRIVER_RESPONSE, offer_id)
    manager = RequestManager()
    executor = RecordingExecutor()
    controller = SimpleNamespace(
        driver_response=DriverResponseModel(utility_threshold=-2.0, response_delay_sec=10.0),
        routing=SimpleNamespace(query_count=0, cache_hit_count=0, cache_hit_rate=0.0),
    )
    engine = SimulationEngine(state, events, controller, executor, manager)
    engine.pending_offers[offer_id] = {
        "request_id": request.order_id,
        "vehicle_id": vehicle.vehicle_id,
        "plan": plan,
        "old_plan": old_plan,
        "edge": edge,
        "offer_time": now,
        "response_deadline": response_time,
    }

    engine.run(now + pd.Timedelta(seconds=9), finalize=False)

    assert request.status == RequestStatus.OFFERED
    assert executor.published == []
    assert manager.transition_records == []

    engine.run(response_time, finalize=False)

    assert request.status == RequestStatus.ASSIGNED
    assert request.assignment_time == response_time
    assert executor.published == [(vehicle.vehicle_id, plan.plan_version, response_time)]
    assert manager.transition_records[-1]["trigger"] == "ASSIGNMENT_CONFIRMED"
    assert engine.offer_records[-1]["response"] == "ACCEPT"
    assert pd.Timestamp(engine.offer_records[-1]["response_time"]) > pd.Timestamp(engine.offer_records[-1]["offer_time"])


def test_preassignment_offer_is_released_when_locked_leg_finishes_before_response():
    """An asynchronous response must not install a plan built on a vanished leg."""
    now = pd.Timestamp("2016-10-23T10:00:00Z")
    response_time = now + pd.Timedelta(seconds=10)
    request = RequestState(
        order_id="reserved-order",
        request_time=now,
        observed_boarding_time=now + pd.Timedelta(minutes=2),
        origin_lon=108.91,
        origin_lat=34.21,
        origin_zone="z0_0",
        destination_lon=108.92,
        destination_lat=34.22,
        destination_zone="z1_1",
        latest_pickup_time=now + pd.Timedelta(minutes=8),
        condition_available=True,
        predicted_service_time_sec=300,
        realized_service_time_sec=300,
        route_length_m=2_000,
        stress_value=0.2,
        status=RequestStatus.OFFERED,
        last_offer_time=now,
        first_offer_time=now,
        offer_round=1,
    )
    old_plan = VehiclePlan("vehicle-1", 0, [], now, "init", True, 0.0)
    vehicle = VehicleState(
        vehicle_id="vehicle-1",
        vehicle_type="HV",
        current_lon=108.90,
        current_lat=34.20,
        current_zone="z0_0",
        online_start=now - pd.Timedelta(hours=1),
        online_end=now + pd.Timedelta(hours=2),
        execution_status=VehicleExecutionStatus.IDLE,
        current_leg=None,
        active_plan=old_plan,
        plan_version=0,
    )
    plan = VehiclePlan(
        vehicle_id=vehicle.vehicle_id,
        plan_version=1,
        stops=[],
        created_time=now,
        trigger="preassignment",
        feasible=True,
        objective_value=1.0,
        reserved_request_ids=[request.order_id],
    )
    edge = CandidateEdge(
        request_id=request.order_id,
        vehicle_id=vehicle.vehicle_id,
        pickup_eta_sec=60,
        marginal_contribution=5.0,
        passenger_gc=10.0,
        driver_utility=1.0,
        stress=0.2,
        objective=5.0,
        metadata={"vehicle_type": "HV"},
    )
    state = SystemState(
        current_time=now,
        requests={request.order_id: request},
        vehicles={vehicle.vehicle_id: vehicle},
        offered_request_ids={request.order_id},
    )
    state.initialize_vehicle_indexes()
    events = EventQueue()
    offer_id = "reserved-order:vehicle-1:1"
    events.push(response_time, EventType.DRIVER_RESPONSE, offer_id)
    manager = RequestManager()
    controller = SimpleNamespace(
        driver_response=DriverResponseModel(utility_threshold=-2.0, response_delay_sec=10.0),
        routing=SimpleNamespace(query_count=0, cache_hit_count=0, cache_hit_rate=0.0),
    )
    engine = SimulationEngine(state, events, controller, RecordingExecutor(), manager)
    engine.pending_offers[offer_id] = {
        "request_id": request.order_id,
        "vehicle_id": vehicle.vehicle_id,
        "plan": plan,
        "old_plan": old_plan,
        "edge": edge,
        "offer_time": now,
        "response_deadline": response_time,
        "preassigned": True,
    }

    engine.run(response_time, finalize=False)

    assert request.status == RequestStatus.PENDING
    assert request.reserved_vehicle_id is None
    assert vehicle.reserved_request_id is None
    assert engine.reservations.records == []
    assert engine.offer_records[-1]["rejection_reason"] == "vehicle_released_before_preassignment_response"
