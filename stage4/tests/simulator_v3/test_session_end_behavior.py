from types import SimpleNamespace

import pandas as pd

from stage4.simulator_v3.entities import RequestState, VehicleLeg, VehiclePlan, VehicleState
from stage4.simulator_v3.enums import EventType, LegType, RequestStatus, VehicleExecutionStatus
from stage4.simulator_v3.event_queue import EventQueue
from stage4.simulator_v3.request_manager import RequestManager
from stage4.simulator_v3.simulation_engine import SimulationEngine
from stage4.simulator_v3.system_state import SystemState
from stage4.simulator_v3.vehicle_executor import VehicleExecutor


def _controller_stub():
    return SimpleNamespace(
        routing=SimpleNamespace(query_count=0, cache_hit_count=0, cache_hit_rate=0.0)
    )


def test_session_end_allows_locked_service_to_finish_then_goes_offline():
    now = pd.Timestamp("2016-10-23T10:00:00Z")
    session_end = now + pd.Timedelta(minutes=1)
    service_end = now + pd.Timedelta(minutes=2)
    request = RequestState(
        order_id="order-1",
        request_time=now - pd.Timedelta(minutes=5),
        observed_boarding_time=now,
        origin_lon=108.90,
        origin_lat=34.20,
        origin_zone="z0_0",
        destination_lon=108.92,
        destination_lat=34.22,
        destination_zone="z1_1",
        latest_pickup_time=now,
        condition_available=True,
        predicted_service_time_sec=120,
        realized_service_time_sec=120,
        route_length_m=1_000,
        stress_value=0.2,
        status=RequestStatus.IN_SERVICE,
        assigned_vehicle_id="vehicle-1",
    )
    leg = VehicleLeg(
        leg_id="leg-service-1",
        vehicle_id="vehicle-1",
        leg_type=LegType.SERVICE,
        request_id=request.order_id,
        start_lon=request.origin_lon,
        start_lat=request.origin_lat,
        end_lon=request.destination_lon,
        end_lat=request.destination_lat,
        planned_start=now,
        planned_end=service_end,
        actual_start=now,
        actual_end=None,
        distance_m=1_000,
        expected_travel_time_sec=120,
        realized_travel_time_sec=120,
        route_source="historical_service_backend",
        odd_feasible=True,
    )
    vehicle = VehicleState(
        vehicle_id="vehicle-1",
        vehicle_type="HV",
        current_lon=request.origin_lon,
        current_lat=request.origin_lat,
        current_zone=request.origin_zone,
        online_start=now - pd.Timedelta(hours=1),
        online_end=session_end,
        execution_status=VehicleExecutionStatus.SERVICE,
        current_leg=leg,
        active_plan=VehiclePlan("vehicle-1", 1, [], now, "locked-service", True, 0.0, [request.order_id]),
        plan_version=1,
        current_request_id=request.order_id,
    )
    state = SystemState(current_time=now, requests={request.order_id: request}, vehicles={vehicle.vehicle_id: vehicle})
    state.initialize_vehicle_indexes()
    state.set_vehicle_status(vehicle.vehicle_id, VehicleExecutionStatus.SERVICE)
    events = EventQueue()
    events.push(session_end, EventType.HV_SESSION_END, vehicle.vehicle_id)
    events.push(service_end, EventType.LEG_COMPLETED, vehicle.vehicle_id, {"leg_id": leg.leg_id})
    manager = RequestManager()
    executor = VehicleExecutor(SimpleNamespace(), events, manager, state)
    engine = SimulationEngine(state, events, _controller_stub(), executor, manager)

    engine.run(session_end, finalize=False)

    assert vehicle.current_leg is leg
    assert vehicle.execution_status == VehicleExecutionStatus.SERVICE
    assert engine.event_execution_records[-1]["result"] == "ACTIVE_LOCKED_TASK"

    engine.run(service_end, finalize=False)

    assert request.status == RequestStatus.COMPLETED
    assert vehicle.current_leg is None
    assert vehicle.execution_status == VehicleExecutionStatus.OFFLINE
    assert vehicle.vehicle_id in state.offline_vehicle_ids
    assert vehicle.vehicle_id not in state.idle_hv_ids


def test_session_end_takes_idle_vehicle_offline_immediately():
    now = pd.Timestamp("2016-10-23T10:00:00Z")
    vehicle = VehicleState(
        vehicle_id="vehicle-idle",
        vehicle_type="HV",
        current_lon=108.9,
        current_lat=34.2,
        current_zone="z0_0",
        online_start=now - pd.Timedelta(hours=1),
        online_end=now,
        execution_status=VehicleExecutionStatus.IDLE,
        current_leg=None,
        active_plan=VehiclePlan("vehicle-idle", 0, [], now, "init", True, 0.0),
        plan_version=0,
    )
    state = SystemState(current_time=now, vehicles={vehicle.vehicle_id: vehicle})
    state.initialize_vehicle_indexes()
    state.set_vehicle_status(vehicle.vehicle_id, VehicleExecutionStatus.IDLE)
    events = EventQueue()
    events.push(now, EventType.HV_SESSION_END, vehicle.vehicle_id)
    manager = RequestManager()
    engine = SimulationEngine(state, events, _controller_stub(), SimpleNamespace(), manager)

    engine.run(now, finalize=False)

    assert vehicle.execution_status == VehicleExecutionStatus.OFFLINE
    assert vehicle.vehicle_id in state.offline_vehicle_ids
