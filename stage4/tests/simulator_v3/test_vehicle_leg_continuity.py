import pandas as pd

from stage4.simulator_v3.entities import RequestState, VehiclePlan, VehicleState
from stage4.simulator_v3.enums import RequestStatus, VehicleExecutionStatus
from stage4.simulator_v3.event_queue import EventQueue
from stage4.simulator_v3.request_manager import RequestManager
from stage4.simulator_v3.routing_engine import RoutingEngine
from stage4.simulator_v3.system_state import SystemState
from stage4.simulator_v3.vehicle_executor import VehicleExecutor
from stage4.simulator_v3.vehicle_plan import assignment_plan


def test_pickup_and_service_legs_are_time_and_space_continuous(tmp_path):
    pd.DataFrame(
        [{"origin_zone": "z0_0", "time_bin": 20, "empty_speed_mps": 8.0}]
    ).to_parquet(tmp_path / "pickup_empty_speed_by_zone_time.parquet", index=False)
    pd.DataFrame(
        [{"origin_zone": "z0_0", "circuity_factor": 1.2}]
    ).to_parquet(tmp_path / "pickup_circuity_by_zone.parquet", index=False)
    routing = RoutingEngine(tmp_path)
    now = pd.Timestamp("2016-10-23T10:00:00Z")
    request = RequestState(
        order_id="order-1",
        request_time=now,
        observed_boarding_time=now + pd.Timedelta(minutes=2),
        origin_lon=108.91,
        origin_lat=34.21,
        origin_zone="z0_0",
        destination_lon=108.93,
        destination_lat=34.23,
        destination_zone="z1_1",
        latest_pickup_time=now + pd.Timedelta(minutes=8),
        condition_available=True,
        predicted_service_time_sec=300,
        realized_service_time_sec=330,
        route_length_m=2_000,
        stress_value=0.2,
        status=RequestStatus.ASSIGNED,
        metadata={"time_bin": 20, "eta_source": "stage2_predicted_eta"},
    )
    vehicle = VehicleState(
        vehicle_id="vehicle-1",
        vehicle_type="AV",
        current_lon=108.90,
        current_lat=34.20,
        current_zone="z0_0",
        online_start=now - pd.Timedelta(hours=1),
        online_end=now + pd.Timedelta(hours=4),
        execution_status=VehicleExecutionStatus.IDLE,
        current_leg=None,
        active_plan=VehiclePlan("vehicle-1", 0, [], now, "init", True, 0.0),
        plan_version=0,
    )
    state = SystemState(current_time=now, requests={request.order_id: request}, vehicles={vehicle.vehicle_id: vehicle})
    state.initialize_vehicle_indexes()
    state.set_vehicle_status(vehicle.vehicle_id, VehicleExecutionStatus.IDLE)
    executor = VehicleExecutor(routing, EventQueue(), RequestManager(), state)
    plan = assignment_plan(
        vehicle,
        request,
        now,
        now + pd.Timedelta(minutes=2),
        now + pd.Timedelta(minutes=2),
        now + pd.Timedelta(minutes=7),
        objective_value=1.0,
        trigger="unit-test",
    )

    pickup_leg = executor.publish_plan(vehicle, plan, state.requests, now)
    assert pickup_leg is not None
    completed_pickup = executor.complete_current_leg(vehicle, state.requests, pickup_leg.planned_end)
    service_leg = vehicle.current_leg

    assert completed_pickup is pickup_leg
    assert service_leg is not None
    assert pickup_leg.actual_end <= service_leg.actual_start
    assert abs(pickup_leg.end_lon - service_leg.start_lon) <= 1e-6
    assert abs(pickup_leg.end_lat - service_leg.start_lat) <= 1e-6
    assert pickup_leg.leg_id != service_leg.leg_id
    assert pickup_leg.realized_travel_time_sec >= 0
    assert service_leg.realized_travel_time_sec >= 0

    completed_service = executor.complete_current_leg(vehicle, state.requests, service_leg.planned_end)

    assert completed_service is service_leg
    assert request.status == RequestStatus.COMPLETED
    assert vehicle.current_leg is None
    assert vehicle.completed_legs == 2
    assert (vehicle.current_lon, vehicle.current_lat) == (
        request.destination_lon,
        request.destination_lat,
    )

