import pandas as pd

from stage4.simulator_v3.entities import RequestState, VehiclePlan, VehicleState
from stage4.simulator_v3.enums import RequestStatus, VehicleExecutionStatus
from stage4.simulator_v3.event_queue import EventQueue
from stage4.simulator_v3.request_manager import RequestManager
from stage4.simulator_v3.routing_engine import RoutingEngine
from stage4.simulator_v3.system_state import SystemState
from stage4.simulator_v3.vehicle_executor import VehicleExecutor
from stage4.simulator_v3.vehicle_plan import assignment_plan


def _routing(tmp_path) -> RoutingEngine:
    pd.DataFrame(
        [{"origin_zone": "z0_0", "time_bin": 20, "empty_speed_mps": 8.0}]
    ).to_parquet(tmp_path / "pickup_empty_speed_by_zone_time.parquet", index=False)
    pd.DataFrame(
        [{"origin_zone": "z0_0", "circuity_factor": 1.25}]
    ).to_parquet(tmp_path / "pickup_circuity_by_zone.parquet", index=False)
    return RoutingEngine(tmp_path)


def test_plan_publication_does_not_move_vehicle_before_leg_completion(tmp_path):
    now = pd.Timestamp("2016-10-23T10:00:00Z")
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
        predicted_service_time_sec=600,
        realized_service_time_sec=620,
        route_length_m=3_000,
        stress_value=0.2,
        status=RequestStatus.ASSIGNED,
        metadata={"time_bin": 20, "eta_source": "stage2_predicted_eta"},
    )
    empty_plan = VehiclePlan("vehicle-1", 0, [], now, "init", True, 0.0)
    vehicle = VehicleState(
        vehicle_id="vehicle-1",
        vehicle_type="HV",
        current_lon=108.90,
        current_lat=34.20,
        current_zone="z0_0",
        online_start=now - pd.Timedelta(hours=1),
        online_end=now + pd.Timedelta(hours=4),
        execution_status=VehicleExecutionStatus.IDLE,
        current_leg=None,
        active_plan=empty_plan,
        plan_version=0,
    )
    state = SystemState(current_time=now, requests={request.order_id: request}, vehicles={vehicle.vehicle_id: vehicle})
    state.initialize_vehicle_indexes()
    state.set_vehicle_status(vehicle.vehicle_id, VehicleExecutionStatus.IDLE)
    manager = RequestManager()
    executor = VehicleExecutor(_routing(tmp_path), EventQueue(), manager, state)
    plan = assignment_plan(
        vehicle,
        request,
        now,
        now + pd.Timedelta(minutes=2),
        now + pd.Timedelta(minutes=2),
        now + pd.Timedelta(minutes=12),
        objective_value=1.0,
        trigger="unit-test",
    )

    initial_position = (vehicle.current_lon, vehicle.current_lat, vehicle.current_zone)
    leg = executor.publish_plan(vehicle, plan, state.requests, now)

    assert leg is not None
    assert leg.start_lon == initial_position[0]
    assert leg.start_lat == initial_position[1]
    assert (vehicle.current_lon, vehicle.current_lat, vehicle.current_zone) == initial_position
    assert vehicle.execution_status == VehicleExecutionStatus.PICKUP
    assert request.status == RequestStatus.PICKUP_STARTED

    executor.complete_current_leg(vehicle, state.requests, leg.planned_end)

    assert (vehicle.current_lon, vehicle.current_lat, vehicle.current_zone) == (
        request.origin_lon,
        request.origin_lat,
        request.origin_zone,
    )

