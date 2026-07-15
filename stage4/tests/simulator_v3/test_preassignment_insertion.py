import pandas as pd

from stage4.simulator_v3.entities import RequestState, VehiclePlan, VehicleState
from stage4.simulator_v3.enums import VehicleExecutionStatus
from stage4.simulator_v3.vehicle_plan import insert_request_after_locked_stops


def test_preassignment_insertion_preserves_current_request_and_adds_reserved_request():
    now = pd.Timestamp("2016-10-23T10:00:00Z")
    active_plan = VehiclePlan(
        vehicle_id="vehicle-1",
        plan_version=2,
        stops=[],
        created_time=now,
        trigger="current-service",
        feasible=True,
        objective_value=0.0,
        assigned_request_ids=["current-order"],
        reserved_request_ids=[],
    )
    vehicle = VehicleState(
        vehicle_id="vehicle-1",
        vehicle_type="HV",
        current_lon=108.9,
        current_lat=34.2,
        current_zone="z0_0",
        online_start=now - pd.Timedelta(hours=1),
        online_end=now + pd.Timedelta(hours=4),
        execution_status=VehicleExecutionStatus.SERVICE,
        current_leg=None,
        active_plan=active_plan,
        plan_version=2,
        current_request_id="current-order",
    )
    request = RequestState(
        order_id="reserved-order",
        request_time=now,
        observed_boarding_time=now + pd.Timedelta(minutes=4),
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
        route_length_m=3000,
        stress_value=0.2,
    )

    plan = insert_request_after_locked_stops(
        vehicle,
        request,
        now,
        now + pd.Timedelta(minutes=3),
        now + pd.Timedelta(minutes=3),
        now + pd.Timedelta(minutes=13),
        objective_value=1.0,
        trigger="preassignment",
    )

    assert vehicle.current_request_id == "current-order"
    assert plan.assigned_request_ids == ["current-order"]
    assert plan.reserved_request_ids == ["reserved-order"]
    assert [stop.request_id for stop in plan.stops] == ["reserved-order", "reserved-order"]
