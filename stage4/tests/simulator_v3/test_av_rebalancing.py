import pandas as pd

from stage4.simulator_v3.entities import VehiclePlan, VehicleState
from stage4.simulator_v3.enums import StopType, VehicleExecutionStatus
from stage4.simulator_v3.idle_management import IdleMovementManager, IdleMovementPolicy


def av(vehicle_id: str, zone: str) -> VehicleState:
    now = pd.Timestamp("2016-10-23T08:00:00Z")
    return VehicleState(
        vehicle_id, "AV", 108.91, 34.21, zone, now, now + pd.Timedelta(hours=8),
        VehicleExecutionStatus.IDLE, None,
        VehiclePlan(vehicle_id, 0, [], now, "init", True, 0.0), 0,
    )


def test_av_rebalancing_uses_training_demand_shortage_and_plan_stop():
    prior = pd.DataFrame([
        {"zone": "z0_0", "time_bin": 16, "forecast_demand": 1.0, "uses_test_day_future_demand": False},
        {"zone": "z1_0", "time_bin": 16, "forecast_demand": 9.0, "uses_test_day_future_demand": False},
    ])
    manager = IdleMovementManager(
        {"grid_size": 0.02, "min_lon": 108.9, "min_lat": 34.2},
        av_demand_prior=prior,
        policy=IdleMovementPolicy(max_share_per_epoch=1.0),
    )
    decisions = manager.build_plans(
        [av("av-1", "z0_0"), av("av-2", "z0_0")],
        pd.Timestamp("2016-10-23T08:00:00Z"),
        "AV",
    )
    assert decisions
    assert all(item.plan.stops[0].stop_type == StopType.AV_REBALANCE for item in decisions)
    assert decisions[0].plan.stops[0].zone == "z1_0"
    record = manager.records[-1]
    assert record["movement_reason"] == "platform_training_demand_rebalancing"
    assert record["forecast_demand"] == 9.0
    assert record["solver"] in {"scipy_linear_sum_assignment", "deterministic_greedy_fallback"}


def test_av_rebalancing_does_not_mutate_vehicle_coordinates():
    prior = pd.DataFrame([
        {"zone": "z0_0", "time_bin": 16, "forecast_demand": 0.0},
        {"zone": "z1_0", "time_bin": 16, "forecast_demand": 1.0},
    ])
    manager = IdleMovementManager(
        {"grid_size": 0.02, "min_lon": 108.9, "min_lat": 34.2},
        av_demand_prior=prior,
        policy=IdleMovementPolicy(max_share_per_epoch=1.0),
    )
    vehicle = av("av-1", "z0_0")
    before = (vehicle.current_lon, vehicle.current_lat, vehicle.current_zone)
    decisions = manager.build_plans([vehicle], pd.Timestamp("2016-10-23T08:00:00Z"), "AV")
    assert decisions
    assert (vehicle.current_lon, vehicle.current_lat, vehicle.current_zone) == before
    assert decisions[0].plan.stops[0].zone == "z1_0"
