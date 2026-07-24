import numpy as np
import pandas as pd

from stage4.simulator_v3.entities import VehiclePlan, VehicleState
from stage4.simulator_v3.enums import StopType, VehicleExecutionStatus
from stage4.simulator_v3.idle_management import IdleMovementManager, IdleMovementPolicy


def vehicle(vehicle_id: str, vehicle_type: str, zone: str) -> VehicleState:
    now = pd.Timestamp("2016-10-23T08:00:00Z")
    return VehicleState(
        vehicle_id,
        vehicle_type,
        108.91,
        34.21,
        zone,
        now - pd.Timedelta(hours=1),
        now + pd.Timedelta(hours=8),
        VehicleExecutionStatus.IDLE,
        None,
        VehiclePlan(vehicle_id, 0, [], now, "init", True, 0.0),
        0,
    )


def test_hv_destination_comes_from_training_transition_and_returns_plan():
    table = pd.DataFrame([
        {
            "origin_zone": "z0_0",
            "destination_zone": "z1_0",
            "time_bin": 16,
            "idle_duration_bin": "05_15",
            "transition_probability": 1.0,
            "sample_count": 40,
            "uses_test_day_future_demand": False,
        }
    ])
    manager = IdleMovementManager(
        {"grid_size": 0.02, "min_lon": 108.9, "min_lat": 34.2},
        hv_transition_table=table,
        policy=IdleMovementPolicy(max_share_per_epoch=1.0, min_hv_idle_sec=300),
        rng=np.random.default_rng(7),
    )
    hv = vehicle("hv-1", "HV", "z0_0")
    manager.build_plans([hv], pd.Timestamp("2016-10-23T08:00:00Z"), "HV")
    decisions = manager.build_plans([hv], pd.Timestamp("2016-10-23T08:05:00Z"), "HV")
    assert len(decisions) == 1
    assert decisions[0].plan.stops[0].stop_type == StopType.HV_REPOSITION
    assert decisions[0].plan.stops[0].zone == "z1_0"
    assert manager.records[-1]["policy_source"] == "origin_time_idle"
    assert manager.records[-1]["training_sample_count"] == 40


def test_hv_policy_rejects_test_day_future_demand_flag():
    table = pd.DataFrame([{
        "origin_zone": "z0_0", "destination_zone": "z1_0", "time_bin": 16,
        "idle_duration_bin": "05_15", "transition_probability": 1.0,
        "sample_count": 1, "uses_test_day_future_demand": True,
    }])
    try:
        IdleMovementManager(
            {"grid_size": 0.02, "min_lon": 108.9, "min_lat": 34.2},
            hv_transition_table=table,
        )
    except ValueError as error:
        assert "future-demand" in str(error)
    else:
        raise AssertionError("test-day future-demand table must be rejected")
