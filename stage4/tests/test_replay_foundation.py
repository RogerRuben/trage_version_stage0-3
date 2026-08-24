from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage4 import replay_foundation as replay


def _orders(gaps_min: list[int]) -> pd.DataFrame:
    starts = [pd.Timestamp("2016-10-31 08:00", tz=replay.TIMEZONE)]
    ends = [starts[0] + pd.Timedelta(minutes=10)]
    for gap in gaps_min:
        starts.append(ends[-1] + pd.Timedelta(minutes=gap))
        ends.append(starts[-1] + pd.Timedelta(minutes=10))
    count = len(starts)
    return pd.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(count)],
            "driver_id": ["d1"] * count,
            "request_time": starts,
            "arrival_time": ends,
            "start_lon": np.arange(count, dtype=float),
            "start_lat": np.arange(count, dtype=float),
            "end_lon": np.arange(count, dtype=float) + 0.1,
            "end_lat": np.arange(count, dtype=float) + 0.1,
            "valid_session_row": [True] * count,
        }
    )


def test_orders_are_sorted_and_fixed_gap_rule_is_strict():
    orders = _orders([90, 91]).iloc[[2, 0, 1]].reset_index(drop=True)
    sessions, gaps, diagnostics = replay.reconstruct_driver_sessions(
        orders, 90, "20161031", 15
    )
    assert sessions["order_count"].tolist() == [2, 1]
    assert sessions["first_order_id"].tolist() == ["o0", "o2"]
    assert diagnostics["session_split_count"] == 1
    assert gaps.tolist() == [5400.0, 5460.0]


def test_negative_gap_does_not_split():
    orders = _orders([0])
    orders.loc[1, "request_time"] = orders.loc[0, "arrival_time"] - pd.Timedelta(
        minutes=1
    )
    sessions, _, diagnostics = replay.reconstruct_driver_sessions(
        orders, 90, "20161031", 15
    )
    assert len(sessions) == 1
    assert diagnostics["negative_gap_count"] == 1


def test_fixed_15_minute_profile_has_96_bins():
    orders = _orders([])
    sessions = replay.reconstruct_driver_sessions(orders, 90, "20161031", 15)[0]
    scaling = replay.build_scaling_profile(orders, orders, sessions, "20161031", 15)
    assert len(scaling) == 96
    assert scaling["time_bin_index"].tolist() == list(range(96))


def test_selected_fleet_preserves_complete_session_windows():
    orders = _orders([])
    sessions = replay.reconstruct_driver_sessions(orders, 90, "20161031", 15)[0]
    scaling = replay.build_scaling_profile(orders, orders, sessions, "20161031", 15)
    fleet = replay.select_fleet_template(sessions, scaling, 20260824)
    source = sessions.set_index("session_id").loc[fleet.iloc[0]["source_session_id"]]
    assert fleet.iloc[0]["availability_start_time"] == source["session_start_time"]
    assert fleet.iloc[0]["availability_end_time"] == source["session_end_time"]


def test_fleet_sampling_is_reproducible_for_seed():
    sessions = pd.concat(
        [
            replay.reconstruct_driver_sessions(
                _orders([]).assign(driver_id=f"d{i}"), 90, "20161031", 15
            )[0]
            for i in range(8)
        ],
        ignore_index=True,
    )
    scaling = replay.build_scaling_profile(
        _orders([]), _orders([]), sessions, "20161031", 15
    )
    first = replay.select_fleet_template(sessions, scaling, 123)
    second = replay.select_fleet_template(sessions, scaling, 123)
    pd.testing.assert_frame_equal(first, second)


def test_eta_multiplier_uses_bin_median_ratio():
    time = pd.Timestamp("2016-10-31 08:02", tz=replay.TIMEZONE)
    orders = pd.DataFrame(
        {
            "request_time": [time, time, time],
            "realized_service_time_s": [100.0, 200.0, 900.0],
            "valhalla_route_time_s": [100.0, 100.0, 100.0],
        }
    )
    calibration, _ = replay.build_eta_calibration(orders, "20161031", 15, 3)
    row = calibration.loc[calibration["time_bin_index"].eq(32)].iloc[0]
    assert row["selected_eta_multiplier"] == 2.0
    assert row["fallback_level"] == "BIN_MEDIAN"


def test_eta_low_sample_falls_back_to_hour_then_global():
    orders = pd.DataFrame(
        {
            "request_time": [
                pd.Timestamp("2016-10-31 08:02", tz=replay.TIMEZONE),
                pd.Timestamp("2016-10-31 08:32", tz=replay.TIMEZONE),
            ],
            "realized_service_time_s": [100.0, 300.0],
            "valhalla_route_time_s": [100.0, 100.0],
        }
    )
    calibration, _ = replay.build_eta_calibration(orders, "20161031", 15, 30)
    hour_row = calibration.loc[calibration["time_bin_index"].eq(33)].iloc[0]
    global_row = calibration.loc[calibration["time_bin_index"].eq(0)].iloc[0]
    assert hour_row["fallback_level"] == "HOUR_MEDIAN"
    assert hour_row["selected_eta_multiplier"] == 2.0
    assert global_row["fallback_level"] == "GLOBAL_MEDIAN"
    assert global_row["selected_eta_multiplier"] == 2.0


def test_config_is_foundation_only(tmp_path: Path):
    payload = {
        "test_date": "20161031",
        "session_gap_split_min": 90,
        "time_bin_min": 15,
        "fleet_sampling_seed": 20260824,
        "pickup_eta_min_bin_sample": 30,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert set(replay.load_config(path)) == replay.CONFIG_KEYS
    forbidden = {"Gamma", "passenger_acceptance", "av_penetration", "dispatch_solver"}
    assert not forbidden & set(payload)
