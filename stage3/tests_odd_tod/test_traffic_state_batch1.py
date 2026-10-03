import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage3.scripts.traffic_state_batch1 import EDGE, ORDER, aggregate_day, classify, prepare_events


def config():
    return json.loads(Path("stage3/config/traffic_state_batch1.json").read_text())


def test_proxy_rules_and_unknown_are_not_zero_imputation():
    frame = pd.DataFrame({"joint_order_count": [3]*6, "reference_order_days": [100]*6,
        "reference_days": [5]*6, "joint_event_span_s": [500]*6,
        "latest_joint_age_s": [60]*6, "pace_ratio": [1., 1.6, 2.5, 2.5, np.nan, 1.],
        "joint_low_speed_share": [.1, .6, .9, .1, .1, .1]})
    frame.loc[5, "joint_order_count"] = 1
    assert classify(frame, config()).tolist() == ["NON_CONGESTED_PROXY", "CONGESTED_PROXY",
        "SEVERE_CONGESTION_PROXY", "MIXED_EVIDENCE", "UNKNOWN", "UNKNOWN"]


def test_direction_order_weight_and_strict_completion_window():
    frame = pd.DataFrame({EDGE: ["e:F", "e:F", "e:F", "e:R"],
        ORDER: ["a", "a", "b", "c"], "date": ["20161009"]*4,
        "traversal_id": [1, 2, 3, 4], "availability_timestamp": [100., 150., 1700., 1800.],
        "canonical_highway": ["primary"]*4, "observed_sec_per_m": [.1, .1, .5, .9],
        "crawl_time_share": [0., 0., 1., 0.], "stop_time_share": [0.]*4,
        "speed_cv_bounded": [np.nan]*4, "acceleration_rms_bounded": [np.nan]*4})
    prepared = prepare_events(frame, 1800)
    assert prepared.window_end.tolist() == [1800, 1800, 1800, 3600]
    cells = aggregate_day(prepared).set_index(EDGE)
    assert cells.loc["e:F", "joint_order_count"] == 2
    assert abs(cells.loc["e:F", "joint_pace"] - .3) < 1e-12
    assert cells.loc["e:F", "joint_low_speed_share"] == .5
    assert np.isnan(cells.loc["e:F", "acceleration_rms_bounded_order_mean"])


def test_scope_excludes_test_and_thresholds_are_not_profiles():
    cfg = config()
    assert max(cfg["train_dates"]) < min(cfg["evaluation_dates"])
    assert "20161031" not in cfg["train_dates"] + cfg["evaluation_dates"]
    assert cfg["status"] == "EXPLORATORY_NOT_DEPLOYABLE"
