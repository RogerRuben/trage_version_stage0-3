import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage3.scripts.traffic_state_batch2 import (
    DIMS, METRICS, dynamic_ratios, exposure_by_route, prediction_proxy, recover_measurements,
)


def test_recovery_measures_partial_traversal_not_full_link():
    intervals = pd.DataFrame({"order_id": ["x", "x"], "traversal_id": [0, 0],
        "gps_interval_id": [1, 2], "measurement_source": ["direct_observed"]*2,
        "label_valid": [True]*2, "interval_start_time": [0., 4.], "interval_end_time": [2., 6.],
        "observed_travel_time_s": [2., 2.], "observed_distance_m": [10., 10.],
        "canonical_edge_uid": ["e", "e"]})
    physical = pd.DataFrame({"order_id": ["x"], "traversal_id": [0],
        "canonical_edge_uid": ["e"], "allocated_distance_m": [40.]})
    row = recover_measurements(intervals, physical).iloc[0]
    assert row.direct_interval_count == 2
    assert row.traversal_distance_coverage == .5
    assert row.window_time_coverage == 4/6
    assert row.maximum_internal_gap_s == 2


def test_prediction_proxy_requires_reference_not_invented_current_orders():
    cfg = json.loads(Path("stage3/config/traffic_state_batch1.json").read_text())
    frame = pd.DataFrame({"pred_pace_p50": [.1, .3, .3], "reference_pace_q20": [.1, .1, np.nan],
        "reference_order_days": [100, 100, 0], "reference_days": [5, 5, 0],
        "pred_crawl": [.1, .8, .8], "pred_stop": [0., .1, .1]})
    assert prediction_proxy(frame, cfg).tolist() == ["NON_CONGESTED_PROXY", "SEVERE_CONGESTION_PROXY", "UNKNOWN"]


def test_unknown_and_missing_sequence_break_continuous_exposure():
    f = pd.DataFrame({"order_id": ["o"]*4, "route_sequence": [0, 1, 3, 4],
        "state": ["CONGESTED_PROXY", "UNKNOWN", "CONGESTED_PROXY", "SEVERE_CONGESTION_PROXY"],
        "w": [5., 5., 5., 5.]})
    row = exposure_by_route(f, "w", "p_").iloc[0]
    assert row.p_unknown_share == .25
    assert row.p_congested_or_severe_share == .75
    assert row.p_longest_congested_run_s == 10


def test_caps_nestedness_and_family_decomposition():
    profiles = {"profiles": [{"profile_id": p, "dynamic_caps": {d: {m: c for m in ("E", "Q", "C")} for d in DIMS}}
        for p, c in [("C", 1.), ("M", 2.), ("A", 3.)]]}
    x = pd.DataFrame({m: [0.5] for m in METRICS})
    x["speed_cv_E"] = 1.5
    y = dynamic_ratios(x, profiles).iloc[0]
    assert y.rho_dynamic_C > y.rho_dynamic_M > y.rho_dynamic_A
    assert not y.crawl_stop_exceeds_C and y.variability_exceeds_C
