from __future__ import annotations

import numpy as np
import pandas as pd

from stage2.v5_2.micro_products import DIMENSIONS, aggregate_original_route_micro_conditions
from stage2.v5_2.performance import static_complexity_audit


def _reference(frame: pd.DataFrame, threshold: float) -> tuple[float, float, float]:
    weights = frame["estimated_travel_time_p50_s"].to_numpy(float)
    values = frame["pred_crawl_share"].to_numpy(float)
    mean = float(np.average(values, weights=weights))
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    p90 = float(values[order][np.flatnonzero(cumulative >= 0.9 * weights.sum())[0]])
    high = float(weights[values >= threshold].sum() / weights.sum())
    return mean, p90, high


def test_optimized_aggregation_matches_small_reference() -> None:
    frame = pd.DataFrame({
        "split": "x", "date": "20161025", "order_id": "a",
        "route_sequence": np.arange(4), "estimated_travel_time_p50_s": [1.0, 2.0, 3.0, 4.0],
        "allocated_distance_m": 10.0, "edge_train_support": [0, 1, 2, 3],
        "support_group": ["unseen", "low", "medium", "high"],
    })
    for column in DIMENSIONS.values():
        frame[column] = [0.1, 0.2, 0.8, 0.9]
    cdf = {"fit_split": "train", "evaluation_rows_used": 0, "thresholds": {name: 0.8 for name in DIMENSIONS}}
    actual = aggregate_original_route_micro_conditions(frame, cdf).iloc[0]
    expected = _reference(frame, 0.8)
    assert np.allclose(
        [actual["crawl_weighted_mean"], actual["crawl_weighted_p90"], actual["crawl_high_exposure_share"]],
        expected,
    )


def test_v5_2_source_has_no_prohibited_dataframe_patterns() -> None:
    report = static_complexity_audit("stage2/v5_2")
    assert report["status"] == "PASS", report["findings"]
