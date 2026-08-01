from __future__ import annotations

import json

import numpy as np
import pandas as pd

from stage2.v4.aggregation import aggregate_route_dimension
from stage2.v4.metrics import (
    binary_metrics,
    continuous_metrics,
    decision_weighted_order_aggregation,
    order_cluster_bootstrap_ci,
)


def test_decision_aggregation_uses_estimated_time_for_lcs_and_length_for_rts() -> None:
    frame = pd.DataFrame(
        {
            "split": ["test", "test"],
            "date": ["20161031", "20161031"],
            "order_id": ["o1", "o1"],
            "estimated_travel_time_s": [1.0, 3.0],
            "route_part_length_m": [3.0, 1.0],
            "prediction": [0.0, 1.0],
            "valid": [True, True],
        }
    )
    lcs = decision_weighted_order_aggregation(
        frame,
        value_column="prediction",
        dimension="lcs",
        mask_column="valid",
    )
    rts = decision_weighted_order_aggregation(
        frame,
        value_column="prediction",
        dimension="rts",
        mask_column="valid",
    )
    assert np.isclose(lcs["prediction"].iloc[0], 0.75)
    assert np.isclose(rts["prediction"].iloc[0], 0.25)


def test_undefined_metrics_are_strict_json_nulls() -> None:
    continuous = continuous_metrics(
        np.array([1.0, 2.0]),
        np.array([1.0, 1.0]),
    )
    binary = binary_metrics(
        np.array([], dtype=float),
        np.array([], dtype=float),
    )
    bootstrap = order_cluster_bootstrap_ci(
        pd.DataFrame(columns=["order_id", "truth", "prediction"]),
        truth_column="truth",
        prediction_column="prediction",
    )
    payload = {
        "continuous": continuous,
        "binary": binary,
        "bootstrap": bootstrap,
    }
    encoded = json.dumps(payload, allow_nan=False)
    assert '"pearson": null' in encoded
    assert '"average_precision": null' in encoded
    assert bootstrap["mae_ci95"] == [None, None]


def test_vectorized_route_aggregation_preserves_sequence_contract() -> None:
    frame = pd.DataFrame(
        {
            "split": ["test"] * 5,
            "date": ["20161031"] * 5,
            "order_id": ["o1"] * 5,
            "route_sequence": [0, 1, 2, 3, 4],
            "route_position_ratio": [0.1, 0.3, 0.7, 0.9, 1.0],
            "pred_lcs_pct": [0.95, 0.95, 0.2, 0.95, np.nan],
            "lcs_tail_probability_calibrated": [0.8, 0.7, 0.1, 0.6, np.nan],
            "estimated_travel_time_s": [1.0, 2.0, 1.0, 1.0, 3.0],
        }
    )
    result = aggregate_route_dimension(
        frame,
        dimension="lcs",
        percentile_column="pred_lcs_pct",
        tail_probability_column="lcs_tail_probability_calibrated",
    ).iloc[0]
    assert np.isclose(result["lcs_weighted_mean"], 0.8)
    assert np.isclose(result["lcs_max"], 0.95)
    assert np.isclose(result["lcs_tail_mean"], 0.58)
    assert np.isclose(result["lcs_tail_persistence"], 0.5)
    assert np.isclose(result["lcs_coverage_share"], 0.8)
    assert result["lcs_tail_event_present"]
    assert result["lcs_top5_route_positions"] == "[0, 1, 3, 2]"
    assert np.isclose(result["lcs_pickup_side_exposure"], 0.95)
    assert np.isclose(result["lcs_dropoff_side_exposure"], 0.95)
