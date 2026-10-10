from __future__ import annotations

import numpy as np
import pandas as pd

from stage2.v5_2.micro_products import DIMENSIONS, aggregate_original_route_micro_conditions


def test_missing_pace_reduces_distance_coverage_and_hides_formal_travel_time() -> None:
    frame = pd.DataFrame({
        "split": ["evaluation"] * 2, "date": ["20161025"] * 2, "order_id": ["a"] * 2,
        "route_sequence": [0, 1], "allocated_distance_m": [90.0, 10.0],
        "estimated_travel_time_p50_s": [np.nan, 2.0], "edge_train_support": [1, 1],
        "support_group": ["low", "low"], "protocol_id": ["development"] * 2,
        "model_id": ["M4"] * 2, "prediction_source": ["fixture"] * 2,
        "route_track": ["historical_original_service_route"] * 2,
        "route_source": ["frozen_stage1_route_parts"] * 2,
        "route_product_version": ["stage1_v3_route_sequence_context.1"] * 2,
    })
    for column in DIMENSIONS.values():
        frame[column] = 0.5
    cdf = {
        "fit_split": "train", "evaluation_rows_used": 0, "protocol_id": "development",
        "model_id": "M4", "prediction_source": "fixture",
        "thresholds": {name: 0.8 for name in DIMENSIONS},
    }
    result = aggregate_original_route_micro_conditions(frame, cdf, minimum_coverage=0.8).iloc[0]
    assert np.isclose(result["pace_prediction_coverage_distance"], 0.1)
    assert np.isnan(result["travel_time_p50_s"])
    assert result["partial_travel_time_p50_s"] == 2.0
    assert bool(result["unknown_flag"])
