from __future__ import annotations

import numpy as np
import pandas as pd

from stage2.v5_2.micro_products import (
    DIMENSIONS,
    aggregate_original_route_micro_conditions,
    aggregate_static_route_complexity,
    weighted_quantile_by_group,
)


def _tokens() -> pd.DataFrame:
    frame = pd.DataFrame({
        "split": "evaluation",
        "date": "20161025",
        "order_id": ["a", "a", "a"],
        "route_sequence": [0, 1, 2],
        "estimated_travel_time_p50_s": [1.0, 1.0, 8.0],
        "allocated_distance_m": [10.0, 10.0, 10.0],
        "edge_train_support": [0, 2, 100],
        "support_group": ["unseen", "low", "high"],
        "protocol_id": "development",
        "model_id": "M4",
        "prediction_source": "fixture",
        "route_track": "historical_original_service_route",
        "route_source": "frozen_stage1_route_parts",
        "route_product_version": "stage1_v3_route_sequence_context.1",
        "canonical_highway": ["primary", "primary", "secondary"],
        "road_class": ["primary", "primary", "secondary"],
        "bridge": [False, True, False],
        "tunnel": [False, False, False],
    })
    for column in DIMENSIONS.values():
        frame[column] = [0.1, 0.5, 0.9]
    return frame


def test_weighted_empirical_p90_is_not_unweighted_quantile() -> None:
    actual = weighted_quantile_by_group(
        np.array([0, 0, 0]), np.array([0.1, 0.5, 0.9]), np.array([1.0, 1.0, 8.0]), 1,
        quantile=0.90,
    )
    assert actual.tolist() == [0.9]


def test_route_aggregation_uses_time_weights_train_cdf_and_route_order() -> None:
    cdf = {
        "fit_split": "train",
        "evaluation_rows_used": 0,
        "protocol_id": "development",
        "model_id": "M4",
        "prediction_source": "fixture",
        "thresholds": {name: 0.8 for name in DIMENSIONS},
    }
    result = aggregate_original_route_micro_conditions(_tokens(), cdf).iloc[0]
    assert np.isclose(result["crawl_weighted_mean"], 0.78)
    assert result["crawl_weighted_p90"] == 0.9
    assert np.isclose(result["crawl_high_exposure_share"], 0.8)
    assert np.isclose(result["crawl_max_consecutive_high_share"], 0.8)
    assert result["micro_condition_coverage"] == 1.0
    assert result["unknown_flag"] == False  # noqa: E712


def test_missing_micro_value_remains_missing_and_reduces_coverage() -> None:
    frame = _tokens()
    frame.loc[:, "pred_rts_raw"] = np.nan
    cdf = {
        "fit_split": "train", "evaluation_rows_used": 0, "protocol_id": "development",
        "model_id": "M4", "prediction_source": "fixture",
        "thresholds": {name: 0.8 for name in DIMENSIONS},
    }
    result = aggregate_original_route_micro_conditions(frame, cdf).iloc[0]
    assert np.isnan(result["rts_weighted_mean"])
    assert result["rts_prediction_coverage"] == 0.0
    assert result["micro_condition_coverage"] == 1.0
    assert result["unknown_flag"] == False  # noqa: E712


def test_missing_core_micro_value_reduces_deployable_coverage() -> None:
    frame = _tokens()
    frame.loc[:, "pred_crawl_share"] = np.nan
    cdf = {
        "fit_split": "train", "evaluation_rows_used": 0, "protocol_id": "development",
        "model_id": "M4", "prediction_source": "fixture",
        "thresholds": {name: 0.8 for name in DIMENSIONS},
    }
    result = aggregate_original_route_micro_conditions(frame, cdf).iloc[0]
    assert result["crawl_prediction_coverage"] == 0.0
    assert result["micro_condition_coverage"] == 0.0
    assert result["unknown_flag"] == True  # noqa: E712


def test_static_unavailable_fields_are_na_not_zero() -> None:
    result = aggregate_static_route_complexity(_tokens()).iloc[0]
    assert pd.isna(result["ramp_exposure_share"])
    assert pd.isna(result["signal_exposure_share"])
    assert np.isclose(result["bridge_exposure_share"], 1 / 3)
    assert np.isclose(result["road_class_transition_rate"], 0.5)
