from __future__ import annotations

import pandas as pd
import pytest

from stage2.v5_2.contracts import Stage2V52ContractError
from stage2.v5_2.micro_products import build_micro_condition_tokens
from stage2.v5_2.transfer_data import _temporal_source


def _inputs(cutoff: float):
    predictions = pd.DataFrame({
        "split": ["evaluation"], "date": ["20161025"], "order_id": ["a"],
        "traversal_id": [1], "allocated_distance_m": [10.0],
        "pred_crawl_time_share": [0.1], "pred_stop_time_share": [0.2],
        "pred_speed_cv_bounded": [0.3], "pred_acceleration_rms_bounded": [0.4],
        "pred_rts_raw": [0.5], "pace_pred_p50": [0.1],
    })
    context = pd.DataFrame({
        "split": ["evaluation"], "date": ["20161025"], "order_id": ["a"],
        "traversal_id": [1], "route_sequence": [0],
        "observed_directed_edge_uid": ["e1"], "canonical_edge_uid": ["c1"],
        "decision_time": [100.0], "feature_cutoff_time": [cutoff], "history_support": [1],
        "route_track": ["historical_original_service_route"],
        "route_source": ["frozen_stage1_route_parts"],
        "route_product_version": ["stage1_v3_route_sequence_context.1"],
    })
    support = {
        "evaluation_support_used": False, "counts": {"e1": 1}, "group_boundaries": [1, 2],
    }
    return predictions, context, support


@pytest.mark.parametrize("cutoff", [100.0, 101.0])
def test_cutoff_equal_to_or_after_decision_fails(cutoff: float) -> None:
    predictions, context, support = _inputs(cutoff)
    with pytest.raises(Stage2V52ContractError, match="strictly earlier"):
        build_micro_condition_tokens(
            predictions, context, support_artifact=support, protocol_id="development",
            prediction_source="model", model_id="M4", model_hash="fixture",
        )


def test_no_history_temporal_provenance_gets_explicit_strict_fallback() -> None:
    frame = pd.DataFrame({
        "decision_time": [100.0], "feature_age_s": [float("nan")],
        "feature_time_check": ["NO_HISTORY"], "history_count": [0],
        "dynamic_available_mask": [False],
    })
    decision, cutoff, age, fallback = _temporal_source(frame)
    assert decision.tolist() == [100.0]
    assert cutoff.tolist() == [99.0]
    assert age.tolist() == [1.0]
    assert fallback.tolist() == [True]


def test_unproven_missing_temporal_provenance_still_fails_closed() -> None:
    frame = pd.DataFrame({"decision_time": [100.0], "feature_age_s": [float("nan")]})
    with pytest.raises(Stage2V52ContractError, match="violating"):
        _temporal_source(frame)
