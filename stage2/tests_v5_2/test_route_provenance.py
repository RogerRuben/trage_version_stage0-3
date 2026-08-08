from __future__ import annotations

import pandas as pd
import pytest

from stage2.v5_2.contracts import Stage2V52ContractError
from stage2.v5_2.micro_products import build_micro_condition_tokens


def _inputs():
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
        "decision_time": [100.0], "feature_cutoff_time": [90.0], "history_support": [1],
        "route_track": ["planned_route"], "route_source": ["planner"],
        "route_product_version": ["oracle.1"],
    })
    support = {
        "evaluation_support_used": False, "counts": {"e1": 1}, "group_boundaries": [1, 2],
    }
    return predictions, context, support


@pytest.mark.parametrize("route_track", ["planned_route", "fallback_route", "oracle_route"])
def test_non_original_route_provenance_fails(route_track: str) -> None:
    predictions, context, support = _inputs()
    context["route_track"] = route_track
    with pytest.raises(Stage2V52ContractError, match="not a frozen original-route"):
        build_micro_condition_tokens(
            predictions, context, support_artifact=support, protocol_id="development",
            prediction_source="model", model_id="M4", model_hash="fixture",
        )
