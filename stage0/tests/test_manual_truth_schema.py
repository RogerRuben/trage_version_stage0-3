import pandas as pd

from stage0.canonical.manual_truth import REVIEW_COLUMNS, review_metrics, validate_review_schema


def review_frame():
    frame = pd.DataFrame({column: [pd.NA, pd.NA] for column in REVIEW_COLUMNS})
    frame["order_id"] = ["a", "b"]
    frame["reviewer_id"] = ["r1", "r2"]
    frame["review_status"] = ["completed", "completed"]
    frame["review_class"] = ["Correct", "Major error"]
    frame["route_correct"] = [True, False]
    frame["major_error"] = [False, True]
    for column in [
        "minor_error", "wrong_road_level", "wrong_direction", "wrong_parallel_road",
        "wrong_bridge_or_tunnel", "unreasonable_detour", "od_endpoint_error",
        "data_limitation",
    ]:
        frame[column] = [False, False]
    return frame


def test_manual_truth_schema_accepts_completed_reviews():
    assert validate_review_schema(review_frame()) == []


def test_manual_truth_schema_rejects_missing_columns():
    errors = validate_review_schema(review_frame().drop(columns=["wrong_direction"]))
    assert errors == ["missing_columns:wrong_direction"]


def test_manual_truth_metrics_are_computed_not_assumed():
    metrics = review_metrics(review_frame(), ["a"])
    assert metrics["core_precision"] == 1.0
    assert metrics["rejected_correct_share"] == 0.0


def test_manual_major_error_metrics_are_computed():
    metrics = review_metrics(review_frame(), ["a"])
    assert metrics["core_major_error_rate"] == 0.0
    assert metrics["core_wrong_direction_rate"] == 0.0
