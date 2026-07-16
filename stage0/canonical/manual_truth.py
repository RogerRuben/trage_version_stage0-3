"""Schema and acceptance metrics for versioned manual route review."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


REVIEW_COLUMNS = [
    "order_id",
    "reviewer_id",
    "review_status",
    "review_class",
    "route_correct",
    "major_error",
    "minor_error",
    "wrong_road_level",
    "wrong_direction",
    "wrong_parallel_road",
    "wrong_bridge_or_tunnel",
    "unreasonable_detour",
    "od_endpoint_error",
    "data_limitation",
    "comments",
]

ALLOWED_REVIEW_STATUS = {"pending", "completed", "needs_adjudication"}
ALLOWED_REVIEW_CLASS = {
    "Correct",
    "Minor error",
    "Major error",
    "Uncertain / data limitation",
}


def validate_review_schema(frame: pd.DataFrame) -> list[str]:
    """Return validation errors without inventing missing review judgments."""

    errors: list[str] = []
    missing = [column for column in REVIEW_COLUMNS if column not in frame]
    if missing:
        errors.append(f"missing_columns:{','.join(missing)}")
        return errors
    invalid_status = set(frame.review_status.dropna().astype(str)) - ALLOWED_REVIEW_STATUS
    if invalid_status:
        errors.append(f"invalid_review_status:{','.join(sorted(invalid_status))}")
    completed = frame.review_status.eq("completed")
    if frame.loc[completed, "reviewer_id"].isna().any():
        errors.append("completed_review_missing_reviewer_id")
    if frame.loc[completed, "route_correct"].isna().any():
        errors.append("completed_review_missing_route_correct")
    if frame.loc[completed, "review_class"].isna().any():
        errors.append("completed_review_missing_review_class")
    invalid_class = set(frame.loc[completed, "review_class"].dropna().astype(str)) - ALLOWED_REVIEW_CLASS
    if invalid_class:
        errors.append(f"invalid_review_class:{','.join(sorted(invalid_class))}")
    return errors


def review_metrics(reviews: pd.DataFrame, core_order_ids: Iterable[str]) -> dict:
    """Calculate acceptance metrics from completed reviews only."""

    errors = validate_review_schema(reviews)
    if errors:
        raise ValueError(";".join(errors))
    completed = reviews.loc[reviews.review_status.eq("completed")].copy()
    completed["order_id"] = completed.order_id.astype(str)
    completed["route_correct"] = completed.route_correct.astype("boolean")
    completed["major_error"] = completed.major_error.astype("boolean")
    core = set(map(str, core_order_ids))
    core_reviews = completed.loc[completed.order_id.isin(core)]
    rejected_reviews = completed.loc[~completed.order_id.isin(core)]
    def share(frame: pd.DataFrame, column: str) -> float | None:
        return float(frame[column].astype("boolean").mean()) if len(frame) else None

    return {
        "completed_reviews": int(len(completed)),
        "core_completed_reviews": int(len(core_reviews)),
        "core_precision": (
            float(core_reviews.route_correct.mean()) if len(core_reviews) else None
        ),
        "core_false_positive_rate": (
            float((~core_reviews.route_correct).mean()) if len(core_reviews) else None
        ),
        "core_major_error_rate": share(core_reviews, "major_error"),
        "core_wrong_direction_rate": share(core_reviews, "wrong_direction"),
        "core_wrong_road_level_rate": (
            float(
                (
                    core_reviews.wrong_road_level.astype("boolean")
                    | core_reviews.wrong_bridge_or_tunnel.astype("boolean")
                ).mean()
            ) if len(core_reviews) else None
        ),
        "core_unreasonable_detour_rate": share(core_reviews, "unreasonable_detour"),
        "core_correct_rate": (
            float(core_reviews.review_class.eq("Correct").mean()) if len(core_reviews) else None
        ),
        "core_minor_error_rate": (
            float(core_reviews.review_class.eq("Minor error").mean()) if len(core_reviews) else None
        ),
        "core_data_limitation_rate": (
            float(core_reviews.review_class.eq("Uncertain / data limitation").mean())
            if len(core_reviews) else None
        ),
        "rejected_correct_share": (
            float(rejected_reviews.route_correct.mean()) if len(rejected_reviews) else None
        ),
        "rejected_usable_share": (
            float(
                rejected_reviews.review_class.isin(["Correct", "Minor error"]).mean()
            ) if len(rejected_reviews) else None
        ),
    }
