"""Schema and acceptance metrics for versioned manual route review."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


REVIEW_COLUMNS = [
    "order_id",
    "reviewer_id",
    "review_status",
    "route_correct",
    "major_error",
    "minor_error",
    "wrong_road_level",
    "wrong_direction",
    "wrong_parallel_road",
    "wrong_bridge_or_tunnel",
    "unreasonable_detour",
    "od_endpoint_error",
    "comments",
]

ALLOWED_REVIEW_STATUS = {"pending", "completed", "needs_adjudication"}


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
    return errors


def review_metrics(reviews: pd.DataFrame, core_order_ids: Iterable[str]) -> dict:
    """Calculate acceptance metrics from completed reviews only."""

    errors = validate_review_schema(reviews)
    if errors:
        raise ValueError(";".join(errors))
    completed = reviews.loc[reviews.review_status.eq("completed")].copy()
    completed["order_id"] = completed.order_id.astype(str)
    completed["route_correct"] = completed.route_correct.astype("boolean")
    core = set(map(str, core_order_ids))
    core_reviews = completed.loc[completed.order_id.isin(core)]
    rejected_reviews = completed.loc[~completed.order_id.isin(core)]
    return {
        "completed_reviews": int(len(completed)),
        "core_completed_reviews": int(len(core_reviews)),
        "core_precision": (
            float(core_reviews.route_correct.mean()) if len(core_reviews) else None
        ),
        "core_false_positive_rate": (
            float((~core_reviews.route_correct).mean()) if len(core_reviews) else None
        ),
        "rejected_correct_share": (
            float(rejected_reviews.route_correct.mean()) if len(rejected_reviews) else None
        ),
    }
