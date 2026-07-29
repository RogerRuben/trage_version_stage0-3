"""Order-level eligibility gate for trajectory-modeling products."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


ELIGIBILITY_COLUMNS = [
    "order_id",
    "modeling_eligible",
    "modeling_exclusion_reasons",
    "valid_point_count",
    "unique_location_count",
    "raw_gps_distance_m",
    "resolved_gps_distance_m",
    "resolved_gps_distance_share",
    "maximum_time_gap_s",
    "maximum_step_distance_m",
    "unobserved_movement_gap_count",
    "unobserved_movement_distance_m",
    "unobserved_movement_distance_share",
    "preprocess_break_count",
    "usable_subtrace_count",
]


def evaluate_modeling_eligibility(
    points: pd.DataFrame,
    preprocess_metrics: dict[str, Any],
    *,
    minimum_valid_points: int = 10,
    minimum_unique_locations: int = 5,
    minimum_raw_gps_distance_m: float = 100.0,
    unobserved_gap_minimum_time_s: float = 20.0,
    unobserved_gap_minimum_distance_m: float = 300.0,
    maximum_unobserved_movement_share: float = 0.20,
) -> dict[str, Any]:
    """Classify whether an order contains enough observed movement to model.

    A long pause alone is not missing movement. An interval is treated as an
    unobserved movement gap only when both its elapsed time and displacement
    cross the configured thresholds.
    """

    order_id = str(points.order_id.iloc[0]) if len(points) else ""
    valid_point_count = int(preprocess_metrics.get("valid_point_count", len(points)))
    usable_subtrace_count = int(preprocess_metrics.get("usable_subtrace_count", 0))
    raw_distance = float(preprocess_metrics.get("raw_order_gps_distance_m", 0.0))
    resolved_distance = float(
        preprocess_metrics.get("resolved_subtrace_gps_distance_m", 0.0)
    )

    if len(points):
        coordinate_columns = (
            ["matching_lon", "matching_lat"]
            if {"matching_lon", "matching_lat"}.issubset(points.columns)
            else ["lon", "lat"]
        )
        unique_location_count = int(points[coordinate_columns].drop_duplicates().shape[0])
        time_gaps = pd.to_numeric(points.get("time_gap_s"), errors="coerce").fillna(0.0)
        step_distances = pd.to_numeric(
            points.get("step_distance_m"), errors="coerce"
        ).fillna(0.0)
    else:
        unique_location_count = 0
        time_gaps = pd.Series(dtype=float)
        step_distances = pd.Series(dtype=float)

    unobserved = time_gaps.ge(unobserved_gap_minimum_time_s) & step_distances.ge(
        unobserved_gap_minimum_distance_m
    )
    unobserved_distance = float(step_distances.loc[unobserved].sum())
    unobserved_share = unobserved_distance / raw_distance if raw_distance > 0 else 0.0
    resolved_share = resolved_distance / raw_distance if raw_distance > 0 else 0.0

    reasons: list[str] = []
    if valid_point_count < minimum_valid_points:
        reasons.append("INSUFFICIENT_VALID_POINTS")
    if unique_location_count < minimum_unique_locations:
        reasons.append("INSUFFICIENT_UNIQUE_LOCATIONS")
    if raw_distance < minimum_raw_gps_distance_m:
        reasons.append("LOW_TOTAL_MOVEMENT")
    if bool(unobserved.any()):
        reasons.append("LARGE_UNOBSERVED_MOVEMENT_GAP")
    if unobserved_share > maximum_unobserved_movement_share:
        reasons.append("EXCESSIVE_UNOBSERVED_MOVEMENT_SHARE")
    if usable_subtrace_count == 0:
        reasons.append("NO_USABLE_SUBTRACE")

    return {
        "order_id": order_id,
        "modeling_eligible": not reasons,
        "modeling_exclusion_reasons": "|".join(reasons),
        "valid_point_count": valid_point_count,
        "unique_location_count": unique_location_count,
        "raw_gps_distance_m": raw_distance,
        "resolved_gps_distance_m": resolved_distance,
        "resolved_gps_distance_share": float(np.clip(resolved_share, 0.0, 1.0)),
        "maximum_time_gap_s": float(time_gaps.max()) if len(time_gaps) else 0.0,
        "maximum_step_distance_m": (
            float(step_distances.max()) if len(step_distances) else 0.0
        ),
        "unobserved_movement_gap_count": int(unobserved.sum()),
        "unobserved_movement_distance_m": unobserved_distance,
        "unobserved_movement_distance_share": float(
            np.clip(unobserved_share, 0.0, 1.0)
        ),
        "preprocess_break_count": int(
            preprocess_metrics.get("preprocess_break_count", 0)
        ),
        "usable_subtrace_count": usable_subtrace_count,
    }
