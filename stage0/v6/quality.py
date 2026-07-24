"""Small, configurable quality classifier for the v6 feasibility study."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def _quantile(values: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.quantile(q)) if len(numeric) else float("nan")


def evaluate_order_quality(
    source_points: pd.DataFrame,
    matched_points: pd.DataFrame,
    route_parts: pd.DataFrame,
    unresolved_intervals: pd.DataFrame,
    thresholds: dict[str, Any],
    *,
    processing_exception: str | None = None,
) -> dict[str, Any]:
    point_count = len(source_points)
    status = matched_points.get(
        "matched_point_status", pd.Series(["unmatched"] * point_count)
    )
    non_unmatched = status.ne("unmatched")
    matched_share = float(non_unmatched.mean()) if point_count else 0.0
    unmatched_share = float(status.eq("unmatched").mean()) if point_count else 1.0
    interpolated_share = float(status.eq("interpolated").mean()) if point_count else 0.0
    snap = pd.to_numeric(
        matched_points.loc[non_unmatched, "distance_from_trace_point_m"],
        errors="coerce",
    )
    snap_p50, snap_p90, snap_p99 = (_quantile(snap, q) for q in (0.5, 0.9, 0.99))
    od_values = []
    if len(matched_points):
        for row in (matched_points.iloc[0], matched_points.iloc[-1]):
            value = pd.to_numeric(
                pd.Series([row.distance_from_trace_point_m]), errors="coerce"
            ).iloc[0]
            od_values.append(float(value) if pd.notna(value) else float("inf"))
    od_error = max(od_values) if od_values else float("inf")

    route_distance = float(
        pd.to_numeric(route_parts.get("length_m", pd.Series(dtype=float)), errors="coerce")
        .fillna(0)
        .sum()
    )
    gps_distance = float(
        pd.to_numeric(
            source_points.get("step_distance_m", pd.Series(dtype=float)), errors="coerce"
        )
        .fillna(0)
        .sum()
    )
    route_gps_ratio = route_distance / gps_distance if gps_distance > 0 else float("nan")
    mapped_distance = float(
        pd.to_numeric(
            route_parts.loc[
                route_parts.get(
                    "canonical_edge_uid", pd.Series(index=route_parts.index, dtype=object)
                ).notna(),
                "length_m",
            ],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )
    mapping_share = mapped_distance / route_distance if route_distance > 0 else 0.0
    inferred_distance = float(
        pd.to_numeric(
            route_parts.loc[
                route_parts.get(
                    "route_source", pd.Series(index=route_parts.index, dtype=object)
                ).eq("inferred"),
                "length_m",
            ],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )
    inferred_share = inferred_distance / route_distance if route_distance > 0 else 0.0
    duration = max(
        float(source_points.timestamp.max() - source_points.timestamp.min()), 0.0
    ) if len(source_points) else 0.0
    unresolved_time = float(
        pd.to_numeric(
            unresolved_intervals.get(
                "unresolved_interval_time_s", pd.Series(dtype=float)
            ),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )
    unresolved_time_share = unresolved_time / duration if duration > 0 else 0.0
    matched_interval_time = 0.0
    for _, group in matched_points.groupby("subtrace_id", sort=False):
        ordered = group.sort_values(["timestamp", "original_point_seq"], kind="stable")
        rows = list(ordered.itertuples(index=False))
        for left, right in zip(rows, rows[1:]):
            valid = (
                left.matched_point_status != "unmatched"
                and right.matched_point_status != "unmatched"
                and not bool(left.route_discontinuity)
                and not bool(right.route_discontinuity)
            )
            if valid:
                matched_interval_time += max(float(right.timestamp - left.timestamp), 0.0)
    matched_interval_share = (
        min(max(matched_interval_time / duration, 0.0), 1.0) if duration > 0 else 0.0
    )
    discontinuities = int(
        matched_points.get(
            "route_discontinuity", pd.Series(dtype=bool)
        ).fillna(False).sum()
    )
    subtrace_count = int(source_points.subtrace_id.nunique()) if len(source_points) else 0

    strict = thresholds["strict"]
    analysis = thresholds["analysis"]
    has_route = route_distance > 0 and len(route_parts) > 0
    strict_checks = {
        "has_valid_route": has_route,
        "no_discontinuity": discontinuities == 0,
        "matched_interval_share": matched_interval_share
        >= float(strict["minimum_matched_interval_share"]),
        "snap_distance": np.isfinite(snap_p90)
        and snap_p90 <= float(strict["maximum_snap_distance_p90_m"]),
        "canonical_mapping": mapping_share
        >= float(strict["minimum_canonical_mapping_share"]),
        "inferred_distance": inferred_share
        <= float(strict["maximum_inferred_distance_share"]),
        "od_endpoint": od_error <= float(strict["maximum_od_endpoint_error_m"]),
        "route_gps_ratio": np.isfinite(route_gps_ratio)
        and float(strict["minimum_route_gps_ratio"])
        <= route_gps_ratio
        <= float(strict["maximum_route_gps_ratio"]),
        "processing": processing_exception is None,
    }
    analysis_checks = {
        "has_valid_route": has_route,
        "matched_interval_share": matched_interval_share
        >= float(analysis["minimum_matched_interval_share"]),
        "canonical_mapping": mapping_share
        >= float(analysis["minimum_canonical_mapping_share"]),
        "od_endpoint": od_error <= float(analysis["maximum_od_endpoint_error_m"]),
        "route_gps_ratio": np.isfinite(route_gps_ratio)
        and float(analysis["minimum_route_gps_ratio"])
        <= route_gps_ratio
        <= float(analysis["maximum_route_gps_ratio"]),
        "processing": processing_exception is None,
    }
    if all(strict_checks.values()):
        quality = "strict_core"
    elif all(analysis_checks.values()):
        quality = "analysis_set"
    else:
        quality = "rejected"
    failed = [name for name, passed in strict_checks.items() if not passed]
    return {
        "order_id": str(source_points.order_id.iloc[0]) if len(source_points) else "",
        "successful_reconstruction": has_route,
        "matched_point_share": matched_share,
        "matched_interval_share": matched_interval_share,
        "unmatched_point_share": unmatched_share,
        "interpolated_point_share": interpolated_share,
        "snap_distance_p50_m": snap_p50,
        "snap_distance_p90_m": snap_p90,
        "snap_distance_p99_m": snap_p99,
        "subtrace_count": subtrace_count,
        "discontinuity_count": discontinuities,
        "inferred_distance_share": inferred_share,
        "unresolved_time_share": unresolved_time_share,
        "od_endpoint_error_m": od_error,
        "gps_distance_m": gps_distance,
        "route_distance_m": route_distance,
        "route_gps_distance_ratio": route_gps_ratio,
        "canonical_edge_mapping_share": mapping_share,
        "route_part_count": int(len(route_parts)),
        "route_quality": quality,
        "formal_analysis_eligible": quality in {"strict_core", "analysis_set"},
        "strict_evaluation_eligible": quality == "strict_core",
        "quality_reasons": "|".join(failed),
        "quality_checks_json": json.dumps(strict_checks, sort_keys=True),
        "processing_exception": processing_exception,
    }
