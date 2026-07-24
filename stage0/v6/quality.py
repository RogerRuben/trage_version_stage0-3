"""Independent route and dynamic-measurement quality classifiers."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def _quantile(values: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.quantile(q)) if len(numeric) else float("nan")


def _sum(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def evaluate_route_quality(
    source_points: pd.DataFrame,
    matched_points: pd.DataFrame,
    route_parts: pd.DataFrame,
    interval_measurements: pd.DataFrame,
    thresholds: dict[str, Any],
    *,
    processing_exception: str | None = None,
) -> dict[str, Any]:
    """Classify static route usability without implying link-time usability."""

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

    route_distance = _sum(route_parts, "length_m")
    raw_gps_distance = float(
        pd.to_numeric(
            source_points.get("step_distance_m", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0).sum()
    )
    no_break = ~source_points.get(
        "preprocess_break_before", pd.Series(False, index=source_points.index)
    ).fillna(False)
    resolved_gps_distance = float(
        pd.to_numeric(
            source_points.loc[no_break, "step_distance_m"], errors="coerce"
        ).fillna(0).sum()
    )
    route_resolved_ratio = (
        route_distance / resolved_gps_distance
        if resolved_gps_distance > 0
        else float("nan")
    )
    route_raw_ratio = (
        route_distance / raw_gps_distance if raw_gps_distance > 0 else float("nan")
    )
    mapped_distance = float(
        pd.to_numeric(
            route_parts.loc[
                route_parts.get(
                    "canonical_edge_uid",
                    pd.Series(index=route_parts.index, dtype=object),
                ).notna(),
                "length_m",
            ],
            errors="coerce",
        ).fillna(0).sum()
    )
    mapping_share = mapped_distance / route_distance if route_distance > 0 else 0.0
    inferred_distance = float(
        pd.to_numeric(
            route_parts.loc[
                route_parts.get(
                    "measurement_source",
                    route_parts.get(
                        "route_source",
                        pd.Series(index=route_parts.index, dtype=object),
                    ),
                ).isin(["engine_interpolated", "inferred"]),
                "length_m",
            ],
            errors="coerce",
        ).fillna(0).sum()
    )
    inferred_share = inferred_distance / route_distance if route_distance > 0 else 0.0
    total_interval_time = _sum(interval_measurements, "interval_duration_s")
    matched_interval_time = float(
        pd.to_numeric(
            interval_measurements.loc[
                interval_measurements.get(
                    "route_interval_supported",
                    pd.Series(False, index=interval_measurements.index),
                ).fillna(False),
                "interval_duration_s",
            ],
            errors="coerce",
        ).fillna(0).sum()
    )
    matched_interval_share = (
        matched_interval_time / total_interval_time if total_interval_time > 0 else 0.0
    )
    preprocess_break_time = float(
        pd.to_numeric(
            interval_measurements.loc[
                interval_measurements.get(
                    "interval_reason",
                    pd.Series("", index=interval_measurements.index),
                ).astype(str).str.startswith("preprocess_"),
                "interval_duration_s",
            ],
            errors="coerce",
        ).fillna(0).sum()
    )
    preprocess_break_share = (
        preprocess_break_time / total_interval_time if total_interval_time > 0 else 0.0
    )
    preprocess_break_count = int(
        interval_measurements.get(
            "interval_reason", pd.Series(dtype=str)
        ).astype(str).str.startswith("preprocess_").sum()
    )
    discontinuities = int(
        matched_points.get("route_discontinuity", pd.Series(dtype=bool))
        .fillna(False)
        .sum()
    )
    subtrace_count = (
        int(source_points.subtrace_id.nunique()) if len(source_points) else 0
    )
    unresolved_gap_distance = float(
        pd.to_numeric(
            interval_measurements.loc[
                interval_measurements.get(
                    "measurement_source",
                    pd.Series("", index=interval_measurements.index),
                ).eq("unresolved"),
                "gps_interval_distance_m",
            ],
            errors="coerce",
        ).fillna(0).sum()
    )

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
        "route_resolved_gps_ratio": np.isfinite(route_resolved_ratio)
        and float(strict["minimum_route_gps_ratio"])
        <= route_resolved_ratio
        <= float(strict["maximum_route_gps_ratio"]),
        "preprocess_break_share": preprocess_break_share
        <= float(strict.get("maximum_preprocess_break_time_share", 1.0)),
        "processing": processing_exception is None,
    }
    analysis_checks = {
        "has_valid_route": has_route,
        "matched_interval_share": matched_interval_share
        >= float(analysis["minimum_matched_interval_share"]),
        "canonical_mapping": mapping_share
        >= float(analysis["minimum_canonical_mapping_share"]),
        "od_endpoint": od_error <= float(analysis["maximum_od_endpoint_error_m"]),
        "route_resolved_gps_ratio": np.isfinite(route_resolved_ratio)
        and float(analysis["minimum_route_gps_ratio"])
        <= route_resolved_ratio
        <= float(analysis["maximum_route_gps_ratio"]),
        "preprocess_break_share": preprocess_break_share
        <= float(analysis.get("maximum_preprocess_break_time_share", 1.0)),
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
        "preprocess_break_count": preprocess_break_count,
        "preprocess_break_time_share": preprocess_break_share,
        "discontinuity_count": discontinuities,
        "inferred_distance_share": inferred_share,
        "od_endpoint_error_m": od_error,
        "raw_order_gps_distance_m": raw_gps_distance,
        "resolved_subtrace_gps_distance_m": resolved_gps_distance,
        "unresolved_gap_distance_m": unresolved_gap_distance,
        "route_distance_m": route_distance,
        "route_resolved_gps_distance_ratio": route_resolved_ratio,
        "route_raw_gps_distance_ratio": route_raw_ratio,
        "route_gps_distance_ratio": route_resolved_ratio,
        "canonical_edge_mapping_share": mapping_share,
        "route_part_count": int(len(route_parts)),
        "route_quality": quality,
        "formal_analysis_eligible": quality in {"strict_core", "analysis_set"},
        "strict_evaluation_eligible": quality == "strict_core",
        "quality_reasons": "|".join(failed),
        "quality_checks_json": json.dumps(strict_checks, sort_keys=True),
        "processing_exception": processing_exception,
    }


def evaluate_dynamic_measurement_quality(
    route_parts: pd.DataFrame,
    link_traversals: pd.DataFrame,
    interval_measurements: pd.DataFrame,
    interval_accounting: pd.DataFrame,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Classify link-time usability independently from static route quality."""

    accounting = (
        interval_accounting.iloc[0].to_dict()
        if len(interval_accounting)
        else {}
    )
    total_time = float(accounting.get("total_interval_time_s", 0.0))
    direct_time = float(accounting.get("direct_observed_time_s", 0.0))
    supported_time = float(accounting.get("interval_supported_time_s", 0.0))
    engine_time = float(accounting.get("engine_allocated_only_time_s", 0.0))
    unresolved_time = float(accounting.get("unresolved_interval_time_s", 0.0))
    route_distance = _sum(route_parts, "length_m")
    direct_distance = float(accounting.get("direct_observed_distance_m", 0.0))
    direct_time_share = direct_time / total_time if total_time > 0 else 0.0
    supported_time_share = supported_time / total_time if total_time > 0 else 0.0
    engine_time_share = engine_time / total_time if total_time > 0 else 0.0
    unresolved_time_share = unresolved_time / total_time if total_time > 0 else 0.0
    direct_distance_share = (
        direct_distance / route_distance if route_distance > 0 else 0.0
    )
    conservation_error = float(accounting.get("time_conservation_error_s", np.inf))
    conservation_valid = bool(accounting.get("time_conservation_valid", False))
    anchor_valid = bool(accounting.get("timestamp_anchor_valid", False))
    anchor_failures = int(accounting.get("timestamp_anchor_failure_count", 0))
    inferred_violations = int(
        accounting.get("inferred_edge_observed_time_violation_count", 0)
    )
    duplicate_allocations = int(
        accounting.get("unresolved_duplicate_allocation_count", 0)
    )
    valid_timed_count = int(accounting.get("valid_timed_traversal_count", 0))
    timed_share = float(accounting.get("timed_traversal_share", 0.0))

    strict = thresholds["dynamic_strict"]
    partial = thresholds["dynamic_partial"]
    strict_checks = {
        "direct_time_share": direct_time_share
        >= float(strict["minimum_direct_observed_interval_time_share"]),
        "unresolved_time_share": unresolved_time_share
        <= float(strict["maximum_unresolved_time_share"]),
        "timestamp_anchor": anchor_valid,
        "time_conservation": conservation_valid,
        "no_inferred_observed_time": inferred_violations == 0,
        "no_duplicate_allocation": duplicate_allocations == 0,
    }
    partial_checks = {
        "has_direct_observation": direct_time_share
        > float(partial.get("minimum_direct_observed_interval_time_share", 0.0)),
        "time_conservation": conservation_valid,
        "timestamp_anchor": anchor_valid,
        "no_inferred_observed_time": inferred_violations == 0,
        "no_duplicate_allocation": duplicate_allocations == 0,
    }
    if all(strict_checks.values()):
        quality = "dynamic_strict"
    elif all(partial_checks.values()):
        quality = "dynamic_partial"
    else:
        quality = "dynamic_unusable"
    return {
        "order_id": (
            str(interval_measurements.order_id.iloc[0])
            if len(interval_measurements)
            else (
                str(route_parts.order_id.iloc[0]) if len(route_parts) else ""
            )
        ),
        "dynamic_measurement_quality": quality,
        "direct_observed_interval_time_share": direct_time_share,
        "direct_observed_distance_share": direct_distance_share,
        "interval_supported_time_share": supported_time_share,
        "engine_allocated_time_share": engine_time_share,
        "unresolved_time_share": unresolved_time_share,
        "valid_timed_traversal_count": valid_timed_count,
        "timed_traversal_share": timed_share,
        "timestamp_anchor_valid": anchor_valid,
        "timestamp_anchor_failure_count": anchor_failures,
        "time_conservation_error_s": conservation_error,
        "time_conservation_valid": conservation_valid,
        "inferred_edge_observed_time_violation_count": inferred_violations,
        "unresolved_duplicate_allocation_count": duplicate_allocations,
        "dynamic_quality_checks_json": json.dumps(strict_checks, sort_keys=True),
    }


def evaluate_order_quality(
    source_points: pd.DataFrame,
    matched_points: pd.DataFrame,
    route_parts: pd.DataFrame,
    unresolved_intervals: pd.DataFrame,
    thresholds: dict[str, Any],
    *,
    interval_measurements: pd.DataFrame | None = None,
    link_traversals: pd.DataFrame | None = None,
    interval_accounting: pd.DataFrame | None = None,
    processing_exception: str | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper returning both independent quality records."""

    intervals = (
        interval_measurements
        if interval_measurements is not None
        else unresolved_intervals
    )
    route = evaluate_route_quality(
        source_points,
        matched_points,
        route_parts,
        intervals,
        thresholds,
        processing_exception=processing_exception,
    )
    dynamic = evaluate_dynamic_measurement_quality(
        route_parts,
        link_traversals if link_traversals is not None else pd.DataFrame(),
        intervals,
        interval_accounting if interval_accounting is not None else pd.DataFrame(),
        thresholds,
    )
    return {**route, **dynamic}
