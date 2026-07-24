"""Order-level conservation audit and frozen three-layer quality semantics."""

from __future__ import annotations

import json
import math
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from .reconstruction import projected_route_distance_m


def _safe_ratio(numerator: float, denominator: float, default: float = math.nan) -> float:
    return float(numerator / denominator) if denominator > 0 else default


def _repeated_share(values: pd.Series) -> float:
    return float(values.duplicated().sum() / len(values)) if len(values) else 0.0


def evaluate_order_quality(
    order_id: str,
    matched_points: pd.DataFrame,
    route_parts: pd.DataFrame,
    traversals: pd.DataFrame,
    movements: pd.DataFrame,
    edges: gpd.GeoDataFrame,
    match_summary: dict[str, Any],
    quality_config: dict[str, Any],
    unresolved_intervals: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Calculate hard/soft predicates without test-dependent tuning."""
    hard_cfg, soft_cfg = quality_config["hard"], quality_config["soft"]
    edge_lookup = edges if edges.index.name == "edge_uid" else edges.set_index("edge_uid", drop=False)
    successful = str(match_summary["matching_mode"]) not in {
        "rejected", "failed_no_continuous_route",
    } and len(route_parts) > 0
    route_distance = float(pd.to_numeric(
        route_parts.get("allocated_distance_m", route_parts.edge_uid.map(edge_lookup.length_m)), errors="coerce"
    ).fillna(0.0).sum()) if successful else 0.0
    xy = matched_points[["metric_x", "metric_y"]].to_numpy(float) if len(matched_points) else np.empty((0, 2))
    observed_distance = float(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])).sum()) if len(xy) > 1 else 0.0
    raw_time = pd.to_numeric(matched_points.timestamp, errors="coerce") if len(matched_points) else pd.Series(dtype=float)
    duration = float(raw_time.max() - raw_time.min()) if len(raw_time) else 0.0
    traversal_time = float(pd.to_numeric(traversals.get("observed_interval_time_s", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    movement_time = float(pd.to_numeric(movements.get("observed_interval_time_s", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    unresolved = unresolved_intervals if unresolved_intervals is not None else pd.DataFrame()
    unresolved_time = float(pd.to_numeric(
        unresolved.get("unresolved_interval_time_s", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0).sum())
    unallocated_time = duration if not successful else 0.0
    time_error = abs(duration - traversal_time - movement_time - unresolved_time - unallocated_time)
    allocated_distance = float(pd.to_numeric(traversals.get("allocated_distance_m", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    internal_distance_error = abs(route_distance - allocated_distance)
    projected_distance = projected_route_distance_m(matched_points, route_parts, edge_lookup) if successful else math.nan
    projected_distance_error = (
        abs(route_distance - projected_distance)
        if math.isfinite(projected_distance)
        else math.nan
    )
    invalid_position_count = int((~route_parts.get(
        "position_distance_valid", pd.Series(False, index=route_parts.index)
    ).fillna(False).astype(bool)).sum()) if successful else 0
    position_audit_applicable = bool(successful)
    alignment_valid = bool(route_parts.get(
        "observed_run_alignment_valid", pd.Series(False, index=route_parts.index)
    ).fillna(False).astype(bool).all()) if successful else False
    projection = pd.to_numeric(
        matched_points.get(
            "gps_to_edge_distance_m",
            pd.Series(np.nan, index=matched_points.index),
        ),
        errors="coerce",
    )
    movement_quality = movements.get("movement_quality", pd.Series(dtype=str))
    movement_reason = movements.get("movement_audit_reason", movement_quality).astype(str)
    gap_count = int(movement_reason.eq("missing_topology").sum())
    restriction_block_count = int(movement_reason.eq("restriction_block").sum())
    unparsed_restriction_count = int(movement_reason.eq("unparsed_restriction_exposure").sum())
    suspicious_level_count = int(movement_reason.eq("suspicious_level_transition").sum())
    transition_type = movements.get("level_transition_type", pd.Series(dtype=str)).astype(str)
    layer_count = int(transition_type.eq("true_layer_discontinuity").sum())
    movement_type = movements.get("movement_type", pd.Series(dtype=str))
    restriction = movements.get("restriction_status", pd.Series(dtype=str)).astype(str)
    restriction_count = int(restriction.str.startswith("forbidden").sum())
    inferred_dynamic_count = int((
        movements.get("movement_observed", pd.Series(False, index=movements.index)).fillna(False).astype(bool)
        & movements.get("dynamic_time_source", pd.Series("", index=movements.index)).ne("observed_direct_movement")
    ).sum())
    path_identity_mismatch_count = 0
    path_distance_mismatch_count = 0
    same_edge_jitter_mismatch_count = 0
    if successful and "selected_path_json" in matched_points:
        import json as _json

        route_edges = route_parts.sort_values("route_sequence").edge_uid.astype(str).tolist()
        ordered_points = matched_points.sort_values("point_seq", kind="stable").reset_index(drop=True)
        cursor = 0
        for point_index in range(1, len(ordered_points)):
            left_uid = str(ordered_points.edge_uid.iloc[point_index - 1])
            right_uid = str(ordered_points.edge_uid.iloc[point_index])
            try:
                expected = [str(value) for value in _json.loads(
                    str(ordered_points.selected_path_json.iloc[point_index])
                )]
            except (TypeError, ValueError, _json.JSONDecodeError):
                path_identity_mismatch_count += 1
                path_distance_mismatch_count += 1
                continue
            found = False
            for start in range(cursor, len(route_edges)):
                if route_edges[start:start + len(expected)] == expected:
                    cursor = start + len(expected) - 1
                    found = True
                    break
            if not found:
                path_identity_mismatch_count += 1
            selected_distance = float(pd.to_numeric(
                ordered_points.selected_path_distance_m.iloc[point_index],
                errors="coerce",
            ))
            left_position = float(pd.to_numeric(
                ordered_points.position_on_edge.iloc[point_index - 1],
                errors="coerce",
            ))
            right_position = float(pd.to_numeric(
                ordered_points.position_on_edge.iloc[point_index],
                errors="coerce",
            ))
            if left_uid == right_uid:
                reconstructed_distance = max(0.0, right_position - left_position)
                observed_jitter_penalty = float(pd.to_numeric(
                    ordered_points.get(
                        "selected_jitter_penalty_m",
                        pd.Series(np.nan, index=ordered_points.index),
                    ).iloc[point_index],
                    errors="coerce",
                ))
                expected_jitter_penalty = (
                    max(0.0, left_position - right_position)
                    * float(
                        hard_cfg.get(
                            "same_edge_jitter_penalty_per_m", 0.25
                        )
                    )
                )
                if (
                    not math.isfinite(observed_jitter_penalty)
                    or abs(
                        observed_jitter_penalty - expected_jitter_penalty
                    )
                    > float(
                        hard_cfg.get(
                            "hmm_path_distance_tolerance_m", 1e-6
                        )
                    )
                ):
                    same_edge_jitter_mismatch_count += 1
            elif expected and expected[0] in edge_lookup.index and expected[-1] in edge_lookup.index:
                reconstructed_distance = (
                    max(0.0, float(edge_lookup.loc[left_uid].length_m) - left_position)
                    + sum(
                        float(edge_lookup.loc[edge_uid].length_m)
                        for edge_uid in expected[1:-1]
                        if edge_uid in edge_lookup.index
                    )
                    + max(0.0, right_position)
                )
            else:
                reconstructed_distance = math.nan
            if (
                not math.isfinite(selected_distance)
                or not math.isfinite(reconstructed_distance)
                or abs(selected_distance - reconstructed_distance)
                > float(hard_cfg.get("hmm_path_distance_tolerance_m", 1e-6))
            ):
                path_distance_mismatch_count += 1
    illegal_uturn = int(((movement_type == "u_turn") & restriction.str.startswith("forbidden")).sum())
    ordered_points = (
        matched_points.sort_values("point_seq", kind="stable")
        if len(matched_points) and "point_seq" in matched_points
        else matched_points
    )
    heading = pd.to_numeric(ordered_points.get("edge_heading_difference_deg", pd.Series(dtype=float)), errors="coerce")
    reliable = ordered_points.get("heading_reliable", pd.Series(False, index=ordered_points.index)).fillna(False).astype(bool)
    edge_values = ordered_points.get("edge_uid", pd.Series("", index=ordered_points.index)).astype(str)
    movement_neighbourhood = edge_values.ne(edge_values.shift()) | edge_values.ne(edge_values.shift(-1))
    raw_direction = reliable & ~movement_neighbourhood & heading.gt(
        float(hard_cfg.get("maximum_heading_direction_difference_deg", 100.0))
    )
    if len(matched_points) > 1 and {"edge_uid", "position_on_edge"} <= set(matched_points.columns):
        positions = pd.to_numeric(ordered_points.position_on_edge, errors="coerce")
        observed_step = pd.to_numeric(ordered_points.get("observed_step_m", 0.0), errors="coerce").fillna(0.0)
        same_edge = edge_values.eq(edge_values.shift())
        severe_reverse = (
            reliable & same_edge
            & positions.diff().lt(-float(hard_cfg.get("same_edge_reverse_tolerance_m", 10.0)))
            & observed_step.ge(float(hard_cfg.get("minimum_direction_displacement_m", 3.0)))
        )
        raw_direction |= severe_reverse
    minimum_run = int(hard_cfg.get("minimum_consecutive_direction_conflicts", 2))
    groups = raw_direction.ne(raw_direction.shift(fill_value=False)).cumsum()
    run_lengths = raw_direction.groupby(groups).transform("sum") if len(raw_direction) else raw_direction
    persistent_direction = raw_direction & run_lengths.ge(minimum_run)
    direction_warning_count = int(raw_direction.sum())
    direction_count = int(persistent_direction.sum())
    inferred_distance = float(pd.to_numeric(
        route_parts.loc[route_parts.is_interpolated, "allocated_distance_m"]
        if successful and "allocated_distance_m" in route_parts else pd.Series(dtype=float), errors="coerce"
    ).fillna(0.0).sum()) if successful else 0.0
    endpoint_error = float(max(projection.iloc[0], projection.iloc[-1])) if len(projection.dropna()) == len(projection) and len(projection) else math.inf
    confidence = float(np.exp(-projection.mean() / 30.0)) if len(projection.dropna()) else 0.0
    route_ratio = _safe_ratio(route_distance, observed_distance)
    path_ratios = pd.to_numeric(
        matched_points.get("path_to_gps_ratio", pd.Series(dtype=float)), errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()
    metrics = {
        "successful_reconstruction": successful,
        "direction_violation_count": direction_count,
        "direction_warning_count": direction_warning_count,
        "topology_gap_count": gap_count,
        "restriction_block_count": restriction_block_count,
        "unparsed_restriction_exposure_count": unparsed_restriction_count,
        "suspicious_level_transition_count": suspicious_level_count,
        "true_layer_discontinuity_count": layer_count,
        "unreasonable_detour_count": int(bool(np.isfinite(route_ratio) and route_ratio > float(soft_cfg["maximum_route_length_ratio"]) * 2)),
        "illegal_u_turn_count": illegal_uturn,
        "layer_violation_count": layer_count,
        "restriction_violation_count": restriction_count,
        "observed_dynamic_label_on_inferred_edge_count": inferred_dynamic_count,
        "hmm_output_path_identity_mismatch_count": path_identity_mismatch_count,
        "hmm_path_distance_mismatch_count": path_distance_mismatch_count,
        "same_edge_jitter_mismatch_count": same_edge_jitter_mismatch_count,
        "raw_movement_audit_available": "movement_audit_reason" in movements.columns,
        "network_snapshot_mismatch_share": float(
            matched_points.get(
                "selected_network_snapshot_mismatch",
                pd.Series(False, index=matched_points.index, dtype="boolean"),
            ).astype("boolean").fillna(False).mean()
        ) if len(matched_points) else 0.0,
        "path_to_gps_ratio_q50": float(path_ratios.quantile(0.50)) if len(path_ratios) else math.nan,
        "path_to_gps_ratio_q90": float(path_ratios.quantile(0.90)) if len(path_ratios) else math.nan,
        "path_to_gps_ratio_q99": float(path_ratios.quantile(0.99)) if len(path_ratios) else math.nan,
        "route_link_count": int(len(route_parts)),
        "observed_distance_m": observed_distance,
        "unallocated_observed_time_s": unallocated_time,
        "unresolved_interval_time_s": unresolved_time,
        "unallocated_observed_distance_m": observed_distance if not successful else 0.0,
        "matched_distance_m": route_distance,
        "projected_matched_movement_distance_m": projected_distance,
        "od_endpoint_error_m": endpoint_error,
        "time_allocation_error_s": time_error,
        "internal_distance_error_m": internal_distance_error,
        "projected_route_distance_error_m": projected_distance_error,
        "invalid_position_aware_distance_count": invalid_position_count,
        "position_audit_applicable_order_count": int(position_audit_applicable),
        "position_audit_not_applicable_match_failure_count": int(not position_audit_applicable),
        "actual_invalid_position_event_count": invalid_position_count,
        "observed_run_alignment_valid": alignment_valid,
        "time_conservation_error_s": time_error,
        "distance_conservation_error_m": internal_distance_error,
        "fallback_share": float(
            str(match_summary["matching_mode"])
            in {"pure_geometric_fallback", "partial_local_hmm_fallback"}
        ),
        "p90_projection_distance_m": float(projection.quantile(0.9)) if len(projection.dropna()) else math.inf,
        "route_length_ratio": route_ratio,
        "interpolated_distance_share": _safe_ratio(inferred_distance, route_distance),
        "matching_confidence": confidence,
        "repeated_link_share": _repeated_share(route_parts.edge_uid) if successful else 1.0,
        "parallel_ambiguity_share": float(matched_points.parallel_ambiguity.mean()) if len(matched_points) else 1.0,
    }
    hard = {
        "successful_reconstruction": successful,
        "direction_continuity": direction_count <= int(hard_cfg["maximum_direction_violations"]),
        "topology_continuity": gap_count <= int(hard_cfg.get("maximum_topology_gaps", 0)),
        "no_unreasonable_detour": metrics["unreasonable_detour_count"] <= int(hard_cfg["maximum_unreasonable_detours"]),
        "reasonable_od_endpoints": endpoint_error <= float(hard_cfg["maximum_od_endpoint_error_m"]),
        "no_illegal_u_turn": illegal_uturn <= int(hard_cfg["maximum_illegal_u_turns"]),
        "no_restriction_violation": max(restriction_count, restriction_block_count, unparsed_restriction_count) == 0,
        "minimum_route_links": len(route_parts) >= int(hard_cfg["minimum_route_links"]),
        "time_distance_conservation": max(time_error, internal_distance_error) <= float(hard_cfg["conservation_tolerance"]),
        "projected_distance_consistency": bool(
            successful
            and math.isfinite(projected_distance_error)
            and projected_distance_error <= float(hard_cfg["conservation_tolerance"])
        ),
        "valid_position_aware_distance": invalid_position_count == 0,
        "observed_run_alignment": alignment_valid,
        "hmm_output_path_identity": path_identity_mismatch_count == 0,
        "hmm_output_path_distance": path_distance_mismatch_count == 0,
        "same_edge_jitter_consistency": same_edge_jitter_mismatch_count == 0,
        "dynamic_label_provenance": inferred_dynamic_count == 0,
        "layer_continuity": layer_count <= int(hard_cfg["maximum_layer_violations"]),
    }
    soft = {
        "fallback_share": metrics["fallback_share"] <= float(soft_cfg["maximum_fallback_share"]),
        "projection_distance": metrics["p90_projection_distance_m"] <= float(soft_cfg["maximum_p90_projection_distance_m"]),
        "route_length_ratio": float(soft_cfg["minimum_route_length_ratio"]) <= route_ratio <= float(soft_cfg["maximum_route_length_ratio"]),
        "interpolated_distance_share": metrics["interpolated_distance_share"] <= float(soft_cfg["maximum_interpolated_distance_share"]),
        "match_confidence": confidence >= float(soft_cfg["minimum_match_confidence"]),
        "repeated_link_share": metrics["repeated_link_share"] <= float(soft_cfg["maximum_repeated_link_share"]),
        "parallel_ambiguity": metrics["parallel_ambiguity_share"] <= float(soft_cfg["maximum_parallel_ambiguity_share"]),
    }
    hard_failed = sorted(name for name, passed in hard.items() if not passed)
    soft_failed = sorted(name for name, passed in soft.items() if not passed)
    quality = "rejected" if hard_failed else ("strict_core" if not soft_failed else "analysis_set")
    return {
        "order_id": order_id,
        **metrics,
        "route_quality": quality,
        "formal_analysis_eligible": not hard_failed,
        "strict_evaluation_eligible": not hard_failed and not soft_failed,
        "hard_error_flags": json.dumps(hard_failed, ensure_ascii=False),
        "soft_quality_flags": json.dumps(soft_failed, ensure_ascii=False),
        "quality_reasons": "|".join(hard_failed + soft_failed),
    }


def conservation_summary(quality: pd.DataFrame, input_orders: int) -> dict[str, Any]:
    output_orders = int(len(quality))
    counts = quality.route_quality.value_counts().to_dict() if len(quality) else {}
    return {
        "input_orders": int(input_orders),
        "output_orders": output_orders,
        "accounting_pass": output_orders == int(input_orders),
        "time_conservation_failures": int((quality.time_conservation_error_s > 1e-6).sum()) if len(quality) else 0,
        "distance_conservation_failures": int((quality.distance_conservation_error_m > 1e-6).sum()) if len(quality) else 0,
        "quality_counts": {str(key): int(value) for key, value in counts.items()},
    }
