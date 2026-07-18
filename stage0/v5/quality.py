"""Order-level conservation audit and frozen three-layer quality semantics."""

from __future__ import annotations

import json
import math
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd


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
) -> dict[str, Any]:
    """Calculate hard/soft predicates without test-dependent tuning."""
    hard_cfg, soft_cfg = quality_config["hard"], quality_config["soft"]
    edge_lookup = edges if edges.index.name == "edge_uid" else edges.set_index("edge_uid", drop=False)
    successful = match_summary["matching_mode"] != "rejected" and len(route_parts) > 0
    route_distance = float(route_parts.edge_uid.map(edge_lookup.length_m).sum()) if successful else 0.0
    xy = matched_points[["metric_x", "metric_y"]].to_numpy(float) if len(matched_points) else np.empty((0, 2))
    observed_distance = float(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])).sum()) if len(xy) > 1 else 0.0
    raw_time = pd.to_numeric(matched_points.timestamp, errors="coerce") if len(matched_points) else pd.Series(dtype=float)
    duration = float(raw_time.max() - raw_time.min()) if len(raw_time) else 0.0
    traversal_time = float(pd.to_numeric(traversals.get("observed_interval_time_s", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    movement_time = float(pd.to_numeric(movements.get("observed_interval_time_s", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    unallocated_time = max(0.0, duration - traversal_time - movement_time) if not successful else 0.0
    time_error = abs(duration - traversal_time - movement_time - unallocated_time)
    allocated_distance = float(pd.to_numeric(traversals.get("allocated_distance_m", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    distance_error = abs(route_distance - allocated_distance)
    projection = pd.to_numeric(matched_points.get("gps_to_edge_distance_m"), errors="coerce")
    movement_quality = movements.get("movement_quality", pd.Series(dtype=str))
    gap_count = int((movement_quality == "missing_movement").sum())
    transition_type = movements.get("level_transition_type", pd.Series(dtype=str)).astype(str)
    layer_count = int(transition_type.eq("unresolved_level_gap").sum())
    movement_type = movements.get("movement_type", pd.Series(dtype=str))
    restriction = movements.get("restriction_status", pd.Series(dtype=str)).astype(str)
    restriction_count = int(restriction.str.startswith("forbidden").sum())
    illegal_uturn = int(((movement_type == "u_turn") & restriction.str.startswith("forbidden")).sum())
    heading = pd.to_numeric(matched_points.get("edge_heading_difference_deg", pd.Series(dtype=float)), errors="coerce")
    direction_count = int((heading > float(hard_cfg.get("maximum_heading_direction_difference_deg", 100.0))).sum())
    if len(matched_points) > 1 and {"edge_uid", "position_on_edge"} <= set(matched_points.columns):
        ordered_points = matched_points.sort_values("point_seq", kind="stable")
        positions = pd.to_numeric(ordered_points.position_on_edge, errors="coerce")
        same_edge = ordered_points.edge_uid.astype(str).eq(ordered_points.edge_uid.astype(str).shift())
        severe_reverse = same_edge & positions.diff().lt(-float(hard_cfg.get("same_edge_reverse_tolerance_m", 10.0)))
        direction_count += int(severe_reverse.sum())
    inferred_distance = float(route_parts.loc[route_parts.is_interpolated, "edge_uid"].map(edge_lookup.length_m).sum()) if successful else 0.0
    endpoint_error = float(max(projection.iloc[0], projection.iloc[-1])) if len(projection.dropna()) == len(projection) and len(projection) else math.inf
    confidence = float(np.exp(-projection.mean() / 30.0)) if len(projection.dropna()) else 0.0
    route_ratio = _safe_ratio(route_distance, observed_distance)
    metrics = {
        "successful_reconstruction": successful,
        "direction_violation_count": direction_count,
        "topology_gap_count": gap_count,
        "unreasonable_detour_count": int(bool(np.isfinite(route_ratio) and route_ratio > float(soft_cfg["maximum_route_length_ratio"]) * 2)),
        "illegal_u_turn_count": illegal_uturn,
        "layer_violation_count": layer_count,
        "restriction_violation_count": restriction_count,
        "route_link_count": int(len(route_parts)),
        "observed_distance_m": observed_distance,
        "unallocated_observed_time_s": unallocated_time,
        "unallocated_observed_distance_m": observed_distance if not successful else 0.0,
        "matched_distance_m": route_distance,
        "od_endpoint_error_m": endpoint_error,
        "time_conservation_error_s": time_error,
        "distance_conservation_error_m": distance_error,
        "fallback_share": float(match_summary["matching_mode"] == "geometric_fallback"),
        "p90_projection_distance_m": float(projection.quantile(0.9)) if len(projection.dropna()) else math.inf,
        "route_length_ratio": route_ratio,
        "interpolated_distance_share": _safe_ratio(inferred_distance, route_distance, 0.0),
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
        "no_restriction_violation": restriction_count == 0,
        "minimum_route_links": len(route_parts) >= int(hard_cfg["minimum_route_links"]),
        "time_distance_conservation": max(time_error, distance_error) <= float(hard_cfg["conservation_tolerance"]),
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
