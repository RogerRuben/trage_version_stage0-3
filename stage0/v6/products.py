"""Adapter from normalized Valhalla rows to Stage 0 traversal products."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


def _unresolved_row(
    order_id: str,
    interval_id: int,
    left: pd.Series,
    right: pd.Series,
    reason: str,
) -> dict[str, Any]:
    return {
        "order_id": order_id,
        "unresolved_interval_id": interval_id,
        "subtrace_id": left.subtrace_id,
        "from_original_point_seq": int(left.original_point_seq),
        "to_original_point_seq": int(right.original_point_seq),
        "from_edge_index": left.edge_index,
        "to_edge_index": right.edge_index,
        "unresolved_interval_time_s": max(float(right.timestamp - left.timestamp), 0.0),
        "interval_time_source": "raw_gps_timestamp",
        "unresolved_reason": reason,
    }


def build_order_products(
    source_points: pd.DataFrame,
    matched_points: pd.DataFrame,
    route_parts: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build traversal, movement, and unresolved-interval products."""

    if source_points.empty:
        raise ValueError("source_points cannot be empty")
    order_id = str(source_points.order_id.iloc[0])
    observed_time: dict[int, float] = defaultdict(float)
    unresolved: list[dict[str, Any]] = []
    resolved_interval_time_s = 0.0
    total_interval_time_s = 0.0
    observed_parts_by_edge: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
    for route in route_parts.itertuples(index=False):
        if (
            route.path_id == 0
            and route.route_source == "observed"
            and pd.notna(route.valhalla_edge_index)
        ):
            observed_parts_by_edge[
                (str(route.subtrace_id), int(route.valhalla_edge_index))
            ].append((int(route.route_sequence), float(route.length_m)))

    for _, group in matched_points.groupby("subtrace_id", sort=False):
        ordered = group.sort_values(["timestamp", "original_point_seq"], kind="stable")
        rows = list(ordered.itertuples(index=False))
        for left_tuple, right_tuple in zip(rows, rows[1:]):
            left, right = pd.Series(left_tuple._asdict()), pd.Series(right_tuple._asdict())
            dt = max(float(right.timestamp - left.timestamp), 0.0)
            total_interval_time_s += dt
            reason = None
            if (
                left.matched_point_status == "unmatched"
                or right.matched_point_status == "unmatched"
            ):
                reason = "unmatched_endpoint"
            elif bool(left.route_discontinuity) or bool(right.route_discontinuity):
                reason = "valhalla_route_discontinuity"
            elif pd.isna(left.edge_index) or pd.isna(right.edge_index):
                reason = "missing_edge_index"
            elif int(right.edge_index) < int(left.edge_index):
                reason = "nonmonotonic_edge_index"
            if reason is not None:
                unresolved.append(
                    _unresolved_row(order_id, len(unresolved), left, right, reason)
                )
                continue
            candidate_parts: list[tuple[int, float]] = []
            for edge_index in range(int(left.edge_index), int(right.edge_index) + 1):
                candidate_parts.extend(
                    observed_parts_by_edge.get((str(left.subtrace_id), edge_index), [])
                )
            weight_sum = sum(max(weight, 0.0) for _, weight in candidate_parts)
            if not candidate_parts or weight_sum <= 0:
                unresolved.append(
                    _unresolved_row(
                        order_id, len(unresolved), left, right, "no_observed_route_part"
                    )
                )
                continue
            resolved_interval_time_s += dt
            for sequence, weight in candidate_parts:
                observed_time[sequence] += dt * max(weight, 0.0) / weight_sum

    traversal_rows: list[dict[str, Any]] = []
    cumulative_time: dict[str, float] = {}
    observed_point_counts = (
        matched_points.loc[matched_points.matched_point_status.eq("matched")]
        .groupby(["subtrace_id", "edge_index"], dropna=True)
        .size()
        .to_dict()
    )
    for _, route in route_parts.sort_values(
        ["subtrace_id", "route_sequence"], kind="stable"
    ).iterrows():
        subtrace_id = str(route.subtrace_id)
        if subtrace_id not in cumulative_time:
            starts = source_points.loc[
                source_points.subtrace_id.eq(subtrace_id), "timestamp"
            ]
            cumulative_time[subtrace_id] = float(starts.min()) if len(starts) else 0.0
        travel_time = float(observed_time.get(int(route.route_sequence), 0.0))
        enter_time = cumulative_time[subtrace_id]
        exit_time = enter_time + travel_time
        cumulative_time[subtrace_id] = exit_time
        link_id = (
            route.canonical_edge_uid
            if pd.notna(route.get("canonical_edge_uid"))
            else f"valhalla:{route.valhalla_edge_id}"
        )
        traversal_rows.append(
            {
                "order_id": order_id,
                "subtrace_id": subtrace_id,
                "traversal_id": len(traversal_rows),
                "route_sequence": int(route.route_sequence),
                "edge_uid": link_id,
                "canonical_edge_uid": route.get("canonical_edge_uid"),
                "valhalla_edge_id": route.valhalla_edge_id,
                "enter_time": enter_time,
                "exit_time": exit_time,
                "travel_time_s": travel_time,
                "entry_position_m": route.get("entry_position_m", np.nan),
                "exit_position_m": route.get("exit_position_m", np.nan),
                "observed_distance_m": (
                    float(route.length_m) if route.route_source == "observed" else 0.0
                ),
                "allocated_distance_m": float(route.length_m),
                "observed_point_count": int(
                    observed_point_counts.get(
                        (subtrace_id, route.valhalla_edge_index), 0
                    )
                ),
                "traversal_source": route.route_source,
                "traversal_quality": route.mapping_status,
                "is_interpolated": bool(route.is_interpolated),
                "interpolated_distance_share": (
                    1.0 if route.route_source == "inferred" else 0.0
                ),
                "observed_interval_time_s": travel_time,
            }
        )
    traversals = pd.DataFrame(traversal_rows)

    movement_rows: list[dict[str, Any]] = []
    ordered_routes = route_parts.sort_values(
        ["subtrace_id", "path_id", "route_sequence"], kind="stable"
    )
    for (_, _), group in ordered_routes.groupby(["subtrace_id", "path_id"], sort=False):
        rows = list(group.itertuples(index=False))
        for left, right in zip(rows, rows[1:]):
            observed = left.route_source == "observed" and right.route_source == "observed"
            continuous = (
                pd.notna(left.canonical_to_node)
                and pd.notna(right.canonical_from_node)
                and int(left.canonical_to_node) == int(right.canonical_from_node)
            )
            movement_rows.append(
                {
                    "order_id": order_id,
                    "subtrace_id": left.subtrace_id,
                    "movement_sequence": len(movement_rows),
                    "from_edge_uid": (
                        left.canonical_edge_uid
                        if pd.notna(left.canonical_edge_uid)
                        else f"valhalla:{left.valhalla_edge_id}"
                    ),
                    "to_edge_uid": (
                        right.canonical_edge_uid
                        if pd.notna(right.canonical_edge_uid)
                        else f"valhalla:{right.valhalla_edge_id}"
                    ),
                    "movement_observed": observed,
                    "observed_interval_time_s": 0.0,
                    "dynamic_time_source": "observed" if observed else "none",
                    "via_node": left.canonical_to_node,
                    "movement_type": "continuous" if continuous else "valhalla_transition",
                    "restriction_status": "valhalla_legal",
                    "movement_quality": "mapped" if continuous else "engine_only",
                }
            )
    movements = pd.DataFrame(movement_rows)
    unresolved_frame = pd.DataFrame(unresolved)
    accounting = pd.DataFrame(
        [
            {
                "order_id": order_id,
                "resolved_interval_time_s": resolved_interval_time_s,
                "total_interval_time_s": total_interval_time_s,
                "unresolved_interval_time_s": float(
                    unresolved_frame.get(
                        "unresolved_interval_time_s", pd.Series(dtype=float)
                    ).sum()
                ),
            }
        ]
    )
    return {
        "route_parts": route_parts.reset_index(drop=True),
        "link_traversals": traversals,
        "turn_movements": movements,
        "unresolved_intervals": unresolved_frame,
        "interval_accounting": accounting,
    }
