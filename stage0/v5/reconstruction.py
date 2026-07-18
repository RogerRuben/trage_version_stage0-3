"""Edge-aware route reconstruction and static inferred-path accounting."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from .routing import CompactMovementRouter


@dataclass
class ReconstructedRoute:
    edge_uids: list[str]
    observed: list[bool]
    status: str
    gap_count: int


class EdgeAwareRouter:
    def __init__(
        self,
        edges: gpd.GeoDataFrame,
        movements: pd.DataFrame,
        config: dict[str, Any],
        movement_router: CompactMovementRouter | None = None,
    ):
        self.edges = edges.set_index("edge_uid", drop=False)
        self.router = movement_router or CompactMovementRouter(edges, movements, config)
        self.maximum = float(config.get("max_route_distance_m", 6000.0))
        self.last_bridge_search_ms = 0.0
        self.last_precomputed_path_count = 0
        self.last_path_search_count = 0

    def bridge(self, left: str, right: str) -> list[str] | None:
        if left == right:
            return [left]
        result = self.router.bridge(left, right, self.maximum)
        return None if result is None else result[0]

    def reconstruct(
        self,
        matched_points: pd.DataFrame,
        precomputed_paths: dict[tuple[str, str], list[str]] | None = None,
    ) -> ReconstructedRoute:
        precomputed = precomputed_paths or {}
        self.last_bridge_search_ms = 0.0
        self.last_precomputed_path_count = 0
        self.last_path_search_count = 0
        if matched_points.empty or matched_points.edge_uid.isna().any():
            return ReconstructedRoute([], [], "rejected", 1)
        states = matched_points.loc[matched_points.edge_uid.ne(matched_points.edge_uid.shift()), "edge_uid"].astype(str).tolist()
        route = [states[0]]
        observed = [True]
        gaps = 0
        for left, right in zip(states[:-1], states[1:]):
            path = precomputed.get((left, right))
            if path is not None:
                self.last_precomputed_path_count += 1
            elif self.router.movement(left, right) is not None:
                path = [left, right]
            else:
                self.last_path_search_count += 1
                search_started = time.perf_counter()
                path = self.bridge(left, right)
                self.last_bridge_search_ms += (time.perf_counter() - search_started) * 1000.0
            if path is None:
                gaps += 1
                route.append(right)
                observed.append(True)
                continue
            for edge_uid in path[1:-1]:
                route.append(edge_uid)
                observed.append(False)
            route.append(right)
            observed.append(True)
        return ReconstructedRoute(route, observed, "complete" if gaps == 0 else "gap", gaps)


def _edge_lookup(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return edges if edges.index.name == "edge_uid" else edges.set_index("edge_uid", drop=False)


def route_parts_frame(order_id: str, route: ReconstructedRoute, edges: gpd.GeoDataFrame) -> pd.DataFrame:
    lookup = _edge_lookup(edges)
    rows = []
    for sequence, (edge_uid, is_observed) in enumerate(zip(route.edge_uids, route.observed)):
        edge = lookup.loc[edge_uid]
        rows.append({
            "order_id": order_id, "route_sequence": sequence, "edge_uid": edge_uid,
            "from_node": int(edge.from_node), "to_node": int(edge.to_node), "edge_key": edge.edge_key,
            "route_source": "observed" if is_observed else "inferred_path",
            "is_interpolated": not is_observed,
        })
    return pd.DataFrame(rows)


def build_traversals(
    order_id: str, matched_points: pd.DataFrame, route_parts: pd.DataFrame, edges: gpd.GeoDataFrame
) -> pd.DataFrame:
    lookup = _edge_lookup(edges)
    total_distance = float(route_parts.edge_uid.map(lookup.length_m).sum()) if len(route_parts) else 0.0
    inferred_distance = float(route_parts.loc[route_parts.is_interpolated, "edge_uid"].map(lookup.length_m).sum()) if len(route_parts) else 0.0
    rows = []
    ordered = matched_points.sort_values("point_seq", kind="stable")
    edge_sequence = ordered.edge_uid.astype(str).tolist()
    timestamps = pd.to_numeric(ordered.timestamp, errors="coerce").to_numpy(float)
    same_edge_time: dict[str, float] = {}
    for left, right, delta in zip(edge_sequence[:-1], edge_sequence[1:], np.diff(timestamps)):
        if left == right and delta >= 0:
            same_edge_time[left] = same_edge_time.get(left, 0.0) + float(delta)
    consumed_edges: set[str] = set()
    observed_groups = {
        edge_uid: group for edge_uid, group in matched_points.groupby("edge_uid", sort=False)
    }
    for row in route_parts.itertuples():
        length = float(lookup.loc[row.edge_uid].length_m)
        observed_group = observed_groups.get(row.edge_uid)
        if row.is_interpolated or observed_group is None:
            enter_time = exit_time = pd.NaT
            travel_time = math.nan
            observed_distance = 0.0
            point_count = 0
            quality = "static_inferred_only"
        else:
            timestamps = pd.to_numeric(observed_group.timestamp, errors="coerce")
            enter_time, exit_time = float(timestamps.min()), float(timestamps.max())
            travel_time = same_edge_time.get(str(row.edge_uid), 0.0) if str(row.edge_uid) not in consumed_edges else 0.0
            consumed_edges.add(str(row.edge_uid))
            xy = observed_group[["metric_x", "metric_y"]].to_numpy(float)
            observed_distance = float(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])).sum()) if len(xy) > 1 else 0.0
            point_count = len(observed_group)
            quality = "observed_support"
        rows.append({
            "order_id": order_id, "edge_uid": row.edge_uid, "route_sequence": int(row.route_sequence),
            "enter_time": enter_time, "exit_time": exit_time, "travel_time_s": travel_time,
            "observed_distance_m": observed_distance, "allocated_distance_m": length,
            "observed_point_count": point_count,
            "traversal_source": "inferred_path" if row.is_interpolated else "observed",
            "traversal_quality": quality, "is_interpolated": bool(row.is_interpolated),
            "interpolated_distance_share": inferred_distance / total_distance if total_distance else 0.0,
            "observed_interval_time_s": travel_time,
        })
    return pd.DataFrame(rows)


def build_movements(
    order_id: str,
    route_parts: pd.DataFrame,
    movement_router: CompactMovementRouter,
    matched_points: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    movement_times: dict[tuple[str, str], float] = {}
    if matched_points is not None and len(matched_points) > 1:
        ordered = matched_points.sort_values("point_seq", kind="stable")
        edge_sequence = ordered.edge_uid.astype(str).tolist()
        timestamps = pd.to_numeric(ordered.timestamp, errors="coerce").to_numpy(float)
        route_edges = route_parts.edge_uid.astype(str).tolist()
        route_cursor = 0
        for left, right, delta in zip(edge_sequence[:-1], edge_sequence[1:], np.diff(timestamps)):
            if left != right and delta >= 0:
                try:
                    left_index = route_edges.index(left, route_cursor)
                    right_index = route_edges.index(right, left_index + 1)
                    allocation_key = (route_edges[left_index], route_edges[left_index + 1])
                    route_cursor = min(left_index + 1, right_index)
                except (ValueError, IndexError):
                    allocation_key = (left, right)
                movement_times[allocation_key] = movement_times.get(allocation_key, 0.0) + float(delta)
    consumed_pairs: set[tuple[str, str]] = set()
    for sequence, (left, right) in enumerate(zip(route_parts.itertuples(), route_parts.iloc[1:].itertuples())):
        key = (left.edge_uid, right.edge_uid)
        movement = movement_router.movement(str(left.edge_uid), str(right.edge_uid))
        if movement is not None:
            row = {
                "via_node": int(movement.via_node), "movement_type": movement.movement_type,
                "turn_angle": float(movement.turn_angle), "restriction_status": movement.restriction_status,
                "level_transition_type": movement.level_transition_type,
                "movement_quality": "valid",
            }
        else:
            level_gap = int(left.to_node) == int(right.from_node)
            row = {
                "via_node": int(left.to_node), "movement_type": "gap", "turn_angle": math.nan,
                "restriction_status": "unknown", "movement_quality": "missing_movement",
                "level_transition_type": "unresolved_level_gap" if level_gap else "topology_gap",
            }
        rows.append({
            "order_id": order_id, "movement_sequence": sequence,
            "from_edge_uid": left.edge_uid, "to_edge_uid": right.edge_uid,
            "movement_observed": not bool(left.is_interpolated) and not bool(right.is_interpolated),
            "observed_interval_time_s": movement_times.get(key, 0.0) if key not in consumed_pairs else 0.0,
            **row,
        })
        consumed_pairs.add(key)
    return pd.DataFrame(rows)
