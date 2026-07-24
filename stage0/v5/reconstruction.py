"""Edge-aware route reconstruction and static inferred-path accounting."""

from __future__ import annotations

import math
import json
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
    gap_reasons: tuple[str, ...] = ()


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

    def bridge(self, left: str, right: str, cutoff: float | None = None) -> list[str] | None:
        if left == right:
            return [left]
        result = self.router.bridge(left, right, self.maximum if cutoff is None else cutoff)
        return None if result is None else result[0]

    def reconstruct(
        self,
        matched_points: pd.DataFrame,
        precomputed_paths: object | None = None,
    ) -> ReconstructedRoute:
        del precomputed_paths  # transition evidence is persisted on the right-hand matched point
        self.last_bridge_search_ms = 0.0
        self.last_precomputed_path_count = 0
        self.last_path_search_count = 0
        if matched_points.empty or matched_points.edge_uid.isna().any():
            return ReconstructedRoute([], [], "rejected", 1, ("missing_selected_state",))
        ordered = (
            matched_points.sort_values("point_seq", kind="stable")
            if "point_seq" in matched_points.columns
            else matched_points
        ).reset_index(drop=True)
        route = [str(ordered.edge_uid.iloc[0])]
        observed = [True]
        gaps = 0
        reasons: list[str] = []
        for point_index in range(1, len(ordered)):
            left = str(ordered.edge_uid.iloc[point_index - 1])
            right = str(ordered.edge_uid.iloc[point_index])
            if left == right:
                continue
            encoded = str(ordered.get("selected_path_json", pd.Series("", index=ordered.index)).iloc[point_index] or "")
            path: list[str] | None = None
            if encoded:
                try:
                    decoded = json.loads(encoded)
                    path = [str(value) for value in decoded]
                except (TypeError, ValueError, json.JSONDecodeError):
                    reasons.append("invalid_selected_path_encoding")
            cutoff = float(pd.to_numeric(
                ordered.get("transition_cutoff_m", pd.Series(np.nan, index=ordered.index)).iloc[point_index],
                errors="coerce",
            ))
            selected_distance = float(pd.to_numeric(
                ordered.get("selected_path_distance_m", pd.Series(np.nan, index=ordered.index)).iloc[point_index],
                errors="coerce",
            ))
            if (
                path is None
                or not path
                or path[0] != left
                or path[-1] != right
                or not math.isfinite(cutoff)
                or not math.isfinite(selected_distance)
                or selected_distance > cutoff + 1e-6
            ):
                path = None
                reasons.append("missing_or_cutoff_inconsistent_transition_path")
            else:
                self.last_precomputed_path_count += 1
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
        return ReconstructedRoute(
            route, observed, "complete" if gaps == 0 else "gap", gaps, tuple(reasons)
        )


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


def _aligned_observed_runs(matched_points: pd.DataFrame, route_parts: pd.DataFrame) -> pd.DataFrame:
    """Align contiguous point-state visits to observed route-part occurrences."""
    if matched_points.empty:
        return pd.DataFrame()
    ordered = matched_points.sort_values("point_seq", kind="stable").reset_index(drop=True).copy()
    ordered["point_run_id"] = ordered.edge_uid.astype(str).ne(
        ordered.edge_uid.astype(str).shift()
    ).cumsum().astype("int32") - 1
    observed_parts = route_parts.loc[~route_parts.is_interpolated].sort_values(
        "route_sequence", kind="stable"
    )
    available = list(observed_parts[["route_sequence", "edge_uid"]].itertuples(index=False, name=None))
    cursor = 0
    rows: list[dict[str, Any]] = []
    for run_id, group in ordered.groupby("point_run_id", sort=True):
        edge_uid = str(group.edge_uid.iloc[0])
        route_sequence = -1
        for position in range(cursor, len(available)):
            if str(available[position][1]) == edge_uid:
                route_sequence = int(available[position][0])
                cursor = position + 1
                break
        timestamps = pd.to_numeric(group.timestamp, errors="coerce").to_numpy(float)
        positions = pd.to_numeric(
            group.get("position_on_edge", pd.Series(np.nan, index=group.index)), errors="coerce"
        ).to_numpy(float)
        xy = group[["metric_x", "metric_y"]].to_numpy(float) if {"metric_x", "metric_y"} <= set(group) else np.empty((0, 2))
        rows.append({
            "point_run_id": int(run_id),
            "route_sequence": route_sequence,
            "edge_uid": edge_uid,
            "enter_time": float(timestamps[0]),
            "exit_time": float(timestamps[-1]),
            "travel_time_s": float(np.diff(timestamps).sum()) if len(timestamps) > 1 else 0.0,
            "entry_position_m": float(positions[0]) if len(positions) and np.isfinite(positions[0]) else math.nan,
            "exit_position_m": float(positions[-1]) if len(positions) and np.isfinite(positions[-1]) else math.nan,
            "projected_progress_m": float(np.maximum(np.diff(positions), 0.0).sum()) if len(positions) > 1 and np.isfinite(positions).all() else math.nan,
            "observed_distance_m": float(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])).sum()) if len(xy) > 1 else 0.0,
            "observed_point_count": int(len(group)),
        })
    return pd.DataFrame(rows)


def _route_allocations(
    route_parts: pd.DataFrame, observed_runs: pd.DataFrame, edges: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Return position-aware route-part distances without modifying the route identity."""
    lookup = _edge_lookup(edges)
    run_lookup = observed_runs.set_index("route_sequence", drop=False) if len(observed_runs) else pd.DataFrame()
    ordered = route_parts.sort_values("route_sequence", kind="stable").reset_index(drop=True)
    allocations: list[dict[str, float]] = []
    last_index = len(ordered) - 1
    for index, part in ordered.iterrows():
        length = float(lookup.loc[str(part.edge_uid)].length_m)
        run = run_lookup.loc[int(part.route_sequence)] if len(run_lookup) and int(part.route_sequence) in run_lookup.index else None
        raw_entry = float(run.entry_position_m) if run is not None and np.isfinite(run.entry_position_m) else 0.0
        raw_exit = float(run.exit_position_m) if run is not None and np.isfinite(run.exit_position_m) else length
        entry = min(max(raw_entry, 0.0), length)
        exit_ = min(max(raw_exit, 0.0), length)
        position_valid = 0.0 <= raw_entry <= length and 0.0 <= raw_exit <= length
        if last_index == 0 and raw_exit < raw_entry:
            position_valid = False
        if last_index == 0:
            allocated = max(0.0, exit_ - entry)
        elif index == 0:
            allocated = max(0.0, length - entry)
        elif index == last_index:
            allocated = max(0.0, exit_)
        else:
            allocated = max(0.0, length)
        allocations.append({
            "route_sequence": int(part.route_sequence),
            "entry_position_m": entry,
            "exit_position_m": exit_,
            "allocated_distance_m": allocated,
            "position_distance_valid": bool(position_valid),
            "position_adjustment_m": abs(raw_entry - entry) + abs(raw_exit - exit_),
        })
    return pd.DataFrame(allocations)


def projected_route_distance_m(
    matched_points: pd.DataFrame, route_parts: pd.DataFrame, edges: gpd.GeoDataFrame,
) -> float:
    """Independently accumulate projected progress and cross-edge route movement."""
    runs = _aligned_observed_runs(matched_points, route_parts).sort_values("point_run_id", kind="stable")
    if runs.empty or (runs.route_sequence < 0).any():
        return math.nan
    lookup = _edge_lookup(edges)
    route_lookup = route_parts.set_index("route_sequence", drop=False)
    total = float(pd.to_numeric(runs.projected_progress_m, errors="coerce").fillna(0.0).sum())
    for left, right in zip(runs.iloc[:-1].itertuples(), runs.iloc[1:].itertuples()):
        left_length = float(lookup.loc[str(left.edge_uid)].length_m)
        left_exit = min(max(float(left.exit_position_m), 0.0), left_length)
        right_length = float(lookup.loc[str(right.edge_uid)].length_m)
        right_entry = min(max(float(right.entry_position_m), 0.0), right_length)
        middle = 0.0
        for sequence in range(int(left.route_sequence) + 1, int(right.route_sequence)):
            if sequence not in route_lookup.index:
                return math.nan
            middle += float(lookup.loc[str(route_lookup.loc[sequence].edge_uid)].length_m)
        total += max(0.0, left_length - left_exit) + middle + max(0.0, right_entry)
    return total


def add_position_aware_route_distances(
    matched_points: pd.DataFrame, route_parts: pd.DataFrame, edges: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Attach partial first/last-edge distance fields to the route product."""
    if route_parts.empty:
        return route_parts.copy()
    runs = _aligned_observed_runs(matched_points, route_parts)
    allocations = _route_allocations(route_parts, runs, edges)
    aligned = bool(
        len(runs) == int((~route_parts.is_interpolated).sum())
        and (runs.route_sequence >= 0).all()
        and runs.route_sequence.is_unique
    )
    result = route_parts.merge(allocations, on="route_sequence", how="left", validate="one_to_one")
    result["observed_run_alignment_valid"] = aligned
    return result


def build_traversals(
    order_id: str, matched_points: pd.DataFrame, route_parts: pd.DataFrame, edges: gpd.GeoDataFrame
) -> pd.DataFrame:
    lookup = _edge_lookup(edges)
    observed_runs = _aligned_observed_runs(matched_points, route_parts)
    allocations = (
        route_parts[[
            "route_sequence", "entry_position_m", "exit_position_m", "allocated_distance_m",
            "position_distance_valid", "position_adjustment_m",
        ]].copy()
        if {"entry_position_m", "exit_position_m", "allocated_distance_m"} <= set(route_parts.columns)
        else _route_allocations(route_parts, observed_runs, lookup)
    )
    allocation_lookup = allocations.set_index("route_sequence") if len(allocations) else pd.DataFrame()
    run_lookup = observed_runs.set_index("route_sequence") if len(observed_runs) else pd.DataFrame()
    total_distance = float(allocations.allocated_distance_m.sum()) if len(allocations) else 0.0
    inferred_sequences = set(route_parts.loc[route_parts.is_interpolated, "route_sequence"].astype(int))
    inferred_distance = float(allocations.loc[allocations.route_sequence.isin(inferred_sequences), "allocated_distance_m"].sum())
    rows = []
    for row in route_parts.itertuples():
        sequence = int(row.route_sequence)
        allocation = allocation_lookup.loc[sequence]
        observed_run = run_lookup.loc[sequence] if len(run_lookup) and sequence in run_lookup.index else None
        if row.is_interpolated or observed_run is None:
            enter_time = exit_time = pd.NaT
            travel_time = math.nan
            observed_distance = 0.0
            point_count = 0
            quality = "static_inferred_only"
        else:
            enter_time, exit_time = float(observed_run.enter_time), float(observed_run.exit_time)
            travel_time = float(observed_run.travel_time_s)
            observed_distance = float(observed_run.observed_distance_m)
            point_count = int(observed_run.observed_point_count)
            quality = "observed_support"
        rows.append({
            "order_id": order_id, "traversal_id": sequence, "edge_uid": row.edge_uid,
            "route_sequence": sequence,
            "enter_time": enter_time, "exit_time": exit_time, "travel_time_s": travel_time,
            "entry_position_m": float(allocation.entry_position_m),
            "exit_position_m": float(allocation.exit_position_m),
            "observed_distance_m": observed_distance,
            "allocated_distance_m": float(allocation.allocated_distance_m),
            "observed_point_count": point_count,
            "traversal_source": "inferred_path" if row.is_interpolated else "observed",
            "traversal_quality": quality, "is_interpolated": bool(row.is_interpolated),
            "interpolated_distance_share": inferred_distance / total_distance if total_distance else 0.0,
            "observed_interval_time_s": travel_time,
        })
    return pd.DataFrame(rows)


def build_unresolved_intervals(
    order_id: str,
    route_parts: pd.DataFrame,
    movement_router: CompactMovementRouter,
    matched_points: pd.DataFrame,
) -> pd.DataFrame:
    """Keep cross-observation time separate when the route contains inferred or illegal legs."""
    runs = _aligned_observed_runs(matched_points, route_parts)
    if len(runs) < 2:
        return pd.DataFrame(columns=["order_id", "unresolved_interval_id", "unresolved_interval_time_s"])
    rows: list[dict[str, Any]] = []
    ordered = runs.sort_values("point_run_id", kind="stable").reset_index(drop=True)
    for interval_id, (left, right) in enumerate(zip(ordered.iloc[:-1].itertuples(), ordered.iloc[1:].itertuples())):
        direct_sequence = int(right.route_sequence) == int(left.route_sequence) + 1
        direct_movement = movement_router.movement(str(left.edge_uid), str(right.edge_uid)) if direct_sequence else None
        if direct_sequence and direct_movement is not None:
            continue
        reason = "inferred_path_between_observations" if int(right.route_sequence) > int(left.route_sequence) + 1 else "unresolved_direct_movement"
        rows.append({
            "order_id": order_id,
            "unresolved_interval_id": int(interval_id),
            "from_point_run_id": int(left.point_run_id),
            "to_point_run_id": int(right.point_run_id),
            "from_edge_uid": str(left.edge_uid),
            "to_edge_uid": str(right.edge_uid),
            "from_route_sequence": int(left.route_sequence),
            "to_route_sequence": int(right.route_sequence),
            "unresolved_interval_time_s": max(0.0, float(right.enter_time) - float(left.exit_time)),
            "interval_time_source": "unresolved_between_observations",
            "unresolved_reason": reason,
        })
    return pd.DataFrame(rows)


def build_movements(
    order_id: str,
    route_parts: pd.DataFrame,
    movement_router: CompactMovementRouter,
    matched_points: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    movement_times: dict[int, float] = {}
    if matched_points is not None and len(matched_points) > 1:
        runs = _aligned_observed_runs(matched_points, route_parts).sort_values("point_run_id", kind="stable")
        for left, right in zip(runs.iloc[:-1].itertuples(), runs.iloc[1:].itertuples()):
            sequence = int(left.route_sequence)
            direct = int(right.route_sequence) == sequence + 1
            if direct and movement_router.movement(str(left.edge_uid), str(right.edge_uid)) is not None:
                movement_times[sequence] = max(0.0, float(right.enter_time) - float(left.exit_time))
    for sequence, (left, right) in enumerate(zip(route_parts.itertuples(), route_parts.iloc[1:].itertuples())):
        key = (left.edge_uid, right.edge_uid)
        movement = movement_router.movement(str(left.edge_uid), str(right.edge_uid))
        raw_movement = movement_router.raw_movement(str(left.edge_uid), str(right.edge_uid))
        if movement is not None:
            row = {
                "via_node": int(movement.via_node), "movement_type": movement.movement_type,
                "turn_angle": float(movement.turn_angle), "restriction_status": movement.restriction_status,
                "level_transition_type": movement.level_transition_type,
                "movement_quality": "valid",
                "movement_audit_reason": "allowed",
            }
        elif raw_movement is not None:
            restriction = str(raw_movement.restriction_status)
            suspicious = raw_movement.level_transition_type == "suspicious_level_jump"
            if restriction.startswith("forbidden"):
                audit_reason = "restriction_block"
            elif restriction == "unresolved_restriction":
                audit_reason = "unparsed_restriction_exposure"
            elif suspicious or not raw_movement.layer_compatibility:
                audit_reason = "suspicious_level_transition"
            else:
                audit_reason = "raw_movement_not_routable"
            row = {
                "via_node": int(raw_movement.via_node),
                "movement_type": raw_movement.movement_type,
                "turn_angle": float(raw_movement.turn_angle),
                "restriction_status": restriction,
                "movement_quality": audit_reason,
                "movement_audit_reason": audit_reason,
                "level_transition_type": raw_movement.level_transition_type,
            }
        else:
            row = {
                "via_node": int(left.to_node), "movement_type": "gap", "turn_angle": math.nan,
                "restriction_status": "missing_topology", "movement_quality": "missing_topology",
                "movement_audit_reason": "missing_topology",
                "level_transition_type": "true_layer_discontinuity"
                if int(left.to_node) == int(right.from_node) else "topology_gap",
            }
        rows.append({
            "order_id": order_id, "movement_sequence": sequence,
            "from_edge_uid": left.edge_uid, "to_edge_uid": right.edge_uid,
            "movement_observed": sequence in movement_times,
            "observed_interval_time_s": movement_times.get(sequence, 0.0),
            "dynamic_time_source": "observed_direct_movement" if sequence in movement_times else "none_static_only",
            **row,
        })
    return pd.DataFrame(rows)
