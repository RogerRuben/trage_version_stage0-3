"""HMM/Viterbi map matcher for partitioned Xi'an Stage0 matched points.

The geometric matcher output is used as the cleaned GPS input and as an explicit
fallback. HMM states are exact projections onto nearby road links; transition
scores compare routed network distance with observed GPS displacement.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage0.canonical.topology import build_multidigraph, minimum_parallel_edge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-dir", type=Path, required=True)
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("stage0_output"))
    parser.add_argument("--date", required=True)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--candidate-radius-m", type=float, default=80.0)
    parser.add_argument("--sigma-z", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--limit-parts", type=int)
    parser.add_argument("--max-orders-per-part", type=int)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    return parser.parse_args()


class HMMRoadNetwork:
    def __init__(self, roads_path: Path, nodes_path: Path, spacing_m: float = 15.0):
        self.roads = gpd.read_parquet(roads_path).to_crs(32649).reset_index(drop=True)
        self.nodes = gpd.read_parquet(nodes_path).to_crs(32649).reset_index(drop=True)
        self.geoms = self.roads.geometry.to_numpy()
        self.lengths = shapely.length(self.geoms).astype(float)
        self.forward = self.roads.oneway_code.fillna("B").astype(str).isin(["F", "B"]).to_numpy()
        self.reverse = self.roads.oneway_code.fillna("B").astype(str).isin(["T", "B"]).to_numpy()
        coords: list[np.ndarray] = []
        road_ids: list[np.ndarray] = []
        for i, line in enumerate(self.geoms):
            distances = np.arange(0, max(line.length, 1.0) + spacing_m / 2, spacing_m)
            points = shapely.line_interpolate_point(line, distances)
            coords.append(np.column_stack([shapely.get_x(points), shapely.get_y(points)]))
            road_ids.append(np.full(len(distances), i, dtype="int32"))
        self.sample_coords = np.vstack(coords)
        self.sample_road_ids = np.concatenate(road_ids)
        self.tree = cKDTree(self.sample_coords)
        self.to_wgs84 = Transformer.from_crs(32649, 4326, always_xy=True)

        self.graph = build_multidigraph(
            {
                "road_idx": i,
                "from_node": int(row.from_node),
                "to_node": int(row.to_node),
                "length": float(self.lengths[i]),
                "oneway_code": str(row.oneway_code) if pd.notna(row.oneway_code) else "B",
                "link_id": str(row.link_id),
            }
            for i, row in self.roads.iterrows()
        )
        self.node_xy = {
            int(row.node_id): (float(row.geometry.x), float(row.geometry.y))
            for _, row in self.nodes.iterrows()
        }
        edge_rows, edge_cols, edge_weights = [], [], []
        minimum_directed: dict[tuple[int, int], float] = {}
        for u, v, data in self.graph.edges(data=True):
            key = (int(u), int(v))
            minimum_directed[key] = min(minimum_directed.get(key, math.inf), float(data["weight"]))
        for (u, v), weight in minimum_directed.items():
            edge_rows.append(u); edge_cols.append(v); edge_weights.append(weight)
        node_count = max(max(self.graph.nodes, default=0), int(self.nodes.node_id.max())) + 1
        self.csgraph = coo_matrix(
            (edge_weights, (edge_rows, edge_cols)), shape=(node_count, node_count), dtype=float
        ).tocsr()

    def heuristic(self, a: int, b: int) -> float:
        pa = self.node_xy.get(int(a)); pb = self.node_xy.get(int(b))
        if pa is None or pb is None:
            return 0.0
        return math.hypot(pa[0] - pb[0], pa[1] - pb[1])

    @lru_cache(maxsize=512)
    def distance_row(self, source: int) -> np.ndarray:
        if source < 0 or source >= self.csgraph.shape[0]:
            return np.full(self.csgraph.shape[0], np.inf)
        return np.asarray(
            dijkstra(self.csgraph, directed=True, indices=source, limit=3000.0, return_predecessors=False)
        )

    def node_distance(self, source: int, target: int) -> float:
        if source == target:
            return 0.0
        if target < 0 or target >= self.csgraph.shape[0]:
            return math.inf
        directed = float(self.distance_row(int(source))[int(target)])
        if math.isfinite(directed):
            return directed
        return math.inf

    def direct_transition_node(self, road_a: int, road_b: int) -> int | None:
        exits = {node for node, _ in self.exit_options(road_a, self.lengths[road_a] / 2)}
        entries = {node for node, _ in self.entry_options(road_b, self.lengths[road_b] / 2)}
        common = sorted(exits & entries)
        return common[0] if common else None

    def is_directed_link_transition(self, road_a: int, road_b: int) -> bool:
        return self.direct_transition_node(road_a, road_b) is not None

    def candidate_arrays(self, x: np.ndarray, y: np.ndarray, k: int, radius: float):
        query_k = max(12, k * 4)
        # The day runner already parallelizes by process. Spawning all CPU
        # threads again for every small order causes severe oversubscription.
        _, sample_idx = self.tree.query(np.column_stack([x, y]), k=query_k, workers=1)
        sampled_roads = self.sample_road_ids[sample_idx]
        candidates = np.full((len(x), k), -1, dtype="int32")
        for i, values in enumerate(sampled_roads):
            unique: list[int] = []
            seen: set[int] = set()
            for value in values:
                value = int(value)
                if value not in seen:
                    seen.add(value); unique.append(value)
                    if len(unique) == k:
                        break
            candidates[i, : len(unique)] = unique
        valid = candidates >= 0
        points = shapely.points(x, y)
        flat_roads = candidates[valid]
        repeated_points = np.repeat(points, k)[valid.ravel()]
        lines = self.geoms[flat_roads]
        distance_flat = shapely.distance(repeated_points, lines)
        position_flat = shapely.line_locate_point(lines, repeated_points)
        distance = np.full(candidates.shape, np.inf, dtype=float)
        position = np.full(candidates.shape, np.nan, dtype=float)
        distance[valid] = distance_flat; position[valid] = position_flat
        distance[distance > radius] = np.inf
        snapped = np.empty(candidates.shape, dtype=object)
        snapped[:] = None
        valid_radius = np.isfinite(distance)
        snapped[valid_radius] = shapely.line_interpolate_point(
            self.geoms[candidates[valid_radius]], position[valid_radius]
        )
        return candidates, distance, position, snapped

    def exit_options(self, road: int, position: float) -> list[tuple[int, float]]:
        row = self.roads.iloc[road]; length = self.lengths[road]
        options: list[tuple[int, float]] = []
        if self.forward[road]: options.append((int(row.to_node), max(0.0, length - position)))
        if self.reverse[road]: options.append((int(row.from_node), max(0.0, position)))
        return options

    def entry_options(self, road: int, position: float) -> list[tuple[int, float]]:
        row = self.roads.iloc[road]; length = self.lengths[road]
        options: list[tuple[int, float]] = []
        if self.forward[road]: options.append((int(row.from_node), max(0.0, position)))
        if self.reverse[road]: options.append((int(row.to_node), max(0.0, length - position)))
        return options

    def transition_distance(self, road_a: int, pos_a: float, road_b: int, pos_b: float) -> float:
        if road_a == road_b:
            values = []
            if self.forward[road_a] and pos_b >= pos_a: values.append(pos_b - pos_a)
            if self.reverse[road_a] and pos_b <= pos_a: values.append(pos_a - pos_b)
            if values:
                return float(min(values))
        best = math.inf
        for exit_node, exit_distance in self.exit_options(road_a, pos_a):
            for entry_node, entry_distance in self.entry_options(road_b, pos_b):
                middle = self.node_distance(exit_node, entry_node)
                if math.isfinite(middle):
                    best = min(best, exit_distance + middle + entry_distance)
        return float(best)

    @lru_cache(maxsize=200_000)
    def link_path_indices(self, road_a: int, road_b: int) -> tuple[tuple[int, ...], str]:
        if road_a == road_b:
            return (road_a,), "same_link"
        best = math.inf; endpoints: tuple[int, int] | None = None
        for exit_node, exit_distance in self.exit_options(road_a, self.lengths[road_a] / 2):
            for entry_node, entry_distance in self.entry_options(road_b, self.lengths[road_b] / 2):
                middle = self.node_distance(exit_node, entry_node)
                total = exit_distance + middle + entry_distance
                if total < best:
                    best = total; endpoints = (exit_node, entry_node)
        if endpoints is None or not math.isfinite(best):
            return (road_a, road_b), "gap"
        source, target = endpoints
        status = "directed"
        try:
            nodes = nx.shortest_path(self.graph, source, target, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return (road_a, road_b), "gap"
        path = [road_a]
        for u, v in zip(nodes[:-1], nodes[1:]):
            path.append(int(minimum_parallel_edge(self.graph, int(u), int(v))["road_idx"]))
        path.append(road_b)
        compact = [path[0]]
        for value in path[1:]:
            if value != compact[-1]: compact.append(value)
        return tuple(compact), status


def calibrate(frame: pd.DataFrame, sigma: float | None, beta: float | None) -> tuple[float, float]:
    distances = frame.gps_to_link_dist_m.replace([np.inf, -np.inf], np.nan).dropna()
    p90 = float(distances.quantile(0.9)) if len(distances) else 15.0
    sigma_z = float(sigma) if sigma is not None else float(np.clip(p90 / 1.645, 8.0, 30.0))
    beta_value = float(beta) if beta is not None else max(40.0, sigma_z * 5.0)
    return sigma_z, beta_value


def topology_audit(roads: np.ndarray, gps_step: np.ndarray, network: HMMRoadNetwork) -> np.ndarray:
    previous = np.roll(roads, 1); previous[0] = roads[0]
    changed = roads != previous
    from_node = network.roads.from_node.to_numpy(); to_node = network.roads.to_node.to_numpy()
    shared = (
        (from_node[roads] == from_node[previous]) | (from_node[roads] == to_node[previous])
        | (to_node[roads] == from_node[previous]) | (to_node[roads] == to_node[previous])
    )
    line_distance = np.zeros(len(roads), dtype=float)
    idx = np.flatnonzero(changed)
    if len(idx):
        line_distance[idx] = shapely.distance(network.geoms[previous[idx]], network.geoms[roads[idx]])
    starts = shapely.get_point(network.geoms[roads], 0); ends = shapely.get_point(network.geoms[roads], -1)
    sx, sy, ex, ey = shapely.get_x(starts), shapely.get_y(starts), shapely.get_x(ends), shapely.get_y(ends)
    psx, psy, pex, pey = np.roll(sx, 1), np.roll(sy, 1), np.roll(ex, 1), np.roll(ey, 1)
    endpoint_min = np.minimum.reduce([
        np.hypot(psx - sx, psy - sy), np.hypot(psx - ex, psy - ey),
        np.hypot(pex - sx, pey - sy), np.hypot(pex - ex, pey - ey),
    ])
    threshold = np.maximum(120.0, gps_step * 2.5 + 40.0)
    gaps = changed & ~shared & (line_distance > 2.0) & (endpoint_min > threshold)
    gaps[0] = False
    return gaps


def viterbi_order(group: pd.DataFrame, network: HMMRoadNetwork, k: int, radius: float, sigma: float, beta: float):
    group = group.sort_values(["timestamp", "source_row"], kind="stable").copy().reset_index(drop=True)
    group["geometric_link_id"] = group.link_id
    geometric_columns = [
        "link_id", "road_class", "road_name", "speed_limit", "oneway_code",
        "gps_to_link_dist_m", "snap_x", "snap_y", "topology_gap",
    ]
    geometric = group[geometric_columns].copy()
    if {"proj_x", "proj_y"}.issubset(group.columns):
        x = group.proj_x.to_numpy(dtype=float); y = group.proj_y.to_numpy(dtype=float)
    else:
        # Older persisted matched points omit projected raw coordinates; reconstruct from WGS84.
        transformer = Transformer.from_crs(4326, 32649, always_xy=True)
        x, y = transformer.transform(group.lon.to_numpy(dtype=float), group.lat.to_numpy(dtype=float))
        x = np.asarray(x); y = np.asarray(y)
    candidates, distances, positions, snapped = network.candidate_arrays(x, y, k, radius)
    n = len(group)
    emission_log = -0.5 * (distances / sigma) ** 2
    emission_log[~np.isfinite(emission_log)] = -60.0
    dp = np.full((n, k), -np.inf, dtype=float)
    back = np.full((n, k), -1, dtype="int16")
    selected_transition_log = np.zeros((n, k), dtype=float)
    selected_routed_distance = np.full((n, k), np.nan, dtype=float)
    dp[0] = emission_log[0]
    gps_step = group.segment_distance_m.fillna(0).to_numpy(dtype=float)
    dt = group.dt_s.fillna(0).to_numpy(dtype=float)
    for i in range(1, n):
        if dt[i] > 120 or not np.isfinite(dp[i - 1]).any():
            dp[i] = emission_log[i] + np.nanmax(dp[i - 1])
            continue
        for j in range(k):
            road_b = int(candidates[i, j])
            if road_b < 0 or not np.isfinite(distances[i, j]):
                continue
            best_score = -np.inf; best_prev = -1; best_transition = -60.0; best_routed = math.inf
            for h in range(k):
                road_a = int(candidates[i - 1, h])
                if road_a < 0 or not np.isfinite(dp[i - 1, h]):
                    continue
                routed = network.transition_distance(road_a, float(positions[i - 1, h]), road_b, float(positions[i, j]))
                if not math.isfinite(routed):
                    transition_log = -60.0
                else:
                    transition_log = -abs(routed - gps_step[i]) / beta
                    if routed > max(500.0, gps_step[i] * 5.0 + 200.0):
                        transition_log -= min(30.0, (routed - gps_step[i]) / beta)
                score = dp[i - 1, h] + transition_log + emission_log[i, j]
                if score > best_score:
                    best_score = score; best_prev = h; best_transition = transition_log; best_routed = routed
            dp[i, j] = best_score; back[i, j] = best_prev
            selected_transition_log[i, j] = best_transition; selected_routed_distance[i, j] = best_routed
    states = np.full(n, -1, dtype="int16")
    states[-1] = int(np.nanargmax(dp[-1])) if np.isfinite(dp[-1]).any() else 0
    for i in range(n - 1, 0, -1):
        previous = int(back[i, states[i]]) if states[i] >= 0 else -1
        states[i - 1] = previous if previous >= 0 else int(np.nanargmax(dp[i - 1]))
    row = np.arange(n)
    roads = candidates[row, states]
    chosen_distance = distances[row, states]
    chosen_position = positions[row, states]
    chosen_snapped = snapped[row, states]
    valid = (roads >= 0) & np.isfinite(chosen_distance) & pd.notna(chosen_snapped)
    matched_fraction = float(valid.mean())
    mean_dist = float(np.nanmean(np.where(valid, chosen_distance, np.nan))) if valid.any() else math.inf
    transition_logs = selected_transition_log[row, states]
    routed_distances = selected_routed_distance[row, states]
    finite_transitions = transition_logs[1:][np.isfinite(transition_logs[1:])]
    transition_quality = math.exp(float(np.mean(np.clip(finite_transitions, -20, 0)))) if len(finite_transitions) else 0.0
    confidence = matched_fraction * math.exp(-mean_dist / max(sigma * 2, 1)) * transition_quality
    fallback = matched_fraction < 0.85 or confidence < 0.03 or not valid.all()
    reason = "" if not fallback else (
        "low_matched_fraction" if matched_fraction < 0.85 else "low_hmm_confidence" if confidence < 0.03 else "candidate_gap"
    )
    if fallback:
        group["candidate_link_id"] = group.link_id
        group["matched_link_id"] = group.link_id
        group["proj_dist_m"] = group.gps_to_link_dist_m
        group["proj_x_hmm"] = group.snap_x
        group["proj_y_hmm"] = group.snap_y
        group["emission_score"] = np.nan; group["transition_score"] = np.nan; group["viterbi_score"] = np.nan
        group["transition_network_dist_m"] = np.nan
        group["path_gap_flag"] = group.topology_gap
        group["matcher_version"] = "hmm_viterbi_fallback_geometric"
    else:
        selected_roads = network.roads.iloc[roads]
        group["candidate_link_id"] = selected_roads.link_id.to_numpy()
        group["matched_link_id"] = selected_roads.link_id.to_numpy()
        group["link_id"] = selected_roads.link_id.to_numpy()
        group["road_class"] = selected_roads.road_class.to_numpy()
        group["road_name"] = selected_roads.road_name.to_numpy()
        group["speed_limit"] = selected_roads.speed_limit.to_numpy()
        group["oneway_code"] = selected_roads.oneway_code.to_numpy()
        group["proj_dist_m"] = chosen_distance
        group["gps_to_link_dist_m"] = chosen_distance
        group["proj_x_hmm"] = shapely.get_x(chosen_snapped)
        group["proj_y_hmm"] = shapely.get_y(chosen_snapped)
        group["snap_x"] = group.proj_x_hmm; group["snap_y"] = group.proj_y_hmm
        group["emission_score"] = np.exp(np.clip(emission_log[row, states], -60, 0))
        group["transition_score"] = np.exp(np.clip(transition_logs, -60, 0))
        group["transition_network_dist_m"] = routed_distances
        group["viterbi_score"] = dp[row, states] / (row + 1)
        detour_limit = np.maximum(500.0, gps_step * 5.0 + 200.0)
        group["path_gap_flag"] = ~np.isfinite(routed_distances)
        group["unreasonable_detour_flag"] = np.isfinite(routed_distances) & (routed_distances > detour_limit)
        group.loc[0, "path_gap_flag"] = False
        group.loc[0, "unreasonable_detour_flag"] = False
        group["topology_gap"] = topology_audit(roads, gps_step, network)
        group["matcher_version"] = "hmm_viterbi"
        geometric_gap_count = int(geometric.topology_gap.fillna(False).sum())
        hmm_gap_count = int(group.topology_gap.sum())
        geometric_p90 = float(geometric.gps_to_link_dist_m.quantile(0.9))
        hmm_p90 = float(group.proj_dist_m.quantile(0.9))
        if hmm_gap_count > geometric_gap_count or (
            hmm_gap_count == geometric_gap_count and hmm_p90 > geometric_p90 + 10.0
        ):
            fallback = True; reason = "non_degradation_guard"
            for column in geometric_columns:
                group[column] = geometric[column].to_numpy()
            group["candidate_link_id"] = group.link_id
            group["matched_link_id"] = group.link_id
            group["proj_dist_m"] = group.gps_to_link_dist_m
            group["proj_x_hmm"] = group.snap_x; group["proj_y_hmm"] = group.snap_y
            group["path_gap_flag"] = group.topology_gap.fillna(False)
            group["unreasonable_detour_flag"] = False
            group["emission_score"] = np.nan; group["transition_score"] = np.nan
            group["viterbi_score"] = np.nan; group["transition_network_dist_m"] = np.nan
            group["matcher_version"] = "hmm_viterbi_fallback_geometric"
    if "unreasonable_detour_flag" not in group:
        group["unreasonable_detour_flag"] = False
    transitions = max(len(group) - 1, 1)
    confidence *= math.exp(-2.0 * float(group.path_gap_flag.sum()) / transitions)
    confidence *= math.exp(-1.0 * float(group.unreasonable_detour_flag.sum()) / transitions)
    group["point_seq"] = np.arange(n, dtype="int32")
    group["matched_fraction"] = matched_fraction
    group["match_confidence"] = confidence
    group["fallback_used"] = fallback
    group["fallback_reason"] = reason
    lon, lat = network.to_wgs84.transform(group.proj_x_hmm.to_numpy(), group.proj_y_hmm.to_numpy())
    group["proj_lon"] = lon; group["proj_lat"] = lat
    same = group.order_id.eq(group.order_id.shift())
    matched_step = pd.Series(
        np.hypot(group.proj_x_hmm.diff(), group.proj_y_hmm.diff())
    ).where(same).fillna(0)
    gps_length = float(group.segment_distance_m.fillna(0).sum())
    matched_length = float(matched_step.sum())
    return group, {
        "order_id": str(group.order_id.iloc[0]), "matched_fraction": matched_fraction,
        "match_confidence": confidence, "fallback_used": fallback, "fallback_reason": reason,
        "p50_match_dist": float(group.proj_dist_m.quantile(0.5)),
        "p90_match_dist": float(group.proj_dist_m.quantile(0.9)),
        "p95_match_dist": float(group.proj_dist_m.quantile(0.95)),
        "topology_gap_count": int(group.topology_gap.fillna(False).sum()),
        "path_gap_count": int(group.path_gap_flag.sum()),
        "unreasonable_detour_count": int(group.unreasonable_detour_flag.sum()),
        "gps_length_m": gps_length, "matched_route_length_m": matched_length,
        "route_length_ratio": matched_length / gps_length if gps_length > 0 else np.nan,
        "link_sequence_change_ratio": float(group.link_id.astype(str).ne(group.geometric_link_id.astype(str)).mean()),
    }


def compact_routes(frame: pd.DataFrame) -> pd.DataFrame:
    changed = frame.order_id.ne(frame.order_id.shift()) | frame.link_id.ne(frame.link_id.shift())
    routes = frame.loc[changed, ["order_id", "link_id", "point_seq"]].copy()
    routes["route_sequence"] = routes.groupby("order_id").cumcount().astype("int32")
    routes["transition_path_status"] = "selected_state_sequence"
    return routes


def main() -> None:
    args = parse_args()
    network = HMMRoadNetwork(args.roads, args.nodes)
    all_files = sorted(args.matched_dir.glob("*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"no Parquet files in {args.matched_dir}")
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker-index must satisfy 0 <= index < worker-count")
    files = [path for i, path in enumerate(all_files) if i % args.worker_count == args.worker_index]
    files = files[: args.limit_parts] if args.limit_parts else files
    point_dir = args.output_root / "hmm_matched_points" / f"day={args.date}"
    route_dir = args.output_root / "hmm_state_sequences" / f"day={args.date}"
    quality_dir = args.output_root / "hmm_quality_parts" / f"day={args.date}"
    for path in [point_dir, route_dir, quality_dir]: path.mkdir(parents=True, exist_ok=True)
    started = time.time(); processed_orders = 0
    for source in files:
        part = source.stem.split("=")[-1].split("_")[-1]
        point_path = point_dir / f"bucket={part}.parquet"
        route_path = route_dir / f"part={part}.parquet"
        quality_path = quality_dir / f"part={part}.parquet"
        if point_path.exists() and quality_path.exists():
            if not route_path.exists():
                existing = pd.read_parquet(point_path)
                compact_routes(existing).to_parquet(route_path, index=False, compression="zstd")
            continue
        frame = pd.read_parquet(source)
        sigma, beta = calibrate(frame, args.sigma_z, args.beta)
        matched_frames: list[pd.DataFrame] = []; quality_rows: list[dict] = []
        groups = frame.groupby("order_id", sort=False)
        for order_no, (_, group) in enumerate(groups, start=1):
            if args.max_orders_per_part and order_no > args.max_orders_per_part:
                break
            matched, quality = viterbi_order(group, network, args.candidates, args.candidate_radius_m, sigma, beta)
            matched_frames.append(matched); quality_rows.append(quality)
            if order_no % 100 == 0:
                print(f"part={part} orders={order_no:,} elapsed={time.time()-started:.1f}s", flush=True)
        result = pd.concat(matched_frames, ignore_index=True)
        quality = pd.DataFrame(quality_rows)
        routes = compact_routes(result)
        result.to_parquet(point_path, index=False, compression="zstd")
        routes.to_parquet(route_path, index=False, compression="zstd")
        quality.to_parquet(quality_path, index=False, compression="zstd")
        processed_orders += len(quality)
        print(f"completed part={part} orders={len(quality):,} fallback={quality.fallback_used.mean():.2%}", flush=True)
    manifest = {
        "date": args.date, "complete": len(list(point_dir.glob('bucket=*.parquet'))) == len(all_files),
        "processed_orders_this_run": processed_orders, "candidate_count": args.candidates,
        "candidate_radius_m": args.candidate_radius_m, "matcher_version": "hmm_viterbi",
        "worker_count": args.worker_count, "worker_index": args.worker_index,
        "seconds": time.time() - started,
    }
    manifest_dir = args.output_root / "manifests"; manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"day={args.date}.hmm.worker={args.worker_index}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
