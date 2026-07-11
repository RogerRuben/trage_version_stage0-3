"""Fast local-topology matcher with FMM-style gap repair.

This matcher keeps the geometric point projection as the primary point-level
match and only invokes shortest-path repair for non-local link transitions.
It is intended as a production-speed alternative to full per-point HMM/Viterbi.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-dir", type=Path, required=True)
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("stage0/output"))
    parser.add_argument("--date", required=True)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--limit-parts", type=int)
    parser.add_argument("--max-orders-per-part", type=int)
    parser.add_argument("--max-repair-detour-ratio", type=float, default=2.5)
    parser.add_argument("--max-repair-extra-m", type=float, default=250.0)
    parser.add_argument("--raw-gap-repair-detour-ratio", type=float, default=6.0)
    parser.add_argument("--raw-gap-repair-extra-m", type=float, default=800.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


class LocalRoadNetwork:
    def __init__(self, roads_path: Path, nodes_path: Path):
        self.roads = gpd.read_parquet(roads_path).to_crs(32649).reset_index(drop=True)
        self.nodes = gpd.read_parquet(nodes_path).to_crs(32649).reset_index(drop=True)
        self.geoms = self.roads.geometry.to_numpy()
        self.lengths = shapely.length(self.geoms).astype(float)
        self.link_lookup = pd.Series(np.arange(len(self.roads), dtype="int32"), index=self.roads.link_id.astype(str))
        self.forward = self.roads.oneway_code.fillna("B").astype(str).isin(["F", "B"]).to_numpy()
        self.reverse = self.roads.oneway_code.fillna("B").astype(str).isin(["T", "B"]).to_numpy()
        self.graph = nx.DiGraph()
        self.undirected_graph = nx.Graph()
        for idx, row in self.roads.iterrows():
            u, v = int(row.from_node), int(row.to_node)
            length = float(self.lengths[idx])
            if self.forward[idx]:
                self._add_edge(self.graph, u, v, length, idx)
            if self.reverse[idx]:
                self._add_edge(self.graph, v, u, length, idx)
            self._add_edge(self.undirected_graph, u, v, length, idx)
        self.to_wgs84 = Transformer.from_crs(32649, 4326, always_xy=True)

    @staticmethod
    def _add_edge(graph, u: int, v: int, length: float, road_idx: int) -> None:
        existing = graph.get_edge_data(u, v)
        if existing is None or length < existing["weight"]:
            graph.add_edge(u, v, weight=length, road_idx=road_idx)

    def road_indices(self, link_ids: pd.Series) -> np.ndarray:
        values = link_ids.astype(str).map(self.link_lookup)
        if values.isna().any():
            missing = link_ids[values.isna()].astype(str).head(5).tolist()
            raise KeyError(f"unknown link_id values: {missing}")
        return values.to_numpy(dtype="int32")

    def shared_node(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        from_node = self.roads.from_node.to_numpy()
        to_node = self.roads.to_node.to_numpy()
        return (
            (from_node[a] == from_node[b]) | (from_node[a] == to_node[b])
            | (to_node[a] == from_node[b]) | (to_node[a] == to_node[b])
        )

    def local_gap_flags(self, roads: np.ndarray, gps_step: np.ndarray) -> np.ndarray:
        previous = np.roll(roads, 1)
        previous[0] = roads[0]
        changed = roads != previous
        shared = self.shared_node(previous, roads)
        line_distance = np.zeros(len(roads), dtype=float)
        idx = np.flatnonzero(changed)
        if len(idx):
            line_distance[idx] = shapely.distance(self.geoms[previous[idx]], self.geoms[roads[idx]])
        starts = shapely.get_point(self.geoms[roads], 0)
        ends = shapely.get_point(self.geoms[roads], -1)
        sx, sy, ex, ey = shapely.get_x(starts), shapely.get_y(starts), shapely.get_x(ends), shapely.get_y(ends)
        psx, psy, pex, pey = np.roll(sx, 1), np.roll(sy, 1), np.roll(ex, 1), np.roll(ey, 1)
        endpoint_min = np.minimum.reduce([
            np.hypot(psx - sx, psy - sy), np.hypot(psx - ex, psy - ey),
            np.hypot(pex - sx, pey - sy), np.hypot(pex - ex, pey - ey),
        ])
        threshold = np.maximum(120.0, np.nan_to_num(gps_step, nan=0.0) * 2.5 + 40.0)
        gaps = changed & ~shared & (line_distance > 2.0) & (endpoint_min > threshold)
        gaps[0] = False
        return gaps

    def exit_options(self, road: int, position: float | None = None) -> list[tuple[int, float]]:
        row = self.roads.iloc[road]
        length = float(self.lengths[road])
        pos = length / 2 if position is None or not math.isfinite(position) else float(position)
        options: list[tuple[int, float]] = []
        if self.forward[road]:
            options.append((int(row.to_node), max(0.0, length - pos)))
        if self.reverse[road]:
            options.append((int(row.from_node), max(0.0, pos)))
        return options

    def entry_options(self, road: int, position: float | None = None) -> list[tuple[int, float]]:
        row = self.roads.iloc[road]
        length = float(self.lengths[road])
        pos = length / 2 if position is None or not math.isfinite(position) else float(position)
        options: list[tuple[int, float]] = []
        if self.forward[road]:
            options.append((int(row.from_node), max(0.0, pos)))
        if self.reverse[road]:
            options.append((int(row.to_node), max(0.0, length - pos)))
        return options

    @lru_cache(maxsize=200_000)
    def link_path_indices(self, road_a: int, road_b: int) -> tuple[tuple[int, ...], str, float]:
        if road_a == road_b:
            return (road_a,), "same_link", 0.0
        if self.shared_node(np.array([road_a], dtype="int32"), np.array([road_b], dtype="int32"))[0]:
            return (road_a, road_b), "direct_topology", 0.0
        best: tuple[float, int, int] | None = None
        for exit_node, exit_distance in self.exit_options(road_a):
            for entry_node, entry_distance in self.entry_options(road_b):
                try:
                    middle = nx.shortest_path_length(self.graph, exit_node, entry_node, weight="weight")
                    total = float(exit_distance + middle + entry_distance)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                if best is None or total < best[0]:
                    best = (total, exit_node, entry_node)
        status = "fmm_directed"
        graph = self.graph
        if best is None:
            for exit_node, exit_distance in self.exit_options(road_a):
                for entry_node, entry_distance in self.entry_options(road_b):
                    try:
                        middle = nx.shortest_path_length(self.undirected_graph, exit_node, entry_node, weight="weight")
                        total = float(exit_distance + middle * 1.25 + 25.0 + entry_distance)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
                    if best is None or total < best[0]:
                        best = (total, exit_node, entry_node)
            status = "fmm_undirected_relaxation"
            graph = self.undirected_graph
        if best is None:
            return (road_a, road_b), "fmm_gap", math.inf
        total, source, target = best
        try:
            nodes = nx.shortest_path(graph, source, target, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return (road_a, road_b), "fmm_gap", math.inf
        path = [road_a]
        for u, v in zip(nodes[:-1], nodes[1:]):
            path.append(int(graph[u][v]["road_idx"]))
        path.append(road_b)
        compact = [path[0]]
        for value in path[1:]:
            if value != compact[-1]:
                compact.append(value)
        return tuple(compact), status, total


def enrich_points(
    group: pd.DataFrame, network: LocalRoadNetwork, sigma: float,
    max_repair_detour_ratio: float, max_repair_extra_m: float,
    raw_gap_repair_detour_ratio: float, raw_gap_repair_extra_m: float,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    group = group.sort_values(["timestamp", "source_row"], kind="stable").copy().reset_index(drop=True)
    roads = network.road_indices(group.link_id)
    gps_step = group.segment_distance_m.fillna(0).to_numpy(dtype=float)
    local_gaps = network.local_gap_flags(roads, gps_step)
    dist = group.gps_to_link_dist_m.to_numpy(dtype=float)
    matched_fraction = float(np.isfinite(dist).mean())
    mean_dist = float(np.nanmean(dist)) if np.isfinite(dist).any() else math.inf
    p90_dist = float(np.nanquantile(dist, 0.90)) if np.isfinite(dist).any() else math.inf
    gap_count = int(local_gaps.sum())
    transitions = max(len(group) - 1, 1)
    confidence = matched_fraction * math.exp(-mean_dist / max(sigma * 2, 1)) * math.exp(-2.0 * gap_count / transitions)
    fallback = p90_dist > 50 or matched_fraction < 0.90
    fallback_reason = "low_point_quality" if fallback else ""
    group["geometric_link_id"] = group.link_id
    group["candidate_link_id"] = group.link_id
    group["matched_link_id"] = group.link_id
    group["proj_dist_m"] = group.gps_to_link_dist_m
    group["proj_x_hmm"] = group.snap_x
    group["proj_y_hmm"] = group.snap_y
    group["emission_score"] = np.exp(-0.5 * (group.proj_dist_m.fillna(9999).to_numpy(dtype=float) / sigma) ** 2)
    group["transition_score"] = 1.0
    group.loc[local_gaps, "transition_score"] = 0.25
    group["viterbi_score"] = np.nan
    group["transition_network_dist_m"] = np.nan
    group["path_gap_flag"] = local_gaps
    group["unreasonable_detour_flag"] = False
    group["topology_gap"] = local_gaps
    group["matcher_version"] = "local_topology_fmm"
    group["point_seq"] = np.arange(len(group), dtype="int32")
    group["matched_fraction"] = matched_fraction
    group["match_confidence"] = confidence
    group["fallback_used"] = fallback
    group["fallback_reason"] = fallback_reason
    lon, lat = network.to_wgs84.transform(group.proj_x_hmm.to_numpy(), group.proj_y_hmm.to_numpy())
    group["proj_lon"] = lon
    group["proj_lat"] = lat

    routes: list[dict] = []
    raw_gap_count = gap_count
    unresolved_gap_count = 0
    changed = group.link_id.ne(group.link_id.shift())
    states = group.loc[changed, ["order_id", "link_id", "point_seq", "timestamp", "segment_distance_m"]].copy()
    if not states.empty:
        first = states.iloc[0]
        routes.append({
            "order_id": first.order_id, "link_id": first.link_id,
            "source_point_seq": int(first.point_seq), "timestamp": int(first.timestamp),
            "is_interpolated": False, "transition_path_status": "start",
        })
        previous = first
        for _, current in states.iloc[1:].iterrows():
            a = int(network.link_lookup[str(previous.link_id)])
            b = int(network.link_lookup[str(current.link_id)])
            current_point_seq = int(current.point_seq)
            is_raw_gap = bool(local_gaps[current_point_seq]) if 0 <= current_point_seq < len(local_gaps) else False
            path, status, routed = network.link_path_indices(a, b)
            gps_distance = float(current.segment_distance_m) if math.isfinite(float(current.segment_distance_m)) else 0.0
            if math.isfinite(routed):
                group.loc[group.point_seq.eq(current_point_seq), "transition_network_dist_m"] = routed
            if is_raw_gap:
                detour_limit = max(1000.0, gps_distance * raw_gap_repair_detour_ratio + raw_gap_repair_extra_m)
            else:
                detour_limit = max(300.0, gps_distance * max_repair_detour_ratio + max_repair_extra_m)
            if status.startswith("fmm") and (not math.isfinite(routed) or routed > detour_limit):
                path = (a, b)
                status = "fmm_repair_rejected" if is_raw_gap else "local_jump_not_repaired"
                group.loc[group.point_seq.eq(current_point_seq), "unreasonable_detour_flag"] = is_raw_gap
                if is_raw_gap:
                    unresolved_gap_count += 1
            link_ids = network.roads.link_id.to_numpy()[list(path)]
            for link_id in link_ids[1:-1]:
                routes.append({
                    "order_id": current.order_id, "link_id": link_id,
                    "source_point_seq": int(current.point_seq), "timestamp": int(current.timestamp),
                    "is_interpolated": True, "transition_path_status": status,
                })
            routes.append({
                "order_id": current.order_id, "link_id": current.link_id,
                "source_point_seq": int(current.point_seq), "timestamp": int(current.timestamp),
                "is_interpolated": False, "transition_path_status": status,
            })
            previous = current
    for sequence, record in enumerate(routes):
        record["route_sequence"] = sequence
    matched_step = pd.Series(np.hypot(group.proj_x_hmm.diff(), group.proj_y_hmm.diff())).fillna(0)
    gps_length = float(group.segment_distance_m.fillna(0).sum())
    matched_length = float(matched_step.sum())
    quality = {
        "order_id": str(group.order_id.iloc[0]),
        "matched_fraction": matched_fraction,
        "match_confidence": confidence,
        "fallback_used": fallback,
        "fallback_reason": fallback_reason,
        "p50_match_dist": float(np.nanquantile(dist, 0.50)) if np.isfinite(dist).any() else math.inf,
        "p90_match_dist": p90_dist,
        "p95_match_dist": float(np.nanquantile(dist, 0.95)) if np.isfinite(dist).any() else math.inf,
        "raw_topology_gap_count": raw_gap_count,
        "topology_gap_count": unresolved_gap_count,
        "path_gap_count": raw_gap_count,
        "unreasonable_detour_count": int(group.unreasonable_detour_flag.sum()),
        "gps_length_m": gps_length,
        "matched_route_length_m": matched_length,
        "route_length_ratio": matched_length / gps_length if gps_length > 0 else np.nan,
        "link_sequence_change_ratio": 0.0,
        "fmm_repair_count": sum(1 for record in routes if record.get("is_interpolated")),
    }
    return group, quality, routes


def calibrate_sigma(frame: pd.DataFrame) -> float:
    distances = frame.gps_to_link_dist_m.replace([np.inf, -np.inf], np.nan).dropna()
    p90 = float(distances.quantile(0.9)) if len(distances) else 15.0
    return float(np.clip(p90 / 1.645, 8.0, 30.0))


def main() -> None:
    args = parse_args()
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker-index must satisfy 0 <= index < worker-count")
    network = LocalRoadNetwork(args.roads, args.nodes)
    all_files = sorted(args.matched_dir.glob("*.parquet"))
    files = [path for idx, path in enumerate(all_files) if idx % args.worker_count == args.worker_index]
    files = files[: args.limit_parts] if args.limit_parts else files
    point_dir = args.output_root / "fast_matched_points" / f"day={args.date}"
    route_dir = args.output_root / "fast_route_parts" / f"day={args.date}"
    quality_dir = args.output_root / "fast_quality_parts" / f"day={args.date}"
    for path in [point_dir, route_dir, quality_dir]:
        path.mkdir(parents=True, exist_ok=True)
    started = time.time()
    processed_orders = 0
    for source in files:
        part = source.stem.split("=")[-1].split("_")[-1]
        point_path = point_dir / f"bucket={part}.parquet"
        route_path = route_dir / f"part={part}.parquet"
        quality_path = quality_dir / f"part={part}.parquet"
        if point_path.exists() and route_path.exists() and quality_path.exists() and not args.force:
            continue
        frame = pd.read_parquet(source)
        sigma = calibrate_sigma(frame)
        matched_frames: list[pd.DataFrame] = []
        quality_rows: list[dict] = []
        route_rows: list[dict] = []
        for order_no, (_, group) in enumerate(frame.groupby("order_id", sort=False), start=1):
            if args.max_orders_per_part and order_no > args.max_orders_per_part:
                break
            matched, quality, routes = enrich_points(
                group, network, sigma, args.max_repair_detour_ratio, args.max_repair_extra_m,
                args.raw_gap_repair_detour_ratio, args.raw_gap_repair_extra_m,
            )
            matched_frames.append(matched)
            quality_rows.append(quality)
            route_rows.extend(routes)
            if order_no % 500 == 0:
                print(f"part={part} orders={order_no:,} elapsed={time.time()-started:.1f}s", flush=True)
        result = pd.concat(matched_frames, ignore_index=True)
        quality = pd.DataFrame(quality_rows)
        routes = pd.DataFrame(route_rows)
        result.to_parquet(point_path, index=False, compression="zstd")
        routes.to_parquet(route_path, index=False, compression="zstd")
        quality.to_parquet(quality_path, index=False, compression="zstd")
        processed_orders += len(quality)
        print(
            f"completed part={part} orders={len(quality):,} fallback={quality.fallback_used.mean():.2%} "
            f"fmm_links={int(quality.fmm_repair_count.sum()):,}",
            flush=True,
        )
    manifest = {
        "date": args.date,
        "complete": len(list(point_dir.glob("bucket=*.parquet"))) == len(all_files),
        "processed_orders_this_run": processed_orders,
        "matcher_version": "local_topology_fmm",
        "worker_count": args.worker_count,
        "worker_index": args.worker_index,
        "seconds": time.time() - started,
    }
    manifest_dir = args.output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"day={args.date}.fast.worker={args.worker_index}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
