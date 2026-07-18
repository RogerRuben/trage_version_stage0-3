"""Fast edge candidates, ambiguity windows, and selective HMM matching."""

from __future__ import annotations

import math
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra
from shapely import get_x, get_y, line_interpolate_point, line_locate_point, points

from .coordinates import gcj02_to_wgs84


def angular_difference(left: np.ndarray | float, right: np.ndarray | float) -> np.ndarray:
    return np.abs((np.asarray(left) - np.asarray(right) + 180.0) % 360.0 - 180.0)


def line_heading(geometry: Any, position: float, delta: float = 3.0) -> float:
    before = max(0.0, position - delta)
    after = min(float(geometry.length), position + delta)
    if after <= before:
        return 0.0
    a = geometry.interpolate(before)
    b = geometry.interpolate(after)
    return math.degrees(math.atan2(b.y - a.y, b.x - a.x))


def local_hmm_windows(
    ambiguous: Iterable[bool], context_points: int = 2, merge_gap_points: int = 2
) -> list[tuple[int, int]]:
    flags = np.asarray(list(ambiguous), dtype=bool)
    indices = np.flatnonzero(flags)
    if not len(indices):
        return []
    segments: list[list[int]] = [[int(indices[0]), int(indices[0])]]
    for value in indices[1:]:
        if int(value) - segments[-1][1] <= merge_gap_points + 1:
            segments[-1][1] = int(value)
        else:
            segments.append([int(value), int(value)])
    return [(max(0, left - context_points), min(len(flags), right + context_points + 1)) for left, right in segments]


@dataclass(frozen=True)
class Candidate:
    edge_uid: str
    edge_index: int
    position: float
    distance: float
    heading_difference: float
    score: float
    rank: int


class BoundedSourceCache:
    def __init__(self, graph: nx.DiGraph, max_sources: int, cutoff: float):
        self.graph = graph
        self.max_sources = int(max_sources)
        self.cutoff = float(cutoff)
        self._rows: OrderedDict[tuple[int, int], dict[int, float]] = OrderedDict()
        self._lock = threading.RLock()
        self.nodes = np.asarray(sorted(int(value) for value in graph.nodes), dtype="int64")
        self.node_index = {int(value): index for index, value in enumerate(self.nodes)}
        sparse = (
            nx.to_scipy_sparse_array(graph, nodelist=self.nodes.tolist(), weight="weight", dtype=float, format="csr")
            if len(self.nodes) else csr_matrix((0, 0), dtype=float)
        )
        self.matrix = csr_matrix(
            (sparse.data, sparse.indices.astype("int32"), sparse.indptr.astype("int32")),
            shape=sparse.shape,
        )

    def distances(self, source: int, requested_cutoff: float | None = None) -> dict[int, float]:
        source = int(source)
        cutoff = self.cutoff if requested_cutoff is None else min(self.cutoff, max(250.0, math.ceil(float(requested_cutoff) / 250.0) * 250.0))
        cache_key = (source, int(cutoff))
        with self._lock:
            if cache_key in self._rows:
                self._rows.move_to_end(cache_key)
                return self._rows[cache_key]
        source_index = self.node_index.get(source)
        if source_index is None:
            values: dict[int, float] = {}
        else:
            distances = scipy_dijkstra(
                self.matrix, directed=True, indices=source_index,
                limit=cutoff, return_predecessors=False,
            )
            finite = np.flatnonzero(np.isfinite(distances))
            values = {int(self.nodes[index]): float(distances[index]) for index in finite}
        with self._lock:
            existing = self._rows.get(cache_key)
            if existing is not None:
                self._rows.move_to_end(cache_key)
                return existing
            self._rows[cache_key] = values
            if len(self._rows) > self.max_sources:
                self._rows.popitem(last=False)
            return self._rows[cache_key]

    def preload(self, sources: Iterable[int], requested_cutoff: float, batch_size: int = 16) -> None:
        """Batch bounded Dijkstra sources for one HMM window into the same LRU."""
        cutoff = min(self.cutoff, max(250.0, math.ceil(float(requested_cutoff) / 250.0) * 250.0))
        unique = sorted({int(source) for source in sources if int(source) in self.node_index})
        with self._lock:
            missing = [source for source in unique if (source, int(cutoff)) not in self._rows]
        for offset in range(0, len(missing), int(batch_size)):
            source_nodes = missing[offset : offset + int(batch_size)]
            source_indices = [self.node_index[source] for source in source_nodes]
            matrix = scipy_dijkstra(
                self.matrix, directed=True, indices=source_indices,
                limit=cutoff, return_predecessors=False,
            )
            if matrix.ndim == 1:
                matrix = matrix[np.newaxis, :]
            with self._lock:
                for source, distances in zip(source_nodes, matrix):
                    finite = np.flatnonzero(np.isfinite(distances))
                    self._rows[(source, int(cutoff))] = {
                        int(self.nodes[index]): float(distances[index]) for index in finite
                    }
                    self._rows.move_to_end((source, int(cutoff)))
                    while len(self._rows) > self.max_sources:
                        self._rows.popitem(last=False)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._rows)


class TransitionEngine:
    def __init__(self, edges: gpd.GeoDataFrame, movements: pd.DataFrame, metric: pd.DataFrame, config: dict[str, Any]):
        self.edges = edges.reset_index(drop=True)
        self.lookup = {str(value): index for index, value in enumerate(self.edges.edge_uid)}
        self.lengths = self.edges.length_m.to_numpy(float)
        self.from_nodes = self.edges.from_node.to_numpy("int64")
        self.to_nodes = self.edges.to_node.to_numpy("int64")
        graph = nx.DiGraph()
        for row in metric.itertuples():
            graph.add_edge(int(row.from_node), int(row.to_node), weight=float(row.routing_cost_m))
        self.cache = BoundedSourceCache(graph, int(config["route_cache_sources"]), float(config["max_route_distance_m"]))
        self.allowed_movements = {
            (str(row.from_edge_uid), str(row.to_edge_uid))
            for row in movements.itertuples()
            if bool(row.layer_compatibility) and not str(row.restriction_status).startswith("forbidden")
        }

    def distance(self, left: Candidate, right: Candidate, requested_cutoff: float | None = None) -> float:
        if left.edge_uid == right.edge_uid:
            return right.position - left.position if right.position >= left.position else math.inf
        if (left.edge_uid, right.edge_uid) in self.allowed_movements:
            return max(0.0, self.lengths[left.edge_index] - left.position) + right.position
        middle = self.cache.distances(int(self.to_nodes[left.edge_index]), requested_cutoff).get(int(self.from_nodes[right.edge_index]), math.inf)
        if not math.isfinite(middle):
            return math.inf
        return max(0.0, self.lengths[left.edge_index] - left.position) + middle + right.position

    def preload_window(self, candidate_rows: list[list[Candidate]], cutoff: float) -> None:
        sources = (
            int(self.to_nodes[candidate.edge_index])
            for row in candidate_rows[:-1]
            for candidate in row
        )
        self.cache.preload(sources, cutoff)


class CandidateIndex:
    def __init__(self, edges: gpd.GeoDataFrame, config: dict[str, Any], cache_dir: str | None = None):
        self.edges = edges.reset_index(drop=True)
        self.config = config
        self.geometries = self.edges.geometry.to_numpy()
        self.edge_uids = self.edges.edge_uid.astype(str).to_numpy()
        self.lengths = self.edges.length_m.to_numpy(float)
        self.parallel_groups = self.edges.parallel_group.to_numpy()
        self.bridges = self.edges.bridge.to_numpy(bool)
        self.tunnels = self.edges.tunnel.to_numpy(bool)
        self.highways = self.edges.highway.astype(str).to_numpy()
        self.candidate_penalties = self.edges.candidate_penalty.to_numpy(float)
        cache = None if cache_dir is None else Path(cache_dir)
        metadata = cache / "metadata.json" if cache else None
        if metadata and metadata.exists() and (cache / "sample_xy.npy").exists() and (cache / "sample_edges.npy").exists():
            stored = json.loads(metadata.read_text(encoding="utf-8"))
            if stored.get("edge_count") == len(self.edges) and stored.get("config") == config:
                self.sample_xy = np.load(cache / "sample_xy.npy", mmap_mode="r")
                self.sample_edges = np.load(cache / "sample_edges.npy", mmap_mode="r")
                self.tree = cKDTree(self.sample_xy)
                return
        sample_xy: list[np.ndarray] = []
        sample_edges: list[np.ndarray] = []
        for index, row in self.edges.iterrows():
            geometry = self.geometries[index]
            coords = np.asarray(geometry.coords)
            if len(coords) >= 3:
                vectors = np.diff(coords, axis=0)
                headings = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
                curvature = float(np.max(angular_difference(headings[1:], headings[:-1]))) if len(headings) > 1 else 0.0
            else:
                curvature = 0.0
            complex_edge = bool(pd.notna(self.parallel_groups[index])) or self.bridges[index] or self.tunnels[index] or self.highways[index].endswith("_link")
            if complex_edge:
                spacing = float(config["spacing_complex_m"])
            elif curvature >= 25:
                spacing = float(config["spacing_curve_m"])
            elif self.lengths[index] >= 250 and curvature < 5:
                spacing = float(config["spacing_straight_m"])
            else:
                spacing = float(config["spacing_urban_m"])
            distances = np.arange(0.0, max(self.lengths[index], 0.01) + spacing * 0.5, spacing)
            sampled = line_interpolate_point(geometry, distances)
            sample_xy.append(np.column_stack([get_x(sampled), get_y(sampled)]))
            sample_edges.append(np.full(len(distances), index, dtype="int32"))
        self.sample_xy = np.vstack(sample_xy)
        self.sample_edges = np.concatenate(sample_edges)
        if cache:
            cache.mkdir(parents=True, exist_ok=True)
            np.save(cache / "sample_xy.npy", self.sample_xy)
            np.save(cache / "sample_edges.npy", self.sample_edges)
            temporary = cache / "metadata.json.tmp"
            temporary.write_text(json.dumps({"edge_count": len(self.edges), "config": config}, sort_keys=True), encoding="utf-8")
            temporary.replace(cache / "metadata.json")
        self.tree = cKDTree(self.sample_xy)

    def candidates(self, x: float, y: float, gps_heading: float) -> list[Candidate]:
        maximum = int(self.config["max_candidates"])
        query_k = min(len(self.sample_xy), maximum * 6)
        _, sampled_indices = self.tree.query([x, y], k=query_k, workers=1)
        unique: list[int] = []
        for sample_index in np.atleast_1d(sampled_indices):
            edge_index = int(self.sample_edges[int(sample_index)])
            if edge_index not in unique:
                unique.append(edge_index)
            if len(unique) >= maximum:
                break
        point = points([x], [y])[0]
        raw: list[Candidate] = []
        for edge_index in unique:
            geometry = self.geometries[edge_index]
            position = float(line_locate_point(geometry, point))
            projected = line_interpolate_point(geometry, position)
            distance = float(point.distance(projected))
            if distance > float(self.config["radius_m"]):
                continue
            heading_delta = float(angular_difference(gps_heading, line_heading(geometry, position))) if math.isfinite(gps_heading) else 90.0
            score = distance + heading_delta / 180.0 * float(self.config["heading_weight_m"]) + self.candidate_penalties[edge_index]
            raw.append(Candidate(self.edge_uids[edge_index], edge_index, position, distance, heading_delta, score, 0))
        raw.sort(key=lambda value: (value.score, value.edge_uid))
        if not raw:
            return []
        first_index = raw[0].edge_index
        complex_area = bool(pd.notna(self.parallel_groups[first_index])) or self.bridges[first_index] or self.tunnels[first_index] or self.highways[first_index].endswith("_link")
        dense = len(raw) >= 5 and raw[min(4, len(raw)-1)].distance <= 30.0
        retained = int(self.config["complex_candidates"] if complex_area else self.config["dense_candidates"] if dense else 3)
        return [Candidate(**{**candidate.__dict__, "rank": rank + 1}) for rank, candidate in enumerate(raw[:retained])]


def _gps_headings(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.gradient(x)
    dy = np.gradient(y)
    heading = np.degrees(np.arctan2(dy, dx))
    heading[np.hypot(dx, dy) < 0.5] = np.nan
    return heading


def _viterbi(candidate_rows: list[list[Candidate]], observed_step: np.ndarray, engine: TransitionEngine, config: dict[str, Any]) -> tuple[list[Candidate], float] | None:
    if any(not row for row in candidate_rows):
        return None
    sigma = float(config["sigma_distance_m"])
    beta = float(config["beta_transition_m"])
    maximum_dynamic_cutoff = min(
        float(config["max_route_distance_m"]),
        max(300.0, float(np.nanmax(observed_step)) * 3.0 + 200.0),
    )
    engine.preload_window(candidate_rows, maximum_dynamic_cutoff)
    scores = np.array([-candidate.score**2 / (2 * sigma**2) for candidate in candidate_rows[0]], dtype=float)
    back: list[np.ndarray] = []
    for point_index in range(1, len(candidate_rows)):
        current = candidate_rows[point_index]
        previous = candidate_rows[point_index - 1]
        next_scores = np.full(len(current), -np.inf)
        pointers = np.full(len(current), -1, dtype="int32")
        for j, right in enumerate(current):
            emission = -right.score**2 / (2 * sigma**2)
            for i, left in enumerate(previous):
                dynamic_cutoff = min(
                    float(config["max_route_distance_m"]),
                    max(300.0, float(observed_step[point_index]) * 3.0 + 200.0),
                )
                network_distance = engine.distance(left, right, dynamic_cutoff)
                if not math.isfinite(network_distance):
                    continue
                transition = -abs(network_distance - float(observed_step[point_index])) / beta
                value = scores[i] + emission + transition
                if value > next_scores[j]:
                    next_scores[j] = value
                    pointers[j] = i
        if not np.isfinite(next_scores).any():
            return None
        scores = next_scores
        back.append(pointers)
    end = int(np.nanargmax(scores))
    path_indices = [end]
    for pointers in reversed(back):
        end = int(pointers[end])
        if end < 0:
            return None
        path_indices.append(end)
    path_indices.reverse()
    return [candidate_rows[index][choice] for index, choice in enumerate(path_indices)], float(np.max(scores))


def _ambiguity_flags(
    candidate_rows: list[list[Candidate]],
    edges: gpd.GeoDataFrame,
    distance_margin_m: float,
) -> tuple[np.ndarray, list[str]]:
    flags = np.zeros(len(candidate_rows), dtype=bool)
    reasons: list[str] = ["" for _ in candidate_rows]
    for index, candidates in enumerate(candidate_rows):
        found: list[str] = []
        if not candidates:
            found.append("no_candidate")
        elif len(candidates) > 1:
            a, b = candidates[:2]
            left, right = edges.iloc[a.edge_index], edges.iloc[b.edge_index]
            close_candidates = b.score - a.score <= float(distance_margin_m)
            if close_candidates:
                found.append("small_emission_margin")
            if pd.notna(left.parallel_group) and left.parallel_group == right.parallel_group:
                found.append("parallel_group")
            if (left.layer, left.bridge, left.tunnel) != (right.layer, right.bridge, right.tunnel):
                found.append("grade_separation")
            if close_candidates and {str(left.highway), str(right.highway)} & {"service", "motorway_link", "trunk_link", "primary_link"}:
                found.append("main_auxiliary")
            if close_candidates and abs(a.heading_difference - b.heading_difference) <= 10:
                found.append("heading_tie")
        flags[index] = bool(found)
        reasons[index] = "|".join(found)
    return flags, reasons


def match_order(
    frame: pd.DataFrame,
    edges: gpd.GeoDataFrame,
    index: CandidateIndex,
    engine: TransitionEngine,
    candidate_config: dict[str, Any],
    hmm_config: dict[str, Any],
    metric_crs: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = frame.sort_values("timestamp", kind="stable").drop_duplicates("timestamp").copy()
    lon = pd.to_numeric(frame.lon, errors="coerce").to_numpy(float)
    lat = pd.to_numeric(frame.lat, errors="coerce").to_numpy(float)
    valid = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lon) <= 180) & (np.abs(lat) <= 90)
    frame = frame.loc[valid].copy()
    lon, lat = lon[valid], lat[valid]
    if candidate_config.get("trajectory_coordinate_interpretation") == "gcj02":
        lon, lat = gcj02_to_wgs84(lon, lat)
    transformer = Transformer.from_crs(4326, metric_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    headings = _gps_headings(np.asarray(x), np.asarray(y))
    candidate_rows = [index.candidates(float(px), float(py), float(heading)) for px, py, heading in zip(x, y, headings)]
    flags, ambiguity_reasons = _ambiguity_flags(
        candidate_rows, edges, float(candidate_config["ambiguity_distance_margin_m"])
    )
    geometric = [row[0] if row else None for row in candidate_rows]
    selected = list(geometric)
    observed_step = np.r_[0.0, np.hypot(np.diff(x), np.diff(y))]
    windows = local_hmm_windows(flags, int(hmm_config["local_context_points"]), int(hmm_config["merge_gap_points"]))
    # Disjoint local windows remain local. Full-order HMM is reserved for pervasive
    # ambiguity or a merged ambiguity window spanning almost the whole order.
    dominant_window = bool(windows) and max(stop - start for start, stop in windows) / max(len(flags), 1) >= 0.8
    full_order = flags.mean() >= float(hmm_config["full_order_ambiguity_share"]) or dominant_window
    hmm_score = math.nan
    fallback_reason = ""
    if full_order:
        solved = _viterbi(candidate_rows, observed_step, engine, hmm_config)
        if solved:
            selected, hmm_score = solved
            mode = "full_order_hmm"
        else:
            mode = "geometric_fallback" if hmm_config.get("geometric_fallback_enabled") else "rejected"
            fallback_reason = "full_order_hmm_no_continuous_path"
    elif windows:
        mode = "local_hmm"
        local_failed = False
        local_scores: list[float] = []
        for start, stop in windows:
            solved = _viterbi(candidate_rows[start:stop], observed_step[start:stop], engine, hmm_config)
            if not solved:
                local_failed = True
                break
            local_path, local_score = solved
            selected[start:stop] = local_path
            local_scores.append(local_score)
        if local_failed:
            solved = _viterbi(candidate_rows, observed_step, engine, hmm_config)
            if solved:
                selected, hmm_score = solved
                mode = "full_order_hmm"
            else:
                mode = "geometric_fallback" if hmm_config.get("geometric_fallback_enabled") else "rejected"
                fallback_reason = "local_and_full_hmm_no_continuous_path"
        else:
            hmm_score = float(sum(local_scores))
    else:
        mode = "fast_deterministic"
    if any(value is None for value in selected):
        mode = "rejected"
        fallback_reason = "one_or_more_points_without_candidate"
    frame["point_seq"] = np.arange(len(frame), dtype="int32")
    frame["source_lon"], frame["source_lat"] = frame.lon, frame.lat
    frame["matching_lon"], frame["matching_lat"] = lon, lat
    frame["metric_x"], frame["metric_y"] = x, y
    frame["candidate_count"] = [len(row) for row in candidate_rows]
    frame["parallel_ambiguity"] = ["parallel_group" in reason for reason in ambiguity_reasons]
    frame["ambiguity_reason"] = ambiguity_reasons
    frame["matching_mode"] = mode
    if mode != "rejected":
        frame["edge_uid"] = [value.edge_uid for value in selected]
        frame["position_on_edge"] = [value.position for value in selected]
        frame["gps_to_edge_distance_m"] = [value.distance for value in selected]
        frame["candidate_rank"] = [value.rank for value in selected]
        frame["emission_margin"] = [row[1].score-row[0].score if len(row)>1 else math.inf for row in candidate_rows]
    else:
        for column in ["edge_uid", "position_on_edge", "gps_to_edge_distance_m", "candidate_rank", "emission_margin"]:
            frame[column] = pd.NA
    frame["viterbi_margin"] = math.nan
    frame["point_quality"] = np.where(flags, "ambiguous", "high_confidence")
    summary = {
        "order_id": str(frame.order_id.iloc[0]) if len(frame) else "",
        "matching_mode": mode, "fallback_used": mode == "geometric_fallback",
        "fallback_reason": fallback_reason, "point_count": len(frame),
        "ambiguity_point_share": float(flags.mean()) if len(flags) else 1.0,
        "parallel_ambiguity_share": float(frame.parallel_ambiguity.mean()) if len(frame) else 0.0,
        "local_window_count": len(windows), "hmm_score": hmm_score,
    }
    return frame, summary
