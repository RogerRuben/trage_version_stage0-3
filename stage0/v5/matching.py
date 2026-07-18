"""Fast edge candidates, ambiguity windows, and selective HMM matching."""

from __future__ import annotations

import math
import json
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import distance as shapely_distance
from shapely import get_x, get_y, line_interpolate_point, line_locate_point, points

from .coordinates import gcj02_to_wgs84
from .routing import CompactMovementRouter, SearchStats


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


class TransitionEngine:
    def __init__(
        self,
        edges: gpd.GeoDataFrame,
        movements: pd.DataFrame,
        metric: pd.DataFrame,
        config: dict[str, Any],
        movement_router: CompactMovementRouter | None = None,
    ):
        self.edges = edges.reset_index(drop=True)
        self.lookup = {str(value): index for index, value in enumerate(self.edges.edge_uid)}
        self.lengths = self.edges.length_m.to_numpy(float)
        self.from_nodes = self.edges.from_node.to_numpy("int64")
        self.to_nodes = self.edges.to_node.to_numpy("int64")
        self.router = movement_router or CompactMovementRouter(edges, movements, config)
        self.jitter_tolerance = float(config.get("same_edge_jitter_tolerance_m", 0.0))
        self.jitter_penalty_per_m = float(config.get("same_edge_jitter_penalty_per_m", 0.25))

    def distance(self, left: Candidate, right: Candidate, requested_cutoff: float | None = None) -> float:
        if left.edge_uid == right.edge_uid:
            delta = right.position - left.position
            return max(0.0, delta) if delta >= -self.jitter_tolerance else math.inf
        if self.router.movement(left.edge_uid, right.edge_uid) is not None:
            return max(0.0, self.lengths[left.edge_index] - left.position) + right.position
        middle = self.router.multi_target_distances(
            [left.edge_uid], [right.edge_uid],
            float(requested_cutoff or self.router.maximum),
        ).get((left.edge_uid, right.edge_uid), math.inf)
        if not math.isfinite(middle):
            return math.inf
        return max(0.0, self.lengths[left.edge_index] - left.position) + middle + right.position

    def transition_matrices(
        self,
        previous: list[Candidate],
        current: list[Candidate],
        requested_cutoff: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        distances = np.full((len(previous), len(current)), math.inf, dtype=float)
        penalties = np.zeros_like(distances)
        unresolved_left: list[str] = []
        unresolved_right: list[str] = []
        for left in previous:
            for right in current:
                if left.edge_uid != right.edge_uid and self.router.movement(left.edge_uid, right.edge_uid) is None:
                    unresolved_left.append(left.edge_uid)
                    unresolved_right.append(right.edge_uid)
        middle_distances = self.router.multi_target_distances(
            unresolved_left, unresolved_right, requested_cutoff,
        )
        for i, left in enumerate(previous):
            for j, right in enumerate(current):
                if left.edge_uid == right.edge_uid:
                    delta = right.position - left.position
                    if delta >= 0:
                        distances[i, j] = delta
                    elif delta >= -self.jitter_tolerance:
                        distances[i, j] = 0.0
                        penalties[i, j] = abs(delta) * self.jitter_penalty_per_m
                    continue
                direct = self.router.movement(left.edge_uid, right.edge_uid)
                middle = 0.0 if direct is not None else middle_distances.get((left.edge_uid, right.edge_uid), math.inf)
                if not math.isfinite(middle):
                    continue
                distances[i, j] = max(0.0, self.lengths[left.edge_index] - left.position) + middle + right.position
        return distances, penalties

    def selected_bridge_paths(
        self, selected: list[Candidate | None], cutoff: float,
    ) -> tuple[dict[tuple[str, str], list[str]], int]:
        paths: dict[tuple[str, str], list[str]] = {}
        requested = 0
        for left, right in zip(selected, selected[1:]):
            if left is None or right is None or left.edge_uid == right.edge_uid:
                continue
            if self.router.movement(left.edge_uid, right.edge_uid) is not None:
                continue
            key = (left.edge_uid, right.edge_uid)
            if key in paths:
                continue
            requested += 1
            result = self.router.bridge_path(*key, cutoff=cutoff)
            if result is not None:
                paths[key] = result[0]
        return paths, requested

    def stats(self) -> SearchStats:
        return self.router.stats()


class CandidateIndex:
    def __init__(
        self,
        edges: gpd.GeoDataFrame,
        config: dict[str, Any],
        cache_dir: str | None = None,
        metric_crs: str = "EPSG:32649",
    ):
        self.edges = edges.reset_index(drop=True)
        self.config = config
        self.transformer = Transformer.from_crs(4326, metric_crs, always_xy=True)
        self.geometries = self.edges.geometry.to_numpy()
        self.edge_uids = self.edges.edge_uid.astype(str).to_numpy()
        self.candidate_aliases = self.edges.get("candidate_alias_uid", self.edges.edge_uid).astype(str).to_numpy()
        self.alias_digest = hashlib.sha256("\n".join(self.candidate_aliases).encode("utf-8")).hexdigest()
        self.cache_hit = False
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
            if (
                stored.get("edge_count") == len(self.edges)
                and stored.get("config") == config
                and stored.get("candidate_alias_digest") == self.alias_digest
            ):
                self.sample_xy = np.load(cache / "sample_xy.npy", mmap_mode="r")
                self.sample_edges = np.load(cache / "sample_edges.npy", mmap_mode="r")
                self.tree = cKDTree(self.sample_xy)
                self.cache_hit = True
                return
        sample_xy: list[np.ndarray] = []
        sample_edges: list[np.ndarray] = []
        for index, row in self.edges.iterrows():
            if self.candidate_aliases[index] != self.edge_uids[index]:
                continue
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
            temporary.write_text(json.dumps({
                "edge_count": len(self.edges), "config": config,
                "candidate_alias_digest": self.alias_digest,
            }, sort_keys=True), encoding="utf-8")
            temporary.replace(cache / "metadata.json")
        self.tree = cKDTree(self.sample_xy)

    def transform(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = self.transformer.transform(lon, lat)
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def candidates_batch(self, x: np.ndarray, y: np.ndarray, gps_headings: np.ndarray) -> list[list[Candidate]]:
        """Batch radius recall and vectorized exact point-to-edge projection for one order."""
        if not len(x):
            return []
        radius = float(self.config["radius_m"])
        maximum_spacing = max(float(self.config[name]) for name in (
            "spacing_complex_m", "spacing_curve_m", "spacing_straight_m", "spacing_urban_m"
        ))
        sample_hits = self.tree.query_ball_point(
            np.column_stack([x, y]), r=radius + maximum_spacing, workers=1,
        )
        point_indices: list[int] = []
        edge_indices: list[int] = []
        for point_index, hits in enumerate(sample_hits):
            if not hits:
                continue
            unique_edges = np.unique(self.sample_edges[np.asarray(hits, dtype="int64")])
            point_indices.extend([point_index] * len(unique_edges))
            edge_indices.extend(map(int, unique_edges))
        if not edge_indices:
            return [[] for _ in x]
        point_index_array = np.asarray(point_indices, dtype="int32")
        edge_index_array = np.asarray(edge_indices, dtype="int32")
        point_geometries = points(x[point_index_array], y[point_index_array])
        edge_geometries = self.geometries[edge_index_array]
        positions = np.asarray(line_locate_point(edge_geometries, point_geometries), dtype=float)
        projected = line_interpolate_point(edge_geometries, positions)
        distances = np.asarray(shapely_distance(point_geometries, projected), dtype=float)
        before = line_interpolate_point(edge_geometries, np.maximum(0.0, positions - 3.0))
        after = line_interpolate_point(edge_geometries, np.minimum(self.lengths[edge_index_array], positions + 3.0))
        edge_headings = np.degrees(np.arctan2(get_y(after) - get_y(before), get_x(after) - get_x(before)))
        heading_delta = angular_difference(gps_headings[point_index_array], edge_headings)
        scores = (
            distances
            + heading_delta / 180.0 * float(self.config["heading_weight_m"])
            + self.candidate_penalties[edge_index_array]
        )
        rows: list[list[Candidate]] = [[] for _ in x]
        valid = np.flatnonzero(distances <= radius)
        valid_points = point_index_array[valid]
        for point_index in range(len(x)):
            left = int(np.searchsorted(valid_points, point_index, side="left"))
            right = int(np.searchsorted(valid_points, point_index, side="right"))
            pair_positions = valid[left:right]
            by_alias: dict[str, Candidate] = {}
            for pair_position in pair_positions:
                edge_index = int(edge_index_array[pair_position])
                candidate = Candidate(
                    str(self.edge_uids[edge_index]), edge_index, float(positions[pair_position]),
                    float(distances[pair_position]), float(heading_delta[pair_position]),
                    float(scores[pair_position]), 0,
                )
                alias = str(self.candidate_aliases[edge_index])
                previous = by_alias.get(alias)
                if previous is None or (candidate.score, candidate.edge_uid) < (previous.score, previous.edge_uid):
                    by_alias[alias] = candidate
            raw = sorted(by_alias.values(), key=lambda value: (value.score, value.edge_uid))
            if not raw:
                continue
            first_index = raw[0].edge_index
            complex_area = (
                bool(pd.notna(self.parallel_groups[first_index])) or self.bridges[first_index]
                or self.tunnels[first_index] or self.highways[first_index].endswith("_link")
            )
            dense = len(raw) >= 5 and raw[4].distance <= 30.0
            retained = int(self.config[
                "complex_candidates" if complex_area else "dense_candidates" if dense else "ordinary_candidates"
            ])
            retained = min(retained, int(self.config["max_candidates"]))
            rows[point_index] = [
                Candidate(**{**candidate.__dict__, "rank": rank + 1})
                for rank, candidate in enumerate(raw[:retained])
            ]
        return rows

    def candidates(self, x: float, y: float, gps_heading: float) -> list[Candidate]:
        return self.candidates_batch(
            np.asarray([x], dtype=float), np.asarray([y], dtype=float),
            np.asarray([gps_heading], dtype=float),
        )[0]


def _gps_headings(x: np.ndarray, y: np.ndarray, minimum_displacement_m: float) -> tuple[np.ndarray, np.ndarray]:
    dx = np.gradient(x)
    dy = np.gradient(y)
    heading = np.degrees(np.arctan2(dy, dx))
    reliable = np.hypot(dx, dy) >= float(minimum_displacement_m)
    reliable_indices = np.flatnonzero(reliable)
    if not len(reliable_indices):
        return np.zeros(len(x), dtype=float), reliable
    # Stationary points inherit the nearest reliable movement heading for candidate scoring but
    # remain excluded from the full-order ambiguity denominator and direction hard checks.
    missing = np.flatnonzero(~reliable)
    for index in missing:
        nearest = reliable_indices[np.argmin(np.abs(reliable_indices - index))]
        heading[index] = heading[nearest]
    return heading, reliable


def _viterbi(
    candidate_rows: list[list[Candidate]],
    observed_step: np.ndarray,
    engine: TransitionEngine,
    config: dict[str, Any],
) -> tuple[list[Candidate], float, float] | None:
    if any(not row for row in candidate_rows):
        return None
    sigma = float(config["sigma_distance_m"])
    beta = float(config["beta_transition_m"])
    scores = np.array([-candidate.score**2 / (2 * sigma**2) for candidate in candidate_rows[0]], dtype=float)
    back: list[np.ndarray] = []
    transition_search_ms = 0.0
    for point_index in range(1, len(candidate_rows)):
        current = candidate_rows[point_index]
        previous = candidate_rows[point_index - 1]
        next_scores = np.full(len(current), -np.inf)
        pointers = np.full(len(current), -1, dtype="int32")
        dynamic_cutoff = min(
            float(config["max_route_distance_m"]),
            max(300.0, float(observed_step[point_index]) * 3.0 + 200.0),
        )
        search_started = time.perf_counter()
        network_distances, jitter_penalties = engine.transition_matrices(previous, current, dynamic_cutoff)
        transition_search_ms += (time.perf_counter() - search_started) * 1000.0
        for j, right in enumerate(current):
            emission = -right.score**2 / (2 * sigma**2)
            for i, left in enumerate(previous):
                network_distance = float(network_distances[i, j])
                if not math.isfinite(network_distance):
                    continue
                transition = (
                    -abs(network_distance - float(observed_step[point_index])) / beta
                    - float(jitter_penalties[i, j])
                )
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
    return (
        [candidate_rows[index][choice] for index, choice in enumerate(path_indices)],
        float(np.max(scores)),
        transition_search_ms,
    )


def ambiguity_segment_count(flags: np.ndarray) -> int:
    padded = np.r_[False, flags.astype(bool), False]
    return int(np.sum(np.diff(padded.astype("int8")) == 1))


def full_order_decision(
    flags: np.ndarray,
    config: dict[str, Any],
    ambiguity_eligible: np.ndarray | None = None,
) -> tuple[bool, str]:
    if not len(flags):
        return True, "empty_order"
    eligible = np.ones(len(flags), dtype=bool) if ambiguity_eligible is None else np.asarray(ambiguity_eligible, dtype=bool)
    share = float(flags[eligible].mean()) if eligible.any() else 0.0
    if share >= float(config["full_order_ambiguity_share"]):
        return True, "raw_ambiguity_share"
    return False, ""


def local_failures_require_full_order(failed_windows: int, config: dict[str, Any]) -> bool:
    return int(failed_windows) >= int(config.get("full_order_min_windows", 4))


def _ambiguity_flags(
    candidate_rows: list[list[Candidate]],
    edges: gpd.GeoDataFrame,
    distance_margin_m: float,
    heading_reliable: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    flags = np.zeros(len(candidate_rows), dtype=bool)
    reasons: list[str] = ["" for _ in candidate_rows]
    for index, candidates in enumerate(candidate_rows):
        found: list[str] = []
        if not candidates:
            found.append("no_candidate")
        elif len(candidates) > 1 and (heading_reliable is None or bool(heading_reliable[index])):
            a, b = candidates[:2]
            left, right = edges.iloc[a.edge_index], edges.iloc[b.edge_index]
            close_candidates = b.score - a.score <= float(distance_margin_m)
            if close_candidates:
                found.append("small_emission_margin")
            spatially_plausible = max(a.distance, b.distance) <= 30.0
            headings_plausible = max(a.heading_difference, b.heading_difference) <= 75.0
            if close_candidates and spatially_plausible and headings_plausible and pd.notna(left.parallel_group) and left.parallel_group == right.parallel_group:
                found.append("parallel_group")
            if close_candidates and spatially_plausible and headings_plausible and (left.layer, left.bridge, left.tunnel) != (right.layer, right.bridge, right.tunnel):
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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    order_started = time.perf_counter()
    engine.router.begin_order()
    stats_before = engine.stats()
    frame = frame.sort_values("timestamp", kind="stable").drop_duplicates("timestamp").copy()
    lon = pd.to_numeric(frame.lon, errors="coerce").to_numpy(float)
    lat = pd.to_numeric(frame.lat, errors="coerce").to_numpy(float)
    valid = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lon) <= 180) & (np.abs(lat) <= 90)
    frame = frame.loc[valid].copy()
    lon, lat = lon[valid], lat[valid]
    transform_started = time.perf_counter()
    if candidate_config.get("trajectory_coordinate_interpretation") == "gcj02":
        lon, lat = gcj02_to_wgs84(lon, lat)
    x, y = index.transform(lon, lat)
    coordinate_transform_ms = (time.perf_counter() - transform_started) * 1000.0
    headings, heading_reliable = _gps_headings(
        x, y, float(candidate_config.get("minimum_heading_displacement_m", 3.0)),
    )
    candidate_started = time.perf_counter()
    candidate_rows = index.candidates_batch(x, y, headings)
    candidate_generation_ms = (time.perf_counter() - candidate_started) * 1000.0
    ambiguity_started = time.perf_counter()
    flags, ambiguity_reasons = _ambiguity_flags(
        candidate_rows, edges, float(candidate_config["ambiguity_distance_margin_m"]), heading_reliable,
    )
    ambiguity_detection_ms = (time.perf_counter() - ambiguity_started) * 1000.0
    geometric = [row[0] if row else None for row in candidate_rows]
    selected = list(geometric)
    observed_step = np.r_[0.0, np.hypot(np.diff(x), np.diff(y))]
    windows = local_hmm_windows(flags, int(hmm_config["local_context_points"]), int(hmm_config["merge_gap_points"]))
    no_candidate = np.asarray([not row for row in candidate_rows], dtype=bool)
    ambiguity_eligible = heading_reliable | no_candidate
    full_order, full_order_trigger_reason = full_order_decision(flags, hmm_config, ambiguity_eligible)
    hmm_score = math.nan
    fallback_reason = ""
    local_hmm_ms = 0.0
    full_hmm_ms = 0.0
    transition_search_ms = 0.0
    local_failed_windows = 0
    local_hmm_attempted = bool(windows) and not full_order
    full_hmm_attempted = False
    full_hmm_succeeded = False
    if full_order:
        full_hmm_attempted = True
        full_started = time.perf_counter()
        solved = _viterbi(candidate_rows, observed_step, engine, hmm_config)
        full_hmm_ms += (time.perf_counter() - full_started) * 1000.0
        if solved:
            full_hmm_succeeded = True
            selected, hmm_score, searched_ms = solved
            transition_search_ms += searched_ms
            mode = "full_order_hmm"
        else:
            mode = "geometric_fallback" if hmm_config.get("geometric_fallback_enabled") else "rejected"
            fallback_reason = "full_order_hmm_no_continuous_path"
    elif windows:
        mode = "local_hmm"
        local_failed = False
        local_scores: list[float] = []
        for start, stop in windows:
            local_started = time.perf_counter()
            solved = _viterbi(candidate_rows[start:stop], observed_step[start:stop], engine, hmm_config)
            local_hmm_ms += (time.perf_counter() - local_started) * 1000.0
            if not solved:
                expanded_start = max(0, start - int(hmm_config.get("local_retry_context_points", 2)))
                expanded_stop = min(len(candidate_rows), stop + int(hmm_config.get("local_retry_context_points", 2)))
                retry_started = time.perf_counter()
                solved = _viterbi(
                    candidate_rows[expanded_start:expanded_stop],
                    observed_step[expanded_start:expanded_stop],
                    engine,
                    hmm_config,
                )
                local_hmm_ms += (time.perf_counter() - retry_started) * 1000.0
                if solved:
                    local_path, local_score, searched_ms = solved
                    selected[expanded_start:expanded_stop] = local_path
                    local_scores.append(local_score)
                    transition_search_ms += searched_ms
                    continue
                local_failed_windows += 1
                local_failed = True
                continue
            local_path, local_score, searched_ms = solved
            selected[start:stop] = local_path
            local_scores.append(local_score)
            transition_search_ms += searched_ms
        if local_failed:
            if local_failures_require_full_order(local_failed_windows, hmm_config):
                full_order_trigger_reason = "multiple_local_window_failures"
                full_hmm_attempted = True
                full_started = time.perf_counter()
                solved = _viterbi(candidate_rows, observed_step, engine, hmm_config)
                full_hmm_ms += (time.perf_counter() - full_started) * 1000.0
                if solved:
                    full_hmm_succeeded = True
                    selected, hmm_score, searched_ms = solved
                    transition_search_ms += searched_ms
                    mode = "full_order_hmm"
                else:
                    mode = "geometric_fallback" if hmm_config.get("geometric_fallback_enabled") else "rejected"
                    fallback_reason = "multiple_local_and_full_hmm_no_continuous_path"
            else:
                mode = "geometric_fallback" if hmm_config.get("geometric_fallback_enabled") else "rejected"
                fallback_reason = "local_window_no_continuous_path"
        else:
            hmm_score = float(sum(local_scores))
    else:
        mode = "fast_deterministic"
    if any(value is None for value in selected):
        mode = "rejected"
        fallback_reason = "one_or_more_points_without_candidate"
    selected_path_started = time.perf_counter()
    selected_bridge_paths, selected_bridge_request_count = engine.selected_bridge_paths(
        selected, float(hmm_config["max_route_distance_m"]),
    )
    selected_path_search_ms = (time.perf_counter() - selected_path_started) * 1000.0
    frame["point_seq"] = np.arange(len(frame), dtype="int32")
    frame["source_lon"], frame["source_lat"] = frame.lon, frame.lat
    frame["matching_lon"], frame["matching_lat"] = lon, lat
    frame["metric_x"], frame["metric_y"] = x, y
    frame["candidate_count"] = [len(row) for row in candidate_rows]
    frame["observed_step_m"] = observed_step
    frame["heading_reliable"] = heading_reliable
    frame["stationary_or_low_motion"] = ~heading_reliable
    frame["parallel_ambiguity"] = ["parallel_group" in reason for reason in ambiguity_reasons]
    frame["ambiguity_reason"] = ambiguity_reasons
    frame["matching_mode"] = mode
    if mode != "rejected":
        frame["edge_uid"] = [value.edge_uid for value in selected]
        frame["position_on_edge"] = [value.position for value in selected]
        frame["gps_to_edge_distance_m"] = [value.distance for value in selected]
        frame["candidate_rank"] = [value.rank for value in selected]
        frame["edge_heading_difference_deg"] = [value.heading_difference for value in selected]
        frame["emission_margin"] = [row[1].score-row[0].score if len(row)>1 else math.inf for row in candidate_rows]
    else:
        for column in ["edge_uid", "position_on_edge", "gps_to_edge_distance_m", "candidate_rank", "edge_heading_difference_deg", "emission_margin"]:
            frame[column] = pd.NA
    frame["viterbi_margin"] = math.nan
    frame["point_quality"] = np.where(flags, "ambiguous", "high_confidence")
    stats_after = engine.stats()
    stats_delta = stats_after.minus(stats_before)
    summary = {
        "order_id": str(frame.order_id.iloc[0]) if len(frame) else "",
        "matching_mode": mode, "fallback_used": mode == "geometric_fallback",
        "fallback_reason": fallback_reason, "point_count": len(frame),
        "ambiguity_point_share": float(flags.mean()) if len(flags) else 1.0,
        "eligible_ambiguity_point_share": float(flags[ambiguity_eligible].mean()) if ambiguity_eligible.any() else 0.0,
        "stationary_point_share": float((~heading_reliable).mean()) if len(heading_reliable) else 1.0,
        "parallel_ambiguity_share": float(frame.parallel_ambiguity.mean()) if len(frame) else 0.0,
        "local_window_count": len(windows), "hmm_score": hmm_score,
        "local_failed_window_count": local_failed_windows,
        "full_order_trigger_reason": full_order_trigger_reason,
        "local_hmm_attempted": local_hmm_attempted,
        "full_hmm_attempted": full_hmm_attempted,
        "full_hmm_succeeded": full_hmm_succeeded,
        "full_hmm_failed": full_hmm_attempted and not full_hmm_succeeded,
        "selected_bridge_request_count": selected_bridge_request_count,
        "selected_bridge_path_count": len(selected_bridge_paths),
        "selected_path_search_ms": selected_path_search_ms,
        "candidate_generation_ms": candidate_generation_ms,
        "coordinate_transform_ms": coordinate_transform_ms,
        "ambiguity_detection_ms": ambiguity_detection_ms,
        "local_hmm_ms": local_hmm_ms,
        "full_hmm_ms": full_hmm_ms,
        "transition_search_ms": transition_search_ms,
        "matching_total_ms": (time.perf_counter() - order_started) * 1000.0,
        "dijkstra_calls": stats_delta.calls,
        "dijkstra_expanded_nodes": stats_delta.expanded_nodes,
        "route_cache_hits": stats_delta.cache_hits,
        "route_cache_misses": stats_delta.cache_misses,
        "distance_search_calls": stats_delta.distance_calls,
        "path_search_calls": stats_delta.path_calls,
        "positive_cache_hits": stats_delta.positive_cache_hits,
        "negative_cache_hits": stats_delta.negative_cache_hits,
        "path_cache_hits": stats_delta.path_cache_hits,
        "_selected_bridge_paths": selected_bridge_paths,
    }
    return frame, summary
