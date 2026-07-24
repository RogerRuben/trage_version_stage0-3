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
from .routing import CompactMovementRouter, SearchStats, TransitionPath


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
    expanded = [
        [
            max(0, left - context_points),
            min(len(flags), right + context_points + 1),
        ]
        for left, right in segments
    ]
    merged: list[list[int]] = []
    for start, stop in expanded:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return [(start, stop) for start, stop in merged]


@dataclass(frozen=True)
class Candidate:
    edge_uid: str
    edge_index: int
    position: float
    distance: float
    heading_difference: float
    score: float
    rank: int
    heading_log_cost: float = 0.0
    edge_prior_log_cost: float = 0.0
    heading_used: bool = True
    projection_emission_cost: float = math.nan
    total_emission_cost: float = math.nan


def candidate_emission_cost(candidate: Candidate, sigma_distance_m: float) -> float:
    """Return the single emission model used by ranking and Viterbi."""
    if math.isfinite(candidate.total_emission_cost):
        return float(candidate.total_emission_cost)
    projection = candidate.distance**2 / (2.0 * sigma_distance_m**2)
    return float(
        projection
        + (candidate.heading_log_cost if candidate.heading_used else 0.0)
        + candidate.edge_prior_log_cost
    )


@dataclass(frozen=True)
class TransitionSelection:
    point_index: int
    from_edge_uid: str
    to_edge_uid: str
    observed_step_m: float
    time_gap_s: float
    transition_cutoff_m: float
    selected_path_distance_m: float
    generalized_routing_cost: float
    path_identifier: str
    edge_uids: tuple[str, ...]
    retry_used: bool = False
    initial_cutoff_m: float | None = None
    search_exact: bool = True
    jitter_penalty_m: float = 0.0


@dataclass(frozen=True)
class TransitionFailure:
    point_index: int
    from_edge_uid: str
    to_edge_uid: str
    observed_step_m: float
    transition_cutoff_m: float
    endpoint_distance_m: float
    reason: str
    raw_movement_status: str
    search_exact: bool = True
    diagnostic_class: str = "unclassified"


def transition_cutoff(observed_step_m: float, time_gap_s: float, config: dict[str, Any]) -> float:
    """Return the frozen per-observation physical search boundary."""
    spatial = (
        float(config.get("transition_cutoff_alpha", 3.0)) * max(0.0, observed_step_m)
        + float(config.get("transition_cutoff_base_m", 200.0))
    )
    # Time only expands the physical feasibility envelope; it never forces a
    # path. The speed cap is an engineering safety bound recorded in config.
    temporal = max(0.0, time_gap_s) * float(config.get("transition_max_speed_mps", 40.0))
    maximum = float(
        config.get(
            "transition_cutoff_max_m",
            config.get("max_route_distance_m", 6_000.0),
        )
    )
    return min(
        maximum,
        max(float(config.get("transition_cutoff_min_m", 300.0)), min(spatial, temporal) if temporal > 0 else spatial),
    )


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
        self._matrix_cache: dict[
            tuple[Any, ...], tuple[np.ndarray, np.ndarray, bool]
        ] = {}
        self._matrix_cache_hits = 0
        self._matrix_cache_misses = 0

    def begin_order(self) -> None:
        self.router.begin_order()
        self._matrix_cache.clear()
        self._matrix_cache_hits = 0
        self._matrix_cache_misses = 0

    @staticmethod
    def _candidate_signature(candidates: list[Candidate]) -> tuple[tuple[str, float], ...]:
        return tuple(
            (candidate.edge_uid, round(float(candidate.position), 6))
            for candidate in candidates
        )

    def evidence_cache_stats(self) -> tuple[int, int]:
        return self._matrix_cache_hits, self._matrix_cache_misses

    def distance(self, left: Candidate, right: Candidate, requested_cutoff: float | None = None) -> float:
        if left.edge_uid == right.edge_uid:
            delta = right.position - left.position
            return max(0.0, delta) if delta >= -self.jitter_tolerance else math.inf
        if self.router.movement(left.edge_uid, right.edge_uid) is not None:
            return max(0.0, self.lengths[left.edge_index] - left.position) + right.position
        endpoint = max(0.0, self.lengths[left.edge_index] - left.position) + right.position
        cutoff = float(requested_cutoff or self.router.maximum)
        transition = self.router.transition_path(
            left.edge_uid, right.edge_uid, max(0.0, cutoff - endpoint)
        )
        if transition is None:
            return math.inf
        return endpoint + transition.physical_distance_m

    def transition_matrices(
        self,
        previous: list[Candidate],
        current: list[Candidate],
        requested_cutoff: float,
        force_retry: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        cache_key = (
            self._candidate_signature(previous),
            self._candidate_signature(current),
            round(float(requested_cutoff), 6),
            bool(force_retry),
        )
        cached = self._matrix_cache.get(cache_key)
        if cached is not None:
            self._matrix_cache_hits += 1
            return cached[0].copy(), cached[1].copy(), cached[2]
        self._matrix_cache_misses += 1
        distances = np.full((len(previous), len(current)), math.inf, dtype=float)
        penalties = np.zeros_like(distances)
        retry_attempted = False
        for i, left in enumerate(previous):
            target_cutoffs: dict[str, float] = {}
            for j, right in enumerate(current):
                if left.edge_uid == right.edge_uid:
                    delta = right.position - left.position
                    if delta >= 0:
                        distances[i, j] = delta if delta <= requested_cutoff else math.inf
                    elif delta >= -self.jitter_tolerance:
                        distances[i, j] = 0.0
                        penalties[i, j] = abs(delta) * self.jitter_penalty_per_m
                    continue
                endpoint_distance = (
                    max(0.0, self.lengths[left.edge_index] - left.position)
                    + right.position
                )
                remaining_cutoff = requested_cutoff - endpoint_distance
                if remaining_cutoff < 0:
                    continue
                target_cutoffs[right.edge_uid] = max(
                    remaining_cutoff,
                    target_cutoffs.get(right.edge_uid, 0.0),
                )
            source_paths = self.router.transition_paths_from_source(
                left.edge_uid, target_cutoffs
            )
            for j, right in enumerate(current):
                if left.edge_uid == right.edge_uid:
                    continue
                endpoint_distance = (
                    max(0.0, self.lengths[left.edge_index] - left.position)
                    + right.position
                )
                remaining_cutoff = requested_cutoff - endpoint_distance
                if remaining_cutoff < 0:
                    continue
                transition = source_paths.get(right.edge_uid)
                if transition is None:
                    continue
                if transition.physical_distance_m > remaining_cutoff:
                    continue
                distances[i, j] = (
                    max(0.0, self.lengths[left.edge_index] - left.position)
                    + transition.physical_distance_m
                    + right.position
                )
                physical_full_edges = (
                    transition.physical_distance_m + self.lengths[right.edge_index]
                )
                penalties[i, j] += max(
                    0.0, transition.generalized_routing_cost - physical_full_edges
                )
        if force_retry or not np.isfinite(distances).any():
            retry_attempted = True
            retry_cutoff = min(
                self.router.maximum,
                float(self.router_config_value(
                    "transition_retry_max_m", 1200.0
                )),
                requested_cutoff
                * float(self.router_config_value("transition_retry_multiplier", 1.5)),
            )
            subset = int(self.router_config_value("transition_retry_candidate_subset", 3))
            for i, left in enumerate(previous[:subset]):
                target_cutoffs: dict[str, float] = {}
                for right in current[:subset]:
                    if left.edge_uid == right.edge_uid:
                        continue
                    endpoint = (
                        max(0.0, self.lengths[left.edge_index] - left.position)
                        + right.position
                    )
                    if endpoint <= retry_cutoff:
                        target_cutoffs[right.edge_uid] = retry_cutoff - endpoint
                physical_reachability = self.router.multi_target_distances(
                    [left.edge_uid], target_cutoffs, max(
                        target_cutoffs.values(), default=0.0
                    )
                )
                target_cutoffs = {
                    right_uid: cutoff
                    for right_uid, cutoff in target_cutoffs.items()
                    if math.isfinite(physical_reachability.get(
                        (left.edge_uid, right_uid), math.inf
                    ))
                    and physical_reachability[(left.edge_uid, right_uid)] <= cutoff
                }
                source_paths = self.router.transition_paths_from_source(
                    left.edge_uid, target_cutoffs
                )
                for j, right in enumerate(current[:subset]):
                    transition = source_paths.get(right.edge_uid)
                    if transition is None:
                        continue
                    endpoint = (
                        max(0.0, self.lengths[left.edge_index] - left.position)
                        + right.position
                    )
                    distance = endpoint + transition.physical_distance_m
                    if distance > retry_cutoff:
                        continue
                    distances[i, j] = distance
                    physical_full_edges = (
                        transition.physical_distance_m + self.lengths[right.edge_index]
                    )
                    penalties[i, j] = (
                        max(
                            0.0,
                            transition.generalized_routing_cost - physical_full_edges,
                        )
                        + float(self.router_config_value(
                            "transition_retry_soft_penalty_m", 180.0
                        ))
                    )
        self._matrix_cache[cache_key] = (
            distances.copy(),
            penalties.copy(),
            retry_attempted,
        )
        return distances, penalties, retry_attempted

    def router_config_value(self, name: str, default: Any) -> Any:
        return getattr(self.router, "config", {}).get(name, default)

    def raw_movement_status(self, left_uid: str, right_uid: str) -> str:
        raw = self.router.raw_movement(left_uid, right_uid)
        return (
            str(raw.restriction_status)
            if raw is not None
            else "no_direct_raw_movement_record"
        )

    def selected_transition_paths(
        self,
        selected: list[Candidate | None],
        observed_step: np.ndarray,
        time_gap: np.ndarray,
        config: dict[str, Any],
        from_point_index: int = 1,
        to_point_index: int | None = None,
    ) -> tuple[list[TransitionSelection], int, list[TransitionFailure]]:
        paths: list[TransitionSelection] = []
        failures: list[TransitionFailure] = []
        requested = 0
        first = max(1, int(from_point_index))
        last = min(
            len(selected) - 1,
            int(to_point_index) if to_point_index is not None else len(selected) - 1,
        )
        for point_index in range(first, last + 1):
            left, right = selected[point_index - 1], selected[point_index]
            if left is None or right is None:
                continue
            cutoff = transition_cutoff(
                float(observed_step[point_index]), float(time_gap[point_index]), config
            )
            if left.edge_uid == right.edge_uid:
                delta = right.position - left.position
                if delta < -self.jitter_tolerance:
                    failures.append(TransitionFailure(
                        point_index, left.edge_uid, right.edge_uid,
                        float(observed_step[point_index]), cutoff, abs(delta),
                        "same_edge_reverse_exceeds_jitter", "same_edge",
                        True,
                        "candidate_state_direction_mismatch",
                    ))
                    continue
                endpoint_distance = max(0.0, delta)
                jitter_penalty = (
                    abs(delta) * self.jitter_penalty_per_m
                    if delta < 0
                    else 0.0
                )
            else:
                endpoint_distance = (
                    max(0.0, self.lengths[left.edge_index] - left.position)
                    + right.position
                )
                jitter_penalty = 0.0
            initial_cutoff = cutoff
            retry_used = False
            if endpoint_distance > cutoff:
                retry_cutoff = min(
                    self.router.maximum,
                    float(self.router_config_value(
                        "transition_retry_max_m", 1200.0
                    )),
                    cutoff
                    * float(self.router_config_value(
                        "transition_retry_multiplier", 1.5
                    )),
                )
                if endpoint_distance > retry_cutoff:
                    failures.append(TransitionFailure(
                        point_index, left.edge_uid, right.edge_uid,
                        float(observed_step[point_index]), cutoff, endpoint_distance,
                        "endpoint_distance_exceeds_retry_cutoff",
                        self.raw_movement_status(
                            left.edge_uid, right.edge_uid
                        ),
                        True,
                        "cutoff_too_small_or_candidate_state_mismatch",
                    ))
                    continue
                cutoff = retry_cutoff
                retry_used = True
            result = self.router.transition_path(
                left.edge_uid, right.edge_uid, cutoff - endpoint_distance
            )
            if result is None:
                retry_cutoff = min(
                    self.router.maximum,
                    float(self.router_config_value(
                        "transition_retry_max_m", 1200.0
                    )),
                    initial_cutoff
                    * float(self.router_config_value(
                        "transition_retry_multiplier", 1.5
                    )),
                )
                if not retry_used and endpoint_distance <= retry_cutoff:
                    remaining_retry = retry_cutoff - endpoint_distance
                    reachable = self.router.multi_target_distances(
                        [left.edge_uid], [right.edge_uid], remaining_retry
                    ).get((left.edge_uid, right.edge_uid), math.inf)
                    if math.isfinite(reachable) and reachable <= remaining_retry:
                        result = self.router.transition_path(
                            left.edge_uid,
                            right.edge_uid,
                            remaining_retry,
                        )
                if result is None:
                    failures.append(TransitionFailure(
                        point_index, left.edge_uid, right.edge_uid,
                        float(observed_step[point_index]), cutoff, endpoint_distance,
                        "no_movement_path_within_retry_cutoff",
                        self.raw_movement_status(
                            left.edge_uid, right.edge_uid
                        ),
                        True,
                        (
                            "restriction_barrier"
                            if self.raw_movement_status(
                                left.edge_uid, right.edge_uid
                            ).startswith("forbidden")
                            else "no_path_within_exact_cutoff"
                        ),
                    ))
                    continue
                if result is not None:
                    cutoff = retry_cutoff
                    retry_used = True
            if left.edge_uid != right.edge_uid and self.router.movement(left.edge_uid, right.edge_uid) is None:
                requested += 1
            full_distance = endpoint_distance + result.physical_distance_m
            paths.append(TransitionSelection(
                point_index=point_index,
                from_edge_uid=left.edge_uid,
                to_edge_uid=right.edge_uid,
                observed_step_m=float(observed_step[point_index]),
                time_gap_s=float(time_gap[point_index]),
                transition_cutoff_m=cutoff,
                selected_path_distance_m=full_distance,
                generalized_routing_cost=(
                    result.generalized_routing_cost + jitter_penalty
                ),
                path_identifier=result.path_identifier,
                edge_uids=result.edge_uids,
                retry_used=retry_used,
                initial_cutoff_m=initial_cutoff,
                search_exact=result.search_exact,
                jitter_penalty_m=jitter_penalty,
            ))
        return paths, requested, failures

    def stats(self) -> SearchStats:
        return self.router.stats()


class CandidateIndex:
    def __init__(
        self,
        edges: gpd.GeoDataFrame,
        config: dict[str, Any],
        cache_dir: str | None = None,
        metric_crs: str = "EPSG:32649",
        emission_config: dict[str, Any] | None = None,
    ):
        self.edges = edges.reset_index(drop=True)
        self.config = config
        emission_values = emission_config or config
        self.emission_sigma_distance_m = float(
            emission_values.get(
                "sigma_distance_m",
                config.get("emission_sigma_distance_m", 15.0),
            )
        )
        self.transformer = Transformer.from_crs(4326, metric_crs, always_xy=True)
        self.geometries = self.edges.geometry.to_numpy()
        self.edge_uids = self.edges.edge_uid.astype(str).to_numpy()
        self.candidate_aliases = self.edges.get("candidate_alias_uid", self.edges.edge_uid).astype(str).to_numpy()
        self.alias_digest = hashlib.sha256("\n".join(self.candidate_aliases).encode("utf-8")).hexdigest()
        identity = hashlib.sha256()
        for edge_uid, geometry in zip(self.edge_uids, self.geometries):
            identity.update(edge_uid.encode("utf-8"))
            identity.update(b"\0")
            identity.update(bytes(geometry.wkb))
        self.edge_identity_digest = identity.hexdigest()
        self.snapshot_mismatch = self.edges.get(
            "network_snapshot_mismatch", pd.Series(False, index=self.edges.index)
        ).fillna(False).astype(bool).to_numpy()
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
                and stored.get("edge_identity_digest") == self.edge_identity_digest
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
                "edge_identity_digest": self.edge_identity_digest,
            }, sort_keys=True), encoding="utf-8")
            temporary.replace(cache / "metadata.json")
        self.tree = cKDTree(self.sample_xy)

    def transform(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = self.transformer.transform(lon, lat)
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def candidates_batch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        gps_headings: np.ndarray,
        heading_reliable: np.ndarray | None = None,
        radius_m: float | None = None,
        candidate_limit_override: int | None = None,
    ) -> list[list[Candidate]]:
        """Batch radius recall and vectorized exact point-to-edge projection for one order."""
        if not len(x):
            return []
        radius = float(radius_m if radius_m is not None else self.config["radius_m"])
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
        reliable = (
            np.ones(len(x), dtype=bool)
            if heading_reliable is None
            else np.asarray(heading_reliable, dtype=bool)
        )
        heading_delta = angular_difference(gps_headings[point_index_array], edge_headings)
        heading_delta = np.where(reliable[point_index_array], heading_delta, 0.0)
        heading_log_cost = (
            heading_delta / 180.0
            * float(self.config.get("heading_log_cost_max", 1.0))
        )
        edge_prior_log_cost = (
            self.candidate_penalties[edge_index_array]
            / max(float(self.config.get("edge_prior_scale_m", 30.0)), 1e-9)
        )
        projection_emission_cost = (
            distances**2 / (2.0 * self.emission_sigma_distance_m**2)
        )
        total_emission_cost = (
            projection_emission_cost
            + np.where(reliable[point_index_array], heading_log_cost, 0.0)
            + edge_prior_log_cost
        )
        # Retained only as a legacy diagnostic.  It no longer determines
        # candidate identity, truncation, rank, or the fast state.
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
                    float(heading_log_cost[pair_position]),
                    float(edge_prior_log_cost[pair_position]),
                    bool(reliable[point_index]),
                    float(projection_emission_cost[pair_position]),
                    float(total_emission_cost[pair_position]),
                )
                alias = str(self.candidate_aliases[edge_index])
                previous = by_alias.get(alias)
                if previous is None or (
                    candidate.total_emission_cost,
                    candidate.edge_uid,
                ) < (
                    previous.total_emission_cost,
                    previous.edge_uid,
                ):
                    by_alias[alias] = candidate
            raw = sorted(
                by_alias.values(),
                key=lambda value: (value.total_emission_cost, value.edge_uid),
            )
            if not raw:
                continue
            probe_count = min(
                len(raw), int(self.config.get("complex_candidate_probe_count", 8))
            )
            complex_area = any(
                bool(pd.notna(self.parallel_groups[candidate.edge_index]))
                or self.bridges[candidate.edge_index]
                or self.tunnels[candidate.edge_index]
                or self.highways[candidate.edge_index].endswith("_link")
                for candidate in raw[:probe_count]
            )
            dense = len(raw) >= 5 and raw[4].distance <= 30.0
            retained = (
                int(candidate_limit_override)
                if candidate_limit_override is not None
                else int(self.config[
                    "complex_candidates"
                    if complex_area
                    else "dense_candidates"
                    if dense
                    else "ordinary_candidates"
                ])
            )
            if candidate_limit_override is None:
                retained = min(retained, int(self.config["max_candidates"]))
            else:
                retained = min(
                    retained,
                    int(
                        self.config.get(
                            "absolute_max_candidates",
                            max(
                                int(self.config["max_candidates"]),
                                int(candidate_limit_override),
                            ),
                        )
                    ),
                )
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
    # An unreliable/near-stationary observation contributes no direction
    # evidence.  We retain a numeric placeholder for schema stability, but the
    # reliability mask prevents it from entering ranking or emission costs.
    heading[~reliable] = 0.0
    return heading, reliable


def _viterbi(
    candidate_rows: list[list[Candidate]],
    observed_step: np.ndarray,
    time_gap: np.ndarray,
    engine: TransitionEngine,
    config: dict[str, Any],
    start_anchor: Candidate | None = None,
    end_anchor: Candidate | None = None,
) -> tuple[list[Candidate], float, float] | None:
    if any(not row for row in candidate_rows):
        return None
    sigma = float(config["sigma_distance_m"])
    beta = float(config["beta_transition_m"])
    def emission(candidate: Candidate) -> float:
        return -candidate_emission_cost(candidate, sigma)

    def anchored(candidate: Candidate, anchor: Candidate | None) -> bool:
        return anchor is None or (
            candidate.edge_uid == anchor.edge_uid
            and abs(candidate.position - anchor.position)
            <= float(config.get("anchor_position_tolerance_m", 1.0))
        )

    scores = np.array([
        emission(candidate)
        if anchored(candidate, start_anchor) else -np.inf
        for candidate in candidate_rows[0]
    ], dtype=float)
    if not np.isfinite(scores).any():
        return None
    back: list[np.ndarray] = []
    transition_search_ms = 0.0
    for point_index in range(1, len(candidate_rows)):
        current = candidate_rows[point_index]
        previous = candidate_rows[point_index - 1]
        next_scores = np.full(len(current), -np.inf)
        pointers = np.full(len(current), -1, dtype="int32")
        dynamic_cutoff = transition_cutoff(
            float(observed_step[point_index]), float(time_gap[point_index]), config
        )
        search_started = time.perf_counter()
        (
            network_distances,
            jitter_penalties,
            retry_already_attempted,
        ) = engine.transition_matrices(previous, current, dynamic_cutoff)
        reachable_previous = np.isfinite(scores)
        if (
            reachable_previous.any()
            and not np.isfinite(network_distances[reachable_previous, :]).any()
            and not retry_already_attempted
        ):
            (
                network_distances,
                jitter_penalties,
                _,
            ) = engine.transition_matrices(
                previous,
                current,
                dynamic_cutoff,
                force_retry=True,
            )
        transition_search_ms += (time.perf_counter() - search_started) * 1000.0
        for j, right in enumerate(current):
            if point_index == len(candidate_rows) - 1 and not anchored(right, end_anchor):
                continue
            emission_score = emission(right)
            for i, left in enumerate(previous):
                network_distance = float(network_distances[i, j])
                if not math.isfinite(network_distance):
                    continue
                transition = (
                    -abs(network_distance - float(observed_step[point_index])) / beta
                    - float(jitter_penalties[i, j])
                    / float(config.get("beta_semantic_cost_m", 60.0))
                )
                value = scores[i] + emission_score + transition
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
) -> tuple[bool, str]:
    if not len(flags):
        return True, "empty_order"
    # ``flags`` is already the union of heading-reliable emission ambiguity
    # and unmasked transition ambiguity.  Accepting another eligibility mask
    # here would allow low-speed transition failures to disappear again.
    share = float(np.asarray(flags, dtype=bool).mean())
    if share >= float(config["full_order_ambiguity_share"]):
        return True, "raw_ambiguity_share"
    return False, ""


def effective_ambiguity_flags(
    emission_flags: np.ndarray,
    transition_flags: np.ndarray,
    heading_reliable: np.ndarray,
) -> np.ndarray:
    """Mask heading-dependent ambiguity without masking route discontinuity."""
    return (
        np.asarray(emission_flags, dtype=bool)
        & np.asarray(heading_reliable, dtype=bool)
    ) | np.asarray(transition_flags, dtype=bool)


def local_failures_require_full_order(failed_windows: int, config: dict[str, Any]) -> bool:
    return int(failed_windows) >= int(config.get("full_order_min_windows", 4))


def _ambiguity_flags(
    candidate_rows: list[list[Candidate]],
    edges: gpd.GeoDataFrame,
    emission_margin_cost: float,
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
            close_candidates = (
                b.total_emission_cost - a.total_emission_cost
                <= float(emission_margin_cost)
            )
            spatially_plausible = max(a.distance, b.distance) <= 30.0
            headings_plausible = max(a.heading_difference, b.heading_difference) <= 75.0
            if close_candidates and spatially_plausible and headings_plausible and pd.notna(left.parallel_group) and left.parallel_group == right.parallel_group:
                found.append("parallel_group")
            if close_candidates and spatially_plausible and headings_plausible and (left.layer, left.bridge, left.tunnel) != (right.layer, right.bridge, right.tunnel):
                found.append("grade_separation")
            if close_candidates and {str(left.highway), str(right.highway)} & {"service", "motorway_link", "trunk_link", "primary_link"}:
                found.append("main_auxiliary")
            if close_candidates and spatially_plausible and abs(a.heading_difference - b.heading_difference) <= 10 and (
                pd.notna(left.parallel_group)
                or str(left.highway) != str(right.highway)
                or (left.layer, left.bridge, left.tunnel) != (right.layer, right.bridge, right.tunnel)
            ):
                found.append("heading_tie")
        flags[index] = bool(found)
        reasons[index] = "|".join(found)
    return flags, reasons


def _transition_ambiguity_flags(
    geometric: list[Candidate | None],
    observed_step: np.ndarray,
    time_gap: np.ndarray,
    engine: TransitionEngine,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str], list[TransitionSelection]]:
    """Flag only emission uncertainty that affects route continuity."""
    flags = np.zeros(len(geometric), dtype=bool)
    reasons = ["" for _ in geometric]
    evidence: list[TransitionSelection] = []
    for point_index, (left, right) in enumerate(zip(geometric, geometric[1:]), start=1):
        found: list[str] = []
        if left is None or right is None:
            continue
        cutoff = transition_cutoff(
            float(observed_step[point_index]), float(time_gap[point_index]), config
        )
        endpoint_distance = (
            max(0.0, engine.lengths[left.edge_index] - left.position) + right.position
        ) if left.edge_uid != right.edge_uid else max(0.0, right.position - left.position)
        path_distance = math.inf
        direct = (
            left.edge_uid == right.edge_uid
            or engine.router.movement(left.edge_uid, right.edge_uid) is not None
        )
        if endpoint_distance > cutoff:
            found.append("endpoint_distance_exceeds_cutoff")
        elif direct:
            path_distance = 0.0
        else:
            # Ambiguity detection needs reachability and physical distance,
            # not a concrete generalized-cost path.  Using the order-local
            # distance frontier here avoids an expensive Pareto path search
            # that HMM may never use.
            path_distance = engine.router.multi_target_distances(
                [left.edge_uid],
                [right.edge_uid],
                cutoff - endpoint_distance,
            ).get((left.edge_uid, right.edge_uid), math.inf)
        if not math.isfinite(path_distance):
            found.append("no_legal_transition")
        else:
            distance = endpoint_distance + path_distance
            ratio = distance / max(float(observed_step[point_index]), 1.0)
            if ratio > float(config.get("transition_ambiguity_path_gps_ratio", 3.0)):
                found.append("network_gps_ratio")
            if (
                left.edge_uid != right.edge_uid
                and not direct
                and path_distance
                > float(config.get("transition_ambiguity_inferred_m", 250.0))
            ):
                found.append("long_inferred_bridge")
            evidence.append(TransitionSelection(
                point_index, left.edge_uid, right.edge_uid,
                float(observed_step[point_index]), float(time_gap[point_index]), cutoff,
                distance, math.nan, "", (),
            ))
        if found:
            flags[max(0, point_index - 1):min(len(flags), point_index + 1)] = True
            reasons[point_index] = "|".join(found)
    return flags, reasons, evidence


def match_order(
    frame: pd.DataFrame,
    edges: gpd.GeoDataFrame,
    index: CandidateIndex,
    engine: TransitionEngine,
    candidate_config: dict[str, Any],
    hmm_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    order_started = time.perf_counter()
    engine.begin_order()
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
    candidate_rows = index.candidates_batch(
        x, y, headings, heading_reliable=heading_reliable
    )
    minimum_candidates = int(candidate_config.get("min_candidates", 2))
    under_minimum_initial = np.asarray(
        [len(row) < minimum_candidates for row in candidate_rows], dtype=bool
    )
    no_candidate_initial = np.asarray([not row for row in candidate_rows], dtype=bool)
    recovered_candidate_count = 0
    no_candidate_recovered_count = 0
    if under_minimum_initial.any():
        recovery_radius = min(
            float(candidate_config.get("no_candidate_recovery_max_radius_m", 200.0)),
            float(candidate_config["radius_m"])
            * float(candidate_config.get("no_candidate_recovery_radius_multiplier", 2.0)),
        )
        recovered_rows = index.candidates_batch(
            x,
            y,
            headings,
            heading_reliable=heading_reliable,
            radius_m=recovery_radius,
        )
        for point_index in np.flatnonzero(under_minimum_initial):
            if len(recovered_rows[int(point_index)]) > len(candidate_rows[int(point_index)]):
                candidate_rows[int(point_index)] = recovered_rows[int(point_index)]
                recovered_candidate_count += 1
                if no_candidate_initial[int(point_index)]:
                    no_candidate_recovered_count += 1
    candidate_generation_ms = (time.perf_counter() - candidate_started) * 1000.0
    timestamp_values = pd.to_numeric(frame.timestamp, errors="coerce").to_numpy(float)
    time_gap = np.r_[0.0, np.maximum(np.diff(timestamp_values), 0.0)]
    observed_step = np.r_[0.0, np.hypot(np.diff(x), np.diff(y))]
    ambiguity_started = time.perf_counter()
    emission_flags, ambiguity_reasons = _ambiguity_flags(
        candidate_rows,
        edges,
        float(candidate_config["ambiguity_emission_margin_cost"]),
        heading_reliable,
    )
    geometric = [row[0] if row else None for row in candidate_rows]
    transition_flags, transition_reasons, fast_transition_evidence = _transition_ambiguity_flags(
        geometric, observed_step, time_gap, engine, hmm_config,
    )
    transition_recovery_points = sorted({
        neighbour
        for point_index, reason in enumerate(transition_reasons)
        if "no_legal_transition" in reason
        or "endpoint_distance_exceeds_cutoff" in reason
        for neighbour in (point_index - 1, point_index)
        if 0 <= neighbour < len(candidate_rows)
    })
    transition_candidate_expansion_count = 0
    if transition_recovery_points:
        expanded_rows = index.candidates_batch(
            x,
            y,
            headings,
            heading_reliable=heading_reliable,
            radius_m=float(
                candidate_config.get("transition_recovery_radius_m", 200.0)
            ),
            candidate_limit_override=int(
                candidate_config.get("transition_recovery_max_candidates", 20)
            ),
        )
        for point_index in transition_recovery_points:
            if len(expanded_rows[point_index]) > len(candidate_rows[point_index]):
                candidate_rows[point_index] = expanded_rows[point_index]
                transition_candidate_expansion_count += 1
        if transition_candidate_expansion_count:
            emission_flags, ambiguity_reasons = _ambiguity_flags(
                candidate_rows,
                edges,
                float(candidate_config["ambiguity_emission_margin_cost"]),
                heading_reliable,
            )
            geometric = [row[0] if row else None for row in candidate_rows]
            (
                transition_flags,
                transition_reasons,
                fast_transition_evidence,
            ) = _transition_ambiguity_flags(
                geometric, observed_step, time_gap, engine, hmm_config,
            )
    # Unreliable headings suppress only emission ambiguity.  A physical
    # transition failure near a stationary/slow point remains valid evidence
    # and must participate in both local windows and full-order escalation.
    flags = effective_ambiguity_flags(
        emission_flags, transition_flags, heading_reliable
    )
    ambiguity_reasons = [
        "|".join(filter(None, (emission_reason, transition_reason)))
        for emission_reason, transition_reason in zip(ambiguity_reasons, transition_reasons)
    ]
    ambiguity_detection_ms = (time.perf_counter() - ambiguity_started) * 1000.0
    selected = list(geometric)
    windows = local_hmm_windows(flags, int(hmm_config["local_context_points"]), int(hmm_config["merge_gap_points"]))
    no_candidate = np.asarray([not row for row in candidate_rows], dtype=bool)
    full_order, full_order_trigger_reason = full_order_decision(
        flags, hmm_config
    )
    hmm_score = math.nan
    fallback_reason = ""
    local_hmm_ms = 0.0
    full_hmm_ms = 0.0
    transition_search_ms = 0.0
    local_failed_windows = 0
    local_boundary_failure_count = 0
    local_internal_failure_count = 0
    failed_window_details: list[dict[str, Any]] = []
    local_hmm_attempted = False
    local_hmm_window_attempt_count = 0
    local_hmm_retry_window_count = 0
    boundary_repair_viterbi_count = 0
    full_hmm_attempted = False
    full_hmm_succeeded = False
    local_patch_count = 0
    if no_candidate.any():
        mode = "rejected"
        fallback_reason = "unrecoverable_no_candidate"
        full_order = False
        windows = []
    elif full_order:
        full_hmm_attempted = True
        full_started = time.perf_counter()
        solved = _viterbi(candidate_rows, observed_step, time_gap, engine, hmm_config)
        full_hmm_ms += (time.perf_counter() - full_started) * 1000.0
        if solved:
            full_hmm_succeeded = True
            selected, hmm_score, searched_ms = solved
            transition_search_ms += searched_ms
            mode = "full_order_hmm"
        else:
            selected = list(geometric)
            mode = "pure_geometric_fallback" if hmm_config.get("geometric_fallback_enabled") else "rejected"
            fallback_reason = "full_order_hmm_no_continuous_path"
    elif windows:
        mode = "local_hmm"
        local_scores: list[float] = []
        for start, stop in windows:
            # Solve with true anchors outside the ambiguous interior.  Only
            # the interior states are written back.
            solve_start = max(0, start - 1)
            solve_stop = min(len(candidate_rows), stop + 1)
            start_anchor = geometric[solve_start] if start > 0 else None
            end_anchor = geometric[solve_stop - 1] if stop < len(candidate_rows) else None
            local_started = time.perf_counter()
            local_hmm_attempted = True
            local_hmm_window_attempt_count += 1
            solved = _viterbi(
                candidate_rows[solve_start:solve_stop],
                observed_step[solve_start:solve_stop],
                time_gap[solve_start:solve_stop],
                engine, hmm_config, start_anchor=start_anchor, end_anchor=end_anchor,
            )
            local_hmm_ms += (time.perf_counter() - local_started) * 1000.0
            if not solved:
                expanded_start = max(0, start - int(hmm_config.get("local_retry_context_points", 2)))
                expanded_stop = min(len(candidate_rows), stop + int(hmm_config.get("local_retry_context_points", 2)))
                retry_solve_start = max(0, expanded_start - 1)
                retry_solve_stop = min(len(candidate_rows), expanded_stop + 1)
                retry_started = time.perf_counter()
                local_hmm_retry_window_count += 1
                retry_start_anchor = (
                    geometric[retry_solve_start] if expanded_start > 0 else None
                )
                retry_end_anchor = (
                    geometric[retry_solve_stop - 1]
                    if expanded_stop < len(candidate_rows)
                    else None
                )
                solved = _viterbi(
                    candidate_rows[retry_solve_start:retry_solve_stop],
                    observed_step[retry_solve_start:retry_solve_stop],
                    time_gap[retry_solve_start:retry_solve_stop],
                    engine,
                    hmm_config,
                    start_anchor=retry_start_anchor,
                    end_anchor=retry_end_anchor,
                )
                local_hmm_ms += (time.perf_counter() - retry_started) * 1000.0
                if solved:
                    local_path, local_score, searched_ms = solved
                    left_offset = expanded_start - retry_solve_start
                    right_offset = left_offset + (expanded_stop - expanded_start)
                    trial = list(selected)
                    trial[expanded_start:expanded_stop] = local_path[
                        left_offset:right_offset
                    ]
                    _, _, patch_failures = engine.selected_transition_paths(
                        trial,
                        observed_step,
                        time_gap,
                        hmm_config,
                        from_point_index=max(1, expanded_start),
                        to_point_index=min(
                            len(trial) - 1, expanded_stop
                        ),
                    )
                    if patch_failures:
                        local_failed_windows += 1
                        local_boundary_failure_count += 1
                        failed_window_details.append({
                            "start": start,
                            "stop": stop,
                            "failure_stage": "expanded_patch_boundary",
                            "failure_reasons": [
                                failure.reason for failure in patch_failures
                            ],
                        })
                        continue
                    selected = trial
                    local_patch_count += 1
                    local_scores.append(local_score)
                    transition_search_ms += searched_ms
                    continue
                local_failed_windows += 1
                local_internal_failure_count += 1
                failed_window_details.append({
                    "start": start,
                    "stop": stop,
                    "anchor_before": (
                        str(start_anchor.edge_uid) if start_anchor is not None else ""
                    ),
                    "anchor_after": (
                        str(end_anchor.edge_uid) if end_anchor is not None else ""
                    ),
                    "candidate_counts": [
                        len(row) for row in candidate_rows[start:stop]
                    ],
                    "failure_stage": "expanded_local_viterbi",
                })
                continue
            local_path, local_score, searched_ms = solved
            left_offset = start - solve_start
            right_offset = left_offset + (stop - start)
            trial = list(selected)
            trial[start:stop] = local_path[left_offset:right_offset]
            _, _, patch_failures = engine.selected_transition_paths(
                trial,
                observed_step,
                time_gap,
                hmm_config,
                from_point_index=max(1, start),
                to_point_index=min(len(trial) - 1, stop),
            )
            if patch_failures:
                local_failed_windows += 1
                local_boundary_failure_count += 1
                failed_window_details.append({
                    "start": start,
                    "stop": stop,
                    "failure_stage": "patch_boundary",
                    "failure_reasons": [
                        failure.reason for failure in patch_failures
                    ],
                })
                continue
            selected = trial
            local_patch_count += 1
            local_scores.append(local_score)
            transition_search_ms += searched_ms
        if local_failed_windows:
            if local_failures_require_full_order(local_failed_windows, hmm_config):
                full_order_trigger_reason = "multiple_local_window_failures"
                full_hmm_attempted = True
                full_started = time.perf_counter()
                solved = _viterbi(candidate_rows, observed_step, time_gap, engine, hmm_config)
                full_hmm_ms += (time.perf_counter() - full_started) * 1000.0
                if solved:
                    full_hmm_succeeded = True
                    selected, hmm_score, searched_ms = solved
                    transition_search_ms += searched_ms
                    mode = "full_order_hmm"
                else:
                    mode = (
                        "partial_local_hmm_fallback"
                        if local_patch_count
                        else "pure_geometric_fallback"
                    ) if hmm_config.get("geometric_fallback_enabled") else "rejected"
                    fallback_reason = "multiple_local_and_full_hmm_no_continuous_path"
            else:
                mode = (
                    "partial_local_hmm_fallback"
                    if local_patch_count
                    else "pure_geometric_fallback"
                ) if hmm_config.get("geometric_fallback_enabled") else "rejected"
                fallback_reason = "local_window_no_continuous_path"
        else:
            hmm_score = float(sum(local_scores))
    else:
        mode = "fast_deterministic"
    if any(value is None for value in selected):
        mode = "rejected"
        fallback_reason = "one_or_more_points_without_candidate"
    boundary_repair_attempt_count = 0
    boundary_repair_success_count = 0
    if mode != "rejected" and selected:
        _, _, probe_failures = engine.selected_transition_paths(
            selected, observed_step, time_gap, hmm_config,
        )
        if probe_failures:
            repair_flags = np.zeros(len(selected), dtype=bool)
            for failure in probe_failures:
                repair_flags[
                    max(0, failure.point_index - 1):
                    min(len(repair_flags), failure.point_index + 1)
                ] = True
            repair_windows = local_hmm_windows(
                repair_flags,
                int(hmm_config.get("boundary_repair_context_points", 1)),
                int(hmm_config.get("boundary_repair_merge_gap_points", 1)),
            )
            for start, stop in repair_windows:
                boundary_repair_attempt_count += 1
                boundary_repair_viterbi_count += 1
                solve_start = max(0, start - 1)
                solve_stop = min(len(candidate_rows), stop + 1)
                solved = _viterbi(
                    candidate_rows[solve_start:solve_stop],
                    observed_step[solve_start:solve_stop],
                    time_gap[solve_start:solve_stop],
                    engine,
                    hmm_config,
                    start_anchor=selected[solve_start] if start > 0 else None,
                    end_anchor=(
                        selected[solve_stop - 1]
                        if stop < len(candidate_rows)
                        else None
                    ),
                )
                if solved is None:
                    continue
                repaired_path, _, searched_ms = solved
                left_offset = start - solve_start
                right_offset = left_offset + (stop - start)
                candidate_patch = repaired_path[left_offset:right_offset]
                trial = list(selected)
                trial[start:stop] = candidate_patch
                _, _, trial_failures = engine.selected_transition_paths(
                    trial,
                    observed_step,
                    time_gap,
                    hmm_config,
                    from_point_index=max(1, start),
                    to_point_index=min(len(trial) - 1, stop),
                )
                if trial_failures:
                    continue
                selected = trial
                boundary_repair_success_count += 1
                transition_search_ms += searched_ms
            if boundary_repair_success_count and mode in {
                "pure_geometric_fallback", "partial_local_hmm_fallback"
            }:
                mode = "partial_local_hmm_repaired"
    pre_transition_validation_mode = mode
    selected_path_started = time.perf_counter()
    (
        selected_transitions,
        selected_bridge_request_count,
        selected_transition_failures,
    ) = engine.selected_transition_paths(
        selected, observed_step, time_gap, hmm_config,
    )
    selected_path_search_ms = (time.perf_counter() - selected_path_started) * 1000.0
    expected_transition_count = max(0, len(selected) - 1) if mode != "rejected" else 0
    if mode != "rejected" and len(selected_transitions) != expected_transition_count:
        # Preserve the actually attempted state sequence for failure
        # diagnostics.  Replacing it with the geometric sequence here made
        # provisional_edge_uid disagree with the transition failures that
        # caused rejection, preventing a faithful post-run audit.
        mode = "failed_no_continuous_route"
        fallback_reason = "selected_transition_exceeds_dynamic_cutoff"
    frame["point_seq"] = np.arange(len(frame), dtype="int32")
    frame["source_lon"], frame["source_lat"] = frame.lon, frame.lat
    frame["matching_lon"], frame["matching_lat"] = lon, lat
    frame["metric_x"], frame["metric_y"] = x, y
    frame["candidate_count"] = [len(row) for row in candidate_rows]
    frame["observed_step_m"] = observed_step
    frame["time_gap_s"] = time_gap
    frame["heading_reliable"] = heading_reliable
    frame["stationary_or_low_motion"] = ~heading_reliable
    frame["parallel_ambiguity"] = ["parallel_group" in reason for reason in ambiguity_reasons]
    frame["ambiguity_reason"] = ambiguity_reasons
    frame["matching_mode"] = mode
    frame["provisional_edge_uid"] = [
        value.edge_uid if value is not None else pd.NA for value in selected
    ]
    frame["provisional_position_on_edge"] = [
        value.position if value is not None else math.nan for value in selected
    ]
    frame["transition_failure_reason"] = ""
    frame["transition_failure_raw_movement_status"] = ""
    frame["transition_failure_search_exact"] = pd.NA
    frame["transition_failure_diagnostic_class"] = ""
    for failure in selected_transition_failures:
        row_index = frame.index[failure.point_index]
        frame.loc[row_index, "transition_failure_reason"] = failure.reason
        frame.loc[row_index, "transition_failure_raw_movement_status"] = (
            failure.raw_movement_status
        )
        frame.loc[row_index, "transition_failure_search_exact"] = (
            failure.search_exact
        )
        frame.loc[row_index, "transition_failure_diagnostic_class"] = (
            failure.diagnostic_class
        )
    accepted_mode = mode not in {"rejected", "failed_no_continuous_route"}
    if accepted_mode:
        frame["edge_uid"] = [value.edge_uid for value in selected]
        frame["position_on_edge"] = [value.position for value in selected]
        frame["gps_to_edge_distance_m"] = [value.distance for value in selected]
        frame["candidate_rank"] = [value.rank for value in selected]
        frame["edge_heading_difference_deg"] = [value.heading_difference for value in selected]
        frame["legacy_candidate_score"] = [value.score for value in selected]
        frame["projection_emission_cost"] = [
            value.projection_emission_cost for value in selected
        ]
        frame["heading_emission_cost"] = [
            value.heading_log_cost if value.heading_used else 0.0
            for value in selected
        ]
        frame["edge_prior_cost"] = [value.edge_prior_log_cost for value in selected]
        frame["total_emission_cost"] = [
            value.total_emission_cost for value in selected
        ]
        frame["emission_margin"] = [
            row[1].total_emission_cost - row[0].total_emission_cost
            if len(row) > 1
            else math.inf
            for row in candidate_rows
        ]
    else:
        for column in [
            "edge_uid",
            "position_on_edge",
            "gps_to_edge_distance_m",
            "candidate_rank",
            "edge_heading_difference_deg",
            "legacy_candidate_score",
            "projection_emission_cost",
            "heading_emission_cost",
            "edge_prior_cost",
            "total_emission_cost",
            "emission_margin",
        ]:
            frame[column] = pd.NA
    frame["viterbi_margin"] = math.nan
    frame["transition_cutoff_m"] = np.nan
    frame["selected_path_distance_m"] = np.nan
    frame["selected_path_routing_cost"] = np.nan
    frame["selected_path_identifier"] = ""
    frame["selected_path_json"] = ""
    frame["transition_retry_used"] = False
    frame["transition_initial_cutoff_m"] = np.nan
    frame["selected_path_search_exact"] = pd.NA
    frame["selected_jitter_penalty_m"] = np.nan
    for transition in selected_transitions if accepted_mode else []:
        frame.loc[frame.index[transition.point_index], "transition_cutoff_m"] = transition.transition_cutoff_m
        frame.loc[frame.index[transition.point_index], "selected_path_distance_m"] = transition.selected_path_distance_m
        frame.loc[frame.index[transition.point_index], "selected_path_routing_cost"] = transition.generalized_routing_cost
        frame.loc[frame.index[transition.point_index], "selected_path_identifier"] = transition.path_identifier
        frame.loc[frame.index[transition.point_index], "transition_retry_used"] = (
            transition.retry_used
        )
        frame.loc[
            frame.index[transition.point_index], "transition_initial_cutoff_m"
        ] = transition.initial_cutoff_m
        frame.loc[
            frame.index[transition.point_index], "selected_path_search_exact"
        ] = transition.search_exact
        frame.loc[
            frame.index[transition.point_index], "selected_jitter_penalty_m"
        ] = transition.jitter_penalty_m
        frame.loc[frame.index[transition.point_index], "selected_path_json"] = json.dumps(
            transition.edge_uids, separators=(",", ":")
        )
    frame["path_to_gps_ratio"] = frame.selected_path_distance_m / frame.observed_step_m.clip(lower=1.0)
    if accepted_mode:
        selected_indices = [value.edge_index for value in selected]
        frame["selected_network_snapshot_mismatch"] = index.snapshot_mismatch[selected_indices]
    else:
        frame["selected_network_snapshot_mismatch"] = pd.NA
    frame["point_quality"] = np.where(flags, "ambiguous", "high_confidence")
    stats_after = engine.stats()
    stats_delta = stats_after.minus(stats_before)
    evidence_cache_hits, evidence_cache_misses = engine.evidence_cache_stats()
    summary = {
        "order_id": str(frame.order_id.iloc[0]) if len(frame) else "",
        "matching_mode": mode,
        "pre_transition_validation_mode": pre_transition_validation_mode,
        "fallback_used": mode in {"pure_geometric_fallback", "partial_local_hmm_fallback"},
        "fallback_reason": fallback_reason, "point_count": len(frame),
        "ambiguity_point_share": float(flags.mean()) if len(flags) else 1.0,
        "effective_ambiguity_point_share": (
            float(flags.mean()) if len(flags) else 1.0
        ),
        "eligible_ambiguity_point_share": float(flags.mean()) if len(flags) else 0.0,
        "emission_ambiguity_reliable_point_share": float(
            (emission_flags & heading_reliable).mean()
        ) if len(flags) else 0.0,
        "transition_ambiguity_point_share": float(
            transition_flags.mean()
        ) if len(flags) else 0.0,
        "stationary_point_share": float((~heading_reliable).mean()) if len(heading_reliable) else 1.0,
        "parallel_ambiguity_share": float(frame.parallel_ambiguity.mean()) if len(frame) else 0.0,
        "local_window_count": len(windows), "hmm_score": hmm_score,
        "local_failed_window_count": local_failed_windows,
        "local_boundary_failure_count": local_boundary_failure_count,
        "local_internal_failure_count": local_internal_failure_count,
        "local_failed_window_details": json.dumps(
            failed_window_details, ensure_ascii=False, separators=(",", ":")
        ),
        "boundary_repair_attempt_count": boundary_repair_attempt_count,
        "boundary_repair_success_count": boundary_repair_success_count,
        "local_patch_count": local_patch_count,
        "no_candidate_initial_count": int(no_candidate_initial.sum()),
        "under_minimum_candidate_initial_count": int(under_minimum_initial.sum()),
        "transition_candidate_expansion_count": transition_candidate_expansion_count,
        "no_candidate_recovered_count": no_candidate_recovered_count,
        "under_minimum_candidate_expansion_count": recovered_candidate_count,
        "full_order_trigger_reason": full_order_trigger_reason,
        "local_hmm_attempted": local_hmm_attempted,
        "local_hmm_order_attempt_count": int(local_hmm_attempted),
        "local_hmm_window_attempt_count": local_hmm_window_attempt_count,
        "local_hmm_retry_window_count": local_hmm_retry_window_count,
        "boundary_repair_viterbi_count": boundary_repair_viterbi_count,
        "full_hmm_attempted": full_hmm_attempted,
        "full_hmm_succeeded": full_hmm_succeeded,
        "full_hmm_failed": full_hmm_attempted and not full_hmm_succeeded,
        "selected_bridge_request_count": selected_bridge_request_count,
        "selected_bridge_path_count": len(selected_transitions),
        "selected_transition_failure_count": len(selected_transition_failures),
        "endpoint_distance_exceeds_cutoff_count": sum(
            failure.reason.startswith("endpoint_distance_exceeds")
            for failure in selected_transition_failures
        ),
        "no_movement_path_within_cutoff_count": sum(
            failure.reason.startswith("no_movement_path_within")
            for failure in selected_transition_failures
        ),
        "transition_retry_used_count": sum(
            transition.retry_used for transition in selected_transitions
        ),
        "failed_transition_reasons": json.dumps(
            [failure.__dict__ for failure in selected_transition_failures],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "provisional_edge_sequence": json.dumps(
            [value.edge_uid if value is not None else None for value in selected],
            separators=(",", ":"),
        ),
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
        "exact_path_search_calls": stats_delta.exact_path_calls,
        "approximate_path_search_calls": stats_delta.approximate_path_calls,
        "approximate_search_unresolved_count": (
            stats_delta.approximate_unresolved
        ),
        "positive_cache_hits": stats_delta.positive_cache_hits,
        "negative_cache_hits": stats_delta.negative_cache_hits,
        "path_cache_hits": stats_delta.path_cache_hits,
        "order_transition_evidence_cache_hits": evidence_cache_hits,
        "order_transition_evidence_cache_misses": evidence_cache_misses,
        "_selected_transitions": selected_transitions,
        "_selected_transition_failures": selected_transition_failures,
        "_fast_transition_evidence": fast_transition_evidence,
    }
    return frame, summary
