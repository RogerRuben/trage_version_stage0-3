"""Compact legal-movement routing shared by HMM transitions and reconstruction.

The HMM API intentionally returns physical distances only.  It never constructs candidate-pair
paths.  Concrete paths are recovered only for selected transitions through ``bridge_path``.
Positive caches are cutoff independent; negative caches store the exact exhaustively searched
cutoff, avoiding rounded-cutoff false negatives.
"""

from __future__ import annotations

import heapq
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MovementRecord:
    from_edge_uid: str
    via_node: int
    to_edge_uid: str
    movement_type: str
    turn_angle: float
    restriction_status: str
    layer_compatibility: bool
    level_transition_type: str
    road_class_transition: str
    merge_diverge_flag: bool


@dataclass(frozen=True)
class SearchStats:
    distance_calls: int = 0
    path_calls: int = 0
    expanded_nodes: int = 0
    positive_cache_hits: int = 0
    negative_cache_hits: int = 0
    path_cache_hits: int = 0
    cache_misses: int = 0

    @property
    def calls(self) -> int:
        return self.distance_calls + self.path_calls

    @property
    def cache_hits(self) -> int:
        return self.positive_cache_hits + self.negative_cache_hits + self.path_cache_hits

    def minus(self, earlier: "SearchStats") -> "SearchStats":
        return SearchStats(**{
            field: int(getattr(self, field) - getattr(earlier, field))
            for field in self.__dataclass_fields__
        })


@dataclass
class _SourceDistanceState:
    """Incremental, order-local physical-distance search state for one source edge."""

    queue: list[tuple[float, int]]
    best: dict[int, float]
    settled: dict[int, float]


class CompactMovementRouter:
    """CSR edge-state graph with distance-only Dijkstra and selected-path A*."""

    def __init__(self, edges: gpd.GeoDataFrame, movements: pd.DataFrame, config: dict[str, Any]) -> None:
        edge_frame = edges.reset_index(drop=True)
        self.edge_uids = edge_frame.edge_uid.astype(str).to_numpy()
        self.uid_to_index = {uid: index for index, uid in enumerate(self.edge_uids)}
        self.lengths = edge_frame.length_m.to_numpy(float)
        routing_penalty = pd.to_numeric(
            edge_frame.get("routing_penalty", pd.Series(1.0, index=edge_frame.index)), errors="coerce"
        ).fillna(1.0).clip(lower=1.0).to_numpy(float)
        candidate_penalty = pd.to_numeric(
            edge_frame.get("candidate_penalty", pd.Series(0.0, index=edge_frame.index)), errors="coerce"
        ).fillna(0.0).clip(lower=0.0).to_numpy(float)
        self.edge_routing_costs = self.lengths * routing_penalty + candidate_penalty
        self.maximum = float(config.get("max_route_distance_m", 6000.0))
        self.cache_limit = int(config.get("bridge_cache_pairs", 50_000))
        self._lock = threading.RLock()
        self._distance_positive: OrderedDict[tuple[int, int], float] = OrderedDict()
        self._distance_negative: OrderedDict[tuple[int, int], float] = OrderedDict()
        self._path_positive: OrderedDict[tuple[int, int], tuple[tuple[int, ...], float, float]] = OrderedDict()
        self._path_negative: OrderedDict[tuple[int, int], float] = OrderedDict()
        self._distance_source_states: dict[int, _SourceDistanceState] = {}
        self._stats = SearchStats()

        starts = np.empty((len(edge_frame), 2), dtype="float64")
        ends = np.empty((len(edge_frame), 2), dtype="float64")
        for index, geometry in enumerate(edge_frame.geometry):
            coordinates = np.asarray(geometry.coords, dtype=float)
            starts[index] = coordinates[0]
            ends[index] = coordinates[-1]
        self.start_xy, self.end_xy = starts, ends

        records: list[MovementRecord] = []
        rows: list[tuple[int, int, float, float, int]] = []
        for row in movements.itertuples(index=False):
            restriction = str(row.restriction_status)
            if restriction.startswith("forbidden") or not bool(row.layer_compatibility):
                continue
            left = self.uid_to_index.get(str(row.from_edge_uid))
            right = self.uid_to_index.get(str(row.to_edge_uid))
            if left is None or right is None:
                continue
            record = MovementRecord(
                from_edge_uid=str(row.from_edge_uid), via_node=int(row.via_node),
                to_edge_uid=str(row.to_edge_uid), movement_type=str(row.movement_type),
                turn_angle=float(row.turn_angle), restriction_status=restriction,
                layer_compatibility=bool(row.layer_compatibility),
                level_transition_type=str(getattr(row, "level_transition_type", "same_level")),
                road_class_transition=str(row.road_class_transition),
                merge_diverge_flag=bool(row.merge_diverge_flag),
            )
            movement_penalty = 0.0
            if record.movement_type == "u_turn":
                movement_penalty += float(config.get("u_turn_penalty_m", 500.0))
            classes = record.road_class_transition.split("->")
            if len(classes) == 2 and classes[0] != classes[1]:
                movement_penalty += float(config.get("road_class_transition_penalty_m", 30.0))
            record_index = len(records)
            records.append(record)
            rows.append((
                left, right, float(self.lengths[right]),
                float(self.edge_routing_costs[right] + movement_penalty), record_index,
            ))
        rows.sort(key=lambda value: (value[0], value[1], value[3]))
        self.records = records
        self.offsets = np.zeros(len(edge_frame) + 1, dtype="int64")
        for left, *_ in rows:
            self.offsets[left + 1] += 1
        np.cumsum(self.offsets, out=self.offsets)
        self.targets = np.asarray([row[1] for row in rows], dtype="int32")
        self.physical_costs = np.asarray([row[2] for row in rows], dtype="float32")
        self.routing_costs = np.asarray([row[3] for row in rows], dtype="float32")
        self.record_indices = np.asarray([row[4] for row in rows], dtype="int32")

    def _increment(self, **values: int) -> None:
        with self._lock:
            payload = {field: getattr(self._stats, field) + int(values.get(field, 0)) for field in self._stats.__dataclass_fields__}
            self._stats = SearchStats(**payload)

    def stats(self) -> SearchStats:
        with self._lock:
            return self._stats

    def begin_order(self) -> None:
        """Drop incremental source frontiers at the order boundary to cap memory use."""
        self._distance_source_states.clear()

    @property
    def cache_size(self) -> int:
        with self._lock:
            return len(self._distance_positive) + len(self._distance_negative) + len(self._path_positive) + len(self._path_negative)

    def _trim(self, cache: OrderedDict[Any, Any]) -> None:
        while len(cache) > self.cache_limit:
            cache.popitem(last=False)

    def movement(self, left_uid: str, right_uid: str) -> MovementRecord | None:
        left = self.uid_to_index.get(str(left_uid))
        right = self.uid_to_index.get(str(right_uid))
        if left is None or right is None:
            return None
        start, stop = int(self.offsets[left]), int(self.offsets[left + 1])
        positions = np.flatnonzero(self.targets[start:stop] == right)
        return None if not len(positions) else self.records[int(self.record_indices[start + int(positions[0])])]

    def _heuristic(self, edge_index: int, target_index: int) -> float:
        difference = self.end_xy[edge_index] - self.start_xy[target_index]
        return float(np.hypot(difference[0], difference[1]))

    def multi_target_distances(
        self, left_uids: Iterable[str], right_uids: Iterable[str], cutoff: float,
    ) -> dict[tuple[str, str], float]:
        """Return physical intermediate distances; never allocate or cache paths."""
        result: dict[tuple[str, str], float] = {}
        unique_right = list(dict.fromkeys(map(str, right_uids)))
        maximum = min(self.maximum, float(cutoff))
        for left_uid in dict.fromkeys(map(str, left_uids)):
            left = self.uid_to_index.get(left_uid)
            if left is None:
                continue
            unresolved: dict[int, str] = {}
            for right_uid in unique_right:
                right = self.uid_to_index.get(right_uid)
                pair = (left_uid, right_uid)
                if right is None:
                    result[pair] = math.inf
                elif left == right or self.movement(left_uid, right_uid) is not None:
                    result[pair] = 0.0
                else:
                    key = (left, right)
                    with self._lock:
                        positive = self._distance_positive.get(key)
                        negative_cutoff = self._distance_negative.get(key, -math.inf)
                        if positive is not None and positive <= maximum:
                            self._distance_positive.move_to_end(key)
                            self._increment(positive_cache_hits=1)
                            result[pair] = positive
                            continue
                        if positive is not None and positive > maximum:
                            self._increment(negative_cache_hits=1)
                            result[pair] = math.inf
                            continue
                        if maximum <= negative_cutoff:
                            self._distance_negative.move_to_end(key)
                            self._increment(negative_cache_hits=1)
                            result[pair] = math.inf
                            continue
                    unresolved[right] = right_uid
            if not unresolved:
                continue
            # Reuse one incremental physical-distance tree for this source during the current
            # order. Reaching a target includes its full edge, so allow that edge length beyond
            # the requested intermediate cutoff and subtract it on output.
            state = self._distance_source_states.get(left)
            if state is None:
                state = _SourceDistanceState(queue=[(0.0, left)], best={left: 0.0}, settled={})
                self._distance_source_states[left] = state
            remaining = set(unresolved)
            for target in tuple(remaining):
                if target not in state.settled:
                    continue
                middle = max(0.0, state.settled[target] - float(self.lengths[target]))
                key = (left, target)
                with self._lock:
                    self._distance_positive[key] = middle
                    self._distance_positive.move_to_end(key)
                    self._trim(self._distance_positive)
                result[(left_uid, unresolved[target])] = middle if middle <= maximum else math.inf
                remaining.remove(target)
                self._increment(positive_cache_hits=1)
            if not remaining:
                continue
            self._increment(distance_calls=1, cache_misses=len(remaining))
            search_limit = maximum + max(float(self.lengths[index]) for index in remaining)
            expanded = 0
            while state.queue and remaining:
                current_distance, current = heapq.heappop(state.queue)
                if current_distance != state.best.get(current) or current in state.settled:
                    continue
                if current_distance > search_limit:
                    heapq.heappush(state.queue, (current_distance, current))
                    break
                state.settled[current] = current_distance
                expanded += 1
                if current in remaining:
                    remaining.remove(current)
                start, stop = int(self.offsets[current]), int(self.offsets[current + 1])
                for position in range(start, stop):
                    target = int(self.targets[position])
                    proposed = current_distance + float(self.physical_costs[position])
                    if target not in state.settled and proposed < state.best.get(target, math.inf):
                        state.best[target] = proposed
                        heapq.heappush(state.queue, (proposed, target))
            self._increment(expanded_nodes=expanded)
            for target, right_uid in unresolved.items():
                key = (left, target)
                settled_distance = state.settled.get(target, math.inf)
                middle = max(0.0, settled_distance - float(self.lengths[target]))
                if math.isfinite(middle):
                    with self._lock:
                        previous = self._distance_positive.get(key)
                        self._distance_positive[key] = middle if previous is None else min(previous, middle)
                        self._distance_positive.move_to_end(key)
                        self._trim(self._distance_positive)
                    result[(left_uid, right_uid)] = middle if middle <= maximum else math.inf
                else:
                    result[(left_uid, right_uid)] = math.inf
                    with self._lock:
                        self._distance_negative[key] = max(maximum, self._distance_negative.get(key, 0.0))
                        self._distance_negative.move_to_end(key)
                        self._trim(self._distance_negative)
        return result

    def bridge_path(self, left_uid: str, right_uid: str, cutoff: float | None = None) -> tuple[list[str], float] | None:
        """Recover one selected path using routing/access penalties, with cutoff-safe caches."""
        left = self.uid_to_index.get(str(left_uid))
        right = self.uid_to_index.get(str(right_uid))
        if left is None or right is None:
            return None
        if left == right:
            return [str(left_uid)], 0.0
        if self.movement(str(left_uid), str(right_uid)) is not None:
            return [str(left_uid), str(right_uid)], 0.0
        maximum = min(self.maximum, float(cutoff) if cutoff is not None else self.maximum)
        key = (left, right)
        with self._lock:
            cached = self._path_positive.get(key)
            negative_cutoff = self._path_negative.get(key, -math.inf)
            if cached is not None and cached[2] <= maximum:
                self._path_positive.move_to_end(key)
                self._increment(path_cache_hits=1)
                return [str(self.edge_uids[index]) for index in cached[0]], cached[2]
            if maximum <= negative_cutoff:
                self._path_negative.move_to_end(key)
                self._increment(negative_cache_hits=1)
                return None
        self._increment(path_calls=1, cache_misses=1)
        # Routing cost chooses the path; physical distance enforces the search boundary.
        queue: list[tuple[float, float, int]] = [(self._heuristic(left, right), 0.0, left)]
        best_routing = {left: 0.0}
        physical = {left: 0.0}
        parent: dict[int, int] = {}
        expanded = 0
        found = False
        search_limit = maximum + float(self.lengths[right])
        while queue:
            _, routing_cost, current = heapq.heappop(queue)
            if routing_cost != best_routing.get(current) or physical[current] > search_limit:
                continue
            expanded += 1
            if current == right:
                found = True
                break
            start, stop = int(self.offsets[current]), int(self.offsets[current + 1])
            for position in range(start, stop):
                target = int(self.targets[position])
                proposed_physical = physical[current] + float(self.physical_costs[position])
                proposed_routing = routing_cost + float(self.routing_costs[position])
                if proposed_physical > search_limit or proposed_routing >= best_routing.get(target, math.inf):
                    continue
                best_routing[target] = proposed_routing
                physical[target] = proposed_physical
                parent[target] = current
                heapq.heappush(queue, (proposed_routing + self._heuristic(target, right), proposed_routing, target))
        self._increment(expanded_nodes=expanded)
        if not found:
            with self._lock:
                self._path_negative[key] = max(maximum, self._path_negative.get(key, 0.0))
                self._path_negative.move_to_end(key)
                self._trim(self._path_negative)
            return None
        path = [right]
        while path[-1] != left:
            path.append(parent[path[-1]])
        path.reverse()
        middle_physical = max(0.0, physical[right] - float(self.lengths[right]))
        if middle_physical > maximum:
            with self._lock:
                self._path_negative[key] = max(maximum, self._path_negative.get(key, 0.0))
                self._trim(self._path_negative)
            return None
        stored = (tuple(path), float(best_routing[right]), middle_physical)
        with self._lock:
            self._path_positive[key] = stored
            self._path_positive.move_to_end(key)
            self._trim(self._path_positive)
        return [str(self.edge_uids[index]) for index in path], middle_physical

    # Compatibility name used by older callers and external scripts.
    def bridge(self, left_uid: str, right_uid: str, cutoff: float | None = None) -> tuple[list[str], float] | None:
        return self.bridge_path(left_uid, right_uid, cutoff)
