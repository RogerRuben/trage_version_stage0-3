"""Compact legal-movement routing shared by HMM transitions and reconstruction."""

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
    calls: int
    expanded_nodes: int
    cache_hits: int
    cache_misses: int

    def minus(self, earlier: "SearchStats") -> "SearchStats":
        return SearchStats(
            self.calls - earlier.calls,
            self.expanded_nodes - earlier.expanded_nodes,
            self.cache_hits - earlier.cache_hits,
            self.cache_misses - earlier.cache_misses,
        )


class CompactMovementRouter:
    """CSR-like edge-state graph with bounded A* and a path LRU.

    Movement rows are indexed once.  No order-level pandas indexing and no dense
    all-node distance vectors are created.
    """

    def __init__(
        self,
        edges: gpd.GeoDataFrame,
        movements: pd.DataFrame,
        config: dict[str, Any],
    ) -> None:
        edge_frame = edges.reset_index(drop=True)
        self.edge_uids = edge_frame.edge_uid.astype(str).to_numpy()
        self.uid_to_index = {uid: index for index, uid in enumerate(self.edge_uids)}
        self.lengths = edge_frame.length_m.to_numpy(float)
        self.maximum = float(config.get("max_route_distance_m", 6000.0))
        self.cache_limit = int(config.get("bridge_cache_pairs", 50_000))
        self._lock = threading.RLock()
        self._cache: OrderedDict[tuple[int, int, int], tuple[tuple[int, ...] | None, float]] = OrderedDict()
        self._calls = self._expanded = self._hits = self._misses = 0

        starts = np.empty((len(edge_frame), 2), dtype="float64")
        ends = np.empty((len(edge_frame), 2), dtype="float64")
        for index, geometry in enumerate(edge_frame.geometry):
            coordinates = np.asarray(geometry.coords, dtype=float)
            starts[index] = coordinates[0]
            ends[index] = coordinates[-1]
        self.start_xy = starts
        self.end_xy = ends

        records: list[MovementRecord] = []
        rows: list[tuple[int, int, float, int]] = []
        for row in movements.itertuples(index=False):
            restriction = str(row.restriction_status)
            if restriction.startswith("forbidden") or not bool(row.layer_compatibility):
                continue
            left = self.uid_to_index.get(str(row.from_edge_uid))
            right = self.uid_to_index.get(str(row.to_edge_uid))
            if left is None or right is None:
                continue
            transition_type = str(getattr(row, "level_transition_type", "same_level"))
            record = MovementRecord(
                from_edge_uid=str(row.from_edge_uid),
                via_node=int(row.via_node),
                to_edge_uid=str(row.to_edge_uid),
                movement_type=str(row.movement_type),
                turn_angle=float(row.turn_angle),
                restriction_status=restriction,
                layer_compatibility=bool(row.layer_compatibility),
                level_transition_type=transition_type,
                road_class_transition=str(row.road_class_transition),
                merge_diverge_flag=bool(row.merge_diverge_flag),
            )
            record_index = len(records)
            records.append(record)
            cost = float(self.lengths[right])
            if record.movement_type == "u_turn":
                cost += float(config.get("u_turn_penalty_m", 500.0))
            classes = record.road_class_transition.split("->")
            if len(classes) == 2 and classes[0] != classes[1]:
                cost += float(config.get("road_class_transition_penalty_m", 30.0))
            rows.append((left, right, cost, record_index))
        rows.sort(key=lambda value: (value[0], value[1], value[2]))
        self.records = records
        self.offsets = np.zeros(len(edge_frame) + 1, dtype="int64")
        for left, _, _, _ in rows:
            self.offsets[left + 1] += 1
        np.cumsum(self.offsets, out=self.offsets)
        self.targets = np.asarray([row[1] for row in rows], dtype="int32")
        self.costs = np.asarray([row[2] for row in rows], dtype="float32")
        self.record_indices = np.asarray([row[3] for row in rows], dtype="int32")

    def stats(self) -> SearchStats:
        with self._lock:
            return SearchStats(self._calls, self._expanded, self._hits, self._misses)

    @property
    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)

    def movement(self, left_uid: str, right_uid: str) -> MovementRecord | None:
        left = self.uid_to_index.get(str(left_uid))
        right = self.uid_to_index.get(str(right_uid))
        if left is None or right is None:
            return None
        start, stop = int(self.offsets[left]), int(self.offsets[left + 1])
        for position in range(start, stop):
            if int(self.targets[position]) == right:
                return self.records[int(self.record_indices[position])]
        return None

    def _heuristic(self, edge_index: int, target_index: int) -> float:
        difference = self.end_xy[edge_index] - self.start_xy[target_index]
        return float(np.hypot(difference[0], difference[1]))

    def bridge(self, left_uid: str, right_uid: str, cutoff: float | None = None) -> tuple[list[str], float] | None:
        left = self.uid_to_index.get(str(left_uid))
        right = self.uid_to_index.get(str(right_uid))
        if left is None or right is None:
            return None
        if left == right:
            return [str(left_uid)], 0.0
        if self.movement(str(left_uid), str(right_uid)) is not None:
            return [str(left_uid), str(right_uid)], float(self.lengths[right])
        maximum = min(self.maximum, float(cutoff) if cutoff is not None else self.maximum)
        cutoff_bucket = int(max(250.0, math.ceil(maximum / 250.0) * 250.0))
        key = (left, right, cutoff_bucket)
        with self._lock:
            self._calls += 1
            cached = self._cache.get(key)
            if cached is not None:
                self._hits += 1
                self._cache.move_to_end(key)
                path, cost = cached
                return None if path is None else ([str(self.edge_uids[index]) for index in path], cost)
            self._misses += 1

        queue: list[tuple[float, float, int]] = [(self._heuristic(left, right), 0.0, left)]
        distances = {left: 0.0}
        parent: dict[int, int] = {}
        expanded = 0
        found = False
        while queue:
            _, current_cost, current = heapq.heappop(queue)
            if current_cost != distances.get(current):
                continue
            if current_cost > maximum:
                break
            expanded += 1
            if current == right:
                found = True
                break
            start, stop = int(self.offsets[current]), int(self.offsets[current + 1])
            for position in range(start, stop):
                target = int(self.targets[position])
                proposed = current_cost + float(self.costs[position])
                if proposed > maximum or proposed >= distances.get(target, math.inf):
                    continue
                distances[target] = proposed
                parent[target] = current
                heapq.heappush(queue, (proposed + self._heuristic(target, right), proposed, target))
        with self._lock:
            self._expanded += expanded
        if found:
            path_indices = [right]
            while path_indices[-1] != left:
                path_indices.append(parent[path_indices[-1]])
            path_indices.reverse()
            stored_path: tuple[int, ...] | None = tuple(path_indices)
            cost = float(distances[right])
        else:
            stored_path = None
            cost = math.inf
        with self._lock:
            self._cache[key] = (stored_path, cost)
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_limit:
                self._cache.popitem(last=False)
        return None if stored_path is None else ([str(self.edge_uids[index]) for index in stored_path], cost)

    def multi_target_bridges(
        self,
        left_uids: Iterable[str],
        right_uids: Iterable[str],
        cutoff: float,
    ) -> dict[tuple[str, str], tuple[list[str], float] | None]:
        """Run one bounded early-terminating search per source for requested targets."""
        result: dict[tuple[str, str], tuple[list[str], float] | None] = {}
        unique_right = list(dict.fromkeys(map(str, right_uids)))
        maximum = min(self.maximum, float(cutoff))
        cutoff_bucket = int(max(250.0, math.ceil(maximum / 250.0) * 250.0))
        for left in dict.fromkeys(map(str, left_uids)):
            left_index = self.uid_to_index.get(left)
            if left_index is None:
                continue
            missing: dict[int, str] = {}
            for right in unique_right:
                if left == right:
                    result[(left, right)] = ([left], 0.0)
                elif self.movement(left, right) is not None:
                    result[(left, right)] = ([left, right], float(self.lengths[self.uid_to_index[right]]))
                else:
                    right_index = self.uid_to_index.get(right)
                    if right_index is None:
                        result[(left, right)] = None
                        continue
                    key = (left_index, right_index, cutoff_bucket)
                    with self._lock:
                        cached = self._cache.get(key)
                        if cached is not None:
                            self._hits += 1
                            self._cache.move_to_end(key)
                            path, cost = cached
                            result[(left, right)] = None if path is None else (
                                [str(self.edge_uids[index]) for index in path], cost
                            )
                            continue
                    missing[right_index] = right
            if not missing:
                continue
            with self._lock:
                self._calls += 1
                self._misses += len(missing)
            queue: list[tuple[float, int]] = [(0.0, left_index)]
            distances = {left_index: 0.0}
            parent: dict[int, int] = {}
            remaining = set(missing)
            expanded = 0
            while queue and remaining:
                current_cost, current = heapq.heappop(queue)
                if current_cost != distances.get(current) or current_cost > maximum:
                    continue
                expanded += 1
                remaining.discard(current)
                start, stop = int(self.offsets[current]), int(self.offsets[current + 1])
                for position in range(start, stop):
                    target = int(self.targets[position])
                    proposed = current_cost + float(self.costs[position])
                    if proposed > maximum or proposed >= distances.get(target, math.inf):
                        continue
                    distances[target] = proposed
                    parent[target] = current
                    heapq.heappush(queue, (proposed, target))
            with self._lock:
                self._expanded += expanded
            for target_index, right in missing.items():
                if target_index in distances and target_index not in remaining:
                    path_indices = [target_index]
                    while path_indices[-1] != left_index:
                        path_indices.append(parent[path_indices[-1]])
                    path_indices.reverse()
                    stored: tuple[int, ...] | None = tuple(path_indices)
                    cost = float(distances[target_index])
                    result[(left, right)] = ([str(self.edge_uids[index]) for index in stored], cost)
                else:
                    stored = None
                    cost = math.inf
                    result[(left, right)] = None
                with self._lock:
                    self._cache[(left_index, target_index, cutoff_bucket)] = (stored, cost)
                    self._cache.move_to_end((left_index, target_index, cutoff_bucket))
                    while len(self._cache) > self.cache_limit:
                        self._cache.popitem(last=False)
        return result
