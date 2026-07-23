"""Compact legal-movement routing shared by HMM transitions and reconstruction.

Every candidate transition is represented by one :class:`TransitionPath`.  The
same path supplies the HMM physical distance and semantic cost and is later
written to the reconstructed route.  This prevents a physical-shortest HMM path
from being silently replaced by a different penalty-shortest output path.
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
class TransitionPath:
    """One cutoff-feasible edge path and both of its cost semantics."""

    edge_uids: tuple[str, ...]
    physical_distance_m: float
    generalized_routing_cost: float
    path_identifier: str


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
        self._path_positive: OrderedDict[tuple[int, int], list[TransitionPath]] = OrderedDict()
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

        raw_records: list[MovementRecord] = []
        raw_rows: list[tuple[int, int, int]] = []
        records: list[MovementRecord] = []
        rows: list[tuple[int, int, float, float, int]] = []
        for row in movements.itertuples(index=False):
            restriction = str(row.restriction_status)
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
            raw_record_index = len(raw_records)
            raw_records.append(record)
            raw_rows.append((left, right, raw_record_index))
            if (
                restriction.startswith("forbidden")
                or restriction == "unresolved_restriction"
                or not bool(row.layer_compatibility)
            ):
                continue
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
        raw_rows.sort(key=lambda value: (value[0], value[1], value[2]))
        self.raw_records = raw_records
        self.raw_offsets = np.zeros(len(edge_frame) + 1, dtype="int64")
        for left, *_ in raw_rows:
            self.raw_offsets[left + 1] += 1
        np.cumsum(self.raw_offsets, out=self.raw_offsets)
        self.raw_targets = np.asarray([row[1] for row in raw_rows], dtype="int32")
        self.raw_record_indices = np.asarray([row[2] for row in raw_rows], dtype="int32")
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

    def raw_movement(self, left_uid: str, right_uid: str) -> MovementRecord | None:
        """Return an audited movement even when it is forbidden or unresolved for routing."""
        left = self.uid_to_index.get(str(left_uid))
        right = self.uid_to_index.get(str(right_uid))
        if left is None or right is None:
            return None
        start, stop = int(self.raw_offsets[left]), int(self.raw_offsets[left + 1])
        positions = np.flatnonzero(self.raw_targets[start:stop] == right)
        if not len(positions):
            return None
        return self.raw_records[int(self.raw_record_indices[start + int(positions[0])])]

    @staticmethod
    def _path_id(edge_uids: tuple[str, ...]) -> str:
        import hashlib

        return hashlib.sha256("\x1f".join(edge_uids).encode("utf-8")).hexdigest()[:24]

    def _direct_transition_path(self, left_uid: str, right_uid: str) -> TransitionPath | None:
        if left_uid == right_uid:
            path = (str(left_uid),)
            return TransitionPath(path, 0.0, 0.0, self._path_id(path))
        movement = self.movement(left_uid, right_uid)
        if movement is None:
            return None
        right = self.uid_to_index[str(right_uid)]
        start, stop = int(self.offsets[self.uid_to_index[str(left_uid)]]), int(
            self.offsets[self.uid_to_index[str(left_uid)] + 1]
        )
        positions = np.flatnonzero(self.targets[start:stop] == right)
        if not len(positions):
            return None
        position = start + int(positions[0])
        path = (str(left_uid), str(right_uid))
        # Intermediate physical distance excludes the target edge; candidate
        # positions are added by TransitionEngine.
        return TransitionPath(
            path,
            0.0,
            max(0.0, float(self.routing_costs[position]) - float(self.lengths[right])),
            self._path_id(path),
        )

    def transition_path(
        self, left_uid: str, right_uid: str, cutoff: float | None = None
    ) -> TransitionPath | None:
        """Return the minimum generalized-cost path under a physical cutoff.

        A Pareto frontier over ``(physical distance, generalized cost)`` is
        maintained for every edge state.  A single-label A* is not valid here:
        a slightly more expensive but much shorter prefix may be the only path
        capable of satisfying the downstream physical-distance budget.
        """

        direct = self._direct_transition_path(str(left_uid), str(right_uid))
        if direct is not None:
            return direct
        left = self.uid_to_index.get(str(left_uid))
        right = self.uid_to_index.get(str(right_uid))
        if left is None or right is None:
            return None
        maximum = min(self.maximum, float(cutoff) if cutoff is not None else self.maximum)
        key = (left, right)
        with self._lock:
            cached_paths = self._path_positive.get(key, [])
            feasible_cached = [path for path in cached_paths if path.physical_distance_m <= maximum]
            negative_cutoff = self._path_negative.get(key, -math.inf)
            if feasible_cached:
                chosen = min(
                    feasible_cached,
                    key=lambda path: (
                        path.generalized_routing_cost,
                        path.physical_distance_m,
                        path.path_identifier,
                    ),
                )
                self._path_positive.move_to_end(key)
                self._increment(path_cache_hits=1)
                return chosen
            if maximum <= negative_cutoff:
                self._path_negative.move_to_end(key)
                self._increment(negative_cache_hits=1)
                return None

        self._increment(path_calls=1, cache_misses=1)
        search_limit = maximum + float(self.lengths[right])
        # label = (generalized cost, physical distance, state, parent_label_id)
        label_state: list[int] = [left]
        label_parent: list[int] = [-1]
        label_physical: list[float] = [0.0]
        label_cost: list[float] = [0.0]
        frontiers: dict[int, list[int]] = {left: [0]}
        queue: list[tuple[float, float, int]] = [
            (self._heuristic(left, right), 0.0, 0)
        ]
        expanded = 0
        chosen_label: int | None = None
        while queue:
            _, queued_cost, label_id = heapq.heappop(queue)
            if queued_cost != label_cost[label_id]:
                continue
            current = label_state[label_id]
            expanded += 1
            if current == right:
                chosen_label = label_id
                break
            start, stop = int(self.offsets[current]), int(self.offsets[current + 1])
            for position in range(start, stop):
                target = int(self.targets[position])
                proposed_physical = label_physical[label_id] + float(self.physical_costs[position])
                if proposed_physical > search_limit:
                    continue
                proposed_cost = label_cost[label_id] + float(self.routing_costs[position])
                target_frontier = frontiers.setdefault(target, [])
                if any(
                    label_physical[existing] <= proposed_physical
                    and label_cost[existing] <= proposed_cost
                    for existing in target_frontier
                ):
                    continue
                target_frontier[:] = [
                    existing
                    for existing in target_frontier
                    if not (
                        proposed_physical <= label_physical[existing]
                        and proposed_cost <= label_cost[existing]
                    )
                ]
                new_label = len(label_state)
                label_state.append(target)
                label_parent.append(label_id)
                label_physical.append(proposed_physical)
                label_cost.append(proposed_cost)
                target_frontier.append(new_label)
                heapq.heappush(
                    queue,
                    (
                        proposed_cost + self._heuristic(target, right),
                        proposed_cost,
                        new_label,
                    ),
                )
        self._increment(expanded_nodes=expanded)
        if chosen_label is None:
            with self._lock:
                self._path_negative[key] = max(maximum, self._path_negative.get(key, 0.0))
                self._path_negative.move_to_end(key)
                self._trim(self._path_negative)
            return None

        indices: list[int] = []
        cursor = chosen_label
        while cursor >= 0:
            indices.append(label_state[cursor])
            cursor = label_parent[cursor]
        indices.reverse()
        middle_physical = max(
            0.0, label_physical[chosen_label] - float(self.lengths[right])
        )
        edge_uids = tuple(str(self.edge_uids[index]) for index in indices)
        result = TransitionPath(
            edge_uids=edge_uids,
            physical_distance_m=middle_physical,
            generalized_routing_cost=float(label_cost[chosen_label]),
            path_identifier=self._path_id(edge_uids),
        )
        with self._lock:
            existing = self._path_positive.setdefault(key, [])
            if all(path.path_identifier != result.path_identifier for path in existing):
                existing.append(result)
            self._path_positive.move_to_end(key)
            self._trim(self._path_positive)
        return result

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
        """Compatibility API backed by the canonical transition path."""
        result = self.transition_path(left_uid, right_uid, cutoff)
        if result is None:
            return None
        return list(result.edge_uids), result.physical_distance_m

    # Compatibility name used by older callers and external scripts.
    def bridge(self, left_uid: str, right_uid: str, cutoff: float | None = None) -> tuple[list[str], float] | None:
        return self.bridge_path(left_uid, right_uid, cutoff)
