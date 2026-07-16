"""Geometry noding primitives for the versioned Stage0 directed network."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import GeometryCollection, MultiPoint, Point
from shapely.ops import split


def topology_level(layer: object, bridge: object, tunnel: object) -> tuple[str, bool, bool]:
    """Return a conservative grade-separation key used for interior crossings."""

    layer_value = "0" if layer is None or str(layer).lower() in {"nan", "none", ""} else str(layer)
    return layer_value, bool(bridge), bool(tunnel)


def intersection_points(geometry: object) -> list[Point]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Point):
        return [geometry]
    if isinstance(geometry, MultiPoint):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        points: list[Point] = []
        for part in geometry.geoms:
            points.extend(intersection_points(part))
        return points
    return []


def split_line_at_points(line, points: Iterable[Point], endpoint_tolerance_m: float = 0.05):
    projected = []
    seen = set()
    for point in points:
        distance = float(line.project(point))
        if distance <= endpoint_tolerance_m or distance >= float(line.length) - endpoint_tolerance_m:
            continue
        key = round(distance, 6)
        if key in seen:
            continue
        seen.add(key)
        projected.append(line.interpolate(distance))
    if not projected:
        return [line]
    pieces = split(line, MultiPoint(projected))
    return [part for part in pieces.geoms if part.length > endpoint_tolerance_m]


class UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype="int64")

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = int(self.parent[value])
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def cluster_endpoints(coordinates: np.ndarray, tolerance_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Cluster nearly identical endpoints and return node id and representative XY."""

    if len(coordinates) == 0:
        return np.empty(0, dtype="int64"), np.empty((0, 2), dtype=float)
    union = UnionFind(len(coordinates))
    for left, right in cKDTree(coordinates).query_pairs(tolerance_m):
        union.union(int(left), int(right))
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(coordinates)):
        groups[union.find(index)].append(index)
    ordered_roots = sorted(groups, key=lambda root: min(groups[root]))
    node_for_root = {root: node_id for node_id, root in enumerate(ordered_roots)}
    representatives = np.vstack([coordinates[groups[root]].mean(axis=0) for root in ordered_roots])
    node_ids = np.array([node_for_root[union.find(index)] for index in range(len(coordinates))], dtype="int64")
    return node_ids, representatives

