"""Geometry noding primitives for the versioned Stage0 directed network."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import GeometryCollection, MultiPoint, Point
from shapely.ops import split

from stage0.canonical.topology import allows_forward, allows_reverse


TRUE_VALUES = {"1", "t", "true", "y", "yes"}
FALSE_VALUES = {"0", "f", "false", "n", "no", "", "nan", "none", "null", "<na>"}


def parse_bool(value: object) -> bool:
    """Parse road-network booleans without treating non-empty 'F' as true."""

    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise ValueError(f"unsupported boolean encoding: {value!r}")


def normalize_layer(value: object) -> str:
    if value is None:
        return "0"
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null", "<na>"}:
        return "0"
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass
    return text


def topology_level(layer: object, bridge: object, tunnel: object) -> tuple[str, bool, bool]:
    """Return a conservative grade-separation key used for interior crossings."""

    return normalize_layer(layer), parse_bool(bridge), parse_bool(tunnel)


def topology_levels_compatible(left: Sequence[object], right: Sequence[object]) -> bool:
    return tuple(left) == tuple(right)


def grade_transition_connector_eligible(
    left_level: Sequence[object],
    right_level: Sequence[object],
    left_direction: Sequence[float],
    right_direction: Sequence[float],
    left_road_class: object,
    right_road_class: object,
    left_road_name: object = None,
    right_road_name: object = None,
    left_ref: object = None,
    right_ref: object = None,
    maximum_angle_degrees: float = 45.0,
) -> bool:
    """Allow an explicit cross-level terminal connector without merging nodes."""

    if topology_levels_compatible(left_level, right_level):
        return False
    left_class, right_class = str(left_road_class), str(right_road_class)
    same_class = left_class == right_class
    ramp_transition = left_class.endswith("_link") or right_class.endswith("_link")
    same_name = bool(
        left_road_name
        and right_road_name
        and str(left_road_name).lower() not in {"nan", "none"}
        and str(left_road_name) == str(right_road_name)
    )
    same_ref = bool(
        left_ref
        and right_ref
        and str(left_ref).lower() not in {"nan", "none"}
        and str(left_ref) == str(right_ref)
    )
    if not (same_class or ramp_transition or same_name or same_ref):
        return False
    left = np.asarray(left_direction, dtype=float)
    right = np.asarray(right_direction, dtype=float)
    left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
    if left_norm <= 0 or right_norm <= 0:
        return False
    cosine = abs(float(np.dot(left / left_norm, right / right_norm)))
    return cosine >= float(np.cos(np.deg2rad(maximum_angle_degrees)))


def endpoint_access(oneway_code: object, endpoint_is_start: bool) -> tuple[bool, bool]:
    """Return (can_arrive, can_depart) for a physical link endpoint."""

    forward = allows_forward(str(oneway_code))
    reverse = allows_reverse(str(oneway_code))
    if endpoint_is_start:
        return reverse, forward
    return forward, reverse


def connector_traversal_directions(
    left_oneway: object,
    left_endpoint_is_start: bool,
    right_oneway: object,
    right_endpoint_is_start: bool,
) -> tuple[bool, bool]:
    """Return legal left->right and right->left transition directions."""

    left_arrive, left_depart = endpoint_access(left_oneway, left_endpoint_is_start)
    right_arrive, right_depart = endpoint_access(right_oneway, right_endpoint_is_start)
    return left_arrive and right_depart, right_arrive and left_depart


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


def cluster_endpoints_by_level(
    coordinates: np.ndarray,
    levels: Sequence[tuple[str, bool, bool]],
    tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, bool, bool]]]:
    """Cluster endpoints spatially only within an identical topology level."""

    if len(coordinates) != len(levels):
        raise ValueError("coordinates and levels must have the same length")
    if len(coordinates) == 0:
        return np.empty(0, dtype="int64"), np.empty((0, 2), dtype=float), []
    members: dict[tuple[str, bool, bool], list[int]] = defaultdict(list)
    for index, level in enumerate(levels):
        members[tuple(level)].append(index)
    node_ids = np.empty(len(coordinates), dtype="int64")
    representatives: list[np.ndarray] = []
    representative_levels: list[tuple[str, bool, bool]] = []
    offset = 0
    for level in sorted(members, key=repr):
        indices = np.asarray(members[level], dtype="int64")
        local_ids, local_representatives = cluster_endpoints(coordinates[indices], tolerance_m)
        node_ids[indices] = local_ids + offset
        representatives.extend(local_representatives)
        representative_levels.extend([level] * len(local_representatives))
        offset += len(local_representatives)
    return node_ids, np.asarray(representatives, dtype=float), representative_levels
