"""Final Stage 0 route-segment and four-axis quality products."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer
from shapely import from_wkb
from shapely.geometry import LineString
from shapely.ops import nearest_points, substring, unary_union


GPS_STATUSES = {"clean", "local_outlier", "sparse_or_ineligible", "unresolved_gap"}
ROUTE_STATUSES = {"route_pass", "route_partial", "route_fail", "route_uncertain"}
DYNAMIC_STATUSES = {"dynamic_strict", "dynamic_partial", "dynamic_unusable"}
CANONICAL_STATUSES = {"unique", "chain_resolved", "ambiguous", "unmapped"}


@dataclass(frozen=True)
class FinalQualityResult:
    route_segments: pd.DataFrame
    order_quality: dict[str, Any]
    point_route_distances: pd.DataFrame


class CanonicalGeometryStore:
    """Read canonical geometry once and reuse it for all orders."""

    def __init__(self, canonical_edges_path: str | Path) -> None:
        table = pq.read_table(
            canonical_edges_path,
            columns=["edge_uid", "geometry", "length_m"],
        ).to_pandas()
        self.geometry = {
            str(row.edge_uid): from_wkb(row.geometry)
            for row in table.itertuples(index=False)
            if row.geometry is not None
        }
        self.length = {
            str(edge_uid): float(length_m)
            for edge_uid, length_m in table[
                ["edge_uid", "length_m"]
            ].itertuples(index=False, name=None)
        }
        self.transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:32649", always_xy=True
        )
        self._clip_cache: dict[tuple[Any, ...], Any] = {}
        self._union_cache: dict[tuple[Any, ...], Any] = {}

    @staticmethod
    def _value(route: Any, name: str, default: Any = None) -> Any:
        if isinstance(route, dict):
            return route.get(name, default)
        return getattr(route, name, default)

    def _clip_key(self, route: Any) -> tuple[Any, ...]:
        def rounded(name: str) -> float | None:
            value = pd.to_numeric(
                pd.Series([self._value(route, name, np.nan)]),
                errors="coerce",
            ).iloc[0]
            return round(float(value), 6) if pd.notna(value) else None

        return (
            str(self._value(route, "canonical_edge_uid", "")),
            rounded("canonical_length_m"),
            rounded("entry_position_m"),
            rounded("exit_position_m"),
        )

    def clip_route_part(self, route: Any) -> Any | None:
        edge_uid = self._value(route, "canonical_edge_uid", None)
        if edge_uid is None or pd.isna(edge_uid):
            return None
        cache_key = self._clip_key(route)
        cached = self._clip_cache.get(cache_key)
        if cached is not None:
            return cached
        geometry = self.geometry.get(str(edge_uid))
        if geometry is None:
            return None
        canonical_length = float(
            self._value(
                route,
                "canonical_length_m",
                self.length.get(str(edge_uid), geometry.length),
            )
        )
        entry = pd.to_numeric(
            pd.Series([self._value(route, "entry_position_m", np.nan)]),
            errors="coerce",
        ).iloc[0]
        exit_ = pd.to_numeric(
            pd.Series([self._value(route, "exit_position_m", np.nan)]),
            errors="coerce",
        ).iloc[0]
        if (
            geometry.geom_type == "LineString"
            and canonical_length > 0
            and pd.notna(entry)
            and pd.notna(exit_)
        ):
            clipped = substring(
                geometry,
                float(np.clip(entry / canonical_length, 0, 1)),
                float(np.clip(exit_ / canonical_length, 0, 1)),
                normalized=True,
            )
        else:
            clipped = geometry
        if len(self._clip_cache) >= 200_000:
            self._clip_cache.clear()
        self._clip_cache[cache_key] = clipped
        return clipped

    def union_route_parts(self, routes: pd.DataFrame) -> Any | None:
        ordered = routes.sort_values("route_sequence", kind="stable")
        records = list(ordered.itertuples(index=False))
        cache_key = tuple(self._clip_key(route) for route in records)
        if cache_key in self._union_cache:
            return self._union_cache[cache_key]
        geometries = [self.clip_route_part(route) for route in records]
        valid = [geometry for geometry in geometries if geometry is not None]
        union = unary_union(valid) if valid else None
        if len(self._union_cache) >= 50_000:
            self._union_cache.clear()
        self._union_cache[cache_key] = union
        return union

    def point_coordinates(self, points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        lon_column = "matching_lon" if "matching_lon" in points else "lon"
        lat_column = "matching_lat" if "matching_lat" in points else "lat"
        return self.transformer.transform(
            pd.to_numeric(points[lon_column], errors="coerce").to_numpy(float),
            pd.to_numeric(points[lat_column], errors="coerce").to_numpy(float),
        )


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _share(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _angle_difference(left: float, right: float) -> float:
    difference = abs((left - right) % 360.0)
    return min(difference, 360.0 - difference)


def _heading(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dy, dx)) % 360.0


def _line_endpoints(geometry: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if geometry is None or geometry.is_empty:
        return None
    if geometry.geom_type == "LineString":
        coordinates = np.asarray(geometry.coords, dtype=float)
        if len(coordinates):
            return coordinates[0], coordinates[-1]
    if geometry.geom_type == "MultiLineString":
        parts = [part for part in geometry.geoms if not part.is_empty]
        if parts:
            return (
                np.asarray(parts[0].coords[0], dtype=float),
                np.asarray(parts[-1].coords[-1], dtype=float),
            )
    return None


def _parallel_features(
    xy: np.ndarray,
    geometry: Any | None,
    distances: np.ndarray,
    duration_s: float,
    gps_distance_m: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "parallel_offset_mean_m": np.nan,
        "parallel_offset_std_m": np.nan,
        "parallel_offset_same_sign_share": 0.0,
        "parallel_heading_difference_deg": np.nan,
        "parallel_window_time_s": 0.0,
        "parallel_window_distance_m": 0.0,
        "parallel_risk": False,
    }
    finite = np.isfinite(distances)
    if geometry is None or finite.sum() < 3 or len(xy) < 3:
        return result
    endpoints = _line_endpoints(geometry)
    if endpoints is None:
        return result
    gps_vector = xy[-1] - xy[0]
    route_start = np.asarray(
        nearest_points(shapely.Point(xy[0]), geometry)[1].coords[0],
        dtype=float,
    )
    route_end = np.asarray(
        nearest_points(shapely.Point(xy[-1]), geometry)[1].coords[0],
        dtype=float,
    )
    route_vector = route_end - route_start
    if np.linalg.norm(gps_vector) < 5:
        return result
    if np.linalg.norm(route_vector) < 5:
        # A union of clipped parallel links can make both nearest projections
        # land on the same junction. The smooth GPS tangent remains a
        # conservative local tangent for testing persistent lateral offset.
        route_vector = gps_vector.copy()
    route_heading = _heading(route_vector[0], route_vector[1])
    gps_heading = _heading(gps_vector[0], gps_vector[1])
    directional_difference = _angle_difference(route_heading, gps_heading)
    heading_difference = min(
        directional_difference, abs(180.0 - directional_difference)
    )
    signs = []
    for point in xy:
        nearest = np.asarray(
            nearest_points(shapely.Point(point), geometry)[1].coords[0],
            dtype=float,
        )
        cross = route_vector[0] * (point[1] - nearest[1]) - route_vector[1] * (
            point[0] - nearest[0]
        )
        signs.append(np.sign(cross))
    signs_array = np.asarray(signs)
    nonzero = signs_array[signs_array != 0]
    same_sign_share = (
        float(max((nonzero > 0).mean(), (nonzero < 0).mean()))
        if len(nonzero)
        else 0.0
    )
    mean_offset = float(np.nanmean(distances))
    std_offset = float(np.nanstd(distances))
    risk = bool(
        mean_offset >= float(settings.get("parallel_minimum_offset_m", 20.0))
        and std_offset <= float(settings.get("parallel_maximum_offset_std_m", 18.0))
        and same_sign_share
        >= float(settings.get("parallel_minimum_same_sign_share", 0.80))
        and heading_difference
        <= float(settings.get("parallel_maximum_heading_difference_deg", 25.0))
        and (
            duration_s >= float(settings.get("parallel_minimum_time_s", 45.0))
            or gps_distance_m
            >= float(settings.get("parallel_minimum_distance_m", 250.0))
        )
    )
    return {
        "parallel_offset_mean_m": mean_offset,
        "parallel_offset_std_m": std_offset,
        "parallel_offset_same_sign_share": same_sign_share,
        "parallel_heading_difference_deg": heading_difference,
        "parallel_window_time_s": duration_s if risk else 0.0,
        "parallel_window_distance_m": gps_distance_m if risk else 0.0,
        "parallel_risk": risk,
    }


def _parallel_window_features(
    points: pd.DataFrame,
    xy: np.ndarray,
    geometry: Any | None,
    point_geometries: list[Any | None],
    distances: np.ndarray,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Find a sustained, smooth offset run instead of averaging an order."""

    empty = {
        "parallel_offset_mean_m": np.nan,
        "parallel_offset_std_m": np.nan,
        "parallel_offset_same_sign_share": 0.0,
        "parallel_heading_difference_deg": np.nan,
        "parallel_window_time_s": 0.0,
        "parallel_window_distance_m": 0.0,
        "parallel_risk": False,
    }
    minimum_offset = float(settings.get("parallel_minimum_offset_m", 20.0))
    candidate = np.isfinite(distances) & (distances >= minimum_offset)
    runs: list[tuple[int, int]] = []
    start = None
    for index, active in enumerate([*candidate.tolist(), False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start >= int(
                settings.get("parallel_minimum_point_count", 5)
            ):
                runs.append((start, index))
            start = None
    best: dict[str, Any] | None = None
    for start, end in runs:
        window_points = points.iloc[start:end]
        window_distances = distances[start:end]
        window_geometry_values = [
            item
            for item in point_geometries[start:end]
            if item is not None and not item.is_empty
        ]
        window_geometry = (
            unary_union(window_geometry_values)
            if window_geometry_values
            else geometry
        )
        duration = float(
            window_points.timestamp.max() - window_points.timestamp.min()
        )
        gps_distance = float(
            _numeric(window_points, "step_distance_m").fillna(0).sum()
        )
        features = _parallel_features(
            xy[start:end],
            window_geometry,
            window_distances,
            duration,
            gps_distance,
            {
                **settings,
                "parallel_minimum_time_s": float(
                    settings.get("parallel_window_minimum_time_s", 15.0)
                ),
                "parallel_minimum_distance_m": float(
                    settings.get("parallel_window_minimum_distance_m", 50.0)
                ),
            },
        )
        nearest_xy: list[np.ndarray] = []
        for local_index, point in enumerate(xy[start:end]):
            point_geometry = point_geometries[start + local_index]
            if point_geometry is None or point_geometry.is_empty:
                point_geometry = window_geometry
            if point_geometry is None or point_geometry.is_empty:
                nearest_xy = []
                break
            nearest = nearest_points(
                shapely.Point(point), point_geometry
            )[1]
            nearest_xy.append(
                np.asarray(nearest.coords[0], dtype=float)
            )
        if len(nearest_xy) >= 2:
            nearest_array = np.asarray(nearest_xy)
            gps_vector = xy[end - 1] - xy[start]
            route_vector = nearest_array[-1] - nearest_array[0]
            if np.linalg.norm(route_vector) < 5 and len(nearest_array) >= 3:
                centered = nearest_array - nearest_array.mean(axis=0)
                _, _, axes = np.linalg.svd(centered, full_matrices=False)
                route_vector = axes[0]
                if np.dot(route_vector, gps_vector) < 0:
                    route_vector = -route_vector
            if np.linalg.norm(gps_vector) >= 5 and np.linalg.norm(route_vector) > 0:
                directional_difference = _angle_difference(
                    _heading(gps_vector[0], gps_vector[1]),
                    _heading(route_vector[0], route_vector[1]),
                )
                features["parallel_heading_difference_deg"] = min(
                    directional_difference,
                    abs(180.0 - directional_difference),
                )
                signs = np.sign(
                    route_vector[0]
                    * (xy[start:end, 1] - nearest_array[:, 1])
                    - route_vector[1]
                    * (xy[start:end, 0] - nearest_array[:, 0])
                )
                nonzero = signs[signs != 0]
                features["parallel_offset_same_sign_share"] = (
                    float(max((nonzero > 0).mean(), (nonzero < 0).mean()))
                    if len(nonzero)
                    else 0.0
                )
                features["parallel_offset_mean_m"] = float(
                    np.nanmean(window_distances)
                )
                features["parallel_offset_std_m"] = float(
                    np.nanstd(window_distances)
                )
        stable_parallel = bool(
            float(np.nanmean(window_distances)) >= minimum_offset
            and float(np.nanstd(window_distances))
            <= float(settings.get("parallel_maximum_offset_std_m", 18.0))
            and (
                duration
                >= float(
                    settings.get("parallel_window_minimum_time_s", 15.0)
                )
                or gps_distance
                >= float(
                    settings.get(
                        "parallel_window_minimum_distance_m", 50.0
                    )
                )
            )
            and math.isfinite(
                float(features["parallel_heading_difference_deg"])
            )
            and float(features["parallel_heading_difference_deg"])
            <= float(
                settings.get(
                    "parallel_relaxed_maximum_heading_difference_deg", 60.0
                )
            )
            and float(features["parallel_offset_same_sign_share"])
            >= float(
                settings.get(
                    "parallel_relaxed_minimum_same_sign_share", 0.60
                )
            )
        )
        if not stable_parallel:
            continue
        features["parallel_risk"] = True
        features["parallel_window_time_s"] = duration
        features["parallel_window_distance_m"] = gps_distance
        if best is None or (
            features["parallel_window_time_s"],
            features["parallel_window_distance_m"],
        ) > (
            best["parallel_window_time_s"],
            best["parallel_window_distance_m"],
        ):
            best = features
    return best or empty


def _longest_uncovered(
    points: pd.DataFrame, uncovered: np.ndarray
) -> tuple[int, float, float]:
    best = (0, 0.0, 0.0)
    start = None
    for index, value in enumerate([*uncovered.tolist(), False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            end = index - 1
            count = end - start + 1
            time_s = float(
                points.timestamp.iloc[end] - points.timestamp.iloc[start]
            )
            distance_m = float(
                _numeric(points.iloc[start : end + 1], "step_distance_m")
                .fillna(0)
                .sum()
            )
            candidate = (count, max(time_s, 0.0), distance_m)
            if (candidate[2], candidate[1], candidate[0]) > (
                best[2],
                best[1],
                best[0],
            ):
                best = candidate
            start = None
    return best


def _canonical_status(routes: pd.DataFrame) -> str:
    if routes.empty or routes.canonical_edge_uid.isna().all():
        return "unmapped"
    statuses = routes.get(
        "mapping_status", pd.Series("", index=routes.index)
    ).astype(str)
    if statuses.str.contains("ambiguous").any():
        return "ambiguous"
    if routes.canonical_edge_uid.isna().any() or statuses.eq("unmapped").any():
        return "unmapped"
    if (
        statuses.str.contains("way_and_node|reverse_oneway").any()
        or routes.groupby("valhalla_edge_index").canonical_edge_uid.nunique().gt(1).any()
    ):
        return "chain_resolved"
    return "unique"


def _dynamic_status(intervals: pd.DataFrame) -> str:
    if intervals.empty:
        return "dynamic_unusable"
    durations = _numeric(intervals, "interval_duration_s").fillna(0)
    total = float(durations.sum())
    direct = float(
        durations.loc[intervals.measurement_source.eq("direct_observed")].sum()
    )
    unresolved = float(
        durations.loc[intervals.measurement_source.eq("unresolved")].sum()
    )
    if total > 0 and direct / total >= 0.70 and unresolved / total <= 0.20:
        return "dynamic_strict"
    if (
        intervals.measurement_source.isin(
            ["direct_observed", "interval_supported"]
        ).any()
    ):
        return "dynamic_partial"
    return "dynamic_unusable"


def _segment_route_status(
    *,
    has_route: bool,
    buffer40: float,
    p90: float,
    uncovered_distance_share: float,
    parallel: dict[str, Any],
    duration_s: float,
    gps_distance_m: float,
    has_gap: bool,
    settings: dict[str, Any],
) -> tuple[str, str]:
    if not has_route:
        return "route_fail", "NO_CORRESPONDING_ROUTE_COMPONENT"
    if (
        buffer40 < float(settings.get("segment_fail_minimum_buffer40_share", 0.35))
        or uncovered_distance_share
        > float(settings.get("segment_fail_maximum_uncovered_distance_share", 0.55))
    ):
        return "route_fail", "MAJOR_RAW_GPS_ROUTE_MISMATCH"
    if parallel["parallel_risk"]:
        if (
            parallel["parallel_offset_mean_m"]
            >= float(settings.get("parallel_fail_offset_m", 40.0))
            and gps_distance_m
            >= float(settings.get("parallel_fail_distance_m", 400.0))
        ):
            return "route_fail", "SUSTAINED_PARALLEL_CORRIDOR_MISMATCH"
        if gps_distance_m >= float(
            settings.get("parallel_uncertain_distance_m", 250.0)
        ):
            return "route_uncertain", "PARALLEL_ROAD_IDENTITY_UNCERTAIN"
        return "route_partial", "LOCAL_PARALLEL_CORRIDOR_MISMATCH"
    if (
        buffer40 < float(settings.get("segment_pass_minimum_buffer40_share", 0.85))
        or uncovered_distance_share
        > float(settings.get("segment_pass_maximum_uncovered_distance_share", 0.15))
        or (math.isfinite(p90) and p90 > 40.0)
    ):
        return "route_partial", "LOCAL_ROUTE_COVERAGE_GAP"
    return "route_pass", ""


def build_final_quality(
    source_points: pd.DataFrame,
    matched_points: pd.DataFrame,
    route_parts: pd.DataFrame,
    interval_measurements: pd.DataFrame,
    geometry_store: CanonicalGeometryStore,
    eligibility: dict[str, Any],
    settings: dict[str, Any],
    *,
    local_retry: dict[str, Any] | None = None,
) -> FinalQualityResult:
    """Build segment quality and order-level four-axis final status."""

    source = source_points.sort_values(
        ["timestamp", "original_point_seq"], kind="stable"
    ).reset_index(drop=True).copy()
    order_id = str(source.order_id.iloc[0])
    matched = matched_points[
        [
            column
            for column in [
                "subtrace_id",
                "original_point_seq",
                "edge_index",
                "matched_point_status",
                "route_discontinuity",
            ]
            if column in matched_points
        ]
    ].copy()
    source = source.merge(
        matched,
        on=["subtrace_id", "original_point_seq"],
        how="left",
        validate="one_to_one",
    )
    source["matched_point_status"] = source.get(
        "matched_point_status", pd.Series("unmatched", index=source.index)
    ).fillna("unmatched")
    source["route_discontinuity"] = source.get(
        "route_discontinuity", pd.Series(False, index=source.index)
    ).fillna(False)
    source["gps_outlier"] = source.get(
        "gps_outlier", pd.Series(False, index=source.index)
    ).fillna(False)

    route_primary = route_parts.loc[
        pd.to_numeric(
            route_parts.get(
                "valhalla_path_id", pd.Series(0, index=route_parts.index)
            ),
            errors="coerce",
        ).fillna(0).eq(0)
    ].copy()
    edge_component = (
        route_primary.dropna(subset=["valhalla_edge_index"])
        .groupby(["subtrace_id", "valhalla_edge_index"], sort=False)
        .path_id.first()
        .to_dict()
    )
    source["route_component_id"] = [
        edge_component.get((str(subtrace), int(edge)))
        if pd.notna(edge)
        else pd.NA
        for subtrace, edge in source[
            ["subtrace_id", "edge_index"]
        ].itertuples(index=False, name=None)
    ]
    source["route_component_id"] = source.groupby(
        "subtrace_id", sort=False
    ).route_component_id.transform(lambda values: values.ffill().bfill())
    previous_status = source.matched_point_status.shift()
    component_key = source.route_component_id.map(
        lambda value: "<NA>" if pd.isna(value) else str(value)
    )
    break_before = (
        source.subtrace_id.ne(source.subtrace_id.shift())
        | source.get(
            "preprocess_break_before", pd.Series(False, index=source.index)
        ).fillna(False)
        | source.gps_outlier.ne(source.gps_outlier.shift())
        | source.matched_point_status.eq("unmatched").ne(
            previous_status.eq("unmatched")
        )
        | component_key.ne(component_key.shift())
        | source.route_discontinuity
        | source.route_discontinuity.shift(fill_value=False)
    ).fillna(False)
    if len(break_before):
        break_before.iloc[0] = True
    source["quality_segment_number"] = break_before.astype(int).cumsum() - 1

    component_geometries: dict[tuple[str, Any], Any] = {}
    component_distances: dict[tuple[str, Any], float] = {}
    edge_geometries: dict[tuple[str, int], Any] = {}
    for key, group in route_primary.groupby(
        ["subtrace_id", "path_id"], sort=False, dropna=False
    ):
        component_geometries[(str(key[0]), key[1])] = (
            geometry_store.union_route_parts(group)
        )
        component_distances[(str(key[0]), key[1])] = float(
            _numeric(group, "length_m").fillna(0).sum()
        )
    for key, group in route_primary.dropna(
        subset=["valhalla_edge_index"]
    ).groupby(["subtrace_id", "valhalla_edge_index"], sort=False):
        geometry = geometry_store.union_route_parts(group)
        if geometry is not None:
            edge_geometries[(str(key[0]), int(key[1]))] = geometry

    x, y = geometry_store.point_coordinates(source)
    source["_metric_x"] = x
    source["_metric_y"] = y
    point_distance_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    for segment_number, points in source.groupby(
        "quality_segment_number", sort=False
    ):
        points = points.copy()
        component_values = points.route_component_id.dropna()
        component_id = (
            component_values.mode().iloc[0] if len(component_values) else pd.NA
        )
        subtrace_id = str(points.subtrace_id.iloc[0])
        component_key = (subtrace_id, component_id)
        component_geometry = (
            component_geometries.get(component_key)
            if pd.notna(component_id)
            else None
        )
        xy = points[["_metric_x", "_metric_y"]].to_numpy(float)
        distances = np.full(len(points), np.inf, dtype=float)
        point_route_geometries: list[Any | None] = [None] * len(points)
        edge_numeric = _numeric(points, "edge_index", np.nan)
        relevant_geometries: list[Any] = []
        for edge_index, indexes in edge_numeric.groupby(
            edge_numeric, dropna=False
        ).groups.items():
            geometry = (
                edge_geometries.get((subtrace_id, int(edge_index)))
                if pd.notna(edge_index)
                else component_geometry
            )
            if geometry is None or geometry.is_empty:
                geometry = component_geometry
            if geometry is None or geometry.is_empty:
                continue
            relevant_geometries.append(geometry)
            positions = points.index.get_indexer(indexes)
            for position in positions:
                point_route_geometries[int(position)] = geometry
            point_shapes = shapely.points(
                xy[positions, 0], xy[positions, 1]
            )
            distances[positions] = np.asarray(
                shapely.distance(point_shapes, geometry), dtype=float
            )
        for point_seq, distance in zip(points.original_point_seq, distances):
            point_distance_rows.append(
                {
                    "order_id": order_id,
                    "subtrace_id": subtrace_id,
                    "segment_id": f"{order_id}:{int(segment_number):04d}",
                    "original_point_seq": int(point_seq),
                    "raw_gps_route_distance_m": float(distance),
                    "route_component_id": component_id,
                }
            )
        gps_distance = float(_numeric(points, "step_distance_m").fillna(0).sum())
        duration = float(points.timestamp.max() - points.timestamp.min())
        buffer20 = float(np.isfinite(distances).sum() and np.mean(distances <= 20))
        buffer40 = float(np.isfinite(distances).sum() and np.mean(distances <= 40))
        buffer80 = float(np.isfinite(distances).sum() and np.mean(distances <= 80))
        uncovered = ~np.isfinite(distances) | (distances > 40)
        uncovered_distance = float(
            _numeric(points, "step_distance_m").fillna(0).to_numpy()[uncovered].sum()
        )
        time_steps = _numeric(points, "time_gap_s").fillna(0).clip(lower=0).to_numpy()
        uncovered_time = float(time_steps[uncovered].sum())
        p90 = float(np.quantile(distances, 0.90)) if len(distances) else np.inf
        p99 = float(np.quantile(distances, 0.99)) if len(distances) else np.inf
        sequence_min = int(points.original_point_seq.min())
        sequence_max = int(points.original_point_seq.max())
        intervals = interval_measurements.loc[
            _numeric(interval_measurements, "from_original_point_seq").ge(sequence_min)
            & _numeric(interval_measurements, "to_original_point_seq").le(sequence_max)
        ]
        routes = (
            route_primary.loc[
                route_primary.subtrace_id.astype(str).eq(subtrace_id)
                & route_primary.path_id.eq(component_id)
            ]
            if pd.notna(component_id)
            else route_primary.iloc[0:0]
        )
        canonical_status = _canonical_status(routes)
        dynamic_status = _dynamic_status(intervals)
        if bool(points.gps_outlier.any()):
            gps_status = "local_outlier"
        elif bool(
            points.get(
                "preprocess_break_before",
                pd.Series(False, index=points.index),
            ).fillna(False).any()
            and gps_distance
            >= float(settings.get("gps_unresolved_gap_distance_m", 300.0))
        ):
            gps_status = "unresolved_gap"
        else:
            gps_status = "clean"
        parallel = _parallel_window_features(
            points,
            xy,
            (
                unary_union(relevant_geometries)
                if relevant_geometries
                else component_geometry
            ),
            point_route_geometries,
            distances,
            settings,
        )
        has_gap = bool(
            points.route_discontinuity.any()
            or points.matched_point_status.eq("unmatched").any()
        )
        route_status, failure_reason = _segment_route_status(
            has_route=component_geometry is not None,
            buffer40=buffer40,
            p90=p90,
            uncovered_distance_share=_share(uncovered_distance, gps_distance),
            parallel=parallel,
            duration_s=duration,
            gps_distance_m=gps_distance,
            has_gap=has_gap,
            settings=settings,
        )
        segment_rows.append(
            {
                "order_id": order_id,
                "segment_id": f"{order_id}:{int(segment_number):04d}",
                "subtrace_id": subtrace_id,
                "route_component_id": component_id,
                "start_point_seq": sequence_min,
                "end_point_seq": sequence_max,
                "start_time": float(points.timestamp.min()),
                "end_time": float(points.timestamp.max()),
                "segment_duration_s": duration,
                "segment_gps_distance_m": gps_distance,
                "segment_route_distance_m": float(
                    component_distances.get(component_key, 0.0)
                ),
                "segment_route_status": route_status,
                "segment_gps_status": gps_status,
                "segment_dynamic_status": dynamic_status,
                "segment_canonical_status": canonical_status,
                "segment_buffer20_share": buffer20,
                "segment_buffer40_share": buffer40,
                "segment_buffer80_share": buffer80,
                "segment_raw_gps_p90_m": p90,
                "segment_raw_gps_p99_m": p99,
                "segment_uncovered_time_share": _share(uncovered_time, duration),
                "segment_uncovered_distance_share": _share(
                    uncovered_distance, gps_distance
                ),
                "segment_failure_reason": failure_reason,
                "segment_is_main_corridor": False,
                "has_topology_gap": bool(points.route_discontinuity.any()),
                "has_unmatched_points": bool(
                    points.matched_point_status.eq("unmatched").any()
                ),
                **parallel,
            }
        )

    segments = pd.DataFrame(segment_rows)
    if len(segments):
        main_candidates = segments.loc[
            segments.segment_gps_status.ne("local_outlier")
        ]
        if main_candidates.empty:
            main_candidates = segments
        main_index = main_candidates.segment_gps_distance_m.fillna(0).idxmax()
        segments.loc[main_index, "segment_is_main_corridor"] = True
    point_distances = pd.DataFrame(point_distance_rows)
    distances = _numeric(
        point_distances, "raw_gps_route_distance_m", np.inf
    ).to_numpy()
    order_point_geometries: list[Any | None] = []
    for subtrace_id, edge_index, component_id in source[
        ["subtrace_id", "edge_index", "route_component_id"]
    ].itertuples(index=False, name=None):
        geometry = (
            edge_geometries.get((str(subtrace_id), int(edge_index)))
            if pd.notna(edge_index)
            else None
        )
        if geometry is None and pd.notna(component_id):
            geometry = component_geometries.get(
                (str(subtrace_id), component_id)
            )
        order_point_geometries.append(geometry)
    all_geometries = [
        geometry
        for geometry in component_geometries.values()
        if geometry is not None and not geometry.is_empty
    ]
    order_parallel = _parallel_window_features(
        source,
        source[["_metric_x", "_metric_y"]].to_numpy(float),
        unary_union(all_geometries) if all_geometries else None,
        order_point_geometries,
        distances,
        settings,
    )
    raw_distance = float(_numeric(source, "step_distance_m").fillna(0).sum())
    raw_time = float(_numeric(source, "time_gap_s").fillna(0).clip(lower=0).sum())
    uncovered = ~np.isfinite(distances) | (distances > 40)
    uncovered_distance = float(
        _numeric(source, "step_distance_m").fillna(0).to_numpy()[uncovered].sum()
    )
    uncovered_time = float(
        _numeric(source, "time_gap_s").fillna(0).clip(lower=0).to_numpy()[uncovered].sum()
    )
    outlier = source.gps_outlier.to_numpy(bool)
    step_distances = _numeric(source, "step_distance_m").fillna(0).to_numpy()
    route_relevant_distance = float(step_distances[~outlier].sum())
    route_relevant_uncovered_distance = float(
        step_distances[uncovered & ~outlier].sum()
    )
    longest_count, longest_time, longest_distance = _longest_uncovered(
        source, uncovered
    )
    component_values = list(component_distances.values())
    route_distance = float(sum(component_values))
    main_component = max(component_values, default=0.0)
    route_component_count = len(
        [value for value in component_values if value > 0]
    )
    meaningful = segments.loc[
        segments.segment_gps_status.ne("local_outlier")
        & (
            segments.segment_gps_distance_m.ge(
                float(settings.get("minimum_meaningful_segment_distance_m", 50.0))
            )
            | segments.segment_duration_s.ge(
                float(settings.get("minimum_meaningful_segment_time_s", 30.0))
            )
        )
    ] if len(segments) else segments
    main = segments.loc[segments.segment_is_main_corridor].iloc[0] if len(segments) else None
    meaningful_fail = bool(
        len(meaningful)
        and meaningful.segment_route_status.eq("route_fail").any()
    )
    major_unsupported_route_interval = bool(
        len(interval_measurements)
        and (
            interval_measurements.measurement_source.eq("unresolved")
            & _numeric(
                interval_measurements, "gps_interval_distance_m"
            ).ge(
                float(
                    settings.get(
                        "route_fail_unsupported_interval_distance_m", 250.0
                    )
                )
            )
            & interval_measurements.get(
                "interval_reason",
                pd.Series("", index=interval_measurements.index),
            )
            .astype(str)
            .str.contains("engine_interpolated_endpoint|inferred_path")
        ).any()
    )
    route_distance_total = float(_numeric(route_primary, "length_m").fillna(0).sum())
    inferred_route_share = _share(
        float(
            _numeric(
                route_primary.loc[
                    route_primary.get(
                        "is_interpolated",
                        pd.Series(False, index=route_primary.index),
                    ).fillna(False),
                ],
                "length_m",
            ).fillna(0).sum()
        ),
        route_distance_total,
    )
    interpolated_point_share = float(
        source.matched_point_status.eq("interpolated").mean()
    )
    corridor_mostly_engine_inferred = bool(
        len(source)
        >= int(settings.get("inferred_corridor_minimum_points", 20))
        and interpolated_point_share
        >= float(
            settings.get(
                "inferred_corridor_minimum_interpolated_point_share", 0.50
            )
        )
        and inferred_route_share
        >= float(
            settings.get(
                "inferred_corridor_minimum_route_distance_share", 0.20
            )
        )
    )
    catastrophic_outlier_route_gap = bool(
        outlier.any()
        and float(step_distances[outlier].max(initial=0.0))
        >= float(
            settings.get(
                "route_fail_catastrophic_outlier_gap_distance_m", 250.0
            )
        )
    )
    if not bool(eligibility.get("modeling_eligible", False)):
        route_status = "route_fail"
    elif (
        major_unsupported_route_interval
        or corridor_mostly_engine_inferred
        or catastrophic_outlier_route_gap
    ):
        route_status = "route_fail"
    elif bool(order_parallel["parallel_risk"]):
        if bool(outlier.any()) and float(np.mean(distances <= 40)) >= 0.95:
            route_status = "route_uncertain"
        else:
            route_status = (
                "route_fail"
                if len(source)
                >= int(
                    settings.get(
                        "parallel_reliable_order_minimum_points", 20
                    )
                )
                else "route_uncertain"
            )
    elif main is None or main.segment_route_status == "route_fail":
        route_status = "route_fail"
    elif (
        _share(
            route_relevant_uncovered_distance,
            route_relevant_distance,
        )
        > float(settings.get("order_fail_maximum_uncovered_distance_share", 0.30))
        or (
            not bool(outlier.any())
            and _share(route_distance - main_component, route_distance)
            > float(
                settings.get(
                    "order_fail_maximum_isolated_component_share", 0.25
                )
            )
        )
    ):
        route_status = "route_fail"
    elif meaningful_fail or meaningful.segment_route_status.eq("route_partial").any():
        route_status = "route_partial"
    elif meaningful.segment_route_status.eq("route_uncertain").any():
        route_status = "route_uncertain"
    elif (
        float(np.mean(distances <= 20))
        < float(settings.get("order_pass_minimum_buffer20_share", 0.97))
        and len(distances)
        and float(np.quantile(distances, 0.99))
        > float(settings.get("order_partial_raw_gps_p99_m", 30.0))
    ):
        route_status = "route_partial"
    else:
        route_status = "route_pass"

    outlier_distance = float(
        _numeric(source, "step_distance_m").fillna(0).to_numpy()[outlier].sum()
    )
    outlier_time = float(
        _numeric(source, "time_gap_s").fillna(0).clip(lower=0).to_numpy()[outlier].sum()
    )
    if not bool(eligibility.get("modeling_eligible", False)):
        gps_status = "sparse_or_ineligible"
    elif bool(outlier.any()):
        gps_status = "local_outlier"
    elif bool(
        eligibility.get("unobserved_movement_gap_count", 0)
        or (
            source.get(
                "preprocess_break_before",
                pd.Series(False, index=source.index),
            ).fillna(False).any()
            and raw_distance
            >= float(settings.get("gps_unresolved_gap_distance_m", 300.0))
        )
    ):
        gps_status = "unresolved_gap"
    else:
        gps_status = "clean"
    canonical_segments = meaningful if len(meaningful) else segments
    canonical_values = (
        set(canonical_segments.segment_canonical_status)
        if len(canonical_segments)
        else {"unmapped"}
    )
    if "unmapped" in canonical_values:
        canonical_status = "unmapped"
    elif "ambiguous" in canonical_values:
        canonical_status = "ambiguous"
    elif "chain_resolved" in canonical_values:
        canonical_status = "chain_resolved"
    else:
        canonical_status = "unique"
    dynamic_values = set(segments.segment_dynamic_status) if len(segments) else set()
    if "dynamic_strict" in dynamic_values and dynamic_values <= {
        "dynamic_strict",
        "dynamic_partial",
    }:
        dynamic_status = "dynamic_strict"
    elif dynamic_values & {"dynamic_strict", "dynamic_partial"}:
        dynamic_status = "dynamic_partial"
    else:
        dynamic_status = "dynamic_unusable"
    parallel_segments = segments.loc[segments.parallel_risk] if len(segments) else segments
    retry = local_retry or {}
    finite_distances = distances[np.isfinite(distances)]
    order_quality = {
        "order_id": order_id,
        "gps_status": gps_status,
        "route_status": route_status,
        "dynamic_status": dynamic_status,
        "canonical_status": canonical_status,
        "has_local_outlier": bool(outlier.any()),
        "has_sparse_interval": "LARGE_UNOBSERVED_MOVEMENT_GAP"
        in str(eligibility.get("modeling_exclusion_reasons", "")),
        "has_preprocess_break": bool(
            source.get(
                "preprocess_break_before", pd.Series(False, index=source.index)
            ).fillna(False).any()
        ),
        "outlier_time_share": _share(outlier_time, raw_time),
        "outlier_distance_share": _share(outlier_distance, raw_distance),
        "raw_gps_route_distance_p50_m": (
            float(np.quantile(finite_distances, 0.50))
            if len(finite_distances)
            else np.inf
        ),
        "raw_gps_route_distance_p90_m": (
            float(np.quantile(finite_distances, 0.90))
            if len(finite_distances)
            else np.inf
        ),
        "raw_gps_route_distance_p99_m": (
            float(np.quantile(finite_distances, 0.99))
            if len(finite_distances)
            else np.inf
        ),
        "raw_gps_route_distance_max_m": (
            float(finite_distances.max()) if len(finite_distances) else np.inf
        ),
        "raw_gps_buffer20_coverage_share": float(np.mean(distances <= 20)),
        "raw_gps_buffer40_coverage_share": float(np.mean(distances <= 40)),
        "raw_gps_buffer80_coverage_share": float(np.mean(distances <= 80)),
        "longest_uncovered_point_count": longest_count,
        "longest_uncovered_time_s": longest_time,
        "longest_uncovered_gps_distance_m": longest_distance,
        "uncovered_time_share": _share(uncovered_time, raw_time),
        "uncovered_distance_share": _share(uncovered_distance, raw_distance),
        "first_segment_coverage_share": (
            float(segments.iloc[0].segment_buffer40_share) if len(segments) else 0.0
        ),
        "last_segment_coverage_share": (
            float(segments.iloc[-1].segment_buffer40_share) if len(segments) else 0.0
        ),
        "route_component_count": route_component_count,
        "main_component_distance_share": _share(main_component, route_distance),
        "isolated_component_distance_share": _share(
            route_distance - main_component, route_distance
        ),
        "discontinuity_affected_time_s": float(
            segments.loc[
                segments.has_topology_gap | segments.has_unmatched_points,
                "segment_duration_s",
            ].sum()
        ) if len(segments) else 0.0,
        "discontinuity_affected_time_share": _share(
            float(
                segments.loc[
                    segments.has_topology_gap | segments.has_unmatched_points,
                    "segment_duration_s",
                ].sum()
            ) if len(segments) else 0.0,
            raw_time,
        ),
        "discontinuity_affected_gps_distance_m": float(
            segments.loc[
                segments.has_topology_gap | segments.has_unmatched_points,
                "segment_gps_distance_m",
            ].sum()
        ) if len(segments) else 0.0,
        "discontinuity_affected_distance_share": _share(
            float(
                segments.loc[
                    segments.has_topology_gap | segments.has_unmatched_points,
                    "segment_gps_distance_m",
                ].sum()
            ) if len(segments) else 0.0,
            raw_distance,
        ),
        "maximum_continuous_gap_time_s": longest_time,
        "maximum_continuous_gap_distance_m": longest_distance,
        "gap_at_start": bool(len(segments) and (
            segments.iloc[0].has_topology_gap
            or segments.iloc[0].has_unmatched_points
        )),
        "gap_at_end": bool(len(segments) and (
            segments.iloc[-1].has_topology_gap
            or segments.iloc[-1].has_unmatched_points
        )),
        "gap_breaks_main_corridor": bool(
            main is not None
            and (main.has_topology_gap or main.has_unmatched_points)
        ),
        "gap_changes_spatial_corridor": bool(route_component_count > 1),
        "parallel_offset_mean_m": order_parallel["parallel_offset_mean_m"],
        "parallel_offset_std_m": order_parallel["parallel_offset_std_m"],
        "parallel_offset_same_sign_share": order_parallel[
            "parallel_offset_same_sign_share"
        ],
        "parallel_heading_difference_deg": order_parallel[
            "parallel_heading_difference_deg"
        ],
        "parallel_window_time_s": order_parallel["parallel_window_time_s"],
        "parallel_window_distance_m": order_parallel[
            "parallel_window_distance_m"
        ],
        "parallel_risk": bool(order_parallel["parallel_risk"]),
        "local_retry_attempted": bool(retry.get("attempted", False)),
        "local_retry_changed_route": bool(retry.get("changed_route", False)),
        "local_retry_improvement": float(retry.get("improvement", 0.0)),
        "local_retry_reason": str(retry.get("reason", "")),
    }
    return FinalQualityResult(
        segments.reset_index(drop=True),
        order_quality,
        point_distances.reset_index(drop=True),
    )
