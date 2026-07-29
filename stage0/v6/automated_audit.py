"""Fast, deterministic audit of existing Stage 0 v6 hot products.

The module deliberately has no matcher dependency.  It reads materialized
Parquet products, computes conservative rule-based audit features, and renders
only the highest-risk manual-review and auto-fail cases.
"""

from __future__ import annotations

import json
import math
import os
import html
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import psutil
from pyproj import CRS, Transformer
from shapely import from_wkb
from shapely.ops import substring
from shapely.ops import transform as transform_geometry

from .config import Stage0V6Config
from .coordinates import gcj02_to_wgs84

AUDIT_COLUMNS = [
    "order_id",
    "date",
    "audit_class",
    "audit_score",
    "manual_review_required",
    "reason_codes",
    "route_quality",
    "dynamic_quality",
    "dynamic_coverage_class",
    "snap_mean_m",
    "snap_p90_m",
    "snap_p99_m",
    "snap_max_m",
    "route_buffer_coverage_share",
    "route_resolved_gps_ratio",
    "route_raw_gps_ratio",
    "od_endpoint_error_m",
    "maximum_implied_speed_mps",
    "speed_violation_count",
    "canonical_mapping_share",
    "topology_violation_count",
    "direction_violation_count",
    "osm_oneway_conflict_count",
    "uturn_count",
    "matched_point_share",
    "matched_route_interval_share",
    "direct_observed_time_share",
    "direct_observed_distance_share",
    "interval_supported_time_share",
    "unresolved_time_share",
    "timed_traversal_share",
    "valid_timed_traversal_count",
    "preprocess_break_count",
    "valhalla_discontinuity_count",
    "engine_interpolated_endpoint_time_share",
    "canonical_not_unique_time_share",
    "inferred_path_time_share",
    "v5_v6_comparison_available",
    "v5_v6_edge_jaccard",
    "v5_v6_route_distance_ratio",
    "modeling_eligible",
    "modeling_exclusion_reasons",
]

INDEX_COLUMNS = [
    "case_index",
    "order_id",
    "date",
    "audit_class",
    "audit_score",
    "primary_reason",
    "secondary_reasons",
    "route_quality",
    "dynamic_quality",
    "snap_p90_m",
    "route_resolved_gps_ratio",
    "unresolved_time_share",
    "v5_v6_edge_jaccard",
    "risk_window_from_seq",
    "risk_window_to_seq",
    "risk_window_reason",
    "image_path",
]

HOT_PRODUCT_COLUMNS = {
    "order_base": ["order_id", "date"],
    "route_parts": [
        "order_id",
        "date",
        "subtrace_id",
        "path_id",
        "route_sequence",
        "canonical_edge_uid",
        "canonical_from_node",
        "canonical_to_node",
        "canonical_traversal_direction",
        "canonical_length_m",
        "entry_position_m",
        "exit_position_m",
        "traversed_against_osm_oneway",
        "valhalla_topology_gap_before",
        "length_m",
        "mapping_status",
    ],
    "link_traversals": ["order_id", "traversal_id", "measurement_source"],
    "interval_measurements": [
        "order_id",
        "gps_interval_id",
        "from_original_point_seq",
        "to_original_point_seq",
        "interval_duration_s",
        "gps_interval_distance_m",
        "measurement_source",
        "interval_reason",
    ],
    "unresolved_intervals": [
        "order_id",
        "gps_interval_id",
        "from_original_point_seq",
        "to_original_point_seq",
        "unresolved_interval_time_s",
        "unresolved_reason",
    ],
    "route_quality": [
        "order_id",
        "successful_reconstruction",
        "matched_point_share",
        "matched_interval_share",
        "preprocess_break_count",
        "discontinuity_count",
        "od_endpoint_error_m",
        "route_distance_m",
        "route_resolved_gps_distance_ratio",
        "route_raw_gps_distance_ratio",
        "canonical_edge_mapping_share",
        "route_quality",
    ],
    "dynamic_measurement_quality": [
        "order_id",
        "dynamic_measurement_quality",
        "direct_observed_interval_time_share",
        "direct_observed_distance_share",
        "interval_supported_time_share",
        "unresolved_time_share",
        "valid_timed_traversal_count",
        "timed_traversal_share",
    ],
    "matched_points": [
        "order_id",
        "original_point_seq",
        "distance_from_trace_point_m",
        "matching_lon",
        "matching_lat",
        "route_discontinuity",
        "matched_point_status",
    ],
    "turn_movements": ["order_id", "movement_sequence", "movement_source"],
    "interval_accounting": [
        "order_id",
        "time_conservation_valid",
        "timestamp_anchor_valid",
    ],
    "preprocess_breaks": ["order_id", "break_reason"],
    "modeling_eligibility": [
        "order_id",
        "modeling_eligible",
        "modeling_exclusion_reasons",
    ],
}


@dataclass(frozen=True)
class AuditRunResult:
    output_dir: Path
    audit_path: Path
    row_count: int
    class_counts: dict[str, int]
    runtime_s: float
    peak_rss_mb: float


@dataclass(frozen=True)
class AuditPackResult:
    output_dir: Path
    index_path: Path
    image_count: int
    runtime_s: float
    peak_rss_mb: float


class PeakRSSMonitor(AbstractContextManager["PeakRSSMonitor"]):
    """Sample process RSS while a bounded command is running."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        process = psutil.Process(os.getpid())
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, process.memory_info().rss)
            self._stop.wait(self.interval_s)
        self.peak_bytes = max(self.peak_bytes, process.memory_info().rss)

    def __enter__(self) -> "PeakRSSMonitor":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024 * 1024)


def _read_partitioned(
    root: Path,
    columns: list[str] | None = None,
    order_ids: set[str] | None = None,
) -> pd.DataFrame:
    files = sorted(root.glob("day=*/*.parquet"))
    if not files:
        return pd.DataFrame(columns=columns or [])
    filters = None
    if order_ids:
        filters = [("order_id", "in", sorted(order_ids))]
    frames = [
        pd.read_parquet(path, columns=columns, filters=filters) for path in files
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    _atomic_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        path,
    )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _canonical_audit(
    routes: pd.DataFrame, canonical_edges: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        "order_id",
        "topology_violation_count",
        "direction_violation_count",
        "osm_oneway_conflict_count",
        "uturn_count",
    ]
    if routes.empty:
        return pd.DataFrame(columns=columns)
    routed = routes.dropna(subset=["canonical_edge_uid"]).copy()
    if routed.empty:
        return pd.DataFrame(columns=columns)
    routed["canonical_edge_uid"] = routed["canonical_edge_uid"].astype(str)
    routed = routed.sort_values(
        ["order_id", "subtrace_id", "path_id", "route_sequence"], kind="stable"
    )
    group = routed.groupby(
        ["order_id", "subtrace_id", "path_id"], sort=False, dropna=False
    )
    routed["_previous_from"] = group["canonical_from_node"].shift()
    routed["_previous_to"] = group["canonical_to_node"].shift()
    has_previous = routed["_previous_to"].notna()
    routed["_topology_violation"] = has_previous & routed[
        "canonical_from_node"
    ].ne(routed["_previous_to"])
    routed["_uturn"] = (
        has_previous
        & routed["canonical_from_node"].eq(routed["_previous_to"])
        & routed["canonical_to_node"].eq(routed["_previous_from"])
    )

    expected_columns = ["edge_uid", "from_node", "to_node"]
    if "direction" in canonical_edges:
        expected_columns.append("direction")
    expected = canonical_edges[expected_columns].copy()
    expected["edge_uid"] = expected["edge_uid"].astype(str)
    expected = expected.rename(
        columns={
            "from_node": "_expected_from",
            "to_node": "_expected_to",
            "direction": "_expected_direction",
        }
    )
    routed = routed.merge(
        expected,
        left_on="canonical_edge_uid",
        right_on="edge_uid",
        how="left",
        validate="many_to_one",
    )
    # A canonical R edge is already stored in traversal direction. Only the
    # explicit reverse-oneway fallback reuses an F geometry/UID and therefore
    # needs the physical endpoints reversed for this consistency check.
    if "_expected_direction" in routed:
        reverse = (
            routed.get(
                "canonical_traversal_direction",
                pd.Series("F", index=routed.index),
            ).eq("R")
            & routed["_expected_direction"].eq("F")
        )
    else:
        reverse = routed.get(
            "traversed_against_osm_oneway",
            pd.Series(False, index=routed.index),
        ).fillna(False)
    expected_from = routed["_expected_from"].where(
        ~reverse, routed["_expected_to"]
    )
    expected_to = routed["_expected_to"].where(
        ~reverse, routed["_expected_from"]
    )
    routed["_direction_violation"] = routed["_expected_from"].notna() & (
        routed["canonical_from_node"].ne(expected_from)
        | routed["canonical_to_node"].ne(expected_to)
    )
    routed["_osm_oneway_conflict"] = routed.get(
        "traversed_against_osm_oneway",
        pd.Series(False, index=routed.index),
    ).fillna(False)
    result = (
        routed.groupby("order_id", sort=False)
        .agg(
            topology_violation_count=("_topology_violation", "sum"),
            direction_violation_count=("_direction_violation", "sum"),
            osm_oneway_conflict_count=("_osm_oneway_conflict", "sum"),
            uturn_count=("_uturn", "sum"),
        )
        .reset_index()
    )
    for column in columns[1:]:
        result[column] = result[column].astype("int64")
    return result[columns]


def _v5_comparison(
    orders: pd.DataFrame,
    v6_routes: pd.DataFrame,
    v5_routes: pd.DataFrame | None,
    v5_quality: pd.DataFrame | None,
) -> pd.DataFrame:
    result = orders[["order_id"]].drop_duplicates().copy()
    result["v5_v6_comparison_available"] = False
    result["v5_v6_edge_jaccard"] = np.nan
    result["v5_v6_route_distance_ratio"] = np.nan
    if v5_routes is None or v5_routes.empty:
        return result

    v5 = v5_routes.dropna(subset=["edge_uid"]).copy()
    v6 = v6_routes.dropna(subset=["canonical_edge_uid"]).copy()
    v5["order_id"] = v5["order_id"].astype(str)
    v6["order_id"] = v6["order_id"].astype(str)
    if v5_quality is not None and not v5_quality.empty:
        valid_ids = set(
            v5_quality.loc[
                v5_quality["successful_reconstruction"].fillna(False), "order_id"
            ].astype(str)
        )
        v5 = v5.loc[v5.order_id.isin(valid_ids)]

    v5_sets = v5.groupby("order_id").edge_uid.agg(
        lambda values: set(map(str, values))
    )
    v6_sets = v6.groupby("order_id").canonical_edge_uid.agg(
        lambda values: set(map(str, values))
    )
    v5_distance = (
        v5.assign(
            _distance=(
                _numeric(v5, "exit_position_m") - _numeric(v5, "entry_position_m")
            ).abs()
        )
        .groupby("order_id")
        ._distance.sum(min_count=1)
    )
    if "allocated_distance_m" in v5:
        allocated = v5.groupby("order_id").allocated_distance_m.sum(min_count=1)
        v5_distance = allocated.where(allocated.gt(0), v5_distance)
    v6_distance = (
        v6.assign(_distance=_numeric(v6, "length_m"))
        .groupby("order_id")
        ._distance.sum(min_count=1)
    )

    for index, order_id in result.order_id.astype(str).items():
        left = v5_sets.get(order_id, set())
        right = v6_sets.get(order_id, set())
        if not left or not right:
            continue
        result.at[index, "v5_v6_comparison_available"] = True
        result.at[index, "v5_v6_edge_jaccard"] = len(left & right) / len(
            left | right
        )
        denominator = float(v5_distance.get(order_id, np.nan))
        numerator = float(v6_distance.get(order_id, np.nan))
        if math.isfinite(denominator) and denominator > 0:
            result.at[index, "v5_v6_route_distance_ratio"] = numerator / denominator
    return result


def _dynamic_coverage_class(frame: pd.DataFrame) -> pd.Series:
    direct = _numeric(frame, "direct_observed_time_share").fillna(0)
    unresolved = _numeric(frame, "unresolved_time_share").fillna(1)
    quality = frame["dynamic_quality"].fillna("dynamic_unusable")
    values = np.full(len(frame), "static_route_only", dtype=object)
    values[(direct > 0) & (direct < 0.20)] = "low_dynamic_coverage"
    values[(direct >= 0.20)] = "direct_time_observations"
    values[(direct >= 0.70) & (unresolved <= 0.20)] = "high_dynamic_coverage"
    values[quality.eq("dynamic_unusable").to_numpy()] = "static_route_only"
    return pd.Series(values, index=frame.index, dtype="string")


def build_audit_features(
    products: dict[str, pd.DataFrame],
    canonical_edges: pd.DataFrame,
    audit_config: dict[str, Any],
    *,
    v5_routes: pd.DataFrame | None = None,
    v5_quality: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one vectorized audit row per order from already-materialized products."""

    base = products["order_base"][["order_id", "date"]].drop_duplicates().copy()
    base[["order_id", "date"]] = base[["order_id", "date"]].astype(str)
    if base.order_id.duplicated().any():
        raise ValueError("order_id must be unique across the fixed audit sample")

    route_quality = products["route_quality"].copy().rename(
        columns={
            "route_resolved_gps_distance_ratio": "route_resolved_gps_ratio",
            "route_raw_gps_distance_ratio": "route_raw_gps_ratio",
            "canonical_edge_mapping_share": "canonical_mapping_share",
            "matched_interval_share": "matched_route_interval_share",
            "discontinuity_count": "valhalla_discontinuity_count",
        }
    )
    route_columns = [
        "order_id",
        "successful_reconstruction",
        "route_quality",
        "matched_point_share",
        "matched_route_interval_share",
        "preprocess_break_count",
        "valhalla_discontinuity_count",
        "od_endpoint_error_m",
        "route_resolved_gps_ratio",
        "route_raw_gps_ratio",
        "canonical_mapping_share",
    ]
    features = base.merge(
        route_quality[route_columns], on="order_id", how="left", validate="one_to_one"
    )

    dynamic = products["dynamic_measurement_quality"].rename(
        columns={
            "dynamic_measurement_quality": "dynamic_quality",
            "direct_observed_interval_time_share": "direct_observed_time_share",
        }
    )
    dynamic_columns = [
        "order_id",
        "dynamic_quality",
        "direct_observed_time_share",
        "direct_observed_distance_share",
        "interval_supported_time_share",
        "unresolved_time_share",
        "timed_traversal_share",
        "valid_timed_traversal_count",
    ]
    features = features.merge(
        dynamic[dynamic_columns], on="order_id", how="left", validate="one_to_one"
    )

    matched = products["matched_points"].copy()
    matched["_snap"] = _numeric(matched, "distance_from_trace_point_m")
    buffer_m = float(audit_config.get("route_buffer_m", 40.0))
    matched["_inside_buffer"] = matched["_snap"].le(buffer_m)
    snap = (
        matched.groupby("order_id", sort=False)
        .agg(
            snap_mean_m=("_snap", "mean"),
            snap_p90_m=("_snap", lambda values: values.quantile(0.90)),
            snap_p99_m=("_snap", lambda values: values.quantile(0.99)),
            snap_max_m=("_snap", "max"),
            _point_count=("_snap", "size"),
            _inside_count=("_inside_buffer", "sum"),
        )
        .reset_index()
    )
    snap["route_buffer_coverage_share"] = (
        snap["_inside_count"] / snap["_point_count"].clip(lower=1)
    )
    features = features.merge(
        snap.drop(columns=["_point_count", "_inside_count"]),
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    intervals = products["interval_measurements"].copy()
    duration = _numeric(intervals, "interval_duration_s")
    intervals["_implied_speed"] = _numeric(
        intervals, "gps_interval_distance_m"
    ).div(duration.where(duration.gt(0)))
    speed_limit = float(audit_config.get("speed_violation_mps", 50.0))
    intervals["_speed_violation"] = intervals["_implied_speed"].gt(speed_limit)
    speed = (
        intervals.groupby("order_id", sort=False)
        .agg(
            maximum_implied_speed_mps=("_implied_speed", "max"),
            speed_violation_count=("_speed_violation", "sum"),
            _total_interval_time_s=("interval_duration_s", "sum"),
        )
        .reset_index()
    )
    speed["speed_violation_count"] = speed["speed_violation_count"].astype("int64")
    features = features.merge(
        speed.drop(columns="_total_interval_time_s"),
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    unresolved = products["unresolved_intervals"].copy()
    unresolved["_time"] = _numeric(unresolved, "unresolved_interval_time_s").fillna(0)
    unresolved_time = unresolved.pivot_table(
        index="order_id",
        columns="unresolved_reason",
        values="_time",
        aggfunc="sum",
        fill_value=0,
    )
    total_time = intervals.groupby("order_id").interval_duration_s.sum()
    reason_mapping = {
        "engine_interpolated_endpoint": "engine_interpolated_endpoint_time_share",
        "same_valhalla_edge_not_unique_canonical_edge": "canonical_not_unique_time_share",
        "inferred_path_between_gps_anchors": "inferred_path_time_share",
    }
    reason_features = pd.DataFrame(index=total_time.index)
    for reason, column in reason_mapping.items():
        numerator = (
            unresolved_time[reason]
            if reason in unresolved_time
            else pd.Series(0.0, index=total_time.index)
        )
        reason_features[column] = numerator.reindex(total_time.index, fill_value=0).div(
            total_time.where(total_time.gt(0))
        )
    features = features.merge(
        reason_features.reset_index(),
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    topology = _canonical_audit(products["route_parts"], canonical_edges)
    features = features.merge(
        topology, on="order_id", how="left", validate="one_to_one"
    )
    link_counts = (
        products["link_traversals"].groupby("order_id").size().rename("link_traversal_count")
    )
    movement_counts = (
        products["turn_movements"].groupby("order_id").size().rename("movement_count")
    )
    features = features.merge(
        link_counts.reset_index(), on="order_id", how="left", validate="one_to_one"
    ).merge(
        movement_counts.reset_index(), on="order_id", how="left", validate="one_to_one"
    )

    accounting = products["interval_accounting"][
        ["order_id", "time_conservation_valid", "timestamp_anchor_valid"]
    ]
    features = features.merge(
        accounting, on="order_id", how="left", validate="one_to_one"
    )
    comparison = _v5_comparison(
        base, products["route_parts"], v5_routes, v5_quality
    )
    features = features.merge(
        comparison, on="order_id", how="left", validate="one_to_one"
    )
    eligibility = products.get("modeling_eligibility", pd.DataFrame()).copy()
    if eligibility.empty:
        features["modeling_eligible"] = True
        features["modeling_exclusion_reasons"] = ""
    else:
        features = features.merge(
            eligibility[
                [
                    "order_id",
                    "modeling_eligible",
                    "modeling_exclusion_reasons",
                ]
            ],
            on="order_id",
            how="left",
            validate="one_to_one",
        )
        features["modeling_eligible"] = features.modeling_eligible.fillna(False)
        features["modeling_exclusion_reasons"] = features[
            "modeling_exclusion_reasons"
        ].fillna("")

    count_columns = [
        "speed_violation_count",
        "topology_violation_count",
        "direction_violation_count",
        "osm_oneway_conflict_count",
        "uturn_count",
        "preprocess_break_count",
        "valhalla_discontinuity_count",
        "valid_timed_traversal_count",
        "link_traversal_count",
        "movement_count",
    ]
    for column in count_columns:
        features[column] = _numeric(features, column).fillna(0).astype("int64")
    share_columns = list(reason_mapping.values())
    for column in share_columns:
        features[column] = _numeric(features, column).fillna(0.0)
    features["successful_reconstruction"] = features[
        "successful_reconstruction"
    ].fillna(False)
    features["v5_v6_comparison_available"] = features[
        "v5_v6_comparison_available"
    ].fillna(False)
    features["dynamic_coverage_class"] = _dynamic_coverage_class(features)
    return features


def _risk_unit(value: Any, low: float, high: float, *, missing: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return missing
    if not math.isfinite(number):
        return 1.0
    if high <= low:
        return float(number >= high)
    return float(np.clip((number - low) / (high - low), 0.0, 1.0))


def _ratio_risk(value: Any) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(ratio) or ratio <= 0:
        return 1.0
    return float(np.clip(abs(math.log(ratio)) / math.log(4.0), 0.0, 1.0))


def _audit_score(row: pd.Series) -> float:
    components = [
        (12, _risk_unit(row.snap_p90_m, 10, 60, missing=1)),
        (8, _risk_unit(row.snap_p99_m, 20, 80, missing=1)),
        (5, _risk_unit(row.snap_max_m, 40, 100, missing=1)),
        (12, _ratio_risk(row.route_resolved_gps_ratio)),
        (8, _risk_unit(row.od_endpoint_error_m, 50, 300, missing=1)),
        (9, _risk_unit(row.maximum_implied_speed_mps, 30, 70)),
        (5, _risk_unit(row.preprocess_break_count, 0, 3)),
        (6, _risk_unit(row.valhalla_discontinuity_count, 0, 3)),
        (9, _risk_unit(row.unresolved_time_share, 0.65, 1.0, missing=1)),
        (7, _risk_unit(1 - float(row.canonical_mapping_share or 0), 0, 0.1)),
        (7, _risk_unit(row.topology_violation_count, 0, 5)),
        (
            7,
            (1 - float(row.v5_v6_edge_jaccard))
            if bool(row.v5_v6_comparison_available)
            and pd.notna(row.v5_v6_edge_jaccard)
            else 0,
        ),
        (5, _risk_unit(row.engine_interpolated_endpoint_time_share, 0.5, 1.0)),
    ]
    return round(sum(weight * risk for weight, risk in components), 3)


def classify_audit_features(
    features: pd.DataFrame, audit_config: dict[str, Any]
) -> pd.DataFrame:
    """Apply conservative, explainable three-way audit rules."""

    output = features.copy()
    classes: list[str] = []
    reasons_by_order: list[str] = []
    scores: list[float] = []
    for _, row in output.iterrows():
        if not bool(row.get("modeling_eligible", True)):
            classes.append("excluded_low_information")
            reasons_by_order.append(
                str(row.get("modeling_exclusion_reasons", ""))
                or "MODELING_INELIGIBLE"
            )
            scores.append(0.0)
            continue
        hard: list[str] = []
        review: list[str] = []
        success = bool(row.successful_reconstruction)
        route_quality = str(row.route_quality)
        dynamic_quality = str(row.dynamic_quality)
        ratio = float(row.route_resolved_gps_ratio)
        endpoint = float(row.od_endpoint_error_m)
        snap_p90 = float(row.snap_p90_m)
        snap_p99 = float(row.snap_p99_m)
        coverage = float(row.route_buffer_coverage_share)
        speed = float(row.maximum_implied_speed_mps)
        canonical = float(row.canonical_mapping_share)
        matched_interval = float(row.matched_route_interval_share)
        unresolved = float(row.unresolved_time_share)
        topology = int(row.topology_violation_count)
        direction = int(row.direction_violation_count)
        discontinuities = int(row.valhalla_discontinuity_count)
        breaks = int(row.preprocess_break_count)

        if not success:
            hard.append("NO_VALID_ROUTE")
        if math.isfinite(ratio) and (ratio <= 0.20 or ratio >= 5.0):
            hard.append("EXTREME_ROUTE_GPS_RATIO")
        if math.isfinite(speed) and speed >= float(
            audit_config.get("hard_speed_mps", 70.0)
        ):
            hard.append("IMPLAUSIBLE_SPEED")
        if int(row.speed_violation_count) >= int(
            audit_config.get("hard_speed_violation_count", 4)
        ):
            hard.append("MULTIPLE_IMPLAUSIBLE_SPEED_INTERVALS")
        if canonical < 0.50:
            hard.append("MASS_UNMAPPED_CANONICAL_EDGES")
        if direction > 0:
            hard.append("CANONICAL_DIRECTION_MISMATCH")
        if topology >= int(audit_config.get("hard_topology_violations", 5)):
            hard.append("CANONICAL_ROUTE_DISCONNECTED")
        if discontinuities >= int(
            audit_config.get("hard_valhalla_discontinuities", 3)
        ):
            hard.append("MULTIPLE_VALHALLA_DISCONTINUITIES")
        if (
            route_quality == "rejected"
            and snap_p90 >= 45
            and not math.isfinite(endpoint)
        ):
            hard.append("REJECTED_HIGH_SNAP_INVALID_ENDPOINT")
        if (
            route_quality == "rejected"
            and matched_interval < 0.20
            and unresolved >= 0.99
        ):
            hard.append("REJECTED_NEAR_ZERO_ROUTE_SUPPORT")

        if route_quality == "rejected":
            review.append("ROUTE_QUALITY_REJECTED")
        if dynamic_quality == "dynamic_unusable":
            review.append("DYNAMIC_UNUSABLE")
        if snap_p90 > 40:
            review.append("HIGH_SNAP_P90")
        if snap_p99 > 80:
            review.append("HIGH_SNAP_P99")
        if coverage < 0.90:
            review.append("LOW_ROUTE_BUFFER_COVERAGE")
        if not math.isfinite(ratio) or ratio < 0.50 or ratio > 2.50:
            review.append("ROUTE_GPS_RATIO_REVIEW")
        if not math.isfinite(endpoint):
            review.append("OD_ENDPOINT_INVALID")
        elif endpoint > 100:
            review.append("HIGH_OD_ENDPOINT_ERROR")
        if canonical < 0.99:
            review.append("LOW_CANONICAL_MAPPING")
        if topology > 1:
            review.append("TOPOLOGY_VIOLATIONS")
        if int(row.uturn_count) > 1:
            review.append("POSSIBLE_UTURNS")
        if speed > float(audit_config.get("review_speed_mps", 45.0)):
            review.append("SPEED_REVIEW")
        if breaks > 1:
            review.append("MULTIPLE_PREPROCESS_BREAKS")
        if discontinuities > 0:
            review.append("VALHALLA_DISCONTINUITY")
        if unresolved > 0.95:
            review.append("EXTREME_UNRESOLVED_TIME")
        if dynamic_quality != "dynamic_unusable" and float(
            row.direct_observed_time_share
        ) < 0.03:
            review.append("LOW_DYNAMIC_COVERAGE")
        if float(row.engine_interpolated_endpoint_time_share) > 0.90:
            review.append("ENGINE_INTERPOLATED_DOMINANT")
        if float(row.canonical_not_unique_time_share) > 0.90:
            review.append("CANONICAL_NOT_UNIQUE_DOMINANT")
        if (
            bool(row.v5_v6_comparison_available)
            and pd.notna(row.v5_v6_edge_jaccard)
            and float(row.v5_v6_edge_jaccard) < 0.35
        ):
            review.append("V5_V6_ROUTE_DIVERGENCE")
        if not bool(row.time_conservation_valid):
            review.append("TIME_CONSERVATION_FAILURE")
        if not bool(row.timestamp_anchor_valid):
            review.append("TIMESTAMP_ANCHOR_FAILURE")

        hard = list(dict.fromkeys(hard))
        review = [reason for reason in dict.fromkeys(review) if reason not in hard]
        if hard:
            audit_class = "auto_fail"
            reasons = [*hard, *review]
        elif review:
            audit_class = "manual_review"
            reasons = review
        else:
            audit_class = "auto_pass"
            reasons = ["CLEAR_AUTOMATED_CHECKS"]
        classes.append(audit_class)
        reasons_by_order.append("|".join(reasons))
        scores.append(_audit_score(row))

    output["audit_class"] = pd.Series(classes, index=output.index, dtype="string")
    output["audit_score"] = scores
    output["manual_review_required"] = output.audit_class.eq("manual_review")
    output["reason_codes"] = pd.Series(
        reasons_by_order, index=output.index, dtype="string"
    )
    first = [column for column in AUDIT_COLUMNS if column in output]
    remainder = [column for column in output if column not in first]
    return output[first + remainder].sort_values(
        ["date", "order_id"], kind="stable"
    ).reset_index(drop=True)


def _load_v5_products(config: Stage0V6Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = config.path("v5_output")
    route_root = root / "route_parts"
    quality_root = root / "route_quality"
    if not route_root.exists() or not quality_root.exists():
        return pd.DataFrame(), pd.DataFrame()
    routes = _read_partitioned(
        route_root,
        [
            "order_id",
            "edge_uid",
            "entry_position_m",
            "exit_position_m",
            "allocated_distance_m",
        ],
    )
    quality = _read_partitioned(
        quality_root, ["order_id", "successful_reconstruction"]
    )
    return routes, quality


def _load_canonical_direction_table(config: Stage0V6Config) -> pd.DataFrame:
    return pq.read_table(
        config.path("canonical_edges"),
        columns=["edge_uid", "from_node", "to_node", "direction"],
    ).to_pandas()


def _audit_config(config: Stage0V6Config) -> dict[str, Any]:
    value = config.data.get("audit", {})
    return value if isinstance(value, dict) else {}


def run_automated_audit(config: Stage0V6Config) -> AuditRunResult:
    """Audit all fixed-600 orders without invoking Valhalla."""

    started = time.perf_counter()
    output = config.path("output")
    hot = output / "hot"
    target = output / "audit"
    required = list(HOT_PRODUCT_COLUMNS)
    missing = [name for name in required if not (hot / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing hot products: {missing}")

    with PeakRSSMonitor() as memory:
        products = {
            name: _read_partitioned(hot / name, columns)
            for name, columns in HOT_PRODUCT_COLUMNS.items()
        }
        v5_routes, v5_quality = _load_v5_products(config)
        canonical = _load_canonical_direction_table(config)
        features = build_audit_features(
            products,
            canonical,
            _audit_config(config),
            v5_routes=v5_routes,
            v5_quality=v5_quality,
        )
        audit = classify_audit_features(features, _audit_config(config))
        expected = int(config.section("sample")["orders_per_day"]) * len(
            config.section("sample")["dates"]
        )
        if len(audit) != expected:
            raise ValueError(f"audit row count mismatch: expected={expected}, got={len(audit)}")
        if audit.order_id.duplicated().any():
            raise ValueError("audit contains duplicate order_id rows")
        if audit[AUDIT_COLUMNS].isna().all(axis=1).any():
            raise ValueError("audit contains an empty result row")

        _atomic_parquet(audit, target / "automated_route_audit.parquet")
        _atomic_csv(audit, target / "automated_route_audit.csv")
        _atomic_parquet(audit, target / "audit_features.parquet")
        for audit_class, filename in [
            ("auto_pass", "auto_pass_orders.csv"),
            ("auto_fail", "auto_fail_orders.csv"),
            ("manual_review", "manual_review_orders.csv"),
            (
                "excluded_low_information",
                "excluded_low_information_orders.csv",
            ),
        ]:
            subset = audit.loc[audit.audit_class.eq(audit_class)].sort_values(
                ["audit_score", "date", "order_id"],
                ascending=[False, True, True],
                kind="stable",
            )
            _atomic_csv(subset, target / filename)

    runtime_s = time.perf_counter() - started
    class_counts = {
        key: int(value)
        for key, value in audit.audit_class.value_counts().sort_index().items()
    }
    manifest = {
        "schema_version": "stage0_v6_automated_audit.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sample_order_sha256": config.section("sample")["expected_sha256"],
        "row_count": int(len(audit)),
        "class_counts": class_counts,
        "runtime_s": runtime_s,
        "peak_rss_mb": memory.peak_mb,
        "source": "existing_stage0_v6_hot_products",
        "hot_products_read_once": required,
        "valhalla_invoked": False,
        "matcher_call_count": 0,
        "processing_exceptions": 0,
        "v5_comparison_available_count": int(
            audit.v5_v6_comparison_available.sum()
        ),
        "audit_config": _audit_config(config),
    }
    _atomic_json(manifest, target / "manifest.json")
    write_automated_audit_report(config)
    return AuditRunResult(
        output_dir=target,
        audit_path=target / "automated_route_audit.parquet",
        row_count=len(audit),
        class_counts=class_counts,
        runtime_s=runtime_s,
        peak_rss_mb=memory.peak_mb,
    )


def select_review_cases(
    audit: pd.DataFrame,
    *,
    manual_limit: int = 30,
    auto_fail_limit: int = 10,
    total_limit: int = 40,
) -> pd.DataFrame:
    """Select a stable, bounded manual-review image queue."""

    manual = audit.loc[audit.audit_class.eq("manual_review")].sort_values(
        ["audit_score", "date", "order_id"],
        ascending=[False, True, True],
        kind="stable",
    )
    failed = audit.loc[audit.audit_class.eq("auto_fail")].sort_values(
        ["audit_score", "date", "order_id"],
        ascending=[False, True, True],
        kind="stable",
    )
    selected = pd.concat(
        [manual.head(manual_limit), failed.head(auto_fail_limit)],
        ignore_index=True,
    ).head(total_limit)
    selected = selected.copy()
    selected.insert(0, "case_index", np.arange(1, len(selected) + 1))
    return selected


def _identify_risk_windows(
    selected_ids: set[str],
    matched: pd.DataFrame,
    intervals: pd.DataFrame,
    radius_points: int = 15,
) -> pd.DataFrame:
    """Locate one local risk window per selected order only."""

    rows: list[dict[str, Any]] = []
    for order_id in sorted(selected_ids):
        point_rows = matched.loc[matched.order_id.astype(str).eq(order_id)].copy()
        interval_rows = intervals.loc[
            intervals.order_id.astype(str).eq(order_id)
        ].copy()
        best_score = -1.0
        center = 0
        reason = "highest_snap"
        if not point_rows.empty:
            snaps = _numeric(point_rows, "distance_from_trace_point_m").fillna(0)
            point_score = snaps / 40.0 + point_rows[
                "route_discontinuity"
            ].fillna(False).astype(float) * 3.0
            winner = point_score.idxmax()
            best_score = float(point_score.loc[winner])
            center = int(point_rows.loc[winner, "original_point_seq"])
            reason = (
                "valhalla_discontinuity"
                if bool(point_rows.loc[winner, "route_discontinuity"])
                else "highest_snap"
            )
        if not interval_rows.empty:
            durations = _numeric(interval_rows, "interval_duration_s")
            speeds = _numeric(interval_rows, "gps_interval_distance_m").div(
                durations.where(durations.gt(0))
            )
            unresolved = interval_rows["measurement_source"].eq("unresolved")
            inferred = interval_rows["interval_reason"].eq(
                "inferred_path_between_gps_anchors"
            )
            interval_score = speeds.fillna(0) / 40.0 + unresolved * 0.7 + inferred * 1.2
            winner = interval_score.idxmax()
            if float(interval_score.loc[winner]) > best_score:
                row = interval_rows.loc[winner]
                center = int(row["from_original_point_seq"] + row["to_original_point_seq"]) // 2
                if float(speeds.loc[winner]) > 45:
                    reason = "maximum_implied_speed"
                elif bool(inferred.loc[winner]):
                    reason = "inferred_path"
                else:
                    reason = str(row["interval_reason"])
        rows.append(
            {
                "order_id": order_id,
                "risk_window_from_seq": max(center - radius_points, 0),
                "risk_window_to_seq": center + radius_points,
                "risk_window_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _load_candidate_raw_points(
    config: Stage0V6Config, order_ids: set[str]
) -> pd.DataFrame:
    root = config.path("fixed_sample_points")
    files = sorted(root.glob("day=*/part=*/fragment=*.parquet"))
    if not files:
        raise FileNotFoundError(f"fixed sample points not found: {root}")
    dataset = ds.dataset([str(path) for path in files], format="parquet")
    table = dataset.to_table(
        columns=["order_id", "date", "timestamp", "lon", "lat"],
        filter=ds.field("order_id").isin(sorted(order_ids)),
    )
    raw = table.to_pandas()
    raw[["order_id", "date"]] = raw[["order_id", "date"]].astype(str)
    raw = raw.sort_values(["order_id", "timestamp"], kind="stable")
    raw["original_point_seq"] = raw.groupby("order_id").cumcount()
    lon, lat = gcj02_to_wgs84(
        raw.lon.to_numpy(dtype=float), raw.lat.to_numpy(dtype=float)
    )
    raw["wgs_lon"] = lon
    raw["wgs_lat"] = lat
    return raw


def _load_geometry_lookup(
    config: Stage0V6Config, edge_ids: set[str]
) -> dict[str, Any]:
    if not edge_ids:
        return {}
    source = config.path("canonical_edges")
    table = pq.read_table(
        source,
        columns=["edge_uid", "geometry"],
        filters=[("edge_uid", "in", sorted(edge_ids))],
    ).to_pandas()
    metadata = pq.read_schema(source).metadata or {}
    geo_metadata = json.loads(metadata.get(b"geo", b"{}"))
    crs_payload = (
        geo_metadata.get("columns", {}).get("geometry", {}).get("crs")
    )
    transformer = (
        Transformer.from_crs(
            CRS.from_json_dict(crs_payload), "EPSG:4326", always_xy=True
        )
        if crs_payload
        else None
    )
    return {
        str(row.edge_uid): (
            transform_geometry(transformer.transform, from_wkb(row.geometry))
            if transformer is not None
            else from_wkb(row.geometry)
        )
        for row in table.itertuples(index=False)
        if row.geometry is not None
    }


def _geometry_lines(geometry: Any) -> list[np.ndarray]:
    if geometry is None:
        return []
    if geometry.geom_type == "LineString":
        return [np.asarray(geometry.coords)]
    if geometry.geom_type == "MultiLineString":
        return [np.asarray(part.coords) for part in geometry.geoms]
    return []


def _clipped_route_lines(
    routes: pd.DataFrame, geometry: dict[str, Any], edge_column: str
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    lines: list[np.ndarray] = []
    gap_points: list[np.ndarray] = []
    previous_end: np.ndarray | None = None
    previous_path: Any = None
    for route in routes.itertuples(index=False):
        edge_id = getattr(route, edge_column, None)
        edge_geometry = geometry.get(str(edge_id)) if pd.notna(edge_id) else None
        if edge_geometry is None:
            previous_end = None
            continue
        clipped = edge_geometry
        canonical_length = pd.to_numeric(
            pd.Series([getattr(route, "canonical_length_m", np.nan)]),
            errors="coerce",
        ).iloc[0]
        entry = pd.to_numeric(
            pd.Series([getattr(route, "entry_position_m", np.nan)]),
            errors="coerce",
        ).iloc[0]
        exit_ = pd.to_numeric(
            pd.Series([getattr(route, "exit_position_m", np.nan)]),
            errors="coerce",
        ).iloc[0]
        if (
            clipped.geom_type == "LineString"
            and pd.notna(canonical_length)
            and float(canonical_length) > 0
            and pd.notna(entry)
            and pd.notna(exit_)
        ):
            clipped = substring(
                clipped,
                float(np.clip(entry / canonical_length, 0, 1)),
                float(np.clip(exit_ / canonical_length, 0, 1)),
                normalized=True,
            )
        current_lines = _geometry_lines(clipped)
        current_path = getattr(route, "path_id", previous_path)
        explicit_gap = bool(
            getattr(route, "valhalla_topology_gap_before", False)
        )
        if (
            current_lines
            and previous_end is not None
            and (explicit_gap or current_path != previous_path)
        ):
            gap_points.extend(
                [previous_end.reshape(1, 2), current_lines[0][:1]]
            )
        lines.extend(current_lines)
        if current_lines:
            previous_end = current_lines[-1][-1]
        previous_path = current_path
    return lines, gap_points


def _plot_case(
    path: Path,
    row: pd.Series,
    raw: pd.DataFrame,
    v6_routes: pd.DataFrame,
    v5_routes: pd.DataFrame,
    geometry: dict[str, Any],
) -> None:
    order_id = str(row.order_id)
    points = raw.loc[raw.order_id.eq(order_id)].sort_values(
        "original_point_seq", kind="stable"
    )
    v6 = v6_routes.loc[v6_routes.order_id.astype(str).eq(order_id)].sort_values(
        ["subtrace_id", "path_id", "route_sequence"], kind="stable"
    )
    v5 = v5_routes.loc[v5_routes.order_id.astype(str).eq(order_id)].sort_values(
        "route_sequence", kind="stable"
    )
    v6_lines, v6_gap_points = _clipped_route_lines(
        v6, geometry, "canonical_edge_uid"
    )
    v5_lines = [
        line
        for edge_id in v5.edge_uid.dropna().astype(str)
        for line in _geometry_lines(geometry.get(edge_id))
    ]
    risk = points.loc[
        points.original_point_seq.between(
            int(row.risk_window_from_seq), int(row.risk_window_to_seq)
        )
    ]
    raw_line = points[["wgs_lon", "wgs_lat"]].to_numpy(dtype=float)
    risk_line = risk[["wgs_lon", "wgs_lat"]].to_numpy(dtype=float)

    def bounds(
        lines: Iterable[np.ndarray], fallback: np.ndarray
    ) -> tuple[float, float, float, float]:
        valid = [line for line in lines if len(line)]
        if len(fallback):
            valid.append(fallback)
        if not valid:
            return (0.0, 1.0, 0.0, 1.0)
        coordinates = np.vstack(valid)
        xmin, ymin = np.nanmin(coordinates, axis=0)
        xmax, ymax = np.nanmax(coordinates, axis=0)
        dx = max(xmax - xmin, 0.001)
        dy = max(ymax - ymin, 0.001)
        return (
            xmin - dx * 0.08,
            xmax + dx * 0.08,
            ymin - dy * 0.08,
            ymax + dy * 0.08,
        )

    global_bounds = bounds([*v5_lines, *v6_lines], raw_line)
    local_bounds = bounds([], risk_line if len(risk_line) else raw_line)

    def project(
        line: np.ndarray,
        viewport: tuple[float, float, float, float],
        coordinate_bounds: tuple[float, float, float, float],
    ) -> str:
        x, y, width, height = viewport
        xmin, xmax, ymin, ymax = coordinate_bounds
        dx, dy = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
        return " ".join(
            f"{x + (lon - xmin) / dx * width:.2f},"
            f"{y + height - (lat - ymin) / dy * height:.2f}"
            for lon, lat in line
        )

    def map_layer(
        viewport: tuple[float, float, float, float],
        coordinate_bounds: tuple[float, float, float, float],
    ) -> str:
        x, y, width, height = viewport
        parts = [
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            'fill="#fafafa" stroke="#999" stroke-width="1"/>'
        ]
        for fraction in (0.25, 0.5, 0.75):
            parts.append(
                f'<line x1="{x + width * fraction:.1f}" y1="{y}" '
                f'x2="{x + width * fraction:.1f}" y2="{y + height}" '
                'stroke="#e4e4e4" stroke-width="1"/>'
            )
            parts.append(
                f'<line x1="{x}" y1="{y + height * fraction:.1f}" '
                f'x2="{x + width}" y2="{y + height * fraction:.1f}" '
                'stroke="#e4e4e4" stroke-width="1"/>'
            )
        for line in v5_lines:
            parts.append(
                f'<polyline points="{project(line, viewport, coordinate_bounds)}" '
                'fill="none" stroke="#377eb8" stroke-width="2.5" '
                'stroke-dasharray="9 6" opacity=".7"/>'
            )
        for line in v6_lines:
            parts.append(
                f'<polyline points="{project(line, viewport, coordinate_bounds)}" '
                'fill="none" stroke="#1b9e77" stroke-width="3.2" opacity=".9" '
                'marker-end="url(#arrow-v6)"/>'
            )
        for point in v6_gap_points:
            px, py = project(point, viewport, coordinate_bounds).split(",")
            parts.append(
                f'<path d="M {float(px)-5:.1f} {float(py)-5:.1f} '
                f'L {float(px)+5:.1f} {float(py)+5:.1f} '
                f'M {float(px)+5:.1f} {float(py)-5:.1f} '
                f'L {float(px)-5:.1f} {float(py)+5:.1f}" '
                'stroke="#d73027" stroke-width="3"/>'
            )
        if len(raw_line):
            parts.append(
                f'<polyline points="{project(raw_line, viewport, coordinate_bounds)}" '
                'fill="none" stroke="#e66101" stroke-width="1.5" opacity=".75"/>'
            )
            for lon, lat in raw_line:
                point = np.asarray([[lon, lat]])
                px, py = project(point, viewport, coordinate_bounds).split(",")
                parts.append(
                    f'<circle cx="{px}" cy="{py}" r="2.0" fill="#e66101" opacity=".8"/>'
                )
        if len(risk_line):
            parts.append(
                f'<polyline points="{project(risk_line, viewport, coordinate_bounds)}" '
                'fill="none" stroke="#d01c8b" stroke-width="5" opacity=".9"/>'
            )
            for lon, lat in risk_line:
                point = np.asarray([[lon, lat]])
                px, py = project(point, viewport, coordinate_bounds).split(",")
                parts.append(
                    f'<circle cx="{px}" cy="{py}" r="3.2" fill="#d01c8b" '
                    'stroke="white" stroke-width=".7"/>'
                )
        if len(raw_line):
            start_x, start_y = project(
                raw_line[:1], viewport, coordinate_bounds
            ).split(",")
            end_x, end_y = project(
                raw_line[-1:], viewport, coordinate_bounds
            ).split(",")
            parts.append(
                f'<circle cx="{start_x}" cy="{start_y}" r="7" fill="#4daf4a" '
                'stroke="black" stroke-width="1.2"/>'
            )
            parts.append(
                f'<path d="M {float(end_x)-6:.1f} {float(end_y)-6:.1f} '
                f'L {float(end_x)+6:.1f} {float(end_y)+6:.1f} '
                f'M {float(end_x)+6:.1f} {float(end_y)-6:.1f} '
                f'L {float(end_x)-6:.1f} {float(end_y)+6:.1f}" '
                'stroke="#e41a1c" stroke-width="4"/>'
            )
        return "".join(parts)

    jaccard = (
        f"{float(row.v5_v6_edge_jaccard):.3f}"
        if pd.notna(row.v5_v6_edge_jaccard)
        else "N/A"
    )
    endpoint = (
        f"{float(row.od_endpoint_error_m):.1f} m"
        if math.isfinite(float(row.od_endpoint_error_m))
        else "invalid"
    )
    secondary = str(row.secondary_reasons) or "-"
    secondary_lines = [
        secondary[index : index + 50] for index in range(0, len(secondary), 50)
    ]
    metrics = [
        f"Case #{int(row.case_index):03d}",
        f"order_id: {order_id}",
        f"date: {row.date}",
        f"class: {row.audit_class}",
        f"score: {float(row.audit_score):.3f}",
        f"primary: {row.primary_reason}",
        "",
        "snap p90 / p99 / max:",
        f"  {float(row.snap_p90_m):.1f} / {float(row.snap_p99_m):.1f} / "
        f"{float(row.snap_max_m):.1f} m",
        f"route/resolved GPS: {float(row.route_resolved_gps_ratio):.3f}",
        f"OD endpoint error: {endpoint}",
        f"unresolved time: {float(row.unresolved_time_share):.1%}",
        f"direct observed time: {float(row.direct_observed_time_share):.1%}",
        f"canonical mapping: {float(row.canonical_mapping_share):.1%}",
        f"v5/v6 edge Jaccard: {jaccard}",
        "",
        "secondary reasons:",
        *secondary_lines,
    ]
    text_lines = "".join(
        f'<text x="1140" y="{512 + index * 22}" class="metric">'
        f"{html.escape(value)}</text>"
        for index, value in enumerate(metrics)
    )
    title = html.escape(
        f"Case #{int(row.case_index):03d} | order {order_id} | "
        f"{row.audit_class} | score={float(row.audit_score):.1f}"
    )
    subtitle = html.escape(f"Primary reason: {row.primary_reason}")
    local_title = html.escape(
        f"Risk window [{int(row.risk_window_from_seq)}, "
        f"{int(row.risk_window_to_seq)}] - {row.risk_window_reason}"
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<style>
text {{ font-family: "Segoe UI", Arial, sans-serif; fill: #222; }}
.title {{ font-size: 24px; font-weight: 700; }}
.subtitle {{ font-size: 16px; font-weight: 600; }}
.metric {{ font-family: Consolas, monospace; font-size: 15px; }}
.legend {{ font-size: 14px; }}
</style>
<defs>
<marker id="arrow-v6" viewBox="0 0 10 10" refX="8" refY="5"
 markerWidth="4" markerHeight="4" orient="auto-start-reverse">
  <path d="M 0 0 L 10 5 L 0 10 z" fill="#1b9e77"/>
</marker>
</defs>
<rect width="1600" height="1000" fill="white"/>
<text x="30" y="38" class="title">{title}</text>
<text x="30" y="66" class="subtitle">{subtitle}</text>
{map_layer((30, 90, 1060, 850), global_bounds)}
<text x="1120" y="78" class="subtitle">{local_title}</text>
{map_layer((1120, 90, 450, 360), local_bounds)}
<rect x="1120" y="485" width="450" height="455" rx="10" fill="#f7f7f7" stroke="#999"/>
{text_lines}
<line x1="55" y1="915" x2="95" y2="915" stroke="#e66101" stroke-width="3"/>
<text x="105" y="920" class="legend">Raw GPS</text>
<line x1="210" y1="915" x2="250" y2="915" stroke="#1b9e77" stroke-width="4"/>
<text x="260" y="920" class="legend">v6 route</text>
<line x1="365" y1="915" x2="405" y2="915" stroke="#377eb8" stroke-width="3" stroke-dasharray="8 5"/>
<text x="415" y="920" class="legend">v5 route</text>
<line x1="520" y1="915" x2="560" y2="915" stroke="#d01c8b" stroke-width="5"/>
<text x="570" y="920" class="legend">Risk window</text>
<circle cx="735" cy="915" r="6" fill="#4daf4a" stroke="black"/>
<text x="748" y="920" class="legend">Start</text>
<path d="M 825 909 L 837 921 M 837 909 L 825 921" stroke="#e41a1c" stroke-width="4"/>
<text x="845" y="920" class="legend">End</text>
</svg>"""
    _atomic_text(svg, path)


def _index_markdown(index: pd.DataFrame) -> str:
    lines = [
        "# Stage 0 v6 Manual Review Pack",
        "",
        "Reply with case numbers only, for example: `1, 3, 7, 12`.",
        "",
        "| Case | Class | Score | Primary reason | Order | Image |",
        "|---:|---|---:|---|---|---|",
    ]
    for row in index.itertuples(index=False):
        lines.append(
            f"| {int(row.case_index)} | {row.audit_class} | "
            f"{float(row.audit_score):.3f} | {row.primary_reason} | "
            f"`{row.order_id}` | [open]({row.image_path}) |"
        )
    lines.append("")
    return "\n".join(lines)


def write_review_indexes(index: pd.DataFrame, target: Path) -> None:
    ordered = index[INDEX_COLUMNS].sort_values("case_index", kind="stable")
    _atomic_csv(ordered, target / "index.csv")
    _atomic_text(_index_markdown(ordered), target / "index.md")


def generate_manual_review_pack(config: Stage0V6Config) -> AuditPackResult:
    """Render a bounded image pack from an existing automated audit."""

    started = time.perf_counter()
    audit_root = config.path("output") / "audit"
    audit_path = audit_root / "automated_route_audit.parquet"
    if not audit_path.exists():
        raise FileNotFoundError(
            "automated audit is missing; run the `audit` command first"
        )
    settings = _audit_config(config)
    with PeakRSSMonitor() as memory:
        audit = pd.read_parquet(audit_path)
        selected = select_review_cases(
            audit,
            manual_limit=int(settings.get("manual_image_limit", 30)),
            auto_fail_limit=int(settings.get("auto_fail_image_limit", 10)),
            total_limit=int(settings.get("total_image_limit", 40)),
        )
        selected_ids = set(selected.order_id.astype(str))
        hot = config.path("output") / "hot"
        matched = _read_partitioned(
            hot / "matched_points",
            HOT_PRODUCT_COLUMNS["matched_points"],
            selected_ids,
        )
        intervals = _read_partitioned(
            hot / "interval_measurements",
            HOT_PRODUCT_COLUMNS["interval_measurements"],
            selected_ids,
        )
        v6_routes = _read_partitioned(
            hot / "route_parts",
            HOT_PRODUCT_COLUMNS["route_parts"],
            selected_ids,
        )
        v5_routes = _read_partitioned(
            config.path("v5_output") / "route_parts",
            [
                "order_id",
                "route_sequence",
                "edge_uid",
                "entry_position_m",
                "exit_position_m",
            ],
            selected_ids,
        )
        raw = _load_candidate_raw_points(config, selected_ids)
        windows = _identify_risk_windows(selected_ids, matched, intervals)
        selected = selected.merge(
            windows, on="order_id", how="left", validate="one_to_one"
        )

        reasons = selected.reason_codes.fillna("").str.split("|")
        selected["primary_reason"] = reasons.str[0].fillna("UNSPECIFIED_RISK")
        selected["secondary_reasons"] = reasons.apply(
            lambda values: "|".join(values[1:]) if isinstance(values, list) else ""
        )
        target = audit_root / "manual_review_pack"
        images = target / "images"
        images.mkdir(parents=True, exist_ok=True)
        selected["image_path"] = selected.apply(
            lambda row: (
                f"images/case_{int(row.case_index):03d}_{row.order_id}.svg"
            ),
            axis=1,
        )
        keep_names = {Path(path).name for path in selected.image_path}
        for stale in images.glob("case_*.svg"):
            if stale.name not in keep_names:
                stale.unlink()

        edge_ids = set(v6_routes.canonical_edge_uid.dropna().astype(str)) | set(
            v5_routes.edge_uid.dropna().astype(str)
        )
        geometry = _load_geometry_lookup(config, edge_ids)
        for row in selected.itertuples(index=False):
            _plot_case(
                target / row.image_path,
                pd.Series(row._asdict()),
                raw,
                v6_routes,
                v5_routes,
                geometry,
            )
        index = selected.copy()
        write_review_indexes(index, target)

    runtime_s = time.perf_counter() - started
    manifest = {
        "schema_version": "stage0_v6_manual_review_pack.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "image_count": int(len(selected)),
        "manual_review_image_count": int(
            selected.audit_class.eq("manual_review").sum()
        ),
        "auto_fail_image_count": int(selected.audit_class.eq("auto_fail").sum()),
        "case_index_min": int(selected.case_index.min()) if len(selected) else None,
        "case_index_max": int(selected.case_index.max()) if len(selected) else None,
        "runtime_s": runtime_s,
        "peak_rss_mb": memory.peak_mb,
        "candidate_raw_orders_loaded": int(raw.order_id.nunique()),
        "all_600_raw_orders_loaded": False,
        "enhanced_audit_order_count": int(len(selected)),
        "valhalla_invoked": False,
        "matcher_call_count": 0,
        "selection_order": "manual_review score-descending, then auto_fail score-descending",
        "limits": {
            "manual_review": int(settings.get("manual_image_limit", 30)),
            "auto_fail": int(settings.get("auto_fail_image_limit", 10)),
            "total": int(settings.get("total_image_limit", 40)),
        },
    }
    _atomic_json(manifest, target / "manifest.json")
    write_automated_audit_report(config)
    return AuditPackResult(
        output_dir=target,
        index_path=target / "index.csv",
        image_count=len(selected),
        runtime_s=runtime_s,
        peak_rss_mb=memory.peak_mb,
    )


def _top_reasons(audit: pd.DataFrame, audit_class: str, limit: int = 10) -> list[tuple[str, int]]:
    subset = audit.loc[audit.audit_class.eq(audit_class), "reason_codes"].fillna("")
    exploded = subset.str.split("|").explode()
    counts = exploded.loc[exploded.ne("")].value_counts().head(limit)
    return [(str(reason), int(count)) for reason, count in counts.items()]


def write_automated_audit_report(config: Stage0V6Config) -> Path:
    audit_root = config.path("output") / "audit"
    audit_path = audit_root / "automated_route_audit.parquet"
    if not audit_path.exists():
        raise FileNotFoundError(audit_path)
    audit = pd.read_parquet(audit_path)
    audit_manifest = json.loads((audit_root / "manifest.json").read_text(encoding="utf-8")) if (
        audit_root / "manifest.json"
    ).exists() else {}
    pack_manifest_path = audit_root / "manual_review_pack" / "manifest.json"
    pack = (
        json.loads(pack_manifest_path.read_text(encoding="utf-8"))
        if pack_manifest_path.exists()
        else {}
    )
    counts = audit.audit_class.value_counts().to_dict()
    reasons = _top_reasons(audit, "manual_review")
    reason_lines = "\n".join(
        f"- `{reason}`: {count}" for reason, count in reasons
    ) or "- None"
    image_count = int(pack.get("image_count", 0))
    index_range = (
        f"{pack.get('case_index_min')}-{pack.get('case_index_max')}"
        if image_count
        else "not generated"
    )
    report = f"""# Stage 0 v6 Automated Audit Report

## Outcome

- Fixed sample audited: **{len(audit)} orders**
- `auto_pass`: **{int(counts.get("auto_pass", 0))}**
- `auto_fail`: **{int(counts.get("auto_fail", 0))}**
- `manual_review`: **{int(counts.get("manual_review", 0))}**
- `excluded_low_information`: **{int(counts.get("excluded_low_information", 0))}**
- Processing exceptions: **{int(audit_manifest.get("processing_exceptions", 0))}**

## Main manual-review triggers

{reason_lines}

## Review images

- Images generated: **{image_count}**
- Case index range: **{index_range}**
- Manual-review images: **{int(pack.get("manual_review_image_count", 0))}**
- Auto-fail images: **{int(pack.get("auto_fail_image_count", 0))}**
- Queue: `stage0/output_v6/audit/manual_review_pack/index.csv`

## Reuse and efficiency

- Audit source: existing `stage0/output_v6/hot/` Parquet products.
- Valhalla reruns: **0**.
- Automated audit runtime: **{float(audit_manifest.get("runtime_s", 0)):.3f} s**.
- Automated audit peak RSS: **{float(audit_manifest.get("peak_rss_mb", 0)):.1f} MB**.
- Image-pack runtime: **{float(pack.get("runtime_s", 0)):.3f} s**.
- Image-pack peak RSS: **{float(pack.get("peak_rss_mb", 0)):.1f} MB**.
- Raw GPS loading was restricted to selected image candidates.
- Stability rematching was intentionally not run: it is optional and would invoke
  the matcher; this audit instead localizes risk windows from existing products.

## Most discriminating indicators

The conservative rules prioritize route reconstruction success, route/GPS distance
ratio, snap p90/p99 and buffer coverage, OD endpoint validity, canonical topology,
Valhalla discontinuities, extreme implied speeds, unresolved-time share, and large
v5/v6 edge-set divergence. Reverse traversal against an OSM one-way tag is
reported as an informational conflict, not a failure. `audit_score` combines
the risk indicators only for ranking; information-poor orders are separated
before the pass/review/fail rules.

Dynamic evidence is reported separately as `high_dynamic_coverage`,
`direct_time_observations`, `low_dynamic_coverage`, or `static_route_only`.
The existence of one directly observed interval is not treated as sufficient
dynamic coverage.

## How to respond

Open `manual_review_pack/index.md` or the numbered SVG image files, then reply with
case numbers only, for example: **`1, 3, 7, 12`**. The numbering is deterministic
for unchanged audit inputs.
"""
    target = config.repo_root / "stage0" / "docs" / "stage0_v6_automated_audit_report.md"
    _atomic_text(report, target)
    return target
