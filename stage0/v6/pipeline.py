"""Fixed-sample execution and artifact writing for Stage 0 v6."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from .canonical_mapper import CanonicalEdgeMapper
from .config import Stage0V6Config
from .eligibility import evaluate_modeling_eligibility
from .final_quality import CanonicalGeometryStore, build_final_quality
from .parser import parse_trace_attributes
from .preprocess import preprocess_order
from .products import build_order_products
from .quality import evaluate_dynamic_measurement_quality, evaluate_route_quality
from .valhalla_client import ValhallaMatcher

PRODUCTS = (
    "matched_points",
    "route_parts",
    "link_traversals",
    "link_interval_observations",
    "turn_movements",
    "unresolved_intervals",
    "interval_measurements",
    "interval_accounting",
    "order_base",
    "route_quality",
    "dynamic_measurement_quality",
    "performance",
    "subtrace_mapping",
    "preprocess_breaks",
    "modeling_eligibility",
    "route_segments",
    "point_route_distances",
    "final_quality",
)


@dataclass(frozen=True)
class FixedSample:
    points: pd.DataFrame
    orders: pd.DataFrame
    sample_sha256: str


def sample_order_sha256(frame: pd.DataFrame) -> str:
    keys = frame.loc[:, ["date", "order_id"]].astype(str)
    if keys.duplicated().any():
        raise ValueError("fixed sample contains duplicate date/order_id keys")
    keys = keys.sort_values(["date", "order_id"], kind="stable")
    payload = "\n".join(
        f"{date}|{order_id}" for date, order_id in keys.itertuples(index=False, name=None)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_fixed_sample(config: Stage0V6Config) -> FixedSample:
    dates = set(map(str, config.section("sample")["dates"]))
    manifest = pd.read_parquet(
        config.path("fixed_sample_manifest"), columns=["date", "order_id"]
    )
    manifest = manifest.loc[manifest.date.astype(str).isin(dates)].copy()
    manifest[["date", "order_id"]] = manifest[["date", "order_id"]].astype(str)
    digest = sample_order_sha256(manifest)
    expected = str(config.section("sample")["expected_sha256"])
    if digest != expected:
        raise ValueError(f"fixed sample SHA mismatch: expected={expected}, observed={digest}")
    expected_per_day = int(config.section("sample")["orders_per_day"])
    counts = manifest.groupby("date").order_id.nunique().to_dict()
    if any(counts.get(date) != expected_per_day for date in dates):
        raise ValueError(f"fixed sample daily counts do not match contract: {counts}")

    root = config.path("fixed_sample_points")
    frames = []
    for date in sorted(dates):
        files = sorted((root / f"day={date}").glob("part=*/fragment=*.parquet"))
        if not files:
            raise FileNotFoundError(f"no fixed sample point fragments for {date}")
        frames.extend(pd.read_parquet(path) for path in files)
    points = pd.concat(frames, ignore_index=True)
    points[["date", "order_id"]] = points[["date", "order_id"]].astype(str)
    points = points.merge(
        manifest.assign(_in_fixed_sample=True),
        on=["date", "order_id"],
        how="inner",
        validate="many_to_one",
    ).drop(columns="_in_fixed_sample")
    observed = points[["date", "order_id"]].drop_duplicates()
    if sample_order_sha256(observed) != digest:
        raise ValueError("materialized point fragments do not match fixed sample manifest")
    return FixedSample(points, manifest, digest)


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_safe_json(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_product(
    frame: pd.DataFrame, target: Path, compression: str = "zstd"
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, compression=compression)
    temporary.replace(target)


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if frame is not None and len(frame)]
    if nonempty:
        return pd.concat(nonempty, ignore_index=True)
    for frame in frames:
        if frame is not None:
            return frame.iloc[0:0].copy()
    return pd.DataFrame()


def _percentiles(values: pd.Series) -> dict[str, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if not len(numeric):
        return {"p50": None, "p90": None, "p99": None}
    return {
        "p50": float(numeric.quantile(0.50)),
        "p90": float(numeric.quantile(0.90)),
        "p99": float(numeric.quantile(0.99)),
    }


def _order_base(
    clean_points: pd.DataFrame,
    products: dict[str, pd.DataFrame],
    quality: dict[str, Any],
    match_metrics: dict[str, Any],
    eligibility: dict[str, Any],
    final_quality: dict[str, Any],
) -> pd.DataFrame:
    route_parts = products["route_parts"]
    return pd.DataFrame(
        [
            {
                "order_id": str(clean_points.order_id.iloc[0]),
                "driver_id": str(clean_points.driver_id.iloc[0])
                if "driver_id" in clean_points
                else "",
                "date": str(clean_points.date.iloc[0]) if "date" in clean_points else "",
                "start_time": float(clean_points.timestamp.min()),
                "end_time": float(clean_points.timestamp.max()),
                "start_node": route_parts.canonical_from_node.iloc[0]
                if len(route_parts)
                else pd.NA,
                "end_node": route_parts.canonical_to_node.iloc[-1]
                if len(route_parts)
                else pd.NA,
                "point_count": int(len(clean_points)),
                "duration_s": float(
                    clean_points.timestamp.max() - clean_points.timestamp.min()
                ),
                "observed_distance_m": float(
                    quality["raw_order_gps_distance_m"]
                ),
                "matched_distance_m": float(quality["route_distance_m"]),
                "matching_mode": (
                    "valhalla_trace_attributes"
                    if eligibility["modeling_eligible"]
                    else "excluded_low_information"
                ),
                "matching_confidence": float(quality["matched_interval_share"]),
                "route_quality": quality["route_quality"],
                "dynamic_measurement_quality": quality[
                    "dynamic_measurement_quality"
                ],
                "gps_status": final_quality["gps_status"],
                "route_status": final_quality["route_status"],
                "dynamic_status": final_quality["dynamic_status"],
                "canonical_status": final_quality["canonical_status"],
                "quality_reasons": quality["quality_reasons"],
                "backend": match_metrics.get("backend"),
                "retry_count": int(match_metrics.get("retry_count", 0)),
                "modeling_eligible": bool(eligibility["modeling_eligible"]),
                "modeling_exclusion_reasons": eligibility[
                    "modeling_exclusion_reasons"
                ],
                "selected_ignore_oneways_count": int(
                    match_metrics.get("selected_ignore_oneways_count", 0)
                ),
                "oneway_candidate_compared_count": int(
                    match_metrics.get("oneway_candidate_compared_count", 0)
                ),
            }
        ]
    )


def run_fixed_sample(
    config: Stage0V6Config,
    *,
    run_label: str,
    matcher: ValhallaMatcher | None = None,
    actor_init_ms: float | None = None,
) -> tuple[dict[str, Any], ValhallaMatcher]:
    """Run the frozen sample and flush each date/bucket atomically."""

    sample = load_fixed_sample(config)
    output_root = config.path("output") / run_label
    if matcher is None:
        init_started = time.perf_counter()
        matcher = ValhallaMatcher(
            config.section("valhalla"),
            valhalla_config_path=config.path("valhalla_config"),
        )
        actor_init_ms = (time.perf_counter() - init_started) * 1000
    elif actor_init_ms is None:
        actor_init_ms = 0.0
    mapper_started = time.perf_counter()
    mapper = CanonicalEdgeMapper.from_parquet(config.path("canonical_edges"))
    geometry_store = CanonicalGeometryStore(config.path("canonical_edges"))
    mapper_init_ms = (time.perf_counter() - mapper_started) * 1000

    runtime = config.section("runtime")
    orders_per_bucket = int(runtime.get("orders_per_bucket", 200))
    compression = str(runtime.get("parquet_compression", "zstd"))
    product_options = config.section("products")
    raw_samples_remaining = int(runtime.get("diagnostic_response_sample_count", 20))
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    wall_started = time.perf_counter()
    processing_exceptions = 0
    parquet_write_s = 0.0
    route_quality_frames: list[pd.DataFrame] = []
    dynamic_quality_frames: list[pd.DataFrame] = []
    performance_frames: list[pd.DataFrame] = []
    eligibility_frames: list[pd.DataFrame] = []
    final_quality_frames: list[pd.DataFrame] = []
    accounting_frames: list[pd.DataFrame] = []
    snap_values: list[float] = []
    bucket_peak_values: list[float] = []

    for date, day_points in sample.points.groupby("date", sort=True):
        order_ids = sorted(day_points.order_id.astype(str).unique())
        for bucket_id, start_index in enumerate(
            range(0, len(order_ids), orders_per_bucket)
        ):
            bucket_started = time.perf_counter()
            bucket_peak = process.memory_info().rss
            bucket_exceptions = 0
            bucket_ids = set(order_ids[start_index : start_index + orders_per_bucket])
            bucket_frames: dict[str, list[pd.DataFrame]] = {
                name: [] for name in PRODUCTS
            }
            for order_id in sorted(bucket_ids):
                raw_order = day_points.loc[
                    day_points.order_id.astype(str).eq(order_id)
                ].copy()
                order_started = time.perf_counter()
                preprocess_ms = match_ms = parse_ms = mapping_ms = 0.0
                product_ms = quality_ms = 0.0
                exception_message = None
                match_metrics_total: dict[str, Any] = {
                    "backend": matcher.backend,
                    "request_point_count": 0,
                    "matched_edge_count": 0,
                    "matched_point_count": 0,
                    "unmatched_point_count": 0,
                    "interpolated_point_count": 0,
                    "discontinuity_count": 0,
                    "retry_count": 0,
                    "response_ms": 0.0,
                    "oneway_candidate_compared_count": 0,
                    "selected_ignore_oneways_count": 0,
                }
                prep = None
                eligibility = None
                final_result = None
                try:
                    started = time.perf_counter()
                    prep = preprocess_order(raw_order, **config.section("preprocess"))
                    preprocess_ms = (time.perf_counter() - started) * 1000
                    clean = prep.points
                    eligibility = evaluate_modeling_eligibility(
                        clean,
                        prep.metrics,
                        **config.section("modeling_eligibility"),
                    )
                    matched_frames: list[pd.DataFrame] = []
                    route_frames: list[pd.DataFrame] = []
                    for subtrace_id, subtrace in clean.groupby(
                        "subtrace_id", sort=False
                    ):
                        usable = bool(
                            eligibility["modeling_eligible"]
                            and subtrace.usable_subtrace.iloc[0]
                        )
                        if usable:
                            started = time.perf_counter()
                            match_result = matcher.match_order(
                                subtrace.reset_index(drop=True)
                            )
                            match_ms += (time.perf_counter() - started) * 1000
                            for key in (
                                "request_point_count",
                                "matched_edge_count",
                                "matched_point_count",
                                "unmatched_point_count",
                                "interpolated_point_count",
                                "discontinuity_count",
                                "retry_count",
                                "response_ms",
                            ):
                                match_metrics_total[key] += (
                                    match_result.get(key, 0) or 0
                                )
                            raw_response = match_result.get("raw_response") or {}
                            match_metrics_total[
                                "oneway_candidate_compared_count"
                            ] += int(
                                bool(
                                    match_result.get(
                                        "oneway_candidate_compared", False
                                    )
                                )
                            )
                            match_metrics_total[
                                "selected_ignore_oneways_count"
                            ] += int(
                                bool(
                                    match_result.get(
                                        "selected_ignore_oneways", False
                                    )
                                )
                            )
                            if match_result.get("status") == "error":
                                exception_message = match_result.get("error_message")
                            if raw_samples_remaining > 0 and raw_response:
                                _write_json(
                                    output_root
                                    / "diagnostics"
                                    / "raw_response_samples"
                                    / (
                                        f"{date}_{order_id}_"
                                        f"{subtrace_id.rsplit(':', 1)[-1]}.json"
                                    ),
                                    raw_response,
                                )
                                raw_samples_remaining -= 1
                        else:
                            raw_response = {}
                            match_metrics_total["unmatched_point_count"] += len(
                                subtrace
                            )
                        started = time.perf_counter()
                        matched, routes = parse_trace_attributes(
                            raw_response,
                            subtrace.reset_index(drop=True),
                            order_id=str(order_id),
                            subtrace_id=str(subtrace_id),
                        )
                        parse_ms += (time.perf_counter() - started) * 1000
                        matched_frames.append(matched)
                        route_frames.append(routes)
                    matched_points = _concat(matched_frames)
                    valhalla_routes = _concat(route_frames)
                    started = time.perf_counter()
                    mapped_routes, _ = mapper.map_route_parts(valhalla_routes)
                    mapping_ms = (time.perf_counter() - started) * 1000
                    started = time.perf_counter()
                    products = build_order_products(
                        clean,
                        matched_points,
                        mapped_routes,
                        preprocess_breaks=prep.preprocess_breaks,
                        position_backtrack_tolerance=float(
                            product_options.get("position_backtrack_tolerance", 0.01)
                        ),
                        enable_engine_allocation=bool(
                            product_options.get("enable_engine_allocation", False)
                        ),
                    )
                    product_ms = (time.perf_counter() - started) * 1000
                    started = time.perf_counter()
                    route_quality = evaluate_route_quality(
                        clean,
                        matched_points,
                        products["route_parts"],
                        products["interval_measurements"],
                        config.section("quality"),
                        processing_exception=exception_message,
                    )
                    dynamic_quality = evaluate_dynamic_measurement_quality(
                        products["route_parts"],
                        products["link_traversals"],
                        products["interval_measurements"],
                        products["interval_accounting"],
                        config.section("quality"),
                    )
                    final_result = build_final_quality(
                        clean,
                        matched_points,
                        products["route_parts"],
                        products["interval_measurements"],
                        geometry_store,
                        eligibility,
                        config.section("final_quality"),
                    )
                    quality_ms = (time.perf_counter() - started) * 1000
                except Exception as exc:
                    processing_exceptions += 1
                    bucket_exceptions += 1
                    exception_message = f"{type(exc).__name__}: {exc}"
                    if prep is None:
                        prep = preprocess_order(
                            raw_order, **config.section("preprocess")
                        )
                    clean = prep.points
                    if eligibility is None:
                        eligibility = evaluate_modeling_eligibility(
                            clean,
                            prep.metrics,
                            **config.section("modeling_eligibility"),
                        )
                    matched_frames = []
                    for subtrace_id, subtrace in clean.groupby(
                        "subtrace_id", sort=False
                    ):
                        matched, _ = parse_trace_attributes(
                            {},
                            subtrace.reset_index(drop=True),
                            order_id=str(order_id),
                            subtrace_id=str(subtrace_id),
                        )
                        matched_frames.append(matched)
                    matched_points = _concat(matched_frames)
                    mapped_routes, _ = mapper.map_route_parts(pd.DataFrame())
                    products = build_order_products(
                        clean,
                        matched_points,
                        mapped_routes,
                        preprocess_breaks=prep.preprocess_breaks,
                    )
                    route_quality = evaluate_route_quality(
                        clean,
                        matched_points,
                        products["route_parts"],
                        products["interval_measurements"],
                        config.section("quality"),
                        processing_exception=exception_message,
                    )
                    dynamic_quality = evaluate_dynamic_measurement_quality(
                        products["route_parts"],
                        products["link_traversals"],
                        products["interval_measurements"],
                        products["interval_accounting"],
                        config.section("quality"),
                    )
                    final_result = build_final_quality(
                        clean,
                        matched_points,
                        products["route_parts"],
                        products["interval_measurements"],
                        geometry_store,
                        eligibility,
                        config.section("final_quality"),
                    )

                dated_products = {
                    "matched_points": matched_points,
                    "route_parts": products["route_parts"],
                    "link_traversals": products["link_traversals"],
                    "link_interval_observations": products[
                        "link_interval_observations"
                    ],
                    "turn_movements": products["turn_movements"],
                    "unresolved_intervals": products["unresolved_intervals"],
                    "interval_measurements": products["interval_measurements"],
                    "interval_accounting": products["interval_accounting"],
                    "subtrace_mapping": prep.mapping,
                    "preprocess_breaks": prep.preprocess_breaks,
                    "modeling_eligibility": pd.DataFrame([eligibility]),
                    "route_segments": final_result.route_segments,
                    "point_route_distances": final_result.point_route_distances,
                    "final_quality": pd.DataFrame([final_result.order_quality]),
                }
                for product, frame in dated_products.items():
                    output_frame = frame.copy()
                    if "date" not in output_frame:
                        output_frame.insert(0, "date", str(date))
                    bucket_frames[product].append(output_frame)
                route_frame = pd.DataFrame([{**route_quality, "date": str(date)}])
                dynamic_frame = pd.DataFrame(
                    [{**dynamic_quality, "date": str(date)}]
                )
                bucket_frames["route_quality"].append(route_frame)
                bucket_frames["dynamic_measurement_quality"].append(dynamic_frame)
                bucket_frames["order_base"].append(
                    _order_base(
                        clean,
                        products,
                        {**route_quality, **dynamic_quality},
                        match_metrics_total,
                        eligibility,
                        final_result.order_quality,
                    )
                )
                total_ms = (time.perf_counter() - order_started) * 1000
                performance_frame = pd.DataFrame(
                    [
                        {
                            "date": str(date),
                            "order_id": str(order_id),
                            "preprocess_ms": preprocess_ms,
                            "matching_ms": match_ms,
                            "parsing_ms": parse_ms,
                            "canonical_mapping_ms": mapping_ms,
                            "product_build_ms": product_ms,
                            "quality_ms": quality_ms,
                            "total_ms": total_ms,
                            "request_point_count": match_metrics_total[
                                "request_point_count"
                            ],
                            "matched_edge_count": match_metrics_total[
                                "matched_edge_count"
                            ],
                            "matched_point_count": match_metrics_total[
                                "matched_point_count"
                            ],
                            "unmatched_point_count": match_metrics_total[
                                "unmatched_point_count"
                            ],
                            "interpolated_point_count": match_metrics_total[
                                "interpolated_point_count"
                            ],
                            "discontinuity_count": match_metrics_total[
                                "discontinuity_count"
                            ],
                            "retry_count": match_metrics_total["retry_count"],
                            "modeling_eligible": bool(
                                eligibility["modeling_eligible"]
                            ),
                            "selected_ignore_oneways_count": match_metrics_total[
                                "selected_ignore_oneways_count"
                            ],
                            "oneway_candidate_compared_count": match_metrics_total[
                                "oneway_candidate_compared_count"
                            ],
                            "processing_exception": exception_message,
                            "rss_bytes": process.memory_info().rss,
                        }
                    ]
                )
                bucket_frames["performance"].append(performance_frame)
                route_quality_frames.append(route_frame)
                dynamic_quality_frames.append(dynamic_frame)
                performance_frames.append(performance_frame)
                eligibility_frames.append(
                    pd.DataFrame([{**eligibility, "date": str(date)}])
                )
                final_quality_frames.append(
                    pd.DataFrame(
                        [{**final_result.order_quality, "date": str(date)}]
                    )
                )
                accounting_frames.append(
                    products["interval_accounting"].assign(date=str(date))
                )
                snap_values.extend(
                    pd.to_numeric(
                        matched_points.get(
                            "distance_from_trace_point_m",
                            pd.Series(dtype=float),
                        ),
                        errors="coerce",
                    ).dropna().astype(float).tolist()
                )
                bucket_peak = max(bucket_peak, process.memory_info().rss)
                peak_rss = max(peak_rss, bucket_peak)

            combined_bucket = {
                name: _concat(items) for name, items in bucket_frames.items()
            }
            write_started = time.perf_counter()
            product_row_counts: dict[str, int] = {}
            for product, frame in combined_bucket.items():
                product_row_counts[product] = int(len(frame))
                _write_product(
                    frame.reset_index(drop=True),
                    output_root
                    / product
                    / f"day={date}"
                    / f"part={bucket_id:03d}.parquet",
                    compression,
                )
            bucket_write_s = time.perf_counter() - write_started
            parquet_write_s += bucket_write_s
            bucket_keys = pd.DataFrame(
                {
                    "date": [str(date)] * len(bucket_ids),
                    "order_id": sorted(bucket_ids),
                }
            )
            bucket_manifest = {
                "schema_version": "stage0_v6_bucket.1",
                "run_label": run_label,
                "date": str(date),
                "bucket_id": bucket_id,
                "input_order_count": len(bucket_ids),
                "output_order_count": int(
                    len(combined_bucket["route_quality"])
                ),
                "processing_exception_count": bucket_exceptions,
                "peak_rss_mb": bucket_peak / (1024**2),
                "runtime_s": time.perf_counter() - bucket_started,
                "parquet_write_s": bucket_write_s,
                "product_row_counts": product_row_counts,
                "sample_key_sha256": sample_order_sha256(bucket_keys),
                "status": (
                    "PASS"
                    if len(combined_bucket["route_quality"]) == len(bucket_ids)
                    and bucket_exceptions == 0
                    else "FAIL"
                ),
            }
            _write_json(
                output_root
                / "manifests"
                / f"day={date}"
                / f"bucket={bucket_id:03d}.json",
                bucket_manifest,
            )
            bucket_peak_values.append(bucket_peak / (1024**2))
            del combined_bucket, bucket_frames

    wall_s = time.perf_counter() - wall_started
    quality = _concat(route_quality_frames)
    dynamic = _concat(dynamic_quality_frames)
    performance = _concat(performance_frames)
    eligibility = _concat(eligibility_frames)
    final_quality = _concat(final_quality_frames)
    accounting = _concat(accounting_frames)
    route_counts = quality.route_quality.value_counts().to_dict()
    dynamic_counts = (
        dynamic.dynamic_measurement_quality.value_counts().to_dict()
    )
    summary = {
        "schema_version": "stage0_v6_valhalla_run.2",
        "run_label": run_label,
        "status": (
            "PASS"
            if (
                len(quality) == len(sample.orders)
                and processing_exceptions == 0
                and bool(accounting.time_conservation_valid.fillna(False).all())
                and bool(accounting.distance_conservation_valid.fillna(False).all())
                and int(accounting.duplicate_interval_allocation_count.sum()) == 0
                and int(
                    accounting.non_direct_observed_time_violation_count.sum()
                )
                == 0
                and int(accounting.traversal_duplicate_distance_count.sum()) == 0
            )
            else "FAIL"
        ),
        "sample_order_sha256": sample.sample_sha256,
        "input_orders": int(len(sample.orders)),
        "output_orders": int(len(quality)),
        "modeling_eligible_orders": int(
            eligibility.modeling_eligible.fillna(False).sum()
        ),
        "excluded_low_information_orders": int(
            (~eligibility.modeling_eligible.fillna(False)).sum()
        ),
        "accounting_pass": len(quality) == len(sample.orders),
        "processing_exception_count": processing_exceptions,
        "quality_counts": {
            str(key): int(value) for key, value in route_counts.items()
        },
        "dynamic_quality_counts": {
            str(key): int(value) for key, value in dynamic_counts.items()
        },
        "gps_status_counts": {
            str(key): int(value)
            for key, value in final_quality.gps_status.value_counts().items()
        },
        "route_status_counts": {
            str(key): int(value)
            for key, value in final_quality.route_status.value_counts().items()
        },
        "dynamic_status_counts": {
            str(key): int(value)
            for key, value in final_quality.dynamic_status.value_counts().items()
        },
        "canonical_status_counts": {
            str(key): int(value)
            for key, value in final_quality.canonical_status.value_counts().items()
        },
        "time_conservation_failure_count": int(
            (~accounting.time_conservation_valid.fillna(False)).sum()
        ),
        "distance_conservation_failure_count": int(
            (~accounting.distance_conservation_valid.fillna(False)).sum()
        ),
        "duplicate_interval_allocation_count": int(
            accounting.duplicate_interval_allocation_count.fillna(0).sum()
        ),
        "non_direct_observed_time_violation_count": int(
            accounting.non_direct_observed_time_violation_count.fillna(0).sum()
        ),
        "traversal_duplicate_distance_count": int(
            accounting.traversal_duplicate_distance_count.fillna(0).sum()
        ),
        "complete_match_orders": int(quality.matched_point_share.eq(1.0).sum()),
        "partial_match_orders": int(
            quality.matched_point_share.between(0, 1, inclusive="neither").sum()
        ),
        "successful_reconstruction_orders": int(
            quality.successful_reconstruction.fillna(False).sum()
        ),
        "eligible_successful_reconstruction_orders": int(
            (
                quality.successful_reconstruction.fillna(False)
                & eligibility.modeling_eligible.fillna(False).reset_index(drop=True)
            ).sum()
        ),
        "no_valid_route_orders": int(
            (~quality.successful_reconstruction.fillna(False)).sum()
        ),
        "route_formal_eligible_orders": int(
            quality.formal_analysis_eligible.fillna(False).sum()
        ),
        "dynamic_usable_orders": int(
            dynamic.dynamic_measurement_quality.isin(
                ["dynamic_strict", "dynamic_partial"]
            ).sum()
        ),
        "static_only_orders": int(
            (
                quality.formal_analysis_eligible.fillna(False)
                & dynamic.dynamic_measurement_quality.eq("dynamic_unusable")
            ).sum()
        ),
        "mean_route_parts": float(quality.route_part_count.mean()),
        "mean_matched_point_share": float(quality.matched_point_share.mean()),
        "mean_matched_interval_share": float(
            quality.matched_interval_share.mean()
        ),
        "mean_inferred_distance_share": float(
            quality.inferred_distance_share.mean()
        ),
        "mean_preprocess_break_count": float(
            quality.preprocess_break_count.mean()
        ),
        "mean_direct_observed_interval_time_share": float(
            dynamic.direct_observed_interval_time_share.mean()
        ),
        "mean_direct_observed_distance_share": float(
            dynamic.direct_observed_distance_share.mean()
        ),
        "mean_interval_supported_time_share": float(
            dynamic.interval_supported_time_share.mean()
        ),
        "mean_engine_allocated_time_share": float(
            dynamic.engine_allocated_time_share.mean()
        ),
        "mean_unresolved_time_share": float(dynamic.unresolved_time_share.mean()),
        "mean_timed_traversal_share": float(
            dynamic.timed_traversal_share.mean()
        ),
        "mean_valid_timed_traversal_count": float(
            dynamic.valid_timed_traversal_count.mean()
        ),
        "timestamp_anchor_failure_order_count": int(
            (~dynamic.timestamp_anchor_valid.fillna(False)).sum()
        ),
        "inferred_edge_observed_time_violation_count": int(
            dynamic.inferred_edge_observed_time_violation_count.sum()
        ),
        "unresolved_duplicate_allocation_count": int(
            dynamic.unresolved_duplicate_allocation_count.sum()
        ),
        "canonical_edge_mapping_share": float(
            quality.canonical_edge_mapping_share.mean()
        ),
        "od_endpoint_error_m": _percentiles(quality.od_endpoint_error_m),
        "snap_distance_m": _percentiles(pd.Series(snap_values, dtype=float)),
        "route_resolved_gps_distance_ratio": _percentiles(
            quality.route_resolved_gps_distance_ratio
        ),
        "route_raw_gps_distance_ratio": _percentiles(
            quality.route_raw_gps_distance_ratio
        ),
        "discontinuity_count": _percentiles(quality.discontinuity_count),
        "unmatched_point_share": _percentiles(quality.unmatched_point_share),
        "total_wall_s": wall_s,
        "actor_init_ms": actor_init_ms,
        "canonical_mapper_init_ms": mapper_init_ms,
        "pure_matching_s": float(performance.matching_ms.sum() / 1000),
        "parsing_s": float(performance.parsing_ms.sum() / 1000),
        "canonical_mapping_s": float(
            performance.canonical_mapping_ms.sum() / 1000
        ),
        "product_build_s": float(performance.product_build_ms.sum() / 1000),
        "quality_evaluation_s": float(performance.quality_ms.sum() / 1000),
        "parquet_write_s": parquet_write_s,
        "order_latency_ms": _percentiles(performance.total_ms),
        "matching_latency_ms": _percentiles(performance.matching_ms),
        "bucket_peak_rss_mb": _percentiles(pd.Series(bucket_peak_values)),
        "peak_rss_mb": peak_rss / (1024**2),
        "python_version": platform.python_version(),
        "pyvalhalla_version": metadata.version("pyvalhalla"),
        "valhalla_version": metadata.version("pyvalhalla"),
        "operating_system": platform.platform(),
        "config_sha256": config.digest,
    }
    _write_json(output_root / "reports" / "summary.json", summary)
    return summary, matcher


def _quantile_across(frame: pd.DataFrame, column: str, q: float) -> float | None:
    numeric = pd.to_numeric(frame[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    return float(numeric.quantile(q)) if len(numeric) else None


def benchmark_cold_and_hot(config: Stage0V6Config) -> dict[str, Any]:
    init_started = time.perf_counter()
    matcher = ValhallaMatcher(
        config.section("valhalla"),
        valhalla_config_path=config.path("valhalla_config"),
    )
    actor_init_ms = (time.perf_counter() - init_started) * 1000
    cold, matcher = run_fixed_sample(
        config,
        run_label="cold",
        matcher=matcher,
        actor_init_ms=actor_init_ms,
    )
    hot, _ = run_fixed_sample(
        config,
        run_label="hot",
        matcher=matcher,
        actor_init_ms=0.0,
    )
    comparison = {
        "sample_order_sha256": cold["sample_order_sha256"],
        "cold": cold,
        "hot": hot,
        "accounting_equal": (
            cold["input_orders"]
            == cold["output_orders"]
            == hot["input_orders"]
            == hot["output_orders"]
        ),
    }
    _write_json(config.path("output") / "reports" / "cold_hot_summary.json", comparison)
    return comparison
