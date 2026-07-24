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
from .parser import parse_trace_attributes
from .preprocess import preprocess_order
from .products import build_order_products
from .quality import evaluate_order_quality
from .valhalla_client import ValhallaMatcher

PRODUCTS = (
    "matched_points",
    "route_parts",
    "link_traversals",
    "turn_movements",
    "unresolved_intervals",
    "order_base",
    "route_quality",
    "performance",
    "subtrace_mapping",
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
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


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
                "observed_distance_m": float(quality["gps_distance_m"]),
                "matched_distance_m": float(quality["route_distance_m"]),
                "matching_mode": "valhalla_trace_attributes",
                "matching_confidence": float(quality["matched_interval_share"]),
                "route_quality": quality["route_quality"],
                "quality_reasons": quality["quality_reasons"],
                "backend": match_metrics.get("backend"),
                "retry_count": int(match_metrics.get("retry_count", 0)),
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
    """Run and account for exactly the frozen 600-order development sample."""

    sample = load_fixed_sample(config)
    output_root = config.path("output") / run_label
    valhalla_config = config.path("valhalla_config")
    if matcher is None:
        init_started = time.perf_counter()
        matcher = ValhallaMatcher(
            config.section("valhalla"), valhalla_config_path=valhalla_config
        )
        actor_init_ms = (time.perf_counter() - init_started) * 1000
    elif actor_init_ms is None:
        actor_init_ms = 0.0
    mapper_started = time.perf_counter()
    mapper = CanonicalEdgeMapper.from_parquet(config.path("canonical_edges"))
    mapper_init_ms = (time.perf_counter() - mapper_started) * 1000

    frames: dict[str, list[pd.DataFrame]] = {name: [] for name in PRODUCTS}
    raw_samples_remaining = int(
        config.section("runtime").get("diagnostic_response_sample_count", 20)
    )
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    wall_started = time.perf_counter()
    processing_exceptions = 0

    order_groups = sample.points.groupby(["date", "order_id"], sort=True)
    for (date, order_id), raw_order in order_groups:
        order_started = time.perf_counter()
        preprocess_ms = match_ms = parse_ms = mapping_ms = product_ms = quality_ms = 0.0
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
        }
        try:
            started = time.perf_counter()
            prep = preprocess_order(raw_order, **config.section("preprocess"))
            preprocess_ms = (time.perf_counter() - started) * 1000
            clean = prep.points
            frames["subtrace_mapping"].append(prep.mapping.assign(date=str(date)))
            matched_frames: list[pd.DataFrame] = []
            route_frames: list[pd.DataFrame] = []
            for subtrace_id, subtrace in clean.groupby("subtrace_id", sort=False):
                usable = bool(subtrace.usable_subtrace.iloc[0])
                if usable:
                    started = time.perf_counter()
                    match_result = matcher.match_order(subtrace.reset_index(drop=True))
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
                        match_metrics_total[key] += match_result.get(key, 0) or 0
                    raw_response = match_result.get("raw_response") or {}
                    if match_result.get("status") == "error":
                        exception_message = match_result.get("error_message")
                    if raw_samples_remaining > 0 and raw_response:
                        _write_json(
                            output_root
                            / "diagnostics"
                            / "raw_response_samples"
                            / f"{date}_{order_id}_{subtrace_id.rsplit(':', 1)[-1]}.json",
                            raw_response,
                        )
                        raw_samples_remaining -= 1
                else:
                    raw_response = {}
                    match_metrics_total["unmatched_point_count"] += len(subtrace)
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
            products = build_order_products(clean, matched_points, mapped_routes)
            product_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            quality = evaluate_order_quality(
                clean,
                matched_points,
                mapped_routes,
                products["unresolved_intervals"],
                config.section("quality"),
                processing_exception=exception_message,
            )
            quality_ms = (time.perf_counter() - started) * 1000
        except Exception as exc:
            processing_exceptions += 1
            exception_message = f"{type(exc).__name__}: {exc}"
            clean = raw_order.copy().reset_index(drop=True)
            clean["original_point_seq"] = np.arange(len(clean))
            clean["subtrace_id"] = f"{order_id}:000"
            clean["step_distance_m"] = 0.0
            matched_points, valhalla_routes = parse_trace_attributes(
                {}, clean, order_id=str(order_id), subtrace_id=f"{order_id}:000"
            )
            mapped_routes, _ = mapper.map_route_parts(valhalla_routes)
            products = build_order_products(clean, matched_points, mapped_routes)
            quality = evaluate_order_quality(
                clean,
                matched_points,
                mapped_routes,
                products["unresolved_intervals"],
                config.section("quality"),
                processing_exception=exception_message,
            )

        for product in (
            "route_parts",
            "link_traversals",
            "turn_movements",
            "unresolved_intervals",
        ):
            frame = products[product].copy()
            if len(frame):
                frame.insert(0, "date", str(date))
            frames[product].append(frame)
        matched_output = matched_points.copy()
        if len(matched_output):
            matched_output.insert(0, "date", str(date))
        frames["matched_points"].append(matched_output)
        quality_frame = pd.DataFrame([{**quality, "date": str(date)}])
        frames["route_quality"].append(quality_frame)
        frames["order_base"].append(
            _order_base(clean, products, quality, match_metrics_total)
        )
        total_ms = (time.perf_counter() - order_started) * 1000
        frames["performance"].append(
            pd.DataFrame(
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
                        "matched_edge_count": match_metrics_total["matched_edge_count"],
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
                        "processing_exception": exception_message,
                        "rss_bytes": process.memory_info().rss,
                    }
                ]
            )
        )
        peak_rss = max(peak_rss, process.memory_info().rss)

    combined = {name: _concat(items) for name, items in frames.items()}
    compression = str(config.section("runtime").get("parquet_compression", "zstd"))
    for product, frame in combined.items():
        if frame.empty:
            continue
        if "date" in frame:
            for date, daily in frame.groupby("date", sort=True):
                _write_product(
                    daily.reset_index(drop=True),
                    output_root / product / f"day={date}" / "part=000.parquet",
                    compression,
                )
        else:
            _write_product(
                frame.reset_index(drop=True),
                output_root / product / "part=000.parquet",
                compression,
            )

    wall_s = time.perf_counter() - wall_started
    quality = combined["route_quality"]
    performance = combined["performance"]
    counts = quality.route_quality.value_counts().to_dict()
    summary = {
        "schema_version": "stage0_v6_valhalla_run.1",
        "run_label": run_label,
        "status": "PASS"
        if len(quality) == len(sample.orders) and processing_exceptions == 0
        else "FAIL",
        "sample_order_sha256": sample.sample_sha256,
        "input_orders": int(len(sample.orders)),
        "output_orders": int(len(quality)),
        "accounting_pass": len(quality) == len(sample.orders),
        "processing_exception_count": processing_exceptions,
        "quality_counts": {str(key): int(value) for key, value in counts.items()},
        "complete_match_orders": int(quality.matched_point_share.eq(1.0).sum()),
        "partial_match_orders": int(
            quality.matched_point_share.between(0, 1, inclusive="neither").sum()
        ),
        "no_valid_match_orders": int(quality.matched_point_share.eq(0).sum()),
        "successful_reconstruction_orders": int(
            quality.successful_reconstruction.fillna(False).sum()
        ),
        "no_valid_route_orders": int(
            (~quality.successful_reconstruction.fillna(False)).sum()
        ),
        "orders_with_valid_subtrace": int(quality.subtrace_count.gt(0).sum()),
        "mean_route_parts": float(quality.route_part_count.mean()),
        "mean_matched_point_share": float(quality.matched_point_share.mean()),
        "mean_matched_interval_share": float(quality.matched_interval_share.mean()),
        "mean_inferred_distance_share": float(quality.inferred_distance_share.mean()),
        "mean_unresolved_time_share": float(quality.unresolved_time_share.mean()),
        "canonical_edge_mapping_share": float(
            quality.canonical_edge_mapping_share.mean()
        ),
        "od_endpoint_error_m": _percentiles(quality.od_endpoint_error_m),
        "snap_distance_m": _percentiles(
            pd.to_numeric(
                combined["matched_points"]["distance_from_trace_point_m"],
                errors="coerce",
            ).dropna()
        ),
        "route_gps_distance_ratio": _percentiles(
            quality.route_gps_distance_ratio
        ),
        "discontinuity_count": _percentiles(quality.discontinuity_count),
        "unmatched_point_share": _percentiles(quality.unmatched_point_share),
        "total_wall_s": wall_s,
        "actor_init_ms": actor_init_ms,
        "canonical_mapper_init_ms": mapper_init_ms,
        "pure_matching_s": float(performance.matching_ms.sum() / 1000),
        "parsing_s": float(performance.parsing_ms.sum() / 1000),
        "product_build_s": float(performance.product_build_ms.sum() / 1000),
        "order_latency_ms": _percentiles(performance.total_ms),
        "matching_latency_ms": _percentiles(performance.matching_ms),
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
