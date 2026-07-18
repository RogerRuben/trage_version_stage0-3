"""Resume-safe, bucketed Stage 0 v5 daily execution."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import psutil

from .archive import (
    extract_daily_archive,
    list_archive_members,
    materialize_sampled_points,
    sampled_orders_path,
    sampling_run_id,
)
from .config import Stage0Config, stable_hash
from .manifest import base_manifest, write_manifest
from .matching import CandidateIndex, TransitionEngine, match_order
from .quality import conservation_summary, evaluate_order_quality
from .reconstruction import EdgeAwareRouter, build_movements, build_traversals, route_parts_frame
from .routing import CompactMovementRouter


LOGGER = logging.getLogger("stage0.v5")


PRODUCTS = ("order_base", "link_traversals", "turn_movements", "route_parts", "route_quality", "performance")


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _output_file(output: Path, product: str, date: str, bucket: int) -> Path:
    return output / product / f"day={date}" / f"part={bucket:03d}.parquet"


def export_case_traces(
    config: Stage0Config,
    repo: Path,
    dates: list[str],
    sample_run_id: str,
) -> dict[str, Any]:
    """Reproducibly retain bounded examples, never every rejected point trace."""
    output, work = config.path("output", repo), config.path("work", repo)
    runtime = config.section("runtime")
    failure_limit = int(runtime.get("case_trace_per_failure_reason_per_day", 5))
    representative_limit = int(runtime.get("case_trace_representative_per_day", 10))
    seed = int(config.section("sampling")["seed"])
    counts: dict[str, int] = {}
    for date in dates:
        files = sorted((output / "route_quality" / f"day={date}").glob("*.parquet"))
        if not files:
            continue
        quality = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)

        def primary_reason(row: pd.Series) -> str:
            for column in ("hard_error_flags", "soft_quality_flags"):
                try:
                    values = json.loads(str(row[column]))
                except (json.JSONDecodeError, TypeError):
                    values = []
                if values:
                    return str(values[0])
            return "representative_pass"

        quality["case_reason"] = quality.apply(primary_reason, axis=1)
        quality["case_hash"] = quality.order_id.astype(str).map(
            lambda order_id: stable_hash(date, order_id, seed=seed)
        )
        selected: list[pd.DataFrame] = []
        rejected = quality.loc[quality.route_quality.eq("rejected")]
        for reason, group in rejected.groupby("case_reason", sort=True):
            chosen = group.nsmallest(failure_limit, "case_hash").copy()
            chosen["stratum_population"] = len(group)
            chosen["selection_probability"] = len(chosen) / max(len(group), 1)
            selected.append(chosen)
        passed = quality.loc[quality.route_quality.ne("rejected")]
        if len(passed):
            chosen = passed.nsmallest(representative_limit, "case_hash").copy()
            chosen["stratum_population"] = len(passed)
            chosen["selection_probability"] = len(chosen) / len(passed)
            selected.append(chosen)
        index = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=quality.columns)
        target = output / "case_traces" / sample_run_id / f"day={date}"
        _write_frame(index, target / "case_trace_index.parquet")
        selected_orders = set(index.order_id.astype(str))
        traces: list[pd.DataFrame] = []
        for path in (work / "sampled_points" / sample_run_id / f"day={date}").glob("part=*/*.parquet"):
            frame = pd.read_parquet(path)
            retained = frame.loc[frame.order_id.astype(str).isin(selected_orders)]
            if len(retained):
                traces.append(retained)
        if traces:
            _write_frame(pd.concat(traces, ignore_index=True), target / "points.parquet")
        counts[date] = len(index)
    return {"case_trace_counts": counts, "case_trace_total": int(sum(counts.values()))}


def _partition_done(
    output: Path,
    date: str,
    bucket: int,
    config_hash: str,
    orders_per_day: int | None = None,
) -> bool:
    manifest = output / "manifests" / "partitions" / f"day={date}" / f"part={bucket:03d}.json"
    if not manifest.exists() or any(not _output_file(output, name, date, bucket).exists() for name in PRODUCTS):
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    sample_matches = orders_per_day is None or payload.get("orders_per_day") == int(orders_per_day)
    return payload.get("status") == "PASS" and payload.get("config_hash") == config_hash and sample_matches


def _empty_quality(order_id: str, reason: str) -> dict[str, Any]:
    return {
        "order_id": order_id,
        "successful_reconstruction": False,
        "direction_violation_count": 1,
        "topology_gap_count": 1,
        "unreasonable_detour_count": 0,
        "illegal_u_turn_count": 0,
        "layer_violation_count": 0,
        "restriction_violation_count": 0,
        "route_link_count": 0,
        "observed_distance_m": 0.0,
        "unallocated_observed_time_s": 0.0,
        "unallocated_observed_distance_m": 0.0,
        "matched_distance_m": 0.0,
        "od_endpoint_error_m": float("inf"),
        "time_conservation_error_s": 0.0,
        "distance_conservation_error_m": 0.0,
        "fallback_share": 0.0,
        "p90_projection_distance_m": float("inf"),
        "route_length_ratio": float("nan"),
        "interpolated_distance_share": 0.0,
        "matching_confidence": 0.0,
        "repeated_link_share": 1.0,
        "parallel_ambiguity_share": 1.0,
        "route_quality": "rejected",
        "formal_analysis_eligible": False,
        "strict_evaluation_eligible": False,
        "hard_error_flags": json.dumps([reason]),
        "soft_quality_flags": "[]",
        "quality_reasons": reason,
    }


def _sampling_lookup(output: Path, run_id: str, date: str) -> pd.DataFrame:
    path = sampled_orders_path(output / "manifests", run_id, date)
    if not path.exists():
        raise FileNotFoundError(f"sampling manifest missing for {date}: {path}")
    return pd.read_parquet(path).set_index("order_id", drop=False)


def prepare_day_points(
    config: Stage0Config,
    repo: Path,
    date: str,
    buckets: int,
    orders_per_day: int,
    sampling_dates: list[str],
    force: bool = False,
) -> dict[str, Any]:
    output = config.path("output", repo)
    run_id = sampling_run_id(sampling_dates, orders_per_day, int(config.section("sampling")["seed"]))
    sampled = pd.read_parquet(sampled_orders_path(output / "manifests", run_id, date))
    members = list_archive_members(config.path("archive", repo), config.path("seven_zip", repo))
    member = next((row["source_member"] for row in members if row["date"] == date), None)
    if member is None:
        raise RuntimeError(f"date {date} absent from archive inventory")
    daily = extract_daily_archive(config.path("archive", repo), member, config.path("seven_zip", repo), config.path("work", repo))
    target = config.path("work", repo) / "sampled_points" / run_id / f"day={date}"
    return materialize_sampled_points(daily, sampled, target, date, buckets=buckets, force=force)


def run_dates(
    config: Stage0Config,
    repo: Path,
    dates: list[str],
    buckets: int = 128,
    resume: bool = True,
    retain_points: bool = False,
    force: bool = False,
    workers: int = 1,
    bucket_shard_index: int = 0,
    bucket_shard_count: int = 1,
    bucket_ids: set[int] | None = None,
    orders_per_day: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    output, work = config.path("output", repo), config.path("work", repo)
    network = output / "network"
    sample_count = int(orders_per_day or config.section("sampling")["orders_per_day"])
    sample_run_id = sampling_run_id(dates, sample_count, int(config.section("sampling")["seed"]))
    edges = gpd.read_parquet(network / "canonical_edges.parquet")
    movements = pd.read_parquet(network / "movement_graph.parquet")
    candidate_config = config.section("candidate")
    hmm_config = config.section("hmm")
    network_config = {**config.section("network"), **hmm_config}
    candidate_index = CandidateIndex(edges, candidate_config, str(work / "candidate_index" / config.digest))
    movement_router = CompactMovementRouter(edges, movements, network_config)
    transition_engine = TransitionEngine(edges, movements, pd.DataFrame(), hmm_config, movement_router)
    edge_router = EdgeAwareRouter(edges, movements, network_config, movement_router)
    edge_lookup = edges.set_index("edge_uid")
    run_rows: list[dict[str, Any]] = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    for date in dates:
        prepare_day_points(config, repo, date, buckets, sample_count, dates, force=force)
        sample_lookup = _sampling_lookup(output, sample_run_id, date)
        day_input_orders = 0
        for bucket in range(int(buckets)):
            if bucket_ids is not None and bucket not in bucket_ids:
                continue
            if bucket % int(bucket_shard_count) != int(bucket_shard_index):
                continue
            fragments = sorted((work / "sampled_points" / sample_run_id / f"day={date}" / f"part={bucket:03d}").glob("*.parquet"))
            if not fragments:
                continue
            if resume and not force and _partition_done(output, date, bucket, config.digest, sample_count):
                LOGGER.info("date=%s bucket=%03d completed partition skipped", date, bucket)
                continue
            bucket_started = time.perf_counter()
            points = pd.concat([pd.read_parquet(path) for path in fragments], ignore_index=True)
            points = points.sort_values(["order_id", "timestamp"], kind="stable")
            input_orders = points.order_id.astype(str).nunique()
            day_input_orders += input_orders
            order_base_rows: list[dict[str, Any]] = []
            quality_rows: list[dict[str, Any]] = []
            traversal_frames: list[pd.DataFrame] = []
            movement_frames: list[pd.DataFrame] = []
            route_frames: list[pd.DataFrame] = []
            retained_point_frames: list[pd.DataFrame] = []
            performance_rows: list[dict[str, Any]] = []
            failed_rows: list[dict[str, Any]] = []

            def process_order(item: tuple[Any, pd.DataFrame]) -> dict[str, Any]:
                order_id, group = item
                order_id = str(order_id)
                failed = None
                reconstruction_ms = movement_build_ms = quality_ms = 0.0
                try:
                    matched, match_summary = match_order(
                        group, edges, candidate_index, transition_engine,
                        candidate_config, hmm_config, config.section("network")["metric_crs"],
                    )
                    reconstruction_started = time.perf_counter()
                    route = edge_router.reconstruct(matched)
                    route_parts = route_parts_frame(order_id, route, edge_lookup) if route.edge_uids else pd.DataFrame()
                    traversals = build_traversals(order_id, matched, route_parts, edge_lookup) if len(route_parts) else pd.DataFrame()
                    reconstruction_ms = (time.perf_counter() - reconstruction_started) * 1000.0
                    movement_started = time.perf_counter()
                    turns = build_movements(order_id, route_parts, movement_router, matched) if len(route_parts) > 1 else pd.DataFrame()
                    movement_build_ms = (time.perf_counter() - movement_started) * 1000.0
                    quality_started = time.perf_counter()
                    quality = evaluate_order_quality(
                        order_id, matched, route_parts, traversals, turns, edge_lookup,
                        match_summary, config.section("quality"),
                    )
                    quality_ms = (time.perf_counter() - quality_started) * 1000.0
                except Exception as error:  # logged and accounted, never silently discarded
                    LOGGER.exception("date=%s bucket=%03d order=%s failed", date, bucket, order_id)
                    match_summary = {"matching_mode": "rejected", "fallback_used": False, "fallback_reason": "exception"}
                    quality = _empty_quality(order_id, "processing_exception")
                    route_parts = pd.DataFrame()
                    matched = pd.DataFrame()
                    traversals = turns = route_parts = pd.DataFrame()
                    failed = {"date": date, "bucket": bucket, "order_id": order_id, "exception_type": type(error).__name__, "reason": str(error)}
                sample = sample_lookup.loc[order_id] if order_id in sample_lookup.index else None
                timestamps = pd.to_numeric(group.timestamp, errors="coerce")
                order_base = {
                    "order_id": order_id,
                    "driver_id": str(group.driver_id.iloc[0]),
                    "date": date,
                    "start_time": float(timestamps.min()),
                    "end_time": float(timestamps.max()),
                    "start_node": int(edge_lookup.loc[route_parts.edge_uid.iloc[0]].from_node) if len(route_parts) else pd.NA,
                    "end_node": int(edge_lookup.loc[route_parts.edge_uid.iloc[-1]].to_node) if len(route_parts) else pd.NA,
                    "point_count": int(len(group)),
                    "duration_s": float(timestamps.max() - timestamps.min()),
                    "observed_distance_m": quality["observed_distance_m"],
                    "matched_distance_m": quality["matched_distance_m"],
                    "matching_mode": match_summary["matching_mode"],
                    "matching_confidence": quality["matching_confidence"],
                    "route_quality": quality["route_quality"],
                    "quality_reasons": quality["quality_reasons"],
                    "sampling_probability": float(sample.sampling_probability) if sample is not None else float("nan"),
                    "sampling_weight": float(sample.sampling_weight) if sample is not None else float("nan"),
                    "fallback_used": bool(match_summary.get("fallback_used", False)),
                    "fallback_reason": str(match_summary.get("fallback_reason", "")),
                }
                return {
                    "order_base": order_base,
                    "quality": {"date": date, "bucket": bucket, **quality},
                    "traversals": traversals,
                    "turns": turns,
                    "route_parts": route_parts,
                    "matched": matched,
                    "failed": failed,
                    "performance": {
                        "date": date,
                        "bucket": bucket,
                        "order_id": order_id,
                        **{key: match_summary.get(key, 0) for key in (
                            "candidate_generation_ms", "ambiguity_detection_ms", "local_hmm_ms",
                            "full_hmm_ms", "transition_search_ms", "matching_total_ms",
                            "dijkstra_calls", "dijkstra_expanded_nodes", "route_cache_hits",
                            "route_cache_misses", "local_window_count", "local_failed_window_count",
                            "full_order_trigger_reason",
                        )},
                        "reconstruction_ms": reconstruction_ms,
                        "movement_build_ms": movement_build_ms,
                        "quality_ms": quality_ms,
                    },
                }

            grouped_orders = list(points.groupby("order_id", sort=False))
            if int(workers) > 1:
                with ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix="stage0-order") as executor:
                    results = list(executor.map(process_order, grouped_orders))
            else:
                results = [process_order(item) for item in grouped_orders]
            for result in results:
                order_base_rows.append(result["order_base"])
                quality_rows.append(result["quality"])
                if len(result["traversals"]): traversal_frames.append(result["traversals"])
                if len(result["turns"]): movement_frames.append(result["turns"])
                if len(result["route_parts"]): route_frames.append(result["route_parts"])
                if result["failed"] is not None: failed_rows.append(result["failed"])
                performance_rows.append(result["performance"])
                matched = result["matched"]
                if len(matched) and retain_points:
                    retained_point_frames.append(matched)
            tables = {
                "order_base": pd.DataFrame(order_base_rows),
                "link_traversals": pd.concat(traversal_frames, ignore_index=True) if traversal_frames else pd.DataFrame(columns=["order_id"]),
                "turn_movements": pd.concat(movement_frames, ignore_index=True) if movement_frames else pd.DataFrame(columns=["order_id"]),
                "route_parts": pd.concat(route_frames, ignore_index=True) if route_frames else pd.DataFrame(columns=["order_id"]),
                "route_quality": pd.DataFrame(quality_rows),
            }
            output_started = time.perf_counter()
            for product, table in tables.items():
                _write_frame(table, _output_file(output, product, date, bucket))
            output_io_ms = (time.perf_counter() - output_started) * 1000.0 / max(input_orders, 1)
            performance = pd.DataFrame(performance_rows)
            performance["output_io_ms"] = output_io_ms
            _write_frame(performance, _output_file(output, "performance", date, bucket))
            if retained_point_frames:
                _write_frame(
                    pd.concat(retained_point_frames, ignore_index=True),
                    output / "matched_points_retained" / sample_run_id / f"day={date}" / f"part={bucket:03d}.parquet",
                )
            if failed_rows:
                _write_frame(pd.DataFrame(failed_rows), output / "failed_orders" / f"day={date}" / f"part={bucket:03d}.parquet")
            partition_manifest = {
                **base_manifest(repo, config.digest, fragments),
                "status": "PASS",
                "date": date,
                "bucket": bucket,
                "orders_per_day": sample_count,
                "sampling_run_id": sample_run_id,
                "input_orders": input_orders,
                "output_orders": len(order_base_rows),
                "failed_orders": len(failed_rows),
                "runtime_sec": time.perf_counter() - bucket_started,
            }
            write_manifest(output / "manifests" / "partitions" / f"day={date}" / f"part={bucket:03d}.json", partition_manifest)
            peak_rss = max(peak_rss, process.memory_info().rss)
            run_rows.append(partition_manifest)
            del points, tables, order_base_rows, quality_rows, traversal_frames, movement_frames, route_frames, retained_point_frames, performance_rows
        LOGGER.info("date=%s processed_input_orders=%d", date, day_input_orders)
    summary = summarize_run(config, repo, dates, sample_count)
    if int(bucket_shard_count) == 1 and summary.get("accounting_pass") and not retain_points:
        summary.update(export_case_traces(config, repo, dates, sample_run_id))
    summary.update({
        "status": "SHARD_COMPLETE" if int(bucket_shard_count) > 1 else ("PASS" if summary["accounting_pass"] else "FAIL"),
        "runtime_sec": time.perf_counter() - started,
        "peak_memory_mb": peak_rss / (1024**2),
        "routing_cache_pairs": movement_router.cache_size,
        "partition_runs": len(run_rows),
        "workers": int(workers),
        "bucket_shard_index": int(bucket_shard_index),
        "bucket_shard_count": int(bucket_shard_count),
        "orders_per_day": sample_count,
        "sampling_run_id": sample_run_id,
    })
    date_scope = "-".join(dates) if len(dates) <= 3 else f"{dates[0]}-{dates[-1]}-{len(dates)}days"
    shard_suffix = (
        f"__shard={int(bucket_shard_index):02d}-of-{int(bucket_shard_count):02d}"
        if int(bucket_shard_count) > 1 else ""
    )
    summary_name = f"stage0_v5_run_summary__{sample_run_id}__dates={date_scope}{shard_suffix}.json"
    write_manifest(output / "reports" / summary_name, summary)
    return summary


def summarize_run(config: Stage0Config, repo: Path, dates: list[str], orders_per_day: int) -> dict[str, Any]:
    output = config.path("output", repo)
    quality_files = [path for date in dates for path in (output / "route_quality" / f"day={date}").glob("*.parquet")]
    order_files = [path for date in dates for path in (output / "order_base" / f"day={date}").glob("*.parquet")]
    quality = pd.concat([pd.read_parquet(path) for path in quality_files], ignore_index=True) if quality_files else pd.DataFrame()
    orders = pd.concat([pd.read_parquet(path) for path in order_files], ignore_index=True) if order_files else pd.DataFrame()
    run_id = sampling_run_id(dates, orders_per_day, int(config.section("sampling")["seed"]))
    sampling_path = output / "manifests" / "sampling_runs" / run_id / "sampling_manifest.parquet"
    if sampling_path.exists():
        sampled = pd.read_parquet(sampling_path, columns=["date", "order_id"])
        expected_orders = int(len(sampled.loc[sampled.date.astype(str).isin(dates)]))
    else:
        expected_orders = len(orders)
    base = conservation_summary(quality, expected_orders) if len(quality) else {"accounting_pass": False, "input_orders": expected_orders, "output_orders": 0}
    modes = orders.matching_mode.value_counts(normalize=True).to_dict() if len(orders) else {}
    return {
        **base,
        "dates": dates,
        "matching_mode_share": {str(k): float(v) for k, v in modes.items()},
        "topology_gap_count": int(quality.topology_gap_count.sum()) if len(quality) else 0,
        "parallel_ambiguity_order_count": int((quality.parallel_ambiguity_share > 0).sum()) if len(quality) else 0,
        "direction_violation_count": int(quality.direction_violation_count.sum()) if len(quality) else 0,
        "layer_violation_count": int(quality.layer_violation_count.sum()) if len(quality) else 0,
        "restriction_violation_count": int(quality.restriction_violation_count.sum()) if len(quality) else 0,
        "mean_inferred_distance_share": float(quality.interpolated_distance_share.mean()) if len(quality) else float("nan"),
    }
