"""Resume-safe, bucketed Stage 0 v5 daily execution."""

from __future__ import annotations

import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import psutil

from .archive import (
    extract_daily_archive,
    list_archive_members,
    materialize_sampled_points,
    sampled_orders_path,
    sampling_run_id,
)
from .config import Stage0Config, config_hash, stable_hash
from .manifest import base_manifest, write_manifest
from .matching import CandidateIndex, TransitionEngine, match_order
from .quality import conservation_summary, evaluate_order_quality
from .reconstruction import (
    EdgeAwareRouter,
    add_position_aware_route_distances,
    build_movements,
    build_traversals,
    build_unresolved_intervals,
    route_parts_frame,
)
from .routing import CompactMovementRouter


LOGGER = logging.getLogger("stage0.v5")


PRODUCTS = (
    "order_base", "link_traversals", "turn_movements", "unresolved_intervals",
    "route_parts", "route_quality", "performance",
)


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
        for path in (work / "matched_diagnostics" / sample_run_id / f"day={date}").glob("*.parquet"):
            frame = pd.read_parquet(path)
            retained = frame.loc[frame.order_id.astype(str).isin(selected_orders)]
            if len(retained):
                traces.append(retained)
        if traces:
            _write_frame(pd.concat(traces, ignore_index=True), target / "points.parquet")
        route_files = sorted((output / "route_parts" / f"day={date}").glob("*.parquet"))
        if route_files:
            routes = pd.concat([pd.read_parquet(path) for path in route_files], ignore_index=True)
            routes = routes.loc[routes.order_id.astype(str).isin(selected_orders)]
            if len(routes):
                _write_frame(routes, target / "route_parts.parquet")
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
        "restriction_block_count": 0,
        "unparsed_restriction_exposure_count": 0,
        "suspicious_level_transition_count": 0,
        "true_layer_discontinuity_count": 0,
        "unreasonable_detour_count": 0,
        "illegal_u_turn_count": 0,
        "layer_violation_count": 0,
        "restriction_violation_count": 0,
        "observed_dynamic_label_on_inferred_edge_count": 0,
        "hmm_output_path_identity_mismatch_count": 0,
        "hmm_path_distance_mismatch_count": 0,
        "same_edge_jitter_mismatch_count": 0,
        "raw_movement_audit_available": False,
        "route_link_count": 0,
        "observed_distance_m": 0.0,
        "unallocated_observed_time_s": 0.0,
        "unallocated_observed_distance_m": 0.0,
        "matched_distance_m": 0.0,
        "od_endpoint_error_m": float("inf"),
        "time_conservation_error_s": 0.0,
        "distance_conservation_error_m": 0.0,
        "time_allocation_error_s": 0.0,
        "internal_distance_error_m": 0.0,
        "projected_route_distance_error_m": float("nan"),
        "invalid_position_aware_distance_count": 0,
        "actual_invalid_position_event_count": 0,
        "position_audit_applicable_order_count": 0,
        "position_audit_not_applicable_match_failure_count": 1,
        "observed_run_alignment_valid": False,
        "unresolved_interval_time_s": 0.0,
        "projected_matched_movement_distance_m": float("nan"),
        "path_to_gps_ratio_q50": float("nan"),
        "path_to_gps_ratio_q90": float("nan"),
        "path_to_gps_ratio_q99": float("nan"),
        "fallback_share": 0.0,
        "p90_projection_distance_m": float("inf"),
        "route_length_ratio": float("nan"),
        "interpolated_distance_share": float("nan"),
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
    diagnostic_mode: str = "sampled",
) -> dict[str, Any]:
    if diagnostic_mode not in {"none", "sampled", "full"}:
        raise ValueError("diagnostic_mode must be none, sampled, or full")
    if int(workers) != 1:
        raise ValueError(
            "Stage 0 v5 order-level threads are disabled: Python routing/dataframe work is GIL-bound. "
            "Use mutually exclusive process-level bucket shards with --workers 1."
        )
    started = time.perf_counter()
    output, work = config.path("output", repo), config.path("work", repo)
    network = output / "network"
    sample_count = int(orders_per_day or config.section("sampling")["orders_per_day"])
    sample_run_id = sampling_run_id(dates, sample_count, int(config.section("sampling")["seed"]))
    load_started = time.perf_counter()
    edges = gpd.read_parquet(network / "canonical_edges.parquet")
    edges_load_ms = (time.perf_counter() - load_started) * 1000.0
    load_started = time.perf_counter()
    movements = pd.read_parquet(network / "movement_graph.parquet")
    movements_load_ms = (time.perf_counter() - load_started) * 1000.0
    candidate_config = config.section("candidate")
    hmm_config = config.section("hmm")
    network_config = {**config.section("network"), **hmm_config}
    initialization_started = time.perf_counter()
    candidate_index = CandidateIndex(
        edges,
        candidate_config,
        str(work / "candidate_index" / config_hash(candidate_config)),
        config.section("network")["metric_crs"],
        hmm_config,
    )
    candidate_index_init_ms = (time.perf_counter() - initialization_started) * 1000.0
    initialization_started = time.perf_counter()
    movement_router = CompactMovementRouter(edges, movements, network_config)
    movement_router_init_ms = (time.perf_counter() - initialization_started) * 1000.0
    transition_engine = TransitionEngine(edges, movements, pd.DataFrame(), hmm_config, movement_router)
    edge_router = EdgeAwareRouter(edges, movements, network_config, movement_router)
    edge_lookup = edges.set_index("edge_uid")
    run_rows: list[dict[str, Any]] = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    diagnostic_counts: dict[tuple[str, str], int] = {}
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
            input_started = time.perf_counter()
            points = pd.concat([pd.read_parquet(path) for path in fragments], ignore_index=True)
            points = points.sort_values(["order_id", "timestamp"], kind="stable")
            bucket_input_ms = (time.perf_counter() - input_started) * 1000.0
            input_orders = points.order_id.astype(str).nunique()
            day_input_orders += input_orders
            order_base_rows: list[dict[str, Any]] = []
            quality_rows: list[dict[str, Any]] = []
            traversal_frames: list[pd.DataFrame] = []
            movement_frames: list[pd.DataFrame] = []
            unresolved_frames: list[pd.DataFrame] = []
            route_frames: list[pd.DataFrame] = []
            retained_point_frames: list[pd.DataFrame] = []
            performance_rows: list[dict[str, Any]] = []
            failed_rows: list[dict[str, Any]] = []
            matched_diagnostic_frames: list[pd.DataFrame] = []

            def process_order(item: tuple[Any, pd.DataFrame]) -> dict[str, Any]:
                order_id, group = item
                order_id = str(order_id)
                failed = None
                route_bridge_search_ms = route_parts_build_ms = traversal_build_ms = 0.0
                movement_build_ms = quality_ms = 0.0
                precomputed_path_count = reconstruction_bridge_request_count = 0
                reconstruction_path_search_count = 0
                reconstruction_expanded_nodes = reconstruction_path_cache_hits = 0
                try:
                    matched, match_summary = match_order(
                        group, edges, candidate_index, transition_engine,
                        candidate_config, hmm_config,
                    )
                    reconstruction_stats_before = movement_router.stats()
                    route = edge_router.reconstruct(matched)
                    reconstruction_stats = movement_router.stats().minus(reconstruction_stats_before)
                    reconstruction_path_search_count = reconstruction_stats.path_calls
                    reconstruction_expanded_nodes = reconstruction_stats.expanded_nodes
                    reconstruction_path_cache_hits = reconstruction_stats.path_cache_hits
                    route_bridge_search_ms = edge_router.last_bridge_search_ms
                    precomputed_path_count = edge_router.last_precomputed_path_count
                    reconstruction_bridge_request_count = edge_router.last_path_search_count
                    parts_started = time.perf_counter()
                    route_parts = route_parts_frame(order_id, route, edge_lookup) if route.edge_uids else pd.DataFrame()
                    if len(route_parts):
                        route_parts = add_position_aware_route_distances(matched, route_parts, edge_lookup)
                    route_parts_build_ms = (time.perf_counter() - parts_started) * 1000.0
                    traversal_started = time.perf_counter()
                    traversals = build_traversals(order_id, matched, route_parts, edge_lookup) if len(route_parts) else pd.DataFrame()
                    traversal_build_ms = (time.perf_counter() - traversal_started) * 1000.0
                    movement_started = time.perf_counter()
                    turns = build_movements(order_id, route_parts, movement_router, matched) if len(route_parts) > 1 else pd.DataFrame()
                    unresolved = build_unresolved_intervals(
                        order_id, route_parts, movement_router, matched,
                    ) if len(route_parts) else pd.DataFrame()
                    movement_build_ms = (time.perf_counter() - movement_started) * 1000.0
                    quality_started = time.perf_counter()
                    quality = evaluate_order_quality(
                        order_id, matched, route_parts, traversals, turns, edge_lookup,
                        match_summary, config.section("quality"), unresolved,
                    )
                    quality_ms = (time.perf_counter() - quality_started) * 1000.0
                except Exception as error:  # logged and accounted, never silently discarded
                    LOGGER.exception("date=%s bucket=%03d order=%s failed", date, bucket, order_id)
                    match_summary = {"matching_mode": "rejected", "fallback_used": False, "fallback_reason": "exception"}
                    quality = _empty_quality(order_id, "processing_exception")
                    route_parts = pd.DataFrame()
                    matched = pd.DataFrame()
                    traversals = turns = unresolved = route_parts = pd.DataFrame()
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
                    "pre_transition_validation_mode": str(
                        match_summary.get(
                            "pre_transition_validation_mode",
                            match_summary["matching_mode"],
                        )
                    ),
                    "matching_confidence": quality["matching_confidence"],
                    "route_quality": quality["route_quality"],
                    "quality_reasons": quality["quality_reasons"],
                    "sampling_probability": float(sample.sampling_probability) if sample is not None else float("nan"),
                    "sampling_weight": float(sample.sampling_weight) if sample is not None else float("nan"),
                    "fallback_used": bool(match_summary.get("fallback_used", False)),
                    "fallback_reason": str(match_summary.get("fallback_reason", "")),
                }
                string_performance_fields = {
                    "full_order_trigger_reason",
                    "local_failed_window_details",
                    "pre_transition_validation_mode",
                    "failed_transition_reasons",
                    "provisional_edge_sequence",
                }
                boolean_performance_fields = {
                    "local_hmm_attempted",
                    "full_hmm_attempted",
                    "full_hmm_succeeded",
                    "full_hmm_failed",
                }
                return {
                    "order_base": order_base,
                    "quality": {"date": date, "bucket": bucket, **quality},
                    "traversals": traversals,
                    "turns": turns,
                    "unresolved": unresolved,
                    "route_parts": route_parts,
                    "matched": matched,
                    "failed": failed,
                    "performance": {
                        "date": date,
                        "bucket": bucket,
                        "order_id": order_id,
                        **{
                            key: match_summary.get(
                                key,
                                ""
                                if key in string_performance_fields
                                else False
                                if key in boolean_performance_fields
                                else 0,
                            )
                            for key in (
                            "coordinate_transform_ms", "candidate_generation_ms", "ambiguity_detection_ms", "local_hmm_ms",
                            "full_hmm_ms", "transition_search_ms", "matching_total_ms",
                            "dijkstra_calls", "dijkstra_expanded_nodes", "route_cache_hits",
                            "route_cache_misses", "local_window_count", "local_failed_window_count",
                            "full_order_trigger_reason", "local_hmm_attempted", "full_hmm_attempted",
                            "local_hmm_order_attempt_count",
                            "local_hmm_window_attempt_count",
                            "local_hmm_retry_window_count",
                            "boundary_repair_viterbi_count",
                            "full_hmm_succeeded", "full_hmm_failed", "stationary_point_share",
                            "effective_ambiguity_point_share",
                            "eligible_ambiguity_point_share", "selected_bridge_request_count",
                            "selected_bridge_path_count", "selected_path_search_ms",
                            "distance_search_calls", "path_search_calls", "positive_cache_hits",
                            "negative_cache_hits", "path_cache_hits",
                            "exact_path_search_calls",
                            "approximate_path_search_calls",
                            "approximate_search_unresolved_count",
                            "order_transition_evidence_cache_hits",
                            "order_transition_evidence_cache_misses",
                            "local_patch_count", "no_candidate_initial_count",
                            "no_candidate_recovered_count",
                            "under_minimum_candidate_expansion_count",
                            "under_minimum_candidate_initial_count",
                            "transition_candidate_expansion_count",
                            "local_boundary_failure_count",
                            "local_internal_failure_count",
                            "boundary_repair_attempt_count",
                            "boundary_repair_success_count",
                            "local_failed_window_details",
                            "pre_transition_validation_mode",
                            "selected_transition_failure_count",
                            "transition_retry_used_count",
                            "endpoint_distance_exceeds_cutoff_count",
                            "no_movement_path_within_cutoff_count",
                            "failed_transition_reasons",
                            "provisional_edge_sequence",
                            )
                        },
                        "route_bridge_search_ms": route_bridge_search_ms,
                        "route_parts_build_ms": route_parts_build_ms,
                        "traversal_build_ms": traversal_build_ms,
                        "precomputed_path_count": precomputed_path_count,
                        "reconstruction_bridge_request_count": reconstruction_bridge_request_count,
                        "reconstruction_path_search_count": reconstruction_path_search_count,
                        "reconstruction_expanded_nodes": reconstruction_expanded_nodes,
                        "reconstruction_path_cache_hits": reconstruction_path_cache_hits,
                        "movement_build_ms": movement_build_ms,
                        "quality_ms": quality_ms,
                    },
                }

            groupby_started = time.perf_counter()
            grouped_orders = points.groupby("order_id", sort=False)
            bucket_groupby_ms = (time.perf_counter() - groupby_started) * 1000.0
            mode_counts: dict[str, int] = {}
            quality_counts: dict[str, int] = {}
            progress_every = int(config.section("runtime").get("log_progress_every_orders", 25))
            for processed_in_bucket, item in enumerate(grouped_orders, start=1):
                result = process_order(item)
                order_base_rows.append(result["order_base"])
                quality_rows.append(result["quality"])
                if len(result["traversals"]): traversal_frames.append(result["traversals"])
                if len(result["turns"]): movement_frames.append(result["turns"])
                if len(result["unresolved"]): unresolved_frames.append(result["unresolved"])
                if len(result["route_parts"]): route_frames.append(result["route_parts"])
                if result["failed"] is not None: failed_rows.append(result["failed"])
                performance_rows.append(result["performance"])
                matched = result["matched"]
                quality_name = str(result["quality"].get("route_quality", "unknown"))
                mode_name = str(result["order_base"].get("matching_mode", "unknown"))
                mode_counts[mode_name] = mode_counts.get(mode_name, 0) + 1
                quality_counts[quality_name] = quality_counts.get(quality_name, 0) + 1
                reasons = str(result["quality"].get("quality_reasons", "")) or "representative"
                diagnostic_key = (date, reasons)
                retain_diagnostic = diagnostic_mode == "full"
                if diagnostic_mode == "sampled":
                    failure_limit = int(config.section("runtime").get(
                        "case_trace_per_failure_reason_per_day", 5
                    ))
                    representative_limit = int(config.section("runtime").get(
                        "case_trace_representative_per_day", 10
                    ))
                    limit = representative_limit if reasons == "representative" else failure_limit
                    # Stable hash chooses cases; the counter enforces a strict
                    # per-day/reason cap without writing all points first.
                    hash_value = stable_hash(date, result["order_base"]["order_id"], reasons, seed=20261009)
                    retain_diagnostic = (
                        diagnostic_counts.get(diagnostic_key, 0) < limit
                        and hash_value % 10_000 < 2_500
                    )
                    if retain_diagnostic:
                        diagnostic_counts[diagnostic_key] = diagnostic_counts.get(diagnostic_key, 0) + 1
                if len(matched) and retain_diagnostic:
                    diagnostic_columns = [column for column in (
                        "order_id", "point_seq", "timestamp", "source_lon", "source_lat",
                        "matching_lon", "matching_lat", "edge_uid", "position_on_edge",
                        "gps_to_edge_distance_m", "candidate_count", "candidate_rank",
                        "emission_margin", "viterbi_margin", "parallel_ambiguity",
                        "ambiguity_reason", "matching_mode", "point_quality", "observed_step_m",
                        "heading_reliable", "stationary_or_low_motion", "edge_heading_difference_deg",
                        "time_gap_s", "transition_cutoff_m", "selected_path_distance_m",
                        "selected_path_routing_cost", "selected_path_identifier",
                        "selected_path_json", "selected_path_search_exact",
                        "selected_jitter_penalty_m", "path_to_gps_ratio",
                        "legacy_candidate_score", "projection_emission_cost",
                        "heading_emission_cost", "edge_prior_cost",
                        "total_emission_cost",
                        "selected_network_snapshot_mismatch",
                        "provisional_edge_uid", "provisional_position_on_edge",
                        "transition_failure_reason",
                        "transition_failure_raw_movement_status",
                        "transition_failure_search_exact",
                        "transition_failure_diagnostic_class",
                        "transition_retry_used", "transition_initial_cutoff_m",
                    ) if column in matched.columns]
                    matched_diagnostic_frames.append(matched[diagnostic_columns].copy())
                if len(matched) and retain_points:
                    retained_point_frames.append(matched)
                if progress_every > 0 and (
                    processed_in_bucket % progress_every == 0
                    or processed_in_bucket == input_orders
                ):
                    elapsed = time.perf_counter() - bucket_started
                    recent = performance_rows[-min(progress_every, len(performance_rows)):]
                    mean_match_ms = float(np.mean([
                        float(row.get("matching_total_ms", 0.0)) for row in recent
                    ])) if recent else 0.0
                    LOGGER.info(
                        "progress date=%s bucket=%03d orders=%d/%d elapsed_s=%.1f "
                        "mean_recent_match_ms=%.1f modes=%s quality=%s rss_mb=%.1f",
                        date, bucket, processed_in_bucket, input_orders, elapsed,
                        mean_match_ms, mode_counts, quality_counts,
                        process.memory_info().rss / 1024**2,
                    )
            tables = {
                "order_base": pd.DataFrame(order_base_rows),
                "link_traversals": pd.concat(traversal_frames, ignore_index=True) if traversal_frames else pd.DataFrame(columns=["order_id"]),
                "turn_movements": pd.concat(movement_frames, ignore_index=True) if movement_frames else pd.DataFrame(columns=["order_id"]),
                "unresolved_intervals": pd.concat(unresolved_frames, ignore_index=True) if unresolved_frames else pd.DataFrame(columns=["order_id"]),
                "route_parts": pd.concat(route_frames, ignore_index=True) if route_frames else pd.DataFrame(columns=["order_id"]),
                "route_quality": pd.DataFrame(quality_rows),
            }
            output_started = time.perf_counter()
            for product, table in tables.items():
                _write_frame(table, _output_file(output, product, date, bucket))
            output_io_ms = (time.perf_counter() - output_started) * 1000.0 / max(input_orders, 1)
            performance = pd.DataFrame(performance_rows)
            performance["output_io_ms"] = output_io_ms
            performance["bucket_input_ms_per_order"] = bucket_input_ms / max(input_orders, 1)
            performance["bucket_groupby_ms_per_order"] = bucket_groupby_ms / max(input_orders, 1)
            diagnostic_started = time.perf_counter()
            if matched_diagnostic_frames:
                _write_frame(
                    pd.concat(matched_diagnostic_frames, ignore_index=True),
                    work / "matched_diagnostics" / sample_run_id / f"day={date}" / f"part={bucket:03d}.parquet",
                )
            diagnostic_io_ms = (time.perf_counter() - diagnostic_started) * 1000.0 / max(input_orders, 1)
            performance["diagnostic_io_ms"] = diagnostic_io_ms
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
            del points, tables, order_base_rows, quality_rows, traversal_frames, movement_frames, unresolved_frames, route_frames, retained_point_frames, performance_rows, matched_diagnostic_frames
        LOGGER.info("date=%s processed_input_orders=%d", date, day_input_orders)
    summary_started = time.perf_counter()
    summary = summarize_run(config, repo, dates, sample_count)
    summary_build_ms = (time.perf_counter() - summary_started) * 1000.0
    case_trace_export_ms = 0.0
    if int(bucket_shard_count) == 1 and summary.get("accounting_pass") and not retain_points:
        case_started = time.perf_counter()
        summary.update(export_case_traces(config, repo, dates, sample_run_id))
        case_trace_export_ms = (time.perf_counter() - case_started) * 1000.0
    runtime_sec = time.perf_counter() - started
    summary.update({
        "status": "SHARD_COMPLETE" if int(bucket_shard_count) > 1 else ("PASS" if summary["accounting_pass"] else "FAIL"),
        "runtime_sec": runtime_sec,
        "peak_memory_mb": peak_rss / (1024**2),
        "routing_cache_pairs": movement_router.cache_size,
        "edges_load_ms": edges_load_ms,
        "movements_load_ms": movements_load_ms,
        "candidate_index_init_ms": candidate_index_init_ms,
        "candidate_index_cache_hit": candidate_index.cache_hit,
        "movement_router_init_ms": movement_router_init_ms,
        "summary_build_ms": summary_build_ms,
        "case_trace_export_ms": case_trace_export_ms,
        "partition_runs": len(run_rows),
        "workers": int(workers),
        "bucket_shard_index": int(bucket_shard_index),
        "bucket_shard_count": int(bucket_shard_count),
        "orders_per_day": sample_count,
        "sampling_run_id": sample_run_id,
    })
    initialization_ms = edges_load_ms + movements_load_ms + candidate_index_init_ms + movement_router_init_ms
    profiled_ms = (
        initialization_ms + float(summary.get("pure_compute_ms", 0.0))
        + float(summary.get("bucket_input_ms", 0.0)) + float(summary.get("bucket_groupby_ms", 0.0))
        + float(summary.get("output_io_ms", 0.0))
        + float(summary.get("diagnostic_io_ms", 0.0))
        + summary_build_ms + case_trace_export_ms
    )
    summary["initialization_ms"] = initialization_ms
    summary["profiled_ms"] = profiled_ms
    summary["unprofiled_ms"] = max(0.0, runtime_sec * 1000.0 - profiled_ms)
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
    run_id = sampling_run_id(dates, orders_per_day, int(config.section("sampling")["seed"]))
    valid_parts: list[tuple[str, int]] = []
    for date in dates:
        for manifest_path in (
            output / "manifests" / "partitions" / f"day={date}"
        ).glob("part=*.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                manifest.get("status") == "PASS"
                and manifest.get("config_hash") == config.digest
                and manifest.get("sampling_run_id") == run_id
            ):
                valid_parts.append((date, int(manifest["bucket"])))
    quality_files = [_output_file(output, "route_quality", date, bucket) for date, bucket in valid_parts]
    order_files = [_output_file(output, "order_base", date, bucket) for date, bucket in valid_parts]
    performance_files = [_output_file(output, "performance", date, bucket) for date, bucket in valid_parts]
    traversal_files = [_output_file(output, "link_traversals", date, bucket) for date, bucket in valid_parts]
    movement_files = [_output_file(output, "turn_movements", date, bucket) for date, bucket in valid_parts]
    failed_files = [
        output / "failed_orders" / f"day={date}" / f"part={bucket:03d}.parquet"
        for date, bucket in valid_parts
        if (output / "failed_orders" / f"day={date}" / f"part={bucket:03d}.parquet").exists()
    ]
    quality = pd.concat([pd.read_parquet(path) for path in quality_files], ignore_index=True) if quality_files else pd.DataFrame()
    orders = pd.concat([pd.read_parquet(path) for path in order_files], ignore_index=True) if order_files else pd.DataFrame()
    performance = pd.concat([pd.read_parquet(path) for path in performance_files], ignore_index=True) if performance_files else pd.DataFrame()
    traversals = pd.concat([pd.read_parquet(path) for path in traversal_files], ignore_index=True) if traversal_files else pd.DataFrame()
    movements = pd.concat([pd.read_parquet(path) for path in movement_files], ignore_index=True) if movement_files else pd.DataFrame()
    failures = pd.concat([pd.read_parquet(path) for path in failed_files], ignore_index=True) if failed_files else pd.DataFrame()
    sampling_path = output / "manifests" / "sampling_runs" / run_id / "sampling_manifest.parquet"
    if sampling_path.exists():
        sampled = pd.read_parquet(sampling_path, columns=["date", "order_id"])
        expected_orders = int(len(sampled.loc[sampled.date.astype(str).isin(dates)]))
        keys = sampled.loc[sampled.date.astype(str).isin(dates), ["date", "order_id"]].astype(str)
        keys = keys.sort_values(["date", "order_id"], kind="stable")
        sample_sha = hashlib.sha256("\n".join(
            f"{date}|{order_id}" for date, order_id in keys.itertuples(index=False, name=None)
        ).encode("utf-8")).hexdigest()
    else:
        expected_orders = len(orders)
        sample_sha = None
    base = conservation_summary(quality, expected_orders) if len(quality) else {"accounting_pass": False, "input_orders": expected_orders, "output_orders": 0}
    modes = orders.matching_mode.value_counts(normalize=True).to_dict() if len(orders) else {}
    def numeric_sum(column: str) -> float:
        return float(pd.to_numeric(performance.get(column, pd.Series(dtype=float)), errors="coerce").fillna(0).sum())

    full_attempts = int(pd.Series(performance.get("full_hmm_attempted", False)).fillna(False).astype(bool).sum()) if len(performance) else 0
    full_successes = int(pd.Series(performance.get("full_hmm_succeeded", False)).fillna(False).astype(bool).sum()) if len(performance) else 0
    local_attempts = int(pd.Series(performance.get("local_hmm_attempted", False)).fillna(False).astype(bool).sum()) if len(performance) else 0
    pure_columns = (
        "matching_total_ms", "route_bridge_search_ms", "route_parts_build_ms",
        "traversal_build_ms", "movement_build_ms", "quality_ms",
    )
    pure_compute_ms = sum(numeric_sum(column) for column in pure_columns)
    precomputed_paths = numeric_sum("precomputed_path_count")
    reconstruction_requests = numeric_sum("reconstruction_bridge_request_count")
    reconstruction_searches = numeric_sum("reconstruction_path_search_count")
    duplicate_traversals = int(traversals.duplicated(["order_id", "traversal_id"]).sum()) if len(traversals) else 0
    inferred_dynamic = int(pd.to_numeric(
        quality.get("observed_dynamic_label_on_inferred_edge_count", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0).sum())
    mismatch_count = int(pd.to_numeric(
        quality.get("hmm_path_distance_mismatch_count", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0).sum())
    same_edge_jitter_mismatch_count = int(pd.to_numeric(
        quality.get(
            "same_edge_jitter_mismatch_count", pd.Series(dtype=float)
        ),
        errors="coerce",
    ).fillna(0).sum())
    non_rejected_gaps = int((
        quality.get("topology_gap_count", pd.Series(0, index=quality.index)).gt(0)
        & quality.route_quality.ne("rejected")
    ).sum()) if len(quality) else 0
    fallback_modes = {"pure_geometric_fallback", "partial_local_hmm_fallback"}
    fallback_share = float(orders.matching_mode.isin(fallback_modes).mean()) if len(orders) else 0.0
    percentile_columns = [
        "coordinate_transform_ms", "candidate_generation_ms", "ambiguity_detection_ms",
        "local_hmm_ms", "full_hmm_ms", "transition_search_ms",
        "selected_path_search_ms", "matching_total_ms", "route_parts_build_ms",
        "traversal_build_ms", "movement_build_ms", "quality_ms", "output_io_ms",
        "diagnostic_io_ms",
    ]
    performance_percentiles = {}
    for column in percentile_columns:
        values = pd.to_numeric(performance.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
        performance_percentiles[column] = {
            "mean": float(values.mean()) if len(values) else None,
            **{
                name: float(values.quantile(q)) if len(values) else None
                for name, q in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99))
            },
        }
    inferred_values = pd.to_numeric(
        quality.get("interpolated_distance_share", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    def inferred_summary(mask: pd.Series) -> dict[str, Any]:
        values = pd.to_numeric(
            quality.loc[mask, "interpolated_distance_share"],
            errors="coerce",
        ).dropna()
        return {
            "applicable_orders": int(len(values)),
            "mean": float(values.mean()) if len(values) else None,
            **{
                name: float(values.quantile(q)) if len(values) else None
                for name, q in (("q50", 0.50), ("q90", 0.90), ("q99", 0.99))
            },
        }
    successful_mask = quality.get(
        "successful_reconstruction", pd.Series(False, index=quality.index)
    ).fillna(False).astype(bool)
    strict_mask = quality.get(
        "strict_evaluation_eligible", pd.Series(False, index=quality.index)
    ).fillna(False).astype(bool)
    analysis_mask = quality.get(
        "formal_analysis_eligible", pd.Series(False, index=quality.index)
    ).fillna(False).astype(bool)
    failure_cross_tab = {}
    if len(orders) and "pre_transition_validation_mode" in orders:
        failure_cross_tab = {
            f"{before}->{after}": int(count)
            for (before, after), count in orders.groupby(
                ["pre_transition_validation_mode", "matching_mode"],
                dropna=False,
            ).size().items()
        }
    search_values = pd.to_numeric(
        performance.get("path_search_calls", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    expanded_values = pd.to_numeric(
        performance.get("dijkstra_expanded_nodes", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    failed_windows = pd.to_numeric(
        performance.get("local_failed_window_count", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0).astype(int)
    failed_transition_events: list[dict[str, Any]] = []
    for encoded in performance.get(
        "failed_transition_reasons", pd.Series(dtype=str)
    ).fillna("[]"):
        try:
            decoded = json.loads(str(encoded))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(decoded, list):
            failed_transition_events.extend(
                item for item in decoded if isinstance(item, dict)
            )
    failed_point_indices = pd.Series(
        [
            event.get("point_index")
            for event in failed_transition_events
            if event.get("point_index") is not None
        ],
        dtype=float,
    )
    raw_failure_status_counts: dict[str, int] = {}
    failure_diagnostic_class_counts: dict[str, int] = {}
    for event in failed_transition_events:
        status = str(event.get("raw_movement_status", "unknown"))
        raw_failure_status_counts[status] = (
            raw_failure_status_counts.get(status, 0) + 1
        )
        diagnostic_class = str(
            event.get("diagnostic_class", "unclassified")
        )
        failure_diagnostic_class_counts[diagnostic_class] = (
            failure_diagnostic_class_counts.get(diagnostic_class, 0) + 1
        )
    return {
        **base,
        "dates": dates,
        "matching_mode_share": {str(k): float(v) for k, v in modes.items()},
        "sampling_seed": int(config.section("sampling")["seed"]),
        "sampling_run_id": run_id,
        "sample_order_sha256": sample_sha,
        "fallback_share": fallback_share,
        "topology_gap_count": int(quality.topology_gap_count.sum()) if len(quality) else 0,
        "parallel_ambiguity_order_count": int((quality.parallel_ambiguity_share > 0).sum()) if len(quality) else 0,
        "direction_violation_count": int(quality.direction_violation_count.sum()) if len(quality) else 0,
        "layer_violation_count": int(quality.layer_violation_count.sum()) if len(quality) else 0,
        "restriction_violation_count": int(quality.restriction_violation_count.sum()) if len(quality) else 0,
        "mean_inferred_distance_share": (
            float(inferred_values.mean()) if len(inferred_values) else None
        ),
        "inferred_distance_scope_note": (
            "Failed reconstructions are N/A, never zero-filled."
        ),
        "inferred_distance_share_by_scope": {
            "successfully_reconstructed": inferred_summary(successful_mask),
            "strict_core": inferred_summary(strict_mask),
            "analysis_eligible": inferred_summary(analysis_mask),
        },
        "inferred_distance_share_distribution": {
            name: float(inferred_values.quantile(q)) if len(inferred_values) else None
            for name, q in (("q50", 0.50), ("q90", 0.90), ("q99", 0.99))
        },
        "path_to_gps_ratio_distribution": {
            name: float(pd.to_numeric(
                quality.get(column, pd.Series(dtype=float)), errors="coerce"
            ).dropna().quantile(q)) if len(pd.to_numeric(
                quality.get(column, pd.Series(dtype=float)), errors="coerce"
            ).dropna()) else None
            for name, column, q in (
                ("q50", "path_to_gps_ratio_q50", 0.50),
                ("q90", "path_to_gps_ratio_q90", 0.90),
                ("q99", "path_to_gps_ratio_q99", 0.99),
            )
        },
        "full_hmm_attempt_count": full_attempts,
        "full_hmm_success_count": full_successes,
        "full_hmm_failure_count": full_attempts - full_successes,
        "full_hmm_attempt_share": full_attempts / len(performance) if len(performance) else 0.0,
        "full_hmm_failure_share": (full_attempts - full_successes) / len(performance) if len(performance) else 0.0,
        "local_hmm_attempt_count": local_attempts,
        "local_hmm_attempt_share": local_attempts / len(performance) if len(performance) else 0.0,
        "local_hmm_window_attempt_count": int(
            numeric_sum("local_hmm_window_attempt_count")
        ),
        "local_hmm_retry_window_count": int(
            numeric_sum("local_hmm_retry_window_count")
        ),
        "boundary_repair_viterbi_count": int(
            numeric_sum("boundary_repair_viterbi_count")
        ),
        "local_window_failure_count": int(numeric_sum("local_failed_window_count")),
        "orders_by_failed_local_window_count": {
            "1": int((failed_windows == 1).sum()),
            "2": int((failed_windows == 2).sum()),
            "3": int((failed_windows == 3).sum()),
            "4_plus": int((failed_windows >= 4).sum()),
        },
        "local_boundary_transition_failure_count": int(
            numeric_sum("local_boundary_failure_count")
        ),
        "local_internal_transition_failure_count": int(
            numeric_sum("local_internal_failure_count")
        ),
        "failed_transition_reason_counts": {
            "endpoint_distance_exceeds_cutoff": int(
                numeric_sum("endpoint_distance_exceeds_cutoff_count")
            ),
            "no_movement_path_within_cutoff": int(
                numeric_sum("no_movement_path_within_cutoff_count")
            ),
        },
        "failed_transition_raw_movement_status_counts": raw_failure_status_counts,
        "failed_transition_raw_movement_status_semantics": (
            "no_direct_raw_movement_record means the candidate pair has no "
            "single direct movement record; it does not imply that no "
            "multi-edge network path exists"
        ),
        "failed_transition_diagnostic_class_counts": (
            failure_diagnostic_class_counts
        ),
        "failed_transition_point_index_distribution": {
            name: float(failed_point_indices.quantile(q))
            if len(failed_point_indices)
            else None
            for name, q in (("p50", 0.50), ("p90", 0.90), ("p99", 0.99))
        },
        "candidate_true_state_missing_proxy": {
            "orders_with_under_minimum_initial_candidates": int((
                pd.to_numeric(
                    performance.get(
                        "under_minimum_candidate_initial_count",
                        pd.Series(dtype=float),
                    ),
                    errors="coerce",
                ).fillna(0) > 0
            ).sum()),
            "under_minimum_candidate_point_count": int(
                numeric_sum("under_minimum_candidate_initial_count")
            ),
            "unrecovered_no_candidate_point_count": int(
                numeric_sum("no_candidate_initial_count")
                - numeric_sum("no_candidate_recovered_count")
            ),
        },
        "pre_validation_to_final_mode_cross_tab": failure_cross_tab,
        "path_searches_per_order_distribution": {
            name: float(search_values.quantile(q)) if len(search_values) else None
            for name, q in (("p50", 0.50), ("p90", 0.90), ("p99", 0.99))
        },
        "expanded_states_per_order_distribution": {
            name: float(expanded_values.quantile(q)) if len(expanded_values) else None
            for name, q in (("p50", 0.50), ("p90", 0.90), ("p99", 0.99))
        },
        "exact_path_search_calls": int(numeric_sum("exact_path_search_calls")),
        "approximate_path_search_calls": int(
            numeric_sum("approximate_path_search_calls")
        ),
        "approximate_search_unresolved_count": int(
            numeric_sum("approximate_search_unresolved_count")
        ),
        "order_transition_evidence_cache_hits": int(
            numeric_sum("order_transition_evidence_cache_hits")
        ),
        "order_transition_evidence_cache_misses": int(
            numeric_sum("order_transition_evidence_cache_misses")
        ),
        "pure_compute_ms": pure_compute_ms,
        "pure_compute_ms_per_order": pure_compute_ms / len(performance) if len(performance) else 0.0,
        "bucket_input_ms": numeric_sum("bucket_input_ms_per_order"),
        "bucket_groupby_ms": numeric_sum("bucket_groupby_ms_per_order"),
        "output_io_ms": numeric_sum("output_io_ms"),
        "diagnostic_io_ms": numeric_sum("diagnostic_io_ms"),
        "precomputed_path_reuse_count": int(precomputed_paths),
        "reconstruction_bridge_request_count": int(reconstruction_requests),
        "reconstruction_path_search_count": int(reconstruction_searches),
        "selected_path_reuse_share": precomputed_paths / max(precomputed_paths + reconstruction_requests, 1.0),
        "processing_exception_count": int(len(failures)),
        "internal_time_conservation_failures": int((quality.time_allocation_error_s > 1e-6).sum()) if len(quality) else 0,
        "internal_distance_conservation_failures": int((quality.internal_distance_error_m > 1e-6).sum()) if len(quality) else 0,
        "duplicate_traversal_instance_error_count": duplicate_traversals,
        "invalid_position_aware_distance_count": int(quality.invalid_position_aware_distance_count.sum()) if len(quality) else 0,
        "position_audit_applicable_order_count": int(pd.to_numeric(
            quality.get("position_audit_applicable_order_count", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0).sum()),
        "position_audit_not_applicable_match_failure_count": int(pd.to_numeric(
            quality.get(
                "position_audit_not_applicable_match_failure_count",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        ).fillna(0).sum()),
        "actual_invalid_position_event_count": int(pd.to_numeric(
            quality.get("actual_invalid_position_event_count", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0).sum()),
        "observed_dynamic_label_on_inferred_edge_count": inferred_dynamic,
        "hmm_path_distance_mismatch_count": mismatch_count,
        "same_edge_jitter_mismatch_count": same_edge_jitter_mismatch_count,
        "final_forbidden_movement_count": int(movements.get(
            "movement_audit_reason", pd.Series(dtype=str)
        ).astype(str).eq("restriction_block").sum()) if len(movements) else 0,
        "non_rejected_topology_gap_order_count": non_rejected_gaps,
        "raw_movement_audit_available": bool(
            len(movements) == 0 or "movement_audit_reason" in movements.columns
        ),
        "performance_percentiles_ms": performance_percentiles,
    }
