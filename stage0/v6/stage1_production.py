"""Resumable, stable-hash production of the frozen Stage 1 input contract."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from stage0.v5.archive import (
    extract_daily_archive,
    iter_daily_chunks,
    list_archive_members,
    scan_daily_orders,
)
from stage0.v5.config import stable_hash

from .config import Stage0V6Config
from .coordinates import haversine_m
from .order_processor import ProcessedOrder, Stage0OrderProcessor
from .pipeline import _write_json, _write_product


CORE_PRODUCTS = (
    "order_base",
    "route_segments",
    "route_parts",
    "link_traversals",
    "link_interval_observations",
    "interval_measurements",
    "turn_movements",
    "gps_quality",
    "route_quality",
    "dynamic_quality",
    "canonical_quality",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_sha256_file(file_path)))
    return digest.hexdigest()


def _git_sha(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _code_identity(config: Stage0V6Config) -> str:
    """Identify the exact executable tree even before the final Git commit."""

    git_sha = _git_sha(config.repo_root)
    digest = hashlib.sha256()
    sources = sorted(
        [
            *(
                config.repo_root / "stage0" / "v6"
            ).glob("*.py"),
            config.source,
        ],
        key=lambda path: path.as_posix(),
    )
    for path in sources:
        digest.update(path.relative_to(config.repo_root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_sha256_file(path)))
    return f"{git_sha}+content.{digest.hexdigest()[:16]}"


def _selection_hex(date: str, order_id: str, seed: int) -> str:
    return f"{stable_hash(date, order_id, seed=seed):016x}"


def _prefilter_order(
    raw: pd.DataFrame, date: str, selection_hash: str, settings: dict[str, Any]
) -> dict[str, Any]:
    frame = raw.copy()
    timestamp = pd.to_numeric(frame.timestamp, errors="coerce")
    lon = pd.to_numeric(frame.lon, errors="coerce")
    lat = pd.to_numeric(frame.lat, errors="coerce")
    valid_mask = (
        timestamp.notna()
        & lon.between(-180, 180)
        & lat.between(-90, 90)
    )
    valid = pd.DataFrame(
        {"timestamp": timestamp[valid_mask], "lon": lon[valid_mask], "lat": lat[valid_mask]}
    ).sort_values("timestamp", kind="stable")
    raw_count = int(len(frame))
    valid_count = int(len(valid))
    invalid_share = 1.0 - valid_count / max(raw_count, 1)
    # The daily archive does not preserve point order: rows for an order can be
    # interleaved and arrive in arbitrary chunk order.  A timestamp reversal is
    # therefore only meaningful when the source supplies an explicit sequence.
    sequence_column = next(
        (
            column
            for column in ("point_seq", "gps_sequence", "source_sequence")
            if column in frame.columns
        ),
        None,
    )
    if sequence_column is None:
        reverse_share = 0.0
    else:
        ordered_timestamp = timestamp.loc[
            pd.to_numeric(frame[sequence_column], errors="coerce")
            .sort_values(kind="stable")
            .index
        ]
        reverse_share = float(
            ordered_timestamp.diff().lt(0).sum() / max(raw_count - 1, 1)
        )
    duplicate_share = float(
        frame.duplicated(["timestamp", "lon", "lat"]).sum() / max(raw_count, 1)
    )
    duration = (
        float(valid.timestamp.iloc[-1] - valid.timestamp.iloc[0])
        if len(valid) > 1
        else 0.0
    )
    distances = np.zeros(len(valid), dtype=float)
    if len(valid) > 1:
        distances[1:] = haversine_m(
            valid.lon.to_numpy()[:-1],
            valid.lat.to_numpy()[:-1],
            valid.lon.to_numpy()[1:],
            valid.lat.to_numpy()[1:],
        )
    time_steps = valid.timestamp.diff().to_numpy(float)
    speeds = np.divide(
        distances,
        time_steps,
        out=np.full(len(valid), np.inf),
        where=time_steps > 0,
    )
    speeds[0] = 0.0
    impossible_share = float(
        np.mean(speeds[1:] > float(settings["impossible_speed_mps"]))
    ) if len(speeds) > 1 else 1.0
    od_distance = (
        float(
            haversine_m(
                np.asarray([valid.lon.iloc[0]]),
                np.asarray([valid.lat.iloc[0]]),
                np.asarray([valid.lon.iloc[-1]]),
                np.asarray([valid.lat.iloc[-1]]),
            )[0]
        )
        if len(valid) > 1
        else 0.0
    )
    reasons: list[str] = []
    if valid_count < int(settings["minimum_valid_points"]):
        reasons.append("TOO_FEW_VALID_POINTS")
    if duration < float(settings["minimum_duration_s"]):
        reasons.append("DURATION_TOO_SHORT")
    if invalid_share > float(settings["maximum_invalid_coordinate_share"]):
        reasons.append("INVALID_COORDINATE_SHARE")
    if reverse_share > float(settings["maximum_reverse_timestamp_share"]):
        reasons.append("TIMESTAMP_SEVERELY_REVERSED")
    if duplicate_share > float(settings["maximum_duplicate_point_share"]):
        reasons.append("DUPLICATE_POINT_SHARE")
    if od_distance < float(settings["minimum_od_distance_m"]):
        reasons.append("OD_DISTANCE_TOO_SHORT")
    if impossible_share > float(
        settings["maximum_impossible_speed_interval_share"]
    ):
        reasons.append("IMPOSSIBLE_SPEED_DOMINATES")
    return {
        "order_id": str(frame.order_id.iloc[0]),
        "date": str(date),
        "candidate_priority_hash": selection_hash,
        "pre_match_eligible": not reasons,
        "pre_match_rejection_reason": "|".join(reasons),
        "raw_point_count": raw_count,
        "valid_point_count": valid_count,
        "valid_duration_s": duration,
        "valid_gps_distance_m": float(distances.sum()),
    }


def _prepare_day_cache(
    config: Stage0V6Config,
    tar_path: Path,
    candidates: pd.DataFrame,
    date: str,
) -> Path:
    settings = config.section("production")
    cache = config.path("work") / "candidate_cache" / f"date={date}"
    success = cache / "_SUCCESS.json"
    if success.exists():
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    lookup = candidates.set_index("order_id").candidate_bucket.to_dict()
    fragment = 0
    rows = 0
    for chunk in iter_daily_chunks(
        tar_path, chunksize=int(settings.get("raw_chunk_size", 500_000))
    ):
        order_ids = chunk.order_id.astype(str)
        selected = order_ids.isin(lookup)
        if not selected.any():
            continue
        materialized = chunk.loc[selected].copy()
        materialized["date"] = date
        materialized["candidate_bucket"] = materialized.order_id.astype(str).map(
            lookup
        )
        rows += len(materialized)
        for bucket, group in materialized.groupby("candidate_bucket", sort=False):
            target = cache / f"bucket={int(bucket):05d}"
            target.mkdir(parents=True, exist_ok=True)
            _write_product(
                group.drop(columns="candidate_bucket"),
                target / f"fragment={fragment:06d}.parquet",
            )
            fragment += 1
    _write_json(
        success,
        {
            "date": date,
            "candidate_count": int(len(candidates)),
            "point_row_count": rows,
            "fragment_count": fragment,
        },
    )
    return cache


def _frames_for_accepted(
    date: str,
    split: str,
    selection_hash: str,
    result: ProcessedOrder,
) -> dict[str, pd.DataFrame]:
    clean = result.preprocess.points
    route_parts = result.products["route_parts"]
    final = result.final_quality.order_quality
    accounting = result.products["interval_accounting"].iloc[0].to_dict()
    order_id = str(clean.order_id.iloc[0])
    order_base = pd.DataFrame(
        [
            {
                "order_id": order_id,
                "date": date,
                "split": split,
                "departure_time": float(clean.timestamp.min()),
                "arrival_time": float(clean.timestamp.max()),
                "start_node": (
                    route_parts.canonical_from_node.iloc[0]
                    if len(route_parts)
                    else pd.NA
                ),
                "end_node": (
                    route_parts.canonical_to_node.iloc[-1]
                    if len(route_parts)
                    else pd.NA
                ),
                "gps_status": final["gps_status"],
                "route_status": final["route_status"],
                "dynamic_status": final["dynamic_status"],
                "canonical_status": final["canonical_status"],
                "selection_hash": selection_hash,
                "stage1_core_eligible": True,
            }
        ]
    )
    gps_fields = {
        key: value
        for key, value in final.items()
        if key == "order_id"
        or key.startswith("gps_")
        or key.startswith("has_")
        or key.startswith("outlier_")
    }
    route_fields = {
        **result.route_quality,
        **{
            key: value
            for key, value in final.items()
            if key
            in {
                "order_id",
                "route_status",
                "raw_gps_route_distance_p50_m",
                "raw_gps_route_distance_p90_m",
                "raw_gps_route_distance_p99_m",
                "raw_gps_route_distance_max_m",
                "raw_gps_buffer20_coverage_share",
                "raw_gps_buffer40_coverage_share",
                "raw_gps_buffer80_coverage_share",
                "longest_uncovered_point_count",
                "longest_uncovered_time_s",
                "longest_uncovered_gps_distance_m",
                "uncovered_time_share",
                "uncovered_distance_share",
                "route_component_count",
                "main_component_distance_share",
                "isolated_component_distance_share",
                "parallel_risk",
                "local_retry_attempted",
                "local_retry_changed_route",
                "local_retry_improvement",
                "local_retry_reason",
            }
            or key.startswith("discontinuity_")
            or key.startswith("maximum_continuous_gap")
            or key.startswith("gap_")
            or key.startswith("parallel_")
        },
    }
    dynamic_fields = {
        **result.dynamic_quality,
        **accounting,
        "dynamic_status": final["dynamic_status"],
    }
    canonical_fields = {
        "order_id": order_id,
        "canonical_status": final["canonical_status"],
        "canonical_edge_count": int(
            route_parts.canonical_edge_uid.dropna().nunique()
        ),
    }
    direct = result.products["link_interval_observations"]
    if len(direct):
        direct = direct.loc[
            direct.measurement_source.eq("direct_observed")
            & direct.label_valid.fillna(False)
        ].copy()
    return {
        "order_base": order_base,
        "route_segments": result.final_quality.route_segments,
        "route_parts": route_parts,
        "link_traversals": result.products["link_traversals"],
        "link_interval_observations": direct,
        "interval_measurements": result.products["interval_measurements"],
        "turn_movements": result.products["turn_movements"],
        "gps_quality": pd.DataFrame([gps_fields]),
        "route_quality": pd.DataFrame([route_fields]),
        "dynamic_quality": pd.DataFrame([dynamic_fields]),
        "canonical_quality": pd.DataFrame([canonical_fields]),
    }


def _write_stage1_bucket(
    root: Path,
    split: str,
    date: str,
    bucket: int,
    frames: dict[str, list[pd.DataFrame]],
    manifest: dict[str, Any],
) -> None:
    target = root / f"split={split}" / f"date={date}" / f"bucket={bucket:05d}"
    target.mkdir(parents=True, exist_ok=True)
    row_counts: dict[str, int] = {}
    for product in CORE_PRODUCTS:
        values = [frame for frame in frames[product] if len(frame)]
        output = pd.concat(values, ignore_index=True) if values else pd.DataFrame()
        row_counts[product] = int(len(output))
        _write_product(output, target / f"{product}.parquet")
    manifest["product_row_counts"] = row_counts
    manifest["status"] = (
        "PASS"
        if manifest["processing_exception_count"] == 0
        else "FAIL"
    )
    _write_json(target / "manifest.json", manifest)


def build_stage1_input(
    config: Stage0V6Config, *, resume: bool = True
) -> dict[str, Any]:
    """Process candidates in stable order until each frozen quota is met."""

    production = config.section("production")
    archive = config.path("archive")
    seven_zip = config.path("seven_zip")
    members = {
        row["date"]: row
        for row in list_archive_members(archive, seven_zip)
    }
    schedule: list[tuple[str, str, int]] = []
    schedule.extend(
        ("train", str(date), int(production["train_daily_target"]))
        for date in production["train_dates"]
    )
    schedule.extend(
        ("validation", str(date), int(production["validation_daily_target"]))
        for date in production["validation_dates"]
    )
    schedule.append(
        ("test", str(production["test_date"]), int(production["test_target"]))
    )
    missing = [date for _, date, _ in schedule if date not in members]
    if missing:
        raise RuntimeError(f"archive is missing production dates: {missing}")
    root = config.path("stage1_input")
    root.mkdir(parents=True, exist_ok=True)
    processor = Stage0OrderProcessor(config)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    seed = int(production["selection_seed"])
    bucket_size = int(production["candidate_bucket_size"])
    code_sha = _code_identity(config)
    tiles_sha = _sha256_directory(config.path("valhalla_tiles"))
    started_all = time.perf_counter()
    daily_results: list[dict[str, Any]] = []

    for split, date, target_count in schedule:
        day_started = time.perf_counter()
        daily_manifest_path = (
            root
            / "manifests"
            / f"split={split}"
            / f"date={date}.json"
        )
        if daily_manifest_path.exists() and resume:
            previous_daily = json.loads(
                daily_manifest_path.read_text(encoding="utf-8")
            )
            if previous_daily.get("completed", False):
                daily_results.append(previous_daily)
                continue
        member = members[date]
        tar_path = extract_daily_archive(
            archive,
            member["source_member"],
            seven_zip,
            config.path("work"),
        )
        candidate_manifest_path = (
            config.path("work")
            / "candidate_manifests"
            / f"date={date}.parquet"
        )
        if candidate_manifest_path.exists() and resume:
            candidates = pd.read_parquet(candidate_manifest_path)
        else:
            candidates = scan_daily_orders(
                tar_path, date, member["source_member"]
            )
            candidates["selection_hash"] = candidates.order_id.astype(str).map(
                lambda order_id: _selection_hex(date, order_id, seed)
            )
            candidates.sort_values(
                ["selection_hash", "order_id"], kind="stable", inplace=True
            )
            candidates.reset_index(drop=True, inplace=True)
            candidates["candidate_bucket"] = (
                np.arange(len(candidates), dtype=np.int64) // bucket_size
            )
            candidate_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            _write_product(candidates, candidate_manifest_path)
        cache = _prepare_day_cache(config, tar_path, candidates, date)
        accepted_so_far = 0
        processed_so_far = 0
        exceptions_so_far = 0
        existing_manifests = sorted(
            (
                root / f"split={split}" / f"date={date}"
            ).glob("bucket=*/manifest.json")
        )
        if resume:
            for path in existing_manifests:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") == "PASS":
                    accepted_so_far += int(payload["accepted_core_count"])
                    processed_so_far += int(payload["matched_count"])
                    exceptions_so_far += int(
                        payload["processing_exception_count"]
                    )
        for bucket, candidate_rows in candidates.groupby(
            "candidate_bucket", sort=True
        ):
            if accepted_so_far >= target_count:
                break
            bucket = int(bucket)
            output_manifest = (
                root
                / f"split={split}"
                / f"date={date}"
                / f"bucket={bucket:05d}"
                / "manifest.json"
            )
            if output_manifest.exists() and resume:
                payload = json.loads(output_manifest.read_text(encoding="utf-8"))
                if payload.get("status") == "PASS":
                    continue
            bucket_started = time.perf_counter()
            fragments = sorted(
                (cache / f"bucket={bucket:05d}").glob("fragment=*.parquet")
            )
            raw_bucket = (
                pd.concat(
                    [pd.read_parquet(path) for path in fragments],
                    ignore_index=True,
                )
                if fragments
                else pd.DataFrame()
            )
            raw_groups = {
                str(order_id): group
                for order_id, group in raw_bucket.groupby(
                    raw_bucket.order_id.astype(str), sort=False
                )
            } if len(raw_bucket) else {}
            frames = {product: [] for product in CORE_PRODUCTS}
            rejection_rows: list[dict[str, Any]] = []
            accepted = rejected = matched = exceptions = preeligible = 0
            for candidate in candidate_rows.sort_values(
                ["selection_hash", "order_id"], kind="stable"
            ).itertuples(index=False):
                if accepted_so_far + accepted >= target_count:
                    break
                order_id = str(candidate.order_id)
                raw_order = raw_groups.get(order_id)
                if raw_order is None:
                    exceptions += 1
                    rejection_rows.append(
                        {
                            "order_id": order_id,
                            "date": date,
                            "selection_hash": candidate.selection_hash,
                            "pre_match_status": "missing_materialized_points",
                            "gps_status": "",
                            "route_status": "",
                            "dynamic_status": "",
                            "canonical_status": "",
                            "rejection_reason": "MISSING_MATERIALIZED_POINTS",
                        }
                    )
                    continue
                prefilter = _prefilter_order(
                    raw_order,
                    date,
                    str(candidate.selection_hash),
                    production["prefilter"],
                )
                if not prefilter["pre_match_eligible"]:
                    rejected += 1
                    rejection_rows.append(
                        {
                            "order_id": order_id,
                            "date": date,
                            "selection_hash": candidate.selection_hash,
                            "pre_match_status": "rejected",
                            "gps_status": "sparse_or_ineligible",
                            "route_status": "",
                            "dynamic_status": "",
                            "canonical_status": "",
                            "rejection_reason": prefilter[
                                "pre_match_rejection_reason"
                            ],
                            **prefilter,
                        }
                    )
                    continue
                preeligible += 1
                try:
                    result = processor.process(raw_order)
                    matched += 1
                except Exception as exc:
                    exceptions += 1
                    rejection_rows.append(
                        {
                            "order_id": order_id,
                            "date": date,
                            "selection_hash": candidate.selection_hash,
                            "pre_match_status": "eligible",
                            "gps_status": "",
                            "route_status": "",
                            "dynamic_status": "",
                            "canonical_status": "",
                            "rejection_reason": (
                                f"PROCESSING_EXCEPTION:{type(exc).__name__}:{exc}"
                            ),
                        }
                    )
                    continue
                final = result.final_quality.order_quality
                if result.stage1_core_eligible:
                    accepted += 1
                    for product, frame in _frames_for_accepted(
                        date,
                        split,
                        str(candidate.selection_hash),
                        result,
                    ).items():
                        frames[product].append(frame)
                else:
                    rejected += 1
                    rejection_rows.append(
                        {
                            "order_id": order_id,
                            "date": date,
                            "selection_hash": candidate.selection_hash,
                            "pre_match_status": "eligible",
                            "gps_status": final["gps_status"],
                            "route_status": final["route_status"],
                            "dynamic_status": final["dynamic_status"],
                            "canonical_status": final["canonical_status"],
                            "rejection_reason": result.stage1_rejection_reason,
                        }
                    )
                peak_rss = max(peak_rss, process.memory_info().rss)
            manifest = {
                "schema_version": "stage1_input_bucket.1",
                "split": split,
                "date": date,
                "bucket": bucket,
                "candidate_count": int(len(candidate_rows)),
                "pre_match_eligible_count": preeligible,
                "matched_count": matched,
                "accepted_core_count": accepted,
                "rejected_count": rejected,
                "processing_exception_count": exceptions,
                "runtime_s": time.perf_counter() - bucket_started,
                "peak_rss_mb": peak_rss / (1024**2),
                "code_sha": code_sha,
                "config_sha": config.digest,
                "tiles_sha": tiles_sha,
            }
            _write_stage1_bucket(
                root, split, date, bucket, frames, manifest
            )
            rejection_target = (
                root
                / "rejections"
                / f"split={split}"
                / f"date={date}"
                / f"bucket={bucket:05d}.parquet"
            )
            _write_product(pd.DataFrame(rejection_rows), rejection_target)
            accepted_so_far += accepted
            processed_so_far += matched
            exceptions_so_far += exceptions
            del raw_bucket, raw_groups, frames
        result = {
            "split": split,
            "date": date,
            "target_count": target_count,
            "accepted_count": accepted_so_far,
            "daily_quota_shortfall": max(target_count - accepted_so_far, 0),
            "matched_count": processed_so_far,
            "processing_exception_count": exceptions_so_far,
            "runtime_s": time.perf_counter() - day_started,
            "completed": True,
        }
        daily_results.append(result)
        _write_json(
            daily_manifest_path,
            result,
        )
        work_root = config.path("work").resolve()
        for generated_cache in (cache, tar_path.parent):
            resolved_cache = generated_cache.resolve()
            if work_root not in resolved_cache.parents:
                raise RuntimeError(
                    f"refusing to remove cache outside work root: {resolved_cache}"
                )
            if resolved_cache.exists():
                shutil.rmtree(resolved_cache)
    summary = {
        "schema_version": "stage1_input_production.1",
        "status": (
            "PASS"
            if sum(row["processing_exception_count"] for row in daily_results)
            == 0
            else "FAIL"
        ),
        "daily": daily_results,
        "accepted_total": int(
            sum(row["accepted_count"] for row in daily_results)
        ),
        "target_total": int(sum(row["target_count"] for row in daily_results)),
        "daily_quota_shortfall": int(
            sum(row["daily_quota_shortfall"] for row in daily_results)
        ),
        "processing_exception_count": int(
            sum(row["processing_exception_count"] for row in daily_results)
        ),
        "runtime_s": time.perf_counter() - started_all,
        "peak_rss_mb": peak_rss / (1024**2),
        "code_sha": code_sha,
        "config_sha": config.digest,
        "tiles_sha": tiles_sha,
    }
    _write_json(root / "manifests" / "production_summary.json", summary)
    return summary


def verify_stage1_input(
    config: Stage0V6Config, input_root: Path | None = None
) -> dict[str, Any]:
    root = input_root or config.path("stage1_input")
    manifests = sorted(root.glob("split=*/date=*/bucket=*/manifest.json"))
    failures: list[str] = []
    accepted = 0
    exceptions = 0
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        accepted += int(payload.get("accepted_core_count", 0))
        exceptions += int(payload.get("processing_exception_count", 0))
        for product in CORE_PRODUCTS:
            file_path = path.parent / f"{product}.parquet"
            if not file_path.exists():
                failures.append(f"missing:{file_path}")
                continue
            observed = len(pd.read_parquet(file_path))
            expected = int(payload["product_row_counts"].get(product, 0))
            if observed != expected:
                failures.append(
                    f"row_count:{file_path}:{observed}!={expected}"
                )
    result = {
        "schema_version": "stage1_input_verification.1",
        "status": "PASS" if manifests and not failures and exceptions == 0 else "FAIL",
        "bucket_manifest_count": len(manifests),
        "accepted_order_count": accepted,
        "processing_exception_count": exceptions,
        "failure_count": len(failures),
        "failures": failures[:100],
    }
    _write_json(root / "manifests" / "verification.json", result)
    return result
