"""Isolated frozen-upstream materialization for preregistered final dates.

This module does not refit Stage 1 or Stage 2 v4 objects.  It applies the
frozen Stage 1 model bundle and frozen v4 Train history to Stage 0 products in
an isolated Stage 2 v5 directory.  Date admission is owned by the v5 split
freeze, while all Stage 1 bucket-level validation and label arithmetic are
reused unchanged.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from stage1.v3.config import load_config as load_stage1_config
from stage1.v3.input_adapter import BucketRef, Stage0Bucket
from stage1.v3 import input_adapter as stage1_adapter
from stage1.v3.io import (
    atomic_output_directory,
    atomic_write_json,
    atomic_write_parquet,
    bucket_input_identity,
    parquet_schema_sha256,
    sha256_file,
)
from stage1.v3.models import load_model_bundle
from stage1.v3.pipeline import OUTPUT_PRODUCTS, _transform_bucket, _validate_output_frames
from stage2.v4.config import load_config as load_stage2_v4_config
from stage2.v4.dataset_builder import (
    _AtomicDailyWriter,
    _base_route_tokens,
    _finalize_track,
)
from stage2.v4.history_index import TemporalHistoryIndex
from stage2.v4.history_store import validate_history_store
from stage2.v4.stage1_adapter import discover_stage1_buckets

from .config import load_config as load_stage2_v5_config
from .data import load_v5_day
from .shards import _atomic_json, _atomic_npz, _sha256, vectorized_chunk_payload


FINAL_ROOT = Path("stage2/output_v5/final_upstream")
ALLOWED_STAGE0_CONFIG_DELTAS = frozenset(
    {
        "paths.output",
        "paths.work",
        "paths.stage1_input",
        "production.train_dates",
        "production.validation_dates",
        "production.test_date",
        "production.test_target",
    }
)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(item, name))
        return result
    return {prefix: value}


def audit_stage0_config_delta(
    frozen_path: str | Path,
    final_path: str | Path,
    *,
    v5_config_path: str | Path,
) -> dict[str, Any]:
    frozen = yaml.safe_load(Path(frozen_path).read_text(encoding="utf-8"))
    final = yaml.safe_load(Path(final_path).read_text(encoding="utf-8"))
    v5 = load_stage2_v5_config(v5_config_path)
    left = _flatten(frozen)
    right = _flatten(final)
    changed = sorted(key for key in left.keys() | right.keys() if left.get(key) != right.get(key))
    unexpected = sorted(set(changed) - ALLOWED_STAGE0_CONFIG_DELTAS)
    expected_dates = v5.section("split")["final_test_dates"]
    materialized_dates = [*final["production"]["validation_dates"], final["production"]["test_date"]]
    checks = {
        "only_isolation_and_date_fields_changed": not unexpected,
        "final_dates_match_split_freeze": materialized_dates == expected_dates,
        "no_stage0_train_dates": final["production"]["train_dates"] == [],
        "daily_quota_is_10000": final["production"]["validation_daily_target"] == 10000 and final["production"]["test_target"] == 10000,
        "selection_seed_unchanged": final["production"]["selection_seed"] == frozen["production"]["selection_seed"],
        "quality_sections_identical": all(final[name] == frozen[name] for name in ("preprocess", "modeling_eligibility", "valhalla", "quality", "products", "final_quality", "stage1_core")),
    }
    return {
        "schema_version": "stage2_v5_final_upstream_plan.1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "changed_fields": changed,
        "unexpected_changed_fields": unexpected,
        "materialized_dates": materialized_dates,
        "frozen_stage0_config_sha256": _sha256(Path(frozen_path)),
        "final_stage0_config_sha256": _sha256(Path(final_path)),
        "final_stage0_output_root": str(final["paths"]["stage1_input"]),
        "new_final_test_consumed": False,
    }


def _discover_stage0_final(input_root: Path, dates: Iterable[str]) -> list[BucketRef]:
    allowed = set(dates)
    refs: list[BucketRef] = []
    for manifest_path in sorted(input_root.glob("split=*/date=*/bucket=*/manifest.json")):
        parts = {
            item.split("=", 1)[0]: item.split("=", 1)[1]
            for item in manifest_path.parent.relative_to(input_root).parts
        }
        date = parts["date"]
        if date not in allowed:
            raise ValueError(f"unexpected date in final Stage 0 input: {date}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        refs.append(BucketRef(parts["split"], date, int(parts["bucket"]), manifest_path.parent, manifest))
    actual = {ref.date for ref in refs}
    if actual != allowed:
        raise ValueError(f"final Stage 0 dates incomplete: expected={sorted(allowed)}, actual={sorted(actual)}")
    return refs


def _load_final_bucket(ref: BucketRef, config) -> Stage0Bucket:
    """Frozen Stage 1 loader with only its old date allowlist externalized."""

    stage1_adapter._validate_manifest(ref)
    disk_manifest = stage1_adapter._read_manifest(ref.path / "manifest.json")
    if disk_manifest != ref.manifest:
        raise ValueError(f"Stage 0 manifest changed while reading {ref.path}")
    products = {
        product: stage1_adapter._read_product(ref, product)
        for product in stage1_adapter.REQUIRED_PRODUCTS
    }
    stage1_adapter._normalize_nullable_dtypes(products)
    stage1_adapter._enrich_direction_lineage(products)
    stage1_adapter._fill_nullable_order_endpoints(products)
    bucket = Stage0Bucket(
        order_base=products["order_base"],
        route_parts=products["route_parts"],
        link_traversals=products["link_traversals"],
        link_interval_observations=products["link_interval_observations"],
        interval_measurements=products["interval_measurements"],
        turn_movements=products["turn_movements"],
        gps_quality=products["gps_quality"],
        route_quality=products["route_quality"],
        dynamic_quality=products["dynamic_quality"],
        canonical_quality=products["canonical_quality"],
    )
    order_ids = stage1_adapter._validate_order_base(ref, bucket.order_base)
    stage1_adapter._validate_measurement_sources(bucket.route_parts, "route_parts")
    stage1_adapter._validate_measurement_sources(bucket.link_traversals, "link_traversals")
    stage1_adapter._validate_direct_intervals(bucket.link_interval_observations, config)
    stage1_adapter._validate_primary_and_foreign_keys(bucket, order_ids)
    stage1_adapter._validate_traversal_observation_totals(bucket.link_traversals, bucket.link_interval_observations, config)
    stage1_adapter._validate_interval_classification(bucket.interval_measurements, bucket.link_interval_observations, bucket.dynamic_quality, config)
    stage1_adapter._validate_turn_movements(bucket.turn_movements, bucket.route_parts)
    stage1_adapter._validate_route_distance_conservation(bucket.route_parts, bucket.link_traversals, bucket.dynamic_quality, config)
    stage1_adapter._validate_dynamic_accounting(bucket.link_interval_observations, bucket.dynamic_quality, config)
    return bucket


def transform_stage1_final(*, repo_root: str | Path = ".", resume: bool = True) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    v5 = load_stage2_v5_config(root / "stage2/config/stage2_v5.json")
    final_dates = list(v5.section("split")["final_test_dates"])
    input_root = root / FINAL_ROOT / "stage1/input_v1"
    output_root = root / FINAL_ROOT / "stage1/output_v3"
    refs = _discover_stage0_final(input_root, final_dates)
    stage1_config = load_stage1_config(root / "stage1/config/stage1_label_schema_v3.json")
    models = load_model_bundle(root / "stage1/models/stage1_v3_final", stage1_config)
    counts = {name: 0 for name in OUTPUT_PRODUCTS}
    transformed = skipped = 0
    started = time.perf_counter()
    for ref in refs:
        target = output_root / f"split={ref.split}" / f"date={ref.date}" / f"bucket={ref.bucket:05d}"
        if resume and (target / "manifest.json").is_file():
            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("model_id") != models.model_id or manifest.get("input_bucket_sha") != bucket_input_identity(ref.path)["bucket_sha"]:
                raise ValueError(f"non-resumable final Stage 1 bucket: {target}")
            for name, count in manifest["product_row_counts"].items():
                counts[name] += int(count)
            skipped += 1
            continue
        bucket = _load_final_bucket(ref, stage1_config)
        products = _transform_bucket(ref, bucket, models, stage1_config)
        _validate_output_frames(products, ref)
        identity = bucket_input_identity(ref.path)
        with atomic_output_directory(target) as temporary:
            output_hashes: dict[str, str] = {}
            output_schemas: dict[str, str] = {}
            product_counts: dict[str, int] = {}
            for name in OUTPUT_PRODUCTS:
                path = temporary / f"{name}.parquet"
                atomic_write_parquet(products[name], path)
                output_hashes[name] = sha256_file(path)
                output_schemas[name] = parquet_schema_sha256(path)
                product_counts[name] = len(products[name])
                counts[name] += len(products[name])
            atomic_write_json(temporary / "manifest.json", {
                "schema_version": "stage1_v3_output_bucket.2",
                "engineering_status": "PASS",
                "scientific_status": "NOT_VALIDATED",
                "application_protocol": "stage2_v5_preregistered_final_frozen_models",
                "split": ref.split,
                "date": ref.date,
                "bucket": ref.bucket,
                "input_bucket_sha": identity["bucket_sha"],
                "input_file_hashes": identity["files"],
                "input_schema_hashes": identity["schemas"],
                "model_id": models.model_id,
                "product_row_counts": product_counts,
                "output_file_hashes": output_hashes,
                "output_schema_hashes": output_schemas,
            })
        transformed += 1
    summary = {
        "schema_version": "stage2_v5_final_stage1_transform.1",
        "engineering_status": "PASS",
        "model_id": models.model_id,
        "dates": final_dates,
        "bucket_count": len(refs),
        "transformed_bucket_count": transformed,
        "resumed_bucket_count": skipped,
        "product_row_counts": counts,
        "runtime_s": time.perf_counter() - started,
        "refit_performed": False,
    }
    atomic_write_json(output_root / "stage2_v5_final_stage1_summary.json", summary)
    return summary


def build_final_route_features(*, repo_root: str | Path = ".", resume: bool = True) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    input_root = root / FINAL_ROOT / "stage1/input_v1"
    stage1_output = root / FINAL_ROOT / "stage1/output_v3"
    output_root = root / FINAL_ROOT / "stage2/route_conditioned_dataset/revealed_route_proxy"
    v4_config = load_stage2_v4_config(root / "stage2/config/stage2_v4.json")
    history_root = root / "stage2/output_v4/causal_history_store"
    history_summary = validate_history_store(history_root, v4_config)
    history = TemporalHistoryIndex.from_store(history_root, v4_config)
    refs = discover_stage1_buckets(stage1_output, input_root)
    by_date: dict[str, list[Any]] = defaultdict(list)
    for ref in refs:
        by_date[ref.date].append(ref)
    reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    for date, day_refs in sorted(by_date.items()):
        path = output_root / f"day={date}.parquet"
        manifest_path = output_root.parent / "manifests" / f"day={date}.json"
        if resume and path.is_file() and manifest_path.is_file():
            reports.append(json.loads(manifest_path.read_text(encoding="utf-8")))
            continue
        writer = _AtomicDailyWriter(path)
        order_ids: set[str] = set()
        try:
            for ref in sorted(day_refs, key=lambda item: item.bucket):
                base = _base_route_tokens(ref, history, v4_config)
                frame = _finalize_track(base, history, v4_config, track="revealed_route_proxy")
                writer.write(frame)
                order_ids.update(frame["order_id"].astype(str).unique())
            writer.publish()
        except Exception:
            writer.abort()
            raise
        report = {
            "schema_version": "stage2_v5_final_route_features.1",
            "engineering_status": "PASS",
            "date": date,
            "row_count": writer.row_count,
            "order_count": len(order_ids),
            "source_bucket_count": len(day_refs),
            "history_event_count": history_summary["event_count"],
            "history_fit_scope": "frozen_stage2_v4_train_only",
            "file_sha256": sha256_file(path),
        }
        atomic_write_json(manifest_path, report)
        reports.append(report)
    summary = {
        "schema_version": "stage2_v5_final_route_feature_summary.1",
        "engineering_status": "PASS",
        "dates": sorted(by_date),
        "day_count": len(reports),
        "row_count": sum(int(item["row_count"]) for item in reports),
        "order_count": sum(int(item["order_count"]) for item in reports),
        "history_refit_performed": False,
        "runtime_s": time.perf_counter() - started,
    }
    atomic_write_json(output_root.parent / "dataset_manifest.json", summary)
    return summary


def build_final_tensor_shards(*, repo_root: str | Path = ".", resume: bool = True) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    v5 = load_stage2_v5_config(root / "stage2/config/stage2_v5.json")
    dates = list(v5.section("split")["final_test_dates"])
    input_root = root / FINAL_ROOT / "stage1/input_v1"
    stage1_output = root / FINAL_ROOT / "stage1/output_v3"
    feature_root = root / FINAL_ROOT / "stage2/route_conditioned_dataset/revealed_route_proxy"
    output = root / FINAL_ROOT / "stage2/tensor_shards"
    artifact_path = root / "stage2/output_v5/tensor_shards/feature_artifacts.json"
    artifacts = json.loads(artifact_path.read_text(encoding="utf-8"))
    _atomic_json(output / "feature_artifacts.json", artifacts)
    source_splits = {
        path.parents[1].name.split("=", 1)[1]: path.parents[2].name.split("=", 1)[1]
        for path in input_root.glob("split=*/date=*/bucket=*/manifest.json")
    }
    manifests: list[dict[str, Any]] = []
    shard_config = v5.section("shards")
    for date in dates:
        day_root = output / "split=final_test" / f"date={date}"
        manifest_path = day_root / "manifest.json"
        if resume and manifest_path.is_file():
            manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
            continue
        started = time.perf_counter()
        frame = load_v5_day(
            date,
            split=source_splits[date],
            repo_root=root,
            stage1_input_root=input_root,
            stage1_output_root=stage1_output,
            route_feature_root=feature_root,
        )
        payload = vectorized_chunk_payload(
            frame,
            artifacts,
            max_seq_len=int(shard_config["max_seq_len"]),
            overlap=int(shard_config["overlap"]),
        )
        files: list[dict[str, Any]] = []
        per_file = int(shard_config["chunks_per_file"])
        for index, start in enumerate(range(0, len(payload["order_id"]), per_file)):
            end = min(start + per_file, len(payload["order_id"]))
            path = day_root / f"shard-{index:05d}.npz"
            _atomic_npz(path, {name: values[start:end] for name, values in payload.items()})
            files.append({"path": path.relative_to(output).as_posix(), "chunk_count": end - start, "sha256": _sha256(path)})
        manifest = {
            "schema_version": "stage2_v5_tensor_day.1",
            "split": "final_test",
            "date": date,
            "source_row_count": len(frame),
            "chunk_count": len(payload["order_id"]),
            "supervision_weight_sum": float(payload["supervision_weight"].sum()),
            "runtime_s": time.perf_counter() - started,
            "files": files,
        }
        _atomic_json(manifest_path, manifest)
        manifests.append(manifest)
    summary = {
        "schema_version": "stage2_v5_final_tensor_shards.1",
        "dates": dates,
        "day_count": len(manifests),
        "source_row_count": sum(int(item["source_row_count"]) for item in manifests),
        "chunk_count": sum(int(item["chunk_count"]) for item in manifests),
        "feature_artifact_sha256": _sha256(artifact_path),
        "refit_performed": False,
        "days": manifests,
    }
    _atomic_json(output / "tensor_manifest.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit-config", "transform-stage1", "build-route-features", "build-shards"))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    if args.command == "audit-config":
        result = audit_stage0_config_delta(
            root / "stage0/config/stage0_v6_final.yaml",
            root / "stage2/config/stage2_v5_final_stage0.yaml",
            v5_config_path=root / "stage2/config/stage2_v5.json",
        )
        output = root / "stage2/docs/v5/stage2_v5_final_upstream_plan.json"
        atomic_write_json(output, result)
    elif args.command == "transform-stage1":
        result = transform_stage1_final(repo_root=root, resume=args.resume)
    elif args.command == "build-route-features":
        result = build_final_route_features(repo_root=root, resume=args.resume)
    else:
        result = build_final_tensor_shards(repo_root=root, resume=args.resume)
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status", result.get("engineering_status", "PASS")) == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
