"""Layered read-only dry runs and the computed v5 performance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import joblib

from .availability import service_time_target_arrays


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _measure(name: str, function) -> tuple[dict[str, Any], Any]:
    process = psutil.Process(os.getpid())
    before = process.memory_info().rss
    started = time.perf_counter()
    result = function()
    runtime = time.perf_counter() - started
    after = process.memory_info().rss
    rows = int(result.get("row_count", 0))
    record = {
        "stage": name,
        "row_count": rows,
        "runtime_s": runtime,
        "peak_rss_mb": max(before, after) / (1024 * 1024),
        "read_bytes": int(result.get("read_bytes", 0)),
        "write_bytes": 0,
        "rows_per_s": rows / max(runtime, 1e-12),
    }
    return record, result


def _bucket(root: Path, split: str, date: str, bucket: str) -> dict[str, Any]:
    traversal = root / "stage1" / "input_v1" / f"split={split}" / f"date={date}" / f"bucket={bucket}" / "link_traversals.parquet"
    labels = root / "stage1" / "output_v3" / f"split={split}" / f"date={date}" / f"bucket={bucket}" / "traversal_labels.parquet"
    frame = pd.read_parquet(traversal, columns=["measurement_source", "observed_travel_time_s", "observed_distance_m"])
    targets = service_time_target_arrays(frame["measurement_source"].to_numpy(), frame["observed_travel_time_s"].to_numpy(), frame["observed_distance_m"].to_numpy())
    label_rows = len(pd.read_parquet(labels, columns=["order_id"]))
    return {
        "row_count": len(frame),
        "label_row_count": label_rows,
        "pace_target_count": int(targets["pace_target_valid"].sum()),
        "read_bytes": traversal.stat().st_size + labels.stat().st_size,
    }


def _day(root: Path, split: str, date: str) -> dict[str, Any]:
    day = root / "stage1" / "input_v1" / f"split={split}" / f"date={date}"
    paths = sorted(day.glob("bucket=*/link_traversals.parquet"))
    rows = pace = read_bytes = 0
    for path in paths:  # Required bounded partition streaming.
        frame = pd.read_parquet(path, columns=["measurement_source", "observed_travel_time_s", "observed_distance_m"])
        target = service_time_target_arrays(frame["measurement_source"].to_numpy(), frame["observed_travel_time_s"].to_numpy(), frame["observed_distance_m"].to_numpy())
        rows += len(frame)
        pace += int(target["pace_target_valid"].sum())
        read_bytes += path.stat().st_size
    return {"row_count": rows, "pace_target_count": pace, "partition_count": len(paths), "read_bytes": read_bytes}


def _legacy_inference(root: Path) -> dict[str, Any]:
    path = root / "stage2" / "output_v4" / "route_conditioned_dataset" / "revealed_route_proxy" / "day=20161031.parquet"
    columns = ["order_id", "route_sequence", "forecast_horizon_s", "estimated_travel_time_s", "route_part_length_m", "observed_sec_per_m_profile_mean"]
    frame = pd.read_parquet(path, columns=columns)
    horizon = pd.to_numeric(frame["forecast_horizon_s"], errors="coerce").to_numpy(float)
    estimate = pd.to_numeric(frame["estimated_travel_time_s"], errors="coerce").to_numpy(float)
    if np.any(np.isfinite(horizon) & (horizon < 0)) or np.any(np.isfinite(estimate) & (estimate < 0)):
        raise ValueError("legacy inference rehearsal encountered invalid causal features")
    return {
        "row_count": len(frame),
        "order_count": int(frame["order_id"].nunique()),
        "zero_distance_token_count": int(
            pd.to_numeric(frame["route_part_length_m"], errors="coerce").eq(0).sum()
        ),
        "read_bytes": path.stat().st_size,
    }


def _completed_runtime_records(root: Path) -> list[dict[str, Any]]:
    """Append already-completed full development stages when available."""

    records: list[dict[str, Any]] = []
    audit_path = root / "stage2/docs/v5/stage2_v5_service_time_target_audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        records.append({
            "stage": "service_time_target_audit",
            "row_count": int(audit["counters"]["traversal_row_count"]),
            "runtime_s": float(audit["runtime_s"]),
            "peak_rss_mb": np.nan,
            "read_bytes": np.nan,
            "write_bytes": np.nan,
            "rows_per_s": int(audit["counters"]["traversal_row_count"]) / float(audit["runtime_s"]),
            "scope": "full_frozen_stage1_audit",
        })
    shard_path = root / "stage2/output_v5/tensor_shards/tensor_manifest.json"
    if shard_path.is_file():
        shards = json.loads(shard_path.read_text(encoding="utf-8"))
        runtime_s = float(sum(float(day["runtime_s"]) for day in shards["days"]))
        records.append({
            "stage": "tensor_shard_build",
            "row_count": int(shards["source_row_count"]),
            "runtime_s": runtime_s,
            "peak_rss_mb": np.nan,
            "read_bytes": np.nan,
            "write_bytes": np.nan,
            "rows_per_s": int(shards["source_row_count"]) / runtime_s,
            "scope": "full_20_day_vectorized_transform",
        })
    baseline_path = root / "stage2/output_v5/baselines/service_time_baselines.joblib"
    if baseline_path.is_file():
        baseline = joblib.load(baseline_path)
        records.append({
            "stage": "strong_baseline_fit",
            "row_count": int(baseline["tree_fit_row_count"]),
            "runtime_s": float(baseline["runtime_s"]),
            "peak_rss_mb": np.nan,
            "read_bytes": np.nan,
            "write_bytes": np.nan,
            "rows_per_s": int(baseline["tree_fit_row_count"]) / float(baseline["runtime_s"]),
            "scope": "train_only_tree_fit_and_evaluation",
        })
    manifests = [("rc_mstnet_v5_horizon_gate_fit", root / "stage2/output_v5/deep_model/model_manifest.json")]
    manifests += [
        (f"ablation_{mode}_fit", root / f"stage2/output_v5/ablations/{mode}/model/model_manifest.json")
        for mode in ("ordinary_concatenation", "without_recent", "without_profile")
    ]
    for stage, path in manifests:
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        records.append({
            "stage": stage,
            "row_count": 15_649_455,
            "runtime_s": float(manifest["runtime_s"]),
            "peak_rss_mb": np.nan,
            "read_bytes": np.nan,
            "write_bytes": np.nan,
            "rows_per_s": np.nan,
            "scope": f"{len(manifest['training_history'])}_epochs_gpu_amp",
        })
    return records


def run_preflight(repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    static_path = root / "stage2/docs/v5/stage2_v5_static_complexity_audit.json"
    benchmark_path = root / "stage2/docs/v5/stage2_v5_performance_benchmarks.json"
    static = json.loads(static_path.read_text(encoding="utf-8"))
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    jobs = (
        ("one_bucket_train", lambda: _bucket(root, "train", "20161009", "00000")),
        ("one_train_day", lambda: _day(root, "train", "20161009")),
        ("one_validation_day", lambda: _day(root, "validation", "20161025")),
        ("legacy_full_day_read_only_inference", lambda: _legacy_inference(root)),
    )
    records: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for name, function in jobs:
        record, detail = _measure(name, function)
        records.append(record)
        details[name] = detail
    runtime = pd.DataFrame([{**record, "scope": "preflight_read_transform"} for record in records])
    measured_rows = runtime["row_count"].sum()
    measured_runtime = runtime["runtime_s"].sum()
    projected_rows = 15_649_455
    expected_full_runtime_s = measured_runtime * projected_rows / max(measured_rows, 1) * 1.5
    maximum_rss = float(runtime["peak_rss_mb"].max())
    conditions = {
        "static_complexity_audit_pass": static.get("status") == "PASS",
        "micro_benchmark_pass": benchmark.get("status") == "PASS",
        "layered_dry_run_pass": all(record["row_count"] > 0 for record in records),
        "peak_rss_acceptable": maximum_rss <= 8192,
        "reference_equivalence_tests_required": True,
        "runtime_projection_finite": bool(np.isfinite(expected_full_runtime_s)),
    }
    report = {
        "schema_version": "stage2_v5_preflight.1",
        "performance_gate_status": "PASS" if all(conditions.values()) else "FAIL",
        "conditions": conditions,
        "runs": records,
        "details": details,
        "expected_full_runtime_s_linear_times_1_5": expected_full_runtime_s,
        "maximum_peak_rss_mb": maximum_rss,
        "static_audit_sha256": _sha256(static_path),
        "benchmark_report_sha256": _sha256(benchmark_path),
        "profile_report_sha256": _sha256(root / "stage2/docs/v5/performance_profile_hotspots.txt"),
        "upstream_rebuild_required": False,
        "upstream_reuse_note": "Rolling-origin evaluation reuses frozen Stage 0/1 and v4 causal route products for 20161009-27 and the 20161031 legacy benchmark.",
    }
    output = root / "stage2/docs/v5"
    completed = _completed_runtime_records(root)
    if completed:
        runtime = pd.concat([runtime, pd.DataFrame(completed)], ignore_index=True)
    runtime.to_csv(output / "runtime_by_stage.csv", index=False)
    runtime.loc[runtime["peak_rss_mb"].notna(), ["stage", "peak_rss_mb"]].to_csv(output / "memory_by_stage.csv", index=False)
    (output / "stage2_v5_preflight.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    report = run_preflight(args.repo_root)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["performance_gate_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
