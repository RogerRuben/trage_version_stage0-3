"""Streaming construction and lookup of strictly timestamped history events."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import Stage2V4Config
from .contracts import Stage2V4ContractError, require_columns
from .io import (
    atomic_write_json,
    atomic_write_parquet,
    sha256_file,
    stage2_v4_code_identity,
)


HISTORY_EVENT_SCHEMA_VERSION = "stage2_v4_history_events.1"
HISTORY_STORE_SCHEMA_VERSION = "stage2_v4_history_store.1"
HISTORY_VALUE_COLUMNS = (
    "crawl_time_share",
    "stop_time_share",
    "speed_cv_bounded",
    "acceleration_rms_bounded",
    "lcs_raw",
    "rts_raw",
    "lcs_tail_event",
    "rts_tail_event",
    "observed_sec_per_m",
)
HISTORY_REQUIRED_COLUMNS = frozenset(
    {
        "split",
        "date",
        "order_id",
        "traversal_id",
        "observed_directed_edge_uid",
        "canonical_highway",
        "observation_window_end_time",
        "time_bin_30m",
        "weekday_type",
        "lcs_available",
        "rts_available",
        *HISTORY_VALUE_COLUMNS,
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage2V4ContractError(f"cannot read history manifest: {path}") from exc
    if not isinstance(value, dict):
        raise Stage2V4ContractError(f"history manifest is not an object: {path}")
    return value


def _output_bucket_paths(stage1_output: Path) -> dict[str, list[Path]]:
    by_date: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(stage1_output.glob("split=*/date=*/bucket=*")):
        if not path.is_dir():
            continue
        date_part = next(
            (part for part in path.relative_to(stage1_output).parts if part.startswith("date=")),
            None,
        )
        if date_part is None:
            raise Stage2V4ContractError(f"invalid Stage 1 output partition: {path}")
        by_date[date_part.split("=", 1)[1]].append(path)
    return dict(by_date)


def _event_frame(path: Path, expected_model_id: str) -> pd.DataFrame:
    manifest = _read_json(path / "manifest.json")
    if (
        manifest.get("engineering_status") != "PASS"
        or manifest.get("model_id") != expected_model_id
    ):
        raise Stage2V4ContractError(f"unbound Stage 1 output bucket: {path}")
    frame = pd.read_parquet(
        path / "traversal_labels.parquet",
        columns=sorted(HISTORY_REQUIRED_COLUMNS),
    )
    require_columns(frame.columns, HISTORY_REQUIRED_COLUMNS, "traversal_labels history")
    frame = frame.rename(
        columns={
            "order_id": "history_order_id",
            "observation_window_end_time": "availability_timestamp",
        }
    )
    timestamp = pd.to_numeric(frame["availability_timestamp"], errors="coerce")
    if not np.isfinite(timestamp.to_numpy(dtype=float, na_value=np.nan)).all():
        raise Stage2V4ContractError(f"history event has missing timestamp: {path}")
    if frame["observed_directed_edge_uid"].isna().any():
        raise Stage2V4ContractError(f"history event has missing directed edge: {path}")
    frame["feature_timestamp"] = timestamp.astype(float)
    frame["availability_timestamp"] = timestamp.astype(float)
    frame["profile_time_bin"] = pd.to_numeric(
        frame["time_bin_30m"], errors="raise"
    ).astype("int16")
    frame["profile_weekday_type"] = frame["weekday_type"].astype(str)
    frame["self_order_excluded"] = True
    return frame


def _resume_valid(
    manifest_path: Path,
    data_path: Path,
    *,
    config_sha: str,
    code_sha: str,
    model_id: str,
) -> dict[str, Any] | None:
    if not manifest_path.is_file() or not data_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    expected = {
        "schema_version": HISTORY_EVENT_SCHEMA_VERSION,
        "engineering_status": "PASS",
        "stage2_config_sha256": config_sha,
        "stage2_code_sha": code_sha,
        "stage1_model_id": model_id,
        "file_sha256": sha256_file(data_path),
        "row_count": int(pq.ParquetFile(data_path).metadata.num_rows),
    }
    if all(manifest.get(key) == value for key, value in expected.items()):
        return manifest
    return None


def build_history_store(
    stage1_output: str | Path,
    output_root: str | Path,
    config: Stage2V4Config,
    *,
    resume: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Write one timestamp-sorted event table per day without full-data concat."""

    started = time.perf_counter()
    source = Path(stage1_output)
    output = Path(output_root)
    events_root = output / "events"
    manifests_root = output / "manifests"
    expected = config.section("stage1_release")
    code_sha = stage2_v4_code_identity(
        (
            "stage2/v4/history_store.py",
            "stage2/v4/contracts.py",
            "stage2/v4/io.py",
        )
    )
    buckets_by_date = _output_bucket_paths(source)
    expected_dates = {
        *config.section("split")["train_dates"],
        *config.section("split")["validation_model_dates"],
        *config.section("split")["calibration_dates"],
        *config.section("split")["test_dates"],
    }
    if set(buckets_by_date) != expected_dates:
        raise Stage2V4ContractError(
            "Stage 1 output dates do not match the frozen Stage 2 v4 split"
        )

    day_manifests: list[dict[str, Any]] = []
    transformed = 0
    resumed = 0
    for date in sorted(buckets_by_date):
        data_path = events_root / f"day={date}.parquet"
        manifest_path = manifests_root / f"day={date}.json"
        existing = (
            _resume_valid(
                manifest_path,
                data_path,
                config_sha=config.digest,
                code_sha=code_sha,
                model_id=expected["model_id"],
            )
            if resume
            else None
        )
        if existing is not None:
            day_manifests.append(existing)
            resumed += 1
            continue
        if (data_path.exists() or manifest_path.exists()) and not force:
            raise Stage2V4ContractError(
                f"history output already exists but is not resumable: {date}; "
                "use --force explicitly"
            )

        frames = [
            _event_frame(path, expected["model_id"])
            for path in buckets_by_date[date]
        ]
        events = pd.concat(frames, ignore_index=True)
        del frames
        events.sort_values(
            ["availability_timestamp", "observed_directed_edge_uid", "history_order_id"],
            kind="stable",
            inplace=True,
            ignore_index=True,
        )
        for column in HISTORY_VALUE_COLUMNS:
            if column.endswith("_event"):
                events[column] = events[column].astype("boolean")
            else:
                events[column] = pd.to_numeric(events[column], errors="coerce")
        atomic_write_parquet(events, data_path)
        manifest = {
            "schema_version": HISTORY_EVENT_SCHEMA_VERSION,
            "engineering_status": "PASS",
            "date": date,
            "stage2_config_sha256": config.digest,
            "stage2_code_sha": code_sha,
            "stage1_model_id": expected["model_id"],
            "source_bucket_count": len(buckets_by_date[date]),
            "row_count": int(len(events)),
            "minimum_availability_timestamp": float(events["availability_timestamp"].min()),
            "maximum_availability_timestamp": float(events["availability_timestamp"].max()),
            "missing_availability_timestamp_count": 0,
            "file_sha256": sha256_file(data_path),
        }
        atomic_write_json(manifest_path, manifest)
        day_manifests.append(manifest)
        transformed += 1

    total_rows = sum(int(item["row_count"]) for item in day_manifests)
    failures: list[str] = []
    if total_rows != expected["traversal_label_count"]:
        failures.append(
            "history event reconciliation: "
            f"expected {expected['traversal_label_count']}, got {total_rows}"
        )
    summary = {
        "schema_version": HISTORY_STORE_SCHEMA_VERSION,
        "engineering_status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "stage2_config_sha256": config.digest,
        "stage2_code_sha": code_sha,
        "stage1_model_id": expected["model_id"],
        "day_count": len(day_manifests),
        "transformed_day_count": transformed,
        "resumed_day_count": resumed,
        "event_count": total_rows,
        "event_files": {
            item["date"]: {
                "row_count": item["row_count"],
                "file_sha256": item["file_sha256"],
            }
            for item in day_manifests
        },
        "runtime_s": time.perf_counter() - started,
    }
    atomic_write_json(output / "history_store_manifest.json", summary)
    if failures:
        raise Stage2V4ContractError("; ".join(failures))
    return summary


def validate_history_store(
    root: str | Path,
    config: Stage2V4Config,
) -> dict[str, Any]:
    source = Path(root)
    summary = _read_json(source / "history_store_manifest.json")
    expected = config.section("stage1_release")
    checks = {
        "schema_version": HISTORY_STORE_SCHEMA_VERSION,
        "engineering_status": "PASS",
        "stage2_config_sha256": config.digest,
        "stage1_model_id": expected["model_id"],
        "event_count": expected["traversal_label_count"],
    }
    for key, value in checks.items():
        if summary.get(key) != value:
            raise Stage2V4ContractError(
                f"history store mismatch for {key}: "
                f"expected {value!r}, got {summary.get(key)!r}"
            )
    files = summary.get("event_files")
    if not isinstance(files, dict):
        raise Stage2V4ContractError("history store has no event file identities")
    for date, identity in files.items():
        path = source / "events" / f"day={date}.parquet"
        if not path.is_file() or sha256_file(path) != identity.get("file_sha256"):
            raise Stage2V4ContractError(f"history event file identity mismatch: {date}")
    return summary
