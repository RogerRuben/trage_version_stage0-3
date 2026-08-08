"""Canonical, protocol-bound feature matrix builder for the M0 tree baseline."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage2.v4.models.baselines import _feature_candidates
from stage2.v5.data import load_v5_day

from .contracts import FORBIDDEN_MODEL_INPUTS, Stage2V52ContractError, validate_model_inputs
from .feature_binding import sha256_path
from .protocols import get_protocol
from .training import M0_MATRIX_SCHEMA_VERSION


TARGET_COLUMNS = {
    "crawl": ("crawl_time_share", "crawl_target_valid"),
    "stop": ("stop_time_share", "stop_target_valid"),
    "speed_cv": ("speed_cv_bounded", "speed_cv_target_valid"),
    "acceleration_rms": ("acceleration_rms_bounded", "acceleration_rms_target_valid"),
    "rts": ("rts_raw", "rts_target_valid"),
}


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_m0_feature_matrix(
    *, protocol_id: str, repo_root: str | Path, route_feature_root: str | Path,
    output_matrix_path: str | Path, output_manifest_path: str | Path,
) -> dict[str, Any]:
    """Materialize the exact protocol Train partition with an immutable feature schema."""
    protocol = get_protocol(protocol_id)
    root = Path(repo_root).resolve()
    feature_root = Path(route_feature_root).resolve()
    frames = [
        load_v5_day(date, split="train", repo_root=root, route_feature_root=feature_root).assign(
            split="train", date=date
        )
        for date in protocol.train_dates
    ]
    if not frames or any(frame.empty for frame in frames):
        raise Stage2V52ContractError("M0 matrix requires non-empty data for every protocol Train date")
    frame = pd.concat(frames, ignore_index=True)
    feature_names = tuple(column for column in _feature_candidates() if column in frame.columns)
    if not feature_names:
        raise Stage2V52ContractError("M0 matrix has no canonical decision-time features")
    validate_model_inputs(feature_names)
    forbidden = sorted(set(feature_names) & FORBIDDEN_MODEL_INPUTS)
    numeric = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    median = numeric.median(axis=0, skipna=True).fillna(0.0)
    features = numeric.fillna(median).to_numpy(np.float32)
    payload: dict[str, np.ndarray] = {
        "features": features,
        "split": frame["split"].astype(str).to_numpy(),
        "date": frame["date"].astype(str).to_numpy(),
        "order_id": frame["order_id"].astype(str).to_numpy(),
        "traversal_id": pd.to_numeric(frame["traversal_id"], errors="raise").to_numpy(np.int64),
    }
    valid_counts: dict[str, int] = {}
    for target, (column, mask_column) in TARGET_COLUMNS.items():
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float32)
        mask = frame[mask_column].fillna(False).to_numpy(bool) & np.isfinite(values)
        payload[target] = np.where(mask, values, 0.0).astype(np.float32)
        payload[f"{target}_valid"] = mask
        valid_counts[target] = int(mask.sum())
    matrix_path = Path(output_matrix_path)
    _atomic_npz(matrix_path, payload)
    source_hashes = {
        date: sha256_path(feature_root / f"day={date}.parquet") for date in protocol.train_dates
    }
    feature_schema = {
        "feature_names": list(feature_names),
        "missing_policy": "Train_partition_median_fitted_on_exact_protocol_dates",
        "median": {name: float(median[name]) for name in feature_names},
        "dtype": "float32",
    }
    manifest = {
        "schema_version": M0_MATRIX_SCHEMA_VERSION,
        "status": "PASS",
        "protocol_id": protocol_id,
        "protocol_hash": protocol.digest,
        "fit_scope": "train_only",
        "fit_dates_observed": list(protocol.train_dates),
        "evaluation_rows_used": 0,
        "row_count": int(len(frame)),
        "feature_count": len(feature_names),
        "feature_schema": feature_schema,
        "feature_schema_hash": _canonical_hash(feature_schema),
        "source_route_hashes": source_hashes,
        "valid_target_counts": valid_counts,
        "forbidden_input_audit": {"status": "PASS" if not forbidden else "FAIL", "fields": forbidden},
        "matrix_path": matrix_path.as_posix(),
        "matrix_sha256": sha256_path(matrix_path),
    }
    if forbidden:
        raise Stage2V52ContractError(f"M0 matrix contains forbidden inputs: {forbidden}")
    _atomic_json(Path(output_manifest_path), manifest)
    return manifest
