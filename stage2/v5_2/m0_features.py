"""Canonical, protocol-bound feature matrix builder for the M0 tree baseline."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage2.v4.models.baselines import _feature_candidates
from stage2.v5.data import load_v5_day

from .contracts import FORBIDDEN_MODEL_INPUTS, Stage2V52ContractError, validate_model_inputs
from .feature_binding import sha256_path
from .micro_metrics import stop_two_part_metrics
from .protocols import get_protocol
from .support_transfer import lookup_train_support
from .training import M0_MATRIX_SCHEMA_VERSION, M0_TRAINING_SCHEMA_VERSION


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


def _stage1_split(date: str) -> str:
    day = int(str(date)[-2:])
    if 9 <= day <= 24:
        return "train"
    if 25 <= day <= 27:
        return "validation"
    if day == 31:
        return "test"
    raise Stage2V52ContractError(f"date has no frozen Stage 1 split: {date}")


def _source_hashes(root: Path, feature_root: Path, date: str) -> dict[str, Any]:
    route = feature_root / f"day={date}.parquet"
    split = _stage1_split(date)
    traversal_root = root / "stage1/input_v1" / f"split={split}" / f"date={date}"
    label_root = root / "stage1/output_v3" / f"split={split}" / f"date={date}"
    traversal_paths = sorted(traversal_root.glob("bucket=*/link_traversals.parquet"))
    label_paths = sorted(label_root.glob("bucket=*/traversal_labels.parquet"))
    if not route.is_file() or not traversal_paths or not label_paths:
        raise Stage2V52ContractError(f"M0 source products are incomplete for {date}")
    return {
        "route_feature": {"path": route.as_posix(), "sha256": sha256_path(route)},
        "stage1_link_traversals": [
            {"path": path.as_posix(), "sha256": sha256_path(path)} for path in traversal_paths
        ],
        "stage1_traversal_labels": [
            {"path": path.as_posix(), "sha256": sha256_path(path)} for path in label_paths
        ],
    }


def _matrix_arrays(
    frames: Iterable[pd.DataFrame], *, feature_names: tuple[str, ...], median: Mapping[str, float],
    split: str, support_artifact: Mapping[str, Any] | None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    pieces: dict[str, list[np.ndarray]] = {
        "features": [], "split": [], "date": [], "order_id": [], "traversal_id": [],
        "support_group_code": [],
    }
    for target in TARGET_COLUMNS:
        pieces[target], pieces[f"{target}_valid"] = [], []
    valid_counts = {target: 0 for target in TARGET_COLUMNS}
    group_codes = {"unseen": 0, "low": 1, "medium": 2, "high": 3}
    for frame in frames:
        numeric = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
        pieces["features"].append(
            numeric.fillna(pd.Series(median)).to_numpy(np.float32)
        )
        pieces["split"].append(np.full(len(frame), split, dtype="U16"))
        pieces["date"].append(np.asarray(frame["date"].astype(str).tolist(), dtype="U8"))
        pieces["order_id"].append(np.asarray(frame["order_id"].astype(str).tolist(), dtype=str))
        pieces["traversal_id"].append(pd.to_numeric(frame["traversal_id"], errors="raise").to_numpy(np.int64))
        if support_artifact is None:
            group = np.full(len(frame), "medium", dtype="U8")
        else:
            _, group = lookup_train_support(frame["observed_directed_edge_uid"], support_artifact)
        pieces["support_group_code"].append(
            np.asarray([group_codes[str(value)] for value in group], dtype=np.int8)
        )
        for target, (column, mask_column) in TARGET_COLUMNS.items():
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float32)
            mask = frame[mask_column].fillna(False).to_numpy(bool) & np.isfinite(values)
            pieces[target].append(np.where(mask, values, 0.0).astype(np.float32))
            pieces[f"{target}_valid"].append(mask)
            valid_counts[target] += int(mask.sum())
    payload = {name: np.concatenate(values, axis=0) for name, values in pieces.items()}
    return payload, valid_counts


def build_m0_feature_matrix(
    *, protocol_id: str, repo_root: str | Path, route_feature_root: str | Path,
    output_matrix_path: str | Path, output_manifest_path: str | Path,
) -> dict[str, Any]:
    """Materialize the exact protocol Train partition with an immutable feature schema."""
    protocol = get_protocol(protocol_id)
    root = Path(repo_root).resolve()
    feature_root = Path(route_feature_root).resolve()
    feature_names: tuple[str, ...] = ()
    median: dict[str, float] = {}
    # Exact Train medians are fitted one feature at a time from disk-backed
    # float64 streams.  This preserves the previous algorithm while avoiding an
    # all-dates x all-features in-memory copy.
    with tempfile.TemporaryDirectory(prefix="stage2-v5-2-m0-median-") as temporary:
        temp_root = Path(temporary)
        feature_paths: dict[str, Path] = {}
        feature_counts: dict[str, int] = {}
        for date in protocol.train_dates:
            frame = load_v5_day(
                date, split="train", repo_root=root, route_feature_root=feature_root,
                extra_columns=("observed_directed_edge_uid",),
            ).assign(split="train", date=date)
            if frame.empty:
                raise Stage2V52ContractError("M0 matrix requires non-empty data for every protocol Train date")
            if not feature_names:
                feature_names = tuple(column for column in _feature_candidates() if column in frame.columns)
                feature_paths = {name: temp_root / f"feature-{index:04d}.f64" for index, name in enumerate(feature_names)}
                feature_counts = {name: 0 for name in feature_names}
            if not set(feature_names) <= set(frame.columns):
                raise Stage2V52ContractError("M0 Train feature columns drift across dates")
            for name in feature_names:
                values = pd.to_numeric(frame[name], errors="coerce").to_numpy(np.float64)
                with feature_paths[name].open("ab") as handle:
                    values.tofile(handle)
                feature_counts[name] += int(len(values))
        if not feature_names:
            raise Stage2V52ContractError("M0 matrix has no canonical decision-time features")
        for name in feature_names:
            values = np.memmap(feature_paths[name], mode="r", dtype=np.float64, shape=(feature_counts[name],))
            median[name] = float(np.nanmedian(values))
            del values
    validate_model_inputs(feature_names)
    forbidden = sorted(set(feature_names) & FORBIDDEN_MODEL_INPUTS)
    median = {name: value if np.isfinite(value) else 0.0 for name, value in median.items()}

    def train_frames() -> Iterable[pd.DataFrame]:
        for date in protocol.train_dates:
            yield load_v5_day(
                date, split="train", repo_root=root, route_feature_root=feature_root,
                extra_columns=("observed_directed_edge_uid",),
            ).assign(split="train", date=date)

    payload, valid_counts = _matrix_arrays(
        train_frames(), feature_names=feature_names, median=median, split="train", support_artifact=None,
    )
    matrix_path = Path(output_matrix_path)
    _atomic_npz(matrix_path, payload)
    source_hashes = {date: _source_hashes(root, feature_root, date) for date in protocol.train_dates}
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
        "row_count": int(len(payload["features"])),
        "feature_count": len(feature_names),
        "feature_schema": feature_schema,
        "feature_schema_hash": _canonical_hash(feature_schema),
        "source_product_hashes": source_hashes,
        "construction_policy": "daily_dataframe_scan_disk_backed_per_feature_exact_median_then_numpy_materialization",
        "valid_target_counts": valid_counts,
        "forbidden_input_audit": {"status": "PASS" if not forbidden else "FAIL", "fields": forbidden},
        "matrix_path": matrix_path.as_posix(),
        "matrix_sha256": sha256_path(matrix_path),
    }
    if forbidden:
        raise Stage2V52ContractError(f"M0 matrix contains forbidden inputs: {forbidden}")
    _atomic_json(Path(output_manifest_path), manifest)
    return manifest


def transform_m0_feature_matrix(
    *, protocol_id: str, role: str, repo_root: str | Path, route_feature_root: str | Path,
    train_matrix_manifest_path: str | Path, support_artifact_path: str | Path,
    output_matrix_path: str | Path, output_manifest_path: str | Path,
) -> dict[str, Any]:
    """Apply the immutable Train schema and medians to Validation/Evaluation partitions."""
    protocol = get_protocol(protocol_id)
    role_dates = {
        "validation": protocol.validation_dates,
        "evaluation": protocol.evaluation_dates,
        "calibration": protocol.calibration_dates,
        "legacy": protocol.legacy_benchmark_dates,
    }
    if role not in role_dates or not role_dates[role]:
        raise Stage2V52ContractError(f"M0 protocol has no transformable role: {role}")
    train_manifest = json.loads(Path(train_matrix_manifest_path).read_text(encoding="utf-8"))
    if (
        train_manifest.get("schema_version") != M0_MATRIX_SCHEMA_VERSION
        or train_manifest.get("protocol_id") != protocol_id
        or train_manifest.get("fit_scope") != "train_only"
    ):
        raise Stage2V52ContractError("M0 transform requires the exact protocol Train schema")
    support = json.loads(Path(support_artifact_path).read_text(encoding="utf-8"))
    if support.get("fit_scope") != "train_only" or support.get("protocol_id") != protocol_id:
        raise Stage2V52ContractError("M0 transform support is not protocol Train-only")
    root, feature_root = Path(repo_root).resolve(), Path(route_feature_root).resolve()
    schema = train_manifest["feature_schema"]
    feature_names = tuple(schema["feature_names"])

    def role_frames() -> Iterable[pd.DataFrame]:
        for date in role_dates[role]:
            frame = load_v5_day(
                date, split=role, repo_root=root, route_feature_root=feature_root,
                extra_columns=("observed_directed_edge_uid",),
            ).assign(split=role, date=date)
            if not set(feature_names) <= set(frame.columns):
                raise Stage2V52ContractError("M0 transform columns differ from Train schema")
            yield frame

    payload, valid_counts = _matrix_arrays(
        role_frames(), feature_names=feature_names, median=schema["median"], split=role,
        support_artifact=support,
    )
    identity = pd.DataFrame({
        "date": payload["date"].astype(str), "order_id": payload["order_id"].astype(str),
        "traversal_id": payload["traversal_id"],
    })
    if identity.duplicated(["date", "order_id", "traversal_id"]).any():
        raise Stage2V52ContractError("M0 transformed role is not unique by physical traversal")
    matrix_path = Path(output_matrix_path)
    _atomic_npz(matrix_path, payload)
    manifest = {
        "schema_version": M0_MATRIX_SCHEMA_VERSION, "status": "PASS",
        "protocol_id": protocol_id, "protocol_hash": protocol.digest, "role": role,
        "evaluation_dates": list(role_dates[role]), "fit_scope": "transform_only",
        "train_feature_schema_hash": train_manifest["feature_schema_hash"],
        "train_matrix_manifest_sha256": sha256_path(train_matrix_manifest_path),
        "support_artifact_sha256": sha256_path(support_artifact_path),
        "source_product_hashes": {
            date: _source_hashes(root, feature_root, date) for date in role_dates[role]
        },
        "row_count": int(len(payload["features"])), "unique_traversal_count": int(len(identity)),
        "valid_target_counts": valid_counts, "matrix_path": matrix_path.as_posix(),
        "matrix_sha256": sha256_path(matrix_path),
    }
    _atomic_json(Path(output_manifest_path), manifest)
    return manifest


def evaluate_m0_baseline(
    *, protocol_id: str, role: str, matrix_path: str | Path, matrix_manifest_path: str | Path,
    model_path: str | Path, training_manifest_path: str | Path, output_path: str | Path,
) -> dict[str, Any]:
    """Evaluate M0 once per unique traversal with overall/low/unseen four-core MAE."""
    import joblib

    manifest = json.loads(Path(matrix_manifest_path).read_text(encoding="utf-8"))
    training = json.loads(Path(training_manifest_path).read_text(encoding="utf-8"))
    if (
        manifest.get("protocol_id") != protocol_id or manifest.get("role") != role
        or manifest.get("matrix_sha256") != sha256_path(matrix_path)
        or training.get("schema_version") != M0_TRAINING_SCHEMA_VERSION
        or training.get("protocol_id") != protocol_id
        or training.get("model_sha256") != sha256_path(model_path)
        or training.get("feature_schema_hash") != manifest.get("train_feature_schema_hash")
    ):
        raise Stage2V52ContractError("M0 evaluator provenance is inconsistent")
    with np.load(matrix_path, allow_pickle=False) as archive:
        model = joblib.load(model_path)
        predictions = model.predict(archive["features"])
        groups = archive["support_group_code"].astype(np.int8)
        metrics: dict[str, Any] = {}
        for group, selector in {
            "overall": np.ones(len(groups), dtype=bool), "low": groups == 1, "unseen": groups == 0,
        }.items():
            metrics[group] = {}
            for target in ("crawl", "stop", "speed_cv", "acceleration_rms"):
                valid = selector & archive[f"{target}_valid"].astype(bool)
                truth = archive[target].astype(float)
                candidate = predictions[f"pred_{target}"].astype(float)
                valid &= np.isfinite(truth) & np.isfinite(candidate)
                count = int(valid.sum())
                metrics[group][target] = {
                    "count": count,
                    "mae": float(np.mean(np.abs(candidate[valid] - truth[valid]))) if count else None,
                }
            stop_valid = selector & archive["stop_valid"].astype(bool)
            stop_truth = archive["stop"].astype(float)
            stop_valid &= np.isfinite(stop_truth)
            metrics[group]["stop_two_part"] = stop_two_part_metrics(
                stop_truth[stop_valid],
                predictions["stop_occurrence_probability"].astype(float)[stop_valid],
                predictions["stop_positive_share"].astype(float)[stop_valid],
            )
        unique_count = int(len(archive["features"]))
    report = {
        "schema_version": "stage2_v5_2_m0_evaluation.1", "status": "PASS",
        "protocol_id": protocol_id, "protocol_hash": get_protocol(protocol_id).digest,
        "model_id": "M0", "role": role, "evaluation_dates": manifest["evaluation_dates"],
        "metrics_by_support": metrics,
        "core_mae": {target: metrics["overall"][target]["mae"] for target in ("crawl", "stop", "speed_cv", "acceleration_rms")},
        "low_support_core_mae": {target: metrics["low"][target]["mae"] for target in ("crawl", "stop", "speed_cv", "acceleration_rms")},
        "unseen_core_mae": {target: metrics["unseen"][target]["mae"] for target in ("crawl", "stop", "speed_cv", "acceleration_rms")},
        "stop_two_part_metrics_by_support": {
            group: values["stop_two_part"] for group, values in metrics.items()
        },
        "unique_traversal_count": unique_count, "duplicate_prediction_count": 0,
        "matrix_manifest_sha256": sha256_path(matrix_manifest_path),
        "training_manifest_sha256": sha256_path(training_manifest_path),
        "model_sha256": sha256_path(model_path),
        "source_product_hashes": manifest["source_product_hashes"],
    }
    if any(
        metrics[group][target]["mae"] is None
        for group in metrics for target in ("crawl", "stop", "speed_cv", "acceleration_rms")
    ):
        report["status"] = "FAIL_INSUFFICIENT_SUPPORT"
    _atomic_json(Path(output_path), report)
    return report
