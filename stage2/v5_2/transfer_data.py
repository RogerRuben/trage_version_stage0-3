"""Protocol-bound builder for aligned v5.2 transfer tensor shards."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage2.v5.data import load_v5_day
from stage2.v5.shards import vectorized_chunk_payload

from .contracts import Stage2V52ContractError, require_columns
from .protocols import get_protocol, protocol_role_dates
from .structure_features import build_static_structure_features, validate_feature_alignment
from .support_transfer import lookup_train_support
from .temporal_adapter import TEMPORAL_FEATURE_NAMES


TRANSFER_TENSOR_SCHEMA_VERSION = "stage2_v5_2_transfer_tensor.2"
TRANSFER_DAY_SCHEMA_VERSION = "stage2_v5_2_transfer_day.2"
TRANSFER_MANIFEST_SCHEMA_VERSION = "stage2_v5_2_transfer_manifest.2"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Stage2V52ContractError(f"expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def build_temporal_features(frame: pd.DataFrame, *, timezone: str = "Asia/Shanghai") -> np.ndarray:
    require_columns(frame.columns, ("decision_time", "forecast_horizon_s"), product="temporal features")
    decision = pd.to_numeric(frame["decision_time"], errors="coerce")
    horizon = pd.to_numeric(frame["forecast_horizon_s"], errors="coerce")
    if decision.isna().any() or horizon.isna().any() or (horizon < 0).any():
        raise Stage2V52ContractError("temporal feature inputs must be finite and horizon non-negative")
    local = pd.to_datetime(decision, unit="s", utc=True).dt.tz_convert(timezone)
    hour = local.dt.hour.to_numpy(float) + local.dt.minute.to_numpy(float) / 60.0
    angle = hour * (2.0 * np.pi / 24.0)
    return np.column_stack((
        np.sin(angle), np.cos(angle), local.dt.dayofweek.to_numpy(float) / 6.0,
        np.log1p(horizon.to_numpy(float)) / 12.0,
    )).astype(np.float32)


def _partition_identity(frame: pd.DataFrame) -> tuple[str, str]:
    partitions = frame[["split", "date"]].astype(str).drop_duplicates()
    if len(partitions) != 1:
        raise Stage2V52ContractError("transfer shard builder accepts exactly one split/date partition")
    return tuple(partitions.iloc[0].tolist())  # type: ignore[return-value]


def _payload_source_positions(payload: Mapping[str, np.ndarray], frame: pd.DataFrame) -> np.ndarray:
    split, date = _partition_identity(frame)
    identity = ["split", "date", "order_id", "traversal_id"]
    source_frame = frame.loc[:, identity].copy()
    source_frame["split"] = source_frame["split"].astype(str)
    source_frame["date"] = source_frame["date"].astype(str)
    source_frame["order_id"] = source_frame["order_id"].astype(str)
    source = pd.MultiIndex.from_frame(source_frame)
    if source.has_duplicates:
        raise Stage2V52ContractError("transfer source traversal identity is duplicated")
    valid = ~payload["pad_mask"]
    order = np.broadcast_to(payload["order_id"][:, None], valid.shape)[valid].astype(str)
    traversal = payload["traversal_id"][valid].astype(np.int64)
    query = pd.MultiIndex.from_arrays(
        (
            np.full(len(order), split, dtype=str), np.full(len(order), date, dtype=str),
            order, traversal,
        ),
        names=identity,
    )
    positions = source.get_indexer(query)
    if np.any(positions < 0):
        raise Stage2V52ContractError("chunk token cannot be aligned to full source identity")
    aligned = np.full(valid.shape, -1, dtype=np.int64)
    aligned[valid] = positions
    return aligned


def _gather_aligned(values: np.ndarray, positions: np.ndarray, fill: float | int) -> np.ndarray:
    valid = positions >= 0
    safe = np.where(valid, positions, 0)
    gathered = values[safe]
    if gathered.ndim == valid.ndim:
        return np.where(valid, gathered, fill)
    return np.where(valid[..., None], gathered, fill)


def _temporal_source(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    decision = pd.to_numeric(frame["decision_time"], errors="coerce").to_numpy(float)
    if "feature_cutoff_time" in frame:
        cutoff = pd.to_numeric(frame["feature_cutoff_time"], errors="coerce").to_numpy(float)
    else:
        require_columns(frame.columns, ("feature_age_s",), product="transfer temporal provenance")
        age_input = pd.to_numeric(frame["feature_age_s"], errors="coerce").to_numpy(float)
        cutoff = decision - age_input
    age = decision - cutoff
    invalid = ~np.isfinite(decision) | ~np.isfinite(cutoff) | ~np.isfinite(age) | (age <= 0)
    if invalid.any():
        raise Stage2V52ContractError(
            f"transfer source has {int(invalid.sum())} rows violating feature_cutoff_time < decision_time"
        )
    return decision.astype(np.float64), cutoff.astype(np.float64), age.astype(np.float64)


def build_transfer_chunk_payload(
    frame: pd.DataFrame,
    v5_artifacts: dict[str, Any],
    *,
    static_artifact: Mapping[str, Any],
    support_artifact: Mapping[str, Any],
    max_seq_len: int,
    overlap: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Generate v5.1/v5.2 tensors from one canonical single-day partition."""
    require_columns(
        frame.columns,
        ("row_id", "split", "date", "order_id", "traversal_id", "route_sequence"),
        product="transfer shard source",
    )
    split, date = _partition_identity(frame)
    row_id = pd.to_numeric(frame["row_id"], errors="coerce")
    if row_id.isna().any() or row_id.duplicated().any() or (row_id % 1 != 0).any():
        raise Stage2V52ContractError("transfer row_id must be a unique finite integer")
    payload = vectorized_chunk_payload(frame, v5_artifacts, max_seq_len=max_seq_len, overlap=overlap)
    payload["split"] = np.full(len(payload["order_id"]), split, dtype=str)
    payload["date"] = np.full(len(payload["order_id"]), date, dtype=str)
    static, feature_names, static_row_id = build_static_structure_features(frame, static_artifact)
    validate_feature_alignment(frame["row_id"].to_numpy(), static_row_id)
    support, support_group = lookup_train_support(frame["observed_directed_edge_uid"], support_artifact)
    group_code = pd.Series(support_group).map(
        {"unseen": 0, "low": 1, "medium": 2, "high": 3}
    ).to_numpy(np.int8)
    temporal = build_temporal_features(frame)
    decision, cutoff, age = _temporal_source(frame)
    positions = _payload_source_positions(payload, frame)
    payload.update({
        "static_edge_features": _gather_aligned(static, positions, 0.0).astype(np.float32),
        "edge_train_support": _gather_aligned(support, positions, 0).astype(np.float32),
        "support_group_code": _gather_aligned(group_code, positions, -1).astype(np.int8),
        "temporal_features": _gather_aligned(temporal, positions, 0.0).astype(np.float32),
        "decision_time": _gather_aligned(decision, positions, np.nan).astype(np.float64),
        "feature_cutoff_time": _gather_aligned(cutoff, positions, np.nan).astype(np.float64),
        "feature_age_s": _gather_aligned(age, positions, np.nan).astype(np.float64),
        "row_id": _gather_aligned(row_id.to_numpy(np.int64), positions, -1),
    })
    return payload, {
        "schema_version": TRANSFER_TENSOR_SCHEMA_VERSION,
        "split": split,
        "date": date,
        "static_feature_names": list(feature_names),
        "static_feature_count": len(feature_names),
        "temporal_feature_names": list(TEMPORAL_FEATURE_NAMES),
        "identity_key": ["split", "date", "order_id", "traversal_id"],
        "row_alignment": "explicit_row_id_scattered_to_caller_order",
    }


def _resolve_route_path(route_feature_root: Path, date: str) -> Path:
    path = route_feature_root / f"day={date}.parquet"
    if not path.is_file():
        raise Stage2V52ContractError(f"missing frozen route-conditioned product: {path}")
    return path


def build_transfer_shards(
    *,
    protocol_id: str,
    repo_root: str | Path,
    route_feature_root: str | Path,
    feature_artifact_path: str | Path,
    support_artifact_path: str | Path,
    static_artifact_path: str | Path,
    stage1_release_manifest_path: str | Path,
    output_root: str | Path,
    max_seq_len: int,
    overlap: int,
    chunks_per_file: int,
) -> dict[str, Any]:
    """Build the canonical protocol=<id>/split=<role>/date=<day> layout."""
    protocol = get_protocol(protocol_id)
    feature = _json(feature_artifact_path)
    support = _json(support_artifact_path)
    static = _json(static_artifact_path)
    if tuple(feature.get("fit_dates", ())) != protocol.train_dates:
        raise Stage2V52ContractError("transfer feature artifact dates differ from protocol Train dates")
    if support.get("protocol_id") != protocol_id or tuple(support.get("fit_dates_observed", ())) != protocol.train_dates:
        raise Stage2V52ContractError("transfer support artifact is not bound to this protocol")
    if static.get("protocol_id") != protocol_id or tuple(static.get("fit_dates_observed", ())) != protocol.train_dates:
        raise Stage2V52ContractError("transfer static artifact is not bound to this protocol")
    if min(max_seq_len, chunks_per_file) <= 0 or not 0 <= overlap < max_seq_len:
        raise Stage2V52ContractError("invalid frozen transfer shard chunk configuration")
    root = Path(repo_root).resolve()
    route_root = Path(route_feature_root).resolve()
    output = Path(output_root).resolve() / f"protocol={protocol_id}"
    stage1_release_path = Path(stage1_release_manifest_path).resolve()
    stage1_release = _json(stage1_release_path)
    if not stage1_release.get("release_tag"):
        raise Stage2V52ContractError("transfer shards require the frozen Stage 1 release manifest")
    day_manifests: list[dict[str, Any]] = []
    temporal_age_sample: list[np.ndarray] = []
    temporal_token_count = 0
    temporal_invalid_count = 0
    minimum_valid_age = np.inf
    for role, dates in protocol_role_dates(protocol_id).items():
        for date in dates:
            started = time.perf_counter()
            route_path = _resolve_route_path(route_root, date)
            frame = load_v5_day(
                date, split=role, repo_root=root, route_feature_root=route_root,
                extra_columns=(
                    "canonical_highway", "road_class", "observed_direction", "bridge", "tunnel",
                    "synthetic_reverse_edge", "osm_direction_disagreement", "feature_age_s",
                ),
            ).copy()
            frame["split"] = role
            frame["date"] = date
            frame["row_id"] = np.arange(len(frame), dtype=np.int64)
            payload, tensor_schema = build_transfer_chunk_payload(
                frame, feature, static_artifact=static, support_artifact=support,
                max_seq_len=max_seq_len, overlap=overlap,
            )
            valid = ~payload["pad_mask"]
            ages = payload["feature_age_s"][valid]
            invalid = ~np.isfinite(ages) | (ages <= 0)
            valid_ages = ages[~invalid]
            temporal_token_count += int(len(ages))
            temporal_invalid_count += int(invalid.sum())
            if len(valid_ages):
                minimum_valid_age = min(minimum_valid_age, float(valid_ages.min()))
                remaining = max(0, 1_000_000 - sum(len(item) for item in temporal_age_sample))
                if remaining:
                    temporal_age_sample.append(valid_ages[:remaining].copy())
            day_root = output / f"split={role}" / f"date={date}"
            files: list[dict[str, Any]] = []
            for index, start in enumerate(range(0, len(payload["order_id"]), chunks_per_file)):
                end = min(start + chunks_per_file, len(payload["order_id"]))
                path = day_root / f"shard-{index:05d}.npz"
                _atomic_npz(path, {name: values[start:end] for name, values in payload.items()})
                files.append({
                    "path": path.relative_to(output).as_posix(),
                    "sha256": _sha256(path), "chunk_count": end - start,
                })
            day_manifest = {
                "schema_version": TRANSFER_DAY_SCHEMA_VERSION,
                "protocol_id": protocol_id, "protocol_hash": protocol.digest,
                "split": role, "date": date,
                "source_route_path": route_path.as_posix(),
                "source_route_sha256": _sha256(route_path),
                "feature_artifact_sha256": _sha256(feature_artifact_path),
                "support_artifact_sha256": _sha256(support_artifact_path),
                "static_artifact_sha256": _sha256(static_artifact_path),
                "stage1_release_manifest_sha256": _sha256(stage1_release_path),
                "source_row_count": int(len(frame)),
                "chunk_count": int(len(payload["order_id"])),
                "chunk_token_count": int(valid.sum()),
                "unique_traversal_count": int(frame[["date", "order_id", "traversal_id"]].drop_duplicates().shape[0]),
                "tensor_schema": tensor_schema,
                "runtime_s": time.perf_counter() - started,
                "files": files,
            }
            _atomic_json(day_root / "manifest.json", day_manifest)
            day_manifests.append(day_manifest)
            del frame, payload
    valid_age = np.concatenate(temporal_age_sample) if temporal_age_sample else np.array([], dtype=float)
    temporal_report = {
        "schema_version": "stage2_v5_2_transfer_temporal_audit.1",
        "status": "PASS" if temporal_invalid_count == 0 and temporal_token_count > 0 else "FAIL",
        "protocol_id": protocol_id,
        "protocol_hash": protocol.digest,
        "temporal_leakage_count": temporal_invalid_count,
        "audited_token_count": temporal_token_count,
        "minimum_feature_age_s": float(minimum_valid_age) if np.isfinite(minimum_valid_age) else None,
        "p01_feature_age_s": float(np.quantile(valid_age, 0.01)) if len(valid_age) else None,
        "p50_feature_age_s": float(np.quantile(valid_age, 0.50)) if len(valid_age) else None,
        "p99_feature_age_s": float(np.quantile(valid_age, 0.99)) if len(valid_age) else None,
        "quantile_sample_policy": "deterministic_first_1000000_valid_tokens",
        "quantile_sample_count": int(len(valid_age)),
    }
    temporal_path = output / "temporal_audit.json"
    _atomic_json(temporal_path, temporal_report)
    manifest = {
        "schema_version": TRANSFER_MANIFEST_SCHEMA_VERSION,
        "status": "PASS" if temporal_report["status"] == "PASS" else "FAIL",
        "protocol_id": protocol_id, "protocol_hash": protocol.digest,
        "canonical_layout": "protocol=<id>/split=<role>/date=<YYYYMMDD>/shard-*.npz",
        "canonical_roles": list(protocol_role_dates(protocol_id)),
        "feature_artifact_sha256": _sha256(feature_artifact_path),
        "support_artifact_sha256": _sha256(support_artifact_path),
        "static_artifact_sha256": _sha256(static_artifact_path),
        "stage1_release_manifest_path": stage1_release_path.as_posix(),
        "stage1_release_manifest_sha256": _sha256(stage1_release_path),
        "stage1_release_tag": stage1_release["release_tag"],
        "temporal_audit_path": temporal_path.relative_to(output).as_posix(),
        "temporal_audit_sha256": _sha256(temporal_path),
        "day_count": len(day_manifests),
        "source_row_count": int(sum(item["source_row_count"] for item in day_manifests)),
        "chunk_count": int(sum(item["chunk_count"] for item in day_manifests)),
        "days": day_manifests,
    }
    _atomic_json(output / "transfer_manifest.json", manifest)
    return manifest
