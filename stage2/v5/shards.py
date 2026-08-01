"""Vectorized continuous-route tensor shard builder for RC-MSTNet v5."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stage2.v4.models.baselines import _feature_candidates

from .config import Stage2V5Config, load_config
from .contracts import Stage2V5ContractError
from .data import PROFILE_PACE_COLUMNS, RECENT_PACE_COLUMNS, load_v5_day


CONTINUOUS_TARGETS = (
    "crawl_time_share", "stop_time_share", "speed_cv_bounded",
    "acceleration_rms_bounded", "rts_raw", "lcs_raw", "pace_sec_per_m",
)
TARGET_MASKS = (
    "crawl_target_valid", "stop_target_valid", "speed_cv_target_valid",
    "acceleration_rms_target_valid", "rts_target_valid", "lcs_target_valid",
    "pace_target_valid",
)
TAIL_TARGETS = ("lcs_tail_event", "rts_tail_event")
TAIL_MASKS = ("lcs_target_valid", "rts_target_valid")
CATEGORY_NAMES = ("edge", "highway", "time_bin", "position_bucket", "route_length_bucket")
RESERVED_TOKENS = ("__PAD__", "__UNSEEN__", "__RARE__", "__MISSING__")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _categorical_values(frame: pd.DataFrame) -> dict[str, pd.Series]:
    position = np.floor(np.clip(pd.to_numeric(frame["route_position_ratio"], errors="coerce").fillna(0.0), 0.0, 0.999999) * 20).astype(int)
    length = pd.to_numeric(frame["route_token_count"], errors="coerce").fillna(0)
    length_bucket = pd.cut(length, bins=[-np.inf, 20, 40, 80, 120, np.inf], labels=False).astype(int)
    raw = {
        "edge": frame["observed_directed_edge_uid"],
        "highway": frame["canonical_highway"],
        "time_bin": pd.to_numeric(frame["estimated_time_bin"], errors="coerce").fillna(-1).astype(int),
        "position_bucket": position,
        "route_length_bucket": length_bucket,
    }
    return {name: values.astype("string").fillna("__MISSING__").astype(str) for name, values in raw.items()}


def _encode(values: pd.Series, vocabulary: dict[str, Any]) -> np.ndarray:
    mapping = vocabulary["token_to_index"]
    normalized = values.astype("string").fillna("__MISSING__").astype(str)
    mapped = normalized.map(mapping)
    seen = normalized.isin(vocabulary["seen_tokens"])
    result = mapped.fillna(mapping["__UNSEEN__"]).to_numpy(dtype=np.int64, copy=True)
    rare = mapped.isna().to_numpy() & seen.to_numpy()
    result[rare] = int(mapping["__RARE__"])
    return result


def fit_feature_artifacts(
    config: Stage2V5Config,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Fit normalization and vocabularies on this protocol's Train dates only."""

    root = Path(repo_root).resolve()
    feature_root = root / "stage2/output_v4/route_conditioned_dataset/revealed_route_proxy"
    dates = list(config.section("split")["train_dates"])
    paths = [feature_root / f"day={date}.parquet" for date in dates]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise Stage2V5ContractError(f"missing frozen route features: {missing}")
    schema = set(pq.read_schema(paths[0]).names)
    numeric_features = tuple(column for column in _feature_candidates() if column in schema)
    sums = np.zeros(len(numeric_features), dtype=np.float64)
    sum_squares = np.zeros(len(numeric_features), dtype=np.float64)
    counts = np.zeros(len(numeric_features), dtype=np.int64)
    category_counts = {name: Counter() for name in CATEGORY_NAMES}
    columns = list(dict.fromkeys([
        *numeric_features,
        "observed_directed_edge_uid", "canonical_highway", "estimated_time_bin",
        "route_position_ratio", "route_token_count",
    ]))
    for path in paths:  # Bounded one-day streaming over frozen route features.
        frame = pd.read_parquet(path, columns=columns)
        numeric = frame.loc[:, numeric_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        finite = np.isfinite(numeric)
        sums += np.where(finite, numeric, 0.0).sum(axis=0)
        sum_squares += np.where(finite, numeric * numeric, 0.0).sum(axis=0)
        counts += finite.sum(axis=0)
        for name, values in _categorical_values(frame).items():
            category_counts[name].update(values.tolist())
        del frame, numeric
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    variance = np.divide(sum_squares, counts, out=np.ones_like(sums), where=counts > 0) - means * means
    std = np.sqrt(np.maximum(variance, 1e-8))
    minimum_edge = int(config.section("shards")["minimum_edge_frequency"])
    vocabularies: dict[str, dict[str, Any]] = {}
    for name, counter in category_counts.items():
        minimum = minimum_edge if name == "edge" else 1
        frequent = sorted(
            str(key) for key, count in counter.items()
            if count >= minimum and str(key) not in RESERVED_TOKENS
        )
        mapping = {token: index for index, token in enumerate(RESERVED_TOKENS)}
        mapping.update({token: index + len(RESERVED_TOKENS) for index, token in enumerate(frequent)})
        vocabularies[name] = {
            "token_to_index": mapping,
            "seen_tokens": sorted(str(key) for key in counter),
            "minimum_frequency": minimum,
        }
    return {
        "schema_version": "stage2_v5_tensor_artifacts.2",
        "stage2_v5_config_sha256": config.digest,
        "fit_dates": dates,
        "numeric_features": list(numeric_features),
        "numeric_mean": means.tolist(),
        "numeric_std": std.tolist(),
        "numeric_count": counts.tolist(),
        "vocabularies": vocabularies,
        "fit_scope": "train_only_per_protocol_or_fold",
        "percentile_supervision_allowed": config.section("split")["protocol_name"] == "legacy_frozen_benchmark",
    }


def _gather(values: np.ndarray, token_index: np.ndarray, valid: np.ndarray, fill: Any) -> np.ndarray:
    safe = np.where(valid, token_index, 0)
    gathered = values[safe]
    if gathered.ndim == valid.ndim:
        return np.where(valid, gathered, fill)
    return np.where(valid[..., None], gathered, fill)


def vectorized_chunk_payload(
    frame: pd.DataFrame,
    artifacts: dict[str, Any],
    *,
    max_seq_len: int,
    overlap: int,
) -> dict[str, np.ndarray]:
    if not 0 <= overlap < max_seq_len:
        raise Stage2V5ContractError("chunk overlap must be in [0,max_seq_len)")
    working = frame.sort_values(["order_id", "route_sequence"], kind="stable").reset_index(drop=True)
    order = working["order_id"].astype(str).to_numpy()
    sequence = pd.to_numeric(working["route_sequence"], errors="raise").to_numpy(np.int64)
    group_start_flag = np.concatenate((np.array([True]), order[1:] != order[:-1]))
    starts = np.flatnonzero(group_start_flag)
    ends = np.concatenate((starts[1:], np.array([len(working)])))
    lengths = ends - starts
    contiguous = group_start_flag | (sequence == np.concatenate((np.array([0]), sequence[:-1] + 1)))
    if not contiguous.all():
        raise Stage2V5ContractError("route sequence is not contiguous before chunking")
    stride = max_seq_len - overlap
    chunk_count = np.maximum(1, np.ceil(np.maximum(lengths - max_seq_len, 0) / stride).astype(np.int64) + 1)
    group_code = np.repeat(np.arange(len(starts), dtype=np.int64), chunk_count)
    group_offsets = np.repeat(np.cumsum(chunk_count) - chunk_count, chunk_count)
    local_chunk = np.arange(len(group_code), dtype=np.int64) - group_offsets
    local_start = np.minimum(local_chunk * stride, np.maximum(lengths[group_code] - max_seq_len, 0))
    absolute_start = starts[group_code] + local_start
    token_index = absolute_start[:, None] + np.arange(max_seq_len, dtype=np.int64)
    valid = token_index < ends[group_code, None]
    flat_token = token_index[valid]
    supervision_count = np.bincount(flat_token, minlength=len(working))
    token_supervision_count = _gather(supervision_count, token_index, valid, 0).astype(np.int32)
    supervision_weight = np.divide(
        1.0,
        token_supervision_count,
        out=np.zeros(token_supervision_count.shape, dtype=np.float32),
        where=token_supervision_count > 0,
    )

    numeric_features = tuple(artifacts["numeric_features"])
    numeric_raw = working.loc[:, numeric_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, na_value=np.nan)
    numeric_missing = ~np.isfinite(numeric_raw)
    mean = np.asarray(artifacts["numeric_mean"], dtype=np.float32)
    std = np.asarray(artifacts["numeric_std"], dtype=np.float32)
    numeric = (np.where(numeric_missing, mean, numeric_raw) - mean) / std
    categories = _categorical_values(working)
    categorical = np.column_stack([_encode(categories[name], artifacts["vocabularies"][name]) for name in CATEGORY_NAMES])
    targets = working.loc[:, CONTINUOUS_TARGETS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, na_value=np.nan)
    target_masks = working.loc[:, TARGET_MASKS].fillna(False).to_numpy(bool)
    targets = np.where(target_masks, targets, 0.0)
    tail_targets = np.column_stack([working[column].astype("boolean").fillna(False).to_numpy(np.float32) for column in TAIL_TARGETS])
    tail_masks = working.loc[:, TAIL_MASKS].fillna(False).to_numpy(bool)
    if not bool(artifacts.get("percentile_supervision_allowed", False)):
        tail_masks[:] = False
    dynamics = target_masks[:, :4].all(axis=1)
    availability = np.column_stack((target_masks[:, 6], target_masks[:, 5], target_masks[:, 4], dynamics)).astype(np.float32)
    recent = working.loc[:, RECENT_PACE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, na_value=np.nan)
    profile = working.loc[:, PROFILE_PACE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, na_value=np.nan)
    history_age = pd.to_numeric(working["feature_age_s"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    history_support = pd.to_numeric(working["observed_sec_per_m_profile_count"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    horizon = pd.to_numeric(working["forecast_horizon_s"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    distance = pd.to_numeric(working["allocated_distance_m"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    traversal_id = pd.to_numeric(working["traversal_id"], errors="raise").to_numpy(np.int64)
    return {
        "numeric": _gather(numeric, token_index, valid, 0.0).astype(np.float32),
        "numeric_missing": _gather(numeric_missing, token_index, valid, True).astype(bool),
        "categorical": _gather(categorical, token_index, valid, 0).astype(np.int64),
        "targets": _gather(targets, token_index, valid, 0.0).astype(np.float32),
        "target_masks": _gather(target_masks, token_index, valid, False).astype(bool),
        "tail_targets": _gather(tail_targets, token_index, valid, 0.0).astype(np.float32),
        "tail_masks": _gather(tail_masks, token_index, valid, False).astype(bool),
        "availability_targets": _gather(availability, token_index, valid, 0.0).astype(np.float32),
        "recent_history": _gather(np.nan_to_num(recent, nan=0.0), token_index, valid, 0.0).astype(np.float32),
        "profile_history": _gather(np.nan_to_num(profile, nan=0.0), token_index, valid, 0.0).astype(np.float32),
        "forecast_horizon_s": _gather(horizon, token_index, valid, 0.0).astype(np.float32),
        "history_age_s": _gather(history_age, token_index, valid, 0.0).astype(np.float32),
        "history_support": _gather(history_support, token_index, valid, 0.0).astype(np.float32),
        "allocated_distance_m": _gather(distance, token_index, valid, 0.0).astype(np.float32),
        "route_sequence": _gather(sequence, token_index, valid, -1).astype(np.int64),
        "traversal_id": _gather(traversal_id, token_index, valid, -1).astype(np.int64),
        "pad_mask": ~valid,
        "overlap_supervision_count": token_supervision_count,
        "supervision_weight": supervision_weight,
        "order_id": order[starts[group_code]].astype(str),
        "chunk_id": local_chunk.astype(np.int32),
    }


def build_shards(config: Stage2V5Config, *, repo_root: str | Path = ".", output_root: str | Path = "stage2/output_v5/tensor_shards", resume: bool = False) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    output = root / output_root
    artifact_path = output / "feature_artifacts.json"
    if resume and artifact_path.is_file():
        artifacts = json.loads(artifact_path.read_text(encoding="utf-8"))
        if artifacts.get("stage2_v5_config_sha256") != config.digest:
            raise Stage2V5ContractError("tensor artifacts do not match this frozen protocol")
    else:
        artifacts = fit_feature_artifacts(config, repo_root=root)
    split = config.section("split")
    _atomic_json(artifact_path, artifacts)
    dates = [("train", date) for date in split["train_dates"]]
    dates += [("validation_model", date) for date in split["validation_model_dates"]]
    dates += [("calibration", date) for date in split["calibration_dates"]]
    dates += [("evaluation", date) for date in split["evaluation_dates"]]
    dates += [("legacy", date) for date in split["legacy_test_dates"]]
    shard = config.section("shards")
    manifests: list[dict[str, Any]] = []
    for split_name, date in dates:
        day_root = output / f"split={split_name}" / f"date={date}"
        manifest_path = day_root / "manifest.json"
        if resume and manifest_path.is_file():
            manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
            continue
        started = time.perf_counter()
        frame = load_v5_day(date, split=split_name, repo_root=root)
        payload = vectorized_chunk_payload(frame, artifacts, max_seq_len=int(shard["max_seq_len"]), overlap=int(shard["overlap"]))
        chunks_per_file = int(shard["chunks_per_file"])
        files: list[dict[str, Any]] = []
        for file_index, start in enumerate(range(0, len(payload["order_id"]), chunks_per_file)):
            end = min(start + chunks_per_file, len(payload["order_id"]))
            path = day_root / f"shard-{file_index:05d}.npz"
            _atomic_npz(path, {name: values[start:end] for name, values in payload.items()})
            files.append({"path": path.relative_to(output).as_posix(), "chunk_count": end - start, "sha256": _sha256(path)})
        manifest = {
            "schema_version": "stage2_v5_tensor_day.1",
            "split": split_name,
            "date": date,
            "source_row_count": len(frame),
            "chunk_count": len(payload["order_id"]),
            "pace_target_token_count_with_overlap": int((payload["target_masks"][..., 6] & ~payload["pad_mask"]).sum()),
            "supervision_weight_sum": float(payload["supervision_weight"].sum()),
            "runtime_s": time.perf_counter() - started,
            "files": files,
        }
        _atomic_json(manifest_path, manifest)
        manifests.append(manifest)
        del frame, payload
    overall = {
        "schema_version": "stage2_v5_tensor_shards.1",
        "config_sha256": config.digest,
        "feature_artifact_sha256": _sha256(artifact_path),
        "day_count": len(manifests),
        "source_row_count": int(sum(item["source_row_count"] for item in manifests)),
        "chunk_count": int(sum(item["chunk_count"] for item in manifests)),
        "days": manifests,
    }
    _atomic_json(output / "tensor_manifest.json", overall)
    return overall


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage2/config/stage2_v5.json")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-root", default="stage2/output_v5/tensor_shards")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = build_shards(load_config(args.config), repo_root=args.repo_root, output_root=args.output_root, resume=args.resume)
    print(json.dumps({key: report[key] for key in ("schema_version", "day_count", "source_row_count", "chunk_count")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
