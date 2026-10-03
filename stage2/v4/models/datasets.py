"""Continuous-overlap chunking and Train-only tensor artifacts."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..config import Stage2V4Config
from ..contracts import Stage2V4ContractError
from ..io import atomic_write_json, sha256_file, stage2_v4_code_identity
from .baselines import _feature_candidates


SHARD_SCHEMA_VERSION = "stage2_v4_tensor_shards.1"
ARTIFACT_SCHEMA_VERSION = "stage2_v4_tensor_artifacts.1"
CONTINUOUS_TARGETS = (
    "crawl_time_share",
    "stop_time_share",
    "speed_cv_bounded",
    "acceleration_rms_bounded",
    "rts_raw",
    "lcs_raw",
)
TARGET_MASKS = (
    "crawl_target_valid",
    "stop_target_valid",
    "speed_cv_target_valid",
    "acceleration_rms_target_valid",
    "rts_target_valid",
    "lcs_target_valid",
)
TAIL_TARGETS = ("lcs_tail_event", "rts_tail_event")
TAIL_MASKS = ("lcs_target_valid", "rts_target_valid")
PAD_TOKEN = "__PAD__"
UNSEEN_TOKEN = "__UNSEEN__"
RARE_TOKEN = "__RARE__"
MISSING_TOKEN = "__MISSING__"
RESERVED_TOKENS = {PAD_TOKEN, UNSEEN_TOKEN, RARE_TOKEN, MISSING_TOKEN}


def continuous_chunk_starts(
    length: int,
    *,
    max_seq_len: int,
    overlap: int,
) -> tuple[int, ...]:
    if length <= 0:
        return ()
    if not (0 <= overlap < max_seq_len):
        raise Stage2V4ContractError("chunk overlap must be in [0, max_seq_len)")
    if length <= max_seq_len:
        return (0,)
    stride = max_seq_len - overlap
    starts = list(range(0, length - max_seq_len + 1, stride))
    last = length - max_seq_len
    if starts[-1] != last:
        starts.append(last)
    return tuple(starts)


def _stage2_split(date: str, config: Stage2V4Config) -> str:
    split = config.section("split")
    if date in split["train_dates"]:
        return "train"
    if date in split["validation_model_dates"]:
        return "validation_model"
    if date in split["calibration_dates"]:
        return "calibration"
    if date in split["test_dates"]:
        return "test"
    raise Stage2V4ContractError(f"date is outside frozen Stage 2 split: {date}")


def _dataset_files(root: Path, dates: Iterable[str]) -> list[Path]:
    files = [root / f"day={date}.parquet" for date in dates]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise Stage2V4ContractError(f"tensor dataset files are missing: {missing}")
    return files


def _categorical_values(frame: pd.DataFrame) -> dict[str, pd.Series]:
    position = np.floor(
        np.clip(
            pd.to_numeric(frame["route_position_ratio"], errors="coerce").fillna(0.0),
            0.0,
            0.999999,
        )
        * 20
    ).astype(int)
    route_length = pd.to_numeric(
        frame["route_token_count"],
        errors="coerce",
    ).fillna(0)
    length_bucket = pd.cut(
        route_length,
        bins=[-np.inf, 20, 40, 80, 120, np.inf],
        labels=False,
    ).astype(int)
    raw = {
        "edge": frame["observed_directed_edge_uid"],
        "highway": frame["canonical_highway"],
        "time_bin": pd.to_numeric(
            frame["estimated_time_bin"],
            errors="coerce",
        ).fillna(-1).astype(int),
        "position_bucket": position,
        "route_length_bucket": length_bucket,
    }
    return {
        name: values.astype("string").fillna(MISSING_TOKEN).astype(str)
        for name, values in raw.items()
    }


def _fit_artifacts(
    train_files: list[Path],
    config: Stage2V4Config,
) -> dict[str, Any]:
    schema = set(pq.read_schema(train_files[0]).names)
    numeric_features = tuple(
        column for column in _feature_candidates() if column in schema
    )
    sums = np.zeros(len(numeric_features), dtype=np.float64)
    sum_squares = np.zeros(len(numeric_features), dtype=np.float64)
    counts = np.zeros(len(numeric_features), dtype=np.int64)
    category_counts = {
        name: Counter()
        for name in (
            "edge",
            "highway",
            "time_bin",
            "position_bucket",
            "route_length_bucket",
        )
    }
    columns = list(
        dict.fromkeys(
            [
                *numeric_features,
                "observed_directed_edge_uid",
                "canonical_highway",
                "estimated_time_bin",
                "route_position_ratio",
                "route_token_count",
            ]
        )
    )
    for path in train_files:
        frame = pd.read_parquet(path, columns=columns)
        numeric = frame.loc[:, numeric_features].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(dtype=np.float64, na_value=np.nan)
        finite = np.isfinite(numeric)
        sums += np.where(finite, numeric, 0.0).sum(axis=0)
        sum_squares += np.where(finite, numeric * numeric, 0.0).sum(axis=0)
        counts += finite.sum(axis=0)
        for name, values in _categorical_values(frame).items():
            category_counts[name].update(values.tolist())
    means = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums),
        where=counts > 0,
    )
    variance = np.divide(
        sum_squares,
        counts,
        out=np.ones_like(sums),
        where=counts > 0,
    ) - means * means
    std = np.sqrt(np.maximum(variance, 1e-8))
    minimum_edge = int(config.section("shards")["minimum_edge_frequency"])
    vocabularies: dict[str, dict[str, Any]] = {}
    for name, counter in category_counts.items():
        minimum = minimum_edge if name == "edge" else 1
        frequent = sorted(
            str(key)
            for key, count in counter.items()
            if count >= minimum and str(key) not in RESERVED_TOKENS
        )
        token_to_index = {
            PAD_TOKEN: 0,
            UNSEEN_TOKEN: 1,
            RARE_TOKEN: 2,
            MISSING_TOKEN: 3,
            **{token: index + 4 for index, token in enumerate(frequent)},
        }
        vocabularies[name] = {
            "token_to_index": token_to_index,
            "seen_tokens": sorted(str(key) for key in counter),
            "minimum_frequency": minimum,
        }
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "stage2_config_sha256": config.digest,
        "fit_dates": list(config.section("split")["train_dates"]),
        "numeric_features": list(numeric_features),
        "numeric_mean": means.tolist(),
        "numeric_std": std.tolist(),
        "numeric_count": counts.tolist(),
        "vocabularies": vocabularies,
    }


def encode_categorical(
    values: pd.Series,
    vocabulary: dict[str, Any],
) -> np.ndarray:
    mapping = vocabulary["token_to_index"]
    seen = set(vocabulary["seen_tokens"])
    encoded = np.empty(len(values), dtype=np.int64)
    normalized = values.astype("string").fillna(MISSING_TOKEN).astype(str)
    for index, value in enumerate(normalized):
        if value in mapping:
            encoded[index] = int(mapping[value])
        elif value in seen:
            encoded[index] = int(mapping[RARE_TOKEN])
        else:
            encoded[index] = int(mapping[UNSEEN_TOKEN])
    return encoded


def _atomic_save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        suffix=".npz",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _pad(values: np.ndarray, length: int, fill: Any) -> np.ndarray:
    shape = (length, *values.shape[1:])
    result = np.full(shape, fill, dtype=values.dtype)
    result[: len(values)] = values
    return result


def _chunk_payload(
    frame: pd.DataFrame,
    artifacts: dict[str, Any],
    *,
    max_seq_len: int,
    overlap: int,
) -> list[dict[str, np.ndarray]]:
    return list(
        _iter_chunk_payloads(
            frame,
            artifacts,
            max_seq_len=max_seq_len,
            overlap=overlap,
        )
    )


def _iter_chunk_payloads(
    frame: pd.DataFrame,
    artifacts: dict[str, Any],
    *,
    max_seq_len: int,
    overlap: int,
):
    numeric_features = artifacts["numeric_features"]
    means = np.asarray(artifacts["numeric_mean"], dtype=np.float32)
    std = np.asarray(artifacts["numeric_std"], dtype=np.float32)
    for order_id, order in frame.groupby("order_id", sort=False, observed=True):
        order = order.sort_values("route_sequence", kind="stable")
        sequence = pd.to_numeric(order["route_sequence"], errors="raise").to_numpy(
            dtype=np.int64
        )
        if len(sequence) > 1 and not np.all(np.diff(sequence) == 1):
            raise Stage2V4ContractError(
                f"route contains non-contiguous sequence before chunking: {order_id}"
            )
        numeric_raw = order.loc[:, numeric_features].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(dtype=np.float32, na_value=np.nan)
        numeric_missing = ~np.isfinite(numeric_raw)
        numeric = np.where(numeric_missing, means, numeric_raw)
        numeric = (numeric - means) / std
        categories = _categorical_values(order)
        categorical = np.column_stack(
            [
                encode_categorical(categories[name], artifacts["vocabularies"][name])
                for name in (
                    "edge",
                    "highway",
                    "time_bin",
                    "position_bucket",
                    "route_length_bucket",
                )
            ]
        )
        targets = order.loc[:, CONTINUOUS_TARGETS].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(dtype=np.float32, na_value=np.nan)
        target_masks = order.loc[:, TARGET_MASKS].fillna(False).to_numpy(dtype=bool)
        targets = np.where(target_masks, targets, 0.0)
        tail_targets = np.column_stack(
            [
                order[column].astype("boolean").fillna(False).to_numpy(dtype=np.float32)
                for column in TAIL_TARGETS
            ]
        )
        tail_masks = order.loc[:, TAIL_MASKS].fillna(False).to_numpy(dtype=bool)
        traversal_id = pd.to_numeric(order["traversal_id"], errors="raise").to_numpy(
            dtype=np.int64
        )
        weights = np.column_stack(
            [
                pd.to_numeric(
                    order["estimated_travel_time_s"],
                    errors="coerce",
                ).fillna(0.0),
                pd.to_numeric(order["route_part_length_m"], errors="coerce").fillna(0.0),
            ]
        ).astype(np.float32)
        for chunk_id, start in enumerate(
            continuous_chunk_starts(
                len(order),
                max_seq_len=max_seq_len,
                overlap=overlap,
            )
        ):
            end = min(start + max_seq_len, len(order))
            valid_length = end - start
            pad_mask = np.ones(max_seq_len, dtype=bool)
            pad_mask[:valid_length] = False
            yield {
                "numeric": _pad(numeric[start:end], max_seq_len, 0.0),
                "numeric_missing": _pad(
                    numeric_missing[start:end],
                    max_seq_len,
                    True,
                ),
                "categorical": _pad(
                    categorical[start:end],
                    max_seq_len,
                    0,
                ),
                "targets": _pad(targets[start:end], max_seq_len, 0.0),
                "target_masks": _pad(
                    target_masks[start:end],
                    max_seq_len,
                    False,
                ),
                "tail_targets": _pad(
                    tail_targets[start:end],
                    max_seq_len,
                    0.0,
                ),
                "tail_masks": _pad(
                    tail_masks[start:end],
                    max_seq_len,
                    False,
                ),
                "route_sequence": _pad(
                    sequence[start:end],
                    max_seq_len,
                    -1,
                ),
                "traversal_id": _pad(
                    traversal_id[start:end],
                    max_seq_len,
                    -1,
                ),
                "aggregation_weights": _pad(
                    weights[start:end],
                    max_seq_len,
                    0.0,
                ),
                "pad_mask": pad_mask,
                "order_id": np.asarray(str(order_id)),
                "chunk_id": np.asarray(chunk_id, dtype=np.int32),
                "chunk_start_sequence": np.asarray(
                    sequence[start],
                    dtype=np.int64,
                ),
                "chunk_end_sequence": np.asarray(
                    sequence[end - 1],
                    dtype=np.int64,
                ),
            }


def _stack_chunks(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.stack([chunk[key] for chunk in chunks])
        for key in chunks[0]
    }


def build_tensor_shards(
    dataset_root: str | Path,
    output_root: str | Path,
    config: Stage2V4Config,
    *,
    resume: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    dataset = Path(dataset_root)
    output = Path(output_root)
    split = config.section("split")
    all_dates = tuple(
        [
            *split["train_dates"],
            *split["validation_model_dates"],
            *split["calibration_dates"],
            *split["test_dates"],
        ]
    )
    train_files = _dataset_files(dataset, split["train_dates"])
    artifact_path = output / "feature_artifacts.json"
    if artifact_path.exists() and resume:
        artifacts = json.loads(artifact_path.read_text(encoding="utf-8"))
        if (
            artifacts.get("schema_version") != ARTIFACT_SCHEMA_VERSION
            or artifacts.get("stage2_config_sha256") != config.digest
        ):
            if not force:
                raise Stage2V4ContractError("tensor artifacts are not resumable")
            artifacts = _fit_artifacts(train_files, config)
            atomic_write_json(artifact_path, artifacts)
    else:
        if artifact_path.exists() and not force:
            raise Stage2V4ContractError("tensor artifacts exist; use --resume or --force")
        artifacts = _fit_artifacts(train_files, config)
        atomic_write_json(artifact_path, artifacts)
    artifact_sha = sha256_file(artifact_path)

    shard_config = config.section("shards")
    max_seq_len = int(shard_config["max_seq_len"])
    overlap = int(shard_config["overlap"])
    chunks_per_file = int(shard_config["chunks_per_file"])
    day_manifests: list[dict[str, Any]] = []
    for date in all_dates:
        stage2_split = _stage2_split(date, config)
        day_root = output / f"split={stage2_split}" / f"date={date}"
        manifest_path = day_root / "manifest.json"
        if manifest_path.is_file() and resume:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("engineering_status") == "PASS"
                and manifest.get("stage2_config_sha256") == config.digest
                and manifest.get("artifact_sha256") == artifact_sha
                and all(
                    (day_root / item["name"]).is_file()
                    and sha256_file(day_root / item["name"]) == item["sha256"]
                    for item in manifest.get("shards", [])
                )
            ):
                day_manifests.append(manifest)
                continue
        elif manifest_path.exists() and not force:
            raise Stage2V4ContractError(
                f"tensor shard day exists; use --resume or --force: {date}"
            )
        source = dataset / f"day={date}.parquet"
        required = list(
            dict.fromkeys(
                [
                    "order_id",
                    "traversal_id",
                    "route_sequence",
                    "observed_directed_edge_uid",
                    "canonical_highway",
                    "estimated_time_bin",
                    "route_position_ratio",
                    "route_token_count",
                    "estimated_travel_time_s",
                    "route_part_length_m",
                    *artifacts["numeric_features"],
                    *CONTINUOUS_TARGETS,
                    *TARGET_MASKS,
                    *TAIL_TARGETS,
                ]
            )
        )
        frame = pd.read_parquet(source, columns=required)
        shard_identities: list[dict[str, Any]] = []
        buffer: list[dict[str, np.ndarray]] = []
        shard_index = 0
        chunk_count = 0
        for chunk in _iter_chunk_payloads(
            frame,
            artifacts,
            max_seq_len=max_seq_len,
            overlap=overlap,
        ):
            buffer.append(chunk)
            chunk_count += 1
            if len(buffer) < chunks_per_file:
                continue
            path = day_root / f"shard={shard_index:05d}.npz"
            _atomic_save_npz(path, _stack_chunks(buffer))
            shard_identities.append(
                {
                    "name": path.name,
                    "chunk_count": len(buffer),
                    "sha256": sha256_file(path),
                }
            )
            buffer = []
            shard_index += 1
        if buffer:
            path = day_root / f"shard={shard_index:05d}.npz"
            _atomic_save_npz(path, _stack_chunks(buffer))
            shard_identities.append(
                {
                    "name": path.name,
                    "chunk_count": len(buffer),
                    "sha256": sha256_file(path),
                }
            )
        manifest = {
            "schema_version": SHARD_SCHEMA_VERSION,
            "engineering_status": "PASS",
            "stage2_config_sha256": config.digest,
            "artifact_sha256": artifact_sha,
            "split": stage2_split,
            "date": date,
            "source_file_sha256": sha256_file(source),
            "order_count": int(frame["order_id"].nunique()),
            "route_token_count": int(len(frame)),
            "chunk_count": chunk_count,
            "max_seq_len": max_seq_len,
            "overlap": overlap,
            "shards": shard_identities,
        }
        atomic_write_json(manifest_path, manifest)
        day_manifests.append(manifest)
        del frame

    summary = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "engineering_status": "PASS",
        "stage2_config_sha256": config.digest,
        "stage2_code_sha": stage2_v4_code_identity(
            (
                "stage2/v4/models/datasets.py",
                "stage2/v4/contracts.py",
            )
        ),
        "artifact_sha256": artifact_sha,
        "day_count": len(day_manifests),
        "order_count": sum(int(item["order_count"]) for item in day_manifests),
        "route_token_count": sum(
            int(item["route_token_count"]) for item in day_manifests
        ),
        "chunk_count": sum(int(item["chunk_count"]) for item in day_manifests),
        "runtime_s": time.perf_counter() - started,
    }
    atomic_write_json(output / "tensor_manifest.json", summary)
    return summary
