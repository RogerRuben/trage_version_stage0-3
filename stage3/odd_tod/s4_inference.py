"""Prediction-only frozen-M3 inference for the Stage 3 S4 Test31 route.

This module deliberately does not use :func:`stage2.v5.data.load_v5_day` or
the Stage 2 evaluator.  Both paths merge realised Stage 1 labels.  S4 reads a
strict whitelist from the frozen route-conditioned product, builds only model
inputs, and executes a target-free forward pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from stage2.v5.shards import (
    CATEGORY_NAMES,
    PROFILE_PACE_COLUMNS,
    RECENT_PACE_COLUMNS,
    _categorical_values,
    _encode,
    _gather,
)
from stage2.v5_2.contracts import Stage2V52ContractError, require_columns
from stage2.v5_2.structure_features import (
    build_static_structure_features,
    validate_feature_alignment,
)
from stage2.v5_2.support_transfer import lookup_train_support
from stage2.v5_2.temporal_adapter import TEMPORAL_FEATURE_NAMES
from stage2.v5_2.transfer_data import (
    _gather_aligned,
    _payload_source_positions,
    _temporal_source,
    build_temporal_features,
)
from stage3.odd_tod.capability_envelope import M3_SHA256, _m3_model
from stage3.odd_tod.network_foundation import atomic_json, sha256_file


TEST31_DATE = "20161031"
TEST31_ROLE = "legacy"
EXPECTED_CHECKPOINT_SHA256 = (
    "965fc491cd77256f7889961d89932ec6be709bab04adcca358ac1b49f47c2cde"
)
ROUTE_REL = Path(
    "stage2/output_v4/route_conditioned_dataset/revealed_route_proxy/"
    "day=20161031.parquet"
)
CHECKPOINT_REL = Path("stage2/output_v5_2/development/M3/epoch_004.pt")
MODEL_MANIFEST_REL = Path("stage2/output_v5_2/development/M3/model_manifest.json")
FEATURE_ARTIFACT_REL = Path(
    "stage2/output_v5/protocols/development/tensor_shards/feature_artifacts.json"
)
STATIC_ARTIFACT_REL = Path("stage2/output_v5_2/development/artifacts/static.json")
SUPPORT_ARTIFACT_REL = Path("stage2/output_v5_2/development/artifacts/support.json")
OUTPUT_REL = Path("stage3/output/odd_tod/s4/test31_m3_predictions.parquet")
MANIFEST_REL = Path("stage3/output/odd_tod/s4/test31_m3_predictions.json")
EXPECTED_ORDER_COUNT = 30_000

EXPECTED_ARTIFACT_SHA256 = {
    "feature": "64f9e55f897862c778a6680e60cc7438da0c3c9c226bf04d9b7ac8fd5568e84e",
    "static": "60f535316f984f3c6f8137cc7d31896a564034b3511a1d717baa94706cc38ac8",
    "support": "4bc907af5990c93f23730978ab4f2c049a5ed2ff7fb87ca9c5157bd6b33c2d98",
}

FORBIDDEN_SOURCE_COLUMNS = frozenset(
    {
        "crawl_time_share",
        "stop_time_share",
        "speed_cv_bounded",
        "acceleration_rms_bounded",
        "lcs_raw",
        "lcs_pct",
        "lcs_tail_event",
        "rts_raw",
        "rts_pct",
        "rts_tail_event",
        "pace_sec_per_m",
        "observed_travel_time_s",
        "observed_distance_m",
        "measurement_source",
        "travel_time_target_valid",
        "travel_time_direct_valid",
        "travel_time_interpolated_valid",
        "travel_time_source_class",
        "pace_target_valid",
        "crawl_target_valid",
        "stop_target_valid",
        "speed_cv_target_valid",
        "acceleration_rms_target_valid",
        "lcs_target_valid",
        "rts_target_valid",
    }
)

PREDICTION_COLUMNS = (
    "date",
    "order_id",
    "route_sequence",
    "traversal_id",
    "split",
    "support_group_code",
    "allocated_distance_m",
    "pred_crawl",
    "pred_stop",
    "pred_speed_cv",
    "pred_acceleration_rms",
    "pred_rts",
    "pred_pace_p50",
    "travel_time_p50_s",
)


class Stage3S4InferenceError(RuntimeError):
    """Raised when the Test31 prediction-only contract is violated."""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Stage3S4InferenceError(f"expected JSON object: {path}")
    return payload


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_columns(feature_artifact: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the only physical Test31 columns S4 is allowed to read."""

    identity_temporal = (
        "split",
        "date",
        "order_id",
        "route_sequence",
        "traversal_id",
        "observed_directed_edge_uid",
        "canonical_highway",
        "road_class",
        "observed_direction",
        "bridge",
        "tunnel",
        "synthetic_reverse_edge",
        "osm_direction_disagreement",
        "route_position_ratio",
        "route_token_count",
        "estimated_time_bin",
        "decision_time",
        "forecast_horizon_s",
        "feature_age_s",
        "history_count",
        "dynamic_available_mask",
        "feature_time_check",
        "route_part_length_m",
    )
    result = tuple(
        dict.fromkeys(
            (
                *identity_temporal,
                *tuple(str(value) for value in feature_artifact["numeric_features"]),
                *RECENT_PACE_COLUMNS,
                *PROFILE_PACE_COLUMNS,
            )
        )
    )
    forbidden = sorted(set(result) & FORBIDDEN_SOURCE_COLUMNS)
    if forbidden:
        raise Stage3S4InferenceError(
            f"frozen feature artifact unexpectedly requests realised targets: {forbidden}"
        )
    return result


def _vectorized_inference_payload(
    frame: pd.DataFrame,
    artifacts: Mapping[str, Any],
    *,
    max_seq_len: int,
    overlap: int,
) -> dict[str, np.ndarray]:
    """Target-free equivalent of the frozen Stage 2 route chunk encoder.

    Chunking, category encoding, normalization and gather operations delegate
    to the exact Stage 2 helpers.  No target or target-mask array is created.
    """

    if not 0 <= overlap < max_seq_len:
        raise Stage3S4InferenceError("overlap must be in [0, max_seq_len)")
    working = frame.sort_values(["order_id", "route_sequence"], kind="stable").reset_index(drop=True)
    order = working["order_id"].astype(str).to_numpy()
    sequence = pd.to_numeric(working["route_sequence"], errors="raise").to_numpy(np.int64)
    group_start = np.concatenate((np.array([True]), order[1:] != order[:-1]))
    starts = np.flatnonzero(group_start)
    ends = np.concatenate((starts[1:], np.array([len(working)])))
    lengths = ends - starts
    contiguous = group_start | (
        sequence == np.concatenate((np.array([0], dtype=np.int64), sequence[:-1] + 1))
    )
    if not contiguous.all() or np.any(sequence[starts] != 0):
        raise Stage3S4InferenceError("Test31 route_sequence is not zero-based and contiguous")
    stride = max_seq_len - overlap
    chunk_count = np.maximum(
        1, np.ceil(np.maximum(lengths - max_seq_len, 0) / stride).astype(np.int64) + 1
    )
    group_code = np.repeat(np.arange(len(starts), dtype=np.int64), chunk_count)
    group_offsets = np.repeat(np.cumsum(chunk_count) - chunk_count, chunk_count)
    local_chunk = np.arange(len(group_code), dtype=np.int64) - group_offsets
    local_start = np.minimum(
        local_chunk * stride, np.maximum(lengths[group_code] - max_seq_len, 0)
    )
    token_index = starts[group_code, None] + local_start[:, None] + np.arange(max_seq_len)
    valid = token_index < ends[group_code, None]

    numeric_names = tuple(str(value) for value in artifacts["numeric_features"])
    numeric_raw = (
        working.loc[:, numeric_names]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32, na_value=np.nan)
    )
    numeric_missing = ~np.isfinite(numeric_raw)
    mean = np.asarray(artifacts["numeric_mean"], dtype=np.float32)
    std = np.asarray(artifacts["numeric_std"], dtype=np.float32)
    numeric = (np.where(numeric_missing, mean, numeric_raw) - mean) / std
    categories = _categorical_values(working)
    categorical = np.column_stack(
        [_encode(categories[name], artifacts["vocabularies"][name]) for name in CATEGORY_NAMES]
    )
    recent = (
        working.loc[:, RECENT_PACE_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32, na_value=np.nan)
    )
    profile = (
        working.loc[:, PROFILE_PACE_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32, na_value=np.nan)
    )
    horizon = pd.to_numeric(working["forecast_horizon_s"], errors="coerce").fillna(0).to_numpy(np.float32)
    age = pd.to_numeric(working["feature_age_s"], errors="coerce").fillna(0).to_numpy(np.float32)
    history_support = (
        pd.to_numeric(working["observed_sec_per_m_profile_count"], errors="coerce")
        .fillna(0)
        .to_numpy(np.float32)
    )
    distance = pd.to_numeric(working["allocated_distance_m"], errors="coerce").fillna(0).to_numpy(np.float32)
    traversal = pd.to_numeric(working["traversal_id"], errors="raise").to_numpy(np.int64)
    return {
        "numeric": _gather(numeric, token_index, valid, 0.0).astype(np.float32),
        "numeric_missing": _gather(numeric_missing, token_index, valid, True).astype(bool),
        "categorical": _gather(categorical, token_index, valid, 0).astype(np.int64),
        "recent_history": _gather(np.nan_to_num(recent, nan=0), token_index, valid, 0).astype(np.float32),
        "profile_history": _gather(np.nan_to_num(profile, nan=0), token_index, valid, 0).astype(np.float32),
        "forecast_horizon_s": _gather(horizon, token_index, valid, 0).astype(np.float32),
        "history_age_s": _gather(age, token_index, valid, 0).astype(np.float32),
        "history_support": _gather(history_support, token_index, valid, 0).astype(np.float32),
        "allocated_distance_m": _gather(distance, token_index, valid, 0).astype(np.float32),
        "route_sequence": _gather(sequence, token_index, valid, -1).astype(np.int64),
        "traversal_id": _gather(traversal, token_index, valid, -1).astype(np.int64),
        "pad_mask": ~valid,
        "order_id": order[starts[group_code]].astype(str),
        "chunk_id": local_chunk.astype(np.int32),
    }


def _inference_payload(
    frame: pd.DataFrame,
    feature_artifact: Mapping[str, Any],
    static_artifact: Mapping[str, Any],
    support_artifact: Mapping[str, Any],
    *,
    max_seq_len: int,
    overlap: int,
) -> dict[str, np.ndarray]:
    payload = _vectorized_inference_payload(
        frame, feature_artifact, max_seq_len=max_seq_len, overlap=overlap
    )
    payload["split"] = np.full(len(payload["order_id"]), TEST31_ROLE, dtype="<U6")
    payload["date"] = np.full(len(payload["order_id"]), TEST31_DATE, dtype="<U8")
    static, _, static_row_id = build_static_structure_features(frame, static_artifact)
    validate_feature_alignment(frame["row_id"].to_numpy(), static_row_id)
    support, support_group = lookup_train_support(
        frame["observed_directed_edge_uid"], support_artifact
    )
    group_code = pd.Series(support_group).map(
        {"unseen": 0, "low": 1, "medium": 2, "high": 3}
    ).to_numpy(np.int8)
    temporal = build_temporal_features(frame)
    decision, cutoff, age, fallback = _temporal_source(frame)
    positions = _payload_source_positions(payload, frame)
    row_id = frame["row_id"].to_numpy(np.int64)
    payload.update(
        {
            "static_edge_features": _gather_aligned(static, positions, 0).astype(np.float32),
            "edge_train_support": _gather_aligned(support, positions, 0).astype(np.float32),
            "support_group_code": _gather_aligned(group_code, positions, -1).astype(np.int8),
            "temporal_features": _gather_aligned(temporal, positions, 0).astype(np.float32),
            "decision_time": _gather_aligned(decision, positions, np.nan).astype(np.float64),
            "feature_cutoff_time": _gather_aligned(cutoff, positions, np.nan).astype(np.float64),
            "feature_age_s": _gather_aligned(age, positions, np.nan).astype(np.float64),
            "no_history_temporal_fallback": _gather_aligned(fallback, positions, False).astype(bool),
            "row_id": _gather_aligned(row_id, positions, -1).astype(np.int64),
        }
    )
    forbidden_arrays = {
        "targets",
        "target_masks",
        "tail_targets",
        "tail_masks",
        "availability_targets",
        "supervision_weight",
    }
    if forbidden_arrays & set(payload):
        raise Stage3S4InferenceError("target-bearing tensor entered the S4 inference payload")
    return payload


def _forward_prediction(model: Any, data: Mapping[str, np.ndarray], index: np.ndarray, device: Any) -> Mapping[str, Any]:
    import torch

    def tensor(name: str, dtype: Any) -> Any:
        return torch.as_tensor(data[name][index], dtype=dtype, device=device)

    temporal_array = tensor("temporal_features", torch.float32)
    temporal = {
        name: temporal_array[..., offset]
        for offset, name in enumerate(TEMPORAL_FEATURE_NAMES)
    }
    return model(
        tensor("numeric", torch.float32),
        tensor("numeric_missing", torch.bool),
        tensor("categorical", torch.long),
        tensor("route_sequence", torch.long).clamp_min(0),
        tensor("pad_mask", torch.bool),
        static_edge_features=tensor("static_edge_features", torch.float32),
        edge_train_support=tensor("edge_train_support", torch.float32),
        temporal_features=temporal,
        recent_history=tensor("recent_history", torch.float32),
        profile_history=tensor("profile_history", torch.float32),
        forecast_horizon_s=tensor("forecast_horizon_s", torch.float32),
        history_age_s=tensor("history_age_s", torch.float32),
        history_support=tensor("history_support", torch.float32),
    )


def _predict_payload(
    model: Any,
    payload: Mapping[str, np.ndarray],
    *,
    batch_size: int,
    device: Any,
) -> tuple[pd.DataFrame, bool]:
    import torch

    output_names = {
        "pred_crawl": "crawl_share",
        "pred_stop": "stop_share",
        "pred_speed_cv": "speed_cv",
        "pred_acceleration_rms": "acceleration_rms",
        "pred_rts": "rts_raw",
        "pred_pace_p50": "pace_pred_p50",
    }
    pieces: list[pd.DataFrame] = []
    finite = True
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(payload["numeric"]), batch_size):
            index = np.arange(start, min(start + batch_size, len(payload["numeric"])))
            output = _forward_prediction(model, payload, index, device)
            valid = ~payload["pad_mask"][index]
            finite &= all(
                bool(torch.isfinite(value).all())
                for value in output.values()
                if isinstance(value, torch.Tensor) and value.is_floating_point()
            )
            order = payload["order_id"][index].astype(str)
            frame = pd.DataFrame(
                {
                    "date": np.broadcast_to(payload["date"][index, None], valid.shape)[valid],
                    "order_id": np.broadcast_to(order[:, None], valid.shape)[valid],
                    "route_sequence": payload["route_sequence"][index][valid].astype(np.int64),
                    "traversal_id": payload["traversal_id"][index][valid].astype(np.int64),
                    "split": np.broadcast_to(payload["split"][index, None], valid.shape)[valid],
                    "support_group_code": payload["support_group_code"][index][valid].astype(np.int8),
                    "allocated_distance_m": payload["allocated_distance_m"][index][valid].astype(np.float32),
                }
            )
            for column, key in output_names.items():
                frame[column] = output[key].float().cpu().numpy()[valid]
            pieces.append(frame)
    if not pieces:
        raise Stage3S4InferenceError("frozen M3 produced no Test31 prediction tokens")
    copies = pd.concat(pieces, ignore_index=True)
    key = ["date", "order_id", "route_sequence", "traversal_id"]
    for column in ("split", "support_group_code", "allocated_distance_m"):
        if (copies.groupby(key, sort=False, observed=True)[column].nunique() > 1).any():
            raise Stage3S4InferenceError(f"overlap copies disagree on {column}")
    aggregations: dict[str, str] = {
        "split": "first",
        "support_group_code": "first",
        "allocated_distance_m": "first",
        **{column: "mean" for column in output_names},
    }
    unique = copies.groupby(key, sort=False, observed=True, as_index=False).agg(aggregations)
    unique["travel_time_p50_s"] = unique["pred_pace_p50"] * unique["allocated_distance_m"]
    return unique.loc[:, PREDICTION_COLUMNS], finite


def _prepare_source(frame: pd.DataFrame, *, row_offset: int) -> pd.DataFrame:
    if frame.empty:
        raise Stage3S4InferenceError("empty Test31 route row group")
    work = frame.copy()
    physical_dates = work["date"].astype(str).str.replace("-", "", regex=False)
    if not physical_dates.eq(TEST31_DATE).all():
        raise Stage3S4InferenceError("source row group contains a non-Test31 date")
    work["split"] = TEST31_ROLE
    work["date"] = TEST31_DATE
    work["order_id"] = work["order_id"].astype(str)
    work["allocated_distance_m"] = pd.to_numeric(
        work["route_part_length_m"], errors="coerce"
    )
    if work["allocated_distance_m"].isna().any() or (work["allocated_distance_m"] < 0).any():
        raise Stage3S4InferenceError("Test31 route has invalid route_part_length_m")
    work["row_id"] = np.arange(row_offset, row_offset + len(work), dtype=np.int64)
    require_columns(
        work.columns,
        ("split", "date", "order_id", "traversal_id", "route_sequence", "row_id"),
        product="S4 Test31 prediction source",
    )
    if work.duplicated(["order_id", "traversal_id"]).any():
        raise Stage3S4InferenceError("Test31 row group duplicates traversal identity")
    return work


def _bound_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if M3_SHA256 != EXPECTED_CHECKPOINT_SHA256:
        raise Stage3S4InferenceError("S3 and S4 frozen M3 constants disagree")
    paths = {
        "checkpoint": root / CHECKPOINT_REL,
        "model_manifest": root / MODEL_MANIFEST_REL,
        "feature": root / FEATURE_ARTIFACT_REL,
        "static": root / STATIC_ARTIFACT_REL,
        "support": root / SUPPORT_ARTIFACT_REL,
        "route": root / ROUTE_REL,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise Stage3S4InferenceError(f"missing S4 inference inputs: {missing}")
    if sha256_file(paths["checkpoint"]) != EXPECTED_CHECKPOINT_SHA256:
        raise Stage3S4InferenceError("frozen M3 checkpoint SHA mismatch")
    for name in ("feature", "static", "support"):
        if sha256_file(paths[name]) != EXPECTED_ARTIFACT_SHA256[name]:
            raise Stage3S4InferenceError(f"frozen {name} artifact SHA mismatch")
    model_manifest = _read_json(paths["model_manifest"])
    if (
        model_manifest.get("model_id") != "M3"
        or model_manifest.get("protocol_id") != "development"
        or model_manifest.get("selected_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256
        or model_manifest.get("source", {}).get("feature_artifact_path")
        != FEATURE_ARTIFACT_REL.as_posix()
        or model_manifest.get("feature_artifact_sha256", EXPECTED_ARTIFACT_SHA256["feature"])
        != EXPECTED_ARTIFACT_SHA256["feature"]
        or model_manifest.get("static_artifact_sha256") != EXPECTED_ARTIFACT_SHA256["static"]
        or model_manifest.get("support_artifact_sha256") != EXPECTED_ARTIFACT_SHA256["support"]
    ):
        raise Stage3S4InferenceError("M3 manifest does not bind the frozen development artifacts")
    feature = _read_json(paths["feature"])
    static = _read_json(paths["static"])
    support = _read_json(paths["support"])
    bindings = {
        name: {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }
    return bindings, feature, static, support


def build_test31_predictions(
    root: str | Path,
    batch_size: int = 128,
    *,
    max_seq_len: int = 128,
    overlap: int = 32,
    output_path: str | Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run prediction-only M3 inference on the exact frozen Test31 route."""

    if batch_size <= 0:
        raise Stage3S4InferenceError("batch_size must be positive")
    root_path = Path(root).resolve()
    destination = (
        Path(output_path).resolve() if output_path is not None else root_path / OUTPUT_REL
    )
    manifest_path = (
        destination.with_name(f"{destination.stem}_manifest.json")
        if output_path is not None
        else root_path / MANIFEST_REL
    )
    bindings, feature, static, support = _bound_inputs(root_path)
    route_path = root_path / ROUTE_REL
    source_columns = _source_columns(feature)
    physical_schema = set(pq.read_schema(route_path).names)
    missing = sorted(set(source_columns) - physical_schema)
    if missing:
        raise Stage3S4InferenceError(f"Test31 route lacks frozen inference fields: {missing}")
    if resume and destination.is_file() and manifest_path.is_file():
        previous = _read_json(manifest_path)
        previous_bindings = previous.get("input_bindings", {})
        bindings_match = all(
            previous_bindings.get(name, {}).get("path") == descriptor["path"]
            and previous_bindings.get(name, {}).get("sha256") == descriptor["sha256"]
            for name, descriptor in bindings.items()
        )
        if (
            previous.get("status") == "PASS"
            and previous.get("date") == TEST31_DATE
            and previous.get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256
            and previous.get("route_sha256") == bindings["route"]["sha256"]
            and previous.get("prediction_sha256") == sha256_file(destination)
            and previous.get("source_column_whitelist_sha256")
            == _canonical_hash({"columns": source_columns})
            and previous.get("artifact_sha256")
            == _canonical_hash({
                key: value for key, value in previous.items()
                if key != "artifact_sha256"
            })
            and bindings_match
            and previous.get("row_count") == pq.ParquetFile(destination).metadata.num_rows
            and previous.get("order_count") == EXPECTED_ORDER_COUNT
            and previous.get("all_outputs_finite") is True
            and previous.get("decision_time_only") is True
            and previous.get("predicted_progression_only") is True
            and previous.get("realized_future_time_used") is False
            and previous.get("prediction_only_forward") is True
            and previous.get("stage1_target_merge_performed") is False
            and previous.get("target_arrays_constructed") is False
            and previous.get("loss_or_metric_path_called") is False
        ):
            existing_schema = pq.read_schema(destination).names
            if existing_schema != list(PREDICTION_COLUMNS):
                raise Stage3S4InferenceError(
                    "cached prediction schema differs from strict S4 prediction-only schema"
                )
            persisted_target_names = sorted(
                column
                for column in existing_schema
                if column in FORBIDDEN_SOURCE_COLUMNS
                or column.startswith("target_")
                or column.endswith("_target_valid")
            )
            if persisted_target_names:
                raise Stage3S4InferenceError(
                    "cached Test31 predictions persist realised targets: "
                    f"{persisted_target_names}"
                )
            if (
                previous.get("realized_target_columns_persisted") is not False
                or previous.get("realized_target_column_names_persisted") != []
            ):
                migrated = dict(previous)
                migrated["realized_target_columns_persisted"] = False
                migrated["realized_target_column_names_persisted"] = []
                migrated["manifest_schema_migrated"] = True
                migrated.pop("artifact_sha256", None)
                migrated["artifact_sha256"] = _canonical_hash(migrated)
                atomic_json(manifest_path, migrated)
                previous = migrated
            # cache_reused is an invocation result, not a persisted scientific
            # fact; keeping the manifest byte-stable preserves its self-hash.
            return {**previous, "cache_reused": True}
        raise Stage3S4InferenceError(
            "cached Test31 prediction exists but its frozen provenance no longer matches; "
            "refusing implicit re-inference in resume mode"
        )

    model, training_manifest, model_binding = _m3_model(root_path)
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    started = time.perf_counter()
    source_rows = prediction_rows = order_count = 0
    all_outputs_finite = True
    previous_last_order: str | None = None
    route_file = pq.ParquetFile(route_path)
    try:
        for row_group in range(route_file.num_row_groups):
            source = route_file.read_row_group(row_group, columns=list(source_columns)).to_pandas()
            source = _prepare_source(source, row_offset=source_rows)
            first_order = str(source["order_id"].iloc[0])
            if previous_last_order == first_order:
                raise Stage3S4InferenceError(
                    "an order crosses Parquet row groups; row-group inference would split its context"
                )
            previous_last_order = str(source["order_id"].iloc[-1])
            payload = _inference_payload(
                source,
                feature,
                static,
                support,
                max_seq_len=max_seq_len,
                overlap=overlap,
            )
            predictions, finite = _predict_payload(
                model, payload, batch_size=batch_size, device=device
            )
            source_identity = source.loc[
                :, ["date", "order_id", "route_sequence", "traversal_id"]
            ]
            predictions = source_identity.merge(
                predictions,
                on=["date", "order_id", "route_sequence", "traversal_id"],
                how="left",
                validate="one_to_one",
            )
            if predictions.loc[:, PREDICTION_COLUMNS[4:]].isna().any().any():
                raise Stage3S4InferenceError("M3 predictions do not cover every Test31 route token")
            predictions = predictions.loc[:, PREDICTION_COLUMNS]
            table = pa.Table.from_pandas(predictions, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            source_rows += len(source)
            prediction_rows += len(predictions)
            order_count += int(source["order_id"].nunique())
            all_outputs_finite &= finite
            del source, payload, predictions, table
        if writer is None:
            raise Stage3S4InferenceError("Test31 inference wrote no row groups")
        writer.close()
        writer = None
        if source_rows != route_file.metadata.num_rows or prediction_rows != source_rows:
            raise Stage3S4InferenceError(
                f"Test31 prediction reconciliation failed: source={source_rows}, output={prediction_rows}"
            )
        if order_count != EXPECTED_ORDER_COUNT:
            raise Stage3S4InferenceError(
                f"Test31 prediction order count differs from frozen 30000: {order_count}"
            )
        if not all_outputs_finite:
            raise Stage3S4InferenceError("frozen M3 emitted a non-finite Test31 prediction")
        os.replace(temporary, destination)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)

    output_schema = pq.read_schema(destination).names
    if output_schema != list(PREDICTION_COLUMNS):
        raise Stage3S4InferenceError("persisted prediction schema differs from strict S4 schema")
    manifest = {
        "schema_version": "stage3_s4_test31_m3_prediction.1",
        "status": "PASS",
        "date": TEST31_DATE,
        "role": TEST31_ROLE,
        "model_id": "M3",
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "route_path": ROUTE_REL.as_posix(),
        "route_sha256": bindings["route"]["sha256"],
        "input_bindings": bindings,
        "source_column_whitelist": list(source_columns),
        "source_column_whitelist_sha256": _canonical_hash({"columns": source_columns}),
        "physical_source_columns_read_count": len(source_columns),
        "realized_target_columns_read": [],
        "realized_target_columns_persisted": False,
        "realized_target_column_names_persisted": [],
        "stage1_target_merge_performed": False,
        "target_arrays_constructed": False,
        "loss_or_metric_path_called": False,
        "prediction_only_forward": True,
        "decision_time_only": True,
        "predicted_progression_only": True,
        "realized_future_time_used": False,
        "feature_artifact_fit_scope": feature.get("fit_scope"),
        "feature_artifact_fit_dates": feature.get("fit_dates"),
        "static_artifact_fit_scope": static.get("fit_scope"),
        "support_artifact_fit_scope": support.get("fit_scope"),
        "model_source_binding": model_binding,
        "model_training_manifest_sha256": bindings["model_manifest"]["sha256"],
        "model_training_protocol": training_manifest.get("protocol_id"),
        "max_seq_len": max_seq_len,
        "overlap": overlap,
        "overlap_merge": "mean_by_date_order_id_route_sequence_traversal_id",
        "parquet_row_group_count": route_file.num_row_groups,
        "source_row_count": source_rows,
        "prediction_row_count": prediction_rows,
        "row_count": prediction_rows,
        "order_count": order_count,
        "all_outputs_finite": all_outputs_finite,
        "output_path": destination.relative_to(root_path).as_posix(),
        "prediction_sha256": sha256_file(destination),
        "prediction_schema": output_schema,
        "runtime_s": time.perf_counter() - started,
        "device": str(device),
        "batch_size": batch_size,
        "test31_route_search_performed": False,
        "test31_fallback_performed": False,
    }
    manifest["artifact_sha256"] = _canonical_hash(manifest)
    atomic_json(manifest_path, manifest)
    return manifest


__all__ = [
    "EXPECTED_CHECKPOINT_SHA256",
    "FORBIDDEN_SOURCE_COLUMNS",
    "PREDICTION_COLUMNS",
    "Stage3S4InferenceError",
    "TEST31_DATE",
    "build_test31_predictions",
]
