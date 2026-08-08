"""Build aligned transfer tensors on top of frozen v5.1 route chunks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from stage2.v5.shards import vectorized_chunk_payload

from .contracts import Stage2V52ContractError, require_columns
from .structure_features import build_static_structure_features, validate_feature_alignment
from .support_transfer import lookup_train_support
from .temporal_adapter import TEMPORAL_FEATURE_NAMES


def build_temporal_features(frame: pd.DataFrame, *, timezone: str = "Asia/Shanghai") -> np.ndarray:
    """Frozen analytical transforms of decision-time-available fields only."""
    require_columns(frame.columns, ("decision_time", "forecast_horizon_s"), product="temporal features")
    decision = pd.to_numeric(frame["decision_time"], errors="coerce")
    horizon = pd.to_numeric(frame["forecast_horizon_s"], errors="coerce")
    if decision.isna().any() or horizon.isna().any() or (horizon < 0).any():
        raise Stage2V52ContractError("temporal feature inputs must be finite and horizon non-negative")
    local = pd.to_datetime(decision, unit="s", utc=True).dt.tz_convert(timezone)
    hour = local.dt.hour.to_numpy(float) + local.dt.minute.to_numpy(float) / 60.0
    angle = hour * (2.0 * np.pi / 24.0)
    return np.column_stack((
        np.sin(angle),
        np.cos(angle),
        local.dt.dayofweek.to_numpy(float) / 6.0,
        np.log1p(horizon.to_numpy(float)) / 12.0,
    )).astype(np.float32)


def _payload_source_positions(payload: Mapping[str, np.ndarray], frame: pd.DataFrame) -> np.ndarray:
    source = pd.MultiIndex.from_frame(frame[["order_id", "traversal_id"]].astype({"order_id": str}))
    if source.has_duplicates:
        raise Stage2V52ContractError("transfer source traversal identity is duplicated")
    valid = ~payload["pad_mask"]
    order = np.broadcast_to(payload["order_id"][:, None], valid.shape)[valid].astype(str)
    traversal = payload["traversal_id"][valid].astype(np.int64)
    query = pd.MultiIndex.from_arrays((order, traversal), names=("order_id", "traversal_id"))
    positions = source.get_indexer(query)
    if np.any(positions < 0):
        raise Stage2V52ContractError("chunk token cannot be aligned to source row")
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


def build_transfer_chunk_payload(
    frame: pd.DataFrame,
    v5_artifacts: dict[str, Any],
    *,
    static_artifact: Mapping[str, Any],
    support_artifact: Mapping[str, Any],
    max_seq_len: int,
    overlap: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Generate all v5.1 and v5.2 tensors from one consistently keyed frame."""
    require_columns(
        frame.columns,
        ("row_id", "split", "date", "order_id", "traversal_id", "route_sequence"),
        product="transfer shard source",
    )
    row_id = pd.to_numeric(frame["row_id"], errors="coerce")
    if row_id.isna().any() or row_id.duplicated().any() or (row_id % 1 != 0).any():
        raise Stage2V52ContractError("transfer row_id must be a unique finite integer")
    payload = vectorized_chunk_payload(
        frame, v5_artifacts, max_seq_len=max_seq_len, overlap=overlap
    )
    static, feature_names, static_row_id = build_static_structure_features(frame, static_artifact)
    validate_feature_alignment(frame["row_id"].to_numpy(), static_row_id)
    support, support_group = lookup_train_support(
        frame["observed_directed_edge_uid"], support_artifact
    )
    group_code = pd.Series(support_group).map(
        {"unseen": 0, "low": 1, "medium": 2, "high": 3}
    ).to_numpy(np.int8)
    temporal = build_temporal_features(frame)
    positions = _payload_source_positions(payload, frame)
    payload.update({
        "static_edge_features": _gather_aligned(static, positions, 0.0).astype(np.float32),
        "edge_train_support": _gather_aligned(support, positions, 0).astype(np.float32),
        "support_group_code": _gather_aligned(group_code, positions, -1).astype(np.int8),
        "temporal_features": _gather_aligned(temporal, positions, 0.0).astype(np.float32),
        "row_id": _gather_aligned(row_id.to_numpy(np.int64), positions, -1),
    })
    return payload, {
        "schema_version": "stage2_v5_2_transfer_tensor_schema.1",
        "static_feature_names": list(feature_names),
        "temporal_feature_names": list(TEMPORAL_FEATURE_NAMES),
        "row_alignment": "explicit_row_id_scattered_to_caller_order",
    }
