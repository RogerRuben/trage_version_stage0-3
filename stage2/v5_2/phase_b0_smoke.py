"""Bounded real-data Phase B0 smoke; never authorizes training by itself."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from stage2.v5.availability import service_time_target_arrays

from .contracts import Stage2V52ContractError
from .protocols import get_protocol, protocol_role_dates
from .support_transfer import validate_embedded_hash
from .training import initialized_transfer_model
from .transfer_data import build_transfer_chunk_payload
from .verification import preflight, sha256_file


def _json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Stage2V52ContractError(f"expected JSON object: {path}")
    return payload


class _RssSampler:
    def __init__(self) -> None:
        import psutil

        self._process = psutil.Process()
        self._stop = threading.Event()
        self.peak = int(self._process.memory_info().rss)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.01):
            self.peak = max(self.peak, int(self._process.memory_info().rss))

    def __enter__(self) -> "_RssSampler":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.peak = max(self.peak, int(self._process.memory_info().rss))


def _tensor_inputs(payload: Mapping[str, np.ndarray]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    import torch
    from .temporal_adapter import TEMPORAL_FEATURE_NAMES

    def tensor(name: str, dtype: Any) -> Any:
        return torch.as_tensor(payload[name], dtype=dtype)

    temporal_array = tensor("temporal_features", torch.float32)
    positional = (
        tensor("numeric", torch.float32), tensor("numeric_missing", torch.bool),
        tensor("categorical", torch.long), tensor("route_sequence", torch.long).clamp_min(0),
        tensor("pad_mask", torch.bool),
    )
    keyword = {
        "static_edge_features": tensor("static_edge_features", torch.float32),
        "edge_train_support": tensor("edge_train_support", torch.float32),
        "temporal_features": {
            name: temporal_array[..., index] for index, name in enumerate(TEMPORAL_FEATURE_NAMES)
        },
        "recent_history": tensor("recent_history", torch.float32),
        "profile_history": tensor("profile_history", torch.float32),
        "forecast_horizon_s": tensor("forecast_horizon_s", torch.float32),
        "history_age_s": tensor("history_age_s", torch.float32),
        "history_support": tensor("history_support", torch.float32),
    }
    return positional, keyword


def _s0_equivalence_and_finite(model: Any, payload: Mapping[str, np.ndarray]) -> dict[str, Any]:
    import torch

    positional, keyword = _tensor_inputs(payload)
    backbone_keyword = {
        name: keyword[name] for name in (
            "recent_history", "profile_history", "forecast_horizon_s", "history_age_s", "history_support"
        )
    }
    model.eval()
    with torch.inference_mode():
        wrapped = model(*positional, **keyword)
        frozen = model.backbone(*positional, **backbone_keyword)
    common = sorted(set(wrapped) & set(frozen))
    mismatches = [
        name for name in common
        if wrapped[name].shape != frozen[name].shape or not torch.equal(wrapped[name], frozen[name])
    ]
    valid = ~np.asarray(payload["pad_mask"], dtype=bool)
    non_finite: dict[str, int] = {}
    for name, value in wrapped.items():
        array = value.detach().cpu().numpy()
        if array.ndim >= 2 and array.shape[:2] == valid.shape:
            count = int((~np.isfinite(array[valid])).sum())
        else:
            count = int((~np.isfinite(array)).sum())
        if count:
            non_finite[name] = count
    if mismatches or non_finite or set(wrapped) != set(frozen):
        raise Stage2V52ContractError(
            f"S0 wrapper differs from frozen v5.1 or emitted non-finite values: "
            f"mismatch={mismatches}, finite={non_finite}"
        )
    return {
        "compared_output_names": common, "exact_tensor_mismatch_count": 0,
        "non_finite_output_count": 0,
    }


def _payload_audit(frame: pd.DataFrame, payload: Mapping[str, np.ndarray]) -> dict[str, Any]:
    valid = ~np.asarray(payload["pad_mask"], dtype=bool)
    ages = np.asarray(payload["feature_age_s"], dtype=float)[valid]
    leakage = int((~np.isfinite(ages) | (ages <= 0)).sum())
    row_ids = np.asarray(payload["row_id"], dtype=np.int64)[valid]
    if leakage or (row_ids < 0).any():
        raise Stage2V52ContractError("Phase B0 smoke found temporal leakage or invalid row alignment")
    unique_rows = np.unique(row_ids)
    expected_rows = np.sort(pd.to_numeric(frame["row_id"], errors="raise").to_numpy(np.int64))
    if not np.array_equal(unique_rows, expected_rows):
        raise Stage2V52ContractError("overlap tokens do not reconcile to each unique source traversal row")
    source_identity = frame[["date", "order_id", "traversal_id"]]
    if source_identity.duplicated().any():
        raise Stage2V52ContractError("smoke bucket is not unique by physical traversal")
    return {
        "chunk_count": int(len(payload["order_id"])),
        "token_count": int(valid.sum()),
        "unique_token_row_count": int(len(unique_rows)),
        "duplicate_overlap_token_count": int(len(row_ids) - len(unique_rows)),
        "unique_traversal_count": int(len(source_identity)),
        "temporal_leakage_count": leakage,
        "minimum_feature_age_s": float(ages.min()),
    }


def _load_explicit_bucket_frame(
    *, role: str, route_feature_path: Path, traversal_path: Path, label_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    for path in (route_feature_path, traversal_path, label_path):
        if not path.is_file():
            raise Stage2V52ContractError(f"Phase B0 input does not exist: {path}")
    if "bucket=" not in traversal_path.as_posix() or "bucket=" not in label_path.as_posix():
        raise Stage2V52ContractError("Phase B0 traversal and label inputs must be explicit bucket files")
    traversal_columns = [
        "order_id", "traversal_id", "measurement_source", "observed_travel_time_s",
        "observed_distance_m", "allocated_distance_m",
    ]
    traversal = pd.read_parquet(traversal_path, columns=traversal_columns)
    labels = pd.read_parquet(
        label_path,
        columns=["order_id", "traversal_id", "observed_sec_per_m", "rts_measurement_available"],
    )
    identity = ["order_id", "traversal_id"]
    if traversal.empty or labels.empty or traversal.duplicated(identity).any() or labels.duplicated(identity).any():
        raise Stage2V52ContractError(f"Phase B0 {role} traversal/label identity is empty or duplicated")
    reconciliation = traversal[[*identity, "measurement_source"]].merge(
        labels[identity], on=identity, how="outer", indicator=True,
    )
    orphan_labels = int(reconciliation["_merge"].eq("right_only").sum())
    missing_direct = int(
        (
            reconciliation["_merge"].eq("left_only")
            & reconciliation["measurement_source"].eq("direct_observed")
        ).sum()
    )
    if orphan_labels or missing_direct:
        raise Stage2V52ContractError(f"Phase B0 {role} traversal/label reconciliation failed")
    non_direct_without_label = int(reconciliation["_merge"].eq("left_only").sum())
    target = service_time_target_arrays(
        traversal["measurement_source"].to_numpy(),
        traversal["observed_travel_time_s"].to_numpy(),
        traversal["observed_distance_m"].to_numpy(),
    )
    targets = traversal.merge(labels, on=identity, how="left", validate="one_to_one")
    pace = pd.to_numeric(targets["observed_sec_per_m"], errors="coerce").to_numpy(float)
    physical = targets["rts_measurement_available"].fillna(False).to_numpy(bool)
    targets["pace_sec_per_m"] = pace
    targets["pace_target_valid"] = target["travel_time_direct_valid"] & physical & np.isfinite(pace) & (pace > 0)
    targets["travel_time_target_valid"] = target["travel_time_target_valid"]
    targets["travel_time_direct_valid"] = target["travel_time_direct_valid"]
    targets["travel_time_interpolated_valid"] = target["travel_time_interpolated_valid"]
    targets["travel_time_source_class"] = target["travel_time_source_class"]
    order_ids = pa.array(sorted(traversal["order_id"].astype(str).unique()))
    table = ds.dataset(route_feature_path, format="parquet").to_table(
        filter=ds.field("order_id").isin(order_ids)
    )
    route = table.to_pandas()
    if route.empty:
        raise Stage2V52ContractError(f"Phase B0 {role} route feature filter returned no rows")
    frame = route.merge(targets, on=identity, how="inner", validate="one_to_one", suffixes=("", "_stage1"))
    if frame.empty:
        raise Stage2V52ContractError(f"Phase B0 {role} bucket has no route/traversal overlap")
    date_values = frame["date"].astype(str).unique() if "date" in frame else []
    if len(date_values) != 1:
        raise Stage2V52ContractError(f"Phase B0 {role} route bucket has mixed or missing date")
    frame["split"] = role
    frame["date"] = str(date_values[0])
    frame["row_id"] = np.arange(len(frame), dtype=np.int64)
    return frame, {
        "route_feature_path": route_feature_path.as_posix(),
        "route_feature_sha256": sha256_file(route_feature_path),
        "traversal_path": traversal_path.as_posix(), "traversal_sha256": sha256_file(traversal_path),
        "label_path": label_path.as_posix(), "label_sha256": sha256_file(label_path),
        "labelled_direct_traversal_count": int(reconciliation["_merge"].eq("both").sum()),
        "non_direct_without_label_count": non_direct_without_label,
        "orphan_label_count": orphan_labels,
        "missing_direct_label_count": missing_direct,
        "traversal_label_reconciliation_mismatch_count": orphan_labels + missing_direct,
    }


def run_phase_b0_smoke(
    *, config_path: str | Path, protocol_id: str,
    source_checkpoint_path: str | Path, feature_artifact_path: str | Path,
    source_model_manifest_path: str | Path, source_config_path: str | Path,
    support_artifact_path: str | Path, static_artifact_path: str | Path,
    train_route_feature_path: str | Path, train_traversal_path: str | Path,
    train_label_path: str | Path, validation_route_feature_path: str | Path,
    validation_traversal_path: str | Path, validation_label_path: str | Path,
    max_seq_len: int, overlap: int, backbone_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Run real v5.1 binding and S0 inference on exactly two explicit joined buckets."""
    started = time.perf_counter()
    source_preflight = preflight(
        config_path=config_path, protocol_id=protocol_id,
        source_checkpoint_path=source_checkpoint_path, feature_artifact_path=feature_artifact_path,
        source_model_manifest_path=source_model_manifest_path, source_config_path=source_config_path,
    )
    feature, support, static = (
        _json(feature_artifact_path), _json(support_artifact_path), _json(static_artifact_path)
    )
    validate_embedded_hash(support, name="Phase B0 support")
    protocol = get_protocol(protocol_id)
    if tuple(feature.get("fit_dates", ())) != protocol.train_dates:
        raise Stage2V52ContractError("Phase B0 feature artifact is not fitted on protocol Train")
    if (
        support.get("protocol_id") != protocol_id
        or tuple(support.get("fit_dates_observed", ())) != protocol.train_dates
        or static.get("protocol_id") != protocol_id
        or tuple(static.get("fit_dates_observed", ())) != protocol.train_dates
    ):
        raise Stage2V52ContractError("Phase B0 support/static artifacts are not protocol aligned")
    role_inputs = {
        "train": (Path(train_route_feature_path), Path(train_traversal_path), Path(train_label_path)),
        "validation": (
            Path(validation_route_feature_path), Path(validation_traversal_path), Path(validation_label_path)
        ),
    }
    payloads: dict[str, dict[str, np.ndarray]] = {}
    schemas: dict[str, dict[str, Any]] = {}
    bucket_results: dict[str, Any] = {}
    with _RssSampler() as rss:
        for role, paths in role_inputs.items():
            frame, source_files = _load_explicit_bucket_frame(
                role=role, route_feature_path=paths[0], traversal_path=paths[1], label_path=paths[2],
            )
            if str(frame["date"].iloc[0]) not in protocol_role_dates(protocol_id)[role]:
                raise Stage2V52ContractError(f"Phase B0 {role} bucket date is outside the frozen protocol role")
            payload, schema = build_transfer_chunk_payload(
                frame, feature, static_artifact=static, support_artifact=support,
                max_seq_len=max_seq_len, overlap=overlap,
            )
            payloads[role], schemas[role] = payload, schema
            bucket_results[role] = {
                **source_files,
                "date": str(frame["date"].iloc[0]), "source_row_count": int(len(frame)),
                **_payload_audit(frame, payload),
            }
        if schemas["train"]["static_feature_names"] != schemas["validation"]["static_feature_names"]:
            raise Stage2V52ContractError("Train/Validation static tensor schemas differ")
        tau = float(support["positive_quantiles"]["p50"])
        model, source_binding = initialized_transfer_model(
            protocol_id=protocol_id, feature_artifact_path=feature_artifact_path,
            checkpoint_path=source_checkpoint_path,
            source_model_manifest_path=source_model_manifest_path,
            source_config_path=source_config_path,
            static_feature_count=int(schemas["train"]["static_feature_count"]),
            support_tau=tau, spatial_mode="identity", temporal_mode="none",
            backbone_kwargs=dict(backbone_kwargs),
        )
        equivalence = {
            role: _s0_equivalence_and_finite(model, payload) for role, payload in payloads.items()
        }
    return {
        "schema_version": "stage2_v5_2_phase_b0_smoke.1", "status": "PASS",
        "scope": "one_explicit_train_bucket_plus_one_explicit_validation_bucket",
        "full_protocol_scanned": False, "real_v5_1_preflight": source_preflight,
        "source_binding": source_binding,
        "feature_artifact_sha256": sha256_file(feature_artifact_path),
        "support_artifact_sha256": sha256_file(support_artifact_path),
        "static_artifact_sha256": sha256_file(static_artifact_path),
        "tensor_schema": schemas["train"], "buckets": bucket_results,
        "s0_exact_numerical_equivalence": equivalence,
        "runtime_s": float(time.perf_counter() - started),
        "peak_rss_mb": float(rss.peak / (1024 * 1024)),
        "authorizes_phase_b1": True,
    }
