"""Protocol-bound v5.2 training, unique-traversal metrics, and M0 baseline."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .contracts import CORE_TRANSFER_TARGETS, Stage2V52ContractError
from .feature_binding import bind_v51_source_model, sha256_path
from .models.rc_mstnet_transfer import RCMSTNetTransfer
from .protocols import get_protocol, protocol_role_dates
from .temporal_adapter import TEMPORAL_FEATURE_NAMES
from .transfer_data import TRANSFER_MANIFEST_SCHEMA_VERSION


TRAINING_SCHEMA_VERSION = "stage2_v5_2_training.2"
M0_MATRIX_SCHEMA_VERSION = "stage2_v5_2_m0_matrix.2"
M0_TRAINING_SCHEMA_VERSION = "stage2_v5_2_m0_training.2"
CORE_METRIC_DEFINITION = "unique_traversal_mean_absolute_error"
PACE_METRIC_DEFINITION = "unique_traversal_pace_p50_mean_absolute_error"
LOSS_COMPONENT_KEYS = frozenset({
    "pace_distribution", "crawl", "stop_occurrence", "stop_positive", "speed_cv",
    "acceleration", "rts", "lcs_consistency", "lcs_tail", "rts_tail", "availability",
})


def validate_loss_weight_schema(component_weights: Mapping[str, float]) -> dict[str, float]:
    """Require an explicit frozen weight for every real v5 loss component."""
    observed = set(component_weights)
    missing = sorted(LOSS_COMPONENT_KEYS - observed)
    unknown = sorted(observed - LOSS_COMPONENT_KEYS)
    if missing or unknown:
        raise Stage2V52ContractError(
            f"loss-weight schema mismatch: missing={missing}, unknown={unknown}"
        )
    weights = {name: float(component_weights[name]) for name in LOSS_COMPONENT_KEYS}
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise Stage2V52ContractError("loss weights must be finite and non-negative")
    return weights


def validate_m5_m4_adoption(
    payload: Mapping[str, Any], *, protocol_id: str, m4_checkpoint_sha256: str,
) -> None:
    if (
        payload.get("schema_version") != "stage2_v5_2_rolling_spatial_adoption.1"
        or payload.get("status") != "PASS"
        or payload.get("verification_status") != "PASS"
        or payload.get("protocol_id") != "rolling_origin_fold_1_2_3"
        or payload.get("decision_scope") != "rolling_origin_three_fold_six_dates"
        or payload.get("adopt") is not True
        or payload.get("selected_m4_checkpoint_sha256_by_protocol", {}).get(protocol_id)
        != m4_checkpoint_sha256
    ):
        raise Stage2V52ContractError("M5 is blocked until formal rolling M4 adoption passes")


def _json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Stage2V52ContractError(f"expected JSON object: {path}")
    return payload


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_training_manifest(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _finite_metric(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def macro_normalized_core_mae(
    candidate_mae: Mapping[str, float | None],
    v5_1_mae: Mapping[str, float | None],
) -> float:
    if set(candidate_mae) != set(CORE_TRANSFER_TARGETS) or set(v5_1_mae) != set(CORE_TRANSFER_TARGETS):
        raise Stage2V52ContractError("checkpoint score requires exactly four core micro targets")
    ratios: list[float] = []
    for target in CORE_TRANSFER_TARGETS:
        candidate = candidate_mae[target]
        baseline = v5_1_mae[target]
        if not _finite_metric(candidate) or not _finite_metric(baseline) or float(baseline) <= 0:
            raise Stage2V52ContractError("checkpoint MAE is missing or baseline is not positive")
        ratios.append(float(candidate) / float(baseline))
    return float(np.mean(ratios))


def checkpoint_candidate(
    *,
    checkpoint_id: str,
    core_mae: Mapping[str, float | None],
    low_support_core_mae: Mapping[str, float | None],
    v5_1_core_mae: Mapping[str, float | None],
    all_outputs_finite: bool,
    temporal_leakage_count: int,
    pace_p50_mae: float | None,
    v5_1_pace_p50_mae: float | None,
    metric_counts: Mapping[str, Any] | None = None,
    rts_metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    sufficient = (
        all(_finite_metric(core_mae.get(target)) for target in CORE_TRANSFER_TARGETS)
        and all(_finite_metric(low_support_core_mae.get(target)) for target in CORE_TRANSFER_TARGETS)
        and _finite_metric(pace_p50_mae) and _finite_metric(v5_1_pace_p50_mae)
        and float(v5_1_pace_p50_mae) > 0
    )
    pace_degradation = (
        (float(pace_p50_mae) - float(v5_1_pace_p50_mae)) / float(v5_1_pace_p50_mae)
        if sufficient else None
    )
    try:
        primary = macro_normalized_core_mae(core_mae, v5_1_core_mae) if sufficient else None
        secondary_low = (
            float(np.mean([float(low_support_core_mae[target]) for target in CORE_TRANSFER_TARGETS]))
            if sufficient else None
        )
    except Stage2V52ContractError:
        sufficient = False
        primary = None
        secondary_low = None
    gates = {
        "all_outputs_finite": bool(all_outputs_finite),
        "temporal_leakage_count_zero": int(temporal_leakage_count) == 0,
        "metrics_have_sufficient_support": sufficient,
        "pace_p50_relative_degradation_at_most_2pct": (
            pace_degradation is not None and pace_degradation <= 0.02
        ),
    }
    return {
        "checkpoint_id": str(checkpoint_id),
        "hard_gates": gates,
        "hard_gate_status": "PASS" if all(gates.values()) else "FAIL",
        "primary_validation_macro_normalized_core_mae": primary,
        "secondary_low_support_macro_core_mae": secondary_low,
        "secondary_pace_p50_mae": float(pace_p50_mae) if _finite_metric(pace_p50_mae) else None,
        "pace_p50_relative_degradation": pace_degradation,
        "core_target_mae": dict(core_mae),
        "low_support_core_target_mae": dict(low_support_core_mae),
        "metric_counts": dict(metric_counts or {}),
        "rts_secondary_diagnostic": dict(rts_metrics or {}),
        "route_distribution_metrics_used": False,
    }


def select_micro_first_checkpoint(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in candidates if row.get("hard_gate_status") == "PASS"]
    if not eligible:
        return {"status": "NO_ELIGIBLE_CHECKPOINT", "selected_checkpoint_id": None}
    selected = min(
        eligible,
        key=lambda row: (
            row["primary_validation_macro_normalized_core_mae"],
            row["secondary_low_support_macro_core_mae"],
            row["secondary_pace_p50_mae"],
            row["checkpoint_id"],
        ),
    )
    return {
        "status": "MICRO_FIRST_CHECKPOINT_SELECTED",
        "selected_checkpoint_id": selected["checkpoint_id"],
        "selection_order": [
            "validation_macro_normalized_mae_over_4_core_micro_targets",
            "low_support_macro_core_micro_mae",
            "pace_p50_mae",
        ],
        "rts_used": False,
        "route_p90_p95_cvar_used": False,
        "selected_metrics": selected,
    }


def initialized_transfer_model(
    *,
    protocol_id: str,
    feature_artifact_path: str | Path,
    checkpoint_path: str | Path,
    source_model_manifest_path: str | Path,
    source_config_path: str | Path,
    static_feature_count: int,
    support_tau: float,
    spatial_mode: str,
    temporal_mode: str,
    backbone_kwargs: dict[str, Any],
) -> tuple[RCMSTNetTransfer, dict[str, Any]]:
    source_binding, feature_binding = bind_v51_source_model(
        protocol_id=protocol_id,
        feature_artifact_path=feature_artifact_path,
        source_checkpoint_path=checkpoint_path,
        source_model_manifest_path=source_model_manifest_path,
        source_config_path=source_config_path,
        backbone_kwargs=backbone_kwargs,
    )
    model = RCMSTNetTransfer(
        numeric_feature_count=len(source_binding.numeric_features),
        binding=feature_binding,
        static_feature_count=static_feature_count,
        support_tau=support_tau,
        spatial_mode=spatial_mode,
        temporal_mode=temporal_mode,
        backbone_kwargs=backbone_kwargs,
    )
    provenance = model.initialize_from_v51(checkpoint_path, source_binding=source_binding)
    return model, provenance


@dataclass
class MicroTreeBaseline:
    """M0: one bounded tree task per micro target and a two-part stop model."""

    random_state: int = 20261009
    max_iter: int = 80
    learning_rate: float = 0.08
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 50

    def __post_init__(self) -> None:
        common = {
            "max_iter": self.max_iter, "learning_rate": self.learning_rate,
            "max_leaf_nodes": self.max_leaf_nodes, "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
        }
        self.regressors = {
            target: HistGradientBoostingRegressor(**common)
            for target in ("crawl", "speed_cv", "acceleration_rms", "rts")
        }
        self.stop_occurrence = HistGradientBoostingClassifier(**common)
        self.stop_positive = HistGradientBoostingRegressor(**common)
        self.fit_manifest: dict[str, Any] | None = None

    def fit(
        self, features: np.ndarray, targets: Mapping[str, np.ndarray], masks: Mapping[str, np.ndarray],
        *, feature_schema_hash: str, train_dates: tuple[str, ...],
    ) -> "MicroTreeBaseline":
        matrix = np.asarray(features, dtype=np.float32)
        for target, model in self.regressors.items():
            values = np.asarray(targets[target], dtype=np.float64)
            valid = np.asarray(masks[target], dtype=bool) & np.isfinite(values)
            if not valid.any():
                raise Stage2V52ContractError(f"M0 has no valid Train rows for {target}")
            model.fit(matrix[valid], values[valid])
        stop = np.asarray(targets["stop"], dtype=np.float64)
        stop_valid = np.asarray(masks["stop"], dtype=bool) & np.isfinite(stop)
        occurrence = stop > 0
        if len(np.unique(occurrence[stop_valid])) != 2:
            raise Stage2V52ContractError("M0 stop occurrence requires both classes")
        self.stop_occurrence.fit(matrix[stop_valid], occurrence[stop_valid].astype(np.int8))
        positive = stop_valid & occurrence
        if not positive.any():
            raise Stage2V52ContractError("M0 stop positive-share model has no positive rows")
        self.stop_positive.fit(matrix[positive], stop[positive])
        self.fit_manifest = {
            "model": "M0_strong_micro_tree", "feature_schema_hash": str(feature_schema_hash),
            "train_dates": list(train_dates), "random_state": self.random_state,
            "targets": [
                "crawl", "stop_occurrence", "stop_positive_share", "speed_cv",
                "acceleration_rms", "rts_secondary",
            ],
            "formal_prediction_policy": "raw_preserved_and_comparison_clipped_to_0_1",
            "pace_only_tree": False,
        }
        return self

    def predict(self, features: np.ndarray) -> dict[str, np.ndarray]:
        if self.fit_manifest is None:
            raise Stage2V52ContractError("micro tree baseline has not been fitted")
        matrix = np.asarray(features, dtype=np.float32)
        occurrence = self.stop_occurrence.predict_proba(matrix)[:, 1]
        positive_raw = self.stop_positive.predict(matrix)
        raw = {target: model.predict(matrix) for target, model in self.regressors.items()}
        result: dict[str, np.ndarray] = {}
        for target, values in raw.items():
            result[f"pred_{target}_raw"] = values
            result[f"pred_{target}"] = np.clip(values, 0.0, 1.0)
        result.update({
            "stop_occurrence_probability": occurrence,
            "stop_positive_share_raw": positive_raw,
            "stop_positive_share": np.clip(positive_raw, 0.0, 1.0),
            "pred_stop_raw": occurrence * positive_raw,
            "pred_stop": np.clip(occurrence * positive_raw, 0.0, 1.0),
        })
        return result


def train_micro_tree_baseline_from_npz(
    *, protocol_id: str, input_path: str | Path, matrix_manifest_path: str | Path,
    output_root: str | Path, random_seed: int,
) -> dict[str, Any]:
    import joblib

    source = Path(input_path)
    matrix_manifest = _json(matrix_manifest_path)
    if matrix_manifest.get("schema_version") != M0_MATRIX_SCHEMA_VERSION:
        raise Stage2V52ContractError("M0 matrix manifest schema is invalid")
    if matrix_manifest.get("protocol_id") != protocol_id:
        raise Stage2V52ContractError("M0 matrix protocol mismatch")
    if matrix_manifest.get("matrix_sha256") != sha256_path(source):
        raise Stage2V52ContractError("M0 matrix hash differs from its canonical manifest")
    if matrix_manifest.get("forbidden_input_audit", {}).get("status") != "PASS":
        raise Stage2V52ContractError("M0 matrix failed forbidden-input audit")
    feature_schema_hash = str(matrix_manifest.get("feature_schema_hash", ""))
    if not feature_schema_hash:
        raise Stage2V52ContractError("M0 matrix manifest has no feature schema hash")
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "features", "split", "date", "crawl", "stop", "speed_cv", "acceleration_rms",
            "rts", "crawl_valid", "stop_valid", "speed_cv_valid", "acceleration_rms_valid", "rts_valid",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise Stage2V52ContractError(f"M0 matrix is missing arrays: {missing}")
        split, dates = archive["split"].astype(str), archive["date"].astype(str)
        protocol = get_protocol(protocol_id)
        if not np.all(split == "train") or tuple(sorted(np.unique(dates))) != tuple(sorted(protocol.train_dates)):
            raise Stage2V52ContractError("M0 matrix is not the exact protocol Train partition")
        features = archive["features"].copy()
        targets = {name: archive[name].copy() for name in ("crawl", "stop", "speed_cv", "acceleration_rms", "rts")}
        masks = {name: archive[f"{name}_valid"].copy() for name in targets}
    model = MicroTreeBaseline(random_state=random_seed).fit(
        features, targets, masks, feature_schema_hash=feature_schema_hash,
        train_dates=protocol.train_dates,
    )
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "m0_micro_tree.joblib"
    temporary = model_path.with_name(f".{model_path.name}.tmp")
    joblib.dump(model, temporary)
    os.replace(temporary, model_path)
    manifest = {
        "schema_version": M0_TRAINING_SCHEMA_VERSION, "status": "PASS",
        "protocol_id": protocol_id, "protocol_hash": protocol.digest,
        "input_sha256": sha256_path(source),
        "matrix_manifest_sha256": sha256_path(matrix_manifest_path),
        "feature_schema_hash": feature_schema_hash,
        "model_sha256": sha256_path(model_path), **dict(model.fit_manifest or {}),
    }
    write_training_manifest(output / "model_manifest.json", manifest)
    return manifest


def _protocol_root(root: Path, protocol_id: str) -> Path:
    candidate = root / f"protocol={protocol_id}"
    resolved = candidate if candidate.is_dir() else root
    manifest_path = resolved / "transfer_manifest.json"
    if not manifest_path.is_file():
        raise Stage2V52ContractError(f"missing transfer manifest: {manifest_path}")
    manifest = _json(manifest_path)
    if (
        manifest.get("schema_version") != TRANSFER_MANIFEST_SCHEMA_VERSION
        or manifest.get("protocol_id") != protocol_id
        or manifest.get("protocol_hash") != get_protocol(protocol_id).digest
    ):
        raise Stage2V52ContractError("transfer shard manifest differs from frozen protocol")
    return resolved


def _protocol_shards(root: Path, protocol_id: str, role: str) -> list[Path]:
    protocol_root = _protocol_root(root, protocol_id)
    roles = protocol_role_dates(protocol_id)
    if role not in roles:
        raise Stage2V52ContractError(f"unknown canonical transfer shard role: {role}")
    dates = roles[role]
    if not dates:
        raise Stage2V52ContractError(f"protocol {protocol_id} has no dates for role {role}")
    paths = [
        path for date in dates
        for path in sorted((protocol_root / f"split={role}" / f"date={date}").glob("shard-*.npz"))
    ]
    missing = [date for date in dates if not list((protocol_root / f"split={role}" / f"date={date}").glob("shard-*.npz"))]
    if missing:
        raise Stage2V52ContractError(f"missing {role} transfer shards for frozen dates: {missing}")
    return paths


def _bound_temporal_audit(root: Path, protocol_id: str) -> tuple[dict[str, Any], str]:
    protocol_root = _protocol_root(root, protocol_id)
    tensor_manifest = _json(protocol_root / "transfer_manifest.json")
    audit_path = protocol_root / str(tensor_manifest.get("temporal_audit_path", ""))
    if not audit_path.is_file() or sha256_path(audit_path) != tensor_manifest.get("temporal_audit_sha256"):
        raise Stage2V52ContractError("transfer temporal audit is missing or hash-mismatched")
    audit = _json(audit_path)
    if (
        audit.get("protocol_id") != protocol_id or audit.get("status") != "PASS"
        or int(audit.get("temporal_leakage_count", -1)) != 0
    ):
        raise Stage2V52ContractError("transfer temporal leakage audit failed")
    return audit, sha256_path(audit_path)


def _batch_indices(
    length: int, size: int, *, shuffle: bool, rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    index = np.arange(length)
    if shuffle:
        if rng is None:
            raise Stage2V52ContractError("shuffled batch order requires an explicit NumPy Generator")
        rng.shuffle(index)
    return [index[start : start + size] for start in range(0, length, size)]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _forward_loss(
    model: RCMSTNetTransfer, data: Mapping[str, np.ndarray], index: np.ndarray, device: Any,
    *, component_weights: Mapping[str, float],
):
    import torch
    from stage2.v5.models.losses import rc_mstnet_v5_loss

    component_weights = validate_loss_weight_schema(component_weights)

    def tensor(name: str, dtype: Any) -> Any:
        return torch.as_tensor(data[name][index], dtype=dtype, device=device)

    pad = tensor("pad_mask", torch.bool)
    temporal_array = tensor("temporal_features", torch.float32)
    temporal = {name: temporal_array[..., offset] for offset, name in enumerate(TEMPORAL_FEATURE_NAMES)}
    output = model(
        tensor("numeric", torch.float32), tensor("numeric_missing", torch.bool),
        tensor("categorical", torch.long), tensor("route_sequence", torch.long).clamp_min(0), pad,
        static_edge_features=tensor("static_edge_features", torch.float32),
        edge_train_support=tensor("edge_train_support", torch.float32), temporal_features=temporal,
        recent_history=tensor("recent_history", torch.float32), profile_history=tensor("profile_history", torch.float32),
        forecast_horizon_s=tensor("forecast_horizon_s", torch.float32), history_age_s=tensor("history_age_s", torch.float32),
        history_support=tensor("history_support", torch.float32),
    )
    target = tensor("targets", torch.float32)
    target_mask = tensor("target_masks", torch.bool)
    tail, tail_mask = tensor("tail_targets", torch.float32), tensor("tail_masks", torch.bool)
    availability, valid = tensor("availability_targets", torch.float32), ~pad
    targets = {
        "crawl_time_share": target[..., 0], "stop_time_share": target[..., 1],
        "speed_cv_bounded": target[..., 2], "acceleration_rms_bounded": target[..., 3],
        "rts_raw": target[..., 4], "lcs_raw": target[..., 5], "pace_sec_per_m": target[..., 6],
        "lcs_tail_event": tail[..., 0], "rts_tail_event": tail[..., 1], "availability": availability,
    }
    masks = {
        "crawl_target_valid": target_mask[..., 0] & valid,
        "stop_target_valid": target_mask[..., 1] & valid,
        "speed_cv_target_valid": target_mask[..., 2] & valid,
        "acceleration_rms_target_valid": target_mask[..., 3] & valid,
        "rts_target_valid": target_mask[..., 4] & valid,
        "lcs_target_valid": target_mask[..., 5] & valid,
        "pace_target_valid": target_mask[..., 6] & valid,
        "availability_valid": valid.unsqueeze(-1).expand_as(availability),
    }
    loss, components = rc_mstnet_v5_loss(
        output, targets, masks, tensor("supervision_weight", torch.float32),
        component_weights=dict(component_weights),
    )
    return loss, output, target, target_mask, pad, tensor("support_group_code", torch.long), components


def collect_unique_predictions(
    model: RCMSTNetTransfer, paths: list[Path], batch_size: int, device: Any,
    *, component_weights: Mapping[str, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Average overlapping chunk predictions before computing any metric."""
    import torch

    prediction_keys = {
        "crawl": "crawl_share", "stop": "stop_share", "speed_cv": "speed_cv",
        "acceleration_rms": "acceleration_rms", "rts": "rts_raw", "pace_p50": "pace_pred_p50",
    }
    target_index = {"crawl": 0, "stop": 1, "speed_cv": 2, "acceleration_rms": 3, "rts": 4, "pace_p50": 6}
    pieces: list[pd.DataFrame] = []
    all_outputs_finite = True
    model.eval()
    with torch.inference_mode():
        for path in paths:
            data = _load_npz(path)
            for indices in _batch_indices(len(data["numeric"]), batch_size, shuffle=False):
                _, output, target, target_mask, pad, support_group, _ = _forward_loss(
                    model, data, indices, device, component_weights=component_weights,
                )
                valid = (~pad).cpu().numpy()
                all_outputs_finite &= all(
                    bool(torch.isfinite(value).all()) for value in output.values()
                    if isinstance(value, torch.Tensor) and value.is_floating_point()
                )
                chunk_split = data["split"][indices].astype(str)
                chunk_date = data["date"][indices].astype(str)
                chunk_order = data["order_id"][indices].astype(str)
                frame = pd.DataFrame({
                    "split": np.broadcast_to(chunk_split[:, None], valid.shape)[valid],
                    "date": np.broadcast_to(chunk_date[:, None], valid.shape)[valid],
                    "order_id": np.broadcast_to(chunk_order[:, None], valid.shape)[valid],
                    "traversal_id": data["traversal_id"][indices][valid].astype(np.int64),
                    "support_group_code": support_group.cpu().numpy()[valid].astype(np.int8),
                    "allocated_distance_m": data["allocated_distance_m"][indices][valid].astype(np.float32),
                })
                target_np, mask_np = target.cpu().numpy(), target_mask.cpu().numpy()
                for name, output_key in prediction_keys.items():
                    frame[f"pred_{name}"] = output[output_key].float().cpu().numpy()[valid]
                    mask = mask_np[..., target_index[name]][valid]
                    frame[f"{name}_valid"] = mask
                    values = target_np[..., target_index[name]][valid]
                    frame[f"target_{name}"] = np.where(mask, values, np.nan)
                pieces.append(frame)
    if not pieces:
        raise Stage2V52ContractError("checkpoint evaluation produced no chunk tokens")
    chunk = pd.concat(pieces, ignore_index=True)
    key = ["date", "order_id", "traversal_id"]
    consistency = chunk.groupby(key, sort=False, observed=True)["support_group_code"].nunique()
    if (consistency > 1).any():
        raise Stage2V52ContractError("overlap copies disagree on support group")
    distance_range = chunk.groupby(key, sort=False, observed=True)["allocated_distance_m"].agg(["min", "max"])
    if (~np.isclose(distance_range["min"], distance_range["max"], atol=1.0e-6, rtol=0)).any():
        raise Stage2V52ContractError("overlap copies disagree on allocated distance")
    for name in prediction_keys:
        group = chunk.groupby(key, sort=False, observed=True)
        if (group[f"{name}_valid"].nunique() > 1).any():
            raise Stage2V52ContractError(f"overlap copies disagree on {name} validity")
        target_range = group[f"target_{name}"].agg(["min", "max"])
        inconsistent_target = (
            target_range["min"].notna() & target_range["max"].notna()
            & ~np.isclose(target_range["min"], target_range["max"], atol=1.0e-7, rtol=0)
        )
        if inconsistent_target.any():
            raise Stage2V52ContractError(f"overlap copies disagree on {name} target")
    aggregations: dict[str, Any] = {
        "split": "first", "support_group_code": "first", "allocated_distance_m": "first",
    }
    for name in prediction_keys:
        aggregations[f"pred_{name}"] = "mean"
        aggregations[f"{name}_valid"] = "max"
        aggregations[f"target_{name}"] = "mean"
    unique = chunk.groupby(key, sort=False, observed=True, as_index=False).agg(aggregations)
    prediction_aliases = {
        "pred_crawl_time_share": "pred_crawl", "pred_stop_time_share": "pred_stop",
        "pred_speed_cv_bounded": "pred_speed_cv",
        "pred_acceleration_rms_bounded": "pred_acceleration_rms",
        "pred_rts_raw": "pred_rts", "pace_pred_p50": "pred_pace_p50",
    }
    mask_aliases = {
        "crawl_target_valid": "crawl_valid", "stop_target_valid": "stop_valid",
        "speed_cv_target_valid": "speed_cv_valid",
        "acceleration_rms_target_valid": "acceleration_rms_valid",
        "rts_target_valid": "rts_valid", "pace_target_valid": "pace_p50_valid",
    }
    for output_name, source_name in prediction_aliases.items():
        unique[output_name] = unique[source_name]
    for output_name, source_name in mask_aliases.items():
        unique[output_name] = unique[source_name]
    return unique, {
        "all_outputs_finite": all_outputs_finite,
        "chunk_token_count": int(len(chunk)),
        "unique_traversal_count": int(len(unique)),
        "duplicate_prediction_count": int(len(chunk) - len(unique)),
        "overlap_merge": "mean_prediction_by_date_order_id_traversal_id",
    }


def unique_traversal_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    groups = {"overall": np.ones(len(frame), dtype=bool), "low": frame["support_group_code"].eq(1).to_numpy(),
              "unseen": frame["support_group_code"].eq(0).to_numpy()}
    output: dict[str, Any] = {"metric_definition": CORE_METRIC_DEFINITION, "groups": {}}
    for group, group_mask in groups.items():
        target_metrics: dict[str, Any] = {}
        for target in CORE_TRANSFER_TARGETS:
            valid = group_mask & frame[f"{target}_valid"].to_numpy(bool)
            valid &= np.isfinite(frame[f"target_{target}"].to_numpy(float))
            valid &= np.isfinite(frame[f"pred_{target}"].to_numpy(float))
            count = int(valid.sum())
            target_metrics[target] = {
                "count": count,
                "mae": float(np.mean(np.abs(
                    frame.loc[valid, f"pred_{target}"].to_numpy(float)
                    - frame.loc[valid, f"target_{target}"].to_numpy(float)
                ))) if count else None,
                "status": "PASS" if count else "INSUFFICIENT_SUPPORT",
            }
        output["groups"][group] = target_metrics
    pace_valid = frame["pace_p50_valid"].to_numpy(bool)
    pace_valid &= np.isfinite(frame["target_pace_p50"].to_numpy(float)) & np.isfinite(frame["pred_pace_p50"].to_numpy(float))
    pace_count = int(pace_valid.sum())
    output["pace_p50"] = {
        "definition": PACE_METRIC_DEFINITION, "count": pace_count,
        "mae": float(np.mean(np.abs(
            frame.loc[pace_valid, "pred_pace_p50"].to_numpy(float)
            - frame.loc[pace_valid, "target_pace_p50"].to_numpy(float)
        ))) if pace_count else None,
        "status": "PASS" if pace_count else "INSUFFICIENT_SUPPORT",
    }
    return output


def _validation_mae(
    model: RCMSTNetTransfer, paths: list[Path], batch_size: int, device: Any,
    *, component_weights: Mapping[str, float],
) -> dict[str, Any]:
    unique, diagnostics = collect_unique_predictions(
        model, paths, batch_size, device, component_weights=component_weights,
    )
    metrics = unique_traversal_metrics(unique)
    return {
        "all_outputs_finite": diagnostics["all_outputs_finite"],
        "core_mae": {target: metrics["groups"]["overall"][target]["mae"] for target in CORE_TRANSFER_TARGETS},
        "low_support_core_mae": {target: metrics["groups"]["low"][target]["mae"] for target in CORE_TRANSFER_TARGETS},
        "pace_p50_mae": metrics["pace_p50"]["mae"],
        "metric_counts": {
            "overall": {target: metrics["groups"]["overall"][target]["count"] for target in CORE_TRANSFER_TARGETS},
            "low": {target: metrics["groups"]["low"][target]["count"] for target in CORE_TRANSFER_TARGETS},
            "unseen": {target: metrics["groups"]["unseen"][target]["count"] for target in CORE_TRANSFER_TARGETS},
            "pace_p50": metrics["pace_p50"]["count"],
            **{name: diagnostics[name] for name in (
                "chunk_token_count", "unique_traversal_count", "duplicate_prediction_count"
            )},
        },
    }


def validate_m1_metric_manifest(
    payload: Mapping[str, Any], *, protocol_id: str, source_checkpoint_sha256: str,
    feature_artifact_sha256: str, tensor_manifest_sha256: str,
    support_artifact_sha256: str,
) -> None:
    protocol = get_protocol(protocol_id)
    if payload.get("schema_version") != "stage2_v5_2_evaluation.2" or payload.get("model_id") != "M1":
        raise Stage2V52ContractError("baseline metric manifest is not a formal M1 evaluation")
    required = {
        "protocol_id": protocol_id, "source_checkpoint_sha256": source_checkpoint_sha256,
        "feature_artifact_sha256": feature_artifact_sha256,
        "tensor_manifest_sha256": tensor_manifest_sha256,
        "support_artifact_sha256": support_artifact_sha256,
        "core_metric_definition": CORE_METRIC_DEFINITION,
        "pace_metric_definition": PACE_METRIC_DEFINITION,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise Stage2V52ContractError("M1 metric manifest differs from current protocol/source")
    if tuple(payload.get("evaluation_dates", ())) != protocol.validation_dates:
        raise Stage2V52ContractError("M1 metric validation dates differ from current protocol")
    if int(payload.get("unique_traversal_count", 0)) <= 0:
        raise Stage2V52ContractError("M1 metric manifest has no unique traversals")


def train_transfer_from_shards(
    *, protocol_id: str, model_id: str, tensor_root: str | Path,
    feature_artifact_path: str | Path, source_checkpoint_path: str | Path,
    source_model_manifest_path: str | Path, source_config_path: str | Path,
    static_artifact_path: str | Path, support_tau: float, backbone_kwargs: dict[str, Any],
    v5_1_metric_manifest: Mapping[str, Any] | None, output_root: str | Path,
    new_branch_lr: float, backbone_lr_ratio: float, shared_freeze_epochs: int,
    maximum_epochs: int, batch_size: int, base_seed: int,
    component_weights: Mapping[str, float], m4_checkpoint_path: str | Path | None = None,
    m4_training_manifest: Mapping[str, Any] | None = None,
    m4_adoption_manifest: Mapping[str, Any] | None = None,
    m4_adoption_manifest_sha256: str | None = None,
    support_artifact_path: str | Path | None = None,
    support_tau_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute M1–M5 on canonical protocol-bound augmented shards."""
    import torch

    modes = {
        "M1": ("identity", "none"), "M2": ("structure_only", "none"),
        "M3": ("concat", "none"), "M4": ("support_aware", "none"),
        "M5": ("support_aware", "zero_shot"),
    }
    if model_id not in modes:
        raise Stage2V52ContractError("deep training model must be M1-M5")
    component_weights = validate_loss_weight_schema(component_weights)
    protocol = get_protocol(protocol_id)
    static_artifact = _json(static_artifact_path)
    if static_artifact.get("protocol_id") != protocol_id:
        raise Stage2V52ContractError("static artifact protocol differs from training protocol")
    static_feature_count = int(static_artifact.get("feature_count", 0))
    if static_feature_count <= 0:
        raise Stage2V52ContractError("static feature count must be derived from a valid artifact")
    temporal_audit, temporal_audit_sha = _bound_temporal_audit(Path(tensor_root), protocol_id)
    tensor_manifest_path = _protocol_root(Path(tensor_root), protocol_id) / "transfer_manifest.json"
    tensor_manifest = _json(tensor_manifest_path)
    if (
        tensor_manifest.get("feature_artifact_sha256") != sha256_path(feature_artifact_path)
        or tensor_manifest.get("static_artifact_sha256") != sha256_path(static_artifact_path)
    ):
        raise Stage2V52ContractError("transfer tensors are not bound to the training artifacts")
    if support_artifact_path is not None and (
        tensor_manifest.get("support_artifact_sha256") != sha256_path(support_artifact_path)
    ):
        raise Stage2V52ContractError("selected tau support artifact differs from transfer tensors")
    tau_provenance = dict(support_tau_provenance or {})
    if model_id == "M4" and protocol_id == "transfer_tuning":
        if (
            tau_provenance.get("kind") != "train_support_quantile_candidate"
            or tau_provenance.get("support_tau_candidate") not in {"p25", "p50", "p75"}
            or float(tau_provenance.get("support_tau_value", float("nan"))) != float(support_tau)
            or tau_provenance.get("support_tau_source_support_sha256")
            != tensor_manifest.get("support_artifact_sha256")
        ):
            raise Stage2V52ContractError("transfer-tuning M4 tau lacks verified P25/P50/P75 provenance")
    elif model_id in {"M4", "M5"}:
        if (
            tau_provenance.get("kind") != "frozen_transfer_tuning_selection"
            or not isinstance(tau_provenance.get("tau_freeze_artifact_sha256"), str)
            or not isinstance(tau_provenance.get("tau_selection_artifact_sha256"), str)
            or tau_provenance.get("support_tau_candidate") not in {"p25", "p50", "p75"}
            or float(tau_provenance.get("support_tau_value", float("nan"))) != float(support_tau)
            or tau_provenance.get("current_protocol_support_artifact_sha256")
            != tensor_manifest.get("support_artifact_sha256")
        ):
            raise Stage2V52ContractError("M4/M5 tau lacks frozen transfer-tuning selection provenance")
    np_rng = np.random.default_rng(int(base_seed))
    torch.manual_seed(int(base_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(base_seed))
    spatial_mode, temporal_mode = modes[model_id]
    model, source = initialized_transfer_model(
        protocol_id=protocol_id, feature_artifact_path=feature_artifact_path,
        checkpoint_path=source_checkpoint_path, source_model_manifest_path=source_model_manifest_path,
        source_config_path=source_config_path, static_feature_count=static_feature_count,
        support_tau=support_tau, spatial_mode=spatial_mode, temporal_mode=temporal_mode,
        backbone_kwargs=backbone_kwargs,
    )
    m4_initialization: dict[str, Any] | None = None
    if model_id == "M5":
        if (
            m4_checkpoint_path is None or m4_training_manifest is None or m4_adoption_manifest is None
            or not isinstance(m4_adoption_manifest_sha256, str) or len(m4_adoption_manifest_sha256) != 64
        ):
            raise Stage2V52ContractError("M5 requires selected M4 checkpoint, training, and adoption manifests")
        if (
            m4_training_manifest.get("schema_version") != TRAINING_SCHEMA_VERSION
            or m4_training_manifest.get("status") != "PASS"
            or m4_training_manifest.get("protocol_id") != protocol_id
            or m4_training_manifest.get("model_id") != "M4"
            or m4_training_manifest.get("selected_checkpoint_sha256") != sha256_path(m4_checkpoint_path)
        ):
            raise Stage2V52ContractError("M4 training manifest does not bind the selected checkpoint")
        validate_m5_m4_adoption(
            m4_adoption_manifest, protocol_id=protocol_id,
            m4_checkpoint_sha256=sha256_path(m4_checkpoint_path),
        )
        m4_initialization = model.initialize_spatial_from_m4(m4_checkpoint_path)
        m4_initialization["m4_adoption_manifest_sha256"] = m4_adoption_manifest_sha256
        model.set_temporal_adapter_only_trainable()
    if model_id != "M1":
        if v5_1_metric_manifest is None:
            raise Stage2V52ContractError(f"{model_id} requires a formal M1 metric manifest")
        validate_m1_metric_manifest(
            v5_1_metric_manifest, protocol_id=protocol_id,
            source_checkpoint_sha256=source["source_checkpoint_sha256"],
            feature_artifact_sha256=source["feature_artifact_sha256"],
            tensor_manifest_sha256=sha256_path(tensor_manifest_path),
            support_artifact_sha256=str(tensor_manifest.get("support_artifact_sha256", "")),
        )
        baseline_core = v5_1_metric_manifest["core_mae"]
        baseline_pace = v5_1_metric_manifest["pace_p50_mae"]
    else:
        baseline_core, baseline_pace = None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_paths = _protocol_shards(Path(tensor_root), protocol_id, "train")
    validation_paths = _protocol_shards(Path(tensor_root), protocol_id, "validation")
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    if model_id == "M1":
        maximum_epochs = 0
    optimizer = (
        torch.optim.AdamW(
            model.optimizer_parameter_groups(new_branch_lr=new_branch_lr, backbone_lr_ratio=backbone_lr_ratio),
            weight_decay=1.0e-4,
        ) if maximum_epochs else None
    )
    for epoch in range(maximum_epochs):
        if model_id == "M5":
            model.set_temporal_adapter_only_trainable()
        else:
            model.set_shared_backbone_frozen(epoch < shared_freeze_epochs)
        model.train()
        for path in train_paths:
            data = _load_npz(path)
            for index in _batch_indices(len(data["numeric"]), batch_size, shuffle=True, rng=np_rng):
                optimizer.zero_grad(set_to_none=True)
                loss, *_ = _forward_loss(model, data, index, device, component_weights=component_weights)
                if not torch.isfinite(loss):
                    raise Stage2V52ContractError("non-finite transfer training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        metrics = _validation_mae(
            model, validation_paths, batch_size, device, component_weights=component_weights,
        )
        candidate = checkpoint_candidate(
            checkpoint_id=f"epoch_{epoch + 1:03d}", core_mae=metrics["core_mae"],
            low_support_core_mae=metrics["low_support_core_mae"],
            v5_1_core_mae=baseline_core, all_outputs_finite=metrics["all_outputs_finite"],
            temporal_leakage_count=int(temporal_audit["temporal_leakage_count"]),
            pace_p50_mae=metrics["pace_p50_mae"], v5_1_pace_p50_mae=baseline_pace,
            metric_counts=metrics["metric_counts"],
        )
        candidates.append(candidate)
        checkpoint_path = output / f"{candidate['checkpoint_id']}.pt"
        temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
        torch.save({
            "model_state_dict": model.state_dict(), "candidate": candidate,
            "protocol_id": protocol_id, "model_id": model_id,
        }, temporary)
        os.replace(temporary, checkpoint_path)
    if model_id == "M1":
        metrics = _validation_mae(
            model, validation_paths, batch_size, device, component_weights=component_weights,
        )
        baseline_core, baseline_pace = metrics["core_mae"], metrics["pace_p50_mae"]
        candidates.append(checkpoint_candidate(
            checkpoint_id="frozen_v5_1", core_mae=metrics["core_mae"],
            low_support_core_mae=metrics["low_support_core_mae"], v5_1_core_mae=baseline_core,
            all_outputs_finite=metrics["all_outputs_finite"],
            temporal_leakage_count=int(temporal_audit["temporal_leakage_count"]),
            pace_p50_mae=metrics["pace_p50_mae"], v5_1_pace_p50_mae=baseline_pace,
            metric_counts=metrics["metric_counts"],
        ))
    selection = select_micro_first_checkpoint(candidates)
    selected_id = selection.get("selected_checkpoint_id")
    if model_id == "M1" and selected_id == "frozen_v5_1":
        selected_path = Path(source_checkpoint_path)
    else:
        selected_path = output / f"{selected_id}.pt" if selected_id else None
    selected_sha = sha256_path(selected_path) if selected_path is not None and selected_path.is_file() else None
    manifest = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "status": "PASS" if selection["status"] == "MICRO_FIRST_CHECKPOINT_SELECTED" else "FAIL",
        "protocol_id": protocol_id, "protocol_hash": protocol.digest, "model_id": model_id,
        "source": source, "m4_initialization": m4_initialization,
        "initialization_policy": (
            "selected_m4_spatial_plus_fresh_temporal_adapter" if model_id == "M5"
            else source["initialization_policy"]
        ),
        "constructor": {
            "static_feature_count": static_feature_count, "support_tau": support_tau,
            "support_tau_provenance": tau_provenance,
            "spatial_mode": spatial_mode, "temporal_mode": temporal_mode,
            "backbone_kwargs": dict(backbone_kwargs),
        },
        "static_artifact_sha256": sha256_path(static_artifact_path),
        "tensor_manifest_path": tensor_manifest_path.as_posix(),
        "tensor_manifest_sha256": sha256_path(tensor_manifest_path),
        "support_artifact_sha256": tensor_manifest.get("support_artifact_sha256"),
        "temporal_audit_sha256": temporal_audit_sha,
        "temporal_leakage_count": int(temporal_audit["temporal_leakage_count"]),
        "backbone_lr": new_branch_lr * backbone_lr_ratio, "new_branch_lr": new_branch_lr,
        "shared_freeze_epochs": shared_freeze_epochs,
        "m5_trainable_scope": "fresh_temporal_adapter_only" if model_id == "M5" else None,
        "random_seed": {
            "base_seed": int(base_seed), "numpy_rng": "numpy.random.default_rng(PCG64)",
            "torch_manual_seed": int(base_seed),
            "torch_cuda_manual_seed_all": int(base_seed) if torch.cuda.is_available() else None,
            "batch_order_policy": "same_seed_same_protocol_same_shard_order",
        },
        "loss_policy": {
            "component_weights": dict(component_weights),
            "rts_role": "legacy_descriptive_diagnostic_not_stage3_deployable",
            "rts_stage3_deployable": False,
            "rts_raw_mask": "rts_target_valid_independent_of_tail_mask",
            "lcs_raw_mask": "lcs_target_valid_independent_of_tail_mask",
        },
        "checkpoint_selection": selection, "candidates": candidates,
        "selected_checkpoint_path": selected_path.as_posix() if selected_path else None,
        "selected_checkpoint_sha256": selected_sha,
    }
    write_training_manifest(output / "model_manifest.json", manifest)
    return manifest
