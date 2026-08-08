"""Micro-first checkpointing, v5.1 initialization, and strong tree baseline."""

from __future__ import annotations

import json
import os
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .contracts import CORE_TRANSFER_TARGETS, Stage2V52ContractError
from .feature_binding import bind_v51_feature_schema
from .models.rc_mstnet_transfer import RCMSTNetTransfer
from .protocols import get_protocol
from .temporal_adapter import TEMPORAL_FEATURE_NAMES


CHECKPOINT_HARD_GATES = (
    "all_outputs_finite",
    "temporal_leakage_count_zero",
    "pace_p50_relative_degradation_at_most_2pct",
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def macro_normalized_core_mae(
    candidate_mae: Mapping[str, float],
    v5_1_mae: Mapping[str, float],
) -> float:
    if set(candidate_mae) != set(CORE_TRANSFER_TARGETS) or set(v5_1_mae) != set(CORE_TRANSFER_TARGETS):
        raise Stage2V52ContractError("checkpoint score requires exactly four core micro targets")
    ratios = []
    for target in CORE_TRANSFER_TARGETS:
        candidate = float(candidate_mae[target])
        baseline = float(v5_1_mae[target])
        if not np.isfinite(candidate) or not np.isfinite(baseline) or baseline <= 0:
            raise Stage2V52ContractError("checkpoint MAE inputs must be finite and baseline positive")
        ratios.append(candidate / baseline)
    return float(np.mean(ratios))


def checkpoint_candidate(
    *,
    checkpoint_id: str,
    core_mae: Mapping[str, float],
    low_support_core_mae: Mapping[str, float],
    v5_1_core_mae: Mapping[str, float],
    all_outputs_finite: bool,
    temporal_leakage_count: int,
    pace_p50_mae: float,
    v5_1_pace_p50_mae: float,
    rts_metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    pace_degradation = (float(pace_p50_mae) - float(v5_1_pace_p50_mae)) / float(v5_1_pace_p50_mae)
    gates = {
        "all_outputs_finite": bool(all_outputs_finite),
        "temporal_leakage_count_zero": int(temporal_leakage_count) == 0,
        "pace_p50_relative_degradation_at_most_2pct": pace_degradation <= 0.02,
    }
    return {
        "checkpoint_id": str(checkpoint_id),
        "hard_gates": gates,
        "hard_gate_status": "PASS" if all(gates.values()) else "FAIL",
        "primary_validation_macro_normalized_core_mae": macro_normalized_core_mae(core_mae, v5_1_core_mae),
        "secondary_low_support_macro_core_mae": float(np.mean([
            float(low_support_core_mae[target]) for target in CORE_TRANSFER_TARGETS
        ])),
        "secondary_pace_p50_mae": float(pace_p50_mae),
        "pace_p50_relative_degradation": pace_degradation,
        "core_target_mae": dict(core_mae),
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
    feature_artifact_path: str | Path,
    checkpoint_path: str | Path,
    source_model_id: str,
    numeric_feature_count: int,
    static_feature_count: int,
    support_tau: float,
    spatial_mode: str,
    temporal_mode: str,
    backbone_kwargs: dict[str, Any],
) -> tuple[RCMSTNetTransfer, dict[str, Any]]:
    import torch

    saved = torch.load(checkpoint_path, map_location="cpu")
    state = saved.get("model_state_dict", saved)
    binding = bind_v51_feature_schema(feature_artifact_path, checkpoint_state=state)
    model = RCMSTNetTransfer(
        numeric_feature_count=numeric_feature_count,
        binding=binding,
        static_feature_count=static_feature_count,
        support_tau=support_tau,
        spatial_mode=spatial_mode,
        temporal_mode=temporal_mode,
        backbone_kwargs=backbone_kwargs,
    )
    provenance = model.initialize_from_v51(
        checkpoint_path,
        source_model_id=source_model_id,
        source_feature_artifact_path=feature_artifact_path,
    )
    return model, provenance


@dataclass
class MicroTreeBaseline:
    """M0: one strong tree task per micro target on the shared feature matrix."""

    random_state: int = 20261009
    max_iter: int = 80
    learning_rate: float = 0.08
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 50

    def __post_init__(self) -> None:
        common = {
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_samples_leaf": self.min_samples_leaf,
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
        self,
        features: np.ndarray,
        targets: Mapping[str, np.ndarray],
        masks: Mapping[str, np.ndarray],
        *,
        feature_schema_hash: str,
        train_dates: tuple[str, ...],
    ) -> "MicroTreeBaseline":
        matrix = np.asarray(features, dtype=np.float32)
        for target, model in self.regressors.items():
            values = np.asarray(targets[target], dtype=np.float64)
            valid = np.asarray(masks[target], dtype=bool) & np.isfinite(values)
            model.fit(matrix[valid], values[valid])
        stop = np.asarray(targets["stop"], dtype=np.float64)
        stop_valid = np.asarray(masks["stop"], dtype=bool) & np.isfinite(stop)
        occurrence = stop > 0
        self.stop_occurrence.fit(matrix[stop_valid], occurrence[stop_valid].astype(np.int8))
        positive = stop_valid & occurrence
        self.stop_positive.fit(matrix[positive], stop[positive])
        self.fit_manifest = {
            "model": "M0_strong_micro_tree",
            "feature_schema_hash": str(feature_schema_hash),
            "train_dates": list(train_dates),
            "targets": ["crawl", "stop_occurrence", "stop_positive_share", "speed_cv", "acceleration_rms", "rts_secondary"],
            "pace_only_tree": False,
        }
        return self

    def predict(self, features: np.ndarray) -> dict[str, np.ndarray]:
        if self.fit_manifest is None:
            raise Stage2V52ContractError("micro tree baseline has not been fitted")
        matrix = np.asarray(features, dtype=np.float32)
        probability = self.stop_occurrence.predict_proba(matrix)[:, 1]
        positive = np.clip(self.stop_positive.predict(matrix), 0.0, 1.0)
        result = {f"pred_{target}": model.predict(matrix) for target, model in self.regressors.items()}
        result.update({
            "stop_occurrence_probability": probability,
            "stop_positive_share": positive,
            "pred_stop": probability * positive,
        })
        return result


def train_micro_tree_baseline_from_npz(
    *,
    protocol_id: str,
    input_path: str | Path,
    feature_schema_hash: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Fit M0 from one protocol-bound Train-only matrix artifact."""
    import joblib

    source = Path(input_path)
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "features", "split", "date", "crawl", "stop", "speed_cv",
            "acceleration_rms", "rts", "crawl_valid", "stop_valid",
            "speed_cv_valid", "acceleration_rms_valid", "rts_valid",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise Stage2V52ContractError(f"M0 matrix is missing arrays: {missing}")
        split = archive["split"].astype(str)
        dates = archive["date"].astype(str)
        if not np.all(split == "train"):
            raise Stage2V52ContractError("M0 matrix contains non-Train rows")
        protocol = get_protocol(protocol_id)
        observed_dates = tuple(sorted(np.unique(dates).tolist()))
        if observed_dates != tuple(sorted(protocol.train_dates)):
            raise Stage2V52ContractError("M0 matrix dates differ from frozen protocol Train dates")
        features = archive["features"].copy()
        targets = {name: archive[name].copy() for name in (
            "crawl", "stop", "speed_cv", "acceleration_rms", "rts"
        )}
        masks = {name: archive[f"{name}_valid"].copy() for name in targets}
    model = MicroTreeBaseline().fit(
        features, targets, masks,
        feature_schema_hash=feature_schema_hash,
        train_dates=protocol.train_dates,
    )
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "m0_micro_tree.joblib"
    temporary = model_path.with_name(f".{model_path.name}.tmp")
    joblib.dump(model, temporary)
    os.replace(temporary, model_path)
    manifest = {
        "schema_version": "stage2_v5_2_m0_training.1",
        "status": "PASS",
        "protocol_id": protocol_id,
        "protocol_hash": protocol.digest,
        "input_sha256": _sha256_file(source),
        "feature_schema_hash": str(feature_schema_hash),
        "model_sha256": _sha256_file(model_path),
        **dict(model.fit_manifest or {}),
    }
    write_training_manifest(output / "model_manifest.json", manifest)
    return manifest


def write_training_manifest(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _protocol_shards(root: Path, protocol_id: str, role: str) -> list[Path]:
    protocol = get_protocol(protocol_id)
    dates = protocol.train_dates if role == "train" else protocol.validation_dates
    paths = [path for date in dates for path in sorted((root / f"split={role}" / f"date={date}").glob("shard-*.npz"))]
    missing = [date for date in dates if not list((root / f"split={role}" / f"date={date}").glob("shard-*.npz"))]
    if missing:
        raise Stage2V52ContractError(f"missing {role} transfer shards for frozen dates: {missing}")
    return paths


def _batch_indices(length: int, size: int, *, shuffle: bool) -> list[np.ndarray]:
    index = np.arange(length)
    if shuffle:
        np.random.shuffle(index)
    return [index[start : start + size] for start in range(0, length, size)]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _forward_loss(model: RCMSTNetTransfer, data: Mapping[str, np.ndarray], index: np.ndarray, device: Any):
    import torch
    from stage2.v5.models.losses import rc_mstnet_v5_loss

    def tensor(name: str, dtype: Any) -> Any:
        return torch.as_tensor(data[name][index], dtype=dtype, device=device)

    pad = tensor("pad_mask", torch.bool)
    temporal_array = tensor("temporal_features", torch.float32)
    temporal = {
        name: temporal_array[..., feature_index]
        for feature_index, name in enumerate(TEMPORAL_FEATURE_NAMES)
    }
    output = model(
        tensor("numeric", torch.float32),
        tensor("numeric_missing", torch.bool),
        tensor("categorical", torch.long),
        tensor("route_sequence", torch.long).clamp_min(0),
        pad,
        static_edge_features=tensor("static_edge_features", torch.float32),
        edge_train_support=tensor("edge_train_support", torch.float32),
        temporal_features=temporal,
        recent_history=tensor("recent_history", torch.float32),
        profile_history=tensor("profile_history", torch.float32),
        forecast_horizon_s=tensor("forecast_horizon_s", torch.float32),
        history_age_s=tensor("history_age_s", torch.float32),
        history_support=tensor("history_support", torch.float32),
    )
    target = tensor("targets", torch.float32)
    target_mask = tensor("target_masks", torch.bool)
    tail = tensor("tail_targets", torch.float32)
    tail_mask = tensor("tail_masks", torch.bool)
    availability = tensor("availability_targets", torch.float32)
    valid = ~pad
    targets = {
        "crawl_time_share": target[..., 0], "stop_time_share": target[..., 1],
        "speed_cv_bounded": target[..., 2], "acceleration_rms_bounded": target[..., 3],
        "rts_raw": target[..., 4], "lcs_raw": target[..., 5],
        "pace_sec_per_m": target[..., 6], "lcs_tail_event": tail[..., 0],
        "rts_tail_event": tail[..., 1], "availability": availability,
    }
    masks = {
        "crawl_target_valid": target_mask[..., 0] & valid,
        "stop_target_valid": target_mask[..., 1] & valid,
        "speed_cv_target_valid": target_mask[..., 2] & valid,
        "acceleration_rms_target_valid": target_mask[..., 3] & valid,
        "rts_target_valid": target_mask[..., 4] & tail_mask[..., 1] & valid,
        "lcs_target_valid": target_mask[..., 5] & tail_mask[..., 0] & valid,
        "pace_target_valid": target_mask[..., 6] & valid,
        "availability_valid": valid.unsqueeze(-1).expand_as(availability),
    }
    loss, components = rc_mstnet_v5_loss(
        output,
        targets,
        masks,
        tensor("supervision_weight", torch.float32),
        component_weights={
            "pace_distribution": 1.0, "crawl": 0.5, "stop_occurrence": 0.25,
            "stop_positive": 0.5, "speed_cv": 0.5, "acceleration": 0.5,
            "rts": 0.5, "lcs_consistency": 0.5, "lcs_tail": 0.25,
            "rts_tail": 0.25, "availability": 0.2,
        },
    )
    return loss, output, target, target_mask, pad, tensor("support_group_code", torch.long), components


def _validation_mae(model: RCMSTNetTransfer, paths: list[Path], batch_size: int, device: Any) -> dict[str, Any]:
    import torch

    prediction_keys = {
        "crawl": "crawl_share", "stop": "stop_share",
        "speed_cv": "speed_cv", "acceleration_rms": "acceleration_rms",
    }
    target_index = {"crawl": 0, "stop": 1, "speed_cv": 2, "acceleration_rms": 3}
    absolute = {name: 0.0 for name in CORE_TRANSFER_TARGETS}
    count = {name: 0 for name in CORE_TRANSFER_TARGETS}
    low_absolute = {name: 0.0 for name in CORE_TRANSFER_TARGETS}
    low_count = {name: 0 for name in CORE_TRANSFER_TARGETS}
    pace_absolute = 0.0
    pace_count = 0
    finite = True
    model.eval()
    with torch.no_grad():
        for path in paths:
            data = _load_npz(path)
            for index in _batch_indices(len(data["numeric"]), batch_size, shuffle=False):
                _, output, target, masks, pad, support_group, _ = _forward_loss(model, data, index, device)
                finite &= all(bool(torch.isfinite(value).all()) for value in output.values() if value.is_floating_point())
                for name in CORE_TRANSFER_TARGETS:
                    valid = masks[..., target_index[name]] & ~pad
                    error = torch.abs(output[prediction_keys[name]] - target[..., target_index[name]])
                    absolute[name] += float(error[valid].sum())
                    count[name] += int(valid.sum())
                    low = valid & support_group.eq(1)
                    low_absolute[name] += float(error[low].sum())
                    low_count[name] += int(low.sum())
                pace_valid = masks[..., 6] & ~pad
                pace_absolute += float(torch.abs(output["pace_pred_p50"] - target[..., 6])[pace_valid].sum())
                pace_count += int(pace_valid.sum())
    return {
        "all_outputs_finite": finite,
        "core_mae": {name: absolute[name] / max(count[name], 1) for name in CORE_TRANSFER_TARGETS},
        "low_support_core_mae": {
            name: low_absolute[name] / max(low_count[name], 1) for name in CORE_TRANSFER_TARGETS
        },
        "pace_p50_mae": pace_absolute / max(pace_count, 1),
    }


def train_transfer_from_shards(
    *,
    protocol_id: str,
    model_id: str,
    tensor_root: str | Path,
    feature_artifact_path: str | Path,
    source_checkpoint_path: str | Path,
    source_model_id: str,
    static_feature_count: int,
    support_tau: float,
    backbone_kwargs: dict[str, Any],
    v5_1_core_mae: Mapping[str, float],
    v5_1_pace_p50_mae: float,
    output_root: str | Path,
    new_branch_lr: float,
    backbone_lr_ratio: float,
    shared_freeze_epochs: int,
    maximum_epochs: int,
    batch_size: int,
    temporal_leakage_count: int,
) -> dict[str, Any]:
    """Execute M1-M5 on protocol-bound augmented NPZ shards."""
    import torch

    modes = {
        "M1": ("identity", "none"), "M2": ("structure_only", "none"),
        "M3": ("concat", "none"), "M4": ("support_aware", "none"),
        "M5": ("support_aware", "zero_shot"),
    }
    if model_id not in modes:
        raise Stage2V52ContractError("deep training model must be M1-M5")
    spatial_mode, temporal_mode = modes[model_id]
    artifact = json.loads(Path(feature_artifact_path).read_text(encoding="utf-8"))
    numeric_feature_count = len(artifact["numeric_features"])
    model, source = initialized_transfer_model(
        feature_artifact_path=feature_artifact_path,
        checkpoint_path=source_checkpoint_path,
        source_model_id=source_model_id,
        numeric_feature_count=numeric_feature_count,
        static_feature_count=static_feature_count,
        support_tau=support_tau,
        spatial_mode=spatial_mode,
        temporal_mode=temporal_mode,
        backbone_kwargs=backbone_kwargs,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_paths = _protocol_shards(Path(tensor_root), protocol_id, "train")
    validation_paths = _protocol_shards(Path(tensor_root), protocol_id, "validation")
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    if model_id == "M1":
        maximum_epochs = 0
    optimizer = torch.optim.AdamW(
        model.optimizer_parameter_groups(
            new_branch_lr=new_branch_lr, backbone_lr_ratio=backbone_lr_ratio
        ),
        weight_decay=1.0e-4,
    ) if maximum_epochs else None
    for epoch in range(maximum_epochs):
        model.set_shared_backbone_frozen(epoch < shared_freeze_epochs)
        model.train()
        for path in train_paths:
            data = _load_npz(path)
            for index in _batch_indices(len(data["numeric"]), batch_size, shuffle=True):
                optimizer.zero_grad(set_to_none=True)
                loss, *_ = _forward_loss(model, data, index, device)
                if not torch.isfinite(loss):
                    raise Stage2V52ContractError("non-finite transfer training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        metrics = _validation_mae(model, validation_paths, batch_size, device)
        candidate = checkpoint_candidate(
            checkpoint_id=f"epoch_{epoch + 1:03d}",
            core_mae=metrics["core_mae"],
            low_support_core_mae=metrics["low_support_core_mae"],
            v5_1_core_mae=v5_1_core_mae,
            all_outputs_finite=metrics["all_outputs_finite"],
            temporal_leakage_count=temporal_leakage_count,
            pace_p50_mae=metrics["pace_p50_mae"],
            v5_1_pace_p50_mae=v5_1_pace_p50_mae,
        )
        candidates.append(candidate)
        checkpoint_path = output / f"{candidate['checkpoint_id']}.pt"
        checkpoint_temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
        torch.save(
            {"model_state_dict": model.state_dict(), "candidate": candidate},
            checkpoint_temporary,
        )
        os.replace(checkpoint_temporary, checkpoint_path)
    if model_id == "M1":
        metrics = _validation_mae(model, validation_paths, batch_size, device)
        candidates.append(checkpoint_candidate(
            checkpoint_id="frozen_v5_1",
            core_mae=metrics["core_mae"], low_support_core_mae=metrics["low_support_core_mae"],
            v5_1_core_mae=v5_1_core_mae, all_outputs_finite=metrics["all_outputs_finite"],
            temporal_leakage_count=temporal_leakage_count, pace_p50_mae=metrics["pace_p50_mae"],
            v5_1_pace_p50_mae=v5_1_pace_p50_mae,
        ))
    selection = select_micro_first_checkpoint(candidates)
    manifest = {
        "schema_version": "stage2_v5_2_training.1",
        "status": "PASS" if selection["status"] == "MICRO_FIRST_CHECKPOINT_SELECTED" else "FAIL",
        "protocol_id": protocol_id,
        "protocol_hash": get_protocol(protocol_id).digest,
        "model_id": model_id,
        "source": source,
        "initialization_policy": source["initialization_policy"],
        "backbone_lr": new_branch_lr * backbone_lr_ratio,
        "new_branch_lr": new_branch_lr,
        "shared_freeze_epochs": shared_freeze_epochs,
        "checkpoint_selection": selection,
        "candidates": candidates,
    }
    write_training_manifest(output / "model_manifest.json", manifest)
    return manifest
