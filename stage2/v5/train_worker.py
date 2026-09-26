"""GPU worker for overlap-weighted RC-MSTNet v5 training and prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import load_inherited_payload
from .models.losses import rc_mstnet_v5_loss
from .models.rc_mstnet_v5 import RCMSTNetV5


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _training_config(path: Path) -> dict[str, Any]:
    return load_inherited_payload(path)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(_canonical(payload))
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pt", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


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


def _shards(root: Path, split: str, dates: list[str]) -> list[Path]:
    paths: list[Path] = []
    for date in dates:
        manifest = _json(root / f"split={split}" / f"date={date}" / "manifest.json")
        paths.extend(root / item["path"] for item in manifest["files"])
    return paths


def _batches(length: int, size: int, shuffle: bool):
    index = np.arange(length)
    if shuffle:
        np.random.shuffle(index)
    for start in range(0, length, size):
        yield index[start : start + size]


def _load_shard(path: Path) -> dict[str, np.ndarray]:
    """Decompress every NPZ member once, never once per mini-batch."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _tensor(values: np.ndarray, indices: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(values[indices], dtype=dtype, device=device)


def _forward_loss(model: RCMSTNetV5, data: Any, indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    pad = _tensor(data["pad_mask"], indices, device, torch.bool)
    outputs = model(
        _tensor(data["numeric"], indices, device, torch.float32),
        _tensor(data["numeric_missing"], indices, device, torch.bool),
        _tensor(data["categorical"], indices, device, torch.long),
        _tensor(data["route_sequence"], indices, device, torch.long).clamp_min(0),
        pad,
        recent_history=_tensor(data["recent_history"], indices, device, torch.float32),
        profile_history=_tensor(data["profile_history"], indices, device, torch.float32),
        forecast_horizon_s=_tensor(data["forecast_horizon_s"], indices, device, torch.float32),
        history_age_s=_tensor(data["history_age_s"], indices, device, torch.float32),
        history_support=_tensor(data["history_support"], indices, device, torch.float32),
    )
    target = _tensor(data["targets"], indices, device, torch.float32)
    target_mask = _tensor(data["target_masks"], indices, device, torch.bool)
    tail = _tensor(data["tail_targets"], indices, device, torch.float32)
    tail_mask = _tensor(data["tail_masks"], indices, device, torch.bool)
    availability = _tensor(data["availability_targets"], indices, device, torch.float32)
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
    supervision = _tensor(data["supervision_weight"], indices, device, torch.float32)
    loss, components = rc_mstnet_v5_loss(
        outputs,
        targets,
        masks,
        supervision,
        component_weights={
            "pace_distribution": 1.0,
            "crawl": 0.5,
            "stop_occurrence": 0.25,
            "stop_positive": 0.5,
            "speed_cv": 0.5,
            "acceleration": 0.5,
            "rts": 0.5,
            "lcs_consistency": 0.5,
            "lcs_tail": 0.25,
            "rts_tail": 0.25,
            "availability": 0.2,
        },
    )
    return loss, outputs, components


def _evaluate(model: RCMSTNetV5, shards: list[Path], batch_size: int, device: torch.device, mixed: bool) -> float:
    return _evaluate_metrics(model, shards, batch_size, device, mixed)[0]


def _evaluate_metrics(
    model: RCMSTNetV5,
    shards: list[Path],
    batch_size: int,
    device: torch.device,
    mixed: bool,
) -> tuple[float, float]:
    model.eval()
    total = 0.0
    distribution_total = 0.0
    chunks = 0
    with torch.no_grad():
        for path in shards:
            data = _load_shard(path)
            for indices in _batches(len(data["numeric"]), batch_size, False):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=mixed):
                    loss, _, components = _forward_loss(model, data, indices, device)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite validation loss in {path.name}")
                total += float(loss.item()) * len(indices)
                distribution_total += float(components["pace_distribution"].item()) * len(indices)
                chunks += len(indices)
    return total / max(chunks, 1), distribution_total / max(chunks, 1)


def _weighted_unique(values: np.ndarray, inverse: np.ndarray, weights: np.ndarray, count: int) -> np.ndarray:
    usable = np.isfinite(values)
    numerator = np.bincount(
        inverse, weights=np.where(usable, values * weights, 0.0), minlength=count
    )
    denominator = np.bincount(
        inverse, weights=np.where(usable, weights, 0.0), minlength=count
    )
    return np.divide(
        numerator, denominator, out=np.full(count, np.nan), where=denominator > 0
    )


def _checkpoint_arrays(chunks: Path) -> dict[str, np.ndarray]:
    names = (
        "pace_log_mu",
        "pace_log_scale",
        "pace_pred_mean",
        "pace_pred_p50",
        "pace_pred_p90",
        "pace_pred_p95",
        "allocated_distance_m",
    )
    storage: dict[str, list[np.ndarray]] = {
        name: [] for name in (*names, "order_id", "traversal_id", "weight", "truth", "target_valid")
    }
    for path in sorted(chunks.glob("split=*/date=*/shard-*.npz")):
        with np.load(path, allow_pickle=False) as data:
            valid = ~data["pad_mask"]
            storage["order_id"].append(
                np.broadcast_to(data["order_id"][:, None], valid.shape)[valid].astype(str)
            )
            storage["traversal_id"].append(data["traversal_id"][valid].astype(np.int64))
            storage["weight"].append(data["supervision_weight"][valid].astype(np.float64))
            storage["truth"].append(data["targets"][..., 6][valid].astype(np.float64))
            storage["target_valid"].append(data["target_masks"][..., 6][valid].astype(bool))
            for field in names:
                storage[field].append(data[field][valid].astype(np.float64))
    if not storage["order_id"]:
        raise RuntimeError("checkpoint candidate produced no validation rows")
    combined = {name: np.concatenate(parts) for name, parts in storage.items()}
    identity = np.rec.fromarrays(
        [combined["order_id"], combined["traversal_id"]],
        names=("order_id", "traversal_id"),
    )
    unique, inverse = np.unique(identity, return_inverse=True)
    count = len(unique)
    weights = combined["weight"]
    total_weight = np.bincount(inverse, weights=weights, minlength=count)
    if not np.allclose(total_weight, 1.0, atol=1.0e-5, rtol=0):
        raise RuntimeError("checkpoint overlap weights do not sum to one")
    merged = {
        "order_id": unique["order_id"].astype(str),
        "traversal_id": unique["traversal_id"].astype(np.int64),
    }
    for field in names:
        merged[field] = _weighted_unique(combined[field], inverse, weights, count)
    valid_weight = weights * combined["target_valid"].astype(float)
    merged["truth"] = _weighted_unique(combined["truth"], inverse, valid_weight, count)
    merged["target_valid"] = (
        np.bincount(inverse, weights=combined["target_valid"].astype(float), minlength=count) > 0
    )
    return merged


def _route_scenario_smoke(arrays: dict[str, np.ndarray], maximum_route_quantile_s: float) -> bool:
    _, inverse = np.unique(arrays["order_id"], return_inverse=True)
    count = int(inverse.max()) + 1 if len(inverse) else 0
    routes = np.column_stack(
        [
            np.bincount(
                inverse,
                weights=arrays[field] * arrays["allocated_distance_m"],
                minlength=count,
            )
            for field in ("pace_pred_p50", "pace_pred_p90", "pace_pred_p95")
        ]
    )
    return bool(
        len(routes)
        and np.isfinite(routes).all()
        and np.all(routes[:, 0] <= routes[:, 1])
        and np.all(routes[:, 1] <= routes[:, 2])
        and float(routes[:, 2].max()) <= maximum_route_quantile_s
    )


def _select_checkpoint(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in candidates if row["hard_gate_status"] == "PASS"]
    if not eligible:
        return {
            "status": "NO_STABLE_CHECKPOINT",
            "selected_checkpoint_id": None,
            "candidate_count": len(candidates),
            "stable_candidate_count": 0,
        }
    selected = min(
        eligible,
        key=lambda row: (
            row["validation_p50_mae"],
            row.get("validation_distribution_loss", row["validation_distribution_nll"]),
            row["validation_mean_mae"],
            row["validation_quantile_coverage_error"],
            row["checkpoint_id"],
        ),
    )
    return {
        "status": "STABLE_CHECKPOINT_SELECTED",
        "selected_checkpoint_id": selected["checkpoint_id"],
        "candidate_count": len(candidates),
        "stable_candidate_count": len(eligible),
        "selected_metrics": selected,
    }


def _evaluate_checkpoint_candidate(
    model: RCMSTNetV5,
    *,
    checkpoint_id: str,
    distribution_loss: float,
    validation_shards: list[Path],
    tensor_root: Path,
    batch_size: int,
    device: torch.device,
    mixed: bool,
    thresholds: dict[str, float],
    temporary_parent: Path,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="checkpoint-selection-", dir=temporary_parent) as name:
        candidate_root = Path(name)
        chunks = candidate_root / "chunks"
        _predict(
            model,
            validation_shards,
            tensor_root,
            chunks,
            batch_size,
            device,
            mixed,
            "validation_prediction_manifest.json",
        )
        arrays = _checkpoint_arrays(chunks)
        valid = (
            arrays["target_valid"]
            & np.isfinite(arrays["truth"])
            & (arrays["truth"] > 0)
        )
        if not valid.any():
            raise RuntimeError("checkpoint validation has no valid pace targets")
        finite_fields = (
            "pace_log_mu", "pace_log_scale", "pace_pred_mean",
            "pace_pred_p50", "pace_pred_p90", "pace_pred_p95",
        )
        finite = all(np.isfinite(arrays[field]).all() for field in finite_fields)
        p50 = arrays["pace_pred_p50"]
        p90 = arrays["pace_pred_p90"]
        p95 = arrays["pace_pred_p95"]
        mean = arrays["pace_pred_mean"]
        truth = arrays["truth"]
        ratio = np.divide(mean, p50, out=np.full_like(mean, np.inf), where=p50 > 0)
        absolute_error = np.abs(mean[valid] - truth[valid])
        squared_error = np.square(mean[valid] - truth[valid])
        mae_total = float(absolute_error.sum())
        rmse_total = float(squared_error.sum())
        coverage_error = max(
            abs(float((truth[valid] <= p90[valid]).mean()) - 0.9),
            abs(float((truth[valid] <= p95[valid]).mean()) - 0.95),
        )
        smoke = _route_scenario_smoke(
            arrays, float(thresholds["maximum_route_cvar95_s"])
        )
        hard_checks = {
            "all_outputs_finite": finite,
            "quantiles_monotonic": bool(
                np.all(p50 > 0) and np.all(p50 <= p90) and np.all(p90 <= p95)
            ),
            "pace_mean_stable": bool(
                float(mean.max()) <= thresholds["maximum_pace_mean_s_per_m"]
                and float(np.quantile(mean, 0.999))
                <= thresholds["maximum_p99_9_pace_mean_s_per_m"]
            ),
            "mean_to_p50_stable": bool(
                float(ratio.max()) <= thresholds["maximum_mean_to_p50_ratio"]
            ),
            "route_scenario_smoke_stable": smoke,
        }
        return {
            "checkpoint_id": checkpoint_id,
            "hard_gate_status": "PASS" if all(hard_checks.values()) else "FAIL",
            "hard_checks": hard_checks,
            "diagnostics": {
                "maximum_row_mae_contribution_share": float(absolute_error.max()) / max(mae_total, np.finfo(float).tiny),
                "maximum_row_rmse_contribution_share": float(squared_error.max()) / max(rmse_total, np.finfo(float).tiny),
                "single_row_contribution_threshold_pass": bool(
                    float(absolute_error.max()) / max(mae_total, np.finfo(float).tiny)
                    <= thresholds["maximum_single_row_mae_contribution_share"]
                    and float(squared_error.max()) / max(rmse_total, np.finfo(float).tiny)
                    <= thresholds["maximum_single_row_rmse_contribution_share"]
                ),
            },
            "validation_p50_mae": float(np.mean(np.abs(p50[valid] - truth[valid]))),
            "validation_distribution_nll": float(distribution_loss),
            "validation_distribution_loss": float(distribution_loss),
            "validation_mean_mae": float(np.mean(absolute_error)),
            "validation_quantile_coverage_error": float(coverage_error),
            "unique_traversal_count": int(len(p50)),
        }


def _predict(model: RCMSTNetV5, shards: list[Path], tensor_root: Path, output_root: Path, batch_size: int, device: torch.device, mixed: bool, name: str) -> dict[str, Any]:
    model.eval()
    files: list[dict[str, Any]] = []
    mixed_precision_fallback_batch_count = 0
    with torch.no_grad():
        for path in shards:
            relative = path.relative_to(tensor_root)
            output_path = output_root / relative
            data = _load_shard(path)
            storage = {key: [] for key in (
                "pred_crawl_time_share", "pred_stop_time_share", "pred_speed_cv_bounded",
                "pred_acceleration_rms_bounded", "pred_rts_raw", "pred_lcs_raw",
                "lcs_tail_score", "rts_tail_score", "pace_log_mu", "pace_log_scale",
                "pace_pred_mean", "pace_pred_p50", "pace_pred_p90", "pace_pred_p95",
                "availability_probability", "history_recent_gate",
                "stop_occurrence_probability", "stop_positive_share",
            )}
            for indices in _batches(len(data["numeric"]), batch_size, False):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=mixed):
                    _, output, _ = _forward_loss(model, data, indices, device)
                # Long out-of-period feature ages can exceed the finite fp16
                # range after Train-only normalization.  Preserve AMP for the
                # normal path, but rerun only a non-finite batch in fp32.  A
                # silent NaN or clipping the frozen benchmark would both be
                # scientifically invalid.
                if any(
                    isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
                    for value in output.values()
                ):
                    with torch.autocast(device_type=device.type, enabled=False):
                        _, output, _ = _forward_loss(model, data, indices, device)
                    mixed_precision_fallback_batch_count += 1
                    if any(
                        isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
                        for value in output.values()
                    ):
                        raise RuntimeError(f"non-finite prediction after fp32 fallback in {path.name}")
                mapped = {
                        "pred_crawl_time_share": output["crawl_share"],
                        "pred_stop_time_share": output["stop_share"],
                        "pred_speed_cv_bounded": output["speed_cv"],
                        "pred_acceleration_rms_bounded": output["acceleration_rms"],
                        "pred_rts_raw": output["rts_raw"],
                        "pred_lcs_raw": output["lcs_reconstructed_raw"],
                        "lcs_tail_score": torch.sigmoid(output["lcs_tail_logit"]),
                        "rts_tail_score": torch.sigmoid(output["rts_tail_logit"]),
                        "pace_log_mu": output["pace_log_mu"], "pace_log_scale": output["pace_log_scale"],
                        "pace_pred_mean": output["pace_pred_mean"], "pace_pred_p50": output["pace_pred_p50"],
                        "pace_pred_p90": output["pace_pred_p90"], "pace_pred_p95": output["pace_pred_p95"],
                        "availability_probability": torch.sigmoid(output["availability_logits"]),
                        "history_recent_gate": output["history_recent_gate"],
                        "stop_occurrence_probability": torch.sigmoid(output["stop_presence_logit"]),
                        "stop_positive_share": output["stop_positive_share"],
                }
                for key, value in mapped.items():
                    storage[key].append(value.float().cpu().numpy())
            payload = {key: np.concatenate(values).astype(np.float32) for key, values in storage.items()}
            for key in (
                "order_id", "traversal_id", "route_sequence", "pad_mask", "targets",
                "target_masks", "tail_targets", "tail_masks", "allocated_distance_m",
                "overlap_supervision_count", "supervision_weight",
                "categorical",
            ):
                payload[key] = data[key]
            _atomic_npz(output_path, payload)
            files.append({"path": relative.as_posix(), "chunk_count": len(payload["order_id"]), "sha256": _sha256(output_path)})
    manifest = {
        "schema_version": "stage2_v5_chunk_predictions.1",
        "status": "PASS",
        "chunk_count": int(sum(item["chunk_count"] for item in files)),
        "mixed_precision_fallback_batch_count": mixed_precision_fallback_batch_count,
        "files": files,
    }
    _atomic_json(output_root / name, manifest)
    return manifest


def train(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = _training_config(args.config)
    deep = config["deep"]
    seed = int(deep["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tensor_root = args.tensor_root
    artifacts = _json(tensor_root / "feature_artifacts.json")
    split = config["split"]
    train_shards = _shards(tensor_root, "train", split["train_dates"])
    validation_shards = _shards(tensor_root, "validation_model", split["validation_model_dates"])
    calibration_shards = _shards(tensor_root, "calibration", split["calibration_dates"])
    evaluation_shards = _shards(tensor_root, "evaluation", split.get("evaluation_dates", []))
    legacy_shards = _shards(tensor_root, "legacy", split.get("legacy_test_dates", []))
    categorical_sizes = tuple(len(artifacts["vocabularies"][name]["token_to_index"]) for name in ("edge", "highway", "time_bin", "position_bucket", "route_length_bucket"))
    model_config = {
        "numeric_feature_count": len(artifacts["numeric_features"]),
        "categorical_sizes": categorical_sizes,
        "hidden_dim": int(deep["hidden_dim"]), "categorical_embedding_dim": int(deep["categorical_embedding_dim"]),
        "transformer_layers": int(deep["transformer_layers"]), "attention_heads": int(deep["attention_heads"]),
        "dropout": float(deep["dropout"]),
        "minimum_log_scale": float(config["distribution"]["minimum_log_scale"]),
        "maximum_log_scale": float(config["distribution"]["maximum_log_scale"]),
        "distribution_family": config["distribution"].get("family", "lognormal"),
        "maximum_log_p50": float(config["distribution"].get("maximum_log_p50", np.log(5.0))),
        "maximum_log_p90_p50_ratio": float(
            config["distribution"].get("maximum_log_p90_p50_ratio", np.log(10.0))
        ),
        "maximum_log_p95_p90_ratio": float(
            config["distribution"].get("maximum_log_p95_p90_ratio", np.log(3.0))
        ),
        "history_mode": args.history_mode,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed = bool(deep.get("mixed_precision", False)) and device.type == "cuda"
    model = RCMSTNetV5(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(deep["learning_rate"]), weight_decay=float(deep["weight_decay"]))
    scaler = torch.cuda.amp.GradScaler(enabled=mixed)
    checkpoint = args.output / "best_model.pt"
    manifest_path = args.output / "model_manifest.json"
    args.output.mkdir(parents=True, exist_ok=True)
    if args.predict_only:
        if not checkpoint.is_file() or not manifest_path.is_file():
            raise RuntimeError("predict-only requires a completed checkpoint and manifest")
        saved = torch.load(checkpoint, map_location=device)
        model.load_state_dict(saved["model_state_dict"], strict=False)
        _predict(model, validation_shards, tensor_root, args.prediction_root, int(deep["batch_size"]), device, mixed, "validation_prediction_manifest.json")
        _predict(model, calibration_shards, tensor_root, args.prediction_root, int(deep["batch_size"]), device, mixed, "calibration_prediction_manifest.json")
        if evaluation_shards:
            _predict(model, evaluation_shards, tensor_root, args.prediction_root, int(deep["batch_size"]), device, mixed, "evaluation_prediction_manifest.json")
        if legacy_shards:
            _predict(model, legacy_shards, tensor_root, args.prediction_root, int(deep["batch_size"]), device, mixed, "legacy_prediction_manifest.json")
        return _json(manifest_path)
    if args.resume and checkpoint.is_file() and manifest_path.is_file():
        manifest = _json(manifest_path)
        if manifest.get("status") == "PASS":
            return manifest
    if (checkpoint.exists() or manifest_path.exists()) and not args.force:
        raise RuntimeError("model output exists; use --resume or --force")
    best = float("inf")
    best_epoch = -1
    patience = 0
    history: list[dict[str, Any]] = []
    checkpoint_candidates: list[dict[str, Any]] = []
    stable_selection = config.get("checkpoint_selection", {}).get("mode") == "stable_unique_traversal"
    stability_thresholds = config.get("stability_thresholds", {})
    batch_size = int(deep["batch_size"])
    for epoch in range(int(deep["maximum_epochs"])):
        epoch_started = time.perf_counter()
        model.train()
        random.shuffle(train_shards)
        total = 0.0
        chunks = 0
        for path in train_shards:
            data = _load_shard(path)
            for indices in _batches(len(data["numeric"]), batch_size, True):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=mixed):
                    loss, _, _ = _forward_loss(model, data, indices, device)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite training loss in {path.name}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(deep["gradient_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
                total += float(loss.item()) * len(indices)
                chunks += len(indices)
        validation, distribution_loss = _evaluate_metrics(
            model, validation_shards, batch_size, device, mixed
        )
        record = {"epoch": epoch, "train_loss": total / max(chunks, 1), "validation_loss": validation, "runtime_s": time.perf_counter() - epoch_started}
        selected_changed = False
        if stable_selection:
            candidate = _evaluate_checkpoint_candidate(
                model,
                checkpoint_id=f"epoch_{epoch:03d}",
                distribution_loss=distribution_loss,
                validation_shards=validation_shards,
                tensor_root=tensor_root,
                batch_size=batch_size,
                device=device,
                mixed=mixed,
                thresholds=stability_thresholds,
                temporary_parent=args.output,
            )
            checkpoint_candidates.append(candidate)
            selection = _select_checkpoint(checkpoint_candidates)
            record["checkpoint_candidate"] = candidate
            record["checkpoint_selection"] = selection
            selected_changed = selection["selected_checkpoint_id"] == candidate["checkpoint_id"]
        history.append(record)
        print(json.dumps(record), flush=True)
        improved = selected_changed if stable_selection else validation < best - 1e-6
        if improved:
            best = (
                float(record["checkpoint_candidate"]["validation_p50_mae"])
                if stable_selection
                else validation
            )
            best_epoch = epoch
            patience = 0
            _atomic_checkpoint(
                checkpoint,
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "best_epoch": best_epoch,
                    "best_validation_loss": validation,
                    "checkpoint_selection": record.get("checkpoint_selection"),
                },
            )
        else:
            patience += 1
            if patience >= int(deep["early_stopping_patience"]):
                break
    if not checkpoint.is_file():
        raise RuntimeError("training completed without a stable checkpoint")
    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    _predict(model, validation_shards, tensor_root, args.prediction_root, batch_size, device, mixed, "validation_prediction_manifest.json")
    _predict(model, calibration_shards, tensor_root, args.prediction_root, batch_size, device, mixed, "calibration_prediction_manifest.json")
    if evaluation_shards:
        _predict(model, evaluation_shards, tensor_root, args.prediction_root, batch_size, device, mixed, "evaluation_prediction_manifest.json")
    if legacy_shards:
        _predict(model, legacy_shards, tensor_root, args.prediction_root, batch_size, device, mixed, "legacy_prediction_manifest.json")
    checkpoint_sha = _sha256(checkpoint)
    model_id = hashlib.sha256(_canonical({"checkpoint_sha256": checkpoint_sha, "config_sha256": hashlib.sha256(_canonical(config)).hexdigest(), "artifact_sha256": _sha256(tensor_root / "feature_artifacts.json")})).hexdigest()
    selected_record = history[best_epoch]
    manifest = {
        "schema_version": "stage2_v5_rc_mstnet.1", "status": "PASS", "model_id": model_id,
        "checkpoint_sha256": checkpoint_sha, "fit_dates": split["train_dates"],
        "validation_dates": split["validation_model_dates"], "calibration_prediction_dates": split["calibration_dates"],
        "evaluation_prediction_dates": split.get("evaluation_dates", []),
        "legacy_prediction_dates": split.get("legacy_test_dates", []),
        "device": str(device), "mixed_precision": mixed,
        "history_mode": args.history_mode,
        "distribution_family": model.distribution_family,
        "best_epoch": best_epoch,
        "best_validation_loss": float(selected_record["validation_loss"]),
        "best_validation_p50_mae": (
            float(selected_record["checkpoint_candidate"]["validation_p50_mae"])
            if stable_selection
            else None
        ),
        "training_history": history,
        "checkpoint_selection_mode": "stable_unique_traversal" if stable_selection else "legacy_validation_loss",
        "checkpoint_candidates": checkpoint_candidates,
        "runtime_s": time.perf_counter() - started,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("stage2/config/stage2_v5.json"))
    parser.add_argument("--tensor-root", type=Path, default=Path("stage2/output_v5/tensor_shards"))
    parser.add_argument("--output", type=Path, default=Path("stage2/output_v5/deep_model"))
    parser.add_argument("--prediction-root", type=Path, default=Path("stage2/output_v5/deep_predictions"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument(
        "--history-mode",
        choices=("gate", "ordinary_concatenation", "without_recent", "without_profile"),
        default="gate",
    )
    manifest = train(parser.parse_args())
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
