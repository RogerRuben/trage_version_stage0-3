"""Dependency-light GPU worker for RC-MSTNet v4 training and prediction."""

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

from stage2.v4.models.losses import rc_mstnet_v4_loss
from stage2.v4.models.rc_mstnet_v4 import RCMSTNetV4


MODEL_SCHEMA_VERSION = "stage2_v4_rc_mstnet.1"
PREDICTION_SCHEMA_VERSION = "stage2_v4_chunk_predictions.1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        suffix=".npz",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_checkpoint(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        suffix=".pt",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _shards(root: Path, split: str, dates: list[str]) -> list[Path]:
    paths: list[Path] = []
    for date in dates:
        manifest_path = root / f"split={split}" / f"date={date}" / "manifest.json"
        manifest = _read_json(manifest_path)
        if manifest.get("engineering_status") != "PASS":
            raise RuntimeError(f"tensor day is not PASS: {date}")
        paths.extend(
            manifest_path.parent / item["name"] for item in manifest["shards"]
        )
    return paths


def _batch_indices(length: int, batch_size: int, shuffle: bool) -> list[np.ndarray]:
    indices = np.arange(length)
    if shuffle:
        np.random.shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, length, batch_size)]


def _tensor(value: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


def _forward_loss(
    model: RCMSTNetV4,
    data: Any,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    numeric = _tensor(data["numeric"][indices], device, torch.float32)
    numeric_missing = _tensor(
        data["numeric_missing"][indices],
        device,
        torch.bool,
    )
    categorical = _tensor(data["categorical"][indices], device, torch.long)
    route_sequence = _tensor(data["route_sequence"][indices], device, torch.long)
    pad_mask = _tensor(data["pad_mask"][indices], device, torch.bool)
    outputs = model(
        numeric,
        numeric_missing,
        categorical,
        route_sequence.clamp_min(0),
        pad_mask,
    )
    target = _tensor(data["targets"][indices], device, torch.float32)
    target_mask = _tensor(data["target_masks"][indices], device, torch.bool)
    tail = _tensor(data["tail_targets"][indices], device, torch.float32)
    tail_mask = _tensor(data["tail_masks"][indices], device, torch.bool)
    targets = {
        "crawl_time_share": target[..., 0],
        "stop_time_share": target[..., 1],
        "speed_cv_bounded": target[..., 2],
        "acceleration_rms_bounded": target[..., 3],
        "rts_raw": target[..., 4],
        "lcs_raw": target[..., 5],
        "lcs_tail_event": tail[..., 0],
        "rts_tail_event": tail[..., 1],
    }
    masks = {
        "crawl_target_valid": target_mask[..., 0] & ~pad_mask,
        "stop_target_valid": target_mask[..., 1] & ~pad_mask,
        "speed_cv_target_valid": target_mask[..., 2] & ~pad_mask,
        "acceleration_rms_target_valid": target_mask[..., 3] & ~pad_mask,
        "rts_target_valid": target_mask[..., 4] & ~pad_mask,
        "lcs_target_valid": target_mask[..., 5] & ~pad_mask,
    }
    masks["lcs_target_valid"] &= tail_mask[..., 0] | target_mask[..., 5]
    masks["rts_target_valid"] &= tail_mask[..., 1] | target_mask[..., 4]
    loss, components = rc_mstnet_v4_loss(outputs, targets, masks)
    return loss, outputs, components


def _evaluate_loss(
    model: RCMSTNetV4,
    shards: list[Path],
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for path in shards:
            with np.load(path, allow_pickle=False) as data:
                for indices in _batch_indices(len(data["numeric"]), batch_size, False):
                    loss, _outputs, _components = _forward_loss(
                        model,
                        data,
                        indices,
                        device,
                    )
                    total += float(loss.item()) * len(indices)
                    count += len(indices)
    return total / max(count, 1)


def _predict_shards(
    model: RCMSTNetV4,
    input_shards: list[Path],
    *,
    tensor_root: Path,
    output_root: Path,
    batch_size: int,
    device: torch.device,
    manifest_name: str,
) -> dict[str, Any]:
    model.eval()
    identities: list[dict[str, Any]] = []
    with torch.no_grad():
        for input_path in input_shards:
            relative = input_path.relative_to(tensor_root)
            output_path = output_root / relative
            with np.load(input_path, allow_pickle=False) as data:
                predictions: dict[str, list[np.ndarray]] = {
                    name: []
                    for name in (
                        "pred_crawl_time_share",
                        "pred_stop_time_share",
                        "pred_speed_cv_bounded",
                        "pred_acceleration_rms_bounded",
                        "pred_rts_raw",
                        "pred_lcs_raw",
                        "lcs_tail_score",
                        "rts_tail_score",
                        "lcs_log_scale",
                        "rts_log_scale",
                    )
                }
                for indices in _batch_indices(len(data["numeric"]), batch_size, False):
                    _loss, outputs, _components = _forward_loss(
                        model,
                        data,
                        indices,
                        device,
                    )
                    mapped = {
                        "pred_crawl_time_share": outputs["crawl_share"],
                        "pred_stop_time_share": outputs["stop_share"],
                        "pred_speed_cv_bounded": outputs["speed_cv"],
                        "pred_acceleration_rms_bounded": outputs["acceleration_rms"],
                        "pred_rts_raw": outputs["rts_raw"],
                        "pred_lcs_raw": outputs["lcs_reconstructed_raw"],
                        "lcs_tail_score": torch.sigmoid(outputs["lcs_tail_logit"]),
                        "rts_tail_score": torch.sigmoid(outputs["rts_tail_logit"]),
                        "lcs_log_scale": outputs["lcs_log_scale"],
                        "rts_log_scale": outputs["rts_log_scale"],
                    }
                    for name, value in mapped.items():
                        predictions[name].append(value.cpu().numpy().astype(np.float32))
                payload = {
                    name: np.concatenate(values, axis=0)
                    for name, values in predictions.items()
                }
                for name in (
                    "order_id",
                    "traversal_id",
                    "route_sequence",
                    "pad_mask",
                    "targets",
                    "target_masks",
                    "tail_targets",
                    "tail_masks",
                    "aggregation_weights",
                ):
                    payload[name] = data[name]
            _atomic_npz(output_path, payload)
            identities.append(
                {
                    "path": relative.as_posix(),
                    "sha256": _sha256(output_path),
                    "chunk_count": int(len(payload["order_id"])),
                }
            )
    manifest = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "engineering_status": "PASS",
        "shard_count": len(identities),
        "chunk_count": sum(item["chunk_count"] for item in identities),
        "shards": identities,
    }
    _atomic_json(output_root / manifest_name, manifest)
    return manifest


def train(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = _read_json(args.config)
    deep = config["deep"]
    split = config["split"]
    seed = int(deep["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tensor_root = Path(args.tensor_root)
    output = Path(args.output)
    prediction_root = Path(args.prediction_root)
    checkpoint = output / "best_model.pt"
    manifest_path = output / "model_manifest.json"
    if manifest_path.is_file() and checkpoint.is_file() and args.resume:
        manifest = _read_json(manifest_path)
        if (
            manifest.get("engineering_status") == "PASS"
            and manifest.get("stage2_config_sha256")
            == hashlib.sha256(_canonical_bytes(config)).hexdigest()
            and manifest.get("checkpoint_sha256") == _sha256(checkpoint)
        ):
            return manifest
    if (manifest_path.exists() or checkpoint.exists()) and not args.force:
        raise RuntimeError("deep model output exists; use --resume or --force")

    artifacts_path = tensor_root / "feature_artifacts.json"
    artifacts = _read_json(artifacts_path)
    train_shards = _shards(tensor_root, "train", split["train_dates"])
    validation_shards = _shards(
        tensor_root,
        "validation_model",
        split["validation_model_dates"],
    )
    calibration_shards = _shards(
        tensor_root,
        "calibration",
        split["calibration_dates"],
    )
    categorical_sizes = tuple(
        len(artifacts["vocabularies"][name]["token_to_index"])
        for name in (
            "edge",
            "highway",
            "time_bin",
            "position_bucket",
            "route_length_bucket",
        )
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RCMSTNetV4(
        numeric_feature_count=len(artifacts["numeric_features"]),
        categorical_sizes=categorical_sizes,
        hidden_dim=int(deep["hidden_dim"]),
        categorical_embedding_dim=int(deep["categorical_embedding_dim"]),
        transformer_layers=int(deep["transformer_layers"]),
        attention_heads=int(deep["attention_heads"]),
        dropout=float(deep["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(deep["learning_rate"]),
        weight_decay=float(deep["weight_decay"]),
    )
    batch_size = int(deep["batch_size"])
    best_loss = float("inf")
    best_epoch = -1
    patience = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(int(deep["maximum_epochs"])):
        model.train()
        random.shuffle(train_shards)
        total = 0.0
        count = 0
        for path in train_shards:
            with np.load(path, allow_pickle=False) as data:
                for indices in _batch_indices(len(data["numeric"]), batch_size, True):
                    optimizer.zero_grad(set_to_none=True)
                    loss, _outputs, _components = _forward_loss(
                        model,
                        data,
                        indices,
                        device,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(deep["gradient_clip_norm"]),
                    )
                    optimizer.step()
                    total += float(loss.item()) * len(indices)
                    count += len(indices)
        train_loss = total / max(count, 1)
        validation_loss = _evaluate_loss(
            model,
            validation_shards,
            batch_size,
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            ),
            flush=True,
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            patience = 0
            _atomic_checkpoint(
                checkpoint,
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "numeric_feature_count": len(artifacts["numeric_features"]),
                        "categorical_sizes": categorical_sizes,
                        "hidden_dim": int(deep["hidden_dim"]),
                        "categorical_embedding_dim": int(
                            deep["categorical_embedding_dim"]
                        ),
                        "transformer_layers": int(deep["transformer_layers"]),
                        "attention_heads": int(deep["attention_heads"]),
                        "dropout": float(deep["dropout"]),
                    },
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                },
            )
        else:
            patience += 1
            if patience >= int(deep["early_stopping_patience"]):
                break

    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    _predict_shards(
        model,
        validation_shards,
        tensor_root=tensor_root,
        output_root=prediction_root,
        batch_size=batch_size,
        device=device,
        manifest_name="validation_model_prediction_manifest.json",
    )
    _predict_shards(
        model,
        calibration_shards,
        tensor_root=tensor_root,
        output_root=prediction_root,
        batch_size=batch_size,
        device=device,
        manifest_name="calibration_prediction_manifest.json",
    )
    checkpoint_sha = _sha256(checkpoint)
    config_sha = hashlib.sha256(_canonical_bytes(config)).hexdigest()
    model_id = hashlib.sha256(
        _canonical_bytes(
            {
                "checkpoint_sha256": checkpoint_sha,
                "config_sha256": config_sha,
                "artifact_sha256": _sha256(artifacts_path),
            }
        )
    ).hexdigest()
    manifest = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "engineering_status": "PASS",
        "stage2_config_sha256": config_sha,
        "model_id": model_id,
        "checkpoint_sha256": checkpoint_sha,
        "artifact_sha256": _sha256(artifacts_path),
        "fit_dates": split["train_dates"],
        "validation_dates": split["validation_model_dates"],
        "calibration_prediction_dates": split["calibration_dates"],
        "tensor_root": str(tensor_root.resolve()),
        "prediction_root": str(prediction_root.resolve()),
        "test_rows_read": 0,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "training_history": history,
        "runtime_s": time.perf_counter() - started,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    manifest = train(_parser().parse_args())
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
