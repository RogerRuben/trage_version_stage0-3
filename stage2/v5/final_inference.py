"""One-shot inference of the frozen v5 model on preregistered final shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import load_config
from .models.rc_mstnet_v5 import RCMSTNetV5
from .train_worker import _json, _predict, _shards


def run(*, repo_root: str | Path = ".") -> dict:
    root = Path(repo_root).resolve()
    config = load_config(root / "stage2/config/stage2_v5.json")
    tensor_root = root / "stage2/output_v5/final_upstream/stage2/tensor_shards"
    checkpoint_path = root / "stage2/output_v5/deep_model/best_model.pt"
    model_manifest = _json(root / "stage2/output_v5/deep_model/model_manifest.json")
    prediction_root = root / "stage2/output_v5/final_upstream/stage2/deep_predictions"
    dates = list(config.section("split")["final_test_dates"])
    shards = _shards(tensor_root, "final_test", dates)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = dict(checkpoint["model_config"])
    model_config["history_mode"] = "gate"
    model = RCMSTNetV5(**model_config)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    allowed_missing = {name for name in missing if name.startswith("ordinary_history_fusion.")}
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(f"frozen checkpoint incompatibility: missing={missing}, unexpected={unexpected}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    deep = config.section("deep")
    prediction = _predict(
        model,
        shards,
        tensor_root,
        prediction_root,
        int(deep["batch_size"]),
        device,
        bool(deep.get("mixed_precision", False)) and device.type == "cuda",
        "final_test_prediction_manifest.json",
    )
    result = {
        "schema_version": "stage2_v5_final_inference.1",
        "status": "PASS",
        "protocol": "one_shot_preregistered",
        "dates": dates,
        "selection_model_id": model_manifest["model_id"],
        "checkpoint_sha256": model_manifest["checkpoint_sha256"],
        "history_mode": "horizon_gate",
        "device": str(device),
        "post_test_tuning_count": 0,
        "prediction": prediction,
    }
    (prediction_root / "final_inference_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    print(json.dumps(run(repo_root=args.repo_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
