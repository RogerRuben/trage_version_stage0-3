"""Frozen-checkpoint prediction worker for the held-out Test date."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from stage2.v4.models.rc_mstnet_v4 import RCMSTNetV4
from stage2.v4.models.train_worker import (
    _atomic_json,
    _predict_shards,
    _read_json,
    _shards,
    _sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--test-date", required=True)
    args = parser.parse_args()
    config = _read_json(args.config)
    if config["split"]["test_dates"] != [args.test_date]:
        raise RuntimeError("prediction worker may read only the frozen Test date")
    manifest = _read_json(args.model_root / "model_manifest.json")
    checkpoint_path = args.model_root / "best_model.pt"
    if (
        manifest.get("engineering_status") != "PASS"
        or manifest.get("checkpoint_sha256") != _sha256(checkpoint_path)
    ):
        raise RuntimeError("frozen deep checkpoint identity mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = RCMSTNetV4(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    shards = _shards(args.tensor_root, "test", [args.test_date])
    result = _predict_shards(
        model,
        shards,
        tensor_root=args.tensor_root,
        output_root=args.prediction_root,
        batch_size=int(config["deep"]["batch_size"]),
        device=device,
        manifest_name="test_prediction_manifest.json",
    )
    result["deep_model_id"] = manifest["model_id"]
    result["stage2_config_sha256"] = manifest["stage2_config_sha256"]
    result["test_date"] = args.test_date
    _atomic_json(args.prediction_root / "test_prediction_manifest.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
