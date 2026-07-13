"""Validation-only calibration and uncertainty for full-day Stage2 predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


TARGETS = ["lcs", "pmis", "rts"]
KEYS = ["order_id", "driver_id", "date", "route_link_id", "route_link_seq"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", choices=["isotonic", "platt"], default="isotonic")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _fit_calibrator(x: np.ndarray, y: np.ndarray, method: str):
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.clip(x[valid], 0, 1)
    y = y[valid].astype(int)
    if len(np.unique(y)) < 2:
        return None
    if method == "platt":
        model = LogisticRegression(max_iter=500)
        model.fit(x.reshape(-1, 1), y)
        return model
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(x, y)
    return model


def _predict_calibrator(model, x: np.ndarray, method: str) -> np.ndarray:
    x = np.clip(x, 0, 1)
    if model is None:
        return x
    if method == "platt":
        return model.predict_proba(x.reshape(-1, 1))[:, 1]
    return model.predict(x)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    calibrated_dir = args.output_root / "calibrated_predictions" / "fold=7"
    uncertainty_dir = args.output_root / "uncertainty_predictions" / "fold=7"
    calibrated_dir.mkdir(parents=True, exist_ok=True)
    uncertainty_dir.mkdir(parents=True, exist_ok=True)
    calibrated_path = calibrated_dir / "test_predictions.parquet"
    uncertainty_path = uncertainty_dir / "test_uncertainty.parquet"
    if (calibrated_path.exists() or uncertainty_path.exists()) and not args.overwrite:
        raise FileExistsError("Outputs exist; pass --overwrite")

    val = pd.read_parquet(args.validation_predictions)
    test = pd.read_parquet(args.test_predictions)
    calibrated = test[KEYS].copy()
    uncertainty = test[KEYS].copy()
    manifest = {"method": args.method, "targets": {}, "validation_predictions": str(args.validation_predictions), "test_predictions": str(args.test_predictions)}
    for target in TARGETS:
        raw_prob = pd.to_numeric(val[f"pred_{target}_tail_prob"], errors="coerce").to_numpy(float)
        truth = val[f"target_{target}_tail"].astype(float).to_numpy()
        valid_label = val[f"{target}_valid"].astype(bool).to_numpy() if f"{target}_valid" in val else np.isfinite(truth)
        calibrator = _fit_calibrator(raw_prob[valid_label], truth[valid_label], args.method)
        full_prob = pd.to_numeric(test[f"pred_{target}_tail_prob"], errors="coerce").to_numpy(float)
        calibrated[f"{target}_tail_prob_calibrated"] = _predict_calibrator(calibrator, full_prob, args.method)

        val_raw = pd.to_numeric(val[f"pred_{target}_raw"], errors="coerce").to_numpy(float)
        val_y = pd.to_numeric(val[f"target_{target}_raw"], errors="coerce").to_numpy(float)
        val_mask = valid_label & np.isfinite(val_raw) & np.isfinite(val_y)
        residual = np.abs(val_y[val_mask] - val_raw[val_mask])
        q90 = float(np.nanquantile(residual, 0.90)) if residual.size else 0.25
        pred_raw = pd.to_numeric(test[f"pred_{target}_raw"], errors="coerce").to_numpy(float)
        uncertainty[f"{target}_uncertainty"] = q90
        uncertainty[f"{target}_lower"] = np.clip(pred_raw - q90, 0, 1)
        uncertainty[f"{target}_upper"] = np.clip(pred_raw + q90, 0, 1)
        uncertainty[f"{target}_ensemble_variance"] = 0.0
        manifest["targets"][target] = {
            "validation_rows": int(valid_label.sum()),
            "calibrator": args.method if calibrator is not None else "identity_single_class",
            "conformal_residual_q90": q90,
        }
    calibrated.to_parquet(calibrated_path, index=False, compression="zstd")
    uncertainty.to_parquet(uncertainty_path, index=False, compression="zstd")
    (args.output_root / "postprocess_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
