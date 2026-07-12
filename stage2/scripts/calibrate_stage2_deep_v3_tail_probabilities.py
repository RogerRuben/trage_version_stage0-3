"""Validation-only calibration for RC-MSTNet tail probabilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage2_deep_v3_utils import LINK_TARGETS, ece_score  # noqa: E402


KEYS = ["order_id", "driver_id", "date", "route_link_id", "route_link_seq"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/deep_v3/calibration"))
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-5, 1 - 1e-5)
    return np.log(probability / (1 - probability))


def scores(y: np.ndarray, probability: np.ndarray) -> dict:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return {
        "rows": len(y), "positive_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, probability)),
        "ap": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "ece": float(ece_score(y.astype(float), probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
    }


def reliability_rows(y: np.ndarray, probability: np.ndarray, bins: int, **labels) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (probability >= low) & ((probability <= high) if index == bins - 1 else (probability < high))
        rows.append({**labels, "bin": index, "low": low, "high": high, "rows": int(mask.sum()), "mean_probability": float(probability[mask].mean()) if mask.any() else np.nan, "observed_rate": float(y[mask].mean()) if mask.any() else np.nan})
    return rows


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    metrics = []
    curves = []
    manifest = {"fit_split": "validation", "test_labels_used_for_fit": False, "folds": {}}
    for fold_root in sorted(args.prediction_root.glob("fold=*")):
        fold = int(fold_root.name.split("=", 1)[-1])
        frames = {split: pd.read_parquet(fold_root / f"{split}_predictions.parquet") for split in ["validation", "test"]}
        output_frames = {split: frame[KEYS].copy() for split, frame in frames.items()}
        fold_methods = {}
        for target in LINK_TARGETS:
            val_mask = frames["validation"][f"{target}_valid"].fillna(False).to_numpy(bool)
            val_y = frames["validation"].loc[val_mask, f"target_{target}_tail"].to_numpy(int)
            val_raw = frames["validation"].loc[val_mask, f"pred_{target}_tail_prob"].to_numpy(float)
            platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit(val_raw).reshape(-1, 1), val_y)
            isotonic = IsotonicRegression(out_of_bounds="clip").fit(val_raw, val_y)
            method_predictions = {}
            for split, frame in frames.items():
                mask = frame[f"{target}_valid"].fillna(False).to_numpy(bool)
                raw = frame.loc[mask, f"pred_{target}_tail_prob"].to_numpy(float)
                method_predictions[split] = {
                    "raw": raw,
                    "platt": platt.predict_proba(logit(raw).reshape(-1, 1))[:, 1],
                    "isotonic": isotonic.predict(raw),
                }
                y = frame.loc[mask, f"target_{target}_tail"].to_numpy(int)
                for method, probability in method_predictions[split].items():
                    metrics.append({"fold": fold, "split": split, "target": target.upper(), "method": method, **scores(y, probability)})
                    curves.extend(reliability_rows(y, probability, args.bins, fold=fold, split=split, target=target.upper(), method=method))
            val_candidates = [row for row in metrics if row["fold"] == fold and row["split"] == "validation" and row["target"] == target.upper()]
            selected = min(val_candidates, key=lambda row: row["brier"])["method"]
            fold_methods[target] = selected
            for split, frame in frames.items():
                mask = frame[f"{target}_valid"].fillna(False).to_numpy(bool)
                calibrated = np.full(len(frame), np.nan, dtype="float32")
                calibrated[mask] = method_predictions[split][selected].astype("float32")
                output_frames[split][f"{target}_tail_prob_raw"] = frame[f"pred_{target}_tail_prob"].astype("float32")
                output_frames[split][f"{target}_tail_prob_calibrated"] = calibrated
                output_frames[split][f"{target}_calibration_method"] = selected
                output_frames[split][f"{target}_valid"] = frame[f"{target}_valid"].fillna(False)
        fold_output = args.output_root / "calibrated_predictions" / f"fold={fold}"
        fold_output.mkdir(parents=True, exist_ok=True)
        for split, frame in output_frames.items():
            frame.to_parquet(fold_output / f"{split}_predictions.parquet", index=False, compression="zstd")
        manifest["folds"][str(fold)] = fold_methods
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(args.output_root / "calibration_metrics.csv", index=False)
    curve_frame = pd.DataFrame(curves)
    curve_root = args.output_root / "calibration_curves"
    curve_root.mkdir(parents=True, exist_ok=True)
    curve_frame.to_csv(curve_root / "reliability_curves.csv", index=False)
    (args.output_root / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    test = metric_frame[metric_frame["split"].eq("test")].groupby(["target", "method"], as_index=False)[["auc", "ap", "brier", "ece", "log_loss"]].mean()
    report = ["# RC-MSTNet tail calibration", "", "Calibrators and method selection use validation labels only. Test labels are evaluation-only.", "", test.to_markdown(index=False, floatfmt=".5f")]
    (args.output_root / "calibration_report.md").write_text("\n".join(report), encoding="utf-8")
    print(test.to_string(index=False))


if __name__ == "__main__":
    main()
