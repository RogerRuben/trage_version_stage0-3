"""Three-seed ensemble and validation-normalized conformal uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["lcs", "pmis", "rts"]
KEYS = ["order_id", "driver_id", "date", "route_link_id", "route_link_seq"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-run", action="append", required=True, help="seed=prediction/root")
    parser.add_argument("--calibration-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/deep_v3/uncertainty"))
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--folds", default=None, help="Optional comma-separated fold ids.")
    return parser.parse_args()


def read_ensemble(roots: dict[int, Path], fold: int, split: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frames = {seed: pd.read_parquet(root / f"fold={fold}" / f"{split}_predictions.parquet") for seed, root in roots.items()}
    reference = frames[sorted(frames)[0]].reset_index(drop=True)
    for seed, frame in frames.items():
        frame = frame.reset_index(drop=True)
        if not frame[KEYS].equals(reference[KEYS]):
            raise ValueError(f"Prediction key mismatch for seed={seed}, fold={fold}, split={split}")
    arrays = {}
    for target in TARGETS:
        arrays[f"{target}_raw"] = np.stack([frame[f"pred_{target}_raw"].to_numpy(float) for frame in frames.values()])
        arrays[f"{target}_tail"] = np.stack([frame[f"pred_{target}_tail_prob"].to_numpy(float) for frame in frames.values()])
    return reference, arrays


def conformal_quantile(values: np.ndarray, coverage: float) -> float:
    values = values[np.isfinite(values)]
    level = min(1.0, np.ceil((len(values) + 1) * coverage) / max(len(values), 1))
    return float(np.quantile(values, level, method="higher"))


def slice_row(frame: pd.DataFrame, mask: np.ndarray, target: str, fold: int, slice_name: str, slice_value: str) -> dict:
    part = frame.loc[mask]
    error = np.abs(part[f"target_{target}_raw"] - part[f"{target}_predicted_mean"])
    return {
        "fold": fold, "target": target.upper(), "slice": slice_name, "value": slice_value,
        "rows": len(part), "coverage": float(part[f"{target}_covered"].mean()) if len(part) else np.nan,
        "mean_width": float(part[f"{target}_interval_width"].mean()) if len(part) else np.nan,
        "mae": float(error.mean()) if len(part) else np.nan,
        "mean_uncertainty": float(part[f"{target}_uncertainty"].mean()) if len(part) else np.nan,
        "uncertainty_error_spearman": float(part[f"{target}_uncertainty"].corr(error, method="spearman")) if len(part) > 1 else np.nan,
    }


def main() -> None:
    args = parse_args()
    roots = {int(seed): Path(path) for seed, path in (spec.split("=", 1) for spec in args.prediction_run)}
    if not roots:
        raise ValueError("At least one prediction root is required")
    args.output_root.mkdir(parents=True, exist_ok=True)
    metrics = []
    slices = []
    requested_folds = None
    if args.folds:
        requested_folds = [int(value.strip()) for value in args.folds.split(",") if value.strip()]
    else:
        first_root = roots[sorted(roots)[0]]
        requested_folds = sorted(int(path.name.split("=", 1)[-1]) for path in first_root.glob("fold=*"))
    if not requested_folds:
        raise FileNotFoundError("No prediction folds found")
    manifest = {
        "coverage": args.coverage,
        "seeds": sorted(roots),
        "fit_split": "validation",
        "test_labels_used_for_fit": False,
        "single_run_mode": len(roots) == 1,
        "folds": {},
    }
    for fold in requested_folds:
        val_frame, val_arrays = read_ensemble(roots, fold, "validation")
        test_frame, test_arrays = read_ensemble(roots, fold, "test")
        output = test_frame[KEYS].copy()
        output["route_link_count"] = test_frame.groupby("order_id")["route_link_seq"].transform("size").astype("int32")
        fold_q = {}
        for target in TARGETS:
            val_valid = val_frame[f"{target}_valid"].fillna(False).to_numpy(bool)
            test_valid = test_frame[f"{target}_valid"].fillna(False).to_numpy(bool)
            val_mean = val_arrays[f"{target}_raw"].mean(axis=0)
            test_mean = test_arrays[f"{target}_raw"].mean(axis=0)
            ddof = 1 if len(roots) > 1 else 0
            val_std = val_arrays[f"{target}_raw"].std(axis=0, ddof=ddof)
            test_std = test_arrays[f"{target}_raw"].std(axis=0, ddof=ddof)
            floor = max(float(np.nanmedian(val_std[val_valid])), 0.005)
            val_scale = val_std + floor
            test_scale = test_std + floor
            val_error = np.abs(val_frame[f"target_{target}_raw"].to_numpy(float) - val_mean)
            q = conformal_quantile(val_error[val_valid] / val_scale[val_valid], args.coverage)
            half_width = q * test_scale
            lower = np.clip(test_mean - half_width, 0, 1)
            upper = np.clip(test_mean + half_width, 0, 1)
            truth = test_frame[f"target_{target}_raw"].to_numpy(float)
            covered = (truth >= lower) & (truth <= upper) & test_valid
            output[f"{target}_predicted_mean"] = test_mean.astype("float32")
            output[f"{target}_lower"] = lower.astype("float32")
            output[f"{target}_upper"] = upper.astype("float32")
            output[f"{target}_interval_width"] = (upper - lower).astype("float32")
            output[f"{target}_ensemble_variance"] = np.var(test_arrays[f"{target}_raw"], axis=0, ddof=ddof).astype("float32")
            output[f"{target}_uncertainty"] = test_scale.astype("float32")
            output[f"{target}_valid"] = test_valid
            output[f"target_{target}_raw"] = truth.astype("float32")
            output[f"target_{target}_tail"] = test_frame[f"target_{target}_tail"].fillna(False)
            output[f"{target}_covered"] = covered
            fold_q[target] = {"normalized_conformal_q": q, "scale_floor": floor}
            valid_frame = output.loc[test_valid].copy()
            slices.append(slice_row(output, test_valid, target, fold, "overall", "all"))
            route_bucket = pd.cut(output["route_link_count"], [-1, 20, 60, np.inf], labels=["short", "medium", "long"])
            for value in route_bucket.dropna().unique():
                slices.append(slice_row(output, test_valid & route_bucket.eq(value).to_numpy(), target, fold, "route_length", str(value)))
            slices.append(slice_row(output, test_valid & output[f"target_{target}_tail"].to_numpy(bool), target, fold, "tail", "high"))
            uncertainty_decile = pd.qcut(valid_frame[f"{target}_uncertainty"], 10, labels=False, duplicates="drop")
            for value in sorted(uncertainty_decile.dropna().unique()):
                index = valid_frame.index[uncertainty_decile.eq(value)]
                mask = output.index.isin(index)
                slices.append(slice_row(output, mask, target, fold, "uncertainty_decile", str(int(value) + 1)))
        if args.calibration_root is not None:
            calibrated = pd.read_parquet(args.calibration_root / "calibrated_predictions" / f"fold={fold}" / "test_predictions.parquet")
            if not calibrated[KEYS].equals(output[KEYS]):
                raise ValueError(f"Calibration key mismatch fold={fold}")
            for target in TARGETS:
                output[f"{target}_tail_prob_calibrated"] = calibrated[f"{target}_tail_prob_calibrated"]
        fold_root = args.output_root / "predictions" / f"fold={fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        output.to_parquet(fold_root / "test_uncertainty.parquet", index=False, compression="zstd")
        manifest["folds"][str(fold)] = fold_q
    slice_frame = pd.DataFrame(slices)
    slice_frame.to_csv(args.output_root / "uncertainty_slice_metrics.csv", index=False)
    overall = slice_frame[slice_frame["slice"].eq("overall")].copy()
    overall.to_csv(args.output_root / "uncertainty_metrics.csv", index=False)
    (args.output_root / "uncertainty_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = overall.groupby("target", as_index=False)[["coverage", "mean_width", "mae", "uncertainty_error_spearman"]].mean()
    report = ["# Deep v3 predictive uncertainty", "", "Intervals use validation-normalized absolute residuals and three-seed ensemble dispersion. Test labels are evaluation-only.", "", summary.to_markdown(index=False, floatfmt=".5f")]
    (args.output_root / "uncertainty_report.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
