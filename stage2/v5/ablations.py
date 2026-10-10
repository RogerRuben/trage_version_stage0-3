"""Same-row forecast-horizon fusion ablation evaluation.

Model selection is based only on the preregistered validation-model dates and
the primary direct-observed pace MAE.  Calibration dates are reported but do
not participate in selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baselines import _paired_bootstrap, continuous_metrics
from .config import load_config


MODEL_ROOTS = {
    "horizon_gate": "stage2/output_v5/predictions",
    "ordinary_concatenation": "stage2/output_v5/ablations/ordinary_concatenation/merged",
    "without_recent": "stage2/output_v5/ablations/without_recent/merged",
    "without_profile": "stage2/output_v5/ablations/without_profile/merged",
}


def _prediction_path(root: Path, relative_root: str, split: str, date: str) -> Path:
    return root / relative_root / f"split={split}" / f"date={date}" / "traversal_predictions.parquet"


def _identity(frame: pd.DataFrame) -> np.ndarray:
    return frame[["order_id", "traversal_id"]].astype(str).to_numpy()


def evaluate_ablations(
    *,
    repo_root: str | Path = ".",
    config_path: str | Path = "stage2/config/stage2_v5.json",
    model_roots: dict[str, str | Path] | None = None,
    report_root: str | Path = "stage2/docs/v5",
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    config = load_config(config_file if config_file.is_absolute() else root / config_file)
    roots = model_roots or MODEL_ROOTS
    split_config = config.section("split")
    dates = [("validation_model", value) for value in split_config["validation_model_dates"]]
    dates += [("calibration", value) for value in split_config["calibration_dates"]]
    dates += [("evaluation", value) for value in split_config["evaluation_dates"]]
    seed = int(config.section("runtime")["random_seed"])
    metric_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []

    for split, date in dates:
        frames = {
            model: pd.read_parquet(_prediction_path(root, relative, split, date))
            for model, relative in roots.items()
        }
        reference = frames["horizon_gate"]
        reference_identity = _identity(reference)
        for model, frame in frames.items():
            if len(frame) != len(reference) or not np.array_equal(_identity(frame), reference_identity):
                raise ValueError(f"ablation identity mismatch: {model} on {date}")
            if not np.array_equal(frame["pace_target_valid"].to_numpy(bool), reference["pace_target_valid"].to_numpy(bool)):
                raise ValueError(f"ablation target mask mismatch: {model} on {date}")
            if not np.allclose(
                frame["pace_sec_per_m"].to_numpy(float),
                reference["pace_sec_per_m"].to_numpy(float),
                equal_nan=True,
            ):
                raise ValueError(f"ablation target mismatch: {model} on {date}")

        truth = reference["pace_sec_per_m"].to_numpy(float)
        valid = reference["pace_target_valid"].to_numpy(bool) & np.isfinite(truth)
        compare = pd.DataFrame({
            "order_id": reference["order_id"].astype(str),
            "truth": np.where(valid, truth, np.nan),
        })
        for model, frame in frames.items():
            prediction = frame["pace_pred_mean"].to_numpy(float)
            compare[model] = prediction
            metrics = continuous_metrics(np.where(valid, truth, np.nan), prediction)
            metric_rows.append({
                "split": split,
                "date": date,
                "target": "pace_sec_per_m",
                "model": model,
                **metrics,
                "history_recent_gate_mean": float(frame["history_recent_gate"].mean()),
            })
            if model != "horizon_gate":
                bootstrap_rows.append({
                    "split": split,
                    "date": date,
                    "left_model": model,
                    "right_model": "horizon_gate",
                    **_paired_bootstrap(compare, model, "horizon_gate", seed=seed),
                })

    metrics = pd.DataFrame(metric_rows)
    bootstraps = pd.DataFrame(bootstrap_rows)
    validation = metrics[metrics["split"].eq("validation_model")].copy()
    validation["weighted_absolute_error"] = validation["mae"] * validation["count"]
    totals = validation.groupby("model", sort=False, observed=True)[["weighted_absolute_error", "count"]].sum()
    aggregate_mae = (totals["weighted_absolute_error"] / totals["count"]).sort_values()
    selected = str(aggregate_mae.index[0])
    gate_mae = float(aggregate_mae["horizon_gate"])
    summary: dict[str, Any] = {
        "schema_version": "stage2_v5_horizon_ablation.1",
        "selection_dates": list(split_config["validation_model_dates"]),
        "selection_metric": "same-row direct-observed pace_sec_per_m MAE",
        "selected_history_mode": selected,
        "selected_validation_mae": float(aggregate_mae.iloc[0]),
        "horizon_gate_validation_mae": gate_mae,
        "horizon_gate_selected": selected == "horizon_gate",
        "validation_mae_by_model": {name: float(value) for name, value in aggregate_mae.items()},
        "development_evaluation_dates": list(split_config["evaluation_dates"]),
        "status": "PASS",
    }
    report = Path(report_root)
    if not report.is_absolute():
        report = root / report
    report.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(report / "horizon_gate_ablation.csv", index=False)
    bootstraps.to_csv(report / "horizon_gate_ablation_bootstrap.csv", index=False)
    (report / "stage2_v5_horizon_ablation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    print(json.dumps(evaluate_ablations(repo_root=args.repo_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
