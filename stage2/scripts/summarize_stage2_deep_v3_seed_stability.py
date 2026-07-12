"""Summarize RC-MSTNet stability across fixed-seed formal runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage2_deep_v3_utils import LINK_TARGETS, order_level_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="seed=path/to/model/output; repeat for every seed")
    parser.add_argument("--prediction-run", action="append", default=[], help="seed=path/to/prediction/folds, used to backfill stable order metrics")
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/deep_v3/seed_stability"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    prediction_roots = {int(seed): Path(path) for seed, path in (spec.split("=", 1) for spec in args.prediction_run)}
    frames = []
    for spec in args.run:
        seed_text, path_text = spec.split("=", 1)
        frame = pd.read_csv(Path(path_text) / "rc_mstnet_metrics_by_fold.csv")
        seed = int(seed_text)
        if seed in prediction_roots:
            order_rows = []
            for fold_root in sorted(prediction_roots[seed].glob("fold=*")):
                fold = int(fold_root.name.split("=", 1)[-1])
                for split in ["validation", "test"]:
                    prediction = pd.read_parquet(fold_root / f"{split}_predictions.parquet")
                    for target in LINK_TARGETS:
                        order_rows.append({"fold": fold, "split": split, "target": target, **order_level_metrics(prediction, target)})
            order_frame = pd.DataFrame(order_rows)
            order_columns = [column for column in order_frame if column.startswith("order_") and column != "order_tail_definition"]
            frame = frame.drop(columns=order_columns, errors="ignore").merge(order_frame[["fold", "split", "target"] + order_columns], on=["fold", "split", "target"], how="left")
        frame["seed"] = seed
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(args.output_root / "seed_metrics.csv", index=False)
    numeric = [column for column in ["auc", "ap", "spearman", "pearson", "mae", "rmse", "lift_top5", "lift_top10", "order_q90_auc", "order_q90_ap", "order_q90_lift_top10"] if column in metrics]
    per_seed = metrics.groupby(["seed", "split", "target"], as_index=False)[numeric].mean()
    rows = []
    for (split, target), group in per_seed.groupby(["split", "target"]):
        for metric in numeric:
            values = group[metric].dropna().to_numpy(float)
            mean = float(values.mean()) if len(values) else np.nan
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            rows.append({"split": split, "target": target.upper(), "metric": metric, "seeds": len(values), "mean": mean, "std": std, "min": float(values.min()) if len(values) else np.nan, "max": float(values.max()) if len(values) else np.nan, "coefficient_of_variation": std / abs(mean) if mean else np.nan})
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_root / "seed_summary.csv", index=False)
    test = summary[summary["split"].eq("test") & summary["metric"].isin(["auc", "ap", "spearman", "lift_top5", "order_q90_lift_top10"])]
    report = ["# RC-MSTNet seed stability", "", "Folds, data keys, budgets, and hyperparameters are fixed. Only initialization, dropout, and batch ordering change.", "", test.to_markdown(index=False, floatfmt=".6f")]
    (args.output_root / "seed_stability_report.md").write_text("\n".join(report), encoding="utf-8")
    print(test.to_string(index=False))


if __name__ == "__main__":
    main()
