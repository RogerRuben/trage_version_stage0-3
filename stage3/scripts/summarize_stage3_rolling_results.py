"""Summarize Stage3 rolling Core vs Core+IIS results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("stage3/results/rolling"))
    parser.add_argument("--output-root", type=Path, default=Path("stage3/results/rolling"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold_dir in sorted(args.results_root.glob("fold_*")):
        try:
            fold = int(fold_dir.name.split("_")[-1])
        except ValueError:
            continue
        for model_dir, model_name in [("core_deepsets", "Core"), ("core_iis_dropout", "Core+IIS+dropout")]:
            path = fold_dir / model_dir / "metrics.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            frame = frame[frame["split"].eq("test")].copy()
            frame["fold"] = fold
            frame["stage3_model"] = model_name
            rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"No Stage3 metric files found under {args.results_root}")
    metrics = pd.concat(rows, ignore_index=True)
    metrics.to_csv(args.output_root / "stage3_rolling_metrics_by_fold.csv", index=False)
    summary = metrics.groupby(["stage3_model", "target"], as_index=False)[["auc", "ap", "brier", "ece", "lift_top5", "lift_top10", "recall_top5", "recall_top10", "precision_top5", "precision_top10", "spearman", "mae", "rmse"]].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(column).strip("_") for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(args.output_root / "stage3_rolling_metrics_summary.csv", index=False)
    pivot = metrics.pivot_table(index=["fold", "target"], columns="stage3_model", values=["auc", "ap", "lift_top10", "ece"])
    deltas = []
    for (fold, target), row in pivot.iterrows():
        out = {"fold": fold, "target": target}
        for metric in ["auc", "ap", "lift_top10", "ece"]:
            if (metric, "Core") in row.index and (metric, "Core+IIS+dropout") in row.index:
                out[f"delta_{metric}"] = row[(metric, "Core+IIS+dropout")] - row[(metric, "Core")]
        deltas.append(out)
    delta_frame = pd.DataFrame(deltas)
    delta_frame.to_csv(args.output_root / "stage3_core_iis_deltas.csv", index=False)
    overall_delta = delta_frame[delta_frame["target"].eq("OVERALL")]
    iis_decision = "optional_gated_branch"
    if len(overall_delta) and overall_delta["delta_ap"].gt(0).mean() >= 2 / 3 and overall_delta["delta_lift_top10"].gt(0).mean() >= 2 / 3:
        iis_decision = "candidate_optional_branch_with_positive_overall_signal"
    report = [
        "# Stage3 rolling results summary",
        "",
        "Core DeepSets and Core+IIS+dropout are evaluated on strict temporal rolling folds.",
        "",
        "## Test metrics summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Core+IIS minus Core deltas",
        "",
        delta_frame.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"## IIS positioning\n\n`{iis_decision}`. IIS is not promoted to an unconditional required modality unless fold-level gains are stable for both AP and Lift@Top10.",
    ]
    (args.output_root / "stage3_rolling_summary_report.md").write_text("\n".join(report), encoding="utf-8")
    manifest = {"folds": sorted(metrics["fold"].unique().tolist()), "models": sorted(metrics["stage3_model"].unique().tolist()), "iis_decision": iis_decision}
    (args.output_root / "stage3_rolling_summary_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
