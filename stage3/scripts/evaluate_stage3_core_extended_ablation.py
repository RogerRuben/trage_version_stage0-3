"""Evaluate Core vs Core+IIS predictions against core/extended overall labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("stage3/results/rolling"))
    parser.add_argument("--target-root", type=Path, default=Path("stage3/output/rolling_order_targets"))
    parser.add_argument("--feature-root", type=Path, default=Path("stage3/output/rolling_order_features"))
    parser.add_argument("--output-root", type=Path, default=Path("stage3/output/core_extended_ablation"))
    return parser.parse_args()


def metrics(y, p) -> dict:
    p = pd.Series(p).clip(1e-6, 1 - 1e-6)
    y = pd.Series(y).astype(bool)
    return {
        "rows": int(len(y)),
        "positive_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else None,
        "ap": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "precision_top10": float(y.loc[p.nlargest(max(1, int(len(p) * 0.10))).index].mean()),
        "lift_top10": float(y.loc[p.nlargest(max(1, int(len(p) * 0.10))).index].mean() / max(y.mean(), 1e-9)),
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold_dir in sorted(args.results_root.glob("fold_*")):
        fold = int(fold_dir.name.split("_")[-1])
        targets = pd.read_parquet(args.target_root / f"fold={fold}" / "split=test" / "order_targets.parquet")
        features = pd.read_parquet(args.feature_root / f"fold={fold}" / "split=test" / "order_features.parquet", columns=["order_id", "iis_applicability_q90"])
        for model_dir, model_name in [("core_deepsets", "Core"), ("core_iis_dropout", "Core+IIS+dropout")]:
            pred = pd.read_parquet(fold_dir / model_dir / "predictions.parquet")
            pred = pred[pred["split"].eq("test") & pred["target"].eq("OVERALL")][["order_id", "pred_probability"]]
            data = targets.merge(pred, on="order_id", validate="one_to_one").merge(features, on="order_id", how="left", validate="one_to_one")
            for label_col, label_name in [("core_overall_high_stress", "core_overall"), ("extended_overall_high_stress", "extended_overall")]:
                rows.append({"fold": fold, "model": model_name, "label": label_name, "slice": "all", **metrics(data[label_col], data["pred_probability"])})
                threshold = data["iis_applicability_q90"].quantile(0.75)
                subset = data[data["iis_applicability_q90"].ge(threshold)]
                rows.append({"fold": fold, "model": model_name, "label": label_name, "slice": "high_iis_applicability_q75", **metrics(subset[label_col], subset["pred_probability"])})
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_root / "core_extended_ablation_metrics.csv", index=False)
    summary = frame.groupby(["model", "label", "slice"], as_index=False)[["auc", "ap", "brier", "lift_top10"]].mean()
    summary.to_csv(args.output_root / "core_extended_ablation_summary.csv", index=False)
    report = ["# IIS target-specific ablation", "", summary.to_markdown(index=False, floatfmt=".4f")]
    (args.output_root / "core_extended_ablation_report.md").write_text("\n".join(report), encoding="utf-8")
    (args.output_root / "manifest.json").write_text(json.dumps({"status": "PASS", "folds": sorted(frame["fold"].unique().tolist())}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
