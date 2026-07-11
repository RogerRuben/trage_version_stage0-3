"""Summarize Stage2 deep upper-bound experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["LCS", "IIS", "RTS", "PMIS"]
BASELINE_AUC = {"LCS": 0.612117, "IIS": 0.651835, "RTS": 0.612089, "PMIS": 0.610686}
BASELINE_SPEARMAN = {"LCS": 0.245000, "IIS": 0.383429, "RTS": 0.144615, "PMIS": 0.217532}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-output", type=Path, default=Path("stage2/output"))
    return parser.parse_args()


def read_metrics(root: Path) -> pd.DataFrame:
    inputs = [
        ("sampled_lightgbm_1m", root / "baselines" / "baseline_metrics_by_target.csv"),
        ("full_tabular_lgbm_3m_tail", root / "deep_baselines" / "full_tabular" / "full_tabular_metrics_by_target.csv"),
        ("sequence_bigru_50k_orders", root / "deep_baselines" / "sequence_model" / "sequence_metrics_by_target.csv"),
        ("gnn_sequence_bigru_50k_orders", root / "deep_baselines" / "gnn_model" / "gnn_metrics_by_target.csv"),
    ]
    frames = []
    for model, path in inputs:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["model"] = model
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def order_tail_separation(prediction_root: Path) -> pd.DataFrame:
    files = [
        ("full_tabular_lgbm_3m_tail", "full_tabular_validation.parquet", "validation"),
        ("full_tabular_lgbm_3m_tail", "full_tabular_test.parquet", "test"),
        ("sequence_bigru_50k_orders", "sequence_validation.parquet", "validation"),
        ("sequence_bigru_50k_orders", "sequence_test.parquet", "test"),
        ("gnn_sequence_bigru_50k_orders", "gnn_sequence_validation.parquet", "validation"),
        ("gnn_sequence_bigru_50k_orders", "gnn_sequence_test.parquet", "test"),
    ]
    rows = []
    for model, filename, split in files:
        path = prediction_root / filename
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        for target in TARGETS:
            pred_col = f"pred_{target.lower()}"
            target_col = f"target_{target.lower()}"
            valid_col = f"{target.lower()}_valid"
            valid = frame[valid_col].fillna(False) & frame[target_col].notna()
            if not valid.any():
                continue
            per_order = frame.loc[valid, ["order_id", pred_col, target_col]].groupby("order_id").agg(
                pred_mean=(pred_col, "mean"),
                pred_max=(pred_col, "max"),
                actual_tail=(target_col, lambda values: float(np.nanmax(values) >= 0.90)),
                actual_mean=(target_col, "mean"),
            ).reset_index()
            base = float(per_order.actual_tail.mean())
            for score_col in ["pred_mean", "pred_max"]:
                row = {
                    "model": model,
                    "split": split,
                    "target": target,
                    "score": score_col,
                    "orders": len(per_order),
                    "base_tail_rate": base,
                }
                for share, label in [(0.10, "top10"), (0.05, "top5")]:
                    n = max(1, int(len(per_order) * share))
                    top = per_order.nlargest(n, score_col)
                    precision = float(top.actual_tail.mean())
                    recall = float(top.actual_tail.sum() / max(per_order.actual_tail.sum(), 1))
                    row[f"precision_at_{label}"] = precision
                    row[f"recall_at_{label}"] = recall
                    row[f"lift_at_{label}"] = precision / base if base > 0 else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.stage2_output
    metrics = read_metrics(root)
    order_tail = order_tail_separation(root / "deep_predictions")
    report_path = root / "deep_model_report.md"
    output_dir = root / "deep_baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "deep_model_metrics_comparison.csv", index=False)
    order_tail.to_csv(output_dir / "order_tail_separation.csv", index=False)

    test = metrics[metrics.split.eq("test")].copy()
    pivot = test.pivot_table(index=["model", "target"], values=["auc", "spearman", "top10_lift", "top5_lift"], aggfunc="first").reset_index()
    best_auc = test.sort_values("auc", ascending=False).groupby("target", as_index=False).first()

    lines = [
        "# Stage2 deep upper-bound modeling report",
        "",
        "## Experiment scope",
        "",
        "This report compares the original 1M-row LightGBM baseline, a stronger tail-weighted full/large tabular baseline, a BiGRU route-sequence model, and a GraphSAGE-style topology-aware route-sequence model.",
        "",
        "Deep sequence/GNN probes used 50,000 train orders and 30,000 validation/test orders with `max_seq_len=128`; full tabular used 3,000,000 valid rows per target plus train-only historical profiles.",
        "",
        "All experiments use the Stage2 feature whitelist. IIS missing labels are masked and never filled with zero.",
        "",
        "## Test metrics",
        "",
        "| model | target | AUC | Spearman | top10 lift | top5 lift |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in pivot.sort_values(["target", "auc"], ascending=[True, False]).itertuples(index=False):
        lines.append(f"| {row.model} | {row.target} | {row.auc:.3f} | {row.spearman:.3f} | {row.top10_lift:.2f} | {row.top5_lift:.2f} |")

    lines += [
        "",
        "## Best test AUC by target",
        "",
        "| target | best model | best AUC | baseline AUC | delta |",
        "|---|---|---:|---:|---:|",
    ]
    for row in best_auc.itertuples(index=False):
        baseline = BASELINE_AUC.get(row.target, np.nan)
        lines.append(f"| {row.target} | {row.model} | {row.auc:.3f} | {baseline:.3f} | {row.auc - baseline:+.3f} |")

    if not order_tail.empty:
        summary = order_tail[(order_tail.split == "test") & (order_tail.score == "pred_max")].copy()
        lines += [
            "",
            "## Order-level tail separation",
            "",
            "`pred_max` asks whether the model can surface orders containing at least one high-stress link.",
            "",
            "| model | target | base tail | precision@top10 | lift@top10 | precision@top5 | lift@top5 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for _, row in summary.iterrows():
            lines.append(
                f"| {row['model']} | {row['target']} | {row['base_tail_rate']:.3f} | "
                f"{row.get('precision_at_top10', np.nan):.3f} | {row.get('lift_at_top10', np.nan):.2f} | "
                f"{row.get('precision_at_top5', np.nan):.3f} | {row.get('lift_at_top5', np.nan):.2f} |"
            )

    lines += [
        "",
        "## Interpretation",
        "",
        "- The strongest improvement comes from the full/large tabular baseline with train-only historical profile features and tail weighting.",
        "- IIS reaches the stated target range in the full tabular run, while sequence and GNN probes improve validation metrics but do not generalize to the test day under the current training setup.",
        "- RTS remains the weakest and most drift-sensitive target; it should be reported separately and should not be hidden inside a composite average.",
        "- The current deep sequence/GNN probes are not yet reliable enough to serve as Stage3 calibrated vector inputs without OOF/rolling prediction and stronger regularization.",
        "",
        "## Recommendation",
        "",
        "Use the full tabular model as the current Stage2 upper-bound baseline. Treat sequence/GNN results as evidence that route context has signal but is highly shift-sensitive. Before Stage3 calibration, generate rolling/OOF predictions and consider target-specific RTS treatment.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
