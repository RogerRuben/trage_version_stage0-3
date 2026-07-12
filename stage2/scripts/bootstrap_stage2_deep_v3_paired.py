"""Paired order-cluster bootstrap for RC-MSTNet minus LightGBM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage2_deep_v3_utils import LINK_TARGETS, metric_dict  # noqa: E402
from summarize_stage2_deep_v3_fold_metrics import aligned_fold  # noqa: E402


METRICS = ["auc", "ap", "spearman", "lift_top5", "lift_top10"]
ORDER_METRICS = ["auc", "ap", "lift_top10"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep-prediction-root", type=Path, required=True)
    parser.add_argument("--lightgbm-oof", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/deep_v3/bootstrap_ci"))
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lightgbm-ablation", default="static_rolling_dynamic_topology_route")
    return parser.parse_args()


def order_table(data: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return data.groupby("order_id").agg(
        truth=("true_raw", lambda value: float(np.nanquantile(value, 0.90))),
        raw=(f"{prefix}_raw", lambda value: float(np.nanquantile(value, 0.90))),
        prob=(f"{prefix}_prob", lambda value: float(np.nanquantile(value, 0.90))),
    )


def evaluate_link(part: pd.DataFrame, prefix: str) -> dict:
    return metric_dict(part["true_raw"].to_numpy(float), part[f"{prefix}_raw"].to_numpy(float), part[f"{prefix}_prob"].to_numpy(float), part["true_tail"].to_numpy(bool))


def evaluate_order(part: pd.DataFrame) -> dict:
    high = part["truth"].to_numpy(float) >= np.nanquantile(part["truth"], 0.90)
    return metric_dict(part["truth"].to_numpy(float), part["raw"].to_numpy(float), part["prob"].to_numpy(float), high)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    lightgbm = pd.read_parquet(args.lightgbm_oof)
    rng = np.random.default_rng(args.seed)
    replicate_rows = []
    for fold_root in sorted(args.deep_prediction_root.glob("fold=*")):
        fold = int(fold_root.name.split("=", 1)[-1])
        for target in LINK_TARGETS:
            data = aligned_fold(fold_root / "test_predictions.parquet", lightgbm, fold, target, args.lightgbm_ablation)
            groups = [indices.to_numpy() for _, indices in pd.Series(np.arange(len(data)), index=data["order_id"]).groupby(level=0)]
            deep_orders = order_table(data, "deep")
            lgbm_orders = order_table(data, "lgbm")
            order_ids = deep_orders.index.to_numpy()
            for replicate in range(args.rounds):
                sampled = rng.integers(0, len(groups), len(groups))
                link_indices = np.concatenate([groups[index] for index in sampled])
                link_part = data.iloc[link_indices]
                sampled_orders = order_ids[sampled]
                deep_order_part = deep_orders.loc[sampled_orders].reset_index(drop=True)
                lgbm_order_part = lgbm_orders.loc[sampled_orders].reset_index(drop=True)
                deep_link = evaluate_link(link_part, "deep")
                lgbm_link = evaluate_link(link_part, "lgbm")
                deep_order = evaluate_order(deep_order_part)
                lgbm_order = evaluate_order(lgbm_order_part)
                row = {"fold": fold, "target": target.upper(), "replicate": replicate}
                row.update({f"delta_{metric}": deep_link[metric] - lgbm_link[metric] for metric in METRICS})
                row.update({f"delta_order_{metric}": deep_order[metric] - lgbm_order[metric] for metric in ORDER_METRICS})
                replicate_rows.append(row)
    replicates = pd.DataFrame(replicate_rows)
    metric_columns = [column for column in replicates if column.startswith("delta_")]
    by_fold = []
    for (fold, target), group in replicates.groupby(["fold", "target"]):
        for metric in metric_columns:
            by_fold.append({"fold": fold, "target": target, "metric": metric, "mean_delta": group[metric].mean(), "ci_low": group[metric].quantile(0.025), "ci_high": group[metric].quantile(0.975), "rounds": args.rounds})
    by_fold = pd.DataFrame(by_fold)
    by_fold.to_csv(args.output_root / "paired_bootstrap_by_fold.csv", index=False)
    fold_mean_replicates = replicates.groupby(["target", "replicate"], as_index=False)[metric_columns].mean()
    by_target = []
    for target, group in fold_mean_replicates.groupby("target"):
        for metric in metric_columns:
            by_target.append({"target": target, "metric": metric, "mean_delta": group[metric].mean(), "ci_low": group[metric].quantile(0.025), "ci_high": group[metric].quantile(0.975), "rounds": args.rounds, "aggregation": "mean_across_folds_per_replicate"})
    by_target = pd.DataFrame(by_target)
    by_target.to_csv(args.output_root / "paired_bootstrap_by_target.csv", index=False)
    report = ["# Paired order-cluster bootstrap", "", "Positive deltas favor RC-MSTNet. Sampling unit is order_id; link rows remain clustered.", "", by_target.to_markdown(index=False, floatfmt=".4f")]
    (args.output_root / "paired_bootstrap_report.md").write_text("\n".join(report), encoding="utf-8")
    (args.output_root / "paired_bootstrap_manifest.json").write_text(json.dumps({"rounds": args.rounds, "seed": args.seed, "block": "order_id", "paired": True}, indent=2), encoding="utf-8")
    print(by_target.to_string(index=False))


if __name__ == "__main__":
    main()
