"""Run the frozen RC-MSTNet A-F structural ablation suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-shard-root", type=Path, required=True)
    parser.add_argument("--fold-config", type=Path, default=Path("rolling_threefold_config.json"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/deep_v3/ablations"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--folds", default="1,2,3")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


CONFIGS = [
    ("A_neural_tabular", ["--no-dynamic-encoder", "--no-local-route-encoder", "--no-route-transformer", "--no-route-aux-head"], "lcs,pmis,rts"),
    ("B_dynamic_temporal", ["--dynamic-encoder", "--no-local-route-encoder", "--no-route-transformer", "--no-route-aux-head"], "lcs,pmis,rts"),
    ("C_local_route", ["--dynamic-encoder", "--local-route-encoder", "--no-route-transformer", "--no-route-aux-head"], "lcs,pmis,rts"),
    ("D_route_transformer_shared", ["--dynamic-encoder", "--local-route-encoder", "--route-transformer", "--no-route-aux-head"], "lcs,pmis,rts"),
    ("E_single_lcs", ["--dynamic-encoder", "--local-route-encoder", "--route-transformer", "--no-route-aux-head"], "lcs"),
    ("E_single_pmis", ["--dynamic-encoder", "--local-route-encoder", "--route-transformer", "--no-route-aux-head"], "pmis"),
    ("E_single_rts", ["--dynamic-encoder", "--local-route-encoder", "--route-transformer", "--no-route-aux-head"], "rts"),
    ("F_route_aux_full", ["--dynamic-encoder", "--local-route-encoder", "--route-transformer", "--route-aux-head"], "lcs,pmis,rts"),
]


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    trainer = Path(__file__).resolve().parent / "train_stage2_rc_mstnet.py"
    metric_parts = []
    cost_rows = []
    for name, switches, active_targets in CONFIGS:
        model_root = args.output_root / name / "model"
        prediction_root = args.output_root / name / "predictions"
        metrics_path = model_root / "rc_mstnet_metrics_by_fold.csv"
        if not (args.skip_existing and metrics_path.exists()):
            command = [
                sys.executable, str(trainer), "--tensor-shard-root", str(args.tensor_shard_root),
                "--fold-config", str(args.fold_config), "--output-root", str(model_root),
                "--prediction-root", str(prediction_root), "--folds", args.folds,
                "--max-train-orders", "100000", "--max-eval-orders", "0", "--max-seq-len", "96",
                "--batch-size", "128", "--epochs", str(args.epochs), "--hidden-dim", "128",
                "--layers", "3", "--heads", "4", "--num-workers", "0", "--seed", str(args.seed),
                "--active-targets", active_targets,
            ] + switches
            subprocess.run(command, check=True)
        metrics = pd.read_csv(metrics_path)
        active = {value.strip() for value in active_targets.split(",")}
        metrics = metrics[metrics["target"].isin(active)].copy()
        metrics["ablation"] = name
        metrics["active_targets"] = active_targets
        metric_parts.append(metrics)
        for manifest_path in sorted(model_root.glob("fold=*/manifest.json")):
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            cost_rows.append({
                "ablation": name, "fold": payload["fold"], "active_targets": active_targets,
                "parameter_count": payload.get("parameter_count"), "training_seconds": payload.get("training_seconds"),
                "validation_prediction_seconds": payload.get("validation_prediction_seconds"),
                "test_prediction_seconds": payload.get("test_prediction_seconds"),
                "cuda_peak_reserved_mb": payload.get("cuda_peak_reserved_bytes", 0) / 2**20,
            })
    metrics = pd.concat(metric_parts, ignore_index=True)
    metrics.to_csv(args.output_root / "ablation_metrics.csv", index=False)
    order_columns = [column for column in metrics if column.startswith("order_")]
    metrics[["ablation", "active_targets", "fold", "split", "target"] + order_columns].to_csv(args.output_root / "ablation_order_metrics.csv", index=False)
    costs = pd.DataFrame(cost_rows)
    costs.to_csv(args.output_root / "ablation_costs.csv", index=False)
    test = metrics[metrics["split"].eq("test")].groupby(["ablation", "target"], as_index=False)[["auc", "ap", "spearman", "lift_top5", "order_q90_lift_top10"]].mean()
    cost_summary = costs.groupby("ablation", as_index=False)[["parameter_count", "training_seconds", "test_prediction_seconds", "cuda_peak_reserved_mb"]].mean()
    report = ["# RC-MSTNet structural ablation", "", f"All ablations use 100k orders/fold, three rolling folds, seed {args.seed}, and {args.epochs} epochs.", "", "## Accuracy", "", test.to_markdown(index=False, floatfmt=".4f"), "", "## Cost", "", cost_summary.to_markdown(index=False, floatfmt=".2f")]
    (args.output_root / "ablation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(test.to_string(index=False))


if __name__ == "__main__":
    main()
