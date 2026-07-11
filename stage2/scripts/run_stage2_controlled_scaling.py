"""Prepare or run controlled Stage2 scaling experiments.

The default mode is dry-run: it writes a reproducible command plan without
starting expensive training. Use ``--execute`` only when the compute budget is
explicitly available.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_BUDGETS = ["30000", "100000", "300000", "1000000"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/link_dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/controlled_scaling"))
    parser.add_argument("--profile-path", type=Path, default=Path("stage2/output/deep_baselines/full_tabular/train_historical_profiles.parquet"))
    parser.add_argument("--budgets", nargs="+", default=DEFAULT_BUDGETS, help="train order budgets")
    parser.add_argument("--models", nargs="+", default=["full_tabular", "route_local", "dual_graph"])
    parser.add_argument("--execute", action="store_true", help="run the plan instead of only writing it")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def command_plan(args: argparse.Namespace) -> list[dict[str, object]]:
    plan = []
    for budget in args.budgets:
        if "full_tabular" in args.models:
            plan.append({
                "name": f"full_tabular_lgbm_{budget}_orders",
                "budget_orders": int(budget),
                "command": [
                    sys.executable,
                    "stage2/scripts/train_stage2_full_tabular.py",
                    "--dataset-root", str(args.dataset_root),
                    "--output-root", str(args.output_root / f"full_tabular_{budget}"),
                    "--prediction-root", str(args.output_root / f"predictions_full_tabular_{budget}"),
                    "--max-train-orders", budget,
                    "--max-train-rows", "all",
                    "--profile-scope", "sampled_train_orders",
                    "--seed", str(args.seed),
                ],
            })
        if "route_local" in args.models:
            plan.append({
                "name": f"route_local_transformer_{budget}_orders",
                "budget_orders": int(budget),
                "command": [
                    sys.executable,
                    "stage2/scripts/train_stage2_route_local_transformer.py",
                    "--dataset-root", str(args.dataset_root),
                    "--output-root", str(args.output_root / f"route_local_{budget}"),
                    "--prediction-root", str(args.output_root / f"predictions_route_local_{budget}"),
                    "--profile-path", str(args.profile_path),
                    "--max-train-orders", budget,
                    "--max-eval-orders", "15000",
                    "--pretrain-epochs", "1",
                    "--epochs", "2",
                    "--seed", str(args.seed),
                ],
            })
        if "dual_graph" in args.models:
            plan.append({
                "name": f"dual_graph_route_transformer_{budget}_orders",
                "budget_orders": int(budget),
                "command": [
                    sys.executable,
                    "stage2/scripts/train_stage2_dual_graph_route_transformer.py",
                    "--dataset-root", str(args.dataset_root),
                    "--output-root", str(args.output_root / f"dual_graph_{budget}"),
                    "--prediction-root", str(args.output_root / f"predictions_dual_graph_{budget}"),
                    "--profile-path", str(args.profile_path),
                    "--max-train-orders", budget,
                    "--max-eval-orders", "15000",
                    "--epochs", "2",
                    "--seed", str(args.seed),
                ],
            })
    return plan


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan = command_plan(args)
    (args.output_root / "scaling_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_root / "scaling_plan.ps1").open("w", encoding="utf-8") as handle:
        for item in plan:
            handle.write("# " + str(item["name"]) + "\n")
            handle.write(" ".join(f'"{part}"' if " " in part else part for part in item["command"]) + "\n\n")
    if not args.execute:
        print(args.output_root / "scaling_plan.json")
        return
    for item in plan:
        print("running", item["name"], flush=True)
        subprocess.run(item["command"], check=True)


if __name__ == "__main__":
    main()
