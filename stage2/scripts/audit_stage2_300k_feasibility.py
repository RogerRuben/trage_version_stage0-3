"""Audit whether current route-conditioned data can support 300k train orders/fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/route_conditioned_dataset_15k/estimated_time_daily"))
    parser.add_argument("--fold-config", type=Path, default=Path("stage2/config/stage2_heldout_20161017_23_config.json"))
    parser.add_argument("--target-train-orders", type=int, default=300000)
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/deep_v3_scaling_300k/feasibility"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.fold_config.read_text(encoding="utf-8"))["folds"]
    rows = []
    for fold in config:
        day_counts = []
        for date in fold["train_dates"]:
            path = args.dataset_root / f"day={date}.parquet"
            if path.exists():
                day_counts.append({"date": date, "orders": int(pd.read_parquet(path, columns=["order_id"])["order_id"].nunique())})
            else:
                day_counts.append({"date": date, "orders": 0})
        total = sum(item["orders"] for item in day_counts)
        rows.append({
            "fold": int(fold["fold"]),
            "train_dates": ",".join(fold["train_dates"]),
            "available_train_orders": total,
            "target_train_orders": args.target_train_orders,
            "feasible_300k": total >= args.target_train_orders,
            "required_orders_per_day_for_7_day_train": int((args.target_train_orders + 6) // 7),
            "day_counts": day_counts,
        })
    table = pd.DataFrame([{k: v for k, v in row.items() if k != "day_counts"} for row in rows])
    table.to_csv(args.output_root / "scaling_300k_feasibility.csv", index=False)
    result = {
        "status": "PASS" if table["feasible_300k"].all() else "BLOCKED_REQUIRES_UPSTREAM_REBUILD",
        "rows": rows,
        "recommendation": "Rebuild route-conditioned upstream to about 45k orders/day before claiming a 300k train-orders/fold scaling point.",
    }
    (args.output_root / "scaling_300k_feasibility.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = [
        "# Stage2 300k scaling feasibility",
        "",
        table.to_markdown(index=False),
        "",
        result["recommendation"],
    ]
    (args.output_root / "scaling_300k_feasibility_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": result["status"]}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
