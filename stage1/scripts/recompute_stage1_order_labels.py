"""Recompute Stage1 order-level labels from existing link-label partitions.

Use this after changing order aggregation logic when link-level labels do not need
to be rebuilt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_stage1_labels import aggregate_orders  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("stage1/output/prediction_split"))
    parser.add_argument("--order-base-root", type=Path, default=Path("stage0/output/order_base"))
    parser.add_argument("--split-config", type=Path, default=Path("split_config.json"))
    parser.add_argument("--dates", default="split", help="'split' or comma-separated YYYYMMDD dates")
    parser.add_argument("--high-threshold", type=float, default=0.90)
    return parser.parse_args()


def select_dates(args: argparse.Namespace) -> list[str]:
    if args.dates != "split":
        return [item.strip() for item in args.dates.split(",") if item.strip()]
    config = json.loads(args.split_config.read_text(encoding="utf-8"))
    return list(config["train_dates"]) + [config["validation_date"], config["test_date"]]


def main() -> None:
    args = parse_args()
    dates = select_dates(args)
    rows = []
    for date in dates:
        label_dir = args.output_root / "link_labels" / f"day={date}"
        files = sorted(label_dir.glob("*.parquet"))
        if not files:
            rows.append({"date": date, "status": "missing_link_labels", "orders": 0})
            continue
        order_parts = []
        for path in files:
            frame = pd.read_parquet(path)
            order_parts.append(aggregate_orders(frame, args.high_threshold))
        orders = pd.concat(order_parts, ignore_index=True)
        base_path = args.order_base_root / f"day={date}.parquet"
        if base_path.exists():
            base = pd.read_parquet(base_path, columns=["order_id", "quality_tier"]).rename(
                columns={"quality_tier": "stage0_quality_tier"}
            )
            orders = orders.merge(base, on="order_id", how="left")
        order_path = args.output_root / "order_labels" / f"day={date}.parquet"
        order_path.parent.mkdir(parents=True, exist_ok=True)
        orders.to_parquet(order_path, index=False, compression="zstd")
        rows.append({"date": date, "status": "complete", "orders": len(orders)})
        print(f"recomputed order labels day={date} orders={len(orders):,}", flush=True)

    reports = args.output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(rows)
    report.to_csv(reports / "recomputed_order_labels.csv", index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
