"""Audit dynamic state and log consistency for Stage4 dispatch runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("stage4/output/pricing_dispatch"))
    parser.add_argument("--output-root", type=Path, default=Path("stage4/output/pricing_dispatch/audits"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    errors: list[str] = []
    for order_path in sorted(args.run_root.glob("fold=*/exp=*/order_log.parquet")):
        base = order_path.parent
        fold = order_path.parts[-3].split("=", 1)[-1]
        exp = order_path.parts[-2]
        order_log = pd.read_parquet(order_path)
        window_log = pd.read_parquet(base / "window_log.parquet")
        vehicle_log = pd.read_parquet(base / "vehicle_log.parquet")
        served = int(order_log.get("served", False).fillna(False).sum())
        cancelled = int(order_log.get("cancelled", False).fillna(False).sum())
        vehicle_orders = int(vehicle_log["order_count"].sum()) if "order_count" in vehicle_log else -1
        negative_pending = int(window_log["pending_orders"].lt(0).sum()) if "pending_orders" in window_log else 0
        negative_available = 0
        for column in ["available_AV", "available_HV", "busy_AV", "busy_HV"]:
            if column in window_log:
                negative_available += int(window_log[column].lt(0).sum())
        rows.append({
            "fold": fold,
            "experiment": exp,
            "served_orders": served,
            "cancelled_orders": cancelled,
            "vehicle_order_count_sum": vehicle_orders,
            "vehicle_order_count_matches_served": vehicle_orders == served,
            "negative_pending_windows": negative_pending,
            "negative_vehicle_state_counts": negative_available,
        })
        if vehicle_orders != served:
            errors.append(f"{fold}/{exp}: vehicle order sum {vehicle_orders} != served {served}")
        if negative_pending or negative_available:
            errors.append(f"{fold}/{exp}: negative dynamic state counts")
    pd.DataFrame(rows).to_csv(args.output_root / "dynamic_dispatch_consistency_audit.csv", index=False)
    result = {"status": "PASS" if not errors else "FAIL", "errors": errors, "rows": rows}
    (args.output_root / "dynamic_dispatch_consistency_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": errors[:10]}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
