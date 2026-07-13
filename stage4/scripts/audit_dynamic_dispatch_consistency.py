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
        final_total = served + cancelled
        total_rows = len(order_log)
        vehicle_orders = int(vehicle_log["order_count"].sum()) if "order_count" in vehicle_log else -1
        negative_pending = int(window_log["pending_orders"].lt(0).sum()) if "pending_orders" in window_log else 0
        negative_available = 0
        for column in ["available_AV", "available_HV", "busy_AV", "busy_HV"]:
            if column in window_log:
                negative_available += int(window_log[column].lt(0).sum())
        overlap_count = 0
        if served and {"assigned_vehicle", "decision_time", "waiting_time_sec", "pickup_time_sec", "service_time_sec"}.issubset(order_log.columns):
            served_rows = order_log[order_log["served"].fillna(False)].copy()
            served_rows["start_time"] = pd.to_datetime(served_rows["decision_time"], errors="coerce") + pd.to_timedelta(served_rows["waiting_time_sec"].fillna(0), unit="s")
            served_rows["end_time"] = served_rows["start_time"] + pd.to_timedelta(served_rows["pickup_time_sec"].fillna(0) + served_rows["service_time_sec"].fillna(0), unit="s")
            for _, vehicle_group in served_rows.sort_values("start_time").groupby("assigned_vehicle"):
                previous_end = None
                for _, row in vehicle_group.iterrows():
                    if previous_end is not None and row["start_time"] < previous_end:
                        overlap_count += 1
                    previous_end = max(previous_end, row["end_time"]) if previous_end is not None else row["end_time"]
        rows.append({
            "fold": fold,
            "experiment": exp,
            "served_orders": served,
            "cancelled_orders": cancelled,
            "served_plus_cancelled": final_total,
            "order_log_rows": total_rows,
            "final_status_covers_all_orders": final_total == total_rows,
            "vehicle_order_count_sum": vehicle_orders,
            "vehicle_order_count_matches_served": vehicle_orders == served,
            "vehicle_service_overlap_count": overlap_count,
            "negative_pending_windows": negative_pending,
            "negative_vehicle_state_counts": negative_available,
        })
        if vehicle_orders != served:
            errors.append(f"{fold}/{exp}: vehicle order sum {vehicle_orders} != served {served}")
        if final_total != total_rows:
            errors.append(f"{fold}/{exp}: served+cancelled {final_total} != order rows {total_rows}")
        if overlap_count:
            errors.append(f"{fold}/{exp}: vehicle service overlap count={overlap_count}")
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
