"""Daily readiness waterfall for Stage3 rolling inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", required=True, help="Comma-separated YYYYMMDD dates.")
    parser.add_argument("--order-base-root", type=Path, default=Path("stage0/output/order_base"))
    parser.add_argument("--route-conditioned-root", type=Path, default=Path("stage2/output/route_conditioned_dataset_15k/estimated_time_daily"))
    parser.add_argument("--strict-target-root", type=Path, default=Path("stage2/output/strict_targets"))
    parser.add_argument("--heldout-root", type=Path, default=Path("stage3/output/stage2_heldout_daily_20161017_23"))
    parser.add_argument("--warehouse-root", type=Path, default=Path("stage3/output/rolling_stage2_prediction_warehouse"))
    parser.add_argument("--output-root", type=Path, default=Path("stage3/output/daily_readiness"))
    return parser.parse_args()


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path, columns=columns)


def day_parts(root: Path, date: str, columns: list[str] | None = None) -> pd.DataFrame:
    paths = sorted((root / f"day={date}").glob("*.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True)


def warehouse_day(root: Path, family: str, date: str, columns: list[str] | None = None) -> pd.DataFrame:
    parts = []
    for path in sorted((root / family).glob(f"fold=*/split=*/day={date}.parquet")):
        parts.append(pd.read_parquet(path, columns=columns))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def order_count(frame: pd.DataFrame) -> int:
    return int(frame["order_id"].nunique()) if not frame.empty and "order_id" in frame else 0


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    for date in [value.strip() for value in args.dates.split(",") if value.strip()]:
        order_base = read_parquet(args.order_base_root / f"day={date}.parquet", columns=["order_id"])
        route = read_parquet(args.route_conditioned_root / f"day={date}.parquet", columns=["order_id", "route_link_id", "route_link_seq"])
        strict = day_parts(args.strict_target_root, date, columns=["order_id", "link_id", "link_seq"])
        link = read_parquet(args.heldout_root / "link_predictions" / f"day={date}.parquet", columns=["order_id", "route_link_id", "route_link_seq", "prediction_date_after_train_end"])
        movement = read_parquet(args.heldout_root / "movement_predictions" / f"day={date}.parquet", columns=["order_id", "movement_seq"])
        wh_link = warehouse_day(args.warehouse_root, "link_predictions", date, columns=["order_id", "route_link_id", "route_link_seq"])
        wh_movement = warehouse_day(args.warehouse_root, "movement_predictions", date, columns=["order_id", "movement_seq"])
        raw_orders = order_count(order_base)
        route_orders = order_count(route)
        strict_orders = order_count(strict)
        heldout_orders = order_count(link)
        iis_orders = order_count(movement)
        joined_orders = order_count(wh_link)
        duplicate_link_keys = int(link.duplicated(["order_id", "route_link_id", "route_link_seq"]).sum()) if not link.empty else 0
        duplicate_movement_keys = int(movement.duplicated(["order_id", "movement_seq"]).sum()) if not movement.empty else 0
        leakage = "PASS" if (not link.empty and link["prediction_date_after_train_end"].fillna(False).all()) else "FAIL"
        ready = (
            raw_orders > 0
            and route_orders >= 14000
            and heldout_orders >= 14000
            and joined_orders >= 14000
            and iis_orders / max(1, heldout_orders) >= 0.95
            and duplicate_link_keys == 0
            and duplicate_movement_keys == 0
            and leakage == "PASS"
        )
        if not ready:
            errors.append(f"day={date} readiness failed")
        rows.append({
            "day": date,
            "raw_orders": raw_orders,
            "route_conditioned_orders": route_orders,
            "strict_target_orders": strict_orders,
            "heldout_prediction_orders": heldout_orders,
            "IIS_movement_orders": iis_orders,
            "Stage3_joined_orders": joined_orders,
            "route_cov": route_orders / max(1, raw_orders),
            "target_cov": strict_orders / max(1, raw_orders),
            "heldout_cov": heldout_orders / max(1, raw_orders),
            "IIS_cov": iis_orders / max(1, heldout_orders),
            "joined_cov": joined_orders / max(1, heldout_orders),
            "warehouse_movement_orders": order_count(wh_movement),
            "duplicate_link_keys": duplicate_link_keys,
            "duplicate_movement_keys": duplicate_movement_keys,
            "leakage": leakage,
            "ready": "PASS" if ready else "FAIL",
        })
    table = pd.DataFrame(rows)
    table.to_csv(args.output_root / "daily_readiness_summary.csv", index=False)
    report = ["# Stage3 daily readiness", "", table.to_markdown(index=False, floatfmt=".4f"), ""]
    if errors:
        report += ["## Errors", "", *[f"- {error}" for error in errors]]
    (args.output_root / "daily_readiness_report.md").write_text("\n".join(report), encoding="utf-8")
    result = {"status": "PASS" if not errors else "FAIL", "errors": errors, "rows": rows}
    (args.output_root / "daily_readiness_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": errors[:10]}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
