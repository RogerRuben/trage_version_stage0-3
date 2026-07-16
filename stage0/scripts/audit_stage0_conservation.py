"""Recompute Stage0 time/distance and failed-order accounting from products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_parts(directory: Path, columns: list[str] | None = None) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(directory)
    return pd.concat([pd.read_parquet(path, columns=columns) for path in files], ignore_index=True)


def audit_day(root: Path, date: str, expected_orders: int, tolerance: float) -> tuple[pd.DataFrame, dict]:
    points = read_parts(root / "hmm_matched_points" / f"day={date}")
    traversals = read_parts(root / "hmm_link_traversals" / f"day={date}")
    movements = read_parts(root / "hmm_turn_movements" / f"day={date}")
    expected = points.assign(
        _time=pd.to_numeric(points.dt_s, errors="coerce").fillna(0).clip(lower=0),
        _distance=pd.to_numeric(points.segment_distance_m, errors="coerce").fillna(0).clip(lower=0),
    ).groupby("order_id").agg(
        gps_interval_time_sec=("_time", "sum"),
        gps_interval_distance_m=("_distance", "sum"),
        raw_start_time=("timestamp", "min"),
        raw_end_time=("timestamp", "max"),
    )
    actual = traversals.groupby("order_id").agg(
        traversal_time_sec=("travel_time_sec", "sum"),
        traversal_distance_m=("observed_distance_m", "sum"),
        negative_traversal_time=("travel_time_sec", lambda x: int((x < 0).sum())),
        negative_traversal_distance=("observed_distance_m", lambda x: int((x < 0).sum())),
    )
    result = expected.join(actual, how="outer").fillna(0)
    result["date"] = date
    result["time_allocation_error_sec"] = result.traversal_time_sec - result.gps_interval_time_sec
    result["distance_allocation_error_m"] = result.traversal_distance_m - result.gps_interval_distance_m
    result["relative_time_error"] = result.time_allocation_error_sec.abs() / result.gps_interval_time_sec.clip(lower=1e-12)
    result["relative_distance_error"] = result.distance_allocation_error_m.abs() / result.gps_interval_distance_m.clip(lower=1e-12)
    result["raw_elapsed_time_sec"] = result.raw_end_time - result.raw_start_time
    result["movement_time_allocated_sec"] = 0.0
    failed_files = sorted((root / "failed_orders" / f"day={date}").glob("*.parquet"))
    failed = pd.concat([pd.read_parquet(path, columns=["order_id"]) for path in failed_files], ignore_index=True)
    failed_unique = int(failed.order_id.astype(str).nunique())
    reconstructed_unique = int(result.index.astype(str).nunique())
    overlap = len(set(result.index.astype(str)) & set(failed.order_id.astype(str)))
    summary = {
        "date": date,
        "expected_orders": expected_orders,
        "reconstructed_orders": reconstructed_unique,
        "explicit_failed_orders": failed_unique,
        "reconstructed_failed_overlap": overlap,
        "failure_accounting_complete": reconstructed_unique + failed_unique == expected_orders and overlap == 0,
        "negative_time_count": int(result.negative_traversal_time.sum()),
        "negative_distance_count": int(result.negative_traversal_distance.sum()),
        "time_conservation_failures": int(result.time_allocation_error_sec.abs().gt(tolerance).sum()),
        "distance_conservation_failures": int(result.distance_allocation_error_m.abs().gt(tolerance).sum()),
        "maximum_absolute_time_error_sec": float(result.time_allocation_error_sec.abs().max()),
        "maximum_absolute_distance_error_m": float(result.distance_allocation_error_m.abs().max()),
        "movement_rows": int(len(movements)),
        "movement_time_duplicated": "travel_time_sec" in movements.columns,
    }
    return result.reset_index(), summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dates", required=True)
    parser.add_argument("--expected-orders", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--order-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    details, summaries = [], []
    for date in args.dates.split(","):
        frame, summary = audit_day(args.root, date.strip(), args.expected_orders, args.tolerance)
        details.append(frame)
        summaries.append(summary)
    detail = pd.concat(details, ignore_index=True)
    args.order_output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.order_output, index=False, encoding="utf-8-sig")
    passed = all(
        row["failure_accounting_complete"]
        and row["negative_time_count"] == 0
        and row["negative_distance_count"] == 0
        and row["time_conservation_failures"] == 0
        and row["distance_conservation_failures"] == 0
        and not row["movement_time_duplicated"]
        for row in summaries
    )
    output = {"status": "PASS" if passed else "FAIL", "days": summaries}
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
