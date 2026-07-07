"""Audit per-day label coverage and dimension-specific cohort fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DIMENSIONS = ["lcs", "iis", "gns", "rts", "pmis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    return parser.parse_args()


def split_lookup(config: dict) -> dict[str, str]:
    result = {date: "train" for date in config["train_dates"]}
    result[config["validation_date"]] = "validation"
    result[config["test_date"]] = "test"
    return result


def main() -> None:
    args = parse_args()
    config = json.loads(args.split_config.read_text(encoding="utf-8"))
    splits = split_lookup(config)
    rows = []
    for date, split in splits.items():
        files = sorted((args.output_root / "link_labels" / f"day={date}").glob("*.parquet"))
        if not files:
            rows.append({"date": date, "split": split, "status": "missing"}); continue
        total = 0
        nonnull = {dimension: 0 for dimension in DIMENSIONS}
        levels = {dimension: {level: 0 for level in range(1, 7)} for dimension in DIMENSIONS}
        quality = {name: 0 for name in ["inferred_path", "low", "high", "usable"]}
        columns = ["traversal_quality"] + [
            column for dimension in DIMENSIONS
            for column in [f"{dimension}_pct_link", f"{dimension}_cohort_level_used"]
        ]
        for path in files:
            frame = pd.read_parquet(path, columns=columns)
            total += len(frame)
            counts = frame.traversal_quality.value_counts()
            for name in quality: quality[name] += int(counts.get(name, 0))
            for dimension in DIMENSIONS:
                valid = frame[f"{dimension}_pct_link"].notna()
                nonnull[dimension] += int(valid.sum())
                value_counts = frame.loc[valid, f"{dimension}_cohort_level_used"].value_counts()
                for level in range(1, 7): levels[dimension][level] += int(value_counts.get(level, 0))
        row = {"date": date, "split": split, "status": "complete", "total_link_rows": total}
        for dimension in DIMENSIONS:
            row[f"{dimension}_nonnull_ratio"] = nonnull[dimension] / total if total else 0
            for level in range(1, 7):
                denominator = nonnull[dimension]
                row[f"{dimension}_cohort_level_{level}_ratio"] = (
                    levels[dimension][level] / denominator if denominator else 0
                )
        row["inferred_path_ratio"] = quality["inferred_path"] / total if total else 0
        row["low_quality_ratio"] = quality["low"] / total if total else 0
        row["high_traversal_quality_ratio"] = quality["high"] / total if total else 0
        row["usable_traversal_quality_ratio"] = quality["usable"] / total if total else 0
        order_path = args.output_root / "order_labels" / f"day={date}.parquet"
        if order_path.exists():
            orders = pd.read_parquet(order_path, columns=[
                "high_odd_exceedance_85", "high_odd_exceedance_90", "high_odd_exceedance_95"
            ])
            for threshold in [85, 90, 95]:
                row[f"high_odd_exceedance_{threshold}_ratio"] = float(orders[f"high_odd_exceedance_{threshold}"].mean())
        rows.append(row)
    report = pd.DataFrame(rows)
    reports = args.output_root / "reports"; reports.mkdir(parents=True, exist_ok=True)
    report.to_csv(reports / "stage1_label_coverage.csv", index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
