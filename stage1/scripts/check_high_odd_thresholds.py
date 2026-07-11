"""Audit high_odd_exceedance threshold monotonicity and cutoff-specific support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DIMENSIONS = ["lcs", "iis", "gns", "rts", "pmis"]
THRESHOLDS = [85, 90, 95]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("stage1/output/prediction_split"))
    parser.add_argument("--split-config", type=Path, default=Path("split_config.json"))
    return parser.parse_args()


def split_lookup(config: dict) -> dict[str, str]:
    result = {date: "train" for date in config["train_dates"]}
    result[config["validation_date"]] = "validation"
    result[config["test_date"]] = "test"
    return result


def read_order_flags(path: Path) -> pd.DataFrame:
    columns = [f"high_odd_exceedance_{threshold}" for threshold in THRESHOLDS]
    return pd.read_parquet(path, columns=columns)


def link_threshold_support(label_dir: Path) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for path in sorted(label_dir.glob("*.parquet")):
        columns = [f"{dimension}_pct_link" for dimension in DIMENSIONS]
        frame = pd.read_parquet(path, columns=columns)
        for dimension in DIMENSIONS:
            series = frame[f"{dimension}_pct_link"].dropna()
            counts[f"{dimension}_valid_rows"] = counts.get(f"{dimension}_valid_rows", 0) + int(len(series))
            totals[f"{dimension}_pct_085_090_rows"] = totals.get(f"{dimension}_pct_085_090_rows", 0) + int(
                series.between(0.85, 0.90, inclusive="left").sum()
            )
            for threshold in [0.85, 0.90, 0.95]:
                key = f"{dimension}_ge_{int(threshold * 100)}_rows"
                totals[key] = totals.get(key, 0) + int(series.ge(threshold).sum())
    return {**totals, **counts}


def main() -> None:
    args = parse_args()
    config = json.loads(args.split_config.read_text(encoding="utf-8"))
    splits = split_lookup(config)
    rows = []
    for date, split in splits.items():
        order_path = args.output_root / "order_labels" / f"day={date}.parquet"
        label_dir = args.output_root / "link_labels" / f"day={date}"
        row: dict[str, object] = {"date": date, "split": split}
        if not order_path.exists() or not label_dir.exists():
            row["status"] = "missing"
            rows.append(row)
            continue
        orders = read_order_flags(order_path)
        row["status"] = "complete"
        row["orders"] = len(orders)
        for threshold in THRESHOLDS:
            column = f"high_odd_exceedance_{threshold}"
            row[f"{column}_ratio"] = float(orders[column].mean()) if len(orders) else 0.0
        row["order_flags_monotone"] = bool(
            ((orders.high_odd_exceedance_85.astype(int) >= orders.high_odd_exceedance_90.astype(int))
             & (orders.high_odd_exceedance_90.astype(int) >= orders.high_odd_exceedance_95.astype(int))).all()
        )
        row["order_85_equals_90_ratio"] = float(
            (orders.high_odd_exceedance_85 == orders.high_odd_exceedance_90).mean()
        ) if len(orders) else 0.0
        row.update(link_threshold_support(label_dir))
        rows.append(row)

    report = pd.DataFrame(rows)
    reports = args.output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / "high_odd_threshold_check.csv"
    report.to_csv(csv_path, index=False)

    lines = ["# High ODD threshold audit", ""]
    for row in report.itertuples(index=False):
        if getattr(row, "status") != "complete":
            lines.append(f"- {row.date}: missing inputs.")
            continue
        lines.append(
            f"- {row.date} ({row.split}): 85={getattr(row, 'high_odd_exceedance_85_ratio'):.2%}, "
            f"90={getattr(row, 'high_odd_exceedance_90_ratio'):.2%}, "
            f"95={getattr(row, 'high_odd_exceedance_95_ratio'):.2%}, "
            f"monotone={getattr(row, 'order_flags_monotone')}, "
            f"85==90 orders={getattr(row, 'order_85_equals_90_ratio'):.2%}."
        )
    (reports / "high_odd_threshold_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
