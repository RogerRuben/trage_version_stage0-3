"""Summarize observed/inferred link traversal coverage for all completed days."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLUMNS = [
    "date", "orders", "observed_traversals", "inferred_traversals",
    "observed_traversal_ratio", "inferred_traversal_ratio",
    "high_traversal_quality_ratio", "usable_traversal_quality_ratio",
    "low_traversal_quality_ratio", "mean_links_per_order",
    "median_links_per_order", "p90_links_per_order",
    "observed_link_length_m", "inferred_link_length_m",
    "observed_link_length_ratio", "inferred_link_length_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--traversal-collection", default="hmm_link_traversals")
    parser.add_argument("--report-name", default="link_level_quality_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); rows = []
    for day_dir in sorted((args.output_root / args.traversal_collection).glob("day=*")):
        date = day_dir.name.split("=")[1]
        order_counts = []
        total = inferred = high = usable = low = 0
        observed_length = inferred_length = 0.0
        orders = set()
        for path in sorted(day_dir.glob("*.parquet")):
            frame = pd.read_parquet(path, columns=["order_id", "traversal_quality", "link_length_m"])
            total += len(frame); inferred += int(frame.traversal_quality.eq("inferred_path").sum())
            inferred_mask = frame.traversal_quality.eq("inferred_path")
            inferred_length += float(frame.loc[inferred_mask, "link_length_m"].sum())
            observed_length += float(frame.loc[~inferred_mask, "link_length_m"].sum())
            high += int(frame.traversal_quality.eq("high").sum())
            usable += int(frame.traversal_quality.eq("usable").sum())
            low += int(frame.traversal_quality.eq("low").sum())
            counts = frame.groupby("order_id").size(); order_counts.append(counts); orders.update(counts.index.astype(str))
        if not order_counts:
            continue
        links = pd.concat(order_counts)
        observed = total - inferred
        rows.append({
            "date": date, "orders": len(orders), "observed_traversals": observed,
            "inferred_traversals": inferred, "observed_traversal_ratio": observed / total if total else 0,
            "inferred_traversal_ratio": inferred / total if total else 0,
            "high_traversal_quality_ratio": high / total if total else 0,
            "usable_traversal_quality_ratio": usable / total if total else 0,
            "low_traversal_quality_ratio": low / total if total else 0,
            "mean_links_per_order": float(links.mean()), "median_links_per_order": float(links.median()),
            "p90_links_per_order": float(links.quantile(0.9)),
            "observed_link_length_m": observed_length, "inferred_link_length_m": inferred_length,
            "observed_link_length_ratio": observed_length / (observed_length + inferred_length),
            "inferred_link_length_ratio": inferred_length / (observed_length + inferred_length),
        })
    report = pd.DataFrame(rows, columns=COLUMNS)
    reports = args.output_root / "reports"; reports.mkdir(parents=True, exist_ok=True)
    report.to_csv(reports / args.report_name, index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
