"""Standardize one completed geometric Stage0 day into partitioned monthly outputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("stage0_output"))
    parser.add_argument("--date", required=True, help="YYYYMMDD")
    parser.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink")
    return parser.parse_args()


def materialize(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    if mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def add_quality_tier(orders: pd.DataFrame) -> pd.DataFrame:
    orders = orders.copy()
    transitions = (orders["clean_point_count"] - 1).clip(lower=1)
    gap_rate = orders["topology_gap_count"].fillna(0) / transitions
    orders["topology_gap_rate"] = gap_rate
    a = (
        orders["p90_gps_to_link_dist_m"].lt(20)
        & orders["route_length_ratio"].between(0.8, 1.3)
        & gap_rate.le(0.01)
        & orders["matching_confidence"].ge(0.7)
    )
    b = (
        orders["p90_gps_to_link_dist_m"].lt(50)
        & orders["route_length_ratio"].between(0.6, 1.6)
        & gap_rate.le(0.05)
        & orders["matching_confidence"].ge(0.4)
    )
    orders["quality_tier"] = "C"
    orders.loc[b, "quality_tier"] = "B"
    orders.loc[a, "quality_tier"] = "A"
    return orders


def approximate_point_quantiles(files: list[Path], column: str, maximum: float, step: float = 0.1) -> dict[float, float]:
    edges = np.arange(0, maximum + step, step)
    counts = np.zeros(len(edges) - 1, dtype="int64")
    total = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.num_row_groups):
            values = parquet.read_row_group(row_group, columns=[column]).column(0).to_numpy()
            values = values[np.isfinite(values)]
            values = np.clip(values, 0, maximum - step / 10)
            counts += np.histogram(values, bins=edges)[0]; total += len(values)
    cumulative = np.cumsum(counts)
    result = {}
    for quantile in [0.5, 0.9, 0.95]:
        index = int(np.searchsorted(cumulative, total * quantile, side="left"))
        result[quantile] = float(edges[min(index, len(edges) - 2)])
    return result


def make_report(
    orders: pd.DataFrame, gps_points: int, date: str,
    point_distance: dict[float, float], point_dt: dict[float, float],
) -> str:
    dist = orders["p90_gps_to_link_dist_m"].quantile([0.5, 0.9, 0.95])
    ratio = orders["route_length_ratio"].quantile([0.1, 0.5, 0.9])
    tiers = orders["quality_tier"].value_counts(normalize=True).reindex(["A", "B", "C"], fill_value=0)
    return f"""# Stage0 daily quality report: {date}

| Metric | Value |
|---|---:|
| Orders | {len(orders):,} |
| GPS points | {gps_points:,} |
| Point sampling interval P50 / P90 / P95 | {point_dt[0.5]:.2f} / {point_dt[0.9]:.2f} / {point_dt[0.95]:.2f} s |
| Point GPS-link distance P50 / P90 / P95 | {point_distance[0.5]:.2f} / {point_distance[0.9]:.2f} / {point_distance[0.95]:.2f} m |
| Order-level P90 distance distribution P50 / P90 / P95 | {dist.loc[0.5]:.2f} / {dist.loc[0.9]:.2f} / {dist.loc[0.95]:.2f} m |
| Route-length ratio P10 / P50 / P90 | {ratio.loc[0.1]:.3f} / {ratio.loc[0.5]:.3f} / {ratio.loc[0.9]:.3f} |
| Topology gaps | {orders['topology_gap_count'].fillna(0).sum():,.0f} |
| Quality A / B / C | {tiers.A:.2%} / {tiers.B:.2%} / {tiers.C:.2%} |

Quality tiers use matching distance, route-length consistency, topology-gap rate, and matching confidence. Topology gaps and route-length ratio are quality controls, not stress labels.
"""


def main() -> None:
    args = parse_args()
    source = args.source_dir
    output = args.output_root
    orders = add_quality_tier(pd.read_parquet(source / "full_day_stage0_orders.parquet"))
    order_path = output / "order_base" / f"day={args.date}.parquet"
    order_path.parent.mkdir(parents=True, exist_ok=True)
    orders.to_parquet(order_path, index=False, compression="zstd")

    point_files = sorted((source / "matched_points").glob("part_*.parquet"))
    route_files = sorted((source / "route_parts").glob("part_*.parquet"))
    if not point_files or not route_files:
        raise FileNotFoundError("source-dir must contain matched_points and route_parts partitions")
    for path in point_files:
        bucket = path.stem.split("_")[-1]
        materialize(path, output / "matched_points" / f"day={args.date}" / f"bucket={bucket}.parquet", args.link_mode)
    for path in route_files:
        part = path.stem.split("_")[-1]
        materialize(path, output / "route_parts" / f"day={args.date}" / f"part={part}.parquet", args.link_mode)

    gps_points = sum(pq.ParquetFile(path).metadata.num_rows for path in point_files)
    point_distance = approximate_point_quantiles(point_files, "gps_to_link_dist_m", maximum=500, step=0.1)
    point_dt = approximate_point_quantiles(point_files, "dt_s", maximum=300, step=0.1)
    report = make_report(orders, gps_points, args.date, point_distance, point_dt)
    report_path = output / "quality_reports" / f"day={args.date}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    manifest = {
        "date": args.date,
        "complete": True,
        "matcher_version": "geometric_topology",
        "orders": int(len(orders)),
        "gps_points": int(gps_points),
        "point_partitions": len(point_files),
        "route_partitions": len(route_files),
    }
    manifest_dir = output / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"day={args.date}.standardize.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
