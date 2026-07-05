"""Correct endpoint-only topology diagnostics using direct line adjacency."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from run_full_day_2017 import make_reports


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    roads = gpd.read_parquet(args.roads).to_crs(32649).reset_index(drop=True)
    geometries = roads.geometry.to_numpy()
    lookup = dict(zip(roads.link_id.astype(str), range(len(roads))))
    point_dir = args.output_dir / "matched_points"
    order_dir = args.output_dir / "order_parts"
    for point_path in sorted(point_dir.glob("part_*.parquet")):
        bucket = point_path.stem.split("_")[-1]
        order_path = order_dir / f"part_{bucket}.parquet"
        points = pd.read_parquet(point_path)
        road_idx = points.link_id.astype(str).map(lookup).to_numpy(dtype=int)
        previous = np.roll(road_idx, 1)
        same_order = points.order_id.eq(points.order_id.shift()).fillna(False).to_numpy(dtype=bool)
        changed = same_order & (road_idx != previous)
        changed_idx = np.flatnonzero(changed)
        adjacent = np.zeros(len(points), dtype=bool)
        if len(changed_idx):
            distance = shapely.distance(geometries[previous[changed_idx]], geometries[road_idx[changed_idx]])
            adjacent[changed_idx] = distance <= 2.0
        points["topology_gap"] = points.topology_gap.fillna(False).to_numpy(dtype=bool) & ~adjacent
        points["parallel_jump"] = points.parallel_jump.fillna(False).to_numpy(dtype=bool) & ~adjacent
        points.to_parquet(point_path, index=False, compression="zstd")

        orders = pd.read_parquet(order_path).set_index("order_id")
        gap = points.groupby("order_id").topology_gap.sum().astype(float)
        parallel = points.groupby("order_id").parallel_jump.sum().astype(float)
        orders["topology_gap_count"] = gap.reindex(orders.index, fill_value=0)
        orders["parallel_jump_count"] = parallel.reindex(orders.index, fill_value=0)
        transitions = (orders.clean_point_count - 1).clip(lower=1).astype(float)
        route_score = np.exp(
            -np.abs(np.log(orders.route_length_ratio.clip(lower=1e-6).to_numpy(dtype=float))) / 0.45
        )
        orders["matching_confidence"] = (
            orders.matched_point_ratio.to_numpy(dtype=float)
            * np.exp(-orders.mean_gps_to_link_dist_m.to_numpy(dtype=float) / 35)
            * route_score
            * np.exp(-8 * orders.topology_gap_count.to_numpy(dtype=float) / transitions.to_numpy())
            * np.exp(-4 * orders.parallel_jump_count.to_numpy(dtype=float) / transitions.to_numpy())
        )
        orders.reset_index().to_parquet(order_path, index=False, compression="zstd")
        print(f"corrected bucket {bucket}", flush=True)

    class Args:
        output_dir = args.output_dir
        input_crs = "gcj02"

    dirs = {
        "points": point_dir,
        "orders": order_dir,
        "figures": args.output_dir / "figures",
    }
    manifest = __import__("json").loads(
        (args.output_dir / "order_buckets" / "manifest.json").read_text(encoding="utf-8")
    )
    make_reports(Args(), dirs, manifest)


if __name__ == "__main__":
    main()
