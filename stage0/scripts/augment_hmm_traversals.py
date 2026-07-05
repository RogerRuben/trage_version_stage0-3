"""Insert routed-but-unobserved links into HMM traversal and movement products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from build_link_products import build_movements, road_vectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed-traversal-dir", type=Path, required=True)
    parser.add_argument("--route-dir", type=Path, required=True)
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--traversal-output-dir", type=Path, required=True)
    parser.add_argument("--movement-output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roads = gpd.read_parquet(args.roads); vectors = road_vectors(roads)
    road_length = roads.set_index("link_id").length_m
    args.traversal_output_dir.mkdir(parents=True, exist_ok=True)
    args.movement_output_dir.mkdir(parents=True, exist_ok=True)
    totals = {"partitions": 0, "observed": 0, "inferred": 0, "movements": 0}
    for route_path in sorted(args.route_dir.glob("*.parquet")):
        part = route_path.stem.split("=")[-1].split("_")[-1]
        traversal_path = next(iter(args.observed_traversal_dir.glob(f"*{part}.parquet")))
        target = args.traversal_output_dir / f"part={part}.parquet"
        movement_target = args.movement_output_dir / f"part={part}.parquet"
        if target.exists() and movement_target.exists(): continue
        observed = pd.read_parquet(traversal_path)
        routes = pd.read_parquet(route_path)
        observed["_occurrence"] = observed.groupby(["order_id", "link_id"]).cumcount()
        noninterpolated = routes.loc[~routes.is_interpolated, ["order_id", "link_id", "route_sequence"]].copy()
        noninterpolated["_occurrence"] = noninterpolated.groupby(["order_id", "link_id"]).cumcount()
        observed = observed.merge(
            noninterpolated, on=["order_id", "link_id", "_occurrence"], how="left", validate="one_to_one"
        )
        missing_sequence = observed.route_sequence.isna()
        if missing_sequence.any():
            raise ValueError(f"{missing_sequence.sum()} observed traversals could not be aligned in part {part}")
        inferred_route = routes[routes.is_interpolated].copy()
        order_info = observed.groupby("order_id").agg(
            driver_id=("driver_id", "first"), date=("date", "first"), matcher_version=("matcher_version", "first")
        )
        inferred = inferred_route.merge(order_info, left_on="order_id", right_index=True, how="left")
        inferred["link_seq"] = inferred.route_sequence.astype("int32")
        inferred["enter_time"] = inferred.timestamp.astype(float); inferred["exit_time"] = inferred.timestamp.astype(float)
        inferred["travel_time_sec"] = 0.0; inferred["link_length_m"] = inferred.link_id.map(road_length)
        for column in ["mean_speed_mps", "median_speed_mps", "min_speed_mps", "speed_cv", "accel_volatility", "mean_match_dist", "p90_match_dist"]:
            inferred[column] = np.nan
        for column in ["low_speed_time_sec", "low_speed_ratio", "stop_time_sec", "stop_duration_ratio", "observed_distance_m"]:
            inferred[column] = 0.0
        inferred["stop_count"] = 0; inferred["point_count"] = 0
        inferred["traversal_quality"] = "inferred_path"
        columns = [column for column in observed.columns if column not in {"_occurrence", "link_seq"}]
        observed["link_seq"] = observed.route_sequence.astype("int32")
        final_columns = [
            "order_id", "driver_id", "date", "link_id", "link_seq", "enter_time", "exit_time",
            "travel_time_sec", "link_length_m", "mean_speed_mps", "median_speed_mps", "min_speed_mps",
            "low_speed_time_sec", "low_speed_ratio", "stop_time_sec", "stop_count", "stop_duration_ratio",
            "speed_cv", "accel_volatility", "point_count", "mean_match_dist", "p90_match_dist",
            "traversal_quality", "matcher_version", "observed_distance_m",
        ]
        traversals = pd.concat([observed[final_columns], inferred[final_columns]], ignore_index=True)
        traversals = traversals.sort_values(["order_id", "link_seq"], kind="stable").reset_index(drop=True)
        movements = build_movements(traversals, roads, vectors)
        traversals.to_parquet(target, index=False, compression="zstd")
        movements.to_parquet(movement_target, index=False, compression="zstd")
        totals["partitions"] += 1; totals["observed"] += len(observed); totals["inferred"] += len(inferred); totals["movements"] += len(movements)
        print(f"part={part} observed={len(observed):,} inferred={len(inferred):,}", flush=True)
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()

