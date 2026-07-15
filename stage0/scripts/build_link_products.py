"""Build link traversals and turn movements from partitioned matched points."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage0.canonical.intersection import prorate_to_upstream_influence
from stage0.canonical.intervals import allocate_by_projected_mileage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-dir", type=Path, required=True)
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("stage0_output"))
    parser.add_argument("--date", required=True)
    parser.add_argument("--matcher-version", default="geometric_topology")
    parser.add_argument("--low-speed-kmh", type=float, default=5.0)
    parser.add_argument("--stop-speed-kmh", type=float, default=2.0)
    parser.add_argument("--stop-duration-sec", type=float, default=5.0)
    parser.add_argument("--limit-parts", type=int)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--min-part-index", type=int, default=0)
    parser.add_argument("--traversal-collection", default="stage0_link_traversals")
    parser.add_argument("--movement-collection", default="stage0_turn_movements")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def road_vectors(roads: gpd.GeoDataFrame) -> dict[str, np.ndarray]:
    projected = roads.to_crs(32649)
    geoms = projected.geometry.to_numpy()
    starts = shapely.get_point(geoms, 0)
    starts_2 = shapely.line_interpolate_point(geoms, np.minimum(5.0, shapely.length(geoms)))
    ends = shapely.get_point(geoms, -1)
    ends_2 = shapely.line_interpolate_point(geoms, np.maximum(shapely.length(geoms) - 5.0, 0.0))
    return {
        "sx": shapely.get_x(starts), "sy": shapely.get_y(starts),
        "s2x": shapely.get_x(starts_2), "s2y": shapely.get_y(starts_2),
        "ex": shapely.get_x(ends), "ey": shapely.get_y(ends),
        "e2x": shapely.get_x(ends_2), "e2y": shapely.get_y(ends_2),
    }


def build_traversals(frame: pd.DataFrame, roads: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    frame = frame.sort_values(["order_id", "timestamp", "source_row"], kind="stable").reset_index(drop=True)
    same_order = frame.order_id.astype("string").eq(frame.order_id.astype("string").shift()).fillna(False)
    link_changed = frame.link_id.astype("string").ne(frame.link_id.astype("string").shift()).fillna(True)
    new_traversal = (~same_order | link_changed).astype(bool)
    frame["_traversal_id"] = new_traversal.cumsum().astype("int64")
    frame["link_seq"] = new_traversal.groupby(frame.order_id).cumsum().astype("int32") - 1
    dt = frame.dt_s.fillna(0).clip(lower=0, upper=120).astype(float)
    distance = frame.segment_distance_m.fillna(0).clip(lower=0).astype(float)
    # A changed-link interval is shared by its adjacent traversals instead of
    # being assigned wholesale to the later link.  When route-expanded mileage
    # is available it should be supplied as ``transition_mileage_prev_m`` and
    # ``transition_mileage_next_m``; the deterministic equal split is an
    # explicit low-support fallback, and totals remain exactly conserved.
    changed = same_order & new_traversal
    prev_extra_time: dict[int, float] = {}
    prev_extra_distance: dict[int, float] = {}
    prev_exit_time: dict[int, float] = {}
    for idx in np.flatnonzero(changed.to_numpy()):
        mileage = [
            float(frame.at[idx, "transition_mileage_prev_m"])
            if "transition_mileage_prev_m" in frame.columns and pd.notna(frame.at[idx, "transition_mileage_prev_m"])
            else 1.0,
            float(frame.at[idx, "transition_mileage_next_m"])
            if "transition_mileage_next_m" in frame.columns and pd.notna(frame.at[idx, "transition_mileage_next_m"])
            else 1.0,
        ]
        allocated_t, allocated_d = allocate_by_projected_mileage(dt.iat[idx], distance.iat[idx], mileage)
        previous_id = int(frame.at[idx - 1, "_traversal_id"])
        prev_extra_time[previous_id] = prev_extra_time.get(previous_id, 0.0) + float(allocated_t[0])
        prev_extra_distance[previous_id] = prev_extra_distance.get(previous_id, 0.0) + float(allocated_d[0])
        prev_exit_time[previous_id] = float(frame.at[idx, "timestamp"] - allocated_t[1])
        dt.iat[idx] = float(allocated_t[1])
        distance.iat[idx] = float(allocated_d[1])
    frame["_interval_s"] = dt
    frame["_interval_start"] = frame.timestamp - dt
    frame["_low_s"] = dt.where(frame.speed_kmh.lt(args.low_speed_kmh), 0.0)
    frame["_distance_m"] = distance

    same_traversal = frame._traversal_id.eq(frame._traversal_id.shift())
    speed_mps = frame.speed_kmh / 3.6
    accel = (speed_mps.diff() / frame.dt_s).where(same_traversal & frame.dt_s.gt(0) & frame.dt_s.le(10))
    frame["_accel"] = accel

    stop = frame.speed_kmh.lt(args.stop_speed_kmh)
    stop_start = stop & (~stop.shift(fill_value=False) | ~same_traversal)
    frame["_stop_run"] = stop_start.cumsum().astype("int64")
    stop_runs = (
        frame.loc[stop].groupby(["_traversal_id", "_stop_run"], sort=False)._interval_s.sum()
    )
    valid_stops = stop_runs[stop_runs >= args.stop_duration_sec]
    stop_count = valid_stops.groupby(level=0).size()
    stop_time = valid_stops.groupby(level=0).sum()

    group = frame.groupby("_traversal_id", sort=False)
    traversals = group.agg(
        order_id=("order_id", "first"), driver_id=("driver_id", "first"),
        link_id=("link_id", "first"), link_seq=("link_seq", "first"),
        enter_time=("_interval_start", "min"), exit_time=("timestamp", "max"),
        travel_time_sec=("_interval_s", "sum"), observed_distance_m=("_distance_m", "sum"),
        median_speed_kmh=("speed_kmh", "median"), min_speed_kmh=("speed_kmh", "min"),
        speed_std_kmh=("speed_kmh", "std"), low_speed_time_sec=("_low_s", "sum"),
        accel_volatility=("_accel", "std"), point_count=("timestamp", "size"),
        mean_match_dist=("gps_to_link_dist_m", "mean"),
        p90_match_dist=("gps_to_link_dist_m", lambda x: x.quantile(0.9)),
    )
    if prev_extra_time:
        extra_t = pd.Series(prev_extra_time)
        extra_d = pd.Series(prev_extra_distance)
        exit_t = pd.Series(prev_exit_time)
        indexes = traversals.index.intersection(extra_t.index)
        traversals.loc[indexes, "travel_time_sec"] += extra_t.loc[indexes]
        traversals.loc[indexes, "observed_distance_m"] += extra_d.loc[indexes]
        traversals.loc[indexes, "exit_time"] = np.maximum(
            traversals.loc[indexes, "exit_time"].to_numpy(dtype=float),
            exit_t.loc[indexes].to_numpy(dtype=float),
        )
    traversals["stop_count"] = stop_count.reindex(traversals.index, fill_value=0).astype("int32")
    traversals["stop_time_sec"] = stop_time.reindex(traversals.index, fill_value=0.0)
    road_length = roads.set_index("link_id").length_m
    traversals["link_length_m"] = traversals.link_id.map(road_length)
    traversals["mean_speed_mps"] = traversals.observed_distance_m / traversals.travel_time_sec.replace(0, np.nan)
    traversals["median_speed_mps"] = traversals.pop("median_speed_kmh") / 3.6
    traversals["min_speed_mps"] = traversals.pop("min_speed_kmh") / 3.6
    speed_mean_kmh = traversals.mean_speed_mps * 3.6
    traversals["speed_cv"] = traversals.pop("speed_std_kmh") / speed_mean_kmh.replace(0, np.nan)
    traversals["low_speed_ratio"] = traversals.low_speed_time_sec / traversals.travel_time_sec.replace(0, np.nan)
    traversals["stop_duration_ratio"] = traversals.stop_time_sec / traversals.travel_time_sec.replace(0, np.nan)
    traversals["date"] = args.date
    traversals["matcher_version"] = args.matcher_version
    high = traversals.point_count.ge(2) & traversals.p90_match_dist.lt(30) & traversals.travel_time_sec.gt(0)
    usable = traversals.p90_match_dist.lt(50) & traversals.travel_time_sec.ge(0)
    traversals["traversal_quality"] = np.select([high, usable], ["high", "usable"], default="low")
    order = [
        "order_id", "driver_id", "date", "link_id", "link_seq", "enter_time", "exit_time",
        "travel_time_sec", "link_length_m", "mean_speed_mps", "median_speed_mps", "min_speed_mps",
        "low_speed_time_sec", "low_speed_ratio", "stop_time_sec", "stop_count", "stop_duration_ratio",
        "speed_cv", "accel_volatility", "point_count", "mean_match_dist", "p90_match_dist",
        "traversal_quality", "matcher_version", "observed_distance_m",
    ]
    return traversals.reset_index(drop=True)[order]


def build_movements(traversals: pd.DataFrame, roads: gpd.GeoDataFrame, vectors: dict[str, np.ndarray]) -> pd.DataFrame:
    roads = roads.reset_index(drop=True)
    lookup = pd.Series(np.arange(len(roads), dtype="int32"), index=roads.link_id.astype(str))
    current = traversals.copy()
    same_next = current.order_id.eq(current.order_id.shift(-1))
    current = current.loc[same_next].copy()
    current["to_link_id"] = traversals.link_id.shift(-1).loc[current.index].to_numpy()
    current["movement_seq"] = current.groupby("order_id").cumcount().astype("int32")
    fi = current.link_id.astype(str).map(lookup).to_numpy(dtype="int32")
    ti = current.to_link_id.astype(str).map(lookup).to_numpy(dtype="int32")
    fu = roads.from_node.to_numpy()[fi]; fv = roads.to_node.to_numpy()[fi]
    tu = roads.from_node.to_numpy()[ti]; tv = roads.to_node.to_numpy()[ti]
    node = np.select([fu == tu, fu == tv, fv == tu, fv == tv], [fu, fu, fv, fv], default=-1).astype("int64")

    ax = np.where(node == fv, vectors["ex"][fi] - vectors["e2x"][fi], vectors["sx"][fi] - vectors["s2x"][fi])
    ay = np.where(node == fv, vectors["ey"][fi] - vectors["e2y"][fi], vectors["sy"][fi] - vectors["s2y"][fi])
    bx = np.where(node == tu, vectors["s2x"][ti] - vectors["sx"][ti], vectors["e2x"][ti] - vectors["ex"][ti])
    by = np.where(node == tu, vectors["s2y"][ti] - vectors["sy"][ti], vectors["e2y"][ti] - vectors["ey"][ti])
    dot = ax * bx + ay * by
    cross = ax * by - ay * bx
    angle = np.degrees(np.arctan2(cross, dot))
    angle[node < 0] = np.nan
    abs_angle = np.abs(angle)
    turn_type = np.select(
        [abs_angle < 30, abs_angle >= 150, angle >= 30, angle <= -30],
        ["straight", "u_turn", "left", "right"], default="unknown",
    )
    degree = pd.concat([roads.from_node, roads.to_node]).value_counts()
    node_degree = pd.Series(node).map(degree).fillna(0).to_numpy(dtype="int16")
    quality = np.where((node >= 0) & current.traversal_quality.ne("low"), "usable", "low")
    influence_distance_m = 75.0
    intersection_low = np.array([
        prorate_to_upstream_influence(value, length, influence_distance_m)
        for value, length in zip(current.low_speed_time_sec, current.link_length_m)
    ])
    intersection_stop = np.array([
        prorate_to_upstream_influence(value, length, influence_distance_m)
        for value, length in zip(current.stop_time_sec, current.link_length_m)
    ])
    return pd.DataFrame({
        "order_id": current.order_id.to_numpy(),
        "from_link_id": current.link_id.to_numpy(), "to_link_id": current.to_link_id.to_numpy(),
        "node_id": pd.array(np.where(node < 0, None, node), dtype="Int64"),
        "movement_seq": current.movement_seq.to_numpy(),
        "turn_angle": angle, "turn_type": turn_type, "node_degree": node_degree,
        "junction_complexity": np.log1p(node_degree),
        "approach_delay_sec": current.low_speed_time_sec.to_numpy(),
        "intersection_low_speed_time": intersection_low,
        "intersection_stop_time": intersection_stop,
        "intersection_influence_distance_m": influence_distance_m,
        "intersection_measurement_method": "upstream_length_prorated_75m",
        "movement_quality": quality,
    })


def main() -> None:
    args = parse_args()
    roads = gpd.read_parquet(args.roads)
    vectors = road_vectors(roads)
    files = sorted(args.matched_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet partitions in {args.matched_dir}")
    all_files = files
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker-index must be in [0, worker-count)")
    files = [
        path for index, path in enumerate(files)
        if index >= args.min_part_index and index % args.worker_count == args.worker_index
    ]
    if args.limit_parts is not None:
        files = files[: args.limit_parts]
    traversal_dir = args.output_root / args.traversal_collection / f"day={args.date}"
    movement_dir = args.output_root / args.movement_collection / f"day={args.date}"
    traversal_dir.mkdir(parents=True, exist_ok=True); movement_dir.mkdir(parents=True, exist_ok=True)
    totals = {"partitions": 0, "traversals": 0, "movements": 0}
    for source in files:
        part = source.stem.split("=")[-1].split("_")[-1]
        traversal_path = traversal_dir / f"part={part}.parquet"
        movement_path = movement_dir / f"part={part}.parquet"
        if traversal_path.exists() and movement_path.exists() and not args.force:
            continue
        frame = pd.read_parquet(source)
        traversals = build_traversals(frame, roads, args)
        movements = build_movements(traversals, roads, vectors)
        traversals.to_parquet(traversal_path, index=False, compression="zstd")
        movements.to_parquet(movement_path, index=False, compression="zstd")
        totals["partitions"] += 1; totals["traversals"] += len(traversals); totals["movements"] += len(movements)
        print(f"part={part} traversals={len(traversals):,} movements={len(movements):,}", flush=True)
    manifest_dir = args.output_root / "manifests"; manifest_dir.mkdir(parents=True, exist_ok=True)
    totals.update({
        "date": args.date,
        "complete": len(list(traversal_dir.glob('part=*.parquet'))) == len(all_files),
        "matcher_version": args.matcher_version,
    })
    manifest_name = f"day={args.date}.link_products.json" if args.worker_count == 1 else f"day={args.date}.link_products.worker={args.worker_index}.json"
    totals.update({"worker_count": args.worker_count, "worker_index": args.worker_index})
    (manifest_dir / manifest_name).write_text(
        json.dumps(totals, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
