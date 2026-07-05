"""Scalable full-day cleaning, matching, semantic fusion, and feature extraction."""

from __future__ import annotations

import argparse
import json
import math
import time
import tarfile
from contextlib import ExitStack
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Point


COLUMNS = ["driver_id", "order_id", "timestamp", "lon", "lat"]
MAX_SPEED_KMH = 150.0
MATCH_RADIUS_M = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--buckets", type=int, default=128)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--input-crs", choices=["wgs84", "gcj02"], default="gcj02")
    parser.add_argument("--limit-buckets", type=int, help="For benchmark/debug runs only")
    parser.add_argument("--max-input-chunks", type=int, help="For streaming smoke tests only")
    return parser.parse_args()


def gcj02_to_wgs84(lon, lat):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    x, y = lon - 105.0, lat - 35.0
    dlat = -100 + 2 * x + 3 * y + 0.2 * y**2 + 0.1 * x * y + 0.2 * np.sqrt(np.abs(x))
    dlat += (20 * np.sin(6 * x * np.pi) + 20 * np.sin(2 * x * np.pi)) * 2 / 3
    dlat += (20 * np.sin(y * np.pi) + 40 * np.sin(y / 3 * np.pi)) * 2 / 3
    dlat += (160 * np.sin(y / 12 * np.pi) + 320 * np.sin(y * np.pi / 30)) * 2 / 3
    dlon = 300 + x + 2 * y + 0.1 * x**2 + 0.1 * x * y + 0.1 * np.sqrt(np.abs(x))
    dlon += (20 * np.sin(6 * x * np.pi) + 20 * np.sin(2 * x * np.pi)) * 2 / 3
    dlon += (20 * np.sin(x * np.pi) + 40 * np.sin(x / 3 * np.pi)) * 2 / 3
    dlon += (150 * np.sin(x / 12 * np.pi) + 300 * np.sin(x / 30 * np.pi)) * 2 / 3
    a, ee = 6_378_245.0, 0.00669342162296594323
    rad = np.radians(lat)
    magic = 1 - ee * np.sin(rad) ** 2
    sqrt_magic = np.sqrt(magic)
    dlat = dlat * 180 / ((a * (1 - ee)) / (magic * sqrt_magic) * np.pi)
    dlon = dlon * 180 / (a / sqrt_magic * np.cos(rad) * np.pi)
    return lon - dlon, lat - dlat


def order_bucket(values: pd.Series, buckets: int) -> np.ndarray:
    return (pd.util.hash_pandas_object(values, index=False).to_numpy(dtype="uint64") % buckets).astype("int16")


def bucketize(args: argparse.Namespace, bucket_dir: Path) -> dict:
    manifest_path = bucket_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") and manifest.get("buckets") == args.buckets:
            print("bucketization: using completed manifest", flush=True)
            return manifest
    bucket_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[int, pq.ParquetWriter] = {}
    counts = np.zeros(args.buckets, dtype="int64")
    source_offset = 0
    started = time.time()
    try:
        with ExitStack() as stack:
            source: object = args.input
            if args.input.name.lower().endswith((".tar.gz", ".tgz")):
                archive = stack.enter_context(tarfile.open(args.input, mode="r:gz"))
                members = [member for member in archive.getmembers() if member.isfile() and "/gps_" in member.name]
                if len(members) != 1:
                    raise ValueError(f"expected one xian GPS member in {args.input}, found {len(members)}")
                extracted = archive.extractfile(members[0])
                if extracted is None:
                    raise OSError(f"could not stream {members[0].name}")
                source = stack.enter_context(extracted)
            reader = pd.read_csv(
                source, header=None, names=COLUMNS, chunksize=args.chunksize,
                dtype={"driver_id": "string", "order_id": "string"},
            )
            for chunk_no, chunk in enumerate(reader, start=1):
                chunk["source_row"] = np.arange(source_offset, source_offset + len(chunk), dtype="int64")
                source_offset += len(chunk)
                chunk["bucket"] = order_bucket(chunk.order_id, args.buckets)
                for bucket, frame in chunk.groupby("bucket", sort=False):
                    bucket = int(bucket)
                    table = pa.Table.from_pandas(frame.drop(columns="bucket"), preserve_index=False)
                    if bucket not in writers:
                        writers[bucket] = pq.ParquetWriter(
                            bucket_dir / f"bucket_{bucket:03d}.parquet", table.schema,
                            compression="zstd", use_dictionary=["driver_id", "order_id"],
                        )
                    writers[bucket].write_table(table)
                    counts[bucket] += len(frame)
                print(
                    f"bucketize chunk={chunk_no} rows={source_offset:,} elapsed={time.time()-started:.1f}s",
                    flush=True,
                )
                if args.max_input_chunks is not None and chunk_no >= args.max_input_chunks:
                    break
    finally:
        for writer in writers.values():
            writer.close()
    manifest = {
        "complete": args.max_input_chunks is None,
        "source": str(args.input.resolve()),
        "rows": int(source_offset),
        "buckets": args.buckets,
        "bucket_counts": counts.tolist(),
        "seconds": time.time() - started,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


class FastRoadMatcher:
    def __init__(self, roads_path: Path, nodes_path: Path, spacing_m: float = 10.0):
        self.roads = gpd.read_parquet(roads_path).to_crs(32649).reset_index(drop=True)
        self.geoms = self.roads.geometry.to_numpy()
        coords, road_ids = [], []
        for i, line in enumerate(self.geoms):
            distance = np.arange(0, line.length + spacing_m / 2, spacing_m)
            points = shapely.line_interpolate_point(line, distance)
            coords.append(np.column_stack([shapely.get_x(points), shapely.get_y(points)]))
            road_ids.append(np.full(len(distance), i, dtype="int32"))
        self.sample_coords = np.vstack(coords)
        self.sample_road_ids = np.concatenate(road_ids)
        self.tree = cKDTree(self.sample_coords)
        self.transformer = Transformer.from_crs(4326, 32649, always_xy=True)
        nodes = gpd.read_parquet(nodes_path).to_crs(32649)
        intersections = nodes[(nodes.street_degree >= 3) | nodes.signal.fillna(False)]
        self.intersection_tree = cKDTree(
            np.column_stack([intersections.geometry.x, intersections.geometry.y])
        )

    def match(self, x: np.ndarray, y: np.ndarray, k: int = 5):
        xy = np.column_stack([x, y])
        _, nearest_samples = self.tree.query(xy, k=k, workers=-1)
        candidates = self.sample_road_ids[nearest_samples]
        points = shapely.points(x, y)
        candidate_lines = self.geoms[candidates.ravel()]
        repeated_points = np.repeat(points, k)
        exact = shapely.distance(repeated_points, candidate_lines).reshape(-1, k)
        best = np.argmin(exact, axis=1)
        row = np.arange(len(x))
        road_idx = candidates[row, best]
        distance = exact[row, best]
        lines = self.geoms[road_idx]
        positions = shapely.line_locate_point(lines, points)
        snapped = shapely.line_interpolate_point(lines, positions)
        intersection_distance = self.intersection_tree.query(xy, k=1, workers=-1)[0]
        return road_idx, distance, shapely.get_x(snapped), shapely.get_y(snapped), intersection_distance


def add_segment_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    same = frame.order_id.eq(frame.order_id.shift()).fillna(False)
    frame["dt_s"] = frame.timestamp.diff().where(same)
    frame["segment_distance_m"] = np.hypot(frame.proj_x.diff(), frame.proj_y.diff()).where(same)
    frame["speed_kmh"] = frame.segment_distance_m / frame.dt_s * 3.6
    return frame


def clean_bucket(raw: pd.DataFrame, matcher: FastRoadMatcher, input_crs: str):
    raw = raw.copy()
    raw["source_lon"] = raw.lon
    raw["source_lat"] = raw.lat
    valid_coord = raw.lon.between(-180, 180) & raw.lat.between(-90, 90)
    invalid_count = (~valid_coord).groupby(raw.order_id).sum().rename("invalid_coord_count")
    raw_count = raw.groupby("order_id").size().rename("point_count")

    source_order = raw.sort_values(["order_id", "source_row"], kind="stable")
    same_source_order = source_order.order_id.eq(source_order.order_id.shift())
    disorder = (
        (source_order.timestamp.diff().lt(0) & same_source_order)
        .groupby(source_order.order_id).sum().rename("time_disorder_count")
    )

    frame = raw.loc[valid_coord].sort_values(["order_id", "timestamp", "source_row"], kind="stable")
    duplicate = frame.duplicated(["order_id", "timestamp"], keep="first")
    duplicate_count = duplicate.groupby(frame.order_id).sum().rename("duplicate_count")
    frame = frame.loc[~duplicate].copy()
    if input_crs == "gcj02":
        frame["lon"], frame["lat"] = gcj02_to_wgs84(frame.source_lon.to_numpy(), frame.source_lat.to_numpy())
    frame["proj_x"], frame["proj_y"] = matcher.transformer.transform(
        frame.lon.to_numpy(), frame.lat.to_numpy()
    )
    frame = add_segment_metrics(frame)

    # Remove isolated spikes only: both adjacent legs are implausible while the
    # direct previous-to-next movement is plausible.
    same_next = frame.order_id.eq(frame.order_id.shift(-1))
    outgoing_bad = frame.speed_kmh.shift(-1).gt(MAX_SPEED_KMH) & same_next
    bridge_dt = frame.timestamp.shift(-1) - frame.timestamp.shift(1)
    bridge_dist = np.hypot(
        frame.proj_x.shift(-1) - frame.proj_x.shift(1),
        frame.proj_y.shift(-1) - frame.proj_y.shift(1),
    )
    bridge_speed = bridge_dist / bridge_dt * 3.6
    spike = frame.speed_kmh.gt(MAX_SPEED_KMH) & outgoing_bad & bridge_speed.le(MAX_SPEED_KMH)
    jump_removed = spike.groupby(frame.order_id).sum().rename("jump_removed_count")
    frame = frame.loc[~spike].copy()
    frame = add_segment_metrics(frame)

    quality_inputs = pd.concat(
        [raw_count, invalid_count, disorder, duplicate_count, jump_removed], axis=1
    ).fillna(0)
    return frame, quality_inputs


def match_bucket(frame: pd.DataFrame, matcher: FastRoadMatcher) -> pd.DataFrame:
    road_idx, distance, snap_x, snap_y, intersection_distance = matcher.match(
        frame.proj_x.to_numpy(), frame.proj_y.to_numpy()
    )
    roads = matcher.roads.iloc[road_idx]
    frame = frame.copy()
    frame["road_idx"] = road_idx
    frame["link_id"] = roads.link_id.to_numpy()
    frame["road_class"] = roads.road_class.to_numpy()
    frame["road_name"] = roads.road_name.to_numpy()
    frame["speed_limit"] = roads.speed_limit.to_numpy()
    frame["oneway_code"] = roads.oneway_code.to_numpy()
    frame["gps_to_link_dist_m"] = distance
    frame["snap_x"] = snap_x
    frame["snap_y"] = snap_y
    frame["intersection_distance_m"] = intersection_distance

    same = frame.order_id.eq(frame.order_id.shift()).fillna(False)
    frame["matched_step_m"] = np.hypot(frame.snap_x.diff(), frame.snap_y.diff()).where(same)
    prev_road = frame.road_idx.shift()
    changed = (same & frame.road_idx.ne(prev_road)).fillna(False)
    prev_u = frame.road_idx.shift().map(matcher.roads.from_node)
    prev_v = frame.road_idx.shift().map(matcher.roads.to_node)
    curr_u = frame.road_idx.map(matcher.roads.from_node)
    curr_v = frame.road_idx.map(matcher.roads.to_node)
    shared = curr_u.eq(prev_u) | curr_u.eq(prev_v) | curr_v.eq(prev_u) | curr_v.eq(prev_v)

    # Endpoint separation catches disconnected/parallel-road switches without
    # requiring millions of shortest-path calls.
    lines = matcher.geoms[frame.road_idx.to_numpy()]
    start = shapely.get_point(lines, 0)
    end = shapely.get_point(lines, -1)
    sx, sy, ex, ey = shapely.get_x(start), shapely.get_y(start), shapely.get_x(end), shapely.get_y(end)
    psx, psy, pex, pey = pd.Series(sx).shift(), pd.Series(sy).shift(), pd.Series(ex).shift(), pd.Series(ey).shift()
    endpoint_min = np.minimum.reduce([
        np.hypot(psx - sx, psy - sy), np.hypot(psx - ex, psy - ey),
        np.hypot(pex - sx, pey - sy), np.hypot(pex - ex, pey - ey),
    ])
    changed_idx = np.flatnonzero(changed.to_numpy())
    line_pair_distance = np.full(len(frame), np.inf, dtype=float)
    if len(changed_idx):
        previous_lines = lines[np.maximum(changed_idx - 1, 0)]
        line_pair_distance[changed_idx] = shapely.distance(previous_lines, lines[changed_idx])
    geometry_adjacent = line_pair_distance <= 2.0
    threshold = np.maximum(120.0, frame.segment_distance_m.fillna(0).to_numpy() * 2.5 + 40)
    frame["topology_gap"] = (
        changed.to_numpy() & ~shared.to_numpy() & ~geometry_adjacent & (endpoint_min > threshold)
    )
    frame["parallel_jump"] = (
        changed.to_numpy() & ~shared.to_numpy() & ~geometry_adjacent
        & (frame.matched_step_m.fillna(0).to_numpy() < 30)
        & (endpoint_min > 120)
    )
    return frame


def grouped_weighted_sum(frame, mask):
    return frame.dt_s.fillna(0).where(mask, 0).groupby(frame.order_id).sum()


def summarize_orders(frame: pd.DataFrame, quality_inputs: pd.DataFrame):
    group = frame.groupby("order_id", sort=False)
    clean_count = group.size().rename("clean_point_count")
    duration = (group.timestamp.max() - group.timestamp.min()).rename("duration_s")
    distance = group.segment_distance_m.sum().rename("distance_m")
    median_dt = group.dt_s.median().rename("median_dt_s")
    max_speed = group.speed_kmh.max().rename("max_speed_kmh")
    quality = quality_inputs.join([clean_count, duration, distance, median_dt, max_speed], how="left")
    quality["retained_ratio"] = quality.clean_point_count / quality.point_count

    def quality_flag(row):
        reasons = []
        if row.clean_point_count < 10: reasons.append("too_few_points")
        if row.duration_s < 60: reasons.append("too_short_duration")
        if row.distance_m < 200: reasons.append("too_short_distance")
        if row.retained_ratio < 0.8: reasons.append("low_point_retention")
        return "good" if not reasons else "low_quality:" + "|".join(reasons)

    quality["quality_flag"] = quality.apply(quality_flag, axis=1)

    matched_ratio = frame.gps_to_link_dist_m.le(MATCH_RADIUS_M).groupby(frame.order_id).mean()
    mean_dist = group.gps_to_link_dist_m.mean()
    p90_dist = group.gps_to_link_dist_m.quantile(0.9)
    route_length = group.matched_step_m.sum()
    topology_gaps = group.topology_gap.sum().astype(float)
    parallel_jumps = group.parallel_jump.sum().astype(float)
    ratio = (route_length / distance).astype(float)
    transitions = (clean_count - 1).clip(lower=1).astype(float)
    route_score = pd.Series(
        np.exp(-np.abs(np.log(ratio.clip(lower=1e-6).to_numpy(dtype=float))) / 0.45),
        index=ratio.index,
    )
    confidence = (
        matched_ratio * np.exp(-mean_dist / 35) * route_score
        * np.exp(-8 * topology_gaps / transitions) * np.exp(-4 * parallel_jumps / transitions)
    )
    match = pd.DataFrame({
        "matched_point_ratio": matched_ratio,
        "mean_gps_to_link_dist_m": mean_dist,
        "p90_gps_to_link_dist_m": p90_dist,
        "matched_route_length_m": route_length,
        "route_length_ratio": ratio,
        "topology_gap_count": topology_gaps,
        "parallel_jump_count": parallel_jumps,
        "matching_confidence": confidence,
    })
    match["matching_success"] = (
        match.matched_point_ratio.ge(0.85) & match.p90_gps_to_link_dist_m.le(MATCH_RADIUS_M)
        & match.route_length_ratio.between(0.8, 1.3)
    )

    total_dt = group.dt_s.sum()
    low_ratio = grouped_weighted_sum(frame, frame.speed_kmh.lt(10)) / total_dt
    stop_mask = frame.speed_kmh.lt(2)
    stop_start = stop_mask & (~stop_mask.shift(fill_value=False) | frame.order_id.ne(frame.order_id.shift()))
    stop_run = stop_start.cumsum()
    stop_runs = frame.dt_s.fillna(0).where(stop_mask, 0).groupby([frame.order_id, stop_run]).sum()
    valid_stops = stop_runs[stop_runs >= 10]
    stop_count = valid_stops.groupby(level=0).size().reindex(clean_count.index, fill_value=0)
    stop_duration = valid_stops.groupby(level=0).sum().reindex(clean_count.index, fill_value=0)

    speed_mps = frame.speed_kmh / 3.6
    same = frame.order_id.eq(frame.order_id.shift())
    acceleration = (speed_mps.diff() / frame.dt_s).where(same & frame.dt_s.le(10))
    dx, dy = frame.proj_x.diff().where(same), frame.proj_y.diff().where(same)
    seg = np.hypot(dx, dy)
    heading = np.degrees(np.arctan2(dx, dy))
    heading_change = ((heading.diff() + 180) % 360 - 180).where(
        same & seg.ge(5) & seg.shift().ge(5)
    )
    intersection_delay = grouped_weighted_sum(
        frame, frame.intersection_distance_m.le(30) & frame.speed_kmh.lt(10)
    )
    known_speed = grouped_weighted_sum(frame, frame.speed_limit.notna()) / total_dt
    dominant_class = (
        frame.dt_s.fillna(0).groupby([frame.order_id, frame.road_class]).sum()
        .groupby(level=0).idxmax().map(lambda value: value[1])
    )
    features = pd.DataFrame({
        "avg_speed_kmh": distance / total_dt * 3.6,
        "median_speed_kmh": group.speed_kmh.median(),
        "low_speed_ratio": low_ratio,
        "stop_count": stop_count,
        "stop_count_km": stop_count / (distance / 1000),
        "stop_duration_ratio": stop_duration / total_dt,
        "speed_std_kmh": group.speed_kmh.std(),
        "acc_std_mps2": acceleration.groupby(frame.order_id).std(),
        "heading_change_sum_deg": heading_change.abs().groupby(frame.order_id).sum(),
        "turn_count": heading_change.abs().ge(45).groupby(frame.order_id).sum(),
        "intersection_delay_s": intersection_delay,
        "curvature_deg_per_km": heading_change.abs().groupby(frame.order_id).sum() / (distance / 1000),
        "dominant_road_class": dominant_class,
        "speed_limit_known_exposure": known_speed,
        "lane_exposure_status": "unavailable_in_source",
    })
    driver = group.driver_id.first().rename("driver_id")
    result = pd.concat([driver, quality, match, features], axis=1).reset_index()
    return result


def process_bucket(bucket: int, args: argparse.Namespace, matcher: FastRoadMatcher, dirs: dict):
    order_out = dirs["orders"] / f"part_{bucket:03d}.parquet"
    point_out = dirs["points"] / f"part_{bucket:03d}.parquet"
    route_out = dirs["routes"] / f"part_{bucket:03d}.parquet"
    exposure_out = dirs["exposure"] / f"part_{bucket:03d}.parquet"
    if all(path.exists() for path in [order_out, point_out, route_out, exposure_out]):
        return "skipped"
    raw = pd.read_parquet(dirs["buckets"] / f"bucket_{bucket:03d}.parquet")
    frame, quality_inputs = clean_bucket(raw, matcher, args.input_crs)
    frame = match_bucket(frame, matcher)
    orders = summarize_orders(frame, quality_inputs)

    route_change = frame.order_id.ne(frame.order_id.shift()) | frame.link_id.ne(frame.link_id.shift())
    routes = frame.loc[route_change, ["order_id", "link_id"]].copy()
    routes["route_sequence"] = routes.groupby("order_id").cumcount()
    route_counts = routes.groupby("order_id").size().rename("matched_link_count")
    orders = orders.merge(route_counts, on="order_id", how="left")

    exposure = frame.dt_s.fillna(0).groupby([frame.order_id, frame.road_class]).sum().rename("seconds").reset_index()
    totals = exposure.groupby("order_id").seconds.transform("sum")
    exposure["exposure_ratio"] = exposure.seconds / totals

    point_columns = [
        "driver_id", "order_id", "source_row", "timestamp", "source_lon", "source_lat",
        "lon", "lat", "dt_s", "segment_distance_m", "speed_kmh", "link_id", "road_class",
        "road_name", "speed_limit", "oneway_code", "gps_to_link_dist_m", "snap_x", "snap_y",
        "intersection_distance_m", "topology_gap", "parallel_jump",
    ]
    frame[point_columns].to_parquet(point_out, index=False, compression="zstd")
    orders.to_parquet(order_out, index=False, compression="zstd")
    routes.to_parquet(route_out, index=False, compression="zstd")
    exposure.to_parquet(exposure_out, index=False, compression="zstd")
    return {"raw_points": len(raw), "clean_points": len(frame), "orders": len(orders)}


def make_reports(args: argparse.Namespace, dirs: dict, manifest: dict):
    order_files = sorted(dirs["orders"].glob("part_*.parquet"))
    orders = pd.concat([pd.read_parquet(path) for path in order_files], ignore_index=True)
    orders.to_parquet(args.output_dir / "full_day_stage0_orders.parquet", index=False, compression="zstd")
    orders.to_csv(args.output_dir / "full_day_stage0_orders.csv.gz", index=False, compression="gzip")
    point_files = sorted(dirs["points"].glob("part_*.parquet"))
    clean_points = sum(pq.ParquetFile(path).metadata.num_rows for path in point_files)
    summary = {
        "verdict": "CONDITIONAL GO" if (
            orders.quality_flag.eq("good").mean() > 0.8
            and orders.matching_success.mean() > 0.85
            and orders.p90_gps_to_link_dist_m.quantile(0.9) < 50
        ) else "NO-GO / FIX FIRST",
        "input_rows": manifest["rows"],
        "orders": int(orders.order_id.nunique()),
        "drivers": int(orders.driver_id.nunique()),
        "clean_points": int(clean_points),
        "point_retention_ratio": clean_points / manifest["rows"],
        "good_order_ratio": float(orders.quality_flag.eq("good").mean()),
        "matching_success_ratio": float(orders.matching_success.mean()),
        "high_confidence_ratio": float(orders.matching_confidence.ge(0.7).mean()),
        "median_order_p90_gps_link_distance_m": float(orders.p90_gps_to_link_dist_m.median()),
        "p90_order_p90_gps_link_distance_m": float(orders.p90_gps_to_link_dist_m.quantile(0.9)),
        "route_ratio_good_ratio": float(orders.route_length_ratio.between(0.8, 1.3).mean()),
        "source_coordinate_declaration": "WGS84 per user-provided schema",
        "empirical_coordinate_interpretation": args.input_crs,
        "road_snapshot": "2017-01-01T20:28:02Z Geofabrik OSM",
        "matcher": "10m densified-road candidate KD-tree + exact geometry projection + topology audit",
        "lane_semantics": "unavailable in the free 2017 shape source",
    }
    (args.output_dir / "full_day_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    quantiles = orders[[
        "clean_point_count", "median_dt_s", "p90_gps_to_link_dist_m", "route_length_ratio",
        "matching_confidence", "low_speed_ratio", "stop_count_km", "speed_std_kmh",
        "intersection_delay_s", "speed_limit_known_exposure",
    ]].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).T
    quantiles.columns = ["p10", "p25", "p50", "p75", "p90"]
    quantiles.to_csv(args.output_dir / "full_day_distribution_quantiles.csv")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    plots = [
        (orders.clean_point_count, "Clean points/order", None),
        (orders.median_dt_s, "Median sampling interval (s)", (0, 20)),
        (orders.p90_gps_to_link_dist_m, "Order P90 GPS-link distance (m)", (0, 80)),
        (orders.low_speed_ratio, "Low-speed ratio", (0, 1)),
        (orders.stop_count_km.replace([np.inf, -np.inf], np.nan), "Stop count/km", (0, 12)),
        (orders.intersection_delay_s, "Intersection delay (s)", (0, 600)),
    ]
    for ax, (series, title, limits) in zip(axes.flat, plots):
        values = series.replace([np.inf, -np.inf], np.nan).dropna()
        if limits:
            values = values[values.between(limits[0], limits[1])]
        ax.hist(values, bins=50, color="#2474b5", alpha=0.85)
        ax.set_title(title); ax.grid(alpha=0.2)
        if limits: ax.set_xlim(*limits)
    fig.tight_layout(); fig.savefig(dirs["figures"] / "full_day_distributions.png", dpi=170); plt.close(fig)

    exposure_files = sorted((args.output_dir / "road_exposure_parts").glob("part_*.parquet"))
    exposure = pd.concat([pd.read_parquet(path) for path in exposure_files], ignore_index=True)
    road_summary = (
        exposure.groupby("road_class").exposure_ratio.sum().div(len(orders))
        .sort_values(ascending=False).rename("mean_exposure_all_orders").reset_index()
    )
    road_summary.to_csv(args.output_dir / "full_day_road_class_exposure_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    shown = road_summary.sort_values("mean_exposure_all_orders")
    ax.barh(shown.road_class, shown.mean_exposure_all_orders, color="#2a9d8f")
    ax.set_xlabel("Mean order exposure ratio")
    ax.set_title("Full-day road-class exposure")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout(); fig.savefig(dirs["figures"] / "full_day_road_class_exposure.png", dpi=170); plt.close(fig)

    q = quantiles
    report = f"""# 西安滴滴轨迹全日 Stage0 报告

## 结论

**{summary['verdict']}**

全日共处理 {summary['input_rows']:,} 个源点、{summary['orders']:,} 个订单、{summary['drivers']:,} 名司机。源字段按 `driver_id, order_id, timestamp, lon, lat` 解释。虽然数据说明将坐标称为 WGS84，但同期 OSM 上的实测距离强烈支持先按 GCJ-02 转为 WGS84；输出同时保留 `source_lon/source_lat` 和转换后的 `lon/lat`。

## 核心指标

| 指标 | 结果 |
|---|---:|
| 点保留率 | {summary['point_retention_ratio']:.2%} |
| 高质量订单比例 | {summary['good_order_ratio']:.2%} |
| 匹配成功订单比例 | {summary['matching_success_ratio']:.2%} |
| 高置信度订单比例 | {summary['high_confidence_ratio']:.2%} |
| 订单 P90 匹配距离中位数 | {summary['median_order_p90_gps_link_distance_m']:.2f} m |
| 订单 P90 匹配距离的 P90 | {summary['p90_order_p90_gps_link_distance_m']:.2f} m |
| route ratio 0.8-1.3 比例 | {summary['route_ratio_good_ratio']:.2%} |
| low-speed ratio 中位数 | {q.loc['low_speed_ratio','p50']:.3f} |
| stop count/km 中位数 | {q.loc['stop_count_km','p50']:.3f} |
| intersection delay 中位数 | {q.loc['intersection_delay_s','p50']:.1f} s |

## 方法与边界

- 路网是 2017-01-01 Geofabrik OSM，时间上接近 2016-10-01 轨迹。
- 匹配器使用 10m 道路加密候选索引、精确线投影以及拓扑/平行道路跳转审计；仍不是完整 HMM/Viterbi。
- 免费 Shapefile 没有 `lanes`，且 `maxspeed` 标注很稀疏；车道暴露不能从该数据可靠构造。
- 2017 免费包没有行政边界层，西安市域产物是保守矩形包络；实际匹配使用完整覆盖轨迹的核心区路网。
"""
    (args.output_dir / "full_day_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dirs = {
        "buckets": args.output_dir / "order_buckets",
        "points": args.output_dir / "matched_points",
        "orders": args.output_dir / "order_parts",
        "routes": args.output_dir / "route_parts",
        "exposure": args.output_dir / "road_exposure_parts",
        "figures": args.output_dir / "figures",
    }
    for path in dirs.values(): path.mkdir(parents=True, exist_ok=True)
    manifest = bucketize(args, dirs["buckets"])
    matcher = FastRoadMatcher(args.roads, args.nodes)
    bucket_total = args.buckets if args.limit_buckets is None else min(args.limit_buckets, args.buckets)
    started = time.time()
    for bucket in range(bucket_total):
        result = process_bucket(bucket, args, matcher, dirs)
        print(
            f"process bucket={bucket+1}/{bucket_total} result={result} elapsed={time.time()-started:.1f}s",
            flush=True,
        )
    if bucket_total == args.buckets:
        make_reports(args, dirs, manifest)


if __name__ == "__main__":
    main()
