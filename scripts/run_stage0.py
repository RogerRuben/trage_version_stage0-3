"""Run the Stage0 DiDi trajectory feasibility experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point


EARTH_RADIUS_M = 6_371_008.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--graphml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-speed-kmh", type=float, default=150.0)
    parser.add_argument("--match-radius-m", type=float, default=50.0)
    parser.add_argument("--input-crs", choices=["wgs84", "gcj02"], default="wgs84")
    return parser.parse_args()


def haversine_m(lon1, lat1, lon2, lat2):
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def gcj02_to_wgs84(lon, lat):
    """Vectorized inverse GCJ-02 approximation for points within China."""
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


def clean_one_order(group: pd.DataFrame, max_speed_kmh: float):
    source = group.sort_values("source_row", kind="stable")
    disorder = int((source["timestamp"].diff() < 0).sum())
    invalid = ~(group["lon"].between(-180, 180) & group["lat"].between(-90, 90))
    valid = group.loc[~invalid].sort_values("timestamp", kind="stable").copy()
    duplicate = valid.duplicated(["timestamp", "lon", "lat"], keep="first")
    duplicate_count = int(duplicate.sum())
    valid = valid.loc[~duplicate]

    kept = []
    jump_removed = 0
    nonpositive_removed = 0
    for idx, row in valid.iterrows():
        if not kept:
            kept.append(idx)
            continue
        prev = valid.loc[kept[-1]]
        dt = float(row.timestamp - prev.timestamp)
        if dt <= 0:
            nonpositive_removed += 1
            continue
        dist = float(haversine_m(prev.lon, prev.lat, row.lon, row.lat))
        speed = dist / dt * 3.6
        if speed > max_speed_kmh:
            jump_removed += 1
            continue
        kept.append(idx)

    cleaned = valid.loc[kept].copy()
    cleaned["dt_s"] = cleaned["timestamp"].diff()
    cleaned["segment_distance_m"] = haversine_m(
        cleaned["lon"].shift(), cleaned["lat"].shift(), cleaned["lon"], cleaned["lat"]
    )
    cleaned.loc[cleaned["dt_s"].isna(), "segment_distance_m"] = np.nan
    cleaned["speed_kmh"] = cleaned["segment_distance_m"] / cleaned["dt_s"] * 3.6

    raw_count = len(group)
    clean_count = len(cleaned)
    duration = float(cleaned.timestamp.max() - cleaned.timestamp.min()) if clean_count >= 2 else 0.0
    distance = float(cleaned.segment_distance_m.sum()) if clean_count else 0.0
    median_dt = float(cleaned.dt_s.median()) if clean_count >= 2 else np.nan
    max_speed = float(cleaned.speed_kmh.max()) if clean_count >= 2 else np.nan
    retained = clean_count / raw_count if raw_count else 0.0
    reasons = []
    if clean_count < 10:
        reasons.append("too_few_points")
    if duration < 60:
        reasons.append("too_short_duration")
    if distance < 200:
        reasons.append("too_short_distance")
    if retained < 0.8:
        reasons.append("low_point_retention")
    quality = "good" if not reasons else "low_quality:" + "|".join(reasons)
    stats = {
        "order_id": str(group.order_id.iloc[0]),
        "driver_id": str(group.driver_id.iloc[0]),
        "point_count": raw_count,
        "clean_point_count": clean_count,
        "duration_s": duration,
        "distance_m": distance,
        "median_dt_s": median_dt,
        "max_speed_kmh": max_speed,
        "time_disorder_count": disorder,
        "duplicate_count": duplicate_count,
        "invalid_coord_count": int(invalid.sum()),
        "nonpositive_dt_removed": nonpositive_removed,
        "jump_removed_count": jump_removed,
        "retained_ratio": retained,
        "quality_flag": quality,
    }
    return cleaned, stats


def compact_sequence(values) -> str:
    output = []
    previous = object()
    for value in values:
        if value != previous:
            output.append(value)
            previous = value
    return json.dumps(output, ensure_ascii=False)


def stop_metrics(group: pd.DataFrame):
    moving = group["speed_kmh"].fillna(np.inf).to_numpy() < 2.0
    dt = group["dt_s"].fillna(0).to_numpy(dtype=float)
    durations = []
    current = 0.0
    for is_stop, seconds in zip(moving, dt):
        if is_stop:
            current += seconds
        elif current:
            durations.append(current)
            current = 0.0
    if current:
        durations.append(current)
    valid = [value for value in durations if value >= 10.0]
    return len(valid), float(sum(valid))


def exposure_json(group: pd.DataFrame, column: str) -> str:
    weights = group["dt_s"].fillna(0).clip(lower=0)
    labels = group[column].fillna("unknown").astype(str)
    totals = weights.groupby(labels).sum()
    denom = float(totals.sum())
    values = {str(k): round(float(v / denom), 6) for k, v in totals.items()} if denom else {}
    return json.dumps(values, ensure_ascii=False, sort_keys=True)


def make_feature_row(group: pd.DataFrame) -> dict:
    duration = float(group.dt_s.fillna(0).sum())
    distance_m = float(group.segment_distance_m.fillna(0).sum())
    weights = group.dt_s.fillna(0).clip(lower=0)
    speeds = group.speed_kmh
    low_ratio = float(weights[speeds < 10].sum() / duration) if duration else np.nan
    stop_count, stop_duration = stop_metrics(group)

    dx = group["proj_x"].diff().to_numpy()
    dy = group["proj_y"].diff().to_numpy()
    segment_len = np.hypot(dx, dy)
    headings = np.degrees(np.arctan2(dx, dy))
    changes = (np.diff(headings) + 180) % 360 - 180
    valid_changes = np.isfinite(changes) & (segment_len[1:] >= 5) & (segment_len[:-1] >= 5)
    heading_sum = float(np.abs(changes[valid_changes]).sum())
    turn_count = int((np.abs(changes[valid_changes]) >= 45).sum())

    speed_mps = speeds.to_numpy(dtype=float) / 3.6
    dt = group.dt_s.to_numpy(dtype=float)
    acceleration = np.diff(speed_mps) / dt[1:]
    acceleration = acceleration[np.isfinite(acceleration) & (dt[1:] <= 10)]
    intersection_delay = float(
        weights[(group["intersection_distance_m"] <= 30) & (speeds < 10)].sum()
    )
    return {
        "order_id": str(group.order_id.iloc[0]),
        "avg_speed_kmh": distance_m / duration * 3.6 if duration else np.nan,
        "median_speed_kmh": float(speeds.median()),
        "low_speed_ratio": low_ratio,
        "stop_count": stop_count,
        "stop_count_km": stop_count / (distance_m / 1000) if distance_m else np.nan,
        "stop_duration_ratio": stop_duration / duration if duration else np.nan,
        "speed_std_kmh": float(speeds.std()),
        "acc_std_mps2": float(np.std(acceleration)) if len(acceleration) else np.nan,
        "heading_change_sum_deg": heading_sum,
        "turn_count": turn_count,
        "intersection_delay_s": intersection_delay,
        "curvature_deg_per_km": heading_sum / (distance_m / 1000) if distance_m else np.nan,
        "road_class_exposure": exposure_json(group, "road_class"),
        "lane_exposure": exposure_json(group, "lane_num"),
    }


def matching_quality(group: pd.DataFrame, radius_m: float) -> dict:
    distances = group["gps_to_link_dist_m"].to_numpy(dtype=float)
    matched = distances <= radius_m
    gps_length = float(group.segment_distance_m.fillna(0).sum())
    route_length = float(group.matched_step_m.fillna(0).sum())
    ratio = route_length / gps_length if gps_length else np.nan
    topology_gaps = int(group.topology_gap.fillna(False).sum())
    parallel_jumps = int(group.parallel_jump.fillna(False).sum())
    transitions = max(len(group) - 1, 1)
    matched_ratio = float(matched.mean())
    mean_dist = float(np.mean(distances))
    p90_dist = float(np.percentile(distances, 90))
    route_score = math.exp(-abs(math.log(max(ratio, 1e-6))) / 0.45) if np.isfinite(ratio) else 0.0
    confidence = (
        matched_ratio
        * math.exp(-mean_dist / 35.0)
        * route_score
        * math.exp(-8.0 * topology_gaps / transitions)
        * math.exp(-4.0 * parallel_jumps / transitions)
    )
    success = matched_ratio >= 0.85 and p90_dist <= radius_m and 0.8 <= ratio <= 1.3
    return {
        "order_id": str(group.order_id.iloc[0]),
        "matched_point_ratio": matched_ratio,
        "mean_gps_to_link_dist_m": mean_dist,
        "p90_gps_to_link_dist_m": p90_dist,
        "gps_length_m": gps_length,
        "matched_route_length_m": route_length,
        "route_length_ratio": ratio,
        "topology_gap_count": topology_gaps,
        "parallel_jump_count": parallel_jumps,
        "matching_confidence": confidence,
        "matching_success": success,
        "matched_route": compact_sequence(group.link_id.fillna("unmatched")),
    }


def plot_distributions(order_quality, match_quality, features, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    panels = [
        (order_quality.clean_point_count, "Clean points/order", None),
        (order_quality.median_dt_s, "Median sampling interval (s)", (0, 20)),
        (match_quality.p90_gps_to_link_dist_m, "P90 GPS-link distance (m)", (0, 100)),
        (features.low_speed_ratio, "Low-speed ratio", (0, 1)),
        (features.stop_count_km.replace([np.inf, -np.inf], np.nan), "Stop count/km", (0, 10)),
        (features.intersection_delay_s, "Intersection delay (s)", None),
    ]
    for ax, (series, title, limits) in zip(axes.flat, panels):
        clean = series.replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(clean, bins=30, color="#2474b5", alpha=0.85)
        ax.set_title(title)
        ax.grid(alpha=0.2)
        if limits:
            ax.set_xlim(*limits)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def summarize_road_exposure(features: pd.DataFrame, output_dir: Path):
    rows = []
    for _, row in features.iterrows():
        for road_class, ratio in json.loads(row.road_class_exposure).items():
            rows.append({"order_id": row.order_id, "road_class": road_class, "exposure_ratio": ratio})
    long = pd.DataFrame(rows)
    summary = (
        long.groupby("road_class").exposure_ratio
        .agg(order_count="count", mean_exposure="mean", median_exposure="median")
        .sort_values("mean_exposure", ascending=False)
        .reset_index()
    )
    # Means must include zero exposure for orders that never visit a class.
    summary["mean_exposure_all_orders"] = summary.road_class.map(
        long.groupby("road_class").exposure_ratio.sum() / len(features)
    )
    summary.to_csv(output_dir / "road_class_exposure_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    shown = summary.sort_values("mean_exposure_all_orders", ascending=True)
    ax.barh(shown.road_class, shown.mean_exposure_all_orders, color="#2a9d8f")
    ax.set_xlabel("Mean order exposure ratio")
    ax.set_title("Road-class exposure across 500 orders")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "road_class_exposure.png", dpi=170)
    plt.close(fig)
    semantic_rows = []
    for feature_name, column in [
        ("road_class", "road_class_exposure"), ("lane_num", "lane_exposure")
    ]:
        unknown = 0.0
        for value in features[column]:
            unknown += json.loads(value).get("unknown", 0.0)
        unknown /= len(features)
        semantic_rows.append(
            {"semantic": feature_name, "known_exposure_ratio": 1 - unknown, "unknown_exposure_ratio": unknown}
        )
    pd.DataFrame(semantic_rows).to_csv(output_dir / "semantic_coverage_summary.csv", index=False)


def plot_case(order_id, label, raw, matched, edges_proj, output):
    raw_o = raw[raw.order_id == order_id]
    clean_o = matched[matched.order_id == order_id]
    if clean_o.empty:
        return
    xmin, xmax = clean_o.proj_x.min() - 250, clean_o.proj_x.max() + 250
    ymin, ymax = clean_o.proj_y.min() - 250, clean_o.proj_y.max() + 250
    nearby = edges_proj.cx[xmin:xmax, ymin:ymax]
    raw_geo = gpd.GeoDataFrame(
        raw_o.copy(), geometry=gpd.points_from_xy(raw_o.lon, raw_o.lat), crs="EPSG:4326"
    ).to_crs(edges_proj.crs)
    clean_rows = set(clean_o.source_row.astype(int))
    dropped = raw_geo[~raw_geo.source_row.astype(int).isin(clean_rows)]
    fig, ax = plt.subplots(figsize=(8, 8))
    nearby.plot(ax=ax, color="#c8c8c8", linewidth=0.7, zorder=1)
    ax.plot(raw_geo.geometry.x, raw_geo.geometry.y, color="#d95f02", linewidth=1, alpha=0.55, label="raw GPS", zorder=2)
    ax.plot(clean_o.proj_x, clean_o.proj_y, color="#1f78b4", linewidth=1.3, label="clean GPS", zorder=3)
    ax.plot(clean_o.snap_x, clean_o.snap_y, color="#1b9e77", linewidth=2, alpha=0.8, label="matched", zorder=4)
    if not dropped.empty:
        ax.scatter(dropped.geometry.x, dropped.geometry.y, marker="x", s=28, color="#e7298a", label="removed", zorder=5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title(f"{label}: {order_id[:12]}…")
    ax.legend(loc="best", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def fmt(value, digits=3):
    return "NA" if pd.isna(value) else f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    out = args.output_dir
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(args.sample)
    raw["source_lon"] = raw["lon"]
    raw["source_lat"] = raw["lat"]
    if args.input_crs == "gcj02":
        raw["lon"], raw["lat"] = gcj02_to_wgs84(raw.source_lon.to_numpy(), raw.source_lat.to_numpy())

    clean_frames = []
    quality_rows = []
    for _, group in raw.groupby("order_id", sort=False):
        cleaned, stats = clean_one_order(group, args.max_speed_kmh)
        clean_frames.append(cleaned)
        quality_rows.append(stats)
    clean = pd.concat(clean_frames, ignore_index=True)
    quality = pd.DataFrame(quality_rows)
    clean.to_parquet(out / "sample_clean.parquet", index=False)
    quality.to_csv(out / "order_quality.csv", index=False)

    graph = ox.load_graphml(args.graphml)
    graph = ox.truncate.truncate_graph_bbox(
        graph,
        north=float(clean.lat.max() + 0.03),
        south=float(clean.lat.min() - 0.03),
        east=float(clean.lon.max() + 0.03),
        west=float(clean.lon.min() - 0.03),
        truncate_by_edge=True,
        retain_all=True,
    )
    graph_proj = ox.project_graph(graph, to_crs="EPSG:32649")
    nodes_proj, edges_proj = ox.graph_to_gdfs(graph_proj)
    edges_reset = edges_proj.reset_index()
    edges_reset["link_id"] = edges_reset.apply(lambda r: f"{r.u}_{r.v}_{r.key}", axis=1)
    edges_reset["road_class"] = edges_reset.highway.apply(
        lambda value: value[0] if isinstance(value, list) and value else value
    )
    edges_reset["lane_num"] = edges_reset.get("lanes", pd.Series(index=edges_reset.index, dtype=object)).apply(
        lambda value: value[0] if isinstance(value, list) and value else value
    )
    edge_lookup = edges_reset.set_index(["u", "v", "key"])

    clean_geo = gpd.GeoDataFrame(
        clean.copy(), geometry=gpd.points_from_xy(clean.lon, clean.lat), crs="EPSG:4326"
    ).to_crs(graph_proj.graph["crs"])
    clean["proj_x"] = clean_geo.geometry.x.to_numpy()
    clean["proj_y"] = clean_geo.geometry.y.to_numpy()
    nearest, distances = ox.distance.nearest_edges(
        graph_proj, X=clean.proj_x.to_numpy(), Y=clean.proj_y.to_numpy(), return_dist=True
    )
    nearest = list(nearest)
    clean["edge_u"] = [edge[0] for edge in nearest]
    clean["edge_v"] = [edge[1] for edge in nearest]
    clean["edge_key"] = [edge[2] for edge in nearest]
    clean["gps_to_link_dist_m"] = np.asarray(distances, dtype=float)

    snap_x, snap_y, link_ids, road_classes, lane_nums = [], [], [], [], []
    for x, y, edge in zip(clean.proj_x, clean.proj_y, nearest):
        record = edge_lookup.loc[edge]
        line = record.geometry
        snap = line.interpolate(line.project(Point(float(x), float(y))))
        snap_x.append(snap.x)
        snap_y.append(snap.y)
        link_ids.append(record.link_id)
        road_classes.append(record.road_class)
        lane_nums.append(record.lane_num)
    clean["snap_x"] = snap_x
    clean["snap_y"] = snap_y
    clean["link_id"] = link_ids
    clean["road_class"] = road_classes
    clean["lane_num"] = lane_nums

    # Nearest major/intersection node for intersection-delay primitives.
    degree_map = dict(nx.Graph(graph_proj).degree())
    intersection_mask = nodes_proj.index.to_series().map(degree_map).fillna(0).ge(3).to_numpy()
    intersection_nodes = nodes_proj.loc[intersection_mask]
    if intersection_nodes.empty:
        clean["intersection_distance_m"] = np.nan
    else:
        tree = cKDTree(np.column_stack([intersection_nodes.geometry.x, intersection_nodes.geometry.y]))
        clean["intersection_distance_m"] = tree.query(np.column_stack([clean.proj_x, clean.proj_y]), k=1)[0]

    # Vectorized topology diagnostics between successive matched edges.
    node_x = nodes_proj.geometry.x.to_dict()
    node_y = nodes_proj.geometry.y.to_dict()
    clean["matched_step_m"] = np.hypot(clean.snap_x.diff(), clean.snap_y.diff())
    clean.loc[clean.order_id.ne(clean.order_id.shift()), "matched_step_m"] = np.nan
    prev_u, prev_v = clean.edge_u.shift(), clean.edge_v.shift()
    same_order = clean.order_id.eq(clean.order_id.shift())
    changed = same_order & ~(
        clean.edge_u.eq(prev_u) & clean.edge_v.eq(prev_v) & clean.edge_key.eq(clean.edge_key.shift())
    )
    shared = clean.edge_u.eq(prev_u) | clean.edge_u.eq(prev_v) | clean.edge_v.eq(prev_u) | clean.edge_v.eq(prev_v)
    pu_x, pu_y = prev_u.map(node_x), prev_u.map(node_y)
    pv_x, pv_y = prev_v.map(node_x), prev_v.map(node_y)
    cu_x, cu_y = clean.edge_u.map(node_x), clean.edge_u.map(node_y)
    cv_x, cv_y = clean.edge_v.map(node_x), clean.edge_v.map(node_y)
    endpoint_min = np.minimum.reduce([
        np.hypot(pu_x - cu_x, pu_y - cu_y), np.hypot(pu_x - cv_x, pu_y - cv_y),
        np.hypot(pv_x - cu_x, pv_y - cu_y), np.hypot(pv_x - cv_x, pv_y - cv_y),
    ])
    threshold = np.maximum(120.0, clean.segment_distance_m.fillna(0).to_numpy() * 2.5 + 40.0)
    clean["topology_gap"] = changed & ~shared & (endpoint_min > threshold)
    clean["parallel_jump"] = changed & ~shared & (clean.matched_step_m < 30) & (endpoint_min > 120)

    match_rows = [matching_quality(group, args.match_radius_m) for _, group in clean.groupby("order_id")]
    match_quality = pd.DataFrame(match_rows)
    feature_rows = [make_feature_row(group) for _, group in clean.groupby("order_id")]
    features = pd.DataFrame(feature_rows)
    match_quality.to_csv(out / "map_matching_quality.csv", index=False)
    features.to_csv(out / "trajectory_features.csv", index=False)
    clean.drop(columns="geometry", errors="ignore").to_parquet(out / "matched_points.parquet", index=False)

    stage0 = quality.merge(match_quality, on="order_id", how="left").merge(features, on="order_id", how="left")
    stage0.to_csv(out / "stage0_order_table.csv", index=False)
    stage0.to_parquet(out / "stage0_order_table.parquet", index=False)
    plot_distributions(quality, match_quality, features, figures / "feature_distributions.png")
    summarize_road_exposure(features, out)

    candidates = [
        ("high_quality", stage0.sort_values("matching_confidence", ascending=False).iloc[0].order_id),
        ("matching_failure", stage0.sort_values("matching_confidence").iloc[0].order_id),
        ("stop_and_go", stage0.sort_values("stop_count_km", ascending=False).iloc[0].order_id),
        ("complex_intersection", stage0.sort_values("intersection_delay_s", ascending=False).iloc[0].order_id),
        ("gps_jump", stage0.sort_values("jump_removed_count", ascending=False).iloc[0].order_id),
    ]
    used = set()
    case_rows = []
    for label, order_id in candidates:
        if order_id in used:
            continue
        used.add(order_id)
        filename = f"case_{label}.png"
        plot_case(order_id, label, raw, clean, edges_reset.set_geometry("geometry"), figures / filename)
        case_rows.append({"case_type": label, "order_id": order_id, "figure": f"figures/{filename}"})
    pd.DataFrame(case_rows).to_csv(out / "visual_case_index.csv", index=False)

    segment_dt = clean.dt_s.dropna()
    dt_1_5 = float(segment_dt.between(1, 5).mean())
    clean_order_retention = float((quality.quality_flag == "good").mean())
    match_success = float(match_quality.matching_success.mean())
    high_conf = float((match_quality.matching_confidence >= 0.7).mean())
    aggregate_p90 = float(np.percentile(clean.gps_to_link_dist_m, 90))
    ratio_good = float(match_quality.route_length_ratio.between(0.8, 1.3).mean())
    nondegenerate = bool(features.low_speed_ratio.quantile(0.75) - features.low_speed_ratio.quantile(0.25) > 0.02)
    pass_flags = {
        "clean_order_retention_gt_80pct": clean_order_retention > 0.8,
        "sampling_interval_1_5s_majority": dt_1_5 > 0.5,
        "matching_success_gt_85pct": match_success > 0.85,
        "aggregate_p90_link_distance_lt_50m": aggregate_p90 < 50,
        "route_ratio_0_8_1_3_majority": ratio_good > 0.8,
        "low_speed_feature_nondegenerate": nondegenerate,
    }
    verdict = "CONDITIONAL GO" if all(pass_flags.values()) else "NO-GO / FIX FIRST"
    summary = {
        "verdict": verdict,
        "orders": int(raw.order_id.nunique()),
        "drivers": int(raw.driver_id.nunique()),
        "raw_points": len(raw),
        "clean_points": len(clean),
        "point_retention_ratio": len(clean) / len(raw),
        "good_order_ratio": clean_order_retention,
        "dt_1_5s_ratio": dt_1_5,
        "matching_success_ratio": match_success,
        "high_confidence_ratio": high_conf,
        "aggregate_mean_gps_link_distance_m": float(clean.gps_to_link_dist_m.mean()),
        "aggregate_p90_gps_link_distance_m": aggregate_p90,
        "route_ratio_good_order_ratio": ratio_good,
        "pass_flags": pass_flags,
        "matcher_scope": "nearest-edge projection plus topology/parallel-road diagnostics; not full HMM",
        "road_network_caveat": "Current 2026 OSM network is used for 2016-10-01 trajectories.",
        "input_coordinate_interpretation": args.input_crs,
    }
    (out / "stage0_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            ("orders", summary["orders"]), ("drivers", summary["drivers"]),
            ("raw_points", summary["raw_points"]), ("clean_points", summary["clean_points"]),
            ("point_retention_ratio", summary["point_retention_ratio"]),
            ("good_order_ratio", summary["good_order_ratio"]),
            ("dt_1_5s_ratio", summary["dt_1_5s_ratio"]),
            ("orders_with_time_disorder_ratio", float((quality.time_disorder_count > 0).mean())),
            ("orders_with_jump_removal_ratio", float((quality.jump_removed_count > 0).mean())),
        ], columns=["metric", "value"]
    ).to_csv(out / "data_quality_summary.csv", index=False)
    pd.DataFrame(
        [
            ("matching_success_ratio", summary["matching_success_ratio"]),
            ("high_confidence_ratio", summary["high_confidence_ratio"]),
            ("aggregate_mean_gps_link_distance_m", summary["aggregate_mean_gps_link_distance_m"]),
            ("aggregate_p90_gps_link_distance_m", summary["aggregate_p90_gps_link_distance_m"]),
            ("route_ratio_good_order_ratio", summary["route_ratio_good_order_ratio"]),
            ("topology_gap_total", int(match_quality.topology_gap_count.sum())),
            ("parallel_jump_total", int(match_quality.parallel_jump_count.sum())),
        ], columns=["metric", "value"]
    ).to_csv(out / "map_matching_summary.csv", index=False)

    report = f"""# 西安滴滴轨迹 Stage0 可行性报告

## 结论

**{verdict}**

本轮使用 2016-10-01 17:00–19:00 晚高峰中抽取的 500 个完整订单。输入坐标按 **{args.input_crs.upper()}** 解释；若为 GCJ02，则在匹配前显式转换为 WGS84，并保留 `source_lon/source_lat`。匹配器为“最近道路投影 + 拓扑/平行道路跳转审计”的轻量 Stage0 实现，不等同于生产级 HMM/Viterbi map matching。路网为 2026-07-05 下载的当前 OSM，与 2016 轨迹存在约十年的时间错配，因此通过也只能解释为“具备进入历史路网/HMM 复核的基础”。

## 数据质量

| 指标 | 结果 |
|---|---:|
| 订单 / 司机 | {summary['orders']} / {summary['drivers']} |
| 原始 / 清洗后 GPS 点 | {summary['raw_points']:,} / {summary['clean_points']:,} |
| 点保留率 | {summary['point_retention_ratio']:.2%} |
| 高质量订单比例 | {summary['good_order_ratio']:.2%} |
| 采样间隔 1–5 秒比例 | {summary['dt_1_5s_ratio']:.2%} |
| 有时间乱序的订单比例 | {(quality.time_disorder_count > 0).mean():.2%} |
| 有异常跳点的订单比例 | {(quality.jump_removed_count > 0).mean():.2%} |

## Map matching 质量

| 指标 | 结果 |
|---|---:|
| 订单匹配成功率 | {summary['matching_success_ratio']:.2%} |
| 高置信度订单比例 | {summary['high_confidence_ratio']:.2%} |
| 全部点平均 GPS-link 距离 | {summary['aggregate_mean_gps_link_distance_m']:.2f} m |
| 全部点 P90 GPS-link 距离 | {summary['aggregate_p90_gps_link_distance_m']:.2f} m |
| route ratio 在 0.8–1.3 的订单比例 | {summary['route_ratio_good_order_ratio']:.2%} |
| 订单置信度中位数 | {match_quality.matching_confidence.median():.3f} |

## 行为特征分布（订单级）

| 特征 | P25 | Median | P75 |
|---|---:|---:|---:|
| low-speed ratio | {fmt(features.low_speed_ratio.quantile(.25))} | {fmt(features.low_speed_ratio.median())} | {fmt(features.low_speed_ratio.quantile(.75))} |
| stop count/km | {fmt(features.stop_count_km.quantile(.25))} | {fmt(features.stop_count_km.median())} | {fmt(features.stop_count_km.quantile(.75))} |
| speed std (km/h) | {fmt(features.speed_std_kmh.quantile(.25))} | {fmt(features.speed_std_kmh.median())} | {fmt(features.speed_std_kmh.quantile(.75))} |
| intersection delay (s) | {fmt(features.intersection_delay_s.quantile(.25),1)} | {fmt(features.intersection_delay_s.median(),1)} | {fmt(features.intersection_delay_s.quantile(.75),1)} |

## 判定门槛

"""
    for key, value in pass_flags.items():
        report += f"- {'PASS' if value else 'FAIL'} — `{key}`\n"
    report += """

## 解释边界与下一步

1. 原始文件无表头，`driver_id/order_id` 是依据基数与每 ID 点数分布推断；应使用数据字典复核。
2. 坐标诊断显示原坐标与 OSM 系统性错位，而 GCJ-02→WGS84 后距离显著收敛；这强烈支持 GCJ-02 假设，但正式研究仍应以数据字典/提供方说明作最终确认。
3. 当前 OSM 不能代表 2016 年道路状态。正式 Stage1 前应获取历史 OSM 快照或同期权威路网，并升级为带方向、候选集与网络转移概率的 HMM/Viterbi matcher。
4. `intersection_delay` 当前使用距拓扑交叉节点 30m 且速度低于 10km/h 的累计秒数，是 primitive indicator，不是信号控制延误的因果估计。

分布总图见 `figures/feature_distributions.png`，典型订单清单见 `visual_case_index.csv`。
"""
    (out / "stage0_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
