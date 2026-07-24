"""Build a deterministic, stratified review pack without fabricating labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage0.canonical.manual_truth import REVIEW_COLUMNS


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", type=Path, nargs="+", required=True)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--core-sample-size", type=int)
    parser.add_argument("--double-review-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--stage0-root", type=Path, required=True)
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geojson", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def _stratified_sample(work: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    groups = list(work.groupby("sampling_stratum", sort=True))
    selected: list[pd.DataFrame] = []
    base = max(1, sample_size // max(1, len(groups)))
    for _, group in groups:
        selected.append(group.nsmallest(min(base, len(group)), "_random"))
    result = pd.concat(selected, ignore_index=True) if selected else work.iloc[:0]
    if len(result) < sample_size:
        remainder = work.loc[~work.order_id.isin(result.order_id)]
        result = pd.concat(
            [result, remainder.nsmallest(sample_size - len(result), "_random")],
            ignore_index=True,
        )
    return result.nsmallest(min(sample_size, len(result)), "_random")


def sample_pack(
    frame: pd.DataFrame, sample_size: int, seed: int, core_sample_size: int | None = None
) -> pd.DataFrame:
    work = frame.copy()
    work["order_id"] = work.order_id.astype(str)
    work["confidence_band"] = pd.qcut(
        work.mean_match_confidence.rank(method="first"), 4,
        labels=["low", "mid_low", "mid_high", "high"],
    )
    work["distance_band"] = pd.qcut(
        work.gps_length_m.rank(method="first"), 3,
        labels=["short", "medium", "long"],
    )
    work["route_feature_stratum"] = np.select(
        [
            work.unreasonable_detour_count.gt(0),
            work.interpolated_distance_share.gt(0.5),
            work.u_turn_count.gt(0),
            work.direction_gap_count.gt(0),
        ],
        ["detour", "long_interpolation", "u_turn", "direction_gap"],
        default="ordinary",
    )
    if "road_context_stratum" not in work:
        work["road_context_stratum"] = "ordinary_road"
    work["sampling_stratum"] = (
        work.route_quality_class.astype(str) + "|" +
        work.confidence_band.astype(str) + "|" +
        work.distance_band.astype(str) + "|" +
        work.route_feature_stratum.astype(str) + "|" +
        work.road_context_stratum.astype(str)
    )
    rng = np.random.default_rng(seed)
    work["_random"] = rng.random(len(work))
    if core_sample_size is None:
        result = _stratified_sample(work, sample_size)
    else:
        if not 0 <= core_sample_size <= sample_size:
            raise ValueError("core_sample_size must be between zero and sample_size")
        core = work.loc[work.route_quality_class.eq("core")]
        non_core = work.loc[~work.route_quality_class.eq("core")]
        if len(core) < core_sample_size or len(non_core) < sample_size - core_sample_size:
            raise ValueError("insufficient Core or non-Core orders for requested review design")
        result = pd.concat(
            [
                _stratified_sample(core, core_sample_size),
                _stratified_sample(non_core, sample_size - core_sample_size),
            ],
            ignore_index=True,
        ).sort_values(["route_quality_class", "date", "order_id"])
    result = result.drop(columns="_random")
    for column in REVIEW_COLUMNS:
        if column not in result:
            result[column] = "pending" if column == "review_status" else pd.NA
    result["review_status"] = "pending"
    return result


def read_parts(directory: Path, columns: list[str]) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(directory)
    return pd.concat([pd.read_parquet(path, columns=columns) for path in files], ignore_index=True)


def route_context(frame: pd.DataFrame, stage0_root: Path, roads: gpd.GeoDataFrame) -> pd.DataFrame:
    lookup = roads.set_index(roads.link_id.astype(str))
    node_degree = pd.concat([
        roads[["from_node"]].rename(columns={"from_node": "node"}),
        roads[["to_node"]].rename(columns={"to_node": "node"}),
    ]).node.value_counts().to_dict()
    rows = []
    for date, date_frame in frame.groupby("date"):
        routes = read_parts(
            stage0_root / "hmm_route_parts" / f"day={date}",
            ["order_id", "link_id", "route_sequence"],
        )
        routes["link_id"] = routes.link_id.astype(str)
        routes = routes.loc[routes.order_id.astype(str).isin(date_frame.order_id.astype(str))]
        for order_id, group in routes.groupby("order_id", sort=False):
            links = [link for link in group.link_id if link in lookup.index]
            selected = lookup.loc[links] if links else roads.iloc[:0]
            bridge = bool(selected.get("topology_bridge", pd.Series(dtype=bool)).fillna(False).any())
            tunnel = bool(selected.get("topology_tunnel", pd.Series(dtype=bool)).fillna(False).any())
            elevated = bool(
                selected.get("topology_layer", pd.Series(dtype=object)).astype(str).ne("0").any()
            )
            maximum_degree = max(
                [node_degree.get(int(node), 0) for node in pd.concat([
                    selected.get("from_node", pd.Series(dtype=int)),
                    selected.get("to_node", pd.Series(dtype=int)),
                ])],
                default=0,
            )
            context = "bridge" if bridge else "tunnel" if tunnel else "elevated" if elevated else (
                "complex_intersection" if maximum_degree >= 6 else "ordinary_road"
            )
            rows.append({
                "date": str(date), "order_id": str(order_id),
                "contains_bridge": bridge, "contains_tunnel": tunnel,
                "contains_nonzero_layer": elevated,
                "maximum_route_node_degree": maximum_degree,
                "road_context_stratum": context,
            })
    return pd.DataFrame(rows)


def write_review_geometries(
    pack: pd.DataFrame, stage0_root: Path, roads: gpd.GeoDataFrame, target: Path
) -> None:
    road_lookup = roads.set_index(roads.link_id.astype(str)).geometry
    features = []
    for date, date_pack in pack.groupby("date"):
        wanted = set(date_pack.order_id.astype(str))
        routes = read_parts(
            stage0_root / "hmm_route_parts" / f"day={date}",
            ["order_id", "link_id", "route_sequence"],
        )
        points = read_parts(
            stage0_root / "hmm_matched_points" / f"day={date}",
            ["order_id", "point_seq", "lon", "lat"],
        )
        for order_id, group in points.loc[points.order_id.astype(str).isin(wanted)].groupby("order_id"):
            ordered = group.sort_values("point_seq")
            if len(ordered) >= 2:
                features.append({
                    "date": str(date), "order_id": str(order_id), "geometry_role": "gps_trace",
                    "geometry": LineString(zip(ordered.lon, ordered.lat)),
                })
        for order_id, group in routes.loc[routes.order_id.astype(str).isin(wanted)].groupby("order_id"):
            links = group.sort_values("route_sequence").link_id.astype(str)
            geometries = [road_lookup[link] for link in links if link in road_lookup.index]
            if geometries:
                features.append({
                    "date": str(date), "order_id": str(order_id), "geometry_role": "matched_route",
                    "geometry": unary_union(geometries),
                })
    result = gpd.GeoDataFrame(features, geometry="geometry", crs=roads.crs).to_crs(4326)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(target, driver="GeoJSON")


def main() -> None:
    args = arguments()
    frames = []
    for path in args.quality:
        frame = pd.read_parquet(path)
        if "date" not in frame:
            date = path.stem.split("=")[-1]
            frame["date"] = date
        frames.append(frame)
    all_quality = pd.concat(frames, ignore_index=True)
    roads = gpd.read_parquet(args.roads)
    contexts = route_context(all_quality, args.stage0_root, roads)
    all_quality["date"] = all_quality.date.astype(str)
    all_quality["order_id"] = all_quality.order_id.astype(str)
    all_quality = all_quality.merge(contexts, on=["date", "order_id"], how="left")
    pack = sample_pack(all_quality, args.sample_size, args.seed, args.core_sample_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pack.to_parquet(args.output, index=False)
    csv_path = args.output.with_suffix(".csv")
    pack.to_csv(csv_path, index=False, encoding="utf-8-sig")
    double_review = pack.sample(
        n=min(args.double_review_size, len(pack)), random_state=args.seed
    ).loc[:, REVIEW_COLUMNS].copy()
    double_review["reviewer_id"] = pd.NA
    double_review["review_status"] = "pending"
    double_review_path = args.output.with_name(args.output.stem + "_double_review.csv")
    double_review.to_csv(double_review_path, index=False, encoding="utf-8-sig")
    write_review_geometries(pack, args.stage0_root, roads, args.geojson)
    manifest = {
        "status": "AWAITING_HUMAN_REVIEW",
        "review_sample_version": "stage0_route_truth_v2",
        "sample_size": int(len(pack)),
        "core_sample_size": int(pack.route_quality_class.eq("core").sum()),
        "double_review_size": int(min(args.double_review_size, len(pack))),
        "seed": args.seed,
        "source_quality_files": [path.as_posix() for path in args.quality],
        "review_geometry": args.geojson.as_posix(),
        "double_review_file": double_review_path.as_posix(),
        "review_requirements": {
            "minimum_completed_orders": 120,
            "recommended_completed_orders": 150,
            "double_review_subset_required": True,
            "minimum_double_review_pairs": 30,
            "minimum_double_review_agreement": 0.80,
            "maximum_core_major_error_rate": 0.15,
            "maximum_core_wrong_direction_rate": 0.05,
            "maximum_core_wrong_road_level_rate": 0.05,
            "maximum_core_unreasonable_detour_rate": 0.10,
        },
        "canonical_promotion_gate": "HOLD",
        "blocker": "Independent human judgments and adjudication are not yet present.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
