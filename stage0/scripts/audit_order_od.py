"""Audit first/last-observation OD proxies and attach route-planning quality flags.

The audited table is deliberately separate from ``order_base``: the empirical
OD proxy is a dispatch-time input candidate, while map-matching quality and
realized trip statistics remain audit-only fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--od-root", type=Path, default=Path("stage0/output/order_od"))
    parser.add_argument("--order-base-root", type=Path, default=Path("stage0/output/order_base"))
    parser.add_argument("--roads", type=Path, default=Path("map_data/xian_2017/xian_2017_core_roads.parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("stage0/output/order_od_audited"))
    parser.add_argument("--report-root", type=Path, default=Path("stage0/output/reports"))
    parser.add_argument("--dates", default="20161009,20161010,20161011,20161012,20161013,20161014,20161015,20161016,20161017,20161018,20161019")
    return parser.parse_args()


def parse_dates(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_road_index(path: Path) -> tuple[gpd.GeoDataFrame, shapely.STRtree]:
    roads = gpd.read_parquet(path).to_crs(32649).reset_index(drop=True)
    roads["link_id"] = roads["link_id"].astype(str)
    return roads, shapely.STRtree(roads.geometry.to_numpy())


def snap(points: np.ndarray, roads: gpd.GeoDataFrame, tree: shapely.STRtree) -> tuple[np.ndarray, np.ndarray]:
    indices, distances = tree.query_nearest(points, all_matches=False, return_distance=True)
    assigned = np.full(len(points), -1, dtype=np.int64)
    values = np.full(len(points), np.nan, dtype=float)
    if np.ndim(indices) == 2:
        assigned[indices[0]] = indices[1]
        values[indices[0]] = distances
    else:
        assigned[:] = np.asarray(indices, dtype=np.int64)
        values[:] = np.asarray(distances, dtype=float)
    links = np.full(len(points), None, dtype=object)
    valid = assigned >= 0
    links[valid] = roads.iloc[assigned[valid]]["link_id"].to_numpy()
    return links, values


def audit_day(date: str, args: argparse.Namespace, roads: gpd.GeoDataFrame, tree: shapely.STRtree) -> tuple[pd.DataFrame, dict]:
    od = pd.read_parquet(args.od_root / f"day={date}.parquet")
    base_path = args.order_base_root / f"day={date}.parquet"
    base = pd.read_parquet(base_path, columns=["order_id", "driver_id", "point_count", "duration_s"]).rename(
        columns={"driver_id": "order_base_driver_id", "point_count": "order_base_point_count", "duration_s": "order_base_duration_sec"}
    )
    od["order_id"] = od["order_id"].astype("string")
    base["order_id"] = base["order_id"].astype("string")
    output = od.merge(base, on="order_id", how="left", validate="one_to_one", indicator=True)
    output["order_base_aligned"] = output["_merge"].eq("both")
    output.drop(columns="_merge", inplace=True)
    output["driver_id_aligned"] = output["order_base_driver_id"].isna() | output["driver_id"].eq(output["order_base_driver_id"])
    output["coordinate_valid"] = (
        output["origin_lon"].between(108.0, 110.5)
        & output["destination_lon"].between(108.0, 110.5)
        & output["origin_lat"].between(33.0, 35.5)
        & output["destination_lat"].between(33.0, 35.5)
    )

    transformer = Transformer.from_crs(4326, 32649, always_xy=True)
    for prefix in ("origin", "destination"):
        x, y = transformer.transform(output[f"{prefix}_lon"].to_numpy(), output[f"{prefix}_lat"].to_numpy())
        links, distances = snap(shapely.points(x, y), roads, tree)
        output[f"{prefix}_snapped_link_id"] = links
        output[f"{prefix}_snap_distance_m"] = distances

    output["point_count_valid"] = output["raw_point_count"].ge(5)
    output["duration_valid"] = output["duration_sec"].between(30, 6 * 3600)
    strict = (
        output["order_base_aligned"] & output["driver_id_aligned"] & output["coordinate_valid"]
        & output["raw_point_count"].ge(10) & output["duration_sec"].between(60, 4 * 3600)
        & output["origin_snap_distance_m"].le(50) & output["destination_snap_distance_m"].le(50)
    )
    usable = (
        output["order_base_aligned"] & output["coordinate_valid"] & output["point_count_valid"] & output["duration_valid"]
        & output["origin_snap_distance_m"].le(100) & output["destination_snap_distance_m"].le(100)
    )
    strict = strict.fillna(False).to_numpy(dtype=bool)
    usable = usable.fillna(False).to_numpy(dtype=bool)
    output["od_quality_flag"] = np.select([strict, usable], ["A", "B"], default="C")
    output["od_route_eligible"] = usable

    summary = {
        "date": date,
        "orders_od": int(len(output)),
        "orders_order_base": int(len(base)),
        "order_base_alignment_ratio": float(output["order_base_aligned"].mean()),
        "driver_alignment_ratio": float(output["driver_id_aligned"].mean()),
        "coordinate_valid_ratio": float(output["coordinate_valid"].mean()),
        "route_eligible_ratio": float(output["od_route_eligible"].mean()),
        "quality_A_ratio": float(output["od_quality_flag"].eq("A").mean()),
        "quality_B_ratio": float(output["od_quality_flag"].eq("B").mean()),
        "quality_C_ratio": float(output["od_quality_flag"].eq("C").mean()),
        "raw_point_count_p10": float(output["raw_point_count"].quantile(0.10)),
        "raw_point_count_p50": float(output["raw_point_count"].quantile(0.50)),
        "raw_point_count_p90": float(output["raw_point_count"].quantile(0.90)),
        "duration_sec_p10": float(output["duration_sec"].quantile(0.10)),
        "duration_sec_p50": float(output["duration_sec"].quantile(0.50)),
        "duration_sec_p90": float(output["duration_sec"].quantile(0.90)),
        "origin_snap_distance_p50_m": float(output["origin_snap_distance_m"].quantile(0.50)),
        "origin_snap_distance_p90_m": float(output["origin_snap_distance_m"].quantile(0.90)),
        "origin_snap_distance_p95_m": float(output["origin_snap_distance_m"].quantile(0.95)),
        "destination_snap_distance_p50_m": float(output["destination_snap_distance_m"].quantile(0.50)),
        "destination_snap_distance_p90_m": float(output["destination_snap_distance_m"].quantile(0.90)),
        "destination_snap_distance_p95_m": float(output["destination_snap_distance_m"].quantile(0.95)),
    }
    return output, summary


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    roads, tree = build_road_index(args.roads)
    summaries = []
    for date in parse_dates(args.dates):
        output, summary = audit_day(date, args, roads, tree)
        output.to_parquet(args.output_root / f"day={date}.parquet", index=False, compression="zstd")
        summaries.append(summary)
        print(f"OD audit day={date} eligible={summary['route_eligible_ratio']:.4f} A={summary['quality_A_ratio']:.4f}", flush=True)
    report = pd.DataFrame(summaries)
    csv_path = args.report_root / "order_od_quality_summary.csv"
    report.to_csv(csv_path, index=False)
    manifest = {
        "od_source": "first_last_gps_observation_proxy",
        "coordinate_interpretation": "raw_gcj02_converted_to_wgs84",
        "quality_A": "aligned, valid, >=10 points, 60s-4h, both endpoint snap distances <=50m",
        "quality_B": "aligned, valid, >=5 points, 30s-6h, both endpoint snap distances <=100m",
        "quality_C": "otherwise; excluded from planned-route construction",
        "days": summaries,
    }
    (args.report_root / "order_od_quality_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [
        "# 20161009–20161019 OD quality audit", "",
        "OD is an empirical first/last-GPS proxy, not a platform-recorded dispatch OD. Coordinates use the Stage0 empirical GCJ-02 to WGS84 interpretation.", "",
        report.to_markdown(index=False, floatfmt=".4f"), "",
        "Only A/B rows are eligible for planned-route experiments; realized trip fields are audit-only and are forbidden model inputs.",
    ]
    (args.report_root / "order_od_quality_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
