"""Clean UTF-8 Xi'an POIs and compute multi-buffer link-level POI exposure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely.strtree import STRtree

from run_full_day_2017 import FastRoadMatcher, gcj02_to_wgs84


CATEGORIES = [
    "school", "hospital", "commercial", "restaurant", "transit", "bus_stop",
    "residential", "office", "scenic", "parking", "other",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poi", type=Path, required=True)
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("stage0_output"))
    parser.add_argument("--input-crs", choices=["auto", "wgs84", "gcj02"], default="auto")
    return parser.parse_args()


def classify_poi(raw_major: pd.Series, raw_middle: pd.Series) -> pd.Series:
    major = raw_major.fillna("").astype(str)
    middle = raw_middle.fillna("").astype(str)
    joined = major + "|" + middle
    result = pd.Series("other", index=major.index, dtype="string")
    result.loc[major.eq("餐饮美食")] = "restaurant"
    result.loc[major.eq("购物消费") | major.eq("金融机构") | middle.str.contains("市场|超市|便利店|商场|购物中心", regex=True)] = "commercial"
    result.loc[major.eq("公司企业") | middle.str.contains("写字楼|产业园|商务中心", regex=True)] = "office"
    result.loc[major.eq("旅游景点") | middle.str.contains("景区|景点|公园|博物馆|纪念馆", regex=True)] = "scenic"
    result.loc[middle.str.contains("住宅区|宿舍|小区", regex=True)] = "residential"
    result.loc[middle.str.contains("停车场|停车区", regex=True)] = "parking"
    result.loc[middle.str.contains("公交站", regex=True)] = "bus_stop"
    result.loc[middle.str.contains("地铁|火车站|高铁站|机场|客运站|长途汽车站", regex=True)] = "transit"
    result.loc[middle.str.contains("医院|诊所|急救|妇幼|疾病防控|卫生院", regex=True)] = "hospital"
    result.loc[middle.str.contains("幼儿园|小学|中学|大学|学院|学校", regex=True)] = "school"
    return result


def coordinate_diagnostic(raw: pd.DataFrame, matcher: FastRoadMatcher) -> dict:
    minx, miny, maxx, maxy = matcher.roads.total_bounds
    inverse = Transformer.from_crs(32649, 4326, always_xy=True)
    west, south = inverse.transform(minx, miny)
    east, north = inverse.transform(maxx, maxy)
    covered = raw[
        raw["经度"].between(west - 0.01, east + 0.01)
        & raw["纬度"].between(south - 0.01, north + 0.01)
    ]
    sample = covered.sample(min(20_000, len(covered)), random_state=20260705)
    direct_x, direct_y = matcher.transformer.transform(sample["经度"].to_numpy(), sample["纬度"].to_numpy())
    _, direct_dist, *_ = matcher.match(np.asarray(direct_x), np.asarray(direct_y))
    lon, lat = gcj02_to_wgs84(sample["经度"].to_numpy(), sample["纬度"].to_numpy())
    conv_x, conv_y = matcher.transformer.transform(lon, lat)
    _, conv_dist, *_ = matcher.match(np.asarray(conv_x), np.asarray(conv_y))
    return {
        "sample_size": int(len(sample)),
        "diagnostic_scope": "POIs within the core-road bounding box",
        "direct_wgs84_p50_m": float(np.quantile(direct_dist, 0.5)),
        "direct_wgs84_p90_m": float(np.quantile(direct_dist, 0.9)),
        "gcj02_to_wgs84_p50_m": float(np.quantile(conv_dist, 0.5)),
        "gcj02_to_wgs84_p90_m": float(np.quantile(conv_dist, 0.9)),
    }


def nearest_distance(roads_geom: np.ndarray, poi_geom: np.ndarray, category: np.ndarray) -> np.ndarray:
    selected = poi_geom[category]
    if not len(selected):
        return np.full(len(roads_geom), np.nan)
    tree = STRtree(selected)
    _, distance = tree.query_nearest(roads_geom, return_distance=True)
    return distance


def build_exposure(roads: gpd.GeoDataFrame, poi: gpd.GeoDataFrame) -> pd.DataFrame:
    roads_proj = roads.to_crs(32649).reset_index(drop=True)
    poi_proj = poi.to_crs(32649).reset_index(drop=True)
    road_geom = roads_proj.geometry.to_numpy()
    poi_geom = poi_proj.geometry.to_numpy()
    categories = poi_proj.poi_category.astype(str).to_numpy()
    tree = STRtree(poi_geom)
    output = pd.DataFrame({"link_id": roads_proj.link_id.astype(str), "link_length_m": roads_proj.length_m})
    for radius in [50, 100, 200]:
        pairs = tree.query(road_geom, predicate="dwithin", distance=float(radius))
        pair_frame = pd.DataFrame({"road": pairs[0], "category": categories[pairs[1]]})
        counts = pair_frame.value_counts(["road", "category"]).unstack(fill_value=0)
        for category in CATEGORIES:
            output[f"poi_count_{radius}m_{category}"] = counts.get(category, pd.Series(dtype=int)).reindex(
                range(len(output)), fill_value=0
            ).to_numpy(dtype="int32")
    buffer_area_km2 = (2 * 100 * output.link_length_m + np.pi * 100**2) / 1_000_000
    for category in CATEGORIES:
        output[f"poi_density_100m_{category}"] = output[f"poi_count_100m_{category}"] / buffer_area_km2
    for category in ["school", "hospital", "transit"]:
        output[f"nearest_{category}_dist"] = nearest_distance(road_geom, poi_geom, categories == category)
    activity_categories = ["school", "hospital", "commercial", "restaurant", "transit", "bus_stop", "office"]
    activity = sum(np.log1p(output[f"poi_count_100m_{category}"].to_numpy()) for category in activity_categories)
    output["activity_intensity_index"] = pd.Series(activity).rank(pct=True).to_numpy()
    return output


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.poi, encoding="utf-8")
    required = {"名称", "大类", "中类", "经度", "纬度"}
    if not required.issubset(raw.columns):
        raise ValueError(f"POI columns missing: {sorted(required - set(raw.columns))}")
    object_columns = raw.select_dtypes(include="object").columns
    replacement_count = int(sum(raw[c].astype(str).str.contains("�", regex=False).sum() for c in object_columns))
    if replacement_count:
        raise UnicodeError(f"detected {replacement_count} replacement characters; refusing to continue")

    matcher = FastRoadMatcher(args.roads, args.nodes)
    diagnostic = coordinate_diagnostic(raw, matcher)
    interpretation = args.input_crs
    if interpretation == "auto":
        interpretation = "gcj02" if diagnostic["gcj02_to_wgs84_p50_m"] < diagnostic["direct_wgs84_p50_m"] else "wgs84"
    lon = raw["经度"].to_numpy(dtype=float); lat = raw["纬度"].to_numpy(dtype=float)
    if interpretation == "gcj02":
        lon, lat = gcj02_to_wgs84(lon, lat)
    valid = np.isfinite(lon) & np.isfinite(lat) & pd.Series(lon).between(107, 111).to_numpy() & pd.Series(lat).between(32, 36).to_numpy()
    poi = pd.DataFrame({
        "poi_id": [f"xian_poi_{i:06d}" for i in range(len(raw))],
        "source_lon": raw["经度"], "source_lat": raw["纬度"], "lon": lon, "lat": lat,
        "poi_raw_type": raw["大类"].astype(str) + "|" + raw["中类"].astype(str),
        "poi_category": classify_poi(raw["大类"], raw["中类"]), "name": raw["名称"],
        "source": args.poi.name, "quality_flag": np.where(valid, "valid", "invalid_geometry"),
    })
    poi = poi.loc[valid].reset_index(drop=True)
    poi_geo = gpd.GeoDataFrame(poi, geometry=gpd.points_from_xy(poi.lon, poi.lat), crs=4326)
    args.output_root.mkdir(parents=True, exist_ok=True)
    cleaned_path = args.output_root / "poi_cleaned.parquet"
    poi_geo.to_parquet(cleaned_path, index=False, compression="zstd")
    roads = gpd.read_parquet(args.roads)
    exposure = build_exposure(roads, poi_geo)
    exposure.to_parquet(args.output_root / "stage0_link_poi_exposure.parquet", index=False, compression="zstd")
    log = {
        "poi_file_path": str(args.poi.resolve()), "encoding_used": "utf-8",
        "row_count": int(len(raw)), "valid_geometry_count": int(len(poi_geo)),
        "replacement_character_count": replacement_count, "coordinate_diagnostic": diagnostic,
        "coordinate_interpretation": interpretation,
        "category_mapping_summary": poi.poi_category.value_counts().to_dict(),
    }
    (args.output_root / "poi_processing_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
