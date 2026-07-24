"""POI cleaning and grade-aware raw category exposure."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from charset_normalizer import from_bytes

from .config import Stage0Config
from .coordinates import gcj02_to_wgs84
from .manifest import base_manifest, write_manifest


CATEGORY_RULES: dict[str, tuple[str, ...]] = {
    "school": ("学校", "教育", "大学", "中学", "小学", "幼儿园", "school", "college"),
    "hospital": ("医院", "诊所", "卫生", "医疗", "hospital", "clinic"),
    "transit": ("公交", "地铁", "车站", "机场", "客运", "transport", "station"),
    "residential": ("住宅", "小区", "公寓", "宿舍", "residential", "community"),
    "leisure": ("公园", "景区", "体育", "影院", "娱乐", "leisure", "park"),
    "parking": ("停车", "parking"),
    "pickup_dropoff_proxy": ("酒店", "宾馆", "机场", "车站", "商场", "景区", "hotel", "mall"),
    "commercial": ("餐饮", "购物", "商场", "公司", "银行", "酒店", "商业", "restaurant", "shop"),
}
STANDARD_CATEGORIES = [*CATEGORY_RULES, "other"]


def detect_encoding(path: Path) -> tuple[str, int]:
    sample = path.read_bytes()[:2_000_000]
    best = from_bytes(sample).best()
    encoding = best.encoding if best and best.encoding else "utf-8"
    decoded = sample.decode(encoding, errors="replace")
    return encoding, decoded.count("\ufffd")


def _find_column(columns: list[str], tokens: tuple[str, ...]) -> str | None:
    normalized = {column: str(column).strip().lower() for column in columns}
    for column, value in normalized.items():
        if any(token in value for token in tokens):
            return column
    return None


def standard_category(value: object) -> str:
    text = str(value or "").lower()
    for category, keywords in CATEGORY_RULES.items():
        if any(keyword.lower() in text for keyword in keywords):
            return category
    return "other"


def _bbox_share(lon: np.ndarray, lat: np.ndarray, bounds: np.ndarray) -> float:
    minx, miny, maxx, maxy = bounds
    return float(((lon >= minx - 0.05) & (lon <= maxx + 0.05) & (lat >= miny - 0.05) & (lat <= maxy + 0.05)).mean())


def deduplicate_nearest_assignments(joined: pd.DataFrame) -> pd.DataFrame:
    """Choose one deterministic nearest edge per POI, including equidistant ties."""
    return joined.sort_values(
        ["poi_id", "edge_distance_m", "edge_uid"], kind="stable"
    ).drop_duplicates("poi_id", keep="first")


def build_poi(config: Stage0Config, repo: Path, force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    source = config.path("poi", repo)
    output = config.path("output", repo) / "poi"
    network_dir = config.path("output", repo) / "network"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "poi_manifest.json"
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "PASS" and existing.get("config_hash") == config.digest:
            return existing
    edges = gpd.read_parquet(network_dir / "canonical_edges.parquet")
    encoding, replacement_count = detect_encoding(source)
    try:
        frame = pd.read_csv(source, encoding=encoding, low_memory=False)
    except (UnicodeDecodeError, pd.errors.ParserError):
        frame = pd.read_csv(source, encoding=encoding, engine="python", on_bad_lines="warn")
    columns = [str(column) for column in frame.columns]
    lon_col = _find_column(columns, ("lon", "lng", "longitude", "经度", "x坐标"))
    lat_col = _find_column(columns, ("lat", "latitude", "纬度", "y坐标"))
    category_col = _find_column(columns, ("category", "type", "类别", "分类", "大类", "中类"))
    name_col = _find_column(columns, ("name", "名称", "poi"))
    if not lon_col or not lat_col:
        raise ValueError(f"POI coordinate columns not found in {columns}")
    lon = pd.to_numeric(frame[lon_col], errors="coerce")
    lat = pd.to_numeric(frame[lat_col], errors="coerce")
    valid = lon.between(70, 140) & lat.between(10, 60)
    cleaned = frame.loc[valid].copy()
    cleaned["source_lon"] = lon.loc[valid].to_numpy()
    cleaned["source_lat"] = lat.loc[valid].to_numpy()
    network_wgs = edges.to_crs(4326)
    wgs_share = _bbox_share(cleaned.source_lon.to_numpy(), cleaned.source_lat.to_numpy(), network_wgs.total_bounds)
    gcj_lon, gcj_lat = gcj02_to_wgs84(cleaned.source_lon.to_numpy(), cleaned.source_lat.to_numpy())
    gcj_share = _bbox_share(gcj_lon, gcj_lat, network_wgs.total_bounds)
    interpretation = "gcj02" if gcj_share > wgs_share + 0.01 else "wgs84"
    if interpretation == "gcj02":
        cleaned["lon"], cleaned["lat"] = gcj_lon, gcj_lat
    else:
        cleaned["lon"], cleaned["lat"] = cleaned.source_lon, cleaned.source_lat
    category_text = cleaned[category_col].astype(str) if category_col else pd.Series("", index=cleaned.index)
    if name_col:
        category_text = category_text + " " + cleaned[name_col].astype(str)
    cleaned["raw_category"] = cleaned[category_col].astype(str) if category_col else "unknown"
    cleaned["standard_category"] = category_text.map(standard_category)
    cleaned["poi_id"] = np.arange(len(cleaned), dtype="int64")
    cleaned = cleaned.drop_duplicates(subset=["lon", "lat", "raw_category", *( [name_col] if name_col else [])]).copy()
    poi = gpd.GeoDataFrame(cleaned, geometry=gpd.points_from_xy(cleaned.lon, cleaned.lat), crs=4326).to_crs(edges.crs)
    ground = edges.loc[(edges.layer.astype(int) == 0) & ~edges.bridge.astype(bool) & ~edges.tunnel.astype(bool)].copy()
    if ground.empty:
        raise RuntimeError("no ground-level edges available for POI exposure")
    joined = gpd.sjoin_nearest(
        poi[["poi_id", "raw_category", "standard_category", "geometry"]],
        ground[["edge_uid", "geometry"]], how="left", max_distance=200.0, distance_col="edge_distance_m"
    )
    # GeoPandas returns every exactly equidistant edge.  POI exposure is one-to-one:
    # distance first, then stable edge_uid resolves ties without duplicating a POI.
    joined = deduplicate_nearest_assignments(joined)
    assigned = joined.dropna(subset=["edge_uid"]).copy()
    counts = (
        assigned.groupby(["edge_uid", "standard_category"]).size().unstack(fill_value=0)
        .reindex(columns=STANDARD_CATEGORIES, fill_value=0)
    )
    exposure = edges[["edge_uid", "layer", "bridge", "tunnel", "highway", "service", "access"]].copy()
    exposure = exposure.merge(counts, how="left", left_on="edge_uid", right_index=True)
    exposure[STANDARD_CATEGORIES] = exposure[STANDARD_CATEGORIES].fillna(0).astype("int32")
    poi.to_parquet(output / "cleaned_poi.parquet", index=False, compression="zstd")
    exposure.to_parquet(output / "link_poi_exposure.parquet", index=False, compression="zstd")
    mapping = (
        cleaned.groupby(["raw_category", "standard_category"]).size().rename("record_count").reset_index()
    )
    mapping.to_csv(output / "poi_category_mapping.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    manifest = {
        **base_manifest(repo, config.digest, [source]), "status": "PASS",
        "encoding": encoding, "replacement_character_count": replacement_count,
        "source_rows": len(frame), "valid_coordinate_rows": int(valid.sum()),
        "duplicate_rows_removed": int(valid.sum() - len(cleaned)),
        "coordinate_interpretation": interpretation,
        "wgs84_bbox_share": wgs_share, "gcj02_bbox_share": gcj_share,
        "assigned_poi_rows": len(assigned), "unassigned_poi_rows": len(poi) - len(assigned),
        "category_counts": cleaned.standard_category.value_counts().to_dict(),
        "runtime_sec": time.perf_counter() - started,
    }
    write_manifest(manifest_path, manifest)
    (output / "poi_quality_report.md").write_text(
        "# Stage 0 v5 POI quality report\n\n"
        f"- Encoding: `{encoding}`; replacement characters in sample: {replacement_count}\n"
        f"- Source rows: {len(frame):,}; valid coordinates: {int(valid.sum()):,}\n"
        f"- Coordinate interpretation: `{interpretation}` (WGS bbox={wgs_share:.3f}, converted bbox={gcj_share:.3f})\n"
        f"- Assigned to grade-compatible ground edges: {len(assigned):,}; unassigned: {len(poi)-len(assigned):,}\n\n"
        "POI is never used by the map matcher or HMM. Ground POIs are not assigned to bridge, tunnel, or non-zero-layer edges. "
        "Exposures are unweighted raw category counts.\n",
        encoding="utf-8",
    )
    return manifest
