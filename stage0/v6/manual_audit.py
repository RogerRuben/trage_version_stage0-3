"""Create a deterministic 100-order human review pack with self-contained maps."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from shapely import from_wkb

from .config import Stage0V6Config
from .coordinates import gcj02_to_wgs84
from .pipeline import load_fixed_sample

ALLOWED_LABELS = "v5正确|v6正确|两者都正确|两者都错误|无法判断"


def _read_partitioned(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("day=*/*.parquet"))
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _stable_hash(order_id: str) -> str:
    return hashlib.sha256(f"stage0-v6-audit|{order_id}".encode()).hexdigest()


def _geometry_lookup(path: Path, edge_ids: set[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    columns = [
        "edge_uid",
        "geometry",
        "highway",
        "bridge",
        "tunnel",
        "layer",
        "parallel_group",
    ]
    schema = pq.read_schema(path)
    frame = pq.read_table(
        path, columns=[column for column in columns if column in schema.names]
    ).to_pandas()
    frame = frame.loc[frame.edge_uid.astype(str).isin(edge_ids)].copy()
    lookup = {
        str(row.edge_uid): from_wkb(row.geometry)
        for row in frame.itertuples(index=False)
        if row.geometry is not None
    }
    return lookup, frame


def _coordinates(geometry: Any) -> list[list[tuple[float, float]]]:
    if geometry is None:
        return []
    if geometry.geom_type == "LineString":
        return [list(geometry.coords)]
    if geometry.geom_type == "MultiLineString":
        return [list(part.coords) for part in geometry.geoms]
    return []


def _svg_map(
    raw_xy: list[tuple[float, float]],
    v5_lines: list[list[tuple[float, float]]],
    v6_lines: list[list[tuple[float, float]]],
    metadata: dict[str, Any],
) -> str:
    all_points = [*raw_xy]
    for line in [*v5_lines, *v6_lines]:
        all_points.extend(line)
    width, height, pad = 960, 640, 40
    if not all_points:
        all_points = [(0.0, 0.0), (1.0, 1.0)]
    xs, ys = zip(*all_points)
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)

    def project(point):
        x = pad + (point[0] - xmin) / dx * (width - 2 * pad)
        y = height - pad - (point[1] - ymin) / dy * (height - 2 * pad)
        return f"{x:.2f},{y:.2f}"

    def polylines(lines, color, stroke, opacity=1.0):
        return "".join(
            f'<polyline points="{" ".join(project(point) for point in line)}" '
            f'fill="none" stroke="{color}" stroke-width="{stroke}" opacity="{opacity}"/>'
            for line in lines
            if len(line) >= 2
        )

    raw_line = [raw_xy] if len(raw_xy) >= 2 else []
    circles = "".join(
        f'<circle cx="{project(point).split(",")[0]}" cy="{project(point).split(",")[1]}" r="1.8" fill="#d62728"/>'
        for point in raw_xy
    )
    meta_text = html.escape(json.dumps(metadata, ensure_ascii=False, indent=2))
    return f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>{html.escape(str(metadata['order_id']))}</title>
<style>body{{font-family:system-ui;margin:20px}}svg{{border:1px solid #bbb;background:#fafafa}}pre{{white-space:pre-wrap}}</style>
<h2>Stage 0 v5/v6 audit: {html.escape(str(metadata['order_id']))}</h2>
<p><span style="color:#d62728">● raw GPS</span> · <span style="color:#1f77b4">━ v5</span> · <span style="color:#2ca02c">━ v6 Valhalla</span></p>
<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}">
<g opacity=".25" stroke="#ddd">{"".join(f'<line x1="{x}" y1="0" x2="{x}" y2="{height}"/>' for x in range(0,width,80))}{"".join(f'<line x1="0" y1="{y}" x2="{width}" y2="{y}"/>' for y in range(0,height,80))}</g>
{polylines(v5_lines, "#1f77b4", 5, .8)}
{polylines(v6_lines, "#2ca02c", 3, .9)}
{polylines(raw_line, "#d62728", 1.5, .75)}
{circles}
</svg><pre>{meta_text}</pre></html>"""


def generate_manual_audit(config: Stage0V6Config) -> Path:
    output = config.path("output")
    target = output / "manual_audit"
    maps = target / "maps"
    maps.mkdir(parents=True, exist_ok=True)
    for stale_map in maps.glob("*.html"):
        stale_map.unlink()
    sample = load_fixed_sample(config)
    v6_quality = _read_partitioned(output / "hot" / "route_quality")
    v6_routes = _read_partitioned(output / "hot" / "route_parts")
    v5_quality = _read_partitioned(config.path("v5_output") / "route_quality")
    v5_routes = _read_partitioned(config.path("v5_output") / "route_parts")
    v5_quality = v5_quality[["order_id", "successful_reconstruction", "route_quality"]].rename(
        columns={
            "successful_reconstruction": "v5_success",
            "route_quality": "v5_quality",
        }
    )
    order_features = v6_quality.merge(v5_quality, on="order_id", how="left")
    point_features = (
        sample.points.groupby("order_id")
        .agg(point_count=("timestamp", "size"), unique_locations=("lon", lambda x: x.nunique()))
        .reset_index()
    )
    order_features = order_features.merge(point_features, on="order_id", how="left")
    order_features["v6_success"] = order_features.successful_reconstruction.fillna(False)
    order_features["v5_success"] = order_features.v5_success.fillna(False)
    order_features["v5_success_v6_failed"] = order_features.v5_success & ~order_features.v6_success
    order_features["v6_success_v5_failed"] = order_features.v6_success & ~order_features.v5_success
    order_features["gps_sparse"] = order_features.point_count.le(
        order_features.point_count.quantile(0.20)
    )
    order_features["low_speed_stop"] = (
        order_features.unique_locations / order_features.point_count
    ).le(0.55)
    v6_sets = (
        v6_routes.dropna(subset=["canonical_edge_uid"])
        .groupby("order_id")
        .canonical_edge_uid.agg(lambda values: set(map(str, values)))
    )
    v5_sets = (
        v5_routes.groupby("order_id").edge_uid.agg(lambda values: set(map(str, values)))
    )
    overlap = {}
    for order_id in order_features.order_id.astype(str):
        left, right = v5_sets.get(order_id, set()), v6_sets.get(order_id, set())
        overlap[order_id] = len(left & right) / max(len(left | right), 1)
    order_features["v5_v6_route_different"] = order_features.order_id.map(overlap).lt(0.8)

    edge_ids = set(v5_routes.edge_uid.astype(str)) | set(
        v6_routes.canonical_edge_uid.dropna().astype(str)
    )
    geometry, edge_meta = _geometry_lookup(config.path("canonical_edges"), edge_ids)
    meta = edge_meta.set_index(edge_meta.edge_uid.astype(str))
    tags_by_order: dict[str, set[str]] = {}
    for row in order_features.itertuples(index=False):
        tags: set[str] = set()
        if row.v5_success_v6_failed:
            tags.add("v5成功但v6失败")
        if row.v6_success_v5_failed:
            tags.add("v6成功但v5失败")
        if row.v5_v6_route_different:
            tags.add("v5与v6路线不同")
        if row.gps_sparse:
            tags.add("GPS稀疏")
        if row.low_speed_stop:
            tags.add("低速停车")
        route_edges = v6_sets.get(str(row.order_id), set())
        route_meta = meta.loc[meta.index.intersection(route_edges)]
        highways = set(route_meta.get("highway", pd.Series(dtype=str)).astype(str))
        if any(value.endswith("_link") for value in highways):
            tags.add("匝道")
        if route_meta.get("bridge", pd.Series(dtype=bool)).fillna(False).any():
            tags.add("桥梁/隧道")
        if route_meta.get("tunnel", pd.Series(dtype=bool)).fillna(False).any():
            tags.add("桥梁/隧道")
        layers = pd.to_numeric(
            route_meta.get("layer", pd.Series(dtype=float)), errors="coerce"
        ).fillna(0)
        has_elevated_structure = (
            layers.ne(0).any()
            or route_meta.get("bridge", pd.Series(dtype=bool)).fillna(False).any()
        )
        if has_elevated_structure:
            tags.add("高架/地面")
        major = {"motorway", "trunk", "primary", "secondary"}
        minor = {"service", "living_street", "residential", "tertiary"}
        if highways.intersection(major) and (
            highways.intersection(minor)
            or any(value.endswith("_link") for value in highways)
        ):
            tags.add("主路/辅路")
        if row.route_part_count >= order_features.route_part_count.quantile(0.80):
            tags.add("复杂交叉口")
        if not tags.intersection({"匝道", "桥梁/隧道", "复杂交叉口", "高架/地面"}):
            tags.add("普通直线路段")
        tags_by_order[str(row.order_id)] = tags

    desired = [
        "普通直线路段",
        "复杂交叉口",
        "高架/地面",
        "主路/辅路",
        "匝道",
        "桥梁/隧道",
        "GPS稀疏",
        "低速停车",
        "v5成功但v6失败",
        "v6成功但v5失败",
        "v5与v6路线不同",
    ]
    selected: list[str] = []
    ordered_ids = sorted(tags_by_order, key=_stable_hash)
    for tag in desired:
        for order_id in ordered_ids:
            if order_id not in selected and tag in tags_by_order[order_id]:
                selected.append(order_id)
                if sum(tag in tags_by_order[item] for item in selected) >= 8:
                    break
    for order_id in ordered_ids:
        if len(selected) >= int(config.section("runtime")["manual_audit_sample_count"]):
            break
        if order_id not in selected:
            selected.append(order_id)

    review_rows = []
    for index, order_id in enumerate(selected):
        raw = sample.points.loc[sample.points.order_id.astype(str).eq(order_id)].sort_values(
            "timestamp", kind="stable"
        )
        lon, lat = gcj02_to_wgs84(raw.lon.to_numpy(), raw.lat.to_numpy())
        raw_xy = list(zip(lon.tolist(), lat.tolist()))
        v5_ids = v5_routes.loc[v5_routes.order_id.astype(str).eq(order_id), "edge_uid"].astype(str)
        v6_ids = v6_routes.loc[
            v6_routes.order_id.astype(str).eq(order_id), "canonical_edge_uid"
        ].dropna().astype(str)
        v5_lines = [
            line for edge_id in v5_ids for line in _coordinates(geometry.get(edge_id))
        ]
        v6_lines = [
            line for edge_id in v6_ids for line in _coordinates(geometry.get(edge_id))
        ]
        feature = order_features.loc[
            order_features.order_id.astype(str).eq(order_id)
        ].iloc[0]
        metadata = {
            "order_id": order_id,
            "tags": sorted(tags_by_order[order_id]),
            "v5_quality": feature.get("v5_quality"),
            "v6_quality": feature.get("route_quality"),
            "v5_success": bool(feature.v5_success),
            "v6_success": bool(feature.v6_success),
            "route_edge_jaccard": overlap[order_id],
        }
        map_name = f"{index:03d}_{order_id}.html"
        (maps / map_name).write_text(
            _svg_map(raw_xy, v5_lines, v6_lines, metadata), encoding="utf-8"
        )
        review_rows.append(
            {
                "review_sequence": index,
                "order_id": order_id,
                "audit_tags": "|".join(sorted(tags_by_order[order_id])),
                "v5_quality": feature.get("v5_quality"),
                "v6_quality": feature.get("route_quality"),
                "v5_success": bool(feature.v5_success),
                "v6_success": bool(feature.v6_success),
                "route_edge_jaccard": overlap[order_id],
                "map_path": f"maps/{map_name}",
                "human_label": "",
                "allowed_labels": ALLOWED_LABELS,
                "review_notes": "",
                "review_status": "pending",
            }
        )
    review = pd.DataFrame(review_rows)
    review.to_csv(target / "manual_review_100.csv", index=False, encoding="utf-8-sig")
    review.to_parquet(target / "manual_review_100.parquet", index=False, compression="zstd")
    manifest = {
        "sample_order_sha256": config.section("sample")["expected_sha256"],
        "selected_orders": int(len(review)),
        "selection_method": "stable hash with multi-label stratum coverage, then deterministic fill",
        "tag_counts": {
            tag: int(review.audit_tags.str.contains(tag, regex=False).sum()) for tag in desired
        },
        "review_complete": False,
        "accuracy_claim_allowed": False,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target
