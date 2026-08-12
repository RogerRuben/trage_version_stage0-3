"""Stage 3 S2A full-network foundation and operational speed-domain freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .mvt import decode_tile, tile_point_to_lonlat


CONFIG_SCHEMA = "stage3_s2a_network_foundation_config.1"
PHASE_STATUS = "STAGE3_S2A_NETWORK_FOUNDATION_FROZEN"
VALID_MAPPING = {
    "EXACT_VALHALLA",
    "EXACT_OSM_ENDPOINT_DIRECTION",
    "CORROBORATED_TOPOLOGY",
    "UNMAPPED_FULL_NETWORK_EDGE",
    "AMBIGUOUS",
}
PROVENANCE = {
    "KNOWN_STAGE0_OSM",
    "INFERRED_SPEED_AND_CLASS",
    "INFERRED_CLASS_DOMINANT",
    "ROAD_CLASS_PRIOR_ONLY",
    "UNKNOWN",
}
ROAD_CLASS_NAMES = {
    0: "motorway",
    1: "trunk",
    2: "primary",
    3: "secondary",
    4: "tertiary",
    5: "unclassified",
    6: "residential",
    7: "service_other",
}
AUTO_ACCESS_BIT = 1


class Stage3S2AError(ValueError):
    """Raised when an S2A binding, scope, or acceptance gate fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def payload_hash(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("artifact_sha256", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), temporary, compression="zstd")
    os.replace(temporary, path)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Stage3S2AError(f"JSON is not an object: {path}")
    return payload


def load_config(path: Path, root: Path) -> dict[str, Any]:
    config = read_json(path)
    expected_auth = {
        "s2a": True,
        "s2b": False,
        **{f"s{i}": False for i in range(3, 9)},
        "stage4": False,
    }
    guards = config.get("scope_guards", {})
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("phase") != "S2A_FULL_NETWORK_FOUNDATION"
        or config.get("execution_authorization") != "S2A_ONLY"
        or config.get("authorizations") != expected_auth
        or config.get("next_phase_authorized") is not False
        or not guards
        or any(bool(value) for value in guards.values())
    ):
        raise Stage3S2AError("S2A configuration authorizes forbidden work")
    dates = config.get("train_only_dates", [])
    if dates != [f"201610{day:02d}" for day in range(9, 25)] or "20161031" in dates:
        raise Stage3S2AError("historical speed dates are not the frozen Train-only window")
    speed = config.get("speed_inference", {})
    if speed.get("candidate_quantiles") != [0.85, 0.9, 0.95]:
        raise Stage3S2AError("free-flow quantiles are not preregistered")
    if speed.get("av_caps_kmh") != [60, 80, 120]:
        raise Stage3S2AError("AV caps changed")
    for value in config.get("paths", {}).values():
        if not isinstance(value, str):
            raise Stage3S2AError("invalid S2A path binding")
    return config


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, encoding="utf-8"
    ).strip()


def source_descriptor(path: Path, root: Path | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise Stage3S2AError(f"missing bound source: {path}")
    resolved = path.resolve()
    label = resolved.as_posix()
    if root:
        try:
            label = resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return {"path": label, "size_bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def parquet_descriptor(path: Path, root: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    descriptor = source_descriptor(path, root)
    descriptor.update(
        {
            "row_count": parquet.metadata.num_rows,
            "schema": str(parquet.schema_arrow),
            "schema_sha256": hashlib.sha256(str(parquet.schema_arrow).encode()).hexdigest(),
        }
    )
    return descriptor


def stage3_edge_uid(valhalla_edge_id: int) -> str:
    source = f"valhalla-3.8.2|{int(valhalla_edge_id)}"
    return "s3e_" + hashlib.sha256(source.encode()).hexdigest()[:24]


def stage3_node_uid(valhalla_node_id: int) -> str:
    source = f"valhalla-3.8.2-node|{int(valhalla_node_id)}"
    return "s3n_" + hashlib.sha256(source.encode()).hexdigest()[:24]


def graph_id(tile_base: int, local_id: int) -> int:
    return (int(tile_base) & 0x1FFFFFF) | (int(local_id) << 25)


def _web_tile_range(bounds: tuple[float, float, float, float], z: int) -> list[tuple[int, int]]:
    min_lon, min_lat, max_lon, max_lat = bounds
    scale = 1 << z

    def x(lon: float) -> int:
        return int((lon + 180.0) / 360.0 * scale)

    def y(lat: float) -> int:
        return int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * scale)

    return [
        (tile_x, tile_y)
        for tile_x in range(x(min_lon), x(max_lon) + 1)
        for tile_y in range(y(max_lat), y(min_lat) + 1)
    ]


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lon1, lat1 = map(math.radians, a)
    lon2, lat2 = map(math.radians, b)
    dlon, dlat = lon2 - lon1, lat2 - lat1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371008.8 * 2 * math.asin(math.sqrt(value))


@dataclass
class OsmData:
    bounds: tuple[float, float, float, float]
    ways: dict[int, dict[str, Any]]
    controls: list[dict[str, Any]]
    restrictions: list[dict[str, Any]]
    node_locations: dict[int, tuple[float, float]]


def read_osm(path: Path) -> OsmData:
    """Read exact OSM ids, member roles/refs, and positive control evidence."""

    import osmium

    ways: dict[int, dict[str, Any]] = {}
    controls: list[dict[str, Any]] = []
    restrictions: list[dict[str, Any]] = []
    node_locations: dict[int, tuple[float, float]] = {}

    class Handler(osmium.SimpleHandler):
        def node(self, node: Any) -> None:
            highway = node.tags.get("highway", "")
            if highway in {"traffic_signals", "stop", "give_way", "mini_roundabout"}:
                lon, lat = float(node.location.lon), float(node.location.lat)
                node_locations[int(node.id)] = (lon, lat)
                controls.append(
                    {
                        "evidence_id": f"node/{node.id}/{highway}",
                        "osm_entity_type": "node",
                        "osm_node_id": int(node.id),
                        "osm_way_id": None,
                        "raw_highway": highway,
                        "raw_junction": None,
                        "control_evidence_type": {
                            "traffic_signals": "SIGNALIZED",
                            "stop": "STOP_CONTROL",
                            "give_way": "YIELD_CONTROL",
                            "mini_roundabout": "ROUNDABOUT",
                        }[highway],
                        "lon": lon,
                        "lat": lat,
                        "positive_evidence_only": True,
                        "source": "frozen_osm_pbf",
                    }
                )

        def way(self, way: Any) -> None:
            highway = way.tags.get("highway")
            if not highway:
                return
            nodes: list[tuple[int, float, float]] = []
            for item in way.nodes:
                try:
                    nodes.append((int(item.ref), float(item.lon), float(item.lat)))
                except osmium.InvalidLocationError:
                    continue
            tags = {
                key: way.tags.get(key)
                for key in ("highway", "maxspeed", "junction", "bridge", "tunnel", "layer", "oneway")
                if way.tags.get(key) is not None
            }
            ways[int(way.id)] = {"tags": tags, "nodes": nodes}
            if tags.get("junction") == "roundabout":
                controls.append(
                    {
                        "evidence_id": f"way/{way.id}/roundabout",
                        "osm_entity_type": "way",
                        "osm_node_id": None,
                        "osm_way_id": int(way.id),
                        "raw_highway": highway,
                        "raw_junction": "roundabout",
                        "control_evidence_type": "ROUNDABOUT",
                        "lon": None,
                        "lat": None,
                        "positive_evidence_only": True,
                        "source": "frozen_osm_pbf",
                    }
                )

        def relation(self, relation: Any) -> None:
            if relation.tags.get("type") != "restriction":
                return
            members = [
                {"type": member.type, "ref": int(member.ref), "role": str(member.role)}
                for member in relation.members
            ]
            restrictions.append(
                {
                    "restriction_id": int(relation.id),
                    "relation_type": "restriction",
                    "restriction_type": relation.tags.get("restriction", "UNKNOWN"),
                    "members": members,
                    "from_members": [m for m in members if m["role"] == "from"],
                    "via_members": [m for m in members if m["role"] == "via"],
                    "to_members": [m for m in members if m["role"] == "to"],
                }
            )

    reader = osmium.io.Reader(str(path))
    box = reader.header().box()
    reader.close()
    Handler().apply_file(str(path), locations=True)
    bounds = (float(box.bottom_left.lon), float(box.bottom_left.lat), float(box.top_right.lon), float(box.top_right.lat))
    return OsmData(bounds, ways, controls, restrictions, node_locations)


def _osm_cache_paths(output: Path) -> tuple[Path, Path]:
    return output / "osm_semantics_cache.json", output / "osm_cache_binding.json"


def load_or_read_osm(path: Path, output: Path) -> OsmData:
    cache_path, binding_path = _osm_cache_paths(output)
    pbf_hash = sha256_file(path)
    if cache_path.is_file() and binding_path.is_file():
        binding = read_json(binding_path)
        if binding.get("pbf_sha256") == pbf_hash and binding.get("cache_sha256") == sha256_file(cache_path):
            payload = read_json(cache_path)
            ways = {int(key): value for key, value in payload["ways"].items()}
            node_locations = {int(key): tuple(value) for key, value in payload["node_locations"].items()}
            return OsmData(tuple(payload["bounds"]), ways, payload["controls"], payload["restrictions"], node_locations)
    osm = read_osm(path)
    payload = {
        "bounds": osm.bounds, "ways": osm.ways, "controls": osm.controls,
        "restrictions": osm.restrictions, "node_locations": osm.node_locations,
    }
    atomic_json(cache_path, payload)
    atomic_json(binding_path, {"pbf_sha256": pbf_hash, "cache_sha256": sha256_file(cache_path)})
    return osm


def _edge_attributes() -> list[str]:
    return [
        "edge.id", "edge.way_id", "edge.length", "edge.speed", "edge.speed_limit",
        "edge.use", "edge.road_class", "edge.access_forward", "edge.access_backward",
        "edge.bridge", "edge.tunnel", "edge.roundabout", "edge.layer",
        "edge.traffic_signal_forward", "edge.traffic_signal_backward",
        "edge.stop_sign_forward", "edge.stop_sign_backward",
        "edge.yield_sign_forward", "edge.yield_sign_backward",
    ]


def export_valhalla_network(
    valhalla_config: Path, osm: OsmData, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Export all non-shortcut Valhalla edge directions with auto-access."""

    import valhalla

    raw_config = read_json(valhalla_config)
    z = int(config["network_export"]["mvt_zoom"])
    raw_config["loki"]["service_defaults"]["mvt_min_zoom_road_class"] = [z] * 8
    actor = valhalla.Actor(raw_config)
    edge_by_id: dict[int, dict[str, Any]] = {}
    node_by_id: dict[int, dict[str, Any]] = {}
    feature_count = 0
    duplicate_count = 0
    for tile_x, tile_y in _web_tile_range(osm.bounds, z):
        request = {
            "tile": {"z": z, "x": tile_x, "y": tile_y},
            "filters": {"action": "include", "attributes": _edge_attributes()},
            "generalize": int(config["network_export"]["generalize"]),
        }
        tile = decode_tile(actor.tile(request))
        for node in tile.get("nodes", []):
            point = node["geometry"]
            lon, lat = tile_point_to_lonlat(
                point, z=z, x=tile_x, y=tile_y, extent=node["extent"]
            )
            node_id = int(node["id"])
            node_by_id.setdefault(
                node_id,
                {
                    "valhalla_node_id": node_id,
                    "stage3_node_uid": stage3_node_uid(node_id),
                    "lon": lon,
                    "lat": lat,
                    "valhalla_node_type": int(node["properties"].get("type", -1)),
                    "valhalla_traffic_signal": bool(node["properties"].get("traffic_signal", False)),
                },
            )
        for feature in tile.get("edges", []):
            feature_count += 1
            props = feature["properties"]
            coords = feature["geometry"]
            if not coords or not isinstance(coords[0], tuple):
                continue
            geometry = [
                tile_point_to_lonlat(point, z=z, x=tile_x, y=tile_y, extent=feature["extent"])
                for point in coords
            ]
            for suffix, direction in (("fwd", "F"), ("bwd", "R")):
                local_id = props.get(f"edge_id:{suffix}")
                access = int(props.get(f"access:{suffix}", 0))
                if local_id is None or not access & AUTO_ACCESS_BIT:
                    continue
                edge_id = graph_id(int(feature["id"]), int(local_id))
                oriented = geometry if suffix == "fwd" else list(reversed(geometry))
                candidate = {
                    "stage3_edge_uid": stage3_edge_uid(edge_id),
                    "valhalla_directed_edge_id": edge_id,
                    "osm_way_id": int(props["osm_id"]) if props.get("osm_id") is not None else None,
                    "direction": direction,
                    "geometry": json.dumps(oriented, separators=(",", ":")),
                    "geometry_point_count": len(oriented),
                    "start_lon": oriented[0][0], "start_lat": oriented[0][1],
                    "end_lon": oriented[-1][0], "end_lat": oriented[-1][1],
                    "length_m": float(props.get("length", 0.0)),
                    "auto_routable": True,
                    "valhalla_access_mask": access,
                    "valhalla_road_class": ROAD_CLASS_NAMES.get(int(props.get("road_class", -1)), "unknown"),
                    "valhalla_road_class_code": int(props.get("road_class", -1)),
                    "valhalla_use_code": int(props.get("use", -1)),
                    "valhalla_speed_limit_kmh": float(props.get("speed_limit", 0.0)),
                    "bridge_valhalla": bool(props.get("bridge", False)),
                    "tunnel_valhalla": bool(props.get("tunnel", False)),
                    "roundabout_valhalla": bool(props.get("roundabout", False)),
                    "valhalla_layer": int(props.get("layer", 0)),
                    "motor_vehicle_routability_source": "frozen_valhalla_forwardaccess_auto_bit",
                }
                existing = edge_by_id.get(edge_id)
                if existing is None or candidate["geometry_point_count"] > existing["geometry_point_count"]:
                    if existing is not None:
                        duplicate_count += 1
                    edge_by_id[edge_id] = candidate
                else:
                    duplicate_count += 1

    edges = pd.DataFrame(sorted(edge_by_id.values(), key=lambda row: row["valhalla_directed_edge_id"]))
    nodes = pd.DataFrame(sorted(node_by_id.values(), key=lambda row: row["valhalla_node_id"]))
    if edges.empty or nodes.empty:
        raise Stage3S2AError("Valhalla full-network export is empty")
    edges, nodes = enrich_osm_and_topology(edges, nodes, osm)
    report = {
        "schema_version": "stage3_s2a_full_network_export.1",
        "status": "PASS",
        "source": "frozen Valhalla 3.8.2 MVT directed edges; auto bit in forwardaccess",
        "auto_access_bit": AUTO_ACCESS_BIT,
        "web_tile_zoom": z,
        "web_tile_count": len(_web_tile_range(osm.bounds, z)),
        "raw_feature_count": feature_count,
        "duplicate_clipped_direction_count": duplicate_count,
        "node_count": len(nodes),
        "directed_edge_count": len(edges),
        "unique_osm_way_count": int(edges["osm_way_id"].nunique()),
        "road_class_distribution": edges["valhalla_road_class"].value_counts().sort_index().to_dict(),
        "bridge_effective_count": int(edges["bridge_effective"].sum()),
        "tunnel_effective_count": int(edges["tunnel_effective"].sum()),
        "nonzero_layer_count": int((edges["osm_layer"].fillna(0) != 0).sum()),
        "identity_algorithm": config["network_export"]["stage3_edge_uid_algorithm"],
        "shortcuts_excluded": True,
        "all_highway_heuristic_used": False,
        "intersection_clustering_performed": False,
    }
    return edges, nodes, report


def load_or_export_network(
    valhalla_config: Path, osm: OsmData, config: Mapping[str, Any], output: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    edges_path = output / "full_network_export_cache_edges.parquet"
    nodes_path = output / "full_network_export_cache_nodes.parquet"
    report_path = output / "full_network_export_cache_report.json"
    binding_path = output / "full_network_export_cache_binding.json"
    source_hash = hashlib.sha256(
        (
            "stage3-s2a-full-network-cache-v3"
            + sha256_file(valhalla_config)
            + json.dumps(config["network_export"], sort_keys=True)
            + sha256_file(_osm_cache_paths(output)[0])
        ).encode()
    ).hexdigest()
    if all(path.is_file() for path in (edges_path, nodes_path, report_path, binding_path)):
        binding = read_json(binding_path)
        if (
            binding.get("source_set_sha256") == source_hash
            and binding.get("edges_sha256") == sha256_file(edges_path)
            and binding.get("nodes_sha256") == sha256_file(nodes_path)
        ):
            return pq.read_table(edges_path).to_pandas(), pq.read_table(nodes_path).to_pandas(), read_json(report_path)
    edges, nodes, report = export_valhalla_network(valhalla_config, osm, config)
    atomic_parquet(edges_path, edges)
    atomic_parquet(nodes_path, nodes)
    atomic_json(report_path, report)
    atomic_json(
        binding_path,
        {"source_set_sha256": source_hash, "edges_sha256": sha256_file(edges_path), "nodes_sha256": sha256_file(nodes_path)},
    )
    return edges, nodes, report


def _nearest_way_node(
    point: tuple[float, float], nodes: Sequence[tuple[int, float, float]], tolerance_m: float = 4.0
) -> int | None:
    if not nodes:
        return None
    candidates = nodes
    # A Valhalla edge endpoint is an OSM way node.  For long ways, a quick
    # coordinate shortlist avoids scanning every node for every split edge.
    if len(nodes) > 12:
        candidates = sorted(
            nodes,
            key=lambda item: abs(item[1] - point[0]) + abs(item[2] - point[1]),
        )[:4]
    best_id = None
    best_distance = math.inf
    for node_id, lon, lat in candidates:
        distance = haversine_m(point, (lon, lat))
        if distance < best_distance:
            best_id, best_distance = node_id, distance
    return int(best_id) if best_id is not None and best_distance <= tolerance_m else None


def enrich_osm_and_topology(
    edges: pd.DataFrame, nodes: pd.DataFrame, osm: OsmData
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mini_roundabout_nodes = {
        int(item["osm_node_id"])
        for item in osm.controls
        if item["control_evidence_type"] == "ROUNDABOUT" and item["osm_node_id"] is not None
    }
    rows: list[dict[str, Any]] = []
    for row in edges.to_dict("records"):
        way = osm.ways.get(int(row["osm_way_id"])) if pd.notna(row["osm_way_id"]) else None
        tags = way["tags"] if way else {}
        way_nodes = way["nodes"] if way else []
        begin = _nearest_way_node((row["start_lon"], row["start_lat"]), way_nodes)
        end = _nearest_way_node((row["end_lon"], row["end_lat"]), way_nodes)
        way_roundabout = tags.get("junction") == "roundabout"
        mini_roundabout_exposure = begin in mini_roundabout_nodes or end in mini_roundabout_nodes
        roundabout_sources = []
        if way_roundabout:
            roundabout_sources.append("OSM_JUNCTION_ROUNDABOUT_WAY")
        if mini_roundabout_exposure:
            roundabout_sources.append("OSM_HIGHWAY_MINI_ROUNDABOUT_NODE")
        bridge_osm = str(tags.get("bridge", "")).lower() not in {"", "no", "false", "0"}
        tunnel_osm = str(tags.get("tunnel", "")).lower() not in {"", "no", "false", "0"}
        try:
            osm_layer = int(float(tags.get("layer", 0)))
        except (TypeError, ValueError):
            osm_layer = None
        row.update(
            {
                "from_osm_node_id": begin,
                "to_osm_node_id": end,
                "osm_highway": tags.get("highway"),
                "osm_maxspeed_raw": tags.get("maxspeed"),
                "osm_layer": osm_layer,
                "bridge_osm": bridge_osm,
                "bridge_effective": bool(row["bridge_valhalla"] or bridge_osm),
                "bridge_conflict": bool(row["bridge_valhalla"] != bridge_osm),
                "tunnel_osm": tunnel_osm,
                "tunnel_effective": bool(row["tunnel_valhalla"] or tunnel_osm),
                "tunnel_conflict": bool(row["tunnel_valhalla"] != tunnel_osm),
                "junction_roundabout_way": way_roundabout,
                "mini_roundabout_node_exposure": mini_roundabout_exposure,
                "roundabout_evidence_source": "+".join(roundabout_sources) if roundabout_sources else None,
                "grade_separation_evidence": bool(bridge_osm or tunnel_osm or (osm_layer not in (None, 0))),
                "road_class_provenance": "Valhalla classification + raw OSM highway; not statutory equivalence",
            }
        )
        rows.append(row)
    edges = pd.DataFrame(rows)

    node_points: dict[tuple[int, int], list[int]] = defaultdict(list)
    for item in nodes.itertuples(index=False):
        node_points[(round(item.lon * 1e6), round(item.lat * 1e6))].append(int(item.valhalla_node_id))
    edge_endpoints: dict[int, list[str]] = defaultdict(list)
    for index, row in edges.iterrows():
        for prefix in ("start", "end"):
            key = (round(float(row[f"{prefix}_lon"]) * 1e6), round(float(row[f"{prefix}_lat"]) * 1e6))
            candidates = node_points.get(key, [])
            if candidates:
                selected = min(candidates)
                edges.at[index, "from_valhalla_node_id" if prefix == "start" else "to_valhalla_node_id"] = selected
                edge_endpoints[selected].append(row["stage3_edge_uid"])
    osm_by_graph: dict[int, set[int]] = defaultdict(set)
    for row in edges.itertuples(index=False):
        if pd.notna(getattr(row, "from_valhalla_node_id", None)) and pd.notna(row.from_osm_node_id):
            osm_by_graph[int(row.from_valhalla_node_id)].add(int(row.from_osm_node_id))
        if pd.notna(getattr(row, "to_valhalla_node_id", None)) and pd.notna(row.to_osm_node_id):
            osm_by_graph[int(row.to_valhalla_node_id)].add(int(row.to_osm_node_id))
    node_rows = []
    for row in nodes.to_dict("records"):
        node_id = int(row["valhalla_node_id"])
        osm_ids = sorted(osm_by_graph.get(node_id, []))
        incident = edge_endpoints.get(node_id, [])
        row.update(
            {
                "osm_node_id": osm_ids[0] if len(osm_ids) == 1 else None,
                "osm_node_mapping_status": "EXACT_EDGE_ENDPOINT" if len(osm_ids) == 1 else ("AMBIGUOUS" if osm_ids else "UNMAPPED"),
                "incident_auto_edge_count": len(incident),
                "graph_degree_undirected_proxy": len(set(incident)),
                "source_provenance": "frozen_valhalla_node_with_osm_endpoint_enrichment",
                "intersection_complex_id": None,
            }
        )
        node_rows.append(row)
    # The MVT nodes layer contains every graph node in the requested web tiles,
    # including nodes that are unrelated to an auto-access edge.  S2A's node
    # product is the motor-vehicle network foundation, so retain only nodes
    # that are endpoints of at least one exported auto-access edge.  Missing
    # endpoints remain explicit on edges when a web-tile-clipped geometry does
    # not coincide with a physical graph node; they are never nearest-guessed.
    nodes = pd.DataFrame(node_rows)
    nodes = nodes[nodes["incident_auto_edge_count"] > 0].reset_index(drop=True)
    node_uid = nodes.set_index("valhalla_node_id")["stage3_node_uid"].to_dict()
    edges["from_stage3_node_uid"] = edges["from_valhalla_node_id"].map(node_uid)
    edges["to_stage3_node_uid"] = edges["to_valhalla_node_id"].map(node_uid)
    return edges, nodes


OBSERVED_COLUMNS = [
    "canonical_edge_uid", "canonical_traversal_direction", "valhalla_edge_id",
    "osm_way_id", "begin_osm_node_id", "end_osm_node_id", "forward",
    "canonical_from_node", "canonical_to_node", "canonical_highway", "road_class",
    "speed_limit", "bridge", "tunnel", "length_m",
]


def read_observed_identities(input_root: Path) -> pd.DataFrame:
    files = sorted(input_root.glob("split=*/date=*/bucket=*/route_parts.parquet"))
    if not files:
        raise Stage3S2AError("no frozen route_parts")
    frames: list[pd.DataFrame] = []
    for path in files:
        table = pq.read_table(path, columns=OBSERVED_COLUMNS)
        frames.append(table.to_pandas().drop_duplicates())
    observed = pd.concat(frames, ignore_index=True).drop_duplicates()
    observed["valhalla_edge_id"] = pd.to_numeric(observed["valhalla_edge_id"], errors="coerce").astype("Int64")
    observed = observed.dropna(subset=["canonical_edge_uid", "canonical_traversal_direction"])
    static = [
        "canonical_edge_uid", "canonical_traversal_direction", "valhalla_edge_id", "osm_way_id",
        "begin_osm_node_id", "end_osm_node_id", "forward", "canonical_from_node",
        "canonical_to_node", "canonical_highway", "road_class", "speed_limit", "bridge", "tunnel",
    ]
    observed = observed.sort_values(static, na_position="last").drop_duplicates(
        ["canonical_edge_uid", "canonical_traversal_direction"], keep="first"
    )
    return observed.reset_index(drop=True)


def load_or_read_observed(input_root: Path, output: Path) -> pd.DataFrame:
    cache = output / "observed_identity_cache.parquet"
    binding_path = output / "observed_identity_cache_binding.json"
    files = sorted(input_root.glob("split=*/date=*/bucket=*/route_parts.parquet"))
    source_hash = hashlib.sha256(
        "\n".join(f"{path.relative_to(input_root).as_posix()}|{path.stat().st_size}" for path in files).encode()
    ).hexdigest()
    if cache.is_file() and binding_path.is_file():
        binding = read_json(binding_path)
        if binding.get("source_set_sha256") == source_hash and binding.get("cache_sha256") == sha256_file(cache):
            return pq.read_table(cache).to_pandas()
    observed = read_observed_identities(input_root)
    atomic_parquet(cache, observed)
    atomic_json(binding_path, {"source_set_sha256": source_hash, "cache_sha256": sha256_file(cache)})
    return observed


def map_observed_network(
    edges: pd.DataFrame, observed: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    by_valhalla: dict[int, list[int]] = defaultdict(list)
    by_endpoint: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, row in edges.iterrows():
        by_valhalla[int(row.valhalla_directed_edge_id)].append(index)
        if pd.notna(row.osm_way_id) and pd.notna(row.from_osm_node_id) and pd.notna(row.to_osm_node_id):
            by_endpoint[(int(row.osm_way_id), int(row.from_osm_node_id), int(row.to_osm_node_id))].append(index)

    mapping_rows: list[dict[str, Any]] = []
    full_members: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in observed.itertuples(index=False):
        candidates: list[int] = []
        method = ""
        valhalla_id = int(row.valhalla_edge_id) if pd.notna(row.valhalla_edge_id) else None
        if valhalla_id is not None:
            candidates = by_valhalla.get(valhalla_id, [])
            method = "EXACT_VALHALLA"
        if not candidates and all(pd.notna(value) for value in (row.osm_way_id, row.begin_osm_node_id, row.end_osm_node_id)):
            candidates = by_endpoint.get(
                (int(row.osm_way_id), int(row.begin_osm_node_id), int(row.end_osm_node_id)), []
            )
            method = "EXACT_OSM_ENDPOINT_DIRECTION"
        status = method if len(candidates) == 1 else ("AMBIGUOUS" if len(candidates) > 1 else "UNMAPPED_FULL_NETWORK_EDGE")
        confidence = "HIGH" if status in {"EXACT_VALHALLA", "EXACT_OSM_ENDPOINT_DIRECTION"} else "LOW"
        base = {
            "canonical_edge_uid": str(row.canonical_edge_uid),
            "canonical_traversal_direction": str(row.canonical_traversal_direction),
            "observed_valhalla_edge_id": valhalla_id,
            "observed_osm_way_id": int(row.osm_way_id) if pd.notna(row.osm_way_id) else None,
            "observed_begin_osm_node_id": int(row.begin_osm_node_id) if pd.notna(row.begin_osm_node_id) else None,
            "observed_end_osm_node_id": int(row.end_osm_node_id) if pd.notna(row.end_osm_node_id) else None,
            "mapping_status": status,
            "mapping_method": status,
            "mapping_confidence": confidence,
            "candidate_count": len(candidates),
        }
        if candidates:
            for candidate in candidates:
                edge = edges.loc[candidate]
                item = {
                    **base,
                    "stage3_edge_uid": edge.stage3_edge_uid,
                    "valhalla_directed_edge_id": int(edge.valhalla_directed_edge_id),
                }
                mapping_rows.append(item)
                full_members[candidate].append(item)
        else:
            mapping_rows.append({**base, "stage3_edge_uid": None, "valhalla_directed_edge_id": None})

    mapping = pd.DataFrame(mapping_rows)
    edges = edges.copy()
    edges["observed_network_member"] = False
    edges["canonical_edge_uid"] = None
    edges["canonical_traversal_direction"] = None
    edges["mapping_status"] = "UNMAPPED_FULL_NETWORK_EDGE"
    edges["mapping_method"] = "UNMAPPED_FULL_NETWORK_EDGE"
    edges["mapping_confidence"] = "NONE"
    edges["stage0_road_class"] = None
    edges["bridge_stage0"] = None
    edges["tunnel_stage0"] = None
    observed_lookup = observed.set_index(["canonical_edge_uid", "canonical_traversal_direction"])
    for index, members in full_members.items():
        edges.at[index, "observed_network_member"] = True
        identities = sorted({(m["canonical_edge_uid"], m["canonical_traversal_direction"]) for m in members})
        edges.at[index, "canonical_edge_uid"] = json.dumps([item[0] for item in identities])
        edges.at[index, "canonical_traversal_direction"] = json.dumps([item[1] for item in identities])
        edges.at[index, "mapping_status"] = "EXACT_VALHALLA" if all(m["mapping_status"] == "EXACT_VALHALLA" for m in members) else members[0]["mapping_status"]
        edges.at[index, "mapping_method"] = edges.at[index, "mapping_status"]
        edges.at[index, "mapping_confidence"] = "HIGH" if len(identities) == 1 else "MEDIUM"
        first = observed_lookup.loc[identities[0]]
        edges.at[index, "stage0_road_class"] = first.road_class
        edges.at[index, "bridge_stage0"] = bool(first.bridge)
        edges.at[index, "tunnel_stage0"] = bool(first.tunnel)
    mapped = mapping[mapping["stage3_edge_uid"].notna()]
    unique_directed = observed[["canonical_edge_uid", "canonical_traversal_direction"]].drop_duplicates()
    mapped_directed = mapped[["canonical_edge_uid", "canonical_traversal_direction"]].drop_duplicates()
    unique_segments = observed[["canonical_edge_uid"]].drop_duplicates()
    mapped_segments = mapped[["canonical_edge_uid"]].drop_duplicates()
    cardinality = mapped.groupby("stage3_edge_uid").size()
    observed_cardinality = mapping.groupby(
        ["canonical_edge_uid", "canonical_traversal_direction"], dropna=False
    )["stage3_edge_uid"].apply(lambda values: values.notna().sum())
    by_class = (
        mapping.merge(observed[["canonical_edge_uid", "canonical_traversal_direction", "road_class"]], on=["canonical_edge_uid", "canonical_traversal_direction"], how="left")
        .groupby("road_class", dropna=False)["stage3_edge_uid"]
        .agg(total="size", mapped=lambda series: series.notna().sum())
        .reset_index()
    )
    context = mapping.merge(
        observed[["canonical_edge_uid", "canonical_traversal_direction", "bridge", "tunnel"]],
        on=["canonical_edge_uid", "canonical_traversal_direction"],
        how="left",
    ).merge(edges[["stage3_edge_uid", "osm_layer"]], on="stage3_edge_uid", how="left")
    context["layer_context"] = context["osm_layer"].fillna(0).ne(0).map({True: "NONZERO", False: "ZERO_OR_UNKNOWN"})
    by_static_context = {
        field: (
            context.groupby(field, dropna=False)["stage3_edge_uid"]
            .agg(total="size", mapped=lambda values: values.notna().sum())
            .reset_index().to_dict("records")
        )
        for field in ("bridge", "tunnel", "layer_context")
    }
    report = {
        "observed_segment_total": len(unique_segments),
        "observed_segment_mapped": len(mapped_segments),
        "observed_segment_mapping_rate": len(mapped_segments) / len(unique_segments),
        "observed_directed_total": len(unique_directed),
        "observed_directed_mapped": len(mapped_directed),
        "observed_directed_mapping_rate": len(mapped_directed) / len(unique_directed),
        "mapping_status_counts": mapping["mapping_status"].value_counts().to_dict(),
        "one_to_one_full_edges": int((cardinality == 1).sum()),
        "many_to_one_full_edges": int((cardinality > 1).sum()),
        "one_to_one_observed_identities": int((observed_cardinality == 1).sum()),
        "one_to_many_observed_identities": int((observed_cardinality > 1).sum()),
        "ambiguous_observed_identities": int((mapping["mapping_status"] == "AMBIGUOUS").sum()),
        "unmapped_observed_identities": int((mapping["mapping_status"] == "UNMAPPED_FULL_NETWORK_EDGE").sum()),
        "unmapped_by_traversal_direction": (
            mapping.loc[mapping["mapping_status"] == "UNMAPPED_FULL_NETWORK_EDGE", "canonical_traversal_direction"]
            .value_counts().sort_index().to_dict()
        ),
        "unmapped_reverse_direction_count": int(
            (
                (mapping["mapping_status"] == "UNMAPPED_FULL_NETWORK_EDGE")
                & (mapping["canonical_traversal_direction"] == "R")
            ).sum()
        ),
        "by_road_class": by_class.to_dict("records"),
        "by_bridge_tunnel_layer_context": by_static_context,
        "geometry_only_mapping_used": False,
        "direction_preserved": True,
    }
    return edges, mapping, report


def build_control_evidence(osm: OsmData, edges: pd.DataFrame) -> pd.DataFrame:
    way_exposure = edges.groupby("osm_way_id")["stage3_edge_uid"].agg(list).to_dict()
    node_exposure: dict[int, list[str]] = defaultdict(list)
    for row in edges.itertuples(index=False):
        if pd.notna(row.from_osm_node_id):
            node_exposure[int(row.from_osm_node_id)].append(row.stage3_edge_uid)
        if pd.notna(row.to_osm_node_id):
            node_exposure[int(row.to_osm_node_id)].append(row.stage3_edge_uid)
    rows = []
    for evidence in osm.controls:
        exposed = (
            node_exposure.get(int(evidence["osm_node_id"]), [])
            if evidence["osm_node_id"] is not None
            else way_exposure.get(int(evidence["osm_way_id"]), [])
        )
        rows.append(
            {
                **evidence,
                "mapped_full_network_edge_count": len(set(exposed)),
                "mapped_stage3_edge_uids": json.dumps(sorted(set(exposed))),
                "missing_tag_means_negative_control": False,
                "poi_used": False,
            }
        )
    return pd.DataFrame(rows)


def build_restrictions(osm: OsmData, edges: pd.DataFrame) -> pd.DataFrame:
    way_to_edges = edges.groupby("osm_way_id")["stage3_edge_uid"].agg(lambda x: sorted(set(x))).to_dict()
    rows = []
    for relation in osm.restrictions:
        from_members, via_members, to_members = relation["from_members"], relation["via_members"], relation["to_members"]
        role_complete = len(from_members) == 1 and len(to_members) == 1 and len(via_members) >= 1
        from_way = from_members[0]["ref"] if role_complete and from_members[0]["type"] == "w" else None
        to_way = to_members[0]["ref"] if role_complete and to_members[0]["type"] == "w" else None
        from_edges = way_to_edges.get(from_way, []) if from_way is not None else []
        to_edges = way_to_edges.get(to_way, []) if to_way is not None else []
        certified = False
        status = "ROLE_PRESERVED_UNCERTAIN_DIRECTED_MAPPING"
        if not role_complete:
            status = "INVALID_OR_INCOMPLETE_ROLES"
        elif from_edges and to_edges:
            status = "MEMBER_WAYS_MAPPED_DIRECTED_ENFORCEMENT_UNCERTAIN"
        else:
            status = "MEMBER_WAY_UNMAPPED"
        rows.append(
            {
                "restriction_id": relation["restriction_id"],
                "relation_type": relation["relation_type"],
                "restriction_type": relation["restriction_type"],
                "members": json.dumps(relation["members"], sort_keys=True),
                "from_member": json.dumps(from_members, sort_keys=True),
                "via_members": json.dumps(via_members, sort_keys=True),
                "to_member": json.dumps(to_members, sort_keys=True),
                "from_stage3_edge_uids": json.dumps(from_edges),
                "to_stage3_edge_uids": json.dumps(to_edges),
                "mapping_status": status,
                "directed_enforcement_certified": certified,
                "geometry_guessing_used": False,
                "reader": "pyosmium_role_ref_preserving",
            }
        )
    return pd.DataFrame(rows)


def anchor_table(observed: pd.DataFrame) -> pd.DataFrame:
    anchors = observed[pd.to_numeric(observed["speed_limit"], errors="coerce") > 0].copy()
    anchors["known_speed_kmh"] = pd.to_numeric(anchors["speed_limit"], errors="coerce")
    anchors["anchor_group"] = anchors["osm_way_id"].astype("Int64").astype(str)
    # S1's frozen 502-unit anchor population is canonical-segment weighted,
    # not duplicated when the same physical segment was traversed both ways.
    anchors = anchors.sort_values(["canonical_edge_uid", "canonical_traversal_direction"]).drop_duplicates(
        ["canonical_edge_uid"]
    )
    return anchors


def read_historical_speeds(
    input_root: Path, train_dates: Sequence[str], quantiles: Sequence[float]
) -> pd.DataFrame:
    frames = []
    seen_dates: set[str] = set()
    columns = [
        "canonical_edge_uid", "observed_speed_mps", "observed_distance_m",
        "observed_travel_time_s", "measurement_source", "time_observation_valid",
    ]
    for date in train_dates:
        files = sorted((input_root / "split=train" / f"date={date}").glob("bucket=*/link_traversals.parquet"))
        if not files:
            raise Stage3S2AError(f"missing Train speed files for {date}")
        seen_dates.add(date)
        for path in files:
            frame = pq.read_table(path, columns=columns).to_pandas()
            mask = (
                (frame["measurement_source"] == "direct_observed")
                & frame["time_observation_valid"].fillna(False)
                & (pd.to_numeric(frame["observed_speed_mps"], errors="coerce") > 0)
                & (pd.to_numeric(frame["observed_speed_mps"], errors="coerce") <= 55.0)
                & (pd.to_numeric(frame["observed_distance_m"], errors="coerce") > 0)
                & (pd.to_numeric(frame["observed_travel_time_s"], errors="coerce") > 0)
            )
            valid = frame.loc[mask, ["canonical_edge_uid", "observed_speed_mps"]].copy()
            valid["speed_kmh"] = pd.to_numeric(valid["observed_speed_mps"]) * 3.6
            frames.append(valid[["canonical_edge_uid", "speed_kmh"]])
    if seen_dates != set(train_dates) or "20161031" in seen_dates:
        raise Stage3S2AError("historical speed source escaped Train-only dates")
    history = pd.concat(frames, ignore_index=True)
    grouped = history.groupby("canonical_edge_uid")["speed_kmh"]
    result = grouped.size().rename("historical_speed_support_n").to_frame()
    for q in quantiles:
        result[f"v_ff_q{int(q * 100)}_kmh"] = grouped.quantile(q)
    return result.reset_index()


def load_or_read_history(
    input_root: Path, output: Path, train_dates: Sequence[str], quantiles: Sequence[float]
) -> pd.DataFrame:
    cache = output / "train_speed_quantile_cache.parquet"
    binding_path = output / "train_speed_quantile_cache_binding.json"
    files = [
        path
        for date in train_dates
        for path in sorted((input_root / "split=train" / f"date={date}").glob("bucket=*/link_traversals.parquet"))
    ]
    source_hash = hashlib.sha256(
        ("|".join(map(str, quantiles)) + "\n" + "\n".join(f"{path.relative_to(input_root).as_posix()}|{path.stat().st_size}" for path in files)).encode()
    ).hexdigest()
    if cache.is_file() and binding_path.is_file():
        binding = read_json(binding_path)
        if binding.get("source_set_sha256") == source_hash and binding.get("cache_sha256") == sha256_file(cache):
            return pq.read_table(cache).to_pandas()
    history = read_historical_speeds(input_root, train_dates, quantiles)
    atomic_parquet(cache, history)
    atomic_json(binding_path, {"source_set_sha256": source_hash, "cache_sha256": sha256_file(cache)})
    return history


def speed_grid(anchors: pd.DataFrame, policy: Mapping[str, Any]) -> np.ndarray:
    regular = np.arange(
        int(policy["minimum_kmh"]),
        int(policy["maximum_kmh"]) + int(policy["increment_kmh"]),
        int(policy["increment_kmh"]),
        dtype=float,
    )
    known = anchors["known_speed_kmh"].dropna().to_numpy(float)
    grid = np.unique(np.concatenate([regular, known]))
    return grid[(grid >= policy["minimum_kmh"]) & (grid <= policy["maximum_kmh"])]


def nearest_class(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    distances = np.abs(values[:, None] - grid[None, :])
    return grid[np.argmin(distances, axis=1)]


def _mode(values: Sequence[float], fallback: float) -> float:
    counter = Counter(float(value) for value in values if np.isfinite(value))
    if not counter:
        return float(fallback)
    highest = max(counter.values())
    return min(value for value, count in counter.items() if count == highest)


def _class_modes(train: pd.DataFrame, minimum_support: int) -> tuple[dict[str, float], float]:
    global_mode = _mode(train["known_speed_kmh"], 60.0)
    modes = {}
    for road_class, group in train.groupby("road_class"):
        modes[str(road_class)] = _mode(group["known_speed_kmh"], global_mode) if len(group) >= minimum_support else global_mode
    return modes, global_mode


def predict_speed(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    method: str,
    quantile: float,
    grid: np.ndarray,
    minimum_class_support: int,
) -> np.ndarray:
    qcol = f"v_ff_q{int(quantile * 100)}_kmh"
    modes, global_mode = _class_modes(train, minimum_class_support)
    prior = np.array([modes.get(str(value), global_mode) for value in test["road_class"]], dtype=float)
    if method == "ROAD_CLASS_MODE_ONLY":
        return prior
    vff = pd.to_numeric(test[qcol], errors="coerce").to_numpy(float)
    baseline = nearest_class(np.nan_to_num(vff / 1.10, nan=global_mode), grid)
    if method == "V_FF_DIV_1P10_NEAREST_CLASS":
        return baseline
    if method != "MAP_SPEED_AND_ROAD_CLASS":
        raise Stage3S2AError(f"unknown speed method {method}")
    train_ratio = pd.to_numeric(train[qcol], errors="coerce").to_numpy(float) / train["known_speed_kmh"].to_numpy(float)
    valid_ratio = train_ratio[np.isfinite(train_ratio) & (train_ratio > 0)]
    if len(valid_ratio) < 10:
        return baseline
    log_ratio = np.log(valid_ratio)
    center = float(np.median(log_ratio))
    scale = max(float(np.median(np.abs(log_ratio - center)) * 1.4826), 0.08)
    class_counts: dict[str, Counter[float]] = defaultdict(Counter)
    global_counts = Counter(train["known_speed_kmh"].astype(float))
    for row in train.itertuples(index=False):
        class_counts[str(row.road_class)][float(row.known_speed_kmh)] += 1
    predictions = []
    for value, road_class in zip(vff, test["road_class"], strict=False):
        counts = class_counts.get(str(road_class), Counter())
        total = sum(counts.values())
        if total < minimum_class_support:
            counts, total = global_counts, sum(global_counts.values())
        scores = []
        for candidate in grid:
            prior_prob = (counts.get(float(candidate), 0) + 1.0) / (total + len(grid))
            if np.isfinite(value) and value > 0:
                z = (math.log(value / candidate) - center) / scale
                likelihood = math.exp(-0.5 * z * z) / scale
            else:
                likelihood = 1.0
            scores.append(math.log(prior_prob) + math.log(max(likelihood, 1e-15)))
        predictions.append(float(grid[int(np.argmax(scores))]))
    return np.asarray(predictions)


def metric_row(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    compatibility = {
        str(cap): float(np.mean((truth <= cap) == (prediction <= cap))) for cap in (60, 80, 120)
    }
    return {
        "compatibility_accuracy_60": compatibility["60"],
        "compatibility_accuracy_80": compatibility["80"],
        "compatibility_accuracy_120": compatibility["120"],
        "macro_scenario_compatibility_accuracy": float(np.mean(list(compatibility.values()))),
        "within_10_kmh_accuracy": float(np.mean(np.abs(truth - prediction) <= 10)),
        "exact_class_accuracy": float(np.mean(truth == prediction)),
        "mae_kmh": float(np.mean(np.abs(truth - prediction))),
    }


def grouped_folds(groups: Sequence[str], folds: int, seed: int) -> np.ndarray:
    unique = sorted(set(map(str, groups)))
    assignment = {
        group: int(hashlib.sha256(f"{seed}|{group}".encode()).hexdigest(), 16) % folds
        for group in unique
    }
    return np.asarray([assignment[str(group)] for group in groups], dtype=int)


def validate_speed_models(
    anchors: pd.DataFrame, history: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    speed_cfg = config["speed_inference"]
    table = anchors.merge(history, on="canonical_edge_uid", how="left")
    grid = speed_grid(table, speed_cfg["speed_class_grid_policy"])
    table["road_class"] = table["road_class"].fillna("unknown").astype(str)
    fold_ids = grouped_folds(table["anchor_group"], int(speed_cfg["grouped_cv_folds"]), int(speed_cfg["seed"]))
    table["cv_fold"] = fold_ids
    rows = []
    fold_rows = []
    complexity = {
        "ROAD_CLASS_MODE_ONLY": 0,
        "V_FF_DIV_1P10_NEAREST_CLASS": 1,
        "MAP_SPEED_AND_ROAD_CLASS": 2,
    }
    for quantile in speed_cfg["candidate_quantiles"]:
        for method in speed_cfg["candidate_methods"]:
            truths, predictions = [], []
            for fold in range(int(speed_cfg["grouped_cv_folds"])):
                train = table[table["cv_fold"] != fold]
                test = table[table["cv_fold"] == fold]
                if test.empty:
                    continue
                pred = predict_speed(
                    train, test, method=method, quantile=float(quantile), grid=grid,
                    minimum_class_support=int(speed_cfg["minimum_class_anchor_support_n"]),
                )
                truth = test["known_speed_kmh"].to_numpy(float)
                metrics = metric_row(truth, pred)
                fold_rows.append({"quantile": quantile, "method": method, "fold": fold, "anchor_count": len(test), **metrics})
                truths.append(truth)
                predictions.append(pred)
            aggregate = metric_row(np.concatenate(truths), np.concatenate(predictions))
            dispersion = pd.DataFrame([row for row in fold_rows if row["quantile"] == quantile and row["method"] == method])
            rows.append(
                {
                    "quantile": quantile, "method": method, "model_complexity": complexity[method],
                    **aggregate,
                    "fold_macro_std": float(dispersion["macro_scenario_compatibility_accuracy"].std(ddof=0)),
                }
            )
    metrics = pd.DataFrame(rows)
    selected = metrics.sort_values(
        ["macro_scenario_compatibility_accuracy", "within_10_kmh_accuracy", "exact_class_accuracy", "model_complexity", "quantile"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).iloc[0].to_dict()
    anchor_binding_columns = ["canonical_edge_uid", "osm_way_id", "known_speed_kmh"]
    anchor_csv = table[anchor_binding_columns].sort_values(anchor_binding_columns).to_csv(index=False, lineterminator="\n")
    validation = {
        "schema_version": "stage3_s2a_speed_validation.1",
        "status": "PASS",
        "selection_source": "known speed anchors only",
        "train_history_dates": config["train_only_dates"],
        "test31_used": False,
        "anchor_count": len(table),
        "anchor_group_count": int(table["anchor_group"].nunique()),
        "anchor_set_sha256": hashlib.sha256(anchor_csv.encode()).hexdigest(),
        "grouping_key": speed_cfg["grouping_key"],
        "duplicate_group_leakage_count": 0,
        "speed_grid_kmh": grid.tolist(),
        "candidate_results": metrics.to_dict("records"),
        "fold_results": fold_rows,
        "selected": selected,
        "selection_order": speed_cfg["selection_metrics"],
        "inferred_values_are_verified_posted_limits": False,
        "ten_percent_is_legal_permission": False,
    }
    return validation, selected, table


def build_speed_domain(
    edges: pd.DataFrame,
    mapping: pd.DataFrame,
    observed: pd.DataFrame,
    history: pd.DataFrame,
    validation: Mapping[str, Any],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    selected = validation["selected"]
    speed_cfg = config["speed_inference"]
    anchors = anchor_table(observed)
    grid = speed_grid(anchors, speed_cfg["speed_class_grid_policy"])
    history_map = history.set_index("canonical_edge_uid").to_dict("index")
    observed_map = observed.set_index(["canonical_edge_uid", "canonical_traversal_direction"]).to_dict("index")
    mapped_by_edge = mapping[mapping["stage3_edge_uid"].notna()].groupby("stage3_edge_uid")
    inference_train = anchors.merge(history, on="canonical_edge_uid", how="left")
    inference_train["road_class"] = inference_train["road_class"].fillna("unknown").astype(str)
    modes, global_mode = _class_modes(inference_train, int(speed_cfg["minimum_class_anchor_support_n"]))
    qcol = f"v_ff_q{int(float(selected['quantile']) * 100)}_kmh"
    rows = []
    for edge in edges.itertuples(index=False):
        mapped = mapped_by_edge.get_group(edge.stage3_edge_uid) if edge.stage3_edge_uid in mapped_by_edge.groups else pd.DataFrame()
        identities = [
            (str(row.canonical_edge_uid), str(row.canonical_traversal_direction))
            for row in mapped.itertuples(index=False)
        ]
        known = []
        histories = []
        for identity in identities:
            observed_row = observed_map.get(identity)
            if observed_row and pd.notna(observed_row.get("speed_limit")) and float(observed_row["speed_limit"]) > 0:
                known.append(float(observed_row["speed_limit"]))
            history_row = history_map.get(identity[0])
            if history_row:
                histories.append(history_row)
        road_class = edge.stage0_road_class if pd.notna(edge.stage0_road_class) else edge.valhalla_road_class
        prior = float(modes.get(str(road_class), global_mode))
        support = int(max([item["historical_speed_support_n"] for item in histories], default=0))
        vff = float(max([item[qcol] for item in histories if pd.notna(item.get(qcol))], default=np.nan))
        posted = bool(known)
        if known:
            value = _mode(known, known[0])
            provenance, confidence, method = "KNOWN_STAGE0_OSM", "HIGH", "known_frozen_stage0_speed_anchor"
        elif support >= int(speed_cfg["minimum_historical_support_n"]) and np.isfinite(vff):
            test = pd.DataFrame([{qcol: vff, "road_class": str(road_class)}])
            value = float(predict_speed(
                inference_train, test, method=str(selected["method"]), quantile=float(selected["quantile"]),
                grid=grid, minimum_class_support=int(speed_cfg["minimum_class_anchor_support_n"]),
            )[0])
            provenance, confidence, method = "INFERRED_SPEED_AND_CLASS", "MEDIUM", str(selected["method"])
        elif support > 0 and np.isfinite(vff):
            value = prior
            provenance, confidence, method = "INFERRED_CLASS_DOMINANT", "LOW", "class_mode_sparse_history"
        elif road_class != "unknown":
            value = prior
            provenance, confidence, method = "ROAD_CLASS_PRIOR_ONLY", "LOW", "hierarchical_road_class_mode"
        else:
            value = np.nan
            provenance, confidence, method = "UNKNOWN", "LOW", "no_usable_evidence"
        rows.append(
            {
                "stage3_edge_uid": edge.stage3_edge_uid,
                "valhalla_directed_edge_id": int(edge.valhalla_directed_edge_id),
                "observed_network_member": bool(edge.observed_network_member),
                "speed_domain_value_kmh": value,
                "speed_domain_provenance": provenance,
                "speed_domain_confidence": confidence,
                "posted_speed_observed": posted,
                "historical_freeflow_speed_kmh": vff if np.isfinite(vff) else None,
                "historical_speed_support_n": support,
                "road_class_prior_kmh": prior,
                "road_class_context": road_class,
                "speed_inference_method": method,
                "selected_freeflow_quantile": float(selected["quantile"]),
                "speed_compatible_C": bool(value <= 60) if np.isfinite(value) else None,
                "speed_compatible_M": bool(value <= 80) if np.isfinite(value) else None,
                "speed_compatible_A": bool(value <= 120) if np.isfinite(value) else None,
                "is_verified_posted_speed_limit": False,
            }
        )
    result = pd.DataFrame(rows)
    if not set(result["speed_domain_provenance"]).issubset(PROVENANCE):
        raise Stage3S2AError("invalid speed provenance")
    return result


def _distribution(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).sort_index().items()}


def build_reports(
    docs: Path,
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    mapping_report: Mapping[str, Any],
    controls: pd.DataFrame,
    restrictions: pd.DataFrame,
    speed: pd.DataFrame,
    validation: Mapping[str, Any],
) -> None:
    selected = validation["selected"]
    # Bind reconciliation to S1's canonical-segment population; full-network
    # directed edge flags below are a different counting unit.
    bridge_reconciliation = {
        "s1_observed_canonical_segment_total": 18992,
        "s1_stage0_bridge_true_segments": 418,
        "s1_osm_bridge_true_segments": 687,
        "s1_osm_true_stage0_false_segments": 269,
        "mapped_observed_segment_count_in_s2a": int(mapping_report["observed_segment_mapped"]),
        "full_network_directed_bridge_effective_count": int(edges["bridge_effective"].sum()),
        "interpretation": "S1 discrepancy is preserved exactly in its canonical-segment unit. S2A effective bridge enriches the full directed network while retaining raw Stage0/Valhalla/OSM fields and conflicts.",
    }
    network_report = f"""# Stage 3 S2A Network Foundation Report

Phase status: `{PHASE_STATUS}`. S2A exports a Stage3-owned complete directed auto-access graph; it does not build intersection complexes.

## Full network

- Nodes: `{len(nodes):,}`
- Directed auto-routable edges: `{len(edges):,}`
- Unique OSM ways: `{edges['osm_way_id'].nunique():,}`
- Authority: frozen Valhalla 3.8.2 directed edges with `forwardaccess & kAutoAccess`.
- Identity: `stage3_edge_uid = s3e_ + sha256('valhalla-3.8.2|' + uint64 GraphId)[:24]`.
- Road-class distribution: `{json.dumps(_distribution(edges, 'valhalla_road_class'), sort_keys=True)}`
- Effective bridge / tunnel edges: `{int(edges['bridge_effective'].sum()):,}` / `{int(edges['tunnel_effective'].sum()):,}`
- Non-zero OSM layer edges: `{int((edges['osm_layer'].fillna(0) != 0).sum()):,}`

## Observed mapping

- Observed segments: `{mapping_report['observed_segment_mapped']:,}` / `{mapping_report['observed_segment_total']:,}` ({mapping_report['observed_segment_mapping_rate']:.3%})
- Observed directed identities: `{mapping_report['observed_directed_mapped']:,}` / `{mapping_report['observed_directed_total']:,}` ({mapping_report['observed_directed_mapping_rate']:.3%})
- Ambiguous / unmatched: `{mapping_report['ambiguous_observed_identities']:,}` / `{mapping_report['unmapped_observed_identities']:,}`
- Unmatched by observed traversal direction: `{json.dumps(mapping_report['unmapped_by_traversal_direction'], sort_keys=True)}`. Reverse identities are not silently projected onto an auto-forbidden direction.
- Geometry-only remapping: `false`.

## Static provenance

- S1 bridge discrepancy reconciliation: `{json.dumps(bridge_reconciliation, ensure_ascii=False)}`
- Raw Stage0, Valhalla, and OSM bridge/tunnel fields remain separate; effective values are OR-enrichment with explicit conflicts.
- OSM highway names are context, not claims of Chinese statutory functional class.

## Controls and restrictions

- Positive control evidence: `{len(controls):,}` rows; `{json.dumps(_distribution(controls, 'control_evidence_type'), sort_keys=True)}`
- Traffic signals mapped to at least one full-network edge: `{int(((controls['control_evidence_type'] == 'SIGNALIZED') & (controls['mapped_full_network_edge_count'] > 0)).sum()):,}`
- Restriction relations: `{len(restrictions):,}`; role-preserving parse: `PASS`.
- Directed enforcement certified: `{int(restrictions['directed_enforcement_certified'].sum()):,}`. Uncertified restrictions remain uncertain; no geometry-guessed enforcement is fabricated.
- Missing OSM control tags are not negative evidence. POI was not used.

## Scope and limitations

- No intersection clustering or tolerance selection occurred.
- No Stage2 training/inference, profile calibration, Test31 fitting, fallback routing, or Stage4 execution occurred.
- MVT geometries are an export representation of the frozen routing graph. Topology identity remains Valhalla GraphId; OSM endpoint ids are best-effort exact endpoint enrichment and never override GraphId.
"""
    atomic_text(docs / "stage3_s2a_network_foundation_report.md", network_report)

    provenance_counts = _distribution(speed, "speed_domain_provenance")
    confidence_counts = _distribution(speed, "speed_domain_confidence")
    anchor_distribution = _distribution(
        speed[speed["speed_domain_provenance"] == "KNOWN_STAGE0_OSM"], "speed_domain_value_kmh"
    )
    history_edge_count = int((speed["historical_speed_support_n"] > 0).sum())
    history_observation_count = int(speed["historical_speed_support_n"].sum())
    observed_speed = speed[speed["observed_network_member"]]
    unobserved = speed[~speed["observed_network_member"]]
    comparison = [
        {
            "q": row["quantile"],
            "method": row["method"],
            "macro": row["macro_scenario_compatibility_accuracy"],
            "within10": row["within_10_kmh_accuracy"],
            "exact": row["exact_class_accuracy"],
            "mae_kmh": row["mae_kmh"],
        }
        for row in validation["candidate_results"]
    ]
    confidence_compatibility = {
        confidence: {
            cap: float(group[f"speed_compatible_{cap}"].dropna().mean())
            for cap in ("C", "M", "A")
        }
        for confidence, group in speed.groupby("speed_domain_confidence")
    }
    speed_report = f"""# Stage 3 S2A Operational Speed-Domain Report

The output is an operational speed-domain proxy. Inferred classes are **not verified posted speed limits**.

## Frozen validation choice

- Exact known-anchor directed identities: `{validation['anchor_count']:,}` in `{validation['anchor_group_count']:,}` physical OSM-way groups
- Anchor set SHA-256: `{validation['anchor_set_sha256']}`
- Train-only history: `20161009–20161024`; Test31 used: `false`
- Selected quantile: `{selected['quantile']}`
- Selected method: `{selected['method']}`
- Compatibility accuracy at 60 / 80 / 120: `{selected['compatibility_accuracy_60']:.6f}` / `{selected['compatibility_accuracy_80']:.6f}` / `{selected['compatibility_accuracy_120']:.6f}`
- Macro / within-10 / exact / MAE: `{selected['macro_scenario_compatibility_accuracy']:.6f}` / `{selected['within_10_kmh_accuracy']:.6f}` / `{selected['exact_class_accuracy']:.6f}` / `{selected['mae_kmh']:.3f}` km/h
- Candidate methods: B0 road-class mode, B1 `v_ff / 1.10` nearest class, B2 simple MAP of robust speed ratio × empirical class prior.
- Known-anchor speed-class distribution: `{json.dumps(anchor_distribution, sort_keys=True)}`
- Train historical support: `{history_edge_count:,}` full-network edges linked to `{history_observation_count:,}` valid direct observations.
- Frozen P85/P90/P95 × B0/B1/B2 comparison: `{json.dumps(comparison, sort_keys=True)}`

## Full-network coverage

- Provenance: `{json.dumps(provenance_counts, sort_keys=True)}`
- Confidence: `{json.dumps(confidence_counts, sort_keys=True)}`
- Unobserved edges: `{len(unobserved):,}`; non-UNKNOWN domain coverage: `{float((unobserved['speed_domain_provenance'] != 'UNKNOWN').mean()):.3%}`
- Observed-edge non-UNKNOWN coverage: `{float((observed_speed['speed_domain_provenance'] != 'UNKNOWN').mean()):.3%}`
- Full-network road-class distribution: `{json.dumps(_distribution(edges, 'valhalla_road_class'), sort_keys=True)}`
- C/M/A compatible shares among known-domain edges: `{float(speed['speed_compatible_C'].dropna().mean()):.3%}` / `{float(speed['speed_compatible_M'].dropna().mean()):.3%}` / `{float(speed['speed_compatible_A'].dropna().mean()):.3%}`
- C/M/A compatibility by confidence: `{json.dumps(confidence_compatibility, sort_keys=True)}`

## Limitations

- The 10% divisor is an empirical behavioral/regulatory inference anchor, not a statement of legal permission.
- Road-class priors are contextual empirical priors; OSM class is not Chinese statutory class.
- Sparse and unobserved edges receive low-confidence class priors, never fabricated historical speeds.
- No route feasibility, route-level F/U/I propagation, intersection complex, or profile envelope was constructed.
"""
    atomic_text(docs / "stage3_s2a_speed_domain_report.md", speed_report)

    provenance = """# Stage 3 S2A Field Provenance

| Product / field | Authority | S2A rule |
|---|---|---|
| directed topology, edge id, auto_routable | Frozen Valhalla 3.8.2 tiles | GraphId identity; include direction iff `forwardaccess & 1` |
| geometry, length, Valhalla class/use/layer | Frozen Valhalla MVT | Non-shortcut edge layer, zoom filtering disabled for all classes |
| OSM way/node, maxspeed, highway, bridge/tunnel/layer/roundabout | Frozen OSM PBF | pyosmium exact ids/tags; raw and effective fields separate |
| observed canonical identity and Stage0 static fields | Frozen Stage1 route_parts | Exact Valhalla id first, exact OSM endpoint direction second; no nearest-only mapping |
| control evidence | Frozen OSM PBF | Positive signal/stop/yield/roundabout evidence only; missing tag is unknown |
| turn restrictions | Frozen OSM PBF relations | pyosmium preserves member type/ref/role; uncertified direction stays uncertain |
| historical free-flow proxy | Frozen Stage1 link_traversals, Train 09–24 | direct_observed, valid time, physical positive bounds; edge-level P85/P90/P95 |
| speed-domain anchor | Stage0 positive speed_limit | Exact anchor set regenerated and SHA-bound |
| inferred speed domain | Frozen validation choice | Discrete grid, simple B0/B1/B2 selection on grouped anchor CV only |

POI is excluded. Inferred speed-domain values are not posted-speed observations. Intersection complexes are absent by design.
"""
    atomic_text(docs / "stage3_s2a_field_provenance.md", provenance)


def verify_products(
    config: Mapping[str, Any],
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    mapping: pd.DataFrame,
    controls: pd.DataFrame,
    restrictions: pd.DataFrame,
    speed: pd.DataFrame,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    failures = []
    guards = config["scope_guards"]
    if any(guards.values()) or config["authorizations"]["s2b"] or config["next_phase_authorized"]:
        failures.append("scope guard")
    if len(edges) <= int(mapping["canonical_edge_uid"].nunique()):
        failures.append("full network is not larger than observed subnetwork")
    if not edges["auto_routable"].all() or set(edges["motor_vehicle_routability_source"]) != {"frozen_valhalla_forwardaccess_auto_bit"}:
        failures.append("routability source")
    if edges["stage3_edge_uid"].duplicated().any() or any(
        stage3_edge_uid(value) != uid
        for value, uid in zip(edges["valhalla_directed_edge_id"], edges["stage3_edge_uid"], strict=False)
    ):
        failures.append("non-deterministic edge identity")
    if not set(mapping["mapping_status"]).issubset(VALID_MAPPING):
        failures.append("invalid mapping class")
    if controls["poi_used"].any() or controls["missing_tag_means_negative_control"].any():
        failures.append("invalid control inference")
    if restrictions["geometry_guessing_used"].any() or set(restrictions["reader"]) != {"pyosmium_role_ref_preserving"}:
        failures.append("restriction roles")
    if validation["test31_used"] or validation["duplicate_group_leakage_count"]:
        failures.append("speed validation leakage")
    if not set(speed["speed_domain_provenance"]).issubset(PROVENANCE) or speed["is_verified_posted_speed_limit"].any():
        failures.append("speed provenance")
    if config["speed_inference"]["av_caps_kmh"] != [60, 80, 120]:
        failures.append("AV caps")
    return {
        "schema_version": "stage3_s2a_verification.1",
        "status": "PASS" if not failures else "FAIL",
        "phase_status": PHASE_STATUS if not failures else "BLOCKED",
        "failures": failures,
        "s2a_complete": not failures,
        "s2b_authorized": False,
        "next_phase_authorized": False,
        "counts": {
            "nodes": len(nodes), "directed_edges": len(edges), "mapping_rows": len(mapping),
            "control_rows": len(controls), "restriction_rows": len(restrictions), "speed_rows": len(speed),
        },
        "scope_guards": guards,
    }


def run(config_path: Path, root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config(config_path, root)
    if git_head(root) != config["execution_base_commit"] and not git_head(root).startswith(config["execution_base_commit"]):
        raise Stage3S2AError("execution HEAD is not authorized S2A base")
    paths = {key: resolve(root, value) for key, value in config["paths"].items()}
    source_keys = [
        "stage0_freeze_manifest", "stage1_release_manifest", "stage2_final_release_manifest",
        "s1_inventory", "s1_provenance", "pbf", "valhalla_config", "valhalla_build_manifest",
    ]
    sources = {key: source_descriptor(paths[key], root) for key in source_keys}
    if sources["pbf"]["sha256"] != config["bindings"]["pbf_sha256"]:
        raise Stage3S2AError("frozen PBF hash changed")

    osm = load_or_read_osm(paths["pbf"], paths["output"])
    edges, nodes, export_report = load_or_export_network(paths["valhalla_config"], osm, config, paths["output"])
    observed = load_or_read_observed(paths["stage1_input"], paths["output"])
    edges, mapping, mapping_report = map_observed_network(edges, observed)
    controls = build_control_evidence(osm, edges)
    restrictions = build_restrictions(osm, edges)
    history = load_or_read_history(
        paths["stage1_input"], paths["output"], config["train_only_dates"], config["speed_inference"]["candidate_quantiles"]
    )
    validation, _, _ = validate_speed_models(anchor_table(observed), history, config)
    speed = build_speed_domain(edges, mapping, observed, history, validation, config)
    verification = verify_products(config, edges, nodes, mapping, controls, restrictions, speed, validation)
    if verification["status"] != "PASS":
        raise Stage3S2AError(f"S2A verification failed: {verification['failures']}")

    output = paths["output"]
    docs = paths["docs"]
    products = {
        "full_network_edges": output / "stage3_full_network_edges.parquet",
        "full_network_nodes": output / "stage3_full_network_nodes.parquet",
        "observed_mapping": output / "stage3_observed_full_network_mapping.parquet",
        "control_evidence": output / "stage3_control_evidence.parquet",
        "turn_restrictions": output / "stage3_turn_restrictions.parquet",
        "speed_domain": output / "stage3_speed_domain.parquet",
    }
    for path, frame in zip(products.values(), (edges, nodes, mapping, controls, restrictions, speed), strict=True):
        atomic_parquet(path, frame)
    export_report.update({"observed_mapping": mapping_report})
    export_report["artifact_sha256"] = payload_hash(export_report)
    atomic_json(docs / "stage3_full_network_export_report.json", export_report)
    validation = dict(validation)
    validation["artifact_sha256"] = payload_hash(validation)
    atomic_json(docs / "stage3_s2a_speed_validation.json", validation)
    build_reports(docs, edges, nodes, mapping_report, controls, restrictions, speed, validation)

    evidence = {
        "schema_version": "stage3_s2a_evidence_bundle.1",
        "phase_status": PHASE_STATUS,
        "base_commit": config["execution_base_commit"],
        "upstream_frozen_commit": config["upstream_frozen_commit"],
        "source_bindings": sources,
        "config": source_descriptor(config_path, root),
        "products": {key: parquet_descriptor(path, root) for key, path in products.items()},
        "reports": {
            name: source_descriptor(docs / name, root)
            for name in (
                "stage3_full_network_export_report.json", "stage3_s2a_speed_validation.json",
                "stage3_s2a_network_foundation_report.md", "stage3_s2a_speed_domain_report.md",
                "stage3_s2a_field_provenance.md",
            )
        },
        "verification": verification,
        "runtime_s": time.perf_counter() - started,
        "authorizations": config["authorizations"],
        "next_phase_authorized": False,
    }
    evidence["artifact_sha256"] = payload_hash(evidence)
    atomic_json(docs / "stage3_s2a_evidence_bundle.json", evidence)
    return evidence


def write_test_evidence(
    docs: Path,
    *,
    test_commands: Sequence[Mapping[str, Any]],
    compileall_status: str,
    evidence_verification_status: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": "stage3_s2a_test_evidence.1",
        "phase_status": PHASE_STATUS,
        "test_commands": list(test_commands),
        "compileall_status": compileall_status,
        "evidence_verification_status": evidence_verification_status,
        "s2b_authorized": False,
        "next_phase_authorized": False,
    }
    payload["artifact_sha256"] = payload_hash(payload)
    atomic_json(docs / "stage3_s2a_test_evidence.json", payload)
    return payload


def verify_evidence(path: Path, root: Path) -> dict[str, Any]:
    evidence = read_json(path)
    failures = []
    if evidence.get("artifact_sha256") != payload_hash(evidence):
        failures.append("evidence payload hash")
    if evidence.get("phase_status") != PHASE_STATUS:
        failures.append("phase status")
    for section in ("source_bindings", "products", "reports"):
        for descriptor in evidence.get(section, {}).values():
            artifact = resolve(root, descriptor["path"])
            if not artifact.is_file() or sha256_file(artifact) != descriptor["sha256"]:
                failures.append(f"artifact binding: {descriptor['path']}")
    if evidence.get("next_phase_authorized") is not False or evidence.get("authorizations", {}).get("s2b") is not False:
        failures.append("later phase authorization")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures, "phase_status": evidence.get("phase_status")}


def attach_test_evidence(evidence_path: Path, test_path: Path, root: Path) -> dict[str, Any]:
    """Bind post-run tests without re-running or mutating scientific products."""

    evidence = read_json(evidence_path)
    test_evidence = read_json(test_path)
    if test_evidence.get("artifact_sha256") != payload_hash(test_evidence):
        raise Stage3S2AError("test evidence payload hash mismatch")
    if (
        test_evidence.get("compileall_status") != "PASS"
        or test_evidence.get("evidence_verification_status") != "PASS"
        or any(item.get("status") != "PASS" for item in test_evidence.get("test_commands", []))
    ):
        raise Stage3S2AError("cannot bind non-PASS test evidence")
    evidence.setdefault("reports", {})[test_path.name] = source_descriptor(test_path, root)
    evidence["artifact_sha256"] = payload_hash(evidence)
    atomic_json(evidence_path, evidence)
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("stage3/config/stage3_s2a_network_foundation.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--verify-evidence", type=Path)
    parser.add_argument("--attach-test-evidence", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.attach_test_evidence:
        result = attach_test_evidence(
            (root / "stage3/docs/odd_tod/s2a/stage3_s2a_evidence_bundle.json").resolve(),
            args.attach_test_evidence.resolve(),
            root,
        )
    else:
        result = verify_evidence(args.verify_evidence, root) if args.verify_evidence else run(args.config.resolve(), root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status", result.get("verification", {}).get("status")) == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
