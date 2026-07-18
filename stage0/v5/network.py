"""Build the canonical directed edge store and edge-aware movement graph."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import osmium
import pandas as pd
from shapely.geometry import LineString, Point

from .config import Stage0Config, sha256_file
from .manifest import base_manifest, write_manifest


TRUE = {"yes", "true", "1"}
FALSE = {"no", "false", "0", ""}


def normalize_bool(value: object) -> bool:
    return str(value or "").strip().lower() in TRUE


def normalize_layer(value: object, bridge: object = None, tunnel: object = None) -> int:
    text = str(value or "").strip()
    if text:
        try:
            return int(float(text))
        except ValueError:
            pass
    if normalize_bool(bridge):
        return 1
    if normalize_bool(tunnel):
        return -1
    return 0


def normalize_oneway(value: object, junction: object = None, highway: object = None) -> str:
    text = str(value or "").strip().lower()
    if text in {"-1", "reverse"}:
        return "reverse"
    if text in TRUE or str(junction or "").lower() == "roundabout" or str(highway or "").lower() == "motorway":
        return "forward"
    return "both"


def stable_edge_uid(osm_way_id: object, segment_seq: int, direction: str) -> str:
    if direction not in {"F", "R"}:
        raise ValueError(f"invalid edge direction: {direction}")
    return f"{int(osm_way_id)}:{int(segment_seq)}:{direction}"


def directed_directions(oneway: str) -> tuple[str, ...]:
    """Return explicit canonical directions for a normalized OSM oneway value."""
    if oneway == "forward":
        return ("F",)
    if oneway == "reverse":
        return ("R",)
    if oneway == "both":
        return ("F", "R")
    raise ValueError(f"unknown normalized oneway value: {oneway}")


def motor_vehicle_eligible(tags: dict[str, str], excluded_highways: set[str]) -> bool:
    highway = tags.get("highway", "").lower()
    if not highway or highway in excluded_highways:
        return False
    motor = tags.get("motor_vehicle", tags.get("motorcar", "")).lower()
    access = tags.get("access", "").lower()
    if motor in {"no", "private"}:
        return False
    if access in {"no", "private"} and motor not in {"yes", "designated", "permissive"}:
        return False
    if highway == "service" and tags.get("service", "").lower() in {"parking_aisle", "driveway"} and motor not in {"yes", "permissive"}:
        return False
    return True


def road_penalty(tags: dict[str, str], config: dict[str, Any]) -> float:
    highway = tags.get("highway", "").lower()
    if highway == "service":
        return float(config["service_penalty"])
    if highway == "track":
        return float(config["track_penalty"])
    if highway == "living_street":
        return float(config["living_street_penalty"])
    return 1.0


class PbfRoadHandler(osmium.SimpleHandler):
    def __init__(self, network_config: dict[str, Any], snapshot: str):
        super().__init__()
        self.network_config = network_config
        self.snapshot = snapshot
        self.edges: list[dict[str, Any]] = []
        self.nodes: dict[int, tuple[float, float]] = {}
        self.restrictions: list[dict[str, Any]] = []
        self.invalid_way_count = 0

    def way(self, way: osmium.osm.Way) -> None:
        tags = {tag.k: tag.v for tag in way.tags}
        if not motor_vehicle_eligible(tags, set(self.network_config["excluded_highways"])):
            return
        locations: list[tuple[int, float, float]] = []
        for node in way.nodes:
            if not node.location.valid():
                self.invalid_way_count += 1
                return
            locations.append((int(node.ref), float(node.lon), float(node.lat)))
        if len(locations) < 2:
            return
        bridge = normalize_bool(tags.get("bridge"))
        tunnel = normalize_bool(tags.get("tunnel"))
        layer = normalize_layer(tags.get("layer"), bridge, tunnel)
        oneway = normalize_oneway(tags.get("oneway"), tags.get("junction"), tags.get("highway"))
        penalty = road_penalty(tags, self.network_config)
        timestamp = str(getattr(way, "timestamp", "") or "")
        for sequence, (left, right) in enumerate(zip(locations[:-1], locations[1:])):
            left_id, left_lon, left_lat = left
            right_id, right_lon, right_lat = right
            if left_id == right_id or (left_lon == right_lon and left_lat == right_lat):
                continue
            self.nodes[left_id] = (left_lon, left_lat)
            self.nodes[right_id] = (right_lon, right_lat)
            base = {
                "physical_way_id": str(way.id),
                "osm_way_id": int(way.id),
                "segment_seq": int(sequence),
                "highway": tags.get("highway"),
                "oneway": oneway,
                "layer": layer,
                "bridge": bridge,
                "tunnel": tunnel,
                "junction": tags.get("junction"),
                "service": tags.get("service"),
                "access": tags.get("access"),
                "motor_vehicle": tags.get("motor_vehicle", tags.get("motorcar")),
                "lanes": tags.get("lanes"),
                "maxspeed": tags.get("maxspeed"),
                "surface": tags.get("surface"),
                "name": tags.get("name"),
                "ref": tags.get("ref"),
                "source_snapshot": self.snapshot,
                "osm_timestamp": timestamp,
                "routing_penalty": penalty,
                "candidate_penalty": max(0.0, penalty - 1.0) * 20.0,
            }
            forward_geometry = LineString([(left_lon, left_lat), (right_lon, right_lat)])
            directions = directed_directions(oneway)
            if "F" in directions:
                self.edges.append({
                    **base,
                    "edge_uid": stable_edge_uid(way.id, sequence, "F"),
                    "direction": "F", "from_node": left_id, "to_node": right_id,
                    "edge_key": stable_edge_uid(way.id, sequence, "F"),
                    "geometry": forward_geometry,
                })
            if "R" in directions:
                reverse_geometry = LineString([(right_lon, right_lat), (left_lon, left_lat)])
                self.edges.append({
                    **base,
                    "edge_uid": stable_edge_uid(way.id, sequence, "R"),
                    "direction": "R", "from_node": right_id, "to_node": left_id,
                    "edge_key": stable_edge_uid(way.id, sequence, "R"),
                    "geometry": reverse_geometry,
                })

    def relation(self, relation: osmium.osm.Relation) -> None:
        tags = {tag.k: tag.v for tag in relation.tags}
        if tags.get("type") != "restriction":
            return
        row: dict[str, Any] = {
            "relation_id": int(relation.id),
            "restriction": tags.get("restriction", "unknown"),
            "from_way": None, "to_way": None, "via_node": None,
            "raw_members": [], "parse_status": "unresolved",
        }
        for member in relation.members:
            row["raw_members"].append(f"{member.type}:{member.ref}:{member.role}")
            if member.role == "from" and member.type == "w":
                row["from_way"] = int(member.ref)
            elif member.role == "to" and member.type == "w":
                row["to_way"] = int(member.ref)
            elif member.role == "via" and member.type == "n":
                row["via_node"] = int(member.ref)
        if all(row[key] is not None for key in ["from_way", "to_way", "via_node"]):
            row["parse_status"] = "parsed_via_node"
        row["raw_members"] = "|".join(row["raw_members"])
        self.restrictions.append(row)


def _bearing_at_end(geometry: LineString, at_end: bool) -> float:
    coords = list(geometry.coords)
    a, b = (coords[-2], coords[-1]) if at_end else (coords[0], coords[1])
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def signed_turn_angle(incoming: LineString, outgoing: LineString) -> float:
    return (( _bearing_at_end(outgoing, False) - _bearing_at_end(incoming, True) + 180.0) % 360.0) - 180.0


def movement_type(angle: float, is_uturn: bool) -> str:
    if is_uturn or abs(angle) >= 150:
        return "u_turn"
    if angle > 35:
        return "left"
    if angle < -35:
        return "right"
    return "straight"


def classify_parallel_edges(edges: gpd.GeoDataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (u, v), group in edges.groupby(["from_node", "to_node"], sort=False):
        if len(group) < 2:
            continue
        records = list(group.itertuples())
        for i, left in enumerate(records):
            for right in records[i + 1:]:
                same_geometry = left.geometry.equals_exact(right.geometry, tolerance=1e-9)
                same_level = (left.layer, left.bridge, left.tunnel) == (right.layer, right.bridge, right.tunnel)
                same_semantics = all(
                    getattr(left, name) == getattr(right, name)
                    for name in ["highway", "access", "service", "maxspeed"]
                )
                if not same_level:
                    category = "grade_separated"
                elif same_geometry and same_semantics:
                    category = "exact_duplicate"
                elif same_geometry:
                    category = "semantic_equivalent"
                elif left.physical_way_id == right.physical_way_id:
                    category = "attribute_conflict"
                else:
                    category = "true_parallel"
                rows.append({
                    "from_node": int(u), "to_node": int(v),
                    "left_edge_uid": left.edge_uid, "right_edge_uid": right.edge_uid,
                    "parallel_category": category, "same_geometry": same_geometry,
                    "same_level": same_level, "same_semantics": same_semantics,
                    "merge_allowed": category in {"exact_duplicate", "semantic_equivalent"},
                })
    return pd.DataFrame(rows)


def build_movement_graph(edges: gpd.GeoDataFrame, restrictions: pd.DataFrame) -> pd.DataFrame:
    incoming: dict[int, list[int]] = defaultdict(list)
    outgoing: dict[int, list[int]] = defaultdict(list)
    for index, row in edges.iterrows():
        incoming[int(row.to_node)].append(index)
        outgoing[int(row.from_node)].append(index)
    restriction_rows = restrictions.loc[restrictions.parse_status.eq("parsed_via_node")] if len(restrictions) else restrictions
    no_lookup = {
        (int(row.from_way), int(row.via_node), int(row.to_way)): str(row.restriction)
        for row in restriction_rows.itertuples()
        if str(row.restriction).startswith("no_")
    }
    only_lookup: dict[tuple[int, int], set[int]] = defaultdict(set)
    for row in restriction_rows.itertuples():
        if str(row.restriction).startswith("only_"):
            only_lookup[(int(row.from_way), int(row.via_node))].add(int(row.to_way))
    rows: list[dict[str, Any]] = []
    for via_node in sorted(set(incoming) & set(outgoing)):
        for left_idx in incoming[via_node]:
            left = edges.loc[left_idx]
            for right_idx in outgoing[via_node]:
                right = edges.loc[right_idx]
                level_ok = (left.layer, bool(left.bridge), bool(left.tunnel)) == (
                    right.layer, bool(right.bridge), bool(right.tunnel)
                ) or str(left.highway).endswith("_link") or str(right.highway).endswith("_link")
                restriction = "allowed"
                key = (int(left.osm_way_id), via_node, int(right.osm_way_id))
                if key in no_lookup:
                    restriction = f"forbidden:{no_lookup[key]}"
                allowed_only = only_lookup.get((int(left.osm_way_id), via_node))
                if allowed_only and int(right.osm_way_id) not in allowed_only:
                    restriction = "forbidden:only_restriction"
                is_uturn = left.from_node == right.to_node and left.to_node == right.from_node
                angle = signed_turn_angle(left.geometry, right.geometry)
                rows.append({
                    "from_edge_uid": left.edge_uid, "via_node": via_node,
                    "to_edge_uid": right.edge_uid,
                    "movement_type": movement_type(angle, is_uturn),
                    "turn_angle": angle, "restriction_status": restriction,
                    "layer_compatibility": bool(level_ok),
                    "road_class_transition": f"{left.highway}->{right.highway}",
                    "merge_diverge_flag": len(incoming[via_node]) > 1 or len(outgoing[via_node]) > 1,
                })
    return pd.DataFrame(rows)


def _pbf_timestamp(path: Path) -> str:
    try:
        reader = osmium.io.Reader(str(path))
        header = reader.header()
        value = header.get("osmosis_replication_timestamp") or header.get("timestamp") or ""
        reader.close()
        return str(value)
    except Exception:
        return ""


def build_network(config: Stage0Config, repo: Path, force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    pbf = config.path("pbf", repo)
    output = config.path("output", repo) / "network"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "network_manifest.json"
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "PASS" and existing.get("config_hash") == config.digest:
            return existing
    snapshot = _pbf_timestamp(pbf) or f"unknown;file_mtime={pbf.stat().st_mtime_ns}"
    handler = PbfRoadHandler(config.section("network"), snapshot)
    handler.apply_file(str(pbf), locations=True, idx="flex_mem")
    if not handler.edges:
        raise RuntimeError("PBF produced no motor-vehicle edges")
    edges = gpd.GeoDataFrame(handler.edges, geometry="geometry", crs=4326).to_crs(config.section("network")["metric_crs"])
    edges["length_m"] = edges.geometry.length.astype(float)
    edges["routing_cost_m"] = edges.length_m * edges.routing_penalty.astype(float)
    edges["network_snapshot_mismatch"] = edges.osm_timestamp.astype(str).str[:10].gt("2016-10-31")
    parallel = classify_parallel_edges(edges)
    if len(parallel):
        group_map: dict[str, str] = {}
        for group_no, row in enumerate(parallel.itertuples()):
            group_id = f"parallel_{group_no:07d}"
            group_map.setdefault(row.left_edge_uid, group_id)
            group_map.setdefault(row.right_edge_uid, group_id)
        edges["parallel_group"] = edges.edge_uid.map(group_map)
    else:
        edges["parallel_group"] = pd.NA
    node_rows = [
        {"node_id": node_id, "geometry": Point(lon, lat)}
        for node_id, (lon, lat) in handler.nodes.items()
    ]
    nodes = gpd.GeoDataFrame(node_rows, geometry="geometry", crs=4326).to_crs(config.section("network")["metric_crs"])
    restrictions = pd.DataFrame(handler.restrictions)
    if restrictions.empty:
        restrictions = pd.DataFrame(columns=["relation_id", "restriction", "from_way", "to_way", "via_node", "raw_members", "parse_status"])
    movements = build_movement_graph(edges, restrictions)
    metric = (
        edges.sort_values(["from_node", "to_node", "routing_cost_m", "edge_uid"])
        .drop_duplicates(["from_node", "to_node"])
        [["from_node", "to_node", "edge_uid", "routing_cost_m", "length_m"]]
    )
    edges.to_parquet(output / "canonical_edges.parquet", index=False, compression="zstd")
    nodes.to_parquet(output / "canonical_nodes.parquet", index=False, compression="zstd")
    movements.to_parquet(output / "movement_graph.parquet", index=False, compression="zstd")
    metric.to_parquet(output / "metric_routing_graph.parquet", index=False, compression="zstd")
    parallel.to_parquet(output / "parallel_edge_audit.parquet", index=False, compression="zstd")
    restrictions.to_parquet(output / "turn_restrictions.parquet", index=False, compression="zstd")
    parsed_restrictions = int(restrictions.parse_status.eq("parsed_via_node").sum())
    manifest = {
        **base_manifest(repo, config.digest, [pbf]),
        "status": "PASS", "source_snapshot": snapshot,
        "edge_count": len(edges), "node_count": len(nodes), "movement_count": len(movements),
        "parallel_pair_count": len(parallel), "restriction_count": len(restrictions),
        "parsed_restriction_count": parsed_restrictions,
        "restriction_coverage": parsed_restrictions / len(restrictions) if len(restrictions) else None,
        "future_timestamp_edge_count": int(edges.network_snapshot_mismatch.sum()),
        "invalid_way_count": handler.invalid_way_count,
        "runtime_sec": time.perf_counter() - started,
    }
    write_manifest(manifest_path, manifest)
    categories = parallel.parallel_category.value_counts().to_dict() if len(parallel) else {}
    (output / "network_quality_report.md").write_text(
        "# Stage 0 v5 network quality report\n\n"
        f"- Status: PASS\n- PBF SHA-256: `{sha256_file(pbf)}`\n- Source timestamp: `{snapshot}`\n"
        f"- Directed edges: {len(edges):,}\n- Nodes: {len(nodes):,}\n- Legal/recorded movements: {len(movements):,}\n"
        f"- Turn restrictions: {len(restrictions):,}; parsed via-node: {parsed_restrictions:,}\n"
        f"- Parallel classifications: `{json.dumps(categories, ensure_ascii=False)}`\n"
        f"- Edges timestamped after 2016-10-31: {int(edges.network_snapshot_mismatch.sum()):,}\n\n"
        "The metric graph collapses node-pair distance alternatives only for bounded distance queries. "
        "Final route reconstruction always uses edge_uid states and the movement graph.\n",
        encoding="utf-8",
    )
    return manifest
