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


TRUE = {
    "yes", "true", "1",
    "viaduct", "movable", "aqueduct",
    "culvert", "building_passage", "avalanche_protector",
}
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
                    # Geometry equivalence alone is insufficient. Candidate
                    # aliasing is decided later after movement and restriction
                    # signatures are available.
                    "merge_allowed": category == "exact_duplicate",
                })
    return pd.DataFrame(rows)


def parallel_components(
    edge_uids: pd.Series,
    audit: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build transitive parallel groups and duplicate-only candidate aliases."""
    values = [str(value) for value in edge_uids]

    def components(rows: pd.DataFrame) -> dict[str, str]:
        touched = set(rows.left_edge_uid.astype(str)) | set(rows.right_edge_uid.astype(str)) if len(rows) else set()
        parent = {value: value for value in touched}

        def find(value: str) -> str:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(left: str, right: str) -> None:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                first, second = sorted((root_left, root_right))
                parent[second] = first

        for row in rows.itertuples():
            union(str(row.left_edge_uid), str(row.right_edge_uid))
        groups: dict[str, list[str]] = defaultdict(list)
        for value in touched:
            groups[find(value)].append(value)
        result: dict[str, str] = {}
        for number, members in enumerate(sorted((sorted(group) for group in groups.values()), key=lambda group: group[0])):
            group_id = f"parallel_{number:07d}"
            result.update({member: group_id for member in members})
        return result

    parallel_groups = components(audit) if len(audit) else {}
    merge_rows = (
        audit.loc[audit.alias_safe.astype(bool)]
        if len(audit) and "alias_safe" in audit.columns
        else audit.iloc[0:0]
    )
    duplicate_groups = components(merge_rows) if len(merge_rows) else {}
    aliases: dict[str, str] = {value: value for value in values}
    by_group: dict[str, list[str]] = defaultdict(list)
    for edge_uid, group in duplicate_groups.items():
        by_group[group].append(edge_uid)
    for members in by_group.values():
        alias = min(members)
        aliases.update({member: alias for member in members})
    return parallel_groups, aliases


def audit_candidate_aliases(
    edges: gpd.GeoDataFrame,
    parallel: pd.DataFrame,
    movements: pd.DataFrame,
) -> pd.DataFrame:
    """Prove exact duplicates safe before collapsing only their candidate state."""
    if parallel.empty:
        return pd.DataFrame(columns=[
            "edge_uid", "alias_uid", "alias_reason", "geometry_equal",
            "semantic_equal", "movement_signature_equal",
            "restriction_signature_equal", "alias_safe",
        ])
    edge_lookup = edges.set_index("edge_uid", drop=False)
    movement_rows: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    restriction_rows: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in movements.itertuples(index=False):
        outgoing = (
            "out", str(row.to_edge_uid), str(row.movement_type),
            str(row.level_transition_type), str(row.road_class_transition),
        )
        incoming = (
            "in", str(row.from_edge_uid), str(row.movement_type),
            str(row.level_transition_type), str(row.road_class_transition),
        )
        movement_rows[str(row.from_edge_uid)].add(outgoing)
        movement_rows[str(row.to_edge_uid)].add(incoming)
        restriction_rows[str(row.from_edge_uid)].add(("out", str(row.restriction_status)))
        restriction_rows[str(row.to_edge_uid)].add(("in", str(row.restriction_status)))
    rows: list[dict[str, Any]] = []
    semantic_fields = (
        "direction", "highway", "access", "motor_vehicle", "service",
        "layer", "bridge", "tunnel", "junction", "maxspeed",
    )
    for pair in parallel.itertuples(index=False):
        left_uid, right_uid = str(pair.left_edge_uid), str(pair.right_edge_uid)
        left, right = edge_lookup.loc[left_uid], edge_lookup.loc[right_uid]
        geometry_equal = bool(pair.same_geometry)
        semantic_equal = all(str(left.get(name, "")) == str(right.get(name, "")) for name in semantic_fields)
        movement_equal = movement_rows[left_uid] == movement_rows[right_uid]
        restriction_equal = restriction_rows[left_uid] == restriction_rows[right_uid]
        safe = (
            str(pair.parallel_category) == "exact_duplicate"
            and geometry_equal and semantic_equal and movement_equal and restriction_equal
        )
        rows.append({
            "edge_uid": right_uid,
            "alias_uid": min(left_uid, right_uid) if safe else right_uid,
            "alias_reason": "proven_exact_duplicate" if safe else "not_aliased",
            "geometry_equal": geometry_equal,
            "semantic_equal": semantic_equal,
            "movement_signature_equal": movement_equal,
            "restriction_signature_equal": restriction_equal,
            "alias_safe": safe,
            "left_edge_uid": left_uid,
            "right_edge_uid": right_uid,
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
    unresolved_pairs = {
        (int(row.from_way), int(row.to_way))
        for row in restrictions.itertuples()
        if str(row.parse_status) != "parsed_via_node"
        and pd.notna(row.from_way) and pd.notna(row.to_way)
    } if len(restrictions) else set()
    rows: list[dict[str, Any]] = []
    for via_node in sorted(set(incoming) & set(outgoing)):
        for left_idx in incoming[via_node]:
            left = edges.loc[left_idx]
            for right_idx in outgoing[via_node]:
                right = edges.loc[right_idx]
                left_link = str(left.highway).endswith("_link")
                right_link = str(right.highway).endswith("_link")
                if left_link or right_link:
                    level_transition = "ramp_transition"
                elif not bool(left.bridge) and bool(right.bridge):
                    level_transition = "bridge_entry"
                elif bool(left.bridge) and not bool(right.bridge):
                    level_transition = "bridge_exit"
                elif not bool(left.tunnel) and bool(right.tunnel):
                    level_transition = "tunnel_entry"
                elif bool(left.tunnel) and not bool(right.tunnel):
                    level_transition = "tunnel_exit"
                elif (left.layer, bool(left.bridge), bool(left.tunnel)) == (
                    right.layer, bool(right.bridge), bool(right.tunnel)
                ):
                    level_transition = "same_level"
                else:
                    # A shared directed OSM node is positive topological evidence.  Attribute
                    # changes are diagnostic unless an independent audit proves the node false.
                    level_transition = "suspicious_level_jump"
                level_ok = True
                restriction = "allowed"
                key = (int(left.osm_way_id), via_node, int(right.osm_way_id))
                if key in no_lookup:
                    restriction = f"forbidden:{no_lookup[key]}"
                allowed_only = only_lookup.get((int(left.osm_way_id), via_node))
                if allowed_only and int(right.osm_way_id) not in allowed_only:
                    restriction = "forbidden:only_restriction"
                if (
                    restriction == "allowed"
                    and (int(left.osm_way_id), int(right.osm_way_id)) in unresolved_pairs
                ):
                    restriction = "unresolved_restriction"
                is_uturn = left.from_node == right.to_node and left.to_node == right.from_node
                angle = signed_turn_angle(left.geometry, right.geometry)
                rows.append({
                    "from_edge_uid": left.edge_uid, "via_node": via_node,
                    "to_edge_uid": right.edge_uid,
                    "movement_type": movement_type(angle, is_uturn),
                    "turn_angle": angle, "restriction_status": restriction,
                    "layer_compatibility": bool(level_ok),
                    "level_transition_type": level_transition,
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
    pbf_stat = pbf.stat()
    pbf_identity = f"size={pbf_stat.st_size};mtime_ns={pbf_stat.st_mtime_ns}"
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_inputs = existing.get("inputs", [])
        recorded_pbf = next(
            (row for row in existing_inputs if Path(str(row.get("path", ""))).resolve() == pbf.resolve()),
            {},
        )
        if (
            existing.get("status") == "PASS"
            and existing.get("config_hash") == config.digest
            and recorded_pbf.get("reproducible_identifier") == pbf_identity
        ):
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
    group_map, _ = parallel_components(edges.edge_uid, parallel)
    edges["parallel_group"] = edges.edge_uid.map(group_map)
    node_rows = [
        {"node_id": node_id, "geometry": Point(lon, lat)}
        for node_id, (lon, lat) in handler.nodes.items()
    ]
    nodes = gpd.GeoDataFrame(node_rows, geometry="geometry", crs=4326).to_crs(config.section("network")["metric_crs"])
    restrictions = pd.DataFrame(handler.restrictions)
    if restrictions.empty:
        restrictions = pd.DataFrame(columns=["relation_id", "restriction", "from_way", "to_way", "via_node", "raw_members", "parse_status"])
    movements = build_movement_graph(edges, restrictions)
    alias_audit = audit_candidate_aliases(edges, parallel, movements)
    _, candidate_alias = parallel_components(edges.edge_uid, alias_audit)
    edges["candidate_alias_uid"] = edges.edge_uid.astype(str).map(candidate_alias)
    allowed_movements = movements.loc[
        ~movements.restriction_status.astype(str).str.startswith("forbidden")
        & movements.restriction_status.ne("unresolved_restriction")
        & movements.layer_compatibility.astype(bool)
    ].copy()
    metric = (
        edges.sort_values(["from_node", "to_node", "routing_cost_m", "edge_uid"])
        .drop_duplicates(["from_node", "to_node"])
        [["from_node", "to_node", "edge_uid", "routing_cost_m", "length_m"]]
    )
    edges.to_parquet(output / "canonical_edges.parquet", index=False, compression="zstd")
    nodes.to_parquet(output / "canonical_nodes.parquet", index=False, compression="zstd")
    movements.to_parquet(output / "movement_graph.parquet", index=False, compression="zstd")
    movements.to_parquet(output / "raw_movement_audit_graph.parquet", index=False, compression="zstd")
    allowed_movements.to_parquet(output / "allowed_movement_graph.parquet", index=False, compression="zstd")
    metric.to_parquet(output / "metric_routing_graph.parquet", index=False, compression="zstd")
    parallel.to_parquet(output / "parallel_edge_audit.parquet", index=False, compression="zstd")
    alias_audit.to_parquet(output / "candidate_alias_audit.parquet", index=False, compression="zstd")
    restrictions.to_parquet(output / "turn_restrictions.parquet", index=False, compression="zstd")
    parsed_restrictions = int(restrictions.parse_status.eq("parsed_via_node").sum())
    manifest = {
        **base_manifest(repo, config.digest, [pbf]),
        "status": "PASS", "source_snapshot": snapshot,
        "edge_count": len(edges), "node_count": len(nodes), "movement_count": len(movements),
        "parallel_pair_count": len(parallel), "restriction_count": len(restrictions),
        "parallel_component_count": int(edges.parallel_group.nunique(dropna=True)),
        "candidate_alias_edge_count": int(edges.candidate_alias_uid.ne(edges.edge_uid.astype(str)).sum()),
        "candidate_state_count": int(edges.candidate_alias_uid.nunique()),
        "safe_candidate_alias_pair_count": int(alias_audit.alias_safe.sum()) if len(alias_audit) else 0,
        "allowed_movement_count": int(len(allowed_movements)),
        "raw_movement_audit_available": True,
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
        f"- Parallel connected components: {int(edges.parallel_group.nunique(dropna=True)):,}\n"
        f"- Duplicate/equivalent candidate aliases: {int(edges.candidate_alias_uid.ne(edges.edge_uid.astype(str)).sum()):,}\n"
        f"- Edges timestamped after 2016-10-31: {int(edges.network_snapshot_mismatch.sum()):,}\n\n"
        "The metric graph collapses node-pair distance alternatives only for bounded distance queries. "
        "Final route reconstruction always uses edge_uid states and the movement graph.\n",
        encoding="utf-8",
    )
    return manifest
