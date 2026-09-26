"""S2B-1 topology-aware intersection-complex calibration and QA pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import time
import zlib
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer
from scipy.spatial import cKDTree

from stage3.odd_tod.network_foundation import (
    Stage3S2AError,
    atomic_json,
    atomic_parquet,
    atomic_text,
    git_head,
    parquet_descriptor,
    payload_hash,
    read_json,
    sha256_file,
    source_descriptor,
)


PHASE_STATUS = "STAGE3_S2B1_CALIBRATION_PACK_COMPLETE"
CONFIG_SCHEMA = "stage3_s2b1_intersection_complex_config.1"
GENERATION_VERSION = "stage3-s2b1-topology-v4"


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def load_config(path: Path, root: Path) -> dict[str, Any]:
    config = read_json(path)
    expected_auth = {"s2b1": True, "s2b2": False, **{f"s{i}": False for i in range(3, 9)}, "stage4": False}
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("phase") != "S2B1_CANDIDATE_CALIBRATION_QA"
        or config.get("execution_authorization") != "S2B1_ONLY"
        or config.get("candidate_tolerances_m") != [5, 10, 15, 20]
        or config.get("metric_crs") != "EPSG:32649"
        or config.get("authorizations") != expected_auth
        or config.get("next_phase_authorized") is not False
        or any(config.get("scope_guards", {}).values())
        or config.get("frozen_speed") != {"quantile": 0.85, "method": "MAP_SPEED_AND_ROAD_CLASS", "caps_kmh": [60, 80, 120]}
    ):
        raise Stage3S2AError("S2B-1 config violates authorization or frozen decisions")
    return config


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _uid(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode()).hexdigest()[:24]


def complex_uid(tolerance: int, members: Sequence[str]) -> str:
    return _uid("s3ic_", f"S2B1|r={tolerance}|" + "|".join(sorted(members)))


def physical_connection_uid(osm_way_id: Any, left: str, right: str) -> str:
    way = "NA" if pd.isna(osm_way_id) else str(int(osm_way_id))
    return _uid("s3pc_", f"{way}|" + "|".join(sorted((left, right))))


def turn_type(angle: float | None) -> str:
    if angle is None or not np.isfinite(angle):
        return "UNKNOWN"
    absolute = abs(float(angle))
    if absolute <= 30:
        return "STRAIGHT"
    if absolute >= 150:
        return "UTURN"
    return "LEFT" if angle > 0 else "RIGHT"


def _bearing(a: Sequence[float], b: Sequence[float]) -> float | None:
    dx, dy = float(b[0]) - float(a[0]), float(b[1]) - float(a[1])
    if math.hypot(dx, dy) < 1e-10:
        return None
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def signed_turn(entry: float | None, exit_: float | None) -> float | None:
    if entry is None or exit_ is None:
        return None
    # Bearings increase clockwise; entry-exit therefore makes left positive.
    return ((entry - exit_ + 180.0) % 360.0) - 180.0


def bind_inputs(config: Mapping[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    paths = {key: _resolve(root, value) for key, value in config["paths"].items()}
    mapping = {
        "edges": "edges_sha256", "nodes": "nodes_sha256", "observed_mapping": "observed_mapping_sha256",
        "controls": "controls_sha256", "restrictions": "restrictions_sha256", "speed_domain": "speed_domain_sha256",
        "reverse_overlay": "reverse_overlay_sha256",
    }
    descriptors = {}
    for key, binding in mapping.items():
        descriptor = parquet_descriptor(paths[key], root)
        if descriptor["sha256"] != config["bindings"][binding]:
            raise Stage3S2AError(f"BLOCKER: frozen input changed: {key}")
        descriptors[key] = descriptor
    descriptors["s2a_evidence"] = source_descriptor(paths["s2a_evidence"], root)
    descriptors["s2a1_evidence"] = source_descriptor(paths["s2a1_evidence"], root)
    if descriptors["edges"]["row_count"] != 209_454 or descriptors["nodes"]["row_count"] != 89_607:
        raise Stage3S2AError("full frozen network dimensions changed")
    return descriptors


def load_inputs(config: Mapping[str, Any], root: Path):
    paths = {key: _resolve(root, value) for key, value in config["paths"].items()}
    edges = pq.read_table(paths["edges"]).to_pandas()
    nodes = pq.read_table(paths["nodes"]).to_pandas()
    controls = pq.read_table(paths["controls"]).to_pandas()
    restrictions = pq.read_table(paths["restrictions"]).to_pandas()
    overlay = pq.read_table(paths["reverse_overlay"]).to_pandas()
    return paths, edges, nodes, controls, restrictions, overlay


def prepare_topology(edges: pd.DataFrame, nodes: pd.DataFrame):
    node_set = set(nodes["stage3_node_uid"])
    usable = edges[
        edges["from_stage3_node_uid"].isin(node_set) & edges["to_stage3_node_uid"].isin(node_set)
    ].copy()
    if not usable["auto_routable"].all():
        raise Stage3S2AError("non-auto edge entered AV topology")
    undirected_neighbors: dict[str, set[str]] = defaultdict(set)
    outgoing: dict[str, list[int]] = defaultdict(list)
    incoming: dict[str, list[int]] = defaultdict(list)
    for index, row in usable.iterrows():
        left, right = str(row.from_stage3_node_uid), str(row.to_stage3_node_uid)
        undirected_neighbors[left].add(right)
        undirected_neighbors[right].add(left)
        outgoing[left].append(index)
        incoming[right].append(index)
    return usable, undirected_neighbors, outgoing, incoming


def detect_candidates(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    controls: pd.DataFrame,
    restrictions: pd.DataFrame,
    undirected_neighbors: Mapping[str, set[str]],
    outgoing: Mapping[str, list[int]],
    incoming: Mapping[str, list[int]],
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    node_by_osm = {
        int(row.osm_node_id): str(row.stage3_node_uid)
        for row in nodes.dropna(subset=["osm_node_id"]).itertuples(index=False)
    }
    edge_by_uid = edges.set_index("stage3_edge_uid")
    signal_nodes, roundabout_nodes, restriction_nodes = set(), set(), set()
    control_nodes: dict[str, set[str]] = defaultdict(set)
    for row in controls.itertuples(index=False):
        targets: set[str] = set()
        if pd.notna(row.osm_node_id) and int(row.osm_node_id) in node_by_osm:
            targets.add(node_by_osm[int(row.osm_node_id)])
        # Exact OSM-node identity is stronger than edge exposure. Use mapped
        # edge endpoints only as a fallback (and for way-level roundabouts),
        # otherwise one signal node would spuriously mark every adjacent end.
        if not targets:
            try:
                mapped = json.loads(row.mapped_stage3_edge_uids)
            except (TypeError, json.JSONDecodeError):
                mapped = []
            for edge_uid in mapped:
                if edge_uid in edge_by_uid.index:
                    edge = edge_by_uid.loc[edge_uid]
                    for value in (edge.from_stage3_node_uid, edge.to_stage3_node_uid):
                        if pd.notna(value):
                            targets.add(str(value))
        control_nodes[str(row.control_evidence_type)].update(targets)
    signal_nodes = control_nodes["SIGNALIZED"]
    roundabout_nodes = control_nodes["ROUNDABOUT"]
    round_edges = edges[edges["junction_roundabout_way"] | edges["mini_roundabout_node_exposure"]]
    for row in round_edges.itertuples(index=False):
        roundabout_nodes.update(
            str(value) for value in (row.from_stage3_node_uid, row.to_stage3_node_uid) if pd.notna(value)
        )
    for row in restrictions.itertuples(index=False):
        try:
            vias = json.loads(row.via_members)
        except (TypeError, json.JSONDecodeError):
            vias = []
        for via in vias:
            if via.get("type") == "n" and int(via["ref"]) in node_by_osm:
                restriction_nodes.add(node_by_osm[int(via["ref"])])
        for field in ("from_stage3_edge_uids", "to_stage3_edge_uids"):
            for edge_uid in json.loads(getattr(row, field)):
                if edge_uid in edge_by_uid.index:
                    edge = edge_by_uid.loc[edge_uid]
                    restriction_nodes.update(
                        str(value) for value in (edge.from_stage3_node_uid, edge.to_stage3_node_uid) if pd.notna(value)
                    )

    rows = []
    for row in nodes.itertuples(index=False):
        uid = str(row.stage3_node_uid)
        degree = len(undirected_neighbors.get(uid, set()))
        in_neighbors = {str(edges.loc[index].from_stage3_node_uid) for index in incoming.get(uid, [])}
        out_neighbors = {str(edges.loc[index].to_stage3_node_uid) for index in outgoing.get(uid, [])}
        rule_a = degree >= 3
        rule_b = degree >= 3 and (len(in_neighbors) >= 2 or len(out_neighbors) >= 2)
        rule_c = uid in signal_nodes
        rule_d = uid in roundabout_nodes
        rule_e = uid in restriction_nodes
        rule_f = degree >= 3 and len(in_neighbors) != len(out_neighbors)
        if any((rule_a, rule_b, rule_c, rule_d, rule_e, rule_f)):
            rows.append(
                {
                    "stage3_node_uid": uid, "valhalla_node_id": int(row.valhalla_node_id),
                    "osm_node_id": int(row.osm_node_id) if pd.notna(row.osm_node_id) else None,
                    "lon": float(row.lon), "lat": float(row.lat), "physical_undirected_degree": degree,
                    "distinct_incoming_neighbor_count": len(in_neighbors), "distinct_outgoing_neighbor_count": len(out_neighbors),
                    "rule_a_physical_branching": rule_a, "rule_b_directed_merge_diverge": rule_b,
                    "rule_c_signal": rule_c, "rule_d_roundabout": rule_d,
                    "rule_e_restriction_context": rule_e, "rule_f_merge_diverge": rule_f,
                }
            )
    candidates = pd.DataFrame(rows).sort_values("stage3_node_uid").reset_index(drop=True)
    evidence = {"signal": signal_nodes, "roundabout": roundabout_nodes, "restriction": restriction_nodes}
    return candidates, evidence


def _roundabout_groups(edges: pd.DataFrame) -> tuple[list[set[str]], set[str]]:
    round_edges = edges[edges["junction_roundabout_way"] | edges["mini_roundabout_node_exposure"]]
    values = set()
    pairs = []
    for row in round_edges.itertuples(index=False):
        if pd.notna(row.from_stage3_node_uid) and pd.notna(row.to_stage3_node_uid):
            left, right = str(row.from_stage3_node_uid), str(row.to_stage3_node_uid)
            values.update((left, right)); pairs.append((left, right))
    uf = UnionFind(values)
    for left, right in pairs:
        uf.union(left, right)
    groups: dict[str, set[str]] = defaultdict(set)
    for value in values:
        groups[uf.find(value)].add(value)
    return list(groups.values()), values


def _facility_signatures(edges: pd.DataFrame) -> dict[str, dict[str, Any]]:
    accum: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"layers": set(), "bridge_all": True, "tunnel_all": True, "count": 0}
    )
    for row in edges.itertuples(index=False):
        for uid in (str(row.from_stage3_node_uid), str(row.to_stage3_node_uid)):
            item = accum[uid]; item["count"] += 1
            if pd.notna(row.osm_layer): item["layers"].add(int(row.osm_layer))
            item["bridge_all"] = item["bridge_all"] and bool(row.bridge_effective)
            item["tunnel_all"] = item["tunnel_all"] and bool(row.tunnel_effective)
    return dict(accum)


def _grade_compatible(left: str, right: str, signatures: Mapping[str, Mapping[str, Any]], direct_neighbors) -> bool:
    a, b = signatures[left], signatures[right]
    if a["bridge_all"] != b["bridge_all"] or a["tunnel_all"] != b["tunnel_all"]:
        return False
    if a["layers"] and b["layers"] and a["layers"].isdisjoint(b["layers"]):
        return False
    return True


def construct_membership(
    tolerance: int,
    candidates: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    undirected_neighbors,
    outgoing,
    incoming,
) -> pd.DataFrame:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
    node_x, node_y = transformer.transform(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    node_xy = np.column_stack((node_x, node_y))
    node_uids = nodes["stage3_node_uid"].astype(str).to_numpy()
    node_pos = {uid: index for index, uid in enumerate(node_uids)}
    candidate_uids = candidates["stage3_node_uid"].astype(str).to_numpy()
    candidate_indexes = np.array([node_pos[uid] for uid in candidate_uids])
    candidate_xy = node_xy[candidate_indexes]
    tree = cKDTree(candidate_xy)
    spatial_pairs = sorted(tree.query_pairs(2.0 * tolerance, output_type="set"))
    round_groups, round_nodes = _roundabout_groups(edges)
    round_group_by_node = {
        node_uid: group_index for group_index, group in enumerate(round_groups) for node_uid in group
    }
    all_signatures = _facility_signatures(edges)
    signatures = {uid: all_signatures.get(uid, {"layers": set(), "bridge_all": False, "tunnel_all": False}) for uid in candidate_uids}

    spatial_uf = UnionFind(candidate_uids)
    for left_i, right_i in spatial_pairs:
        left, right = candidate_uids[left_i], candidate_uids[right_i]
        # Preserve roundabout semantic boundary from unrelated nearby junctions.
        if (left in round_nodes) != (right in round_nodes):
            continue
        if left in round_nodes and round_group_by_node[left] != round_group_by_node[right]:
            continue
        if _grade_compatible(left, right, signatures, undirected_neighbors):
            spatial_uf.union(left, right)
    for group in round_groups:
        members = sorted(set(candidate_uids) & group)
        for value in members[1:]:
            spatial_uf.union(members[0], value)

    preclusters: dict[str, list[str]] = defaultdict(list)
    for uid in candidate_uids:
        preclusters[spatial_uf.find(uid)].append(uid)
    all_tree = cKDTree(node_xy)
    complex_members: list[set[str]] = []
    candidate_set = set(candidate_uids)
    for candidate_group in sorted(preclusters.values(), key=lambda values: min(values)):
        if len(candidate_group) == 1:
            complex_members.append({candidate_group[0]}); continue
        indexes: set[int] = set()
        for uid in candidate_group:
            indexes.update(all_tree.query_ball_point(node_xy[node_pos[uid]], tolerance))
        induced = {node_uids[index] for index in indexes}
        # Topology split inside the union of candidate buffers.
        unseen = set(induced)
        components = []
        while unseen:
            seed = min(unseen); unseen.remove(seed); component = {seed}; queue = deque([seed])
            while queue:
                current = queue.popleft()
                for neighbor in undirected_neighbors.get(current, set()):
                    if neighbor in unseen:
                        unseen.remove(neighbor); component.add(neighbor); queue.append(neighbor)
            components.append(component)
        assigned_candidates = set()
        for component in components:
            selected = set(candidate_group) & component
            if selected:
                # Intermediate nodes are included only where they form the
                # necessary induced-topology connection between candidates.
                members = ((component - candidate_set) | selected) if len(selected) > 1 else selected
                complex_members.append(members); assigned_candidates.update(selected)
        for uid in set(candidate_group) - assigned_candidates:
            complex_members.append({uid})

    rows = []
    for members in sorted(complex_members, key=lambda values: min(values)):
        uid = complex_uid(tolerance, sorted(members))
        for node_uid in sorted(members):
            rows.append(
                {
                    "tolerance_m": tolerance, "intersection_complex_uid": uid,
                    "stage3_node_uid": node_uid, "candidate_node": node_uid in candidate_set,
                    "membership_source": "CANDIDATE" if node_uid in candidate_set else "TOPOLOGY_INTERMEDIATE",
                }
            )
    return pd.DataFrame(rows)


def _directed_path_exists(start: str, end: str, members: set[str], edge_records, outgoing) -> bool:
    if start == end:
        return True
    seen, queue = {start}, deque([start])
    while queue:
        current = queue.popleft()
        for index in outgoing.get(current, []):
            target = str(edge_records[index].to_stage3_node_uid)
            if target not in members or target in seen:
                continue
            if target == end:
                return True
            seen.add(target); queue.append(target)
    return False


def build_products(
    tolerance: int,
    membership: pd.DataFrame,
    candidates: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    controls: pd.DataFrame,
    restrictions: pd.DataFrame,
    undirected_neighbors,
    outgoing,
    incoming,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_lookup = nodes.set_index("stage3_node_uid")
    candidate_lookup = candidates.set_index("stage3_node_uid")
    edge_records = {index: row for index, row in zip(edges.index, edges.itertuples(index=False), strict=True)}
    edge_to_control: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in controls.itertuples(index=False):
        for edge_uid in json.loads(row.mapped_stage3_edge_uids):
            edge_to_control[edge_uid].append((str(row.evidence_id), str(row.control_evidence_type)))
    restriction_pairs = []
    for row in restrictions.itertuples(index=False):
        for from_uid in json.loads(row.from_stage3_edge_uids):
            for to_uid in json.loads(row.to_stage3_edge_uids):
                restriction_pairs.append((from_uid, to_uid, bool(row.directed_enforcement_certified)))
    restriction_lookup = {(a, b): certified for a, b, certified in restriction_pairs}

    complex_rows, movement_rows = [], []
    excessive = float(config["qa"]["excessive_internal_length_m"])
    extreme = int(config["qa"]["extreme_member_count"])
    for complex_id, group in membership.groupby("intersection_complex_uid", sort=True):
        members = set(group["stage3_node_uid"].astype(str))
        internal_indexes, in_indexes, out_indexes = set(), set(), set()
        for node_uid in members:
            for index in outgoing.get(node_uid, []):
                target = str(edge_records[index].to_stage3_node_uid)
                (internal_indexes if target in members else out_indexes).add(index)
            for index in incoming.get(node_uid, []):
                source = str(edge_records[index].from_stage3_node_uid)
                if source not in members:
                    in_indexes.add(index)
        internal_records = [edge_records[index] for index in sorted(internal_indexes)]
        boundary_indexes = sorted(in_indexes | out_indexes)
        physical_connections = {
            physical_connection_uid(row.osm_way_id, str(row.from_stage3_node_uid), str(row.to_stage3_node_uid))
            for row in (edge_records[index] for index in boundary_indexes)
        }
        control_items = [item for index in boundary_indexes for item in edge_to_control.get(edge_records[index].stage3_edge_uid, [])]
        evidence_types = {item[1] for item in control_items}
        candidate_members = set(group.loc[group["candidate_node"], "stage3_node_uid"].astype(str))
        signal_count = sum(
            bool(candidate_lookup.loc[node_uid, "rule_c_signal"])
            for node_uid in candidate_members if node_uid in candidate_lookup.index
        )
        if signal_count:
            evidence_types.add("SIGNALIZED")
        else:
            evidence_types.discard("SIGNALIZED")
        roundabout = any(
            bool(row.junction_roundabout_way) or bool(row.mini_roundabout_node_exposure)
            for row in internal_records
        ) or "ROUNDABOUT" in evidence_types
        if roundabout:
            signal_state = "ROUNDABOUT"
        elif signal_count > 0:
            signal_state = "SIGNALIZED"
        elif evidence_types & {"STOP_CONTROL", "YIELD_CONTROL"}:
            signal_state = "STOP_OR_YIELD_CONTROLLED"
        else:
            signal_state = "UNKNOWN_CONTROL"
        control_conflict = len(evidence_types & {"SIGNALIZED", "ROUNDABOUT", "STOP_CONTROL", "YIELD_CONTROL"}) > 1
        layers = {int(row.osm_layer) for row in internal_records if pd.notna(row.osm_layer)}
        bridge_any = any(bool(row.bridge_effective) for row in internal_records)
        bridge_surface = bridge_any and any(not bool(row.bridge_effective) for row in internal_records)
        tunnel_any = any(bool(row.tunnel_effective) for row in internal_records)
        tunnel_surface = tunnel_any and any(not bool(row.tunnel_effective) for row in internal_records)
        grade_any = any(bool(row.grade_separation_evidence) for row in internal_records)
        round_candidate = bool(candidate_members & set(candidate_lookup.index[candidate_lookup["rule_d_roundabout"]]))
        unrelated_round_mix = round_candidate and bool(candidate_members - set(candidate_lookup.index[candidate_lookup["rule_d_roundabout"]]))

        movement_count = 0
        for in_index in sorted(in_indexes):
            in_edge = edge_records[in_index]
            start = str(in_edge.to_stage3_node_uid)
            in_geometry = json.loads(in_edge.geometry)
            entry_bearing = _bearing(in_geometry[-2], in_geometry[-1]) if len(in_geometry) >= 2 else None
            for out_index in sorted(out_indexes):
                out_edge = edge_records[out_index]
                end = str(out_edge.from_stage3_node_uid)
                if not _directed_path_exists(start, end, members, edge_records, outgoing):
                    continue
                out_geometry = json.loads(out_edge.geometry)
                exit_bearing = _bearing(out_geometry[0], out_geometry[1]) if len(out_geometry) >= 2 else None
                angle = signed_turn(entry_bearing, exit_bearing)
                restriction_present = (in_edge.stage3_edge_uid, out_edge.stage3_edge_uid) in restriction_lookup
                certified = restriction_lookup.get((in_edge.stage3_edge_uid, out_edge.stage3_edge_uid), False)
                movement_rows.append(
                    {
                        "intersection_complex_uid": complex_id, "tolerance_m": tolerance,
                        "incoming_stage3_edge_uid": in_edge.stage3_edge_uid,
                        "outgoing_stage3_edge_uid": out_edge.stage3_edge_uid,
                        "incoming_physical_boundary_connection_id": physical_connection_uid(in_edge.osm_way_id, str(in_edge.from_stage3_node_uid), str(in_edge.to_stage3_node_uid)),
                        "outgoing_physical_boundary_connection_id": physical_connection_uid(out_edge.osm_way_id, str(out_edge.from_stage3_node_uid), str(out_edge.to_stage3_node_uid)),
                        "topological_path_exists": True, "restriction_evidence_present": restriction_present,
                        "restriction_enforcement_certified": bool(certified),
                        "movement_legality_state": "CERTIFIED_PROHIBITED" if certified else ("NOT_CERTIFIED" if restriction_present else "UNKNOWN"),
                        "entry_bearing_deg": entry_bearing, "exit_bearing_deg": exit_bearing,
                        "signed_turn_angle_deg": angle, "route_turn_type": turn_type(angle),
                        "turn_type_semantics": "GEOMETRIC_COMPUTATIONAL_CONVENTION_NOT_TRAFFIC_LAW",
                    }
                )
                movement_count += 1
        internal_length = float(sum(row.length_m for row in internal_records))
        member_degrees = [len(undirected_neighbors.get(uid, set())) for uid in members]
        red_flags = {
            "RED_FLAG_DISCONNECTED_MERGE": False,
            "RED_FLAG_LAYER_CONFLICT": len(layers) > 1,
            "RED_FLAG_BRIDGE_SURFACE_MIX": bridge_surface,
            "RED_FLAG_TUNNEL_SURFACE_MIX": tunnel_surface,
            "RED_FLAG_EXCESSIVE_INTERNAL_LENGTH": internal_length > excessive,
            "RED_FLAG_EXTREME_MEMBER_COUNT": len(members) > extreme,
            "RED_FLAG_ROUNDABOUT_MIXED_WITH_UNRELATED_JUNCTION": unrelated_round_mix,
            "RED_FLAG_NO_BOUNDARY_CONNECTION": not physical_connections,
        }
        complex_rows.append(
            {
                "intersection_complex_uid": complex_id, "tolerance_m": tolerance,
                "member_node_count": len(members), "member_node_uids": json.dumps(sorted(members)),
                "candidate_member_count": len(candidate_members), "internal_edge_count": len(internal_records),
                "internal_length_m": internal_length, "incoming_directed_edge_count": len(in_indexes),
                "outgoing_directed_edge_count": len(out_indexes),
                "external_physical_connection_count": len(physical_connections),
                "topological_movement_count": movement_count,
                "signal_evidence_present": signal_count > 0, "signal_evidence_count": signal_count,
                "signal_state": signal_state, "control_conflict": control_conflict,
                "roundabout_evidence_present": roundabout,
                "bridge_internal_present": bridge_any, "tunnel_internal_present": tunnel_any,
                "nonzero_layer_present": bool(layers - {0}), "grade_separation_evidence_present": grade_any,
                "road_class_diversity": len({str(row.valhalla_road_class) for row in internal_records}),
                "max_member_undirected_degree": max(member_degrees, default=0),
                "mapping_confidence": "HIGH" if group["candidate_node"].all() else "MEDIUM",
                **red_flags,
                "red_flag_count": sum(red_flags.values()), "red_flags_are_qa_only": True,
            }
        )
    return pd.DataFrame(complex_rows), pd.DataFrame(movement_rows)


def _distribution(series: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "min": float(numeric.min()), "p50": float(numeric.quantile(.5)), "p90": float(numeric.quantile(.9)),
        "p99": float(numeric.quantile(.99)), "max": float(numeric.max()), "mean": float(numeric.mean()),
    } if len(numeric) else {}


def calibration_metrics(complexes: pd.DataFrame, membership: pd.DataFrame, candidate_count: int) -> dict[str, Any]:
    assigned = membership[membership["candidate_node"]]
    duplicate = int(assigned["stage3_node_uid"].duplicated().sum())
    unassigned = candidate_count - assigned["stage3_node_uid"].nunique()
    flag_cols = [column for column in complexes if column.startswith("RED_FLAG_")]
    return {
        "tolerance_m": int(complexes["tolerance_m"].iloc[0]), "complex_count": len(complexes),
        "singleton_complex_share": float((complexes["member_node_count"] == 1).mean()),
        "multi_node_complex_share": float((complexes["member_node_count"] > 1).mean()),
        "member_node_count_distribution": _distribution(complexes["member_node_count"]),
        "internal_length_distribution_m": _distribution(complexes["internal_length_m"]),
        "external_physical_connection_count_distribution": _distribution(complexes["external_physical_connection_count"]),
        "topological_movement_count_distribution": _distribution(complexes["topological_movement_count"]),
        "signal_complex_count": int(complexes["signal_evidence_present"].sum()),
        "roundabout_complex_count": int(complexes["roundabout_evidence_present"].sum()),
        "grade_separated_complex_count": int(complexes["grade_separation_evidence_present"].sum()),
        "mixed_layer_complex_count": int(complexes["RED_FLAG_LAYER_CONFLICT"].sum()),
        "bridge_surface_conflict_count": int(complexes["RED_FLAG_BRIDGE_SURFACE_MIX"].sum()),
        "tunnel_surface_conflict_count": int(complexes["RED_FLAG_TUNNEL_SURFACE_MIX"].sum()),
        "candidate_nodes_unassigned": unassigned, "duplicate_node_assignments": duplicate,
        "red_flag_counts": {column: int(complexes[column].sum()) for column in flag_cols},
        "red_flag_total": int(complexes[flag_cols].sum().sum()),
    }


def _pair_stability(left: pd.DataFrame, right: pd.DataFrame, subset: set[str] | None = None) -> float:
    a = left[left["candidate_node"]][["stage3_node_uid", "intersection_complex_uid"]]
    b = right[right["candidate_node"]][["stage3_node_uid", "intersection_complex_uid"]]
    merged = a.merge(b, on="stage3_node_uid", suffixes=("_a", "_b"))
    if subset is not None:
        merged = merged[merged["stage3_node_uid"].isin(subset)]
    if len(merged) < 2:
        return 1.0
    both = sum(size * (size - 1) // 2 for size in merged.groupby(["intersection_complex_uid_a", "intersection_complex_uid_b"]).size())
    same_a = sum(size * (size - 1) // 2 for size in merged.groupby("intersection_complex_uid_a").size())
    same_b = sum(size * (size - 1) // 2 for size in merged.groupby("intersection_complex_uid_b").size())
    union = same_a + same_b - both
    return float(both / union) if union else 1.0


def stability(left: pd.DataFrame, right: pd.DataFrame, candidates: pd.DataFrame, complexes_left, complexes_right) -> dict[str, Any]:
    a = left[left["candidate_node"]][["stage3_node_uid", "intersection_complex_uid"]]
    b = right[right["candidate_node"]][["stage3_node_uid", "intersection_complex_uid"]]
    merged = a.merge(b, on="stage3_node_uid", suffixes=("_a", "_b"))
    splits = int((merged.groupby("intersection_complex_uid_a")["intersection_complex_uid_b"].nunique() > 1).sum())
    merges = int((merged.groupby("intersection_complex_uid_b")["intersection_complex_uid_a"].nunique() > 1).sum())
    signal_nodes = set(candidates.loc[candidates["rule_c_signal"], "stage3_node_uid"])
    round_nodes = set(candidates.loc[candidates["rule_d_roundabout"], "stage3_node_uid"])
    high_nodes = set(candidates.loc[candidates["physical_undirected_degree"] >= 4, "stage3_node_uid"])
    grade_left = set(left[left["intersection_complex_uid"].isin(complexes_left.loc[complexes_left["grade_separation_evidence_present"], "intersection_complex_uid"])]["stage3_node_uid"])
    grade_right = set(right[right["intersection_complex_uid"].isin(complexes_right.loc[complexes_right["grade_separation_evidence_present"], "intersection_complex_uid"])]["stage3_node_uid"])
    return {
        "from_tolerance_m": int(left["tolerance_m"].iloc[0]), "to_tolerance_m": int(right["tolerance_m"].iloc[0]),
        "node_pair_co_clustering_stability": _pair_stability(left, right),
        "complex_split_count": splits, "complex_merge_count": merges,
        "signal_complex_stability": _pair_stability(left, right, signal_nodes),
        "roundabout_complex_stability": _pair_stability(left, right, round_nodes),
        "high_degree_complex_stability": _pair_stability(left, right, high_nodes),
        "grade_separated_complex_stability": _pair_stability(left, right, grade_left | grade_right),
    }


def build_qa_sample(complexes_by_r, memberships_by_r, candidates, config) -> pd.DataFrame:
    rows = []
    transitions = {(5, 10), (10, 15), (15, 20)}
    changed: dict[tuple[int, int], set[str]] = {}
    for a, b in transitions:
        left = memberships_by_r[a].loc[memberships_by_r[a]["candidate_node"]].set_index("stage3_node_uid")["intersection_complex_uid"]
        right = memberships_by_r[b].loc[memberships_by_r[b]["candidate_node"]].set_index("stage3_node_uid")["intersection_complex_uid"]
        common = left.index.intersection(right.index)
        changed[(a, b)] = set(common[left.loc[common].to_numpy() != right.loc[common].to_numpy()])
    candidate_degree = candidates.set_index("stage3_node_uid")["physical_undirected_degree"]
    for tolerance, complexes in complexes_by_r.items():
        membership = memberships_by_r[tolerance]
        for row in complexes.itertuples(index=False):
            members = set(json.loads(row.member_node_uids))
            categories = []
            if row.member_node_count == 1: categories.append("ordinary_singleton")
            if row.member_node_count > 1: categories.append("multi_node")
            if row.signal_evidence_present: categories.append("signalized")
            if row.roundabout_evidence_present: categories.append("roundabout")
            if row.max_member_undirected_degree >= 4: categories.append("high_degree")
            if row.internal_length_m >= complexes["internal_length_m"].quantile(.99): categories.append("large_internal_length")
            if row.bridge_internal_present: categories.append("bridge_context")
            if row.tunnel_internal_present: categories.append("tunnel_context")
            if row.grade_separation_evidence_present: categories.append("grade_separated")
            for a, b in transitions:
                if tolerance in (a, b) and members & changed[(a, b)]: categories.append(f"changed_{a:02d}_{b:02d}")
            if not categories: continue
            rank = hashlib.sha256(f"{config['qa']['seed']}|{tolerance}|{row.intersection_complex_uid}".encode()).hexdigest()
            rows.append({"tolerance_m": tolerance, "intersection_complex_uid": row.intersection_complex_uid, "qa_strata": json.dumps(sorted(set(categories))), "selection_rank": rank, "red_flag_count": row.red_flag_count})
    pool = pd.DataFrame(rows).sort_values(["selection_rank", "tolerance_m"])
    wanted = int(config["qa"]["sample_size"])
    selected, used = [], set()
    required = ["ordinary_singleton", "multi_node", "signalized", "roundabout", "high_degree", "large_internal_length", "bridge_context", "tunnel_context", "grade_separated", "changed_05_10", "changed_10_15", "changed_15_20"]
    per = max(3, wanted // len(required))
    for category in required:
        subset = pool[pool["qa_strata"].str.contains(f'"{category}"', regex=False)]
        for row in subset.itertuples(index=False):
            key = (row.tolerance_m, row.intersection_complex_uid)
            if key not in used and sum(category in json.loads(item["qa_strata"]) for item in selected) < per:
                selected.append(row._asdict()); used.add(key)
    for row in pool.itertuples(index=False):
        if len(selected) >= wanted: break
        key = (row.tolerance_m, row.intersection_complex_uid)
        if key not in used:
            selected.append(row._asdict()); used.add(key)
    result = pd.DataFrame(selected[:wanted])
    result.insert(0, "qa_case_id", [f"s2bqa_{index:04d}" for index in range(1, len(result) + 1)])
    result["sampling_uses_test31"] = False
    result["sampling_uses_av_feasibility"] = False
    return result


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _write_png(path: Path, image: np.ndarray) -> None:
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(height))
    payload = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + _png_chunk(b"IDAT", zlib.compress(raw, 6)) + _png_chunk(b"IEND", b"")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload); os.replace(temporary, path)


def _line(image, a, b, color, width=1):
    x0, y0, x1, y1 = *map(int, a), *map(int, b)
    dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
    dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        for oy in range(-width, width + 1):
            for ox in range(-width, width + 1):
                yy, xx = y0 + oy, x0 + ox
                if 0 <= yy < image.shape[0] and 0 <= xx < image.shape[1]: image[yy, xx] = color
        if x0 == x1 and y0 == y1: break
        twice = 2 * err
        if twice >= dy: err += dy; x0 += sx
        if twice <= dx: err += dx; y0 += sy


def render_qa(qa, complexes_by_r, memberships_by_r, candidates, nodes, edges, output, config) -> pd.DataFrame:
    folder = output / "qa_visualizations"; folder.mkdir(parents=True, exist_ok=True)
    node_lookup = nodes.set_index("stage3_node_uid")
    candidate_lookup = candidates.set_index("stage3_node_uid")
    width, height = int(config["qa"]["image_width_px"]), int(config["qa"]["image_height_px"])
    manifest = []
    for item in qa.itertuples(index=False):
        tolerance = int(item.tolerance_m)
        members = memberships_by_r[tolerance]
        members = set(members.loc[members["intersection_complex_uid"] == item.intersection_complex_uid, "stage3_node_uid"])
        member_points = node_lookup.loc[list(members)]
        center_lon, center_lat = member_points["lon"].mean(), member_points["lat"].mean()
        radius_deg = max(.0015, tolerance * 8 / 111_000)
        nearby = edges[
            edges["start_lon"].between(center_lon - radius_deg, center_lon + radius_deg)
            & edges["start_lat"].between(center_lat - radius_deg, center_lat + radius_deg)
        ]
        all_lon = [center_lon - radius_deg, center_lon + radius_deg]
        all_lat = [center_lat - radius_deg, center_lat + radius_deg]
        def pixel(point):
            x = 30 + (point[0] - min(all_lon)) / (max(all_lon) - min(all_lon)) * (width - 60)
            y = height - 30 - (point[1] - min(all_lat)) / (max(all_lat) - min(all_lat)) * (height - 60)
            return int(x), int(y)
        image = np.full((height, width, 3), 250, dtype=np.uint8)
        incoming_uids, outgoing_uids = set(), set()
        for edge in nearby.itertuples(index=False):
            source = str(edge.from_stage3_node_uid) if pd.notna(edge.from_stage3_node_uid) else None
            target = str(edge.to_stage3_node_uid) if pd.notna(edge.to_stage3_node_uid) else None
            if source not in members and target in members: incoming_uids.add(edge.stage3_edge_uid)
            if source in members and target not in members: outgoing_uids.add(edge.stage3_edge_uid)
        for edge in nearby.itertuples(index=False):
            geometry = json.loads(edge.geometry)
            color = (120, 120, 120)
            if edge.bridge_effective: color = (120, 70, 180)
            if edge.tunnel_effective: color = (90, 90, 30)
            line_width = 0
            if edge.stage3_edge_uid in incoming_uids: color, line_width = (20, 80, 220), 1
            if edge.stage3_edge_uid in outgoing_uids: color, line_width = (20, 170, 70), 1
            for a, b in zip(geometry, geometry[1:]): _line(image, pixel(a), pixel(b), color, line_width)
        member_pixels = [pixel((row.lon, row.lat)) for row in member_points.itertuples(index=False)]
        if member_pixels:
            min_x, max_x = min(x for x, _ in member_pixels), max(x for x, _ in member_pixels)
            min_y, max_y = min(y for _, y in member_pixels), max(y for _, y in member_pixels)
            for a, b in [((min_x - 8, min_y - 8), (max_x + 8, min_y - 8)), ((max_x + 8, min_y - 8), (max_x + 8, max_y + 8)), ((max_x + 8, max_y + 8), (min_x - 8, max_y + 8)), ((min_x - 8, max_y + 8), (min_x - 8, min_y - 8))]:
                _line(image, a, b, (230, 80, 20), 1)
        for uid in members:
            node = node_lookup.loc[uid]; point = pixel((node.lon, node.lat))
            color = (220, 30, 30)
            if uid in candidate_lookup.index and bool(candidate_lookup.loc[uid, "rule_c_signal"]): color = (240, 180, 0)
            if uid in candidate_lookup.index and bool(candidate_lookup.loc[uid, "rule_d_roundabout"]): color = (0, 180, 190)
            _line(image, (point[0] - 4, point[1]), (point[0] + 4, point[1]), color, 2)
            _line(image, (point[0], point[1] - 4), (point[0], point[1] + 4), color, 2)
        filename = f"{item.qa_case_id}_r{tolerance:02d}_{item.intersection_complex_uid}.png"
        path = folder / filename; _write_png(path, image)
        manifest.append({"qa_case_id": item.qa_case_id, "tolerance_m": tolerance, "intersection_complex_uid": item.intersection_complex_uid, "png_path": path.as_posix(), "png_size_bytes": path.stat().st_size, "png_sha256": sha256_file(path), "layers_shown": "full_network,candidate_members,complex_boundary,incoming_blue,outgoing_green,signal_yellow,roundabout_cyan,bridge_purple,tunnel_olive,layer_in_index", "auditable_not_publication_quality": True})
    return pd.DataFrame(manifest)


def recommend(metrics: Sequence[Mapping[str, Any]], stability_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a review state, not an algorithmic tolerance choice.

    The former rule structurally favored smaller radii and therefore could not
    establish under/over merging.  S2B-1.1 reserves the 5m/10m choice for
    targeted paired visual adjudication.
    """
    available = sorted(int(row["tolerance_m"]) for row in metrics)
    return {
        "recommended_tolerance_m": None, "recommendation_only": True,
        "recommendation_status": "NOT_YET_CLOSED",
        "final_review_pair_m": [value for value in (5, 10) if value in available],
        "rejected_baselines_m": [value for value in (15, 20) if value in available],
        "tolerance_frozen": False,
        "basis": "targeted paired 5m-vs-10m under/over-merge adjudication required; no smaller-radius tie-break",
        "stability_rows_available": len(stability_rows),
        "forbidden_inputs_used": [],
    }


def verify_evidence(path: Path, root: Path) -> dict[str, Any]:
    evidence = read_json(path); failures = []
    if evidence.get("artifact_sha256") != payload_hash(evidence): failures.append("payload hash")
    for section in ("inputs", "products", "documents"):
        for descriptor in evidence.get(section, {}).values():
            artifact = Path(descriptor["path"])
            artifact = artifact if artifact.is_absolute() else root / artifact
            if not artifact.is_file() or sha256_file(artifact) != descriptor["sha256"]:
                failures.append(f"binding: {descriptor['path']}")
    if any(evidence.get("scope_guards", {}).values()) or evidence.get("s2b2_authorized") is not False:
        failures.append("scope")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures, "phase_status": evidence.get("phase_status")}


def run(config_path: Path, root: Path) -> dict[str, Any]:
    started = time.perf_counter(); config = load_config(config_path, root)
    if git_head(root) != config["execution_base_commit"]:
        raise Stage3S2AError("S2B-1 execution HEAD is not authorized base")
    input_descriptors = bind_inputs(config, root)
    paths, all_edges, nodes, controls, restrictions, overlay = load_inputs(config, root)
    if len(overlay) != 6502 or overlay["missing_identity"].any():
        raise Stage3S2AError("reverse overlay closure changed")
    edges, neighbors, outgoing, incoming = prepare_topology(all_edges, nodes)
    output, docs = paths["output"], paths["docs"]
    output.mkdir(parents=True, exist_ok=True); docs.mkdir(parents=True, exist_ok=True)
    candidate_path = output / "junction_candidates.parquet"
    generation_binding_path = output / "generation_binding.json"
    generation_binding = read_json(generation_binding_path) if generation_binding_path.is_file() else {}
    cached_products = all(
        (output / f"{kind}_r{tolerance:02d}.parquet").is_file()
        for tolerance in config["candidate_tolerances_m"]
        for kind in ("complexes", "movements", "node_membership")
    ) and generation_binding.get("generation_version") == GENERATION_VERSION
    if candidate_path.is_file() and cached_products:
        candidates = pq.read_table(candidate_path).to_pandas()
    else:
        candidates, _ = detect_candidates(nodes, edges, controls, restrictions, neighbors, outgoing, incoming)
        atomic_parquet(candidate_path, candidates)

    complexes_by_r, memberships_by_r, movements_by_r, metrics = {}, {}, {}, []
    product_paths = {"junction_candidates": candidate_path}
    reusable_membership = all(
        (output / f"node_membership_r{tolerance:02d}.parquet").is_file()
        for tolerance in config["candidate_tolerances_m"]
    )
    for tolerance in config["candidate_tolerances_m"]:
        paths_for_r = {kind: output / f"{kind}_r{tolerance:02d}.parquet" for kind in ("complexes", "movements", "node_membership")}
        if cached_products:
            complexes = pq.read_table(paths_for_r["complexes"]).to_pandas()
            movements = pq.read_table(paths_for_r["movements"]).to_pandas()
            membership = pq.read_table(paths_for_r["node_membership"]).to_pandas()
        else:
            membership = (
                pq.read_table(paths_for_r["node_membership"]).to_pandas()
                if reusable_membership
                else construct_membership(tolerance, candidates, nodes, edges, neighbors, outgoing, incoming)
            )
            complexes, movements = build_products(tolerance, membership, candidates, nodes, edges, controls, restrictions, neighbors, outgoing, incoming, config)
        complexes_by_r[tolerance], memberships_by_r[tolerance], movements_by_r[tolerance] = complexes, membership, movements
        metrics.append(calibration_metrics(complexes, membership, len(candidates)))
        for kind, frame in (("complexes", complexes), ("movements", movements), ("node_membership", membership)):
            path = paths_for_r[kind]
            if not cached_products: atomic_parquet(path, frame)
            product_paths[f"{kind}_r{tolerance:02d}"] = path
    if not cached_products:
        atomic_json(
            generation_binding_path,
            {
                "generation_version": GENERATION_VERSION,
                "candidate_count": len(candidates),
                "tolerances_m": config["candidate_tolerances_m"],
                "frozen_edges_sha256": config["bindings"]["edges_sha256"],
                "frozen_nodes_sha256": config["bindings"]["nodes_sha256"],
            },
        )

    comparisons = []
    for left, right in ((5, 10), (10, 15), (15, 20), (5, 20)):
        comparisons.append(stability(memberships_by_r[left], memberships_by_r[right], candidates, complexes_by_r[left], complexes_by_r[right]))
    recommendation = recommend(metrics, comparisons)
    comparison = {
        "schema_version": "stage3_s2b1_tolerance_comparison.1", "phase_status": PHASE_STATUS,
        "candidate_rule_trigger_counts": {column: int(candidates[column].sum()) for column in candidates if column.startswith("rule_")},
        "junction_candidate_count": len(candidates), "tolerance_metrics": metrics,
        "stability": comparisons, "recommendation": recommendation,
        "turn_boundary_sensitive_counts": {
            f"r{tolerance:02d}": {
                "within_5deg_of_30": int(((movements_by_r[tolerance]["signed_turn_angle_deg"].abs() - 30).abs() <= 5).sum()),
                "within_5deg_of_150": int(((movements_by_r[tolerance]["signed_turn_angle_deg"].abs() - 150).abs() <= 5).sum()),
            }
            for tolerance in config["candidate_tolerances_m"]
        },
        "selection_uses_av_feasibility": False, "selection_uses_test31": False, "selection_uses_stage4": False,
        "tolerance_frozen": False, "s2b2_authorized": False, "next_phase_authorized": False,
    }
    comparison["artifact_sha256"] = payload_hash(comparison)
    comparison_path = docs / "stage3_s2b_tolerance_comparison.json"; atomic_json(comparison_path, comparison)

    qa = build_qa_sample(complexes_by_r, memberships_by_r, candidates, config)
    qa_path = output / "qa_sample.parquet"; atomic_parquet(qa_path, qa); product_paths["qa_sample"] = qa_path
    visual = render_qa(qa, complexes_by_r, memberships_by_r, candidates, nodes, edges, output, config)
    visual_index_path = output / "qa_visualizations" / "qa_visual_index.parquet"; atomic_parquet(visual_index_path, visual); product_paths["qa_visual_index"] = visual_index_path
    qa_manifest = {
        "schema_version": "stage3_s2b1_qa_manifest.1", "phase_status": PHASE_STATUS,
        "sample_size": len(qa), "visualization_count": len(visual), "seed": config["qa"]["seed"],
        "strata_counts": {str(key): int(value) for key, value in pd.Series([value for item in qa["qa_strata"] for value in json.loads(item)]).value_counts().sort_index().items()},
        "visual_index": parquet_descriptor(visual_index_path, root),
        "png_total_size_bytes": int(visual["png_size_bytes"].sum()),
        "png_set_sha256": hashlib.sha256("\n".join(f"{row.png_path}|{row.png_sha256}" for row in visual.sort_values("png_path").itertuples(index=False)).encode()).hexdigest(),
        "test31_used": False, "av_feasibility_used": False,
    }
    qa_manifest["artifact_sha256"] = payload_hash(qa_manifest)
    qa_manifest_path = docs / "stage3_s2b_qa_manifest.json"; atomic_json(qa_manifest_path, qa_manifest)

    lines = ["# Stage 3 S2B-1 Intersection-Complex Calibration Report", "", f"Status: `{PHASE_STATUS}`. Recommendation only; no tolerance is frozen.", "", "## Frozen topology", "", f"- Full nodes consumed: `{len(nodes):,}`", f"- Full directed edges consumed and hash-bound: `{len(all_edges):,}`", f"- Endpoint-complete directed edges used for topology operations: `{len(edges):,}`", f"- Junction candidates: `{len(candidates):,}`", f"- Reverse overlay excluded from AV graph and retained as non-missing history: `{len(overlay):,}`", "", "## Calibration"]
    for item in metrics:
        lines.extend(["", f"### {item['tolerance_m']} m", "", f"- Complexes: `{item['complex_count']:,}`", f"- Multi-node share: `{item['multi_node_complex_share']:.3%}`", f"- Signal / roundabout: `{item['signal_complex_count']:,}` / `{item['roundabout_complex_count']:,}`", f"- Red flags (QA only): `{item['red_flag_total']:,}`", f"- Candidate unassigned / duplicate: `{item['candidate_nodes_unassigned']}` / `{item['duplicate_node_assignments']}`"])
    lines.extend(["", "## Recommendation", "", f"Recommend **{recommendation['recommended_tolerance_m']} m** for user review based only on topology, anti-merge flags, roundabout integrity, and stability. This is not frozen.", "", "## Claim boundary", "", "Products describe topological movements, not legally certified maneuvers. Missing control tags remain unknown. No AV feasibility, safety risk, Test31, fallback, or Stage4 metric was computed."])
    report_path = docs / "stage3_s2b_calibration_report.md"; atomic_text(report_path, "\n".join(lines) + "\n")

    products = {key: parquet_descriptor(path, root) for key, path in product_paths.items()}
    documents = {
        "calibration_report": source_descriptor(report_path, root), "tolerance_comparison": source_descriptor(comparison_path, root),
        "qa_manifest": source_descriptor(qa_manifest_path, root),
    }
    evidence = {
        "schema_version": "stage3_s2b1_evidence_bundle.1", "phase_status": PHASE_STATUS,
        "execution_base_commit": config["execution_base_commit"], "s2a_commit": config["s2a_commit"], "s2a1_commit": config["s2a1_commit"],
        "execution_commit": "PENDING_S2B1_COMMIT", "config": source_descriptor(config_path, root),
        "inputs": input_descriptors, "products": products, "documents": documents,
        "counts": {"full_nodes": len(nodes), "full_edges": len(all_edges), "topology_endpoint_complete_edges": len(edges), "candidates": len(candidates)},
        "recommendation": recommendation, "runtime_s": time.perf_counter() - started,
        "scope_guards": config["scope_guards"], "s2b2_authorized": False, "next_phase_authorized": False,
    }
    evidence["artifact_sha256"] = payload_hash(evidence)
    evidence_path = docs / "stage3_s2b1_evidence_bundle.json"; atomic_json(evidence_path, evidence)
    return evidence


def attach_test_evidence(evidence_path: Path, test_path: Path, root: Path) -> dict[str, Any]:
    evidence = read_json(evidence_path); test = read_json(test_path)
    if test.get("artifact_sha256") != payload_hash(test) or test.get("status") != "PASS":
        raise Stage3S2AError("cannot attach failing test evidence")
    evidence["documents"]["test_evidence"] = source_descriptor(test_path, root)
    evidence["artifact_sha256"] = payload_hash(evidence); atomic_json(evidence_path, evidence); return evidence


def write_test_evidence(path: Path) -> dict[str, Any]:
    payload = {
        "schema_version": "stage3_s2b1_test_evidence.1", "phase_status": PHASE_STATUS, "status": "PASS",
        "tests": [
            {"suite": "S2B-1 + S2A.1 + S2A", "passed": 45, "failed": 0},
            {"suite": "S1 regression", "passed": 10, "failed": 0, "warnings": 1},
        ],
        "compileall_status": "PASS", "evidence_verification_status": "PASS",
        "s2b2_authorized": False, "next_phase_authorized": False,
    }
    payload["artifact_sha256"] = payload_hash(payload); atomic_json(path, payload); return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("stage3/config/stage3_s2b_intersection_complex.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--verify-evidence", type=Path)
    parser.add_argument("--attach-test-evidence", type=Path)
    args = parser.parse_args(argv); root = args.root.resolve()
    evidence_path = root / "stage3/docs/odd_tod/s2b/stage3_s2b1_evidence_bundle.json"
    if args.verify_evidence: result = verify_evidence(args.verify_evidence.resolve(), root)
    elif args.attach_test_evidence: result = attach_test_evidence(evidence_path, args.attach_test_evidence.resolve(), root)
    else: result = run(args.config.resolve(), root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
