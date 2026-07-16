"""Build a versioned, intersection-noded Stage0 network without overwriting v2."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage0.canonical.noding import (
    cluster_endpoints_by_level,
    grade_transition_connector_eligible,
    intersection_points,
    parse_bool,
    split_line_at_points,
    topology_level,
    topology_levels_compatible,
)
from stage0.canonical.topology import allows_forward, allows_reverse
from canonical_pipeline.manifest import sha256_file


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--endpoint-snap-tolerance-m", type=float, default=0.75)
    parser.add_argument("--interior-tolerance-m", type=float, default=0.05)
    parser.add_argument("--network-version", default="xian_2017_core_noded_v4")
    parser.add_argument("--cross-level-connector-max-angle-deg", type=float, default=45.0)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def bool_value(value: object) -> bool:
    return parse_bool(value)


def graph_metrics(roads: pd.DataFrame) -> dict:
    graph = nx.DiGraph()
    for row in roads.itertuples(index=False):
        if allows_forward(row.oneway_code):
            graph.add_edge(int(row.from_node), int(row.to_node))
        if allows_reverse(row.oneway_code):
            graph.add_edge(int(row.to_node), int(row.from_node))
    weak = list(nx.weakly_connected_components(graph))
    strong = list(nx.strongly_connected_components(graph))
    largest_weak = max(weak, key=len, default=set())
    in_largest = roads.from_node.isin(largest_weak) & roads.to_node.isin(largest_weak)
    return {
        "nodes": graph.number_of_nodes(),
        "directed_edges": graph.number_of_edges(),
        "weak_components": len(weak),
        "strong_components": len(strong),
        "largest_weak_component_nodes": max(map(len, weak), default=0),
        "largest_weak_component_share": max(map(len, weak), default=0) / max(1, graph.number_of_nodes()),
        "largest_weak_component_links": int(in_largest.sum()),
        "largest_weak_component_link_share": float(in_largest.mean()),
        "largest_weak_component_length_m": float(roads.loc[in_largest, "length_m"].sum()),
        "largest_weak_component_length_share": float(
            roads.loc[in_largest, "length_m"].sum() / max(float(roads.length_m.sum()), 1.0)
        ),
    }


def main() -> None:
    args = arguments()
    source = gpd.read_parquet(args.roads).to_crs(32649).reset_index(drop=True)
    source["_level"] = [
        topology_level(
            getattr(row, "layer", None),
            bool_value(getattr(row, "bridge", None)),
            bool_value(getattr(row, "tunnel", None)),
        )
        for row in source.itertuples(index=False)
    ]
    split_points: dict[int, list] = defaultdict(list)
    intersection_pairs = 0
    grade_separated_pairs = 0
    overlap_pairs = 0
    pair_index = source.sindex.query_bulk(source.geometry, predicate="intersects")
    for left_index, right_index in zip(pair_index[0], pair_index[1]):
        left_index, right_index = int(left_index), int(right_index)
        if left_index >= right_index:
            continue
        if not topology_levels_compatible(
            source.at[left_index, "_level"], source.at[right_index, "_level"]
        ):
            grade_separated_pairs += 1
            continue
        intersection = source.geometry.iloc[left_index].intersection(source.geometry.iloc[right_index])
        points = intersection_points(intersection)
        if not points:
            if not intersection.is_empty:
                overlap_pairs += 1
            continue
        intersection_pairs += 1
        split_points[left_index].extend(points)
        split_points[right_index].extend(points)

    segment_rows = []
    split_source_links = 0
    for source_index, row in source.iterrows():
        pieces = split_line_at_points(
            row.geometry,
            split_points.get(int(source_index), []),
            endpoint_tolerance_m=args.interior_tolerance_m,
        )
        split_source_links += int(len(pieces) > 1)
        level = tuple(row["_level"])
        base = row.drop(labels=["geometry", "_level"]).to_dict()
        for segment_index, geometry in enumerate(pieces):
            segment_rows.append({
                **base,
                "source_link_id": str(row.link_id),
                "segment_index": segment_index,
                "topology_layer": level[0],
                "topology_bridge": level[1],
                "topology_tunnel": level[2],
                "geometry": geometry,
            })
    segments = gpd.GeoDataFrame(segment_rows, geometry="geometry", crs=source.crs)

    endpoints = []
    endpoint_directions = []
    endpoint_segment_indices = []
    for segment_index, line in enumerate(segments.geometry):
        coordinates = list(line.coords)
        endpoints.append(coordinates[0])
        endpoints.append(coordinates[-1])
        endpoint_directions.append(np.asarray(coordinates[1]) - np.asarray(coordinates[0]))
        endpoint_directions.append(np.asarray(coordinates[-2]) - np.asarray(coordinates[-1]))
        endpoint_segment_indices.extend([segment_index, segment_index])
    endpoint_array = np.asarray(endpoints, dtype=float)
    endpoint_levels = [
        level
        for row in segments.itertuples(index=False)
        for level in [(row.topology_layer, bool(row.topology_bridge), bool(row.topology_tunnel))] * 2
    ]
    endpoint_nodes, representatives, representative_levels = cluster_endpoints_by_level(
        endpoint_array, endpoint_levels, args.endpoint_snap_tolerance_m
    )
    from_nodes = endpoint_nodes[0::2]
    to_nodes = endpoint_nodes[1::2]
    snapped_geometry = []
    for index, line in enumerate(segments.geometry):
        coords = list(line.coords)
        coords[0] = tuple(representatives[from_nodes[index]])
        coords[-1] = tuple(representatives[to_nodes[index]])
        snapped_geometry.append(LineString(coords))
    segments["geometry"] = snapped_geometry
    segments["from_node"] = from_nodes
    segments["to_node"] = to_nodes
    segments["length_m"] = segments.geometry.length
    segments["link_id"] = [
        f"{source_link_id}__n{segment_index}"
        for source_link_id, segment_index in zip(segments.source_link_id, segments.segment_index)
    ]
    segments["topology_version"] = args.network_version
    segments["topology_connector"] = False
    segments["candidate_eligible"] = True
    segments = segments.drop(columns=[column for column in ["_level"] if column in segments])

    connector_pairs: set[tuple[int, int]] = set()
    cross_level_endpoint_pairs = 0
    rejected_connector_pairs = 0
    for left_endpoint, right_endpoint in cKDTree(endpoint_array).query_pairs(
        args.endpoint_snap_tolerance_m
    ):
        left_endpoint, right_endpoint = int(left_endpoint), int(right_endpoint)
        left_segment = int(endpoint_segment_indices[left_endpoint])
        right_segment = int(endpoint_segment_indices[right_endpoint])
        if left_segment == right_segment:
            continue
        left_level, right_level = endpoint_levels[left_endpoint], endpoint_levels[right_endpoint]
        if topology_levels_compatible(left_level, right_level):
            continue
        cross_level_endpoint_pairs += 1
        left_row, right_row = segments.iloc[left_segment], segments.iloc[right_segment]
        eligible = grade_transition_connector_eligible(
            left_level,
            right_level,
            endpoint_directions[left_endpoint],
            endpoint_directions[right_endpoint],
            left_row.road_class,
            right_row.road_class,
            left_row.road_name,
            right_row.road_name,
            getattr(left_row, "ref", None),
            getattr(right_row, "ref", None),
            args.cross_level_connector_max_angle_deg,
        )
        left_node, right_node = int(endpoint_nodes[left_endpoint]), int(endpoint_nodes[right_endpoint])
        node_pair = tuple(sorted((left_node, right_node)))
        if not eligible or left_node == right_node or node_pair in connector_pairs:
            rejected_connector_pairs += 1
            continue
        connector_pairs.add(node_pair)

    if connector_pairs:
        connector_rows = []
        template = {column: None for column in segments.columns}
        for connector_index, (left_node, right_node) in enumerate(sorted(connector_pairs)):
            geometry = LineString([representatives[left_node], representatives[right_node]])
            connector_rows.append({
                **template,
                "link_id": f"grade_transition_connector__{connector_index}",
                "source_link_id": "grade_transition_connector",
                "segment_index": connector_index,
                "from_node": left_node,
                "to_node": right_node,
                "oneway_code": "B",
                "oneway_direction": "both",
                "road_class": "topology_connector",
                "road_name": "grade_transition_connector",
                "length_m": float(geometry.length),
                "topology_layer": "transition",
                "topology_bridge": False,
                "topology_tunnel": False,
                "topology_version": args.network_version,
                "topology_connector": True,
                "candidate_eligible": False,
                "geometry": geometry,
            })
        segments = pd.concat(
            [segments, gpd.GeoDataFrame(connector_rows, geometry="geometry", crs=segments.crs)],
            ignore_index=True,
        )
        segments = gpd.GeoDataFrame(segments, geometry="geometry", crs=source.crs)

    degree_graph = nx.Graph()
    degree_graph.add_edges_from(zip(segments.from_node.astype(int), segments.to_node.astype(int)))
    degree = dict(degree_graph.degree())
    nodes = gpd.GeoDataFrame(
        {
            "node_id": np.arange(len(representatives), dtype="int64"),
            "street_degree": [degree.get(index, 0) for index in range(len(representatives))],
            "topology_layer": [level[0] for level in representative_levels],
            "topology_bridge": [level[1] for level in representative_levels],
            "topology_tunnel": [level[2] for level in representative_levels],
            "topology_version": args.network_version,
        },
        geometry=gpd.points_from_xy(representatives[:, 0], representatives[:, 1]),
        crs=segments.crs,
    )
    nodes["intersection_type"] = np.where(nodes.street_degree >= 3, "intersection", "none")
    segments_wgs = segments.to_crs(4326)
    nodes_wgs = nodes.to_crs(4326)
    nodes_wgs["lon"] = nodes_wgs.geometry.x
    nodes_wgs["lat"] = nodes_wgs.geometry.y

    args.output_root.mkdir(parents=True, exist_ok=True)
    roads_path = args.output_root / "roads.parquet"
    nodes_path = args.output_root / "nodes.parquet"
    segments_wgs.to_parquet(roads_path, index=False)
    nodes_wgs.to_parquet(nodes_path, index=False)
    original_metrics = graph_metrics(source)
    noded_metrics = graph_metrics(segments)
    audit_path = args.audit or args.output_root / "network_audit.json"
    manifest_path = args.manifest or args.output_root / "network_manifest.json"
    audit = {
        "status": "DIAGNOSTIC_PASS",
        "network_version": args.network_version,
        "source_network": args.roads.as_posix(),
        "source_links": int(len(source)),
        "noded_links": int(len(segments)),
        "split_source_links": split_source_links,
        "same_level_intersection_pairs": intersection_pairs,
        "grade_separated_pairs_not_noded": grade_separated_pairs,
        "overlap_pairs_not_automatically_merged": overlap_pairs,
        "cross_level_endpoint_pairs": cross_level_endpoint_pairs,
        "grade_transition_connectors": len(connector_pairs),
        "rejected_cross_level_connector_pairs": rejected_connector_pairs,
        "cross_level_connector_max_angle_deg": args.cross_level_connector_max_angle_deg,
        "endpoint_snap_tolerance_m": args.endpoint_snap_tolerance_m,
        "original_graph": original_metrics,
        "noded_graph": noded_metrics,
        "outputs": {"roads": roads_path.as_posix(), "nodes": nodes_path.as_posix()},
        "canonical_promotion_gate": "HOLD",
        "promotion_requirements": [
            "Rematch the fixed train/validation/test chain against this version.",
            "Run route-quality and manual-truth audits.",
            "Verify grade-separated and overlapping-road exceptions.",
        ],
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "artifact_version": args.network_version,
        "status": "exploratory",
        "audit_status": "DIAGNOSTIC_PASS",
        "created_by_commit": commit,
        "registered_by_commit": commit,
        "source_network": args.roads.as_posix(),
        "source_network_sha256": sha256_file(args.roads),
        "endpoint_snap_tolerance_m": args.endpoint_snap_tolerance_m,
        "interior_tolerance_m": args.interior_tolerance_m,
        "files": {
            "roads": {"path": roads_path.as_posix(), "sha256": sha256_file(roads_path)},
            "nodes": {"path": nodes_path.as_posix(), "sha256": sha256_file(nodes_path)},
            "audit": {"path": audit_path.as_posix(), "sha256": sha256_file(audit_path)},
        },
        "canonical_promotion_gate": "HOLD",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
