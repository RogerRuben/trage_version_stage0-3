"""Build a versioned, intersection-noded Stage0 network without overwriting v2."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage0.canonical.noding import (
    cluster_endpoints,
    intersection_points,
    split_line_at_points,
    topology_level,
)
from stage0.canonical.topology import allows_forward, allows_reverse


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--endpoint-snap-tolerance-m", type=float, default=0.75)
    parser.add_argument("--interior-tolerance-m", type=float, default=0.05)
    parser.add_argument("--audit", type=Path, required=True)
    return parser.parse_args()


def bool_value(value: object) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def graph_metrics(roads: pd.DataFrame) -> dict:
    graph = nx.DiGraph()
    for row in roads.itertuples(index=False):
        if allows_forward(row.oneway_code):
            graph.add_edge(int(row.from_node), int(row.to_node))
        if allows_reverse(row.oneway_code):
            graph.add_edge(int(row.to_node), int(row.from_node))
    weak = list(nx.weakly_connected_components(graph))
    strong = list(nx.strongly_connected_components(graph))
    return {
        "nodes": graph.number_of_nodes(),
        "directed_edges": graph.number_of_edges(),
        "weak_components": len(weak),
        "strong_components": len(strong),
        "largest_weak_component_nodes": max(map(len, weak), default=0),
        "largest_weak_component_share": max(map(len, weak), default=0) / max(1, graph.number_of_nodes()),
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
        if source.at[left_index, "_level"] != source.at[right_index, "_level"]:
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
        base = row.drop(labels=["geometry", "_level"]).to_dict()
        for segment_index, geometry in enumerate(pieces):
            segment_rows.append({
                **base,
                "source_link_id": str(row.link_id),
                "segment_index": segment_index,
                "geometry": geometry,
            })
    segments = gpd.GeoDataFrame(segment_rows, geometry="geometry", crs=source.crs)

    endpoints = []
    for line in segments.geometry:
        endpoints.append(line.coords[0])
        endpoints.append(line.coords[-1])
    endpoint_array = np.asarray(endpoints, dtype=float)
    endpoint_nodes, representatives = cluster_endpoints(endpoint_array, args.endpoint_snap_tolerance_m)
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
    segments["topology_version"] = "xian_2017_core_noded_v3"
    segments = segments.drop(columns=[column for column in ["_level"] if column in segments])

    degree_graph = nx.Graph()
    degree_graph.add_edges_from(zip(segments.from_node.astype(int), segments.to_node.astype(int)))
    degree = dict(degree_graph.degree())
    nodes = gpd.GeoDataFrame(
        {
            "node_id": np.arange(len(representatives), dtype="int64"),
            "street_degree": [degree.get(index, 0) for index in range(len(representatives))],
            "topology_version": "xian_2017_core_noded_v3",
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
    roads_path = args.output_root / "xian_2017_core_noded_v3_roads.parquet"
    nodes_path = args.output_root / "xian_2017_core_noded_v3_nodes.parquet"
    segments_wgs.to_parquet(roads_path, index=False)
    nodes_wgs.to_parquet(nodes_path, index=False)
    original_metrics = graph_metrics(source)
    noded_metrics = graph_metrics(segments)
    audit = {
        "status": "DIAGNOSTIC_PASS",
        "network_version": "xian_2017_core_noded_v3",
        "source_network": args.roads.as_posix(),
        "source_links": int(len(source)),
        "noded_links": int(len(segments)),
        "split_source_links": split_source_links,
        "same_level_intersection_pairs": intersection_pairs,
        "grade_separated_pairs_not_noded": grade_separated_pairs,
        "overlap_pairs_not_automatically_merged": overlap_pairs,
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
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

