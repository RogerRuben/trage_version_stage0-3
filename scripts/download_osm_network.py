"""Download the OSM drive network covering the Stage0 Xi'an sample."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--margin-deg", type=float, default=0.018)
    parser.add_argument(
        "--region",
        choices=["fixed-xian-core", "sample"],
        default="fixed-xian-core",
        help="Use the public fixed Xi'an core bbox by default; sample mode derives a private bbox.",
    )
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def first(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


def parse_number(value):
    value = first(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.region == "fixed-xian-core":
        # Public, reusable Xi'an urban-region extent. It deliberately does not
        # encode or disclose the private trajectory sample's spatial bounds.
        west, south, east, north = 108.70, 34.00, 109.20, 34.55
        coverage = "Public fixed Xi'an urban/core-region extent; not trajectory-derived"
    else:
        if args.sample is None:
            raise ValueError("--sample is required when --region=sample")
        sample = pd.read_parquet(args.sample, columns=["lon", "lat"])
        west = float(sample.lon.min() - args.margin_deg)
        east = float(sample.lon.max() + args.margin_deg)
        south = float(sample.lat.min() - args.margin_deg)
        north = float(sample.lat.max() + args.margin_deg)
        coverage = "Stage0 sample bounds plus margin; not the full Xi'an municipality"

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(args.output_dir / "osm_cache")
    ox.settings.log_console = True
    ox.settings.requests_timeout = 300
    ox.settings.overpass_rate_limit = True
    ox.settings.useful_tags_node = list(set(ox.settings.useful_tags_node + ["highway", "crossing"]))
    ox.settings.useful_tags_way = list(
        set(
            ox.settings.useful_tags_way
            + ["highway", "lanes", "maxspeed", "name", "oneway", "junction", "surface"]
        )
    )

    graphml_path = args.output_dir / "xian_stage0_drive.graphml"
    if args.reuse_existing and graphml_path.exists():
        graph = ox.load_graphml(graphml_path)
    else:
        graph = ox.graph_from_bbox(
            north=north,
            south=south,
            east=east,
            west=west,
            network_type="drive",
            simplify=True,
            retain_all=True,
            truncate_by_edge=True,
        )
        graph = ox.add_edge_speeds(graph)
        graph = ox.add_edge_travel_times(graph)
        graph.graph["downloaded_utc"] = datetime.now(timezone.utc).isoformat()
        graph.graph["stage0_bbox"] = json.dumps([west, south, east, north])
        ox.save_graphml(graph, graphml_path)

    nodes, edges = ox.graph_to_gdfs(graph)
    nodes = nodes.reset_index()
    edges = edges.reset_index()
    undirected = nx.Graph(graph)
    nodes["street_degree"] = nodes["osmid"].map(dict(undirected.degree())).fillna(0).astype(int)
    nodes["signal"] = nodes.get("highway", pd.Series(index=nodes.index, dtype=object)).apply(
        lambda value: "traffic_signals" in str(value)
    )
    nodes["intersection_type"] = "none"
    nodes.loc[nodes["street_degree"] >= 3, "intersection_type"] = "intersection"
    nodes.loc[nodes["signal"], "intersection_type"] = "signalized_intersection"

    edges["link_id"] = edges.apply(lambda r: f"{r['u']}_{r['v']}_{r['key']}", axis=1)
    edges["from_node"] = edges["u"]
    edges["to_node"] = edges["v"]
    edges["road_class"] = edges["highway"].apply(first)
    edges["lane_num"] = edges.get("lanes", pd.Series(index=edges.index, dtype=object)).apply(parse_number)
    edges["speed_limit"] = edges.get("maxspeed", pd.Series(index=edges.index, dtype=object)).apply(parse_number)
    edges["road_name"] = edges.get("name", pd.Series(index=edges.index, dtype=object)).apply(first)
    edges["signal"] = edges["from_node"].isin(nodes.loc[nodes.signal, "osmid"]) | edges["to_node"].isin(
        nodes.loc[nodes.signal, "osmid"]
    )
    intersection_nodes = set(nodes.loc[nodes.street_degree >= 3, "osmid"])
    edges["intersection_type"] = "none"
    edges.loc[
        edges["from_node"].isin(intersection_nodes) | edges["to_node"].isin(intersection_nodes),
        "intersection_type",
    ] = "intersection_adjacent"
    edges.loc[edges["signal"], "intersection_type"] = "signal_adjacent"

    edge_cols = [
        "link_id", "from_node", "to_node", "key", "geometry", "length", "road_class",
        "lane_num", "speed_limit", "oneway", "signal", "intersection_type", "road_name",
        "highway", "lanes", "maxspeed", "surface", "travel_time",
    ]
    edge_cols = [c for c in edge_cols if c in edges.columns]
    node_cols = [
        "osmid", "x", "y", "geometry", "street_degree", "signal", "intersection_type", "highway"
    ]
    node_cols = [c for c in node_cols if c in nodes.columns]
    clean_edges = gpd.GeoDataFrame(edges[edge_cols], geometry="geometry", crs=edges.crs)
    clean_nodes = gpd.GeoDataFrame(nodes[node_cols], geometry="geometry", crs=nodes.crs)
    for frame in (clean_edges, clean_nodes):
        for column in frame.select_dtypes(include="object").columns:
            if column != "geometry":
                frame[column] = frame[column].apply(
                    lambda value: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, tuple, dict))
                    else value
                )
    clean_edges.to_file(args.output_dir / "xian_stage0_edges.gpkg", layer="edges", driver="GPKG")
    clean_nodes.to_file(args.output_dir / "xian_stage0_nodes.gpkg", layer="nodes", driver="GPKG")
    combined = args.output_dir / "xian_stage0_network.gpkg"
    # The bundled Fiona cannot reliably append a second layer on Windows.
    # Keep a convenient edges-only GPKG and publish nodes as a separate GPKG.
    clean_edges.to_file(combined, layer="edges", driver="GPKG", mode="w")
    clean_edges.to_parquet(args.output_dir / "xian_stage0_edges.parquet", index=False)
    clean_nodes.to_parquet(args.output_dir / "xian_stage0_nodes.parquet", index=False)

    metadata = {
        "source": "OpenStreetMap via Overpass API",
        "downloaded_utc": graph.graph["downloaded_utc"],
        "network_type": "drive",
        "coverage": coverage,
        "bbox_wgs84": {"west": west, "south": south, "east": east, "north": north},
        "node_count": len(graph.nodes),
        "directed_edge_count": len(graph.edges),
        "crs": str(graph.graph.get("crs")),
        "trajectory_date": "2016-10-01",
        "temporal_caveat": "Current OSM is being matched to 2016 trajectories.",
        "artifacts": {
            "graph": "xian_stage0_drive.graphml",
            "edges": ["xian_stage0_edges.gpkg", "xian_stage0_edges.parquet"],
            "nodes": ["xian_stage0_nodes.gpkg", "xian_stage0_nodes.parquet"],
            "edges_only_convenience_gpkg": "xian_stage0_network.gpkg"
        },
    }
    (args.output_dir / "network_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
