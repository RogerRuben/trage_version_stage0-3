"""Extract Xi'an roads and traffic points from the 2017 Geofabrik China shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, box


# Conservative municipality envelope from the commonly published Xi'an extent.
# The 2017 free archive has no admin boundary layer, so this is explicitly not a
# boundary-accurate administrative clip.
XIAN_ENVELOPE = (107.65, 33.65, 109.82, 34.75)
XIAN_CORE = (108.70, 34.00, 109.20, 34.55)
NON_DRIVABLE = {"footway", "path", "steps", "cycleway", "bridleway", "pedestrian"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def clip_lines(frame: gpd.GeoDataFrame, bounds) -> gpd.GeoDataFrame:
    result = frame.clip(box(*bounds)).explode(index_parts=False).reset_index(drop=True)
    result = result[result.geometry.geom_type == "LineString"].copy()
    result = result[~result.geometry.is_empty]
    return result


def add_topology(frame: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    frame = frame.reset_index(drop=True).copy()
    endpoint_to_id: dict[tuple[float, float], int] = {}

    def node_id(coord) -> int:
        key = (round(float(coord[0]), 7), round(float(coord[1]), 7))
        if key not in endpoint_to_id:
            endpoint_to_id[key] = len(endpoint_to_id)
        return endpoint_to_id[key]

    frame["from_node"] = [node_id(line.coords[0]) for line in frame.geometry]
    frame["to_node"] = [node_id(line.coords[-1]) for line in frame.geometry]
    frame["link_id"] = [f"{osm_id}_{i}" for i, osm_id in enumerate(frame.osm_id.astype(str))]
    frame["road_class"] = frame["fclass"]
    frame["road_name"] = frame["name"]
    frame["speed_limit"] = frame["maxspeed"].replace(0, np.nan)
    frame["lane_num"] = np.nan
    frame["oneway_code"] = frame["oneway"]
    frame["oneway_direction"] = frame["oneway"].map(
        {"F": "geometry_direction", "T": "opposite_geometry", "B": "both"}
    ).fillna("unknown")
    projected = frame.to_crs(32649)
    frame["length_m"] = projected.length.to_numpy()

    reverse = {value: key for key, value in endpoint_to_id.items()}
    nodes = gpd.GeoDataFrame(
        {
            "node_id": list(reverse),
            "lon": [reverse[i][0] for i in reverse],
            "lat": [reverse[i][1] for i in reverse],
        },
        geometry=gpd.points_from_xy(
            [reverse[i][0] for i in reverse], [reverse[i][1] for i in reverse]
        ),
        crs=4326,
    )
    graph = nx.Graph()
    graph.add_edges_from(zip(frame.from_node, frame.to_node))
    degree = dict(graph.degree())
    nodes["street_degree"] = nodes.node_id.map(degree).fillna(0).astype(int)
    nodes["intersection_type"] = np.where(nodes.street_degree >= 3, "intersection", "none")
    return frame, nodes


def add_signal_semantics(
    roads: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, traffic: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    signal = traffic[traffic.fclass == "traffic_signals"].copy()
    roads = roads.copy()
    nodes = nodes.copy()
    if signal.empty:
        roads["signal"] = False
        nodes["signal"] = False
        return roads, nodes
    signal_p = signal.to_crs(32649)
    roads_p = roads.to_crs(32649)
    nodes_p = nodes.to_crs(32649)
    road_join = gpd.sjoin_nearest(
        roads_p[["link_id", "geometry"]], signal_p[["geometry"]], how="left", distance_col="signal_dist"
    )
    road_min = road_join.groupby("link_id").signal_dist.min()
    roads["signal"] = roads.link_id.map(road_min).le(20).fillna(False)
    node_join = gpd.sjoin_nearest(
        nodes_p[["node_id", "geometry"]], signal_p[["geometry"]], how="left", distance_col="signal_dist"
    )
    node_min = node_join.groupby("node_id").signal_dist.min()
    nodes["signal"] = nodes.node_id.map(node_min).le(20).fillna(False)
    nodes.loc[nodes.signal, "intersection_type"] = "signalized_intersection"
    return roads, nodes


def build_graph(roads: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(crs="EPSG:4326", name="Xi'an 2017 Geofabrik drive network")
    for row in nodes.itertuples():
        graph.add_node(
            int(row.node_id), x=float(row.lon), y=float(row.lat),
            street_degree=int(row.street_degree), signal=bool(row.signal),
            intersection_type=row.intersection_type,
        )
    columns = [
        "link_id", "osm_id", "road_class", "road_name", "speed_limit", "oneway_code",
        "oneway_direction", "layer", "bridge", "tunnel", "length_m", "signal",
    ]
    for row in roads.itertuples():
        attrs = {column: getattr(row, column) for column in columns}
        attrs["geometry"] = row.geometry
        if row.oneway_code in ("F", "B"):
            graph.add_edge(int(row.from_node), int(row.to_node), **attrs)
        if row.oneway_code in ("T", "B"):
            reverse_attrs = attrs.copy()
            reverse_attrs["geometry"] = LineString(list(row.geometry.coords)[::-1])
            reverse_attrs["link_id"] = f"{row.link_id}_rev"
            graph.add_edge(int(row.to_node), int(row.from_node), **reverse_attrs)
    return graph


def save_region(
    name: str,
    roads: gpd.GeoDataFrame,
    traffic: gpd.GeoDataFrame,
    output_dir: Path,
    write_graph: bool,
) -> dict:
    roads, nodes = add_topology(roads)
    roads, nodes = add_signal_semantics(roads, nodes, traffic)
    roads.to_parquet(output_dir / f"{name}_roads.parquet", index=False)
    nodes.to_parquet(output_dir / f"{name}_nodes.parquet", index=False)
    traffic.to_parquet(output_dir / f"{name}_traffic.parquet", index=False)
    roads.to_file(output_dir / f"{name}_roads.gpkg", layer="roads", driver="GPKG")
    nodes.to_file(output_dir / f"{name}_nodes.gpkg", layer="nodes", driver="GPKG")
    traffic.to_file(output_dir / f"{name}_traffic.gpkg", layer="traffic", driver="GPKG")
    if write_graph:
        graph = build_graph(roads, nodes)
        ox.save_graphml(graph, output_dir / f"{name}_drive.graphml")
        directed_edges = graph.number_of_edges()
    else:
        directed_edges = int((roads.oneway_code.eq("B").astype(int) + 1).sum())
    return {
        "roads": len(roads),
        "nodes": len(nodes),
        "directed_edges": directed_edges,
        "traffic_points": len(traffic),
        "traffic_signals": int((traffic.fclass == "traffic_signals").sum()),
        "known_speed_limit_ratio": float(roads.speed_limit.notna().mean()),
        "road_class_counts": roads.road_class.value_counts().to_dict(),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roads_path = args.source_dir / "gis_osm_roads_free_1.shp"
    traffic_path = args.source_dir / "gis_osm_traffic_free_1.shp"
    roads_all = gpd.read_file(roads_path, bbox=XIAN_ENVELOPE)
    traffic_all = gpd.read_file(traffic_path, bbox=XIAN_ENVELOPE)
    drive = roads_all[~roads_all.fclass.isin(NON_DRIVABLE)].copy()
    envelope_roads = clip_lines(drive, XIAN_ENVELOPE)
    envelope_traffic = traffic_all.clip(box(*XIAN_ENVELOPE)).reset_index(drop=True)
    core_roads = clip_lines(envelope_roads, XIAN_CORE)
    core_traffic = envelope_traffic.clip(box(*XIAN_CORE)).reset_index(drop=True)

    metadata = {
        "source": str(roads_path.resolve()),
        "source_snapshot_utc": "2017-01-01T20:28:02Z",
        "source_document": "osm-data-in-gis-formats-free.pdf",
        "crs": "EPSG:4326 (WGS84)",
        "oneway_codes": {
            "F": "only in LineString direction",
            "T": "only opposite LineString direction",
            "B": "both directions",
        },
        "excluded_non_drivable_classes": sorted(NON_DRIVABLE),
        "lane_semantics": "Unavailable in the Geofabrik free shape schema",
        "admin_boundary_caveat": "The 2017 free archive has no adminareas layer; envelope output is a conservative rectangular clip.",
        "municipality_envelope_bbox": XIAN_ENVELOPE,
        "trajectory_core_bbox": XIAN_CORE,
        "municipality_envelope": save_region(
            "xian_2017_envelope", envelope_roads, envelope_traffic, args.output_dir, False
        ),
        "trajectory_core": save_region(
            "xian_2017_core", core_roads, core_traffic, args.output_dir, True
        ),
    }
    (args.output_dir / "xian_2017_network_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
