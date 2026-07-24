"""Normalize Valhalla trace attributes into stable Stage 0 tables."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


MATCHED_POINT_COLUMNS = [
    "order_id",
    "subtrace_id",
    "original_point_seq",
    "timestamp",
    "matched_point_status",
    "edge_index",
    "valhalla_edge_id",
    "osm_way_id",
    "begin_osm_node_id",
    "end_osm_node_id",
    "forward",
    "percent_along",
    "distance_from_trace_point_m",
    "matching_lon",
    "matching_lat",
    "route_discontinuity",
    "begin_route_discontinuity",
    "end_route_discontinuity",
]

ROUTE_PART_COLUMNS = [
    "order_id",
    "subtrace_id",
    "path_id",
    "route_sequence",
    "valhalla_edge_index",
    "valhalla_edge_id",
    "osm_way_id",
    "begin_osm_node_id",
    "end_osm_node_id",
    "forward",
    "source_percent_along",
    "target_percent_along",
    "length_m",
    "road_class",
    "bridge",
    "tunnel",
    "speed_limit",
    "edge_elapsed_time_s",
    "is_interpolated",
    "route_source",
]


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _edge_identifiers(edge: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    end_node = edge.get("end_node") or {}
    return (
        edge.get("id"),
        edge.get("way_id"),
        edge.get("node_id"),
        end_node.get("node_id"),
    )


def parse_trace_attributes(
    raw: dict[str, Any],
    source_points: pd.DataFrame,
    *,
    order_id: str,
    subtrace_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``matched_points`` and ``route_parts`` for one subtrace."""

    primary_edges = list(raw.get("edges") or [])
    matched_items = list(raw.get("matched_points") or [])
    point_rows: list[dict[str, Any]] = []
    for index in range(len(source_points)):
        item = matched_items[index] if index < len(matched_items) else {"type": "unmatched"}
        edge_index = item.get("edge_index")
        normalized_edge_index = (
            int(edge_index)
            if edge_index is not None
            and 0 <= int(edge_index) < len(primary_edges)
            else None
        )
        edge = (
            primary_edges[normalized_edge_index]
            if normalized_edge_index is not None
            else {}
        )
        edge_id, way_id, begin_node, end_node = _edge_identifiers(edge)
        source = source_points.iloc[index]
        begin_disc = bool(item.get("begin_route_discontinuity", False))
        end_disc = bool(item.get("end_route_discontinuity", False))
        point_rows.append(
            {
                "order_id": str(order_id),
                "subtrace_id": str(subtrace_id),
                "original_point_seq": int(source["original_point_seq"]),
                "timestamp": float(source["timestamp"]),
                "matched_point_status": str(item.get("type", "unmatched")),
                "edge_index": (
                    normalized_edge_index if normalized_edge_index is not None else pd.NA
                ),
                "valhalla_edge_id": str(edge_id) if edge_id is not None else pd.NA,
                "osm_way_id": int(way_id) if way_id is not None else pd.NA,
                "begin_osm_node_id": int(begin_node) if begin_node is not None else pd.NA,
                "end_osm_node_id": int(end_node) if end_node is not None else pd.NA,
                "forward": bool(edge.get("forward")) if edge else pd.NA,
                "percent_along": item.get("distance_along_edge", np.nan),
                "distance_from_trace_point_m": item.get(
                    "distance_from_trace_point", np.nan
                ),
                "matching_lon": item.get("lon", np.nan),
                "matching_lat": item.get("lat", np.nan),
                "route_discontinuity": begin_disc or end_disc,
                "begin_route_discontinuity": begin_disc,
                "end_route_discontinuity": end_disc,
            }
        )
    matched_points = pd.DataFrame(point_rows, columns=MATCHED_POINT_COLUMNS)

    actual_matched_edges = set(
        pd.to_numeric(
            matched_points.loc[
                matched_points.matched_point_status.eq("matched"), "edge_index"
            ],
            errors="coerce",
        ).dropna().astype(int)
    )
    paths = [raw, *list(raw.get("alternate_paths") or [])]
    route_rows: list[dict[str, Any]] = []
    route_sequence = 0
    for path_id, path in enumerate(paths):
        if not isinstance(path, dict):
            continue
        for edge_index, edge in enumerate(path.get("edges") or []):
            edge_id, way_id, begin_node, end_node = _edge_identifiers(edge)
            observed = path_id == 0 and edge_index in actual_matched_edges
            route_rows.append(
                {
                    "order_id": str(order_id),
                    "subtrace_id": str(subtrace_id),
                    "path_id": path_id,
                    "route_sequence": route_sequence,
                    "valhalla_edge_index": edge_index,
                    "valhalla_edge_id": str(edge_id) if edge_id is not None else pd.NA,
                    "osm_way_id": int(way_id) if way_id is not None else pd.NA,
                    "begin_osm_node_id": (
                        int(begin_node) if begin_node is not None else pd.NA
                    ),
                    "end_osm_node_id": int(end_node) if end_node is not None else pd.NA,
                    "forward": bool(edge.get("forward", True)),
                    "source_percent_along": float(edge.get("source_percent_along", 0.0)),
                    "target_percent_along": float(edge.get("target_percent_along", 1.0)),
                    "length_m": float(edge.get("length", 0.0)) * 1000.0,
                    "road_class": edge.get("road_class"),
                    "bridge": bool(edge.get("bridge", False)),
                    "tunnel": bool(edge.get("tunnel", False)),
                    "speed_limit": edge.get("speed_limit"),
                    "edge_elapsed_time_s": (edge.get("end_node") or {}).get(
                        "elapsed_time", np.nan
                    ),
                    "is_interpolated": not observed,
                    "route_source": "observed" if observed else "inferred",
                }
            )
            route_sequence += 1
    route_parts = pd.DataFrame(route_rows, columns=ROUTE_PART_COLUMNS)
    return matched_points, route_parts
