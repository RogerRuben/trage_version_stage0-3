"""Build shortest or historical-fastest planned routes from OD proxies.

OD coordinates are first/last GPS observations converted to WGS84. Routing uses
the directed 2017 road graph and Stage1 train-fitted travel-time references.
Planned links are aligned to realized strict targets only where the order
actually traversed the same link; unmatched planned links retain null labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer


TARGETS = ["lcs", "iis", "rts", "pmis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order-od-root", type=Path, default=Path("stage0/output/order_od"))
    parser.add_argument("--strict-target-root", type=Path, default=Path("stage2/output/strict_targets"))
    parser.add_argument("--roads", type=Path, default=Path("map_data/xian_2017/xian_2017_core_roads.parquet"))
    parser.add_argument("--reference-root", type=Path, default=Path("stage1/output/prediction_split/models/travel_time_reference"))
    parser.add_argument("--poi-exposure", type=Path, default=Path("stage0/output/poi/stage0_link_poi_exposure.parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/od_planned_routes"))
    parser.add_argument(
        "--route-source", choices=["historical_fastest_path", "shortest_path"],
        default="historical_fastest_path",
    )
    parser.add_argument("--routing-component", choices=["largest", "all"], default="all")
    parser.add_argument("--dates", default="20161009,20161010,20161011,20161012,20161013,20161014,20161015,20161016,20161017,20161018,20161019")
    parser.add_argument("--max-orders-per-day", type=int, default=2000, help="0 means all orders")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def parse_dates(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_references(root: Path) -> tuple[pd.Series, pd.Series, float]:
    level2 = pd.read_parquet(root / "level2.parquet").set_index("key")["reference_sec_per_m"]
    level5 = pd.read_parquet(root / "level5.parquet").set_index("key")["reference_sec_per_m"]
    global_model = pd.read_parquet(root / "level6.parquet")
    global_value = float(global_model.loc[global_model["key"].eq("global"), "reference_sec_per_m"].iloc[0])
    return level2, level5, global_value


def road_graph(path: Path, reference_root: Path, poi_exposure: Path, routing_component: str):
    roads = gpd.read_parquet(path).to_crs(32649).reset_index(drop=True)
    roads["link_id"] = roads["link_id"].astype(str)
    roads["planned_link_length_m"] = shapely.length(roads.geometry.to_numpy())
    midpoint = shapely.line_interpolate_point(roads.geometry.to_numpy(), roads["planned_link_length_m"].to_numpy() / 2)
    roads["area_grid"] = [f"{int(x // 1000)}_{int(y // 1000)}" for x, y in zip(shapely.get_x(midpoint), shapely.get_y(midpoint))]
    degree = pd.concat([roads["from_node"], roads["to_node"]]).value_counts()
    roads["endpoint_degree"] = (
        roads["from_node"].map(degree).fillna(0).to_numpy() + roads["to_node"].map(degree).fillna(0).to_numpy()
    ) / 2
    roads["link_fragmentation"] = (1000 / roads["planned_link_length_m"].clip(lower=20)).clip(upper=20)
    roads["minor_road"] = roads["road_class"].astype(str).isin({"residential", "unclassified", "service", "living_street", "track"}).astype("int8")
    if poi_exposure.exists():
        exposure = pd.read_parquet(poi_exposure).drop(columns=["link_length_m"], errors="ignore")
        exposure["link_id"] = exposure["link_id"].astype(str)
        roads = roads.merge(exposure, on="link_id", how="left", validate="one_to_one")
    level2, level5, global_value = load_references(reference_root)
    for mode in ["peak", "offpeak"]:
        keys = roads["link_id"] + "|" + mode
        roads[f"sec_per_m_{mode}"] = keys.map(level2).fillna(roads["road_class"].astype(str).map(level5)).fillna(global_value)
        roads[f"travel_time_{mode}"] = roads["planned_link_length_m"] * roads[f"sec_per_m_{mode}"]
    graph = nx.MultiDiGraph()
    for row in roads.itertuples():
        attrs = {
            "link_id": row.link_id,
            "weight_peak": float(row.travel_time_peak),
            "weight_offpeak": float(row.travel_time_offpeak),
            "weight_length": float(row.planned_link_length_m),
        }
        code = str(getattr(row, "oneway_code", "B"))
        if code in {"F", "B"}:
            graph.add_edge(int(row.from_node), int(row.to_node), **attrs)
        if code in {"T", "B"}:
            graph.add_edge(int(row.to_node), int(row.from_node), **attrs)
    if routing_component == "largest":
        component = max(nx.weakly_connected_components(graph), key=len)
        graph = graph.subgraph(component).copy()
        roads = roads[roads["from_node"].isin(component) & roads["to_node"].isin(component)].reset_index(drop=True)
    tree = shapely.STRtree(roads.geometry.to_numpy())
    # The distributed OSM line layer retains many whole ways whose internal
    # at-grade intersections are not represented by shared endpoint IDs.  A
    # geometry-noded link graph is therefore kept as an explicit fallback.
    link_graph = nx.Graph()
    for row in roads.itertuples():
        link_graph.add_node(
            row.link_id, weight_length=float(row.planned_link_length_m),
            weight_peak=float(row.travel_time_peak), weight_offpeak=float(row.travel_time_offpeak),
        )
    pairs = tree.query(roads.geometry.to_numpy(), predicate="dwithin", distance=3.0)
    for left, right in zip(pairs[0], pairs[1]):
        if left >= right:
            continue
        a, b = roads.iloc[int(left)], roads.iloc[int(right)]
        if int(a.get("layer", 0)) != int(b.get("layer", 0)):
            continue
        attrs = {
            name: (float(link_graph.nodes[a.link_id][name]) + float(link_graph.nodes[b.link_id][name])) / 2
            for name in ["weight_length", "weight_peak", "weight_offpeak"]
        }
        link_graph.add_edge(str(a.link_id), str(b.link_id), **attrs)
    return roads, graph, link_graph, tree


def snap_links(od: pd.DataFrame, roads: pd.DataFrame, tree: shapely.STRtree) -> pd.DataFrame:
    transformer = Transformer.from_crs(4326, 32649, always_xy=True)
    output = od.copy()
    for prefix in ["origin", "destination"]:
        x, y = transformer.transform(output[f"{prefix}_lon"].to_numpy(), output[f"{prefix}_lat"].to_numpy())
        points = shapely.points(x, y)
        indices, distances = tree.query_nearest(points, all_matches=False, return_distance=True)
        if np.ndim(indices) == 2:
            input_index, road_index = indices[0], indices[1]
            assigned = np.full(len(output), -1, dtype=int)
            assigned[input_index] = road_index
            distance_values = np.full(len(output), np.nan)
            distance_values[input_index] = distances
        else:
            assigned = np.asarray(indices, dtype=int)
            distance_values = np.asarray(distances, dtype=float)
        output[f"{prefix}_snapped_link_id"] = roads.iloc[assigned]["link_id"].to_numpy()
        output[f"{prefix}_snap_distance_m"] = distance_values
    return output


def path_links(
    graph: nx.MultiDiGraph, roads_lookup: pd.DataFrame, first: str, last: str,
    mode: str, route_source: str,
) -> list[str] | None:
    if first not in roads_lookup.index or last not in roads_lookup.index:
        return None
    origin = roads_lookup.loc[first]
    destination = roads_lookup.loc[last]
    pairs = [
        (int(origin.to_node), int(destination.from_node)),
        (int(origin.from_node), int(destination.from_node)),
        (int(origin.to_node), int(destination.to_node)),
        (int(origin.from_node), int(destination.to_node)),
    ]
    weight = "weight_length" if route_source == "shortest_path" else f"weight_{mode}"
    best = None
    for source, target in pairs:
        try:
            nodes = nx.shortest_path(graph, source, target, weight=weight)
            cost = nx.shortest_path_length(graph, source, target, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        links = [first]
        for u, v in zip(nodes[:-1], nodes[1:]):
            edge = min(graph.get_edge_data(u, v).values(), key=lambda item: item[weight])
            links.append(str(edge["link_id"]))
        links.append(last)
        links = [value for index, value in enumerate(links) if index == 0 or value != links[index - 1]]
        if best is None or cost < best[0]:
            best = (float(cost), links)
    return None if best is None else best[1]


def path_link_graph(graph: nx.Graph, first: str, last: str, mode: str, route_source: str) -> list[str] | None:
    weight = "weight_length" if route_source == "shortest_path" else f"weight_{mode}"
    try:
        return [str(value) for value in nx.shortest_path(graph, first, last, weight=weight)]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def load_realized_targets(root: Path, date: str, orders: set[str]) -> pd.DataFrame:
    parts = []
    for path in sorted((root / f"day={date}").glob("*.parquet")):
        frame = pd.read_parquet(path)
        frame = frame[frame["order_id"].isin(orders)]
        if not frame.empty:
            parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def is_peak(timestamp: pd.Timestamp) -> bool:
    local = timestamp.tz_convert("Asia/Shanghai")
    minute = local.hour * 60 + local.minute
    return (7 * 60 <= minute <= 9 * 60 + 30) or (17 * 60 <= minute <= 19 * 60 + 30)


def build_day(
    od: pd.DataFrame, roads: pd.DataFrame, graph: nx.MultiDiGraph, link_graph: nx.Graph,
    targets: pd.DataFrame, route_source: str,
) -> pd.DataFrame:
    lookup = roads.set_index("link_id")
    undirected = graph.to_undirected()
    if not targets.empty:
        targets = targets.sort_values(["order_id", "link_seq"]).copy()
        targets["link_occurrence"] = targets.groupby(["order_id", "link_id"]).cumcount()
        target_lookup = targets.set_index(["order_id", "link_id", "link_occurrence"])
    else:
        target_lookup = None
    rows = []
    for order in od.itertuples(index=False):
        dispatch = pd.to_datetime(float(order.origin_timestamp), unit="s", utc=True).round("us")
        mode = "peak" if is_peak(dispatch) else "offpeak"
        links = path_links(
            graph, lookup, str(order.origin_snapped_link_id), str(order.destination_snapped_link_id), mode, route_source,
        )
        routing_fallback = "none"
        if not links:
            links = path_links(
                undirected, lookup, str(order.origin_snapped_link_id), str(order.destination_snapped_link_id), mode, route_source,
            )
            routing_fallback = "undirected_connectivity"
        if not links:
            links = path_link_graph(
                link_graph, str(order.origin_snapped_link_id), str(order.destination_snapped_link_id), mode, route_source,
            )
            routing_fallback = "geometry_noded_link_graph"
        if not links:
            continue
        route = pd.DataFrame({"order_id": order.order_id, "planned_link_id": links})
        route["planned_link_seq"] = np.arange(len(route), dtype="int32")
        static_columns = [
            "road_class", "area_grid", "planned_link_length_m", "endpoint_degree", "link_fragmentation", "minor_road",
            f"sec_per_m_{mode}", "activity_intensity_index",
        ] + [column for column in lookup.columns if column.startswith("poi_density_100m_")]
        static_columns = [column for column in static_columns if column in lookup.columns]
        route = route.join(lookup[static_columns], on="planned_link_id")
        route["estimated_link_travel_time_sec"] = route["planned_link_length_m"] * route[f"sec_per_m_{mode}"]
        elapsed = route["estimated_link_travel_time_sec"].shift(fill_value=0).cumsum()
        route["estimated_link_entry_time"] = (dispatch + pd.to_timedelta(elapsed, unit="s")).dt.round("us")
        route["dispatch_time"] = dispatch
        route["date"] = order.date
        route["driver_id"] = order.driver_id
        route["route_source"] = route_source
        route["od_source"] = order.od_source
        route["origin_snap_distance_m"] = order.origin_snap_distance_m
        route["destination_snap_distance_m"] = order.destination_snap_distance_m
        route["routing_time_mode"] = mode
        route["routing_fallback"] = routing_fallback
        route["planned_route_link_count"] = len(route)
        route["position_ratio"] = route["planned_link_seq"] / max(len(route) - 1, 1)
        total_length = route["planned_link_length_m"].sum()
        route["distance_to_destination_ratio"] = 1 - route["planned_link_length_m"].cumsum() / max(total_length, 1)
        if target_lookup is not None:
            route["link_occurrence"] = route.groupby(["order_id", "planned_link_id"]).cumcount()
            index = pd.MultiIndex.from_arrays([route["order_id"], route["planned_link_id"], route["link_occurrence"]])
            aligned = target_lookup.reindex(index).reset_index(drop=True)
            for column in aligned.columns:
                if column not in {"order_id", "link_id", "driver_id", "date", "link_seq", "time_bin", "traversal_quality"}:
                    route[column] = aligned[column].to_numpy()
        route["realized_label_available"] = route["target_lcs_raw"].notna() if "target_lcs_raw" in route else False
        rows.append(route)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    roads, graph, link_graph, tree = road_graph(args.roads, args.reference_root, args.poi_exposure, args.routing_component)
    manifest_path = args.output_root / "planned_route_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("route_source") != args.route_source:
            raise ValueError("output root already contains a different route_source")
    else:
        manifest = {"route_source": args.route_source, "od_source": "first_last_gps_observation", "days": {}}
    for number, date in enumerate(parse_dates(args.dates)):
        od = pd.read_parquet(args.order_od_root / f"day={date}.parquet")
        if "od_route_eligible" in od:
            od = od[od["od_route_eligible"]].copy()
        else:
            od = od[od["raw_point_count"].ge(5) & od["duration_sec"].ge(30)].copy()
        if args.max_orders_per_day > 0 and len(od) > args.max_orders_per_day:
            od = od.sample(n=args.max_orders_per_day, random_state=args.seed + number)
        # Audited OD may contain nearest-link snaps against the complete road
        # layer.  Routing must re-snap against the selected routable component.
        for prefix in ["origin", "destination"]:
            if f"{prefix}_snapped_link_id" in od:
                od.rename(columns={
                    f"{prefix}_snapped_link_id": f"{prefix}_nearest_road_link_id",
                    f"{prefix}_snap_distance_m": f"{prefix}_nearest_road_snap_distance_m",
                }, inplace=True)
        od = snap_links(od, roads, tree)
        targets = load_realized_targets(args.strict_target_root, date, set(od["order_id"]))
        output = build_day(od, roads, graph, link_graph, targets, args.route_source)
        target = args.output_root / f"day={date}.parquet"
        output.to_parquet(target, index=False, compression="zstd")
        orders_output = int(output["order_id"].nunique()) if len(output) else 0
        manifest["days"][date] = {
            "orders_input": int(len(od)), "orders_routed": orders_output,
            "route_success_ratio": orders_output / max(len(od), 1), "rows": int(len(output)),
            "realized_label_link_ratio": float(output["realized_label_available"].mean()) if len(output) else None,
            "origin_snap_p90_m": float(od["origin_snap_distance_m"].quantile(0.90)),
            "destination_snap_p90_m": float(od["destination_snap_distance_m"].quantile(0.90)),
            "undirected_fallback_ratio": float(output.drop_duplicates("order_id")["routing_fallback"].eq("undirected_connectivity").mean()) if len(output) else None,
            "geometry_link_fallback_ratio": float(output.drop_duplicates("order_id")["routing_fallback"].eq("geometry_noded_link_graph").mean()) if len(output) else None,
        }
        print(f"planned routes day={date} {manifest['days'][date]}", flush=True)
    manifest.update({
        "max_orders_per_day": args.max_orders_per_day,
        "routing_component": args.routing_component,
        "estimated_entry_time_source": "train_fitted_hierarchical_travel_time_reference",
        "fully_deployable": False,
        "deployment_caveat": "first/last GPS are empirical proxies for dispatch-time OD; no platform route/ETA log is available",
        "topology_fallback": "same-layer geometry-noded link graph within 3m; used only after directed and undirected endpoint topology fail",
        "topology_fallback_caveat": "the OSM line layer is not fully noded at internal intersections; fallback use and planned/actual overlap must be audited",
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
