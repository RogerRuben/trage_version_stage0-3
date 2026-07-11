"""Build explicit oracle or planned-route Stage2 route tables.

`actual_matched_route` is an oracle export. `shortest_path` replaces the interior
route with a directed road-network shortest path between first/last matched-link
endpoint proxies. The latter is a routing prototype, not fully deployable until
true order OD coordinates replace those endpoint proxies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_stage2_full_tabular import collect_order_budget  # noqa: E402


ID_COLUMNS = ["order_id", "driver_id", "date", "link_id", "link_seq", "enter_time"]
TARGET_COLUMNS = [
    "target_lcs_pct", "target_iis_pct", "target_rts_pct", "target_pmis_pct",
    "lcs_valid", "iis_valid", "rts_valid", "pmis_valid",
]
DEFAULT_SPEED_KPH = {
    "motorway": 70, "trunk": 55, "primary": 45, "secondary": 35,
    "tertiary": 30, "residential": 22, "service": 18, "unclassified": 25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("stage2/output/link_dataset"))
    parser.add_argument("--roads", type=Path, default=Path("map_data/xian_2017/xian_2017_core_roads.parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/planned_route_dataset"))
    parser.add_argument("--route-source", choices=["actual_matched_route", "shortest_path"], default="actual_matched_route")
    parser.add_argument("--max-orders-per-split", type=int, default=1000, help="0 means all orders")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def read_orders(path: Path, maximum: int, seed: int) -> pd.DataFrame:
    selected = None if maximum <= 0 else collect_order_budget(path, maximum, seed)
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    columns = [column for column in ID_COLUMNS + TARGET_COLUMNS if column in available]
    parts = []
    for group_no in range(parquet.metadata.num_row_groups):
        frame = parquet.read_row_group(group_no, columns=columns).to_pandas()
        if selected is not None:
            frame = frame[frame["order_id"].isin(selected)]
        if not frame.empty:
            parts.append(frame)
    return pd.concat(parts, ignore_index=True).sort_values(["order_id", "link_seq"]) if parts else pd.DataFrame(columns=columns)


def road_data(path: Path) -> tuple[gpd.GeoDataFrame, nx.MultiDiGraph]:
    roads = gpd.read_parquet(path).copy()
    roads["link_id"] = roads["link_id"].astype(str)
    projected = roads.to_crs(32649)
    roads["planned_link_length_m"] = projected.geometry.length.to_numpy()
    graph = nx.MultiDiGraph()
    for row in roads.itertuples():
        attrs = {"link_id": row.link_id, "weight": float(row.planned_link_length_m)}
        code = str(getattr(row, "oneway_code", "B"))
        if code in {"F", "B"}:
            graph.add_edge(int(row.from_node), int(row.to_node), **attrs)
        if code in {"T", "B"}:
            graph.add_edge(int(row.to_node), int(row.from_node), **attrs)
    return roads, graph


def shortest_links(graph: nx.MultiDiGraph, roads: pd.DataFrame, first: str, last: str) -> list[str] | None:
    lookup = roads.set_index("link_id")
    if first not in lookup.index or last not in lookup.index:
        return None
    origin = lookup.loc[first]
    destination = lookup.loc[last]
    pairs = [
        (int(origin.to_node), int(destination.from_node)),
        (int(origin.from_node), int(destination.from_node)),
        (int(origin.to_node), int(destination.to_node)),
        (int(origin.from_node), int(destination.to_node)),
    ]
    best: tuple[float, list[str]] | None = None
    for source, target in pairs:
        try:
            nodes = nx.shortest_path(graph, source, target, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        links = [first]
        cost = 0.0
        for u, v in zip(nodes[:-1], nodes[1:]):
            options = graph.get_edge_data(u, v)
            edge = min(options.values(), key=lambda item: item["weight"])
            links.append(str(edge["link_id"]))
            cost += float(edge["weight"])
        links.append(last)
        links = [value for index, value in enumerate(links) if index == 0 or value != links[index - 1]]
        if best is None or cost < best[0]:
            best = (cost, links)
    return None if best is None else best[1]


def planned_rows(actual: pd.DataFrame, roads: pd.DataFrame, graph: nx.MultiDiGraph) -> pd.DataFrame:
    lookup = roads.set_index("link_id")
    outputs: list[pd.DataFrame] = []
    for order_id, group in actual.groupby("order_id", sort=False):
        group = group.sort_values("link_seq")
        first, last = str(group.iloc[0].link_id), str(group.iloc[-1].link_id)
        links = shortest_links(graph, roads, first, last)
        if not links:
            continue
        route = pd.DataFrame({"order_id": order_id, "planned_link_id": links})
        route["planned_link_seq"] = np.arange(len(route), dtype="int32")
        route = route.join(lookup[["road_class", "planned_link_length_m"]], on="planned_link_id")
        speed = route["road_class"].map(DEFAULT_SPEED_KPH).fillna(25).astype(float)
        route["estimated_link_travel_time_sec"] = route["planned_link_length_m"] / (speed / 3.6)
        dispatch = pd.to_datetime(float(group.iloc[0].enter_time), unit="s", utc=True)
        route["estimated_link_entry_time"] = (
            dispatch + pd.to_timedelta(route["estimated_link_travel_time_sec"].shift(fill_value=0).cumsum(), unit="s")
        ).dt.round("us")
        route["dispatch_time"] = dispatch.round("us")
        route["origin_link_proxy"] = first
        route["destination_link_proxy"] = last
        route["driver_id"] = group.iloc[0].get("driver_id")
        route["date"] = group.iloc[0].get("date")
        actual_unique = group.drop_duplicates("link_id").set_index("link_id")
        for column in TARGET_COLUMNS:
            if column in actual_unique:
                route[column] = route["planned_link_id"].map(actual_unique[column])
        route["label_alignment_status"] = np.where(route["planned_link_id"].isin(actual_unique.index), "planned_link_observed_in_actual_route", "planned_link_unobserved")
        outputs.append(route)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def actual_rows(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.rename(columns={"link_id": "planned_link_id", "link_seq": "planned_link_seq", "enter_time": "estimated_link_entry_time"}).copy()
    output["estimated_link_entry_time"] = pd.to_datetime(output["estimated_link_entry_time"], unit="s", utc=True).dt.round("us")
    dispatch = output.groupby("order_id")["estimated_link_entry_time"].transform("min")
    output["dispatch_time"] = dispatch
    output["label_alignment_status"] = "oracle_exact_actual_route"
    return output


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    roads = graph = None
    if args.route_source == "shortest_path":
        roads, graph = road_data(args.roads)
    counts = {}
    overlap = {}
    order_counts = {}
    route_success = {}
    for split in ["train", "validation", "test"]:
        actual = read_orders(args.dataset_root / f"{split}.parquet", args.max_orders_per_split, args.seed)
        output = actual_rows(actual) if args.route_source == "actual_matched_route" else planned_rows(actual, roads, graph)
        output["route_source"] = args.route_source
        output["experiment_track"] = "oracle_route_upper_bound" if args.route_source == "actual_matched_route" else "planned_route_proxy"
        output.to_parquet(args.output_root / f"{split}_planned_routes.parquet", index=False, compression="zstd")
        counts[split] = int(len(output))
        order_counts[split] = {"input": int(actual["order_id"].nunique()), "output": int(output["order_id"].nunique()) if len(output) else 0}
        route_success[split] = order_counts[split]["output"] / max(order_counts[split]["input"], 1)
        overlap[split] = float(output["label_alignment_status"].ne("planned_link_unobserved").mean()) if len(output) else None
    manifest = {
        "route_source": args.route_source,
        "rows": counts,
        "orders": order_counts,
        "route_success_ratio": route_success,
        "max_orders_per_split": args.max_orders_per_split,
        "route_endpoint_source": "actual_first_last_matched_link_proxy",
        "fully_deployable": False,
        "reason_not_fully_deployable": "true dispatch-time OD and external routing output are absent",
        "label_alignment_ratio": overlap,
    }
    (args.output_root / "planned_route_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
