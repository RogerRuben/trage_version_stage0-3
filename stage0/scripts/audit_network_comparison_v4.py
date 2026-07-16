"""Compare the fixed Stage0 diagnostic sample across network versions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage0.canonical.topology import allows_forward, allows_reverse, legal_entries, legal_exits


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def graph_and_metrics(path: Path) -> tuple[nx.DiGraph, dict[str, dict], dict]:
    roads = gpd.read_parquet(path)
    graph = nx.DiGraph()
    lookup = {}
    for row in roads.itertuples(index=False):
        link = str(row.link_id)
        u, v = int(row.from_node), int(row.to_node)
        code = str(row.oneway_code) if pd.notna(row.oneway_code) else "B"
        length = max(0.0, float(row.length_m))
        lookup[link] = {"u": u, "v": v, "code": code, "length": length}
        if allows_forward(code):
            graph.add_edge(u, v, length_m=length)
        if allows_reverse(code):
            graph.add_edge(v, u, length_m=length)
    weak = list(nx.weakly_connected_components(graph))
    strong = list(nx.strongly_connected_components(graph))
    largest = max(weak, key=len, default=set())
    in_largest = roads.from_node.isin(largest) & roads.to_node.isin(largest)
    metrics = {
        "node_count": graph.number_of_nodes(),
        "link_count": int(len(roads)),
        "weak_components": len(weak),
        "strong_components": len(strong),
        "largest_wcc_node_share": len(largest) / max(1, graph.number_of_nodes()),
        "largest_wcc_link_share": float(in_largest.mean()),
        "largest_wcc_length_share": float(
            roads.loc[in_largest, "length_m"].sum() / max(float(roads.length_m.sum()), 1.0)
        ),
    }
    return graph, lookup, metrics


def read_route_parts(root: Path, date: str) -> pd.DataFrame:
    files = sorted((root / "hmm_route_parts" / f"day={date}").glob("*.parquet"))
    return pd.concat(
        [pd.read_parquet(path, columns=["order_id", "link_id", "route_sequence"]) for path in files],
        ignore_index=True,
    )


def directed_reachability(graph: nx.DiGraph, lookup: dict[str, dict], routes: pd.DataFrame) -> dict:
    reachable = 0
    eligible = 0
    for _, group in routes.groupby("order_id", sort=False):
        links = group.sort_values("route_sequence").link_id.astype(str)
        first = lookup.get(links.iloc[0])
        last = lookup.get(links.iloc[-1])
        if first is None or last is None:
            continue
        eligible += 1
        sources = legal_entries(first["u"], first["v"], first["code"])
        targets = legal_exits(last["u"], last["v"], last["code"])
        if any(source in graph and target in graph and nx.has_path(graph, source, target)
               for source in sources for target in targets):
            reachable += 1
    return {
        "directed_od_eligible_orders": eligible,
        "directed_od_reachable_orders": reachable,
        "directed_od_reachability": reachable / max(1, eligible),
    }


def route_length_metrics(quality: pd.DataFrame) -> dict:
    return {
        "mean_matched_route_length_m": float(quality.matched_route_length_m.mean()),
        "p95_matched_route_length_m": float(quality.matched_route_length_m.quantile(0.95)),
        "mean_route_length_ratio": float(quality.route_length_ratio.mean()),
        "p95_route_length_ratio": float(quality.route_length_ratio.quantile(0.95)),
    }


def main() -> None:
    args = arguments()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = []
    for variant in config["variants"]:
        graph, lookup, metrics = graph_and_metrics(Path(variant["roads"]))
        routes = read_route_parts(Path(variant["stage0_root"]), config["date"])
        quality = pd.read_parquet(variant["quality"])
        summary = pd.read_csv(variant["summary"]).iloc[0].to_dict()
        row = {
            "variant": variant["name"],
            **metrics,
            **directed_reachability(graph, lookup, routes),
            **summary,
            **route_length_metrics(quality),
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    v3 = frame.loc[frame.variant.eq("noded_v3")].iloc[0]
    v4 = frame.loc[frame.variant.eq("noded_v4")].iloc[0]
    stop_reasons = []
    if v4.core_share_of_input < v3.core_share_of_input - config["gates"]["maximum_core_drop"]:
        stop_reasons.append("v4_core_share_drop_exceeds_gate")
    if v4.failed_match_orders > v3.failed_match_orders + config["gates"]["maximum_failed_order_increase"]:
        stop_reasons.append("v4_failed_orders_increased")
    if v4.mean_matched_route_length_m > v3.mean_matched_route_length_m * (1 + config["gates"]["maximum_mean_route_inflation"]):
        stop_reasons.append("v4_route_length_inflation")
    if v4.directed_od_reachability >= config["gates"]["suspicious_reachability"]:
        stop_reasons.append("v4_directed_reachability_requires_manual_review")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    result = {
        "status": "STOP_FOR_MANUAL_REVIEW" if stop_reasons else "DIAGNOSTIC_PASS",
        "fixed_sample_date": config["date"],
        "fixed_sample_orders": int(frame.input_orders.max()),
        "comparison": rows,
        "stop_reasons": stop_reasons,
        "canonical_promotion_gate": "HOLD",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
