"""Audit graph-only connector use in expanded diagnostic routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


def read_parts(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(directory)
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--route-dir", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--review-sample", type=Path, required=True)
    parser.add_argument("--review-geojson", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    args = parser.parse_args()
    roads = gpd.read_parquet(args.roads)
    connectors = roads.loc[roads.topology_connector.fillna(False).astype(bool)].copy()
    connectors["link_id"] = connectors.link_id.astype(str)
    routes = read_parts(args.route_dir)
    routes["link_id"] = routes.link_id.astype(str)
    routes["order_id"] = routes.order_id.astype(str)
    routes["route_max_sequence"] = routes.groupby("order_id").route_sequence.transform("max")
    used = routes.loc[routes.link_id.isin(set(connectors.link_id))].copy()
    used["route_position"] = pd.cut(
        used.route_sequence / used.route_max_sequence.clip(lower=1),
        bins=[-0.01, 0.1, 0.9, 1.01], labels=["origin_near", "internal", "destination_near"],
    ).astype(str)
    details = used.merge(
        connectors.drop(columns="geometry"), on="link_id", how="left", validate="many_to_one"
    )
    per_order = details.groupby("order_id").agg(
        connector_count=("link_id", "size"),
        connector_distance_m=("length_m", "sum"),
        internal_connector_count=("route_position", lambda x: int((x == "internal").sum())),
        od_near_connector_count=("route_position", lambda x: int((x != "internal").sum())),
    ) if len(details) else pd.DataFrame()
    per_connector = details.groupby("link_id").agg(
        order_count=("order_id", "nunique"),
        occurrence_count=("order_id", "size"),
        connector_distance_m=("length_m", "first"),
        oneway_code=("oneway_code", "first"),
        connector_semantics=("connector_semantics", "first"),
    ).reset_index() if len(details) else pd.DataFrame()
    total_orders = int(routes.order_id.nunique())
    used_orders = int(details.order_id.nunique()) if len(details) else 0
    top_count = int(per_connector.order_count.max()) if len(per_connector) else 0
    summary = {
        "status": "DIAGNOSTIC_PASS",
        "date": args.date,
        "total_routes": total_orders,
        "orders_using_connector": used_orders,
        "connector_order_share": used_orders / max(1, total_orders),
        "connector_occurrences": int(len(details)),
        "unique_connectors_used": int(details.link_id.nunique()) if len(details) else 0,
        "mean_connectors_per_using_order": float(per_order.connector_count.mean()) if len(per_order) else 0.0,
        "p95_connectors_per_using_order": float(per_order.connector_count.quantile(0.95)) if len(per_order) else 0.0,
        "cumulative_connector_distance_m": float(details.length_m.sum()) if len(details) else 0.0,
        "maximum_connector_distance_m": float(details.length_m.max()) if len(details) else 0.0,
        "internal_connector_occurrences": int(details.route_position.eq("internal").sum()) if len(details) else 0,
        "od_near_connector_occurrences": int(details.route_position.ne("internal").sum()) if len(details) else 0,
        "maximum_orders_using_single_connector": top_count,
        "maximum_connector_order_share": top_count / max(1, total_orders),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(args.summary, index=False, encoding="utf-8-sig")
    args.details.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.details, index=False, encoding="utf-8-sig")
    if len(per_connector):
        per_connector["semantic_group"] = per_connector.connector_semantics.fillna("unknown")
        top = per_connector.sort_values(["order_count", "link_id"], ascending=[False, True]).head(20)
        remaining = per_connector.loc[~per_connector.link_id.isin(top.link_id)]
        diverse = remaining.groupby(["oneway_code", "semantic_group"], group_keys=False).head(1)
        sample = pd.concat([top, diverse], ignore_index=True).drop_duplicates("link_id").head(args.sample_size)
        if len(sample) < min(args.sample_size, len(per_connector)):
            extra = remaining.loc[~remaining.link_id.isin(sample.link_id)].sort_values("link_id")
            sample = pd.concat([sample, extra.head(args.sample_size - len(sample))], ignore_index=True)
    else:
        sample = per_connector
    for column in [
        "reviewer_id", "review_status", "direction_correct", "level_transition_correct",
        "ramp_or_terminal_evidence", "major_error", "data_limitation", "comments",
    ]:
        sample[column] = "pending" if column == "review_status" else pd.NA
    args.review_sample.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.review_sample, index=False, encoding="utf-8-sig")
    review_geometry = connectors.loc[connectors.link_id.isin(sample.link_id)].merge(
        sample[["link_id", "order_count", "occurrence_count"]], on="link_id", how="inner"
    ).to_crs(4326)
    args.review_geojson.parent.mkdir(parents=True, exist_ok=True)
    review_geometry.to_file(args.review_geojson, driver="GeoJSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
