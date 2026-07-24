"""Audit canonical Stage 0 route topology and interval conservation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage0.canonical.topology import is_directed_transition


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0-root", type=Path, required=True)
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--dates", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    return parser.parse_args()


def road_lookup(path: Path) -> dict[str, dict]:
    roads = gpd.read_parquet(path)
    return {
        str(row.link_id): {
            "link_id": str(row.link_id),
            "from_node": int(row.from_node),
            "to_node": int(row.to_node),
            "oneway_code": str(row.oneway_code) if pd.notna(row.oneway_code) else "B",
        }
        for _, row in roads.iterrows()
    }


def audit_day(root: Path, date: str, roads: dict[str, dict]) -> tuple[dict, list[dict]]:
    route_files = sorted((root / "hmm_route_parts" / f"day={date}").glob("*.parquet"))
    point_files = sorted((root / "hmm_matched_points" / f"day={date}").glob("*.parquet"))
    traversal_files = sorted((root / "hmm_observed_link_traversals" / f"day={date}").glob("*.parquet"))
    if not route_files or not point_files or not traversal_files:
        raise FileNotFoundError(f"incomplete Stage0 day {date}")
    illegal = 0
    flagged_illegal = 0
    unflagged_illegal = 0
    unknown_links = 0
    transitions = 0
    details: list[dict] = []
    order_count = 0
    for path in route_files:
        frame = pd.read_parquet(path, columns=["order_id", "route_sequence", "link_id", "transition_path_status"])
        order_count += frame.order_id.nunique()
        for order_id, group in frame.groupby("order_id", sort=False):
            ordered = group.sort_values("route_sequence")
            links = ordered.link_id.astype(str).tolist()
            statuses = ordered.transition_path_status.astype(str).tolist()
            for transition_no, (left_id, right_id) in enumerate(zip(links, links[1:]), start=1):
                transitions += 1
                left = roads.get(left_id); right = roads.get(right_id)
                if left is None or right is None:
                    unknown_links += 1
                    continue
                if not is_directed_transition(
                    left["from_node"], left["to_node"], left["oneway_code"],
                    right["from_node"], right["to_node"], right["oneway_code"],
                ):
                    illegal += 1
                    explicitly_flagged = statuses[transition_no] == "gap"
                    flagged_illegal += int(explicitly_flagged)
                    unflagged_illegal += int(not explicitly_flagged)
                    if len(details) < 100:
                        details.append({
                            "date": date, "order_id": str(order_id),
                            "from_link_id": left_id, "to_link_id": right_id,
                            "reason": "flagged_direction_gap" if explicitly_flagged else "unflagged_illegal_directed_transition",
                        })

    point_parts = []
    for path in point_files:
        frame = pd.read_parquet(path, columns=["order_id", "dt_s", "segment_distance_m"])
        frame["dt_s"] = pd.to_numeric(frame.dt_s, errors="coerce").fillna(0).clip(lower=0, upper=120)
        frame["segment_distance_m"] = pd.to_numeric(frame.segment_distance_m, errors="coerce").fillna(0).clip(lower=0)
        point_parts.append(frame.groupby("order_id", as_index=False).agg(
            point_time_sec=("dt_s", "sum"), point_distance_m=("segment_distance_m", "sum")
        ))
    points = pd.concat(point_parts).groupby("order_id", as_index=False).sum()
    traversal_parts = []
    for path in traversal_files:
        frame = pd.read_parquet(path, columns=["order_id", "travel_time_sec", "observed_distance_m"])
        traversal_parts.append(frame.groupby("order_id", as_index=False).agg(
            traversal_time_sec=("travel_time_sec", "sum"),
            traversal_distance_m=("observed_distance_m", "sum"),
        ))
    traversals = pd.concat(traversal_parts).groupby("order_id", as_index=False).sum()
    joined = points.merge(traversals, on="order_id", how="outer", indicator=True)
    joined["time_error_sec"] = (joined.point_time_sec - joined.traversal_time_sec).abs()
    joined["distance_error_m"] = (joined.point_distance_m - joined.traversal_distance_m).abs()
    time_fail = int(joined.time_error_sec.gt(1e-6).sum())
    distance_fail = int(joined.distance_error_m.gt(1e-6).sum())
    summary = {
        "date": date,
        "orders": int(order_count),
        "route_transitions": transitions,
        "illegal_directed_transitions": illegal,
        "flagged_direction_gaps": flagged_illegal,
        "unflagged_illegal_directed_transitions": unflagged_illegal,
        "unknown_road_link_transitions": unknown_links,
        "interval_orders": int(len(joined)),
        "time_conservation_failures": time_fail,
        "distance_conservation_failures": distance_fail,
        "maximum_time_error_sec": float(joined.time_error_sec.max()),
        "maximum_distance_error_m": float(joined.distance_error_m.max()),
    }
    summary["status"] = "PASS" if unflagged_illegal == 0 and unknown_links == 0 and time_fail == 0 and distance_fail == 0 else "FAIL"
    return summary, details


def main() -> None:
    args = arguments()
    roads = road_lookup(args.roads)
    summaries = []
    details = []
    for date in [item.strip() for item in args.dates.split(",") if item.strip()]:
        summary, day_details = audit_day(args.stage0_root, date, roads)
        summaries.append(summary); details.extend(day_details)
    result = {
        "status": "PASS" if all(item["status"] == "PASS" for item in summaries) else "FAIL",
        "audit_version": "canonical_stage0_v2",
        "parallel_edge_policy": "MultiDiGraph_preserve_all_route_minimum_edge_only",
        "interval_policy": "adjacent_traversal_proportional_allocation_exact_conservation",
        "iis_influence_policy": "upstream_75m_prorated",
        "days": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.details.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(details, columns=["date", "order_id", "from_link_id", "to_link_id", "reason"]).to_csv(args.details, index=False)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
