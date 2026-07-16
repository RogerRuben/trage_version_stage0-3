"""Classify Stage0 routes as Core, Extended, or Rejected.

Extended eligibility is based on an actual bounded bridge in the frozen directed
road graph.  The script does not mutate or silently repair the matched route; it
only publishes repair evidence for robustness-set governance.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage0.canonical.topology import (
    allows_forward,
    allows_reverse,
    is_directed_transition,
    legal_entries,
    legal_exits,
)
from stage0.canonical.route_quality import (
    core_threshold_flags,
    projection_metrics,
    route_sequence_metrics,
)
from stage0.canonical.quality_layers import classify_quality_layer


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0-root", type=Path, required=True)
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--dates", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--gap-details", type=Path, required=True)
    return parser.parse_args()


def load_network(path: Path) -> tuple[pd.DataFrame, dict[str, dict], nx.DiGraph]:
    roads = gpd.read_parquet(path).copy()
    roads["link_id"] = roads.link_id.astype(str)
    roads["length_m"] = pd.to_numeric(roads.length_m, errors="coerce").fillna(0).clip(lower=0)
    lookup = {
        row.link_id: {
            "from_node": int(row.from_node),
            "to_node": int(row.to_node),
            "oneway_code": str(row.oneway_code) if pd.notna(row.oneway_code) else "B",
            "length_m": float(row.length_m),
            "source_link_id": str(getattr(row, "source_link_id", row.link_id)),
        }
        for row in roads.itertuples(index=False)
    }
    graph = nx.DiGraph()
    for row in roads.itertuples(index=False):
        u, v, length = int(row.from_node), int(row.to_node), float(row.length_m)
        code = str(row.oneway_code) if pd.notna(row.oneway_code) else "B"
        directed = []
        if allows_forward(code):
            directed.append((u, v))
        if allows_reverse(code):
            directed.append((v, u))
        for left, right in directed:
            prior = graph.get_edge_data(left, right)
            if prior is None or length < float(prior["length_m"]):
                graph.add_edge(left, right, length_m=length, link_id=row.link_id)
    return roads, lookup, graph


def bridge_gap(
    left_id: str,
    right_id: str,
    lookup: dict[str, dict],
    graph: nx.DiGraph,
    maximum_distance_m: float,
    maximum_links: int,
    cache: dict[tuple[str, str], dict],
) -> dict:
    key = (left_id, right_id)
    if key in cache:
        return cache[key]
    left, right = lookup.get(left_id), lookup.get(right_id)
    if left is None or right is None:
        result = {
            "bridge_found": False,
            "bridge_repairable": False,
            "bridge_distance_m": np.nan,
            "bridge_link_count": np.nan,
            "bridge_node_path": "",
            "bridge_reason": "unknown_link",
        }
        cache[key] = result
        return result
    sources = sorted(legal_exits(left["from_node"], left["to_node"], left["oneway_code"]))
    targets = set(legal_entries(right["from_node"], right["to_node"], right["oneway_code"]))
    best: tuple[float, list[int]] | None = None
    for source in sources:
        if source not in graph:
            continue
        distances, paths = nx.single_source_dijkstra(
            graph, source, cutoff=maximum_distance_m, weight="length_m"
        )
        for target in targets:
            if target not in distances:
                continue
            candidate = (float(distances[target]), [int(value) for value in paths[target]])
            if best is None or (candidate[0], len(candidate[1])) < (best[0], len(best[1])):
                best = candidate
    if best is None:
        result = {
            "bridge_found": False,
            "bridge_repairable": False,
            "bridge_distance_m": np.nan,
            "bridge_link_count": np.nan,
            "bridge_node_path": "",
            "bridge_reason": "no_directed_path_within_distance_cap",
        }
    else:
        distance, node_path = best
        link_count = max(0, len(node_path) - 1)
        repairable = link_count <= maximum_links and distance <= maximum_distance_m
        result = {
            "bridge_found": True,
            "bridge_repairable": bool(repairable),
            "bridge_distance_m": distance,
            "bridge_link_count": link_count,
            "bridge_node_path": "|".join(map(str, node_path)),
            "bridge_reason": "bounded_directed_bridge" if repairable else "bridge_exceeds_link_cap",
        }
    cache[key] = result
    return result


def read_parts(directory: Path, columns: list[str]) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(directory)
    return pd.concat([pd.read_parquet(path, columns=columns) for path in files], ignore_index=True)


def classify_day(
    date: str,
    stage0_root: Path,
    lookup: dict[str, dict],
    graph: nx.DiGraph,
    config: dict,
    cache: dict[tuple[str, str], dict],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    routes = read_parts(
        stage0_root / "hmm_route_parts" / f"day={date}",
        [
            "order_id", "link_id", "route_sequence", "transition_path_status",
            "is_interpolated",
        ],
    )
    points = read_parts(
        stage0_root / "hmm_matched_points" / f"day={date}",
        [
            "order_id", "point_seq", "match_confidence", "fallback_used",
            "matcher_version", "proj_dist_m", "source_lon", "source_lat",
            "lon", "lat", "proj_lon", "proj_lat", "segment_distance_m",
            "unreasonable_detour_flag",
        ],
    )
    point_rows = []
    for order_id, group in points.groupby("order_id", sort=False):
        ordered_points = group.sort_values("point_seq")
        metrics = projection_metrics(
            ordered_points.proj_dist_m.to_numpy(),
            ordered_points.lon.to_numpy(),
            ordered_points.lat.to_numpy(),
            ordered_points.proj_lon.to_numpy(),
            ordered_points.proj_lat.to_numpy(),
        )
        modes = ordered_points.matcher_version.mode()
        point_rows.append({
            "order_id": str(order_id),
            "mean_match_confidence": float(ordered_points.match_confidence.mean()),
            "fallback_point_share": float(ordered_points.fallback_used.mean()),
            "matcher_version": modes.iat[0] if len(modes) else "unknown",
            "gps_length_m": float(
                pd.to_numeric(ordered_points.segment_distance_m, errors="coerce").fillna(0).clip(lower=0).sum()
            ),
            "unreasonable_detour_count": int(ordered_points.unreasonable_detour_flag.fillna(False).sum()),
            **metrics,
        })
    point_quality = pd.DataFrame(point_rows)
    point_metrics_by_order = {row["order_id"]: row for row in point_rows}
    extended_cfg = config["extended"]
    link_lengths = {link: float(data["length_m"]) for link, data in lookup.items()}
    source_link_ids = {link: str(data["source_link_id"]) for link, data in lookup.items()}
    gap_rows: list[dict] = []
    order_rows: list[dict] = []
    for order_id, group in routes.groupby("order_id", sort=False):
        ordered = group.sort_values("route_sequence")
        links = ordered.link_id.astype(str).tolist()
        gps_length = point_metrics_by_order.get(str(order_id), {}).get("gps_length_m", np.nan)
        sequence_metrics = route_sequence_metrics(
            links,
            ordered.is_interpolated.fillna(False).to_numpy(),
            link_lengths,
            source_link_ids,
            float(gps_length),
        )
        gaps: list[dict] = []
        for transition_index, (left_id, right_id) in enumerate(zip(links, links[1:]), start=1):
            left, right = lookup.get(left_id), lookup.get(right_id)
            legal = bool(
                left is not None
                and right is not None
                and is_directed_transition(
                    left["from_node"], left["to_node"], left["oneway_code"],
                    right["from_node"], right["to_node"], right["oneway_code"],
                )
            )
            if legal:
                continue
            evidence = bridge_gap(
                left_id,
                right_id,
                lookup,
                graph,
                float(extended_cfg["maximum_bridge_distance_m_per_gap"]),
                int(extended_cfg["maximum_bridge_links_per_gap"]),
                cache,
            )
            row = {
                "date": date,
                "order_id": str(order_id),
                "transition_index": transition_index,
                "from_link_id": left_id,
                "to_link_id": right_id,
                **evidence,
            }
            gaps.append(row)
            gap_rows.append(row)
        transitions = max(0, len(links) - 1)
        gap_count = len(gaps)
        order_rows.append({
            "date": date,
            "order_id": str(order_id),
            "route_link_count": len(links),
            "route_transition_count": transitions,
            "direction_gap_count": gap_count,
            "direction_gap_share": gap_count / max(1, transitions),
            "all_gaps_bridgeable": bool(gap_count > 0 and all(row["bridge_repairable"] for row in gaps)),
            "maximum_bridge_distance_m": max((row["bridge_distance_m"] for row in gaps if pd.notna(row["bridge_distance_m"])), default=np.nan),
            "maximum_bridge_link_count": max((row["bridge_link_count"] for row in gaps if pd.notna(row["bridge_link_count"])), default=np.nan),
            **sequence_metrics,
        })
    quality = pd.DataFrame(order_rows).merge(point_quality, on="order_id", how="left", validate="one_to_one")
    core_cfg = config["core"]
    v4_metrics = "maximum_p90_projection_distance_m" in core_cfg
    if v4_metrics:
        flags = pd.DataFrame(
            [core_threshold_flags(row, core_cfg) for row in quality.to_dict("records")],
            index=quality.index,
        )
        quality = pd.concat([quality, flags], axis=1)
        quality["core_route_link_count_ok"] = quality.route_link_count.ge(
            int(core_cfg["minimum_route_links"])
        )
        core = quality.core_all_thresholds_pass & quality.core_route_link_count_ok
        extended_flags = pd.DataFrame(
            [core_threshold_flags(row, extended_cfg) for row in quality.to_dict("records")],
            index=quality.index,
        ).rename(columns=lambda column: column.replace("core_", "extended_"))
        quality = pd.concat([quality, extended_flags], axis=1)
        quality["extended_route_link_count_ok"] = quality.route_link_count.ge(
            int(core_cfg["minimum_route_links"])
        )
        extended_common = quality.extended_all_thresholds_pass & quality.extended_route_link_count_ok
    else:
        common_core = (
            quality.route_link_count.ge(int(core_cfg["minimum_route_links"]))
            & quality.mean_match_confidence.ge(float(core_cfg["minimum_mean_match_confidence"]))
            & quality.fallback_point_share.le(float(core_cfg["maximum_fallback_point_share"]))
        )
        core = common_core & quality.direction_gap_count.le(int(core_cfg["maximum_direction_gaps"]))
        extended_common = (
            quality.route_link_count.ge(int(core_cfg["minimum_route_links"]))
            & quality.mean_match_confidence.ge(float(extended_cfg["minimum_mean_match_confidence"]))
            & quality.fallback_point_share.le(float(extended_cfg["maximum_fallback_point_share"]))
        )
    extended = (
        ~core
        & extended_common
        & quality.direction_gap_count.between(1, int(extended_cfg["maximum_direction_gaps"]))
        & quality.direction_gap_share.le(float(extended_cfg["maximum_gap_share"]))
        & quality.all_gaps_bridgeable
    )
    quality["route_quality_class"] = np.select([core, extended], ["core", "extended"], default="rejected")
    quality["formal_training_eligible"] = core
    quality["robustness_only_eligible"] = extended
    reason_conditions = [
        core,
        extended,
        quality.mean_match_confidence.lt(float(extended_cfg["minimum_mean_match_confidence"])),
        quality.fallback_point_share.gt(float(extended_cfg["maximum_fallback_point_share"])),
        quality.direction_gap_count.gt(int(extended_cfg["maximum_direction_gaps"])),
        quality.direction_gap_count.gt(0) & ~quality.all_gaps_bridgeable,
    ]
    reason_values = [
        "fully_directed_continuous_and_quality_thresholds_passed",
        "bounded_graph_bridgeable_gap",
        "low_match_confidence",
        "excessive_geometric_fallback",
        "too_many_direction_gaps",
        "unbridgeable_direction_gap",
    ]
    if v4_metrics:
        reason_conditions.extend([
            ~quality.core_projection_ok,
            ~quality.core_route_length_ratio_ok,
            ~quality.core_interpolation_ok,
            ~(quality.core_origin_error_ok & quality.core_destination_error_ok),
            ~quality.core_no_unreasonable_detour,
            ~quality.core_u_turn_ok,
            ~quality.core_repeated_link_ok,
        ])
        reason_values.extend([
            "projection_distance_threshold_failure",
            "route_length_ratio_threshold_failure",
            "interpolated_distance_share_threshold_failure",
            "od_endpoint_projection_threshold_failure",
            "unreasonable_detour_detected",
            "u_turn_detected",
            "repeated_link_share_threshold_failure",
        ])
    quality["quality_reason"] = np.select(
        reason_conditions, reason_values, default="route_quality_threshold_failure"
    )
    if v4_metrics:
        final_layers = pd.DataFrame(
            [classify_quality_layer(row) for row in quality.to_dict("records")],
            index=quality.index,
        )
        quality = pd.concat([quality, final_layers], axis=1)
    counts = quality.route_quality_class.value_counts().to_dict()
    final_counts = (
        quality.route_quality_class_v4_final.value_counts().to_dict()
        if "route_quality_class_v4_final" in quality else {}
    )
    failed_files = sorted((stage0_root / "failed_orders" / f"day={date}").glob("*.parquet"))
    failed_orders = sum(
        len(pd.read_parquet(path, columns=["order_id"])) for path in failed_files
    )
    input_orders = len(quality) + failed_orders
    summary = {
        "date": date,
        "input_orders": int(input_orders),
        "successfully_reconstructed_orders": int(len(quality)),
        "failed_match_orders": int(failed_orders),
        "reconstruction_coverage": float(len(quality) / max(1, input_orders)),
        "orders": int(len(quality)),
        "core_orders": int(counts.get("core", 0)),
        "extended_orders": int(counts.get("extended", 0)),
        "rejected_orders": int(counts.get("rejected", 0)),
        "core_share_of_input": float(counts.get("core", 0) / max(1, input_orders)),
        "failed_match_share_of_input": float(failed_orders / max(1, input_orders)),
        "core_share": float(quality.route_quality_class.eq("core").mean()),
        "extended_share": float(quality.route_quality_class.eq("extended").mean()),
        "rejected_share": float(quality.route_quality_class.eq("rejected").mean()),
        "strict_core_orders_v4_final_candidate": int(final_counts.get("strict_core", 0)),
        "analysis_set_orders_v4_final_candidate": int(final_counts.get("analysis_set", 0)),
        "rejected_orders_v4_final_candidate": int(final_counts.get("rejected", 0)),
        "formal_analysis_eligible_share": float(
            quality.formal_analysis_eligible.mean()
        ) if "formal_analysis_eligible" in quality else None,
        "orders_with_gap": int(quality.direction_gap_count.gt(0).sum()),
        "orders_with_all_gaps_bridgeable": int(quality.all_gaps_bridgeable.sum()),
        "mean_direction_gaps": float(quality.direction_gap_count.mean()),
        "geometric_fallback_orders": int(quality.matcher_version.str.contains("fallback", case=False, na=False).sum()),
    }
    return quality, pd.DataFrame(gap_rows), summary


def main() -> None:
    args = arguments()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _, lookup, graph = load_network(args.roads)
    cache: dict[tuple[str, str], dict] = {}
    summaries = []
    all_gaps = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    for date in [value.strip() for value in args.dates.split(",") if value.strip()]:
        quality, gaps, summary = classify_day(date, args.stage0_root, lookup, graph, config, cache)
        target = args.output_root / "order_route_quality" / f"day={date}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        quality.to_parquet(target, index=False, compression="zstd")
        summaries.append(summary)
        all_gaps.append(gaps)
    summary_frame = pd.DataFrame(summaries)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(args.summary, index=False, encoding="utf-8-sig")
    args.gap_details.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_gaps, ignore_index=True).to_csv(args.gap_details, index=False, encoding="utf-8-sig")
    result = {
        "status": "DIAGNOSTIC_PASS",
        "schema_version": config["schema_version"],
        "classification_config": args.config.as_posix(),
        "network": args.roads.as_posix(),
        "days": summaries,
        "gap_pair_cache_size": len(cache),
        "manual_truth_audit": "AWAITING_HUMAN_REVIEW",
        "canonical_promotion_gate": "HOLD",
        "promotion_blockers": [
            "Full-data Core/Extended coverage has not been computed.",
            "Versioned manual route truth sample has not been reviewed.",
            "Bridge evidence is classification-only; no route is silently repaired.",
        ],
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
