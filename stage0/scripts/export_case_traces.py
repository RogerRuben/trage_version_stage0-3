"""Export a small auditable set of point/route traces before daily pruning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_full_day_2017 import order_bucket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--buckets", type=int, default=128)
    parser.add_argument("--comparison-collection", default="matcher_comparison")
    parser.add_argument("--matched-collection", default="hmm_matched_points")
    parser.add_argument("--route-collection", default="hmm_route_parts")
    parser.add_argument("--traversal-collection", default="hmm_link_traversals")
    parser.add_argument("--movement-collection", default="hmm_turn_movements")
    parser.add_argument("--poi-behavior-collection", default="stage0_order_link_poi_behavior")
    parser.add_argument("--matched-suffix", default="hmm_points")
    return parser.parse_args()


def choose(candidates: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    output = []; used = set()
    for label, order_id, reason in candidates:
        if order_id and order_id not in used:
            output.append((label, str(order_id), reason)); used.add(str(order_id))
    return output


def main() -> None:
    args = parse_args(); root = args.output_root
    comparison_path = root / args.comparison_collection / f"day={args.date}" / "order_comparison.parquet"
    comparison = pd.read_parquet(comparison_path)
    candidates: list[tuple[str, str, str]] = []
    improvement = comparison.assign(delta=comparison.geo_topology_gap_count - comparison.topology_gap_count)
    if len(improvement):
        row = improvement.sort_values(["delta", "link_sequence_change_ratio"], ascending=False).iloc[0]
        candidates.append(("hmm_improvement", row.order_id, f"topology_gap_reduction={row.delta}"))
    fallback = comparison[comparison.fallback_used]
    if len(fallback):
        row = fallback.sort_values("geo_match_confidence").iloc[0]
        candidates.append(("hmm_fallback", row.order_id, str(row.fallback_reason)))

    traversal_summaries = []
    traversal_dir = root / args.traversal_collection / f"day={args.date}"
    for path in sorted(traversal_dir.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=[
            "order_id", "low_speed_ratio", "stop_duration_ratio", "speed_cv", "travel_time_sec",
            "observed_distance_m", "traversal_quality",
        ])
        summary = frame.groupby("order_id").agg(
            low_speed=("low_speed_ratio", "mean"), stop_ratio=("stop_duration_ratio", "mean"),
            speed_cv=("speed_cv", "median"), duration=("travel_time_sec", "sum"),
            distance=("observed_distance_m", "sum"),
            high_quality=("traversal_quality", lambda x: x.eq("high").mean()),
        )
        summary["part"] = path.stem.split("=")[-1].split("_")[-1]
        traversal_summaries.append(summary.reset_index())
    traversals = pd.concat(traversal_summaries, ignore_index=True)
    traversals["lcs_proxy"] = (
        traversals[["low_speed", "stop_ratio"]].mean(axis=1)
        + traversals.speed_cv.fillna(0).clip(0, 2) / 4
    ).clip(0, 1.5)
    traversals["sec_per_km"] = traversals.duration / (traversals.distance / 1000).replace(0, np.nan)
    behavior_eligible = traversals.distance.ge(500) & traversals.high_quality.ge(0.5)
    eligible = traversals[behavior_eligible] if behavior_eligible.any() else traversals
    row = eligible.sort_values("lcs_proxy", ascending=False).iloc[0]
    candidates.append(("high_lcs_candidate", row.order_id, f"lcs_proxy={row.lcs_proxy:.3f}"))
    rts_eligible = eligible[eligible.sec_per_km.between(60, 3600)]
    row = (rts_eligible if len(rts_eligible) else eligible).sort_values("sec_per_km", ascending=False).iloc[0]
    candidates.append(("high_rts_candidate", row.order_id, f"sec_per_km={row.sec_per_km:.1f}"))
    smooth = eligible[eligible.high_quality.ge(0.8)].sort_values("lcs_proxy")
    if len(smooth): candidates.append(("normal_smooth", smooth.iloc[0].order_id, "low_lcs_high_quality"))

    movement_summaries = []
    for path in sorted((root / args.movement_collection / f"day={args.date}").glob("*.parquet")):
        frame = pd.read_parquet(path, columns=[
            "order_id", "turn_angle", "node_degree", "intersection_low_speed_time", "movement_quality"
        ])
        summary = frame.groupby("order_id").agg(
            turn_angle=("turn_angle", lambda x: x.abs().mean()), node_degree=("node_degree", "mean"),
            intersection_delay=("intersection_low_speed_time", "sum"),
            usable=("movement_quality", lambda x: x.eq("usable").mean()),
        ).reset_index()
        movement_summaries.append(summary)
    if movement_summaries:
        movements = pd.concat(movement_summaries, ignore_index=True)
        movements["iis_proxy"] = movements.intersection_delay.rank(pct=True) + movements.turn_angle.rank(pct=True)
        row = movements.sort_values("iis_proxy", ascending=False).iloc[0]
        candidates.append(("high_iis_candidate", row.order_id, f"iis_proxy={row.iis_proxy:.3f}"))

    behavior_summaries = []
    for path in sorted((root / args.poi_behavior_collection / f"day={args.date}").glob("*.parquet")):
        frame = pd.read_parquet(path, columns=[
            "order_id", "activity_intensity_index", "low_speed_ratio_on_poi_link",
            "stop_time_on_poi_link", "poi_interaction_candidate",
        ])
        behavior_summaries.append(frame.groupby("order_id").agg(
            activity=("activity_intensity_index", "mean"), poi_low_speed=("low_speed_ratio_on_poi_link", "mean"),
            poi_stop=("stop_time_on_poi_link", "sum"), interactions=("poi_interaction_candidate", "sum"),
        ).reset_index())
    if behavior_summaries:
        behavior = pd.concat(behavior_summaries, ignore_index=True)
        stop_component = np.log1p(behavior.poi_stop.clip(lower=0)) / np.log(601)
        behavior["pmis_proxy"] = (
            behavior.activity.clip(0, 1) * (behavior.poi_low_speed.clip(0, 1) + stop_component.clip(0, 1)) / 2
        ).clip(0, 1)
        row = behavior.sort_values("pmis_proxy", ascending=False).iloc[0]
        candidates.append(("high_pmis_candidate", row.order_id, f"pmis_proxy={row.pmis_proxy:.3f}"))

    order_base = pd.read_parquet(root / "order_base" / f"day={args.date}.parquet")
    row = order_base.sort_values(["quality_tier", "matching_confidence"], ascending=[False, True]).iloc[0]
    candidates.append(("low_quality_matching", row.order_id, f"quality={row.quality_tier}"))
    selected = choose(candidates)

    case_dir = root / "case_traces" / f"day={args.date}"; case_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for label, order_id, reason in selected:
        bucket = int(order_bucket(pd.Series([order_id], dtype="string"), args.buckets)[0])
        part = f"{bucket:03d}"
        geo_path = next(iter((root / "matched_points" / f"day={args.date}").glob(f"*{part}.parquet")))
        hmm_path = next(iter((root / args.matched_collection / f"day={args.date}").glob(f"*{part}.parquet")))
        route_path = next(iter((root / args.route_collection / f"day={args.date}").glob(f"*{part}.parquet")))
        geo = pd.read_parquet(geo_path, filters=[[('order_id', '==', order_id)]])
        hmm = pd.read_parquet(hmm_path, filters=[[('order_id', '==', order_id)]])
        route = pd.read_parquet(route_path, filters=[[('order_id', '==', order_id)]])
        prefix = f"order_id={order_id}"
        geo.to_parquet(case_dir / f"{prefix}_geo_points.parquet", index=False, compression="zstd")
        hmm.to_parquet(case_dir / f"{prefix}_{args.matched_suffix}.parquet", index=False, compression="zstd")
        route.to_parquet(case_dir / f"{prefix}_route.parquet", index=False, compression="zstd")
        index_rows.append({"case_type": label, "order_id": order_id, "bucket": bucket, "reason": reason})
    pd.DataFrame(index_rows).to_csv(case_dir / "case_index.csv", index=False)
    print(pd.DataFrame(index_rows).to_string(index=False))


if __name__ == "__main__":
    main()
