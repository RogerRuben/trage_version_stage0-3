"""Expand non-adjacent HMM state transitions into auditable network link paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hmm_viterbi_matcher import HMMRoadNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-dir", type=Path, required=True)
    parser.add_argument("--roads", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def expand_order(group: pd.DataFrame, network: HMMRoadNetwork, lookup: pd.Series) -> list[dict]:
    group = group.sort_values("point_seq")
    changed = group.link_id.ne(group.link_id.shift())
    states = group.loc[changed, ["order_id", "link_id", "point_seq", "timestamp"]]
    if states.empty: return []
    records = [{
        "order_id": states.order_id.iloc[0], "link_id": states.link_id.iloc[0],
        "source_point_seq": int(states.point_seq.iloc[0]), "timestamp": int(states.timestamp.iloc[0]),
        "is_interpolated": False, "transition_path_status": "start",
    }]
    previous = states.iloc[0]
    for _, current in states.iloc[1:].iterrows():
        a = int(lookup[str(previous.link_id)]); b = int(lookup[str(current.link_id)])
        if network.is_directed_link_transition(a, b):
            path, status = (a, b), "direct_topology"
        else:
            path, status = network.link_path_indices(a, b)
        link_ids = network.roads.link_id.to_numpy()[list(path)]
        for link_id in link_ids[1:-1]:
            records.append({
                "order_id": current.order_id, "link_id": link_id,
                "source_point_seq": int(current.point_seq), "timestamp": int(current.timestamp),
                "is_interpolated": True, "transition_path_status": status,
            })
        records.append({
            "order_id": current.order_id, "link_id": current.link_id,
            "source_point_seq": int(current.point_seq), "timestamp": int(current.timestamp),
            "is_interpolated": False, "transition_path_status": status,
        })
        previous = current
    for sequence, record in enumerate(records): record["route_sequence"] = sequence
    return records


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    network = HMMRoadNetwork(args.roads, args.nodes)
    lookup = pd.Series(range(len(network.roads)), index=network.roads.link_id.astype(str))
    totals = {"partitions": 0, "rows": 0, "interpolated_rows": 0}
    for source in sorted(args.matched_dir.glob("*.parquet")):
        part = source.stem.split("=")[-1].split("_")[-1]
        target = args.output_dir / f"part={part}.parquet"
        if target.exists() and not args.force: continue
        frame = pd.read_parquet(source, columns=["order_id", "link_id", "point_seq", "timestamp"])
        records = []
        for _, group in frame.groupby("order_id", sort=False):
            records.extend(expand_order(group, network, lookup))
        routes = pd.DataFrame(records)
        routes.to_parquet(target, index=False, compression="zstd")
        totals["partitions"] += 1; totals["rows"] += len(routes)
        totals["interpolated_rows"] += int(routes.is_interpolated.sum())
        print(f"part={part} rows={len(routes):,} interpolated={routes.is_interpolated.sum():,}", flush=True)
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
