"""Build full strictly lagged link/area/network state lookup tables.

State at five-minute bin ``b`` uses only completed bins before ``b``. The lookup
is independent of any current order and can therefore be joined to planned
routes by estimated entry time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_stage2_lagged_traffic_features import (  # noqa: E402
    METRICS, aggregate_bins, directed_adjacency, group_window_features, neighbor_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primitive-root", type=Path, default=Path("stage1/output/prediction_split/primitives"))
    parser.add_argument("--roads", type=Path, default=Path("map_data/xian_2017/xian_2017_core_roads.parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/lagged_state_store"))
    parser.add_argument("--dates", default="20161009,20161010,20161011,20161012,20161013,20161014,20161015,20161016,20161017,20161018,20161019")
    parser.add_argument("--bin-minutes", type=int, default=5)
    return parser.parse_args()


def combine(parts: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    frame = pd.concat(parts, ignore_index=True)
    value_columns = [column for column in frame.columns if column not in keys + ["bin_time"]]
    return frame.groupby(keys + ["bin_time"], as_index=False, dropna=False, observed=True)[value_columns].sum()


def roll_aggregates(aggregates: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    if keys:
        grouper = keys[0] if len(keys) == 1 else keys
        parts = [group_window_features(group, prefix) for _, group in aggregates.groupby(grouper, observed=True, sort=False)]
        return pd.concat(parts, ignore_index=True)
    return group_window_features(aggregates, prefix)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    dates = [part.strip() for part in args.dates.split(",") if part.strip()]
    all_link, all_area, all_network = [], [], []
    day_rows = {}
    requested = ["enter_time", "link_id", "area_grid"] + METRICS
    for date in dates:
        link_parts, area_parts, network_parts = [], [], []
        rows = 0
        for path in sorted((args.primitive_root / f"day={date}").glob("*.parquet")):
            frame = pd.read_parquet(path, columns=requested)
            frame["link_id"] = frame["link_id"].astype(str)
            frame["area_grid"] = frame["area_grid"].astype("string")
            frame["target_prediction_timestamp"] = pd.to_datetime(frame["enter_time"], unit="s", utc=True, errors="coerce")
            frame = frame[frame["target_prediction_timestamp"].notna()].copy()
            frame["bin_time"] = frame["target_prediction_timestamp"].dt.floor(f"{args.bin_minutes}min")
            link_parts.append(aggregate_bins(frame, ["link_id"]))
            area_parts.append(aggregate_bins(frame, ["area_grid"]))
            network_parts.append(aggregate_bins(frame, []))
            rows += len(frame)
        if not link_parts:
            raise FileNotFoundError(f"no primitives for day={date}")
        all_link.append(combine(link_parts, ["link_id"]))
        all_area.append(combine(area_parts, ["area_grid"]))
        all_network.append(combine(network_parts, []))
        day_rows[date] = rows
        print(f"state aggregation day={date} rows={rows:,}", flush=True)
    link_aggregates = pd.concat(all_link, ignore_index=True).sort_values(["link_id", "bin_time"])
    area_aggregates = pd.concat(all_area, ignore_index=True).sort_values(["area_grid", "bin_time"])
    network_aggregates = pd.concat(all_network, ignore_index=True).sort_values("bin_time")
    link = roll_aggregates(link_aggregates, ["link_id"], "link")
    area = roll_aggregates(area_aggregates, ["area_grid"], "area")
    network = roll_aggregates(network_aggregates, [], "network")
    link_duplicate_rows = int(link.duplicated(["link_id", "bin_time"], keep=False).sum())
    if link_duplicate_rows:
        # Duplicate OSM link IDs can occur when the source line layer contains
        # multiple geometries for one logical link.  Their completed-bin state
        # is identical by key here; collapse deterministically before topology
        # propagation and record the event in the manifest.
        link = link.groupby(["link_id", "bin_time"], as_index=False, observed=True).first()
    upstream_adjacency, downstream_adjacency = directed_adjacency(args.roads)
    upstream = neighbor_features(link, upstream_adjacency, "upstream")
    downstream = neighbor_features(link, downstream_adjacency, "downstream")
    link = link.merge(upstream, on=["link_id", "bin_time"], how="left", validate="one_to_one")
    link = link.merge(downstream, on=["link_id", "bin_time"], how="left", validate="one_to_one")
    for frame in [link, area, network]:
        frame["feature_timestamp"] = frame["bin_time"]
        frame["availability_timestamp"] = frame["bin_time"] - pd.Timedelta(milliseconds=1)
    link.to_parquet(args.output_root / "link_state.parquet", index=False, compression="zstd")
    area.to_parquet(args.output_root / "area_state.parquet", index=False, compression="zstd")
    network.to_parquet(args.output_root / "network_state.parquet", index=False, compression="zstd")
    manifest = {
        "dates": dates, "bin_minutes": args.bin_minutes, "same_bin_excluded": True,
        "availability_rule": "availability_timestamp < planned estimated link entry time",
        "physical_upstream_downstream": True, "source_rows_by_day": day_rows,
        "link_state_duplicate_rows_collapsed": link_duplicate_rows,
        "lookup_rows": {"link": len(link), "area": len(area), "network": len(network)},
    }
    (args.output_root / "lagged_state_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
