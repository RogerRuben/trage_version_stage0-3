"""Put the actual-route oracle on a dispatch-time estimated-entry clock.

The route itself remains post-trip oracle information.  Entry times and all
state joins are nevertheless reconstructed causally so the product isolates
route-choice error from traffic-state/target predictability.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_stage2_od_planned_routes import is_peak, road_graph  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual-root", type=Path, default=Path("stage2/output/routes/actual_route_oracle"))
    parser.add_argument("--order-od-root", type=Path, default=Path("stage0/output/order_od_audited"))
    parser.add_argument("--roads", type=Path, default=Path("map_data/xian_2017/xian_2017_core_roads.parquet"))
    parser.add_argument("--reference-root", type=Path, default=Path("stage1/output/prediction_split/models/travel_time_reference"))
    parser.add_argument("--poi-exposure", type=Path, default=Path("stage0/output/poi/stage0_link_poi_exposure.parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/routes/actual_route_planned_time_oracle"))
    parser.add_argument("--dates", default="20161009,20161010,20161011,20161012,20161013,20161014,20161015,20161016,20161017,20161018,20161019")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    roads, _, _, _ = road_graph(args.roads, args.reference_root, args.poi_exposure, "all")
    lookup = roads.set_index("link_id")
    static = [
        "road_class", "area_grid", "planned_link_length_m", "endpoint_degree", "link_fragmentation", "minor_road",
        "activity_intensity_index",
    ] + [column for column in roads.columns if column.startswith("poi_density_100m_")]
    manifest = {"route_source": "actual_route_oracle", "entry_clock": "dispatch-time historical reference", "deployable": False, "days": {}}
    for date in [part.strip() for part in args.dates.split(",") if part.strip()]:
        actual = pd.read_parquet(args.actual_root / f"day={date}.parquet").sort_values(["order_id", "actual_link_seq"])
        od = pd.read_parquet(args.order_od_root / f"day={date}.parquet", columns=["order_id", "origin_timestamp"])
        frame = actual.merge(od, on="order_id", how="left", validate="many_to_one")
        frame["planned_link_id"] = frame["actual_link_id"].astype(str)
        frame["planned_link_seq"] = frame["actual_link_seq"].astype("int32")
        frame = frame.join(lookup[static], on="planned_link_id", rsuffix="_road")
        dispatch = pd.to_datetime(frame["origin_timestamp"], unit="s", utc=True).dt.round("us")
        order_dispatch = dispatch.groupby(frame["order_id"]).transform("first")
        frame["dispatch_time"] = order_dispatch
        peak = order_dispatch.map(is_peak)
        sec_peak = frame["planned_link_id"].map(lookup["sec_per_m_peak"])
        sec_offpeak = frame["planned_link_id"].map(lookup["sec_per_m_offpeak"])
        frame["estimated_link_travel_time_sec"] = np.where(peak, sec_peak, sec_offpeak) * frame["planned_link_length_m"]
        elapsed = frame.groupby("order_id")["estimated_link_travel_time_sec"].transform(lambda value: value.shift(fill_value=0).cumsum())
        frame["estimated_link_entry_time"] = (order_dispatch + pd.to_timedelta(elapsed, unit="s")).dt.round("us")
        frame["planned_route_link_count"] = frame.groupby("order_id")["planned_link_id"].transform("size")
        frame["position_ratio"] = frame["planned_link_seq"] / (frame["planned_route_link_count"] - 1).clip(lower=1)
        total = frame.groupby("order_id")["planned_link_length_m"].transform("sum").clip(lower=1)
        frame["distance_to_destination_ratio"] = 1 - frame.groupby("order_id")["planned_link_length_m"].cumsum() / total
        frame["route_source"] = "actual_route_oracle"
        frame["routing_fallback"] = "post_trip_oracle"
        frame["realized_label_available"] = frame["target_lcs_raw"].notna()
        frame.to_parquet(args.output_root / f"day={date}.parquet", index=False, compression="zstd")
        manifest["days"][date] = {
            "orders": int(frame["order_id"].nunique()), "rows": len(frame),
            "estimated_entry_time_coverage": float(frame["estimated_link_entry_time"].notna().mean()),
            "realized_label_link_ratio": float(frame["realized_label_available"].mean()),
        }
        print(f"actual-route planned-time oracle day={date} {manifest['days'][date]}", flush=True)
    (args.output_root / "actual_route_planned_time_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
