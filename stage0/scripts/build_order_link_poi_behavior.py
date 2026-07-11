"""Merge link traversal behavior with static link-level POI exposure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traversal-dir", type=Path, required=True)
    parser.add_argument("--poi-exposure", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("stage0_output"))
    parser.add_argument("--date", required=True)
    parser.add_argument("--activity-threshold", type=float, default=0.75)
    parser.add_argument("--output-collection", default="stage0_order_link_poi_behavior")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exposure = pd.read_parquet(args.poi_exposure)
    keep = ["link_id", "activity_intensity_index"] + [
        f"poi_density_100m_{category}"
        for category in ["school", "hospital", "commercial", "restaurant", "transit", "bus_stop", "residential", "office", "scenic", "parking"]
    ]
    exposure = exposure[keep].rename(columns={
        "poi_density_100m_school": "poi_density_school",
        "poi_density_100m_hospital": "poi_density_hospital",
        "poi_density_100m_commercial": "poi_density_commercial",
        "poi_density_100m_restaurant": "poi_density_restaurant",
        "poi_density_100m_transit": "poi_density_transit",
        "poi_density_100m_bus_stop": "poi_density_bus_stop",
        "poi_density_100m_residential": "poi_density_residential",
        "poi_density_100m_office": "poi_density_office",
        "poi_density_100m_scenic": "poi_density_scenic",
        "poi_density_100m_parking": "poi_density_parking",
    })
    output_dir = args.output_root / args.output_collection / f"day={args.date}"
    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for source in sorted(args.traversal_dir.glob("*.parquet")):
        part = source.stem.split("=")[-1].split("_")[-1]
        target = output_dir / f"part={part}.parquet"
        if target.exists() and not args.force:
            continue
        traversal = pd.read_parquet(source)
        merged = traversal.merge(exposure, on="link_id", how="left", validate="many_to_one")
        density_columns = [column for column in merged.columns if column.startswith("poi_density_")]
        merged[density_columns + ["activity_intensity_index"]] = merged[
            density_columns + ["activity_intensity_index"]
        ].fillna(0)
        exposed = merged.activity_intensity_index.ge(args.activity_threshold)
        merged["low_speed_ratio_on_poi_link"] = merged.low_speed_ratio.where(exposed, 0.0)
        merged["stop_time_on_poi_link"] = merged.stop_time_sec.where(exposed, 0.0)
        merged["delay_on_poi_link"] = np.nan
        merged["delay_reference_status"] = "pending_monthly_link_baseline"
        merged["poi_interaction_candidate"] = exposed & (
            merged.low_speed_ratio.ge(0.25) | merged.stop_time_sec.gt(0)
        )
        columns = [
            "order_id", "driver_id", "date", "link_id", "link_seq", "enter_time", "exit_time",
            *density_columns, "activity_intensity_index", "low_speed_ratio_on_poi_link",
            "stop_time_on_poi_link", "delay_on_poi_link", "delay_reference_status",
            "poi_interaction_candidate", "traversal_quality", "matcher_version",
        ]
        merged[columns].to_parquet(target, index=False, compression="zstd")
        total += len(merged)
        print(f"part={part} rows={len(merged):,}", flush=True)
    manifest = {"date": args.date, "complete": True, "rows_written_this_run": total, "poi_buffer_m": 100}
    manifest_dir = args.output_root / "manifests"; manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"day={args.date}.poi_behavior.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
