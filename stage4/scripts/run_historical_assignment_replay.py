"""Replay full-day historical assignments for ABM supply sanity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders", type=Path, default=Path("stage4/data/test_day_20161023_historical_orders.parquet"))
    parser.add_argument("--sessions", type=Path, default=Path("stage4/data/hv_agent_sessions_20161023.parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("stage4/output/single_day_20161023/historical_replay"))
    parser.add_argument("--results-dir", type=Path, default=Path("stage4/docs/results"))
    return parser.parse_args()


def _haversine_m(lon1, lat1, lon2, lat2):
    r = 6371000.0
    lon1 = np.radians(lon1); lat1 = np.radians(lat1); lon2 = np.radians(lon2); lat2 = np.radians(lat2)
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    orders = pd.read_parquet(args.orders)
    sessions = pd.read_parquet(args.sessions)
    orders["origin_time"] = pd.to_datetime(orders["origin_timestamp"], unit="s", utc=True, errors="coerce")
    orders["destination_time"] = pd.to_datetime(orders["destination_timestamp"], unit="s", utc=True, errors="coerce")
    sessions["online_start"] = pd.to_datetime(sessions["online_start"], utc=True, errors="coerce")
    sessions["online_end"] = pd.to_datetime(sessions["online_end"], utc=True, errors="coerce")
    session_map = {
        driver: group.sort_values("online_start")
        for driver, group in sessions.groupby("driver_id", sort=False)
    }
    rows = []
    for _, order in orders.iterrows():
        driver = order["historical_driver_id"]
        group = session_map.get(driver)
        if group is None:
            rows.append({"order_id": order["order_id"], "driver_id": driver, "replay_status": "missing_driver"})
            continue
        mask = group["online_start"].le(order["origin_time"]) & group["online_end"].ge(order["destination_time"])
        if not mask.any():
            rows.append({"order_id": order["order_id"], "driver_id": driver, "replay_status": "cross_session_conflict"})
            continue
        session = group.loc[mask].iloc[0]
        rows.append({"order_id": order["order_id"], "driver_id": driver, "session_id": session["session_id"], "replay_status": "candidate"})
    replay = pd.DataFrame(rows)
    merged = orders.merge(replay, on=["order_id"], how="left")
    conflict_flags = []
    spatial_jumps = []
    for driver, group in merged[merged["replay_status"].eq("candidate")].sort_values(["historical_driver_id", "origin_time"]).groupby("historical_driver_id"):
        prev_end = group["destination_time"].shift()
        overlap = group["origin_time"].lt(prev_end)
        conflict_flags.extend(overlap.fillna(False).tolist())
        prev_lon = group["destination_lon"].shift()
        prev_lat = group["destination_lat"].shift()
        dist = _haversine_m(prev_lon, prev_lat, group["origin_lon"], group["origin_lat"])
        spatial_jumps.extend(pd.Series(dist).fillna(0).tolist())
    merged.loc[merged["replay_status"].eq("candidate"), "time_overlap_conflict"] = conflict_flags
    merged.loc[merged["replay_status"].eq("candidate"), "previous_dropoff_to_origin_m"] = spatial_jumps
    merged["time_overlap_conflict"] = merged["time_overlap_conflict"].fillna(False).astype(bool)
    merged["spatial_continuity_conflict"] = pd.to_numeric(merged["previous_dropoff_to_origin_m"], errors="coerce").gt(6000).fillna(False)
    merged["replay_success"] = merged["replay_status"].eq("candidate") & ~merged["time_overlap_conflict"]
    merged.to_parquet(args.output_root / "order_replay_log.parquet", index=False, compression="zstd")
    total = len(merged)
    summary = {
        "total_historical_orders": int(total),
        "replayed_orders": int(merged["replay_success"].sum()),
        "replay_success_rate": float(merged["replay_success"].mean()),
        "time_conflict_rate": float(merged["time_overlap_conflict"].mean()),
        "spatial_continuity_conflict_rate": float(merged["spatial_continuity_conflict"].mean()),
        "missing_driver_rate": float(merged["replay_status"].eq("missing_driver").mean()),
        "cross_session_conflict_rate": float(merged["replay_status"].eq("cross_session_conflict").mean()),
        "overlapping_service_rate": float(merged["time_overlap_conflict"].mean()),
        "p95_spatial_jump_m": float(pd.to_numeric(merged["previous_dropoff_to_origin_m"], errors="coerce").quantile(0.95)),
    }
    pd.DataFrame([summary]).to_csv(args.results_dir / "historical_replay_full_day_summary.csv", index=False)
    (args.output_root / "historical_replay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = ["# Historical Assignment Replay", "", pd.DataFrame([summary]).to_markdown(index=False, floatfmt=".4f")]
    Path("stage4/docs/historical_assignment_replay_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
