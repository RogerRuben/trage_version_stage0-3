"""Reusable initial fleet snapshots for fair Stage4 scenario comparison."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def zone_id(lon: float, lat: float, grid: float) -> str:
    if pd.isna(lon) or pd.isna(lat):
        return "unknown"
    return f"{math.floor(float(lon) / grid)}:{math.floor(float(lat) / grid)}"


def fleet_hash(frame: pd.DataFrame) -> str:
    cols = [
        "vehicle_id",
        "vehicle_type",
        "initial_lon",
        "initial_lat",
        "initial_zone",
        "shift_start",
        "shift_end",
        "available_time",
        "initial_status",
    ]
    payload = frame[cols].sort_values("vehicle_id").to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def snapshot_path(root: Path, fold: int, supply: str, av_penetration: float) -> Path:
    return root / f"fold={fold}" / f"supply={supply}" / f"av_penetration={av_penetration:.2f}"


def build_initial_fleet_snapshot(
    orders: pd.DataFrame,
    total_fleet: int,
    av_penetration: float,
    grid: float,
    seed: int,
    fold: int,
    supply: str,
    output_root: Path,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Build or load one deterministic fleet snapshot.

    The same `(fold, supply, av_penetration, seed)` tuple always maps to the same
    vehicle ids, initial coordinates, shifts, and online status. Experiments read
    and deep-copy this snapshot; they must not advance a shared RNG stream.
    """

    root = snapshot_path(output_root, fold, supply, av_penetration)
    fleet_file = root / "initial_fleet.parquet"
    manifest_file = root / "manifest.json"
    if fleet_file.exists() and manifest_file.exists() and not overwrite:
        frame = pd.read_parquet(fleet_file)
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        return frame, manifest

    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    av_count = int(round(total_fleet * av_penetration))
    start = orders["decision_time"].min()
    end = orders["decision_time"].max() + pd.Timedelta(hours=3)
    first_window = orders[orders["decision_time"].le(start + pd.Timedelta(minutes=30))]
    seed_points = first_window[["origin_lon", "origin_lat"]].dropna()
    source = "first_30min_observable_demand"
    if seed_points.empty:
        seed_points = orders[["origin_lon", "origin_lat"]].dropna().head(1)
        source = "first_available_observable_origin"
    sample = seed_points.sample(total_fleet, replace=True, random_state=seed)
    rows = []
    for idx, (_, row) in enumerate(sample.iterrows()):
        vehicle_type = "AV" if idx < av_count else "HV"
        shift_start = start if vehicle_type == "AV" else start + pd.Timedelta(minutes=int(rng.integers(0, 120)))
        shift_end = end if vehicle_type == "AV" else end - pd.Timedelta(minutes=int(rng.integers(0, 120)))
        lon = float(row["origin_lon"])
        lat = float(row["origin_lat"])
        rows.append({
            "vehicle_id": f"{vehicle_type}_{idx}",
            "vehicle_type": vehicle_type,
            "initial_lon": lon,
            "initial_lat": lat,
            "initial_zone": zone_id(lon, lat, grid),
            "shift_start": str(shift_start),
            "shift_end": str(shift_end),
            "available_time": str(shift_start),
            "initial_status": "OFFLINE" if shift_start > start else "IDLE",
            "initialization_source": source,
            "initialization_seed": seed,
            "fold": fold,
            "supply_scenario": supply,
            "av_penetration": av_penetration,
        })
    frame = pd.DataFrame(rows)
    digest = fleet_hash(frame)
    frame["initial_fleet_hash"] = digest
    manifest = {
        "fold": fold,
        "supply_scenario": supply,
        "av_penetration": av_penetration,
        "initialization_seed": seed,
        "total_fleet": total_fleet,
        "av_count": av_count,
        "hv_count": total_fleet - av_count,
        "initialization_source": source,
        "initial_fleet_hash": digest,
        "rows": len(frame),
    }
    frame.to_parquet(fleet_file, index=False, compression="zstd")
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return frame, manifest
