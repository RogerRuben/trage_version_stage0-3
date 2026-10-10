"""Audit Test31 demand and replay-fleet spatial representativeness on a fixed grid."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

from stage4.replay_foundation import load_full_test31_orders


OUTPUT_REL = Path("stage4/output/paper_enhancement/efficient_repositioning")
DOC_REL = Path("stage4/docs/paper_redesign/fleet_spatial_representativeness.md")
REPLAY_REL = Path("stage4/input/replay_foundation/stage4_order_replay_base.parquet")
FLEET_REL = Path("stage4/input/replay_foundation/replay_fleet_template.parquet")
REFERENCE_LAT_DEG = 34.25
GRID_M = 1000.0
WINDOWS = {
    "ALL_DAY": (0, 24 * 60),
    "MORNING_PEAK": (7 * 60, 9 * 60),
    "EVENING_17_1859": (17 * 60, 19 * 60),
}
DATASETS = ("FULL_TEST31", "REPLAY_30K", "SELECTED_FLEET_STARTS")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def _minute_of_day(values: pd.Series) -> pd.Series:
    local = pd.to_datetime(values, utc=True).dt.tz_convert("Asia/Shanghai")
    return local.dt.hour * 60 + local.dt.minute


def _grid(frame: pd.DataFrame, lon: str, lat: str, timestamp: str, name: str) -> pd.DataFrame:
    work = frame[[lon, lat, timestamp]].copy()
    work[lon] = pd.to_numeric(work[lon], errors="coerce")
    work[lat] = pd.to_numeric(work[lat], errors="coerce")
    work = work.loc[np.isfinite(work[lon]) & np.isfinite(work[lat])].copy()
    x_scale = 111_320.0 * math.cos(math.radians(REFERENCE_LAT_DEG))
    work["grid_x"] = np.floor(work[lon].to_numpy(float) * x_scale / GRID_M).astype(int)
    work["grid_y"] = np.floor(work[lat].to_numpy(float) * 110_540.0 / GRID_M).astype(int)
    work["minute"] = _minute_of_day(work[timestamp]).to_numpy()
    rows: list[pd.DataFrame] = []
    for window, (start, end) in WINDOWS.items():
        selected = work.loc[work["minute"].between(start, end - 1)]
        counts = selected.groupby(["grid_x", "grid_y"], sort=True).size().rename("count").reset_index()
        counts["share"] = counts["count"] / counts["count"].sum()
        counts.insert(0, "dataset", name)
        counts.insert(1, "window", window)
        rows.append(counts)
    return pd.concat(rows, ignore_index=True)


def _top_cells(series: pd.Series) -> set[tuple[int, int]]:
    occupied = series[series.gt(0)].sort_values(ascending=False, kind="mergesort")
    count = max(1, int(math.ceil(len(occupied) * 0.10)))
    return set(occupied.head(count).index)


def spatial_metrics(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, float | int]:
    left_s = left.set_index(["grid_x", "grid_y"])["share"]
    right_s = right.set_index(["grid_x", "grid_y"])["share"]
    cells = left_s.index.union(right_s.index)
    p = left_s.reindex(cells, fill_value=0.0)
    q = right_s.reindex(cells, fill_value=0.0)
    rho = spearmanr(p.to_numpy(), q.to_numpy()).statistic
    left_hot = _top_cells(p)
    right_hot = _top_cells(q)
    return {
        "union_cell_count": int(len(cells)),
        "left_occupied_cell_count": int((p > 0).sum()),
        "right_occupied_cell_count": int((q > 0).sum()),
        "total_variation_distance": float(0.5 * np.abs(p - q).sum()),
        "jensen_shannon_divergence": float(jensenshannon(p, q, base=2.0) ** 2),
        "spearman_occupied_share": float(rho),
        "top_10pct_hotspot_jaccard": float(
            len(left_hot & right_hot) / len(left_hot | right_hot)
        ),
    }


def _load_shares(root: Path) -> pd.DataFrame:
    full, diagnostics = load_full_test31_orders(root, "20161031")
    if diagnostics["source_order_count"] != len(full):
        raise ValueError("full Test31 source accounting mismatch")
    replay = pd.read_parquet(root / REPLAY_REL)
    replay = replay.loc[replay["profile_id"].astype(str).eq("M")].copy()
    if replay["order_id"].duplicated().any() or len(replay) != 30_000:
        raise ValueError("replay demand must contain 30,000 unique Profile M orders")
    fleet = pd.read_parquet(root / FLEET_REL)
    frames = [
        _grid(full, "start_lon_wgs84", "start_lat_wgs84", "request_time", DATASETS[0]),
        _grid(replay, "pickup_lon_wgs84", "pickup_lat_wgs84", "request_time", DATASETS[1]),
        _grid(
            fleet,
            "initial_lon_wgs84",
            "initial_lat_wgs84",
            "availability_start_time",
            DATASETS[2],
        ),
    ]
    return pd.concat(frames, ignore_index=True)


def _plot(shares: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    for row, window in enumerate(WINDOWS):
        for column, dataset in enumerate(DATASETS):
            data = shares.loc[
                shares["window"].eq(window) & shares["dataset"].eq(dataset)
            ]
            axes[row, column].scatter(
                data["grid_x"], data["grid_y"], s=8 + 500 * data["share"],
                c=data["share"], cmap="viridis", linewidths=0,
            )
            axes[row, column].set_title(f"{window}\n{dataset}", fontsize=9)
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(root: Path) -> dict[str, Any]:
    shares = _load_shares(root)
    pairs = (
        ("FULL_TEST31", "REPLAY_30K"),
        ("REPLAY_30K", "SELECTED_FLEET_STARTS"),
        ("FULL_TEST31", "SELECTED_FLEET_STARTS"),
    )
    rows = []
    for window in WINDOWS:
        for left_name, right_name in pairs:
            subset = shares.loc[shares["window"].eq(window)]
            metrics = spatial_metrics(
                subset.loc[subset["dataset"].eq(left_name)],
                subset.loc[subset["dataset"].eq(right_name)],
            )
            rows.append(
                {"window": window, "left_dataset": left_name, "right_dataset": right_name, **metrics}
            )
    metrics = pd.DataFrame(rows)
    output = root / OUTPUT_REL
    _atomic_csv(metrics, output / "spatial_representativeness.csv")
    _atomic_parquet(shares, output / "spatial_cell_shares.parquet")
    _plot(shares, output / "spatial_diagnostic.png")
    return {
        "grid_m": GRID_M,
        "reference_lat_deg": REFERENCE_LAT_DEG,
        "windows": list(WINDOWS),
        "metrics_rows": len(metrics),
        "metrics": metrics.to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve()), indent=2, default=str))


if __name__ == "__main__":
    main()
