"""Attach strictly lagged state and rolling raw profiles to planned routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["lcs", "iis", "rts", "pmis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planned-route-root", type=Path, default=Path("stage2/output/od_planned_routes"))
    parser.add_argument("--state-root", type=Path, default=Path("stage2/output/lagged_state_store"))
    parser.add_argument("--strict-target-root", type=Path, default=Path("stage2/output/strict_targets"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/planned_route_causal_dataset"))
    parser.add_argument("--dates", default="20161009,20161010,20161011,20161012,20161013,20161014,20161015,20161016,20161017,20161018,20161019")
    parser.add_argument("--history-days", type=int, default=7)
    parser.add_argument("--max-state-age-minutes", type=int, default=60)
    return parser.parse_args()


def asof_by_group(left: pd.DataFrame, right: pd.DataFrame, left_group: str, right_group: str, left_time: str, right_time: str, tolerance: pd.Timedelta) -> pd.DataFrame:
    parts = []
    right = right[right[right_group].isin(left[left_group].dropna().unique())]
    right_groups = {key: group.sort_values(right_time) for key, group in right.groupby(right_group, observed=True, sort=False)}
    for key, group in left.groupby(left_group, observed=True, sort=False):
        state = right_groups.get(key)
        if state is None:
            parts.append(group)
            continue
        merged = pd.merge_asof(
            group.sort_values(left_time), state.drop(columns=[right_group]).sort_values(right_time),
            left_on=left_time, right_on=right_time, direction="backward", tolerance=tolerance,
        )
        parts.append(merged)
    return pd.concat(parts, ignore_index=True) if parts else left


def combine_history(model_root: Path, dates: list[str], target: str) -> pd.DataFrame:
    parts = [pd.read_parquet(model_root / f"day={date}" / f"{target}.parquet") for date in dates]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).groupby(["link_id", "time_bin"], as_index=False)[["sum", "sum_sq", "count"]].sum()


def add_profile(frame: pd.DataFrame, history: pd.DataFrame, target: str) -> pd.DataFrame:
    mean_column = f"rolling_{target}_raw_mean"
    std_column = f"rolling_{target}_raw_std"
    count_column = f"rolling_{target}_history_count"
    if history.empty:
        frame[mean_column] = frame[std_column] = np.nan
        frame[count_column] = 0
        return frame
    history = history.copy()
    history[mean_column] = history["sum"] / history["count"].clip(lower=1)
    history[std_column] = np.sqrt(((history["sum_sq"] - history["sum"] ** 2 / history["count"].clip(lower=1)) / (history["count"] - 1).clip(lower=1)).clip(lower=0))
    lookup = history.set_index(["link_id", "time_bin"])
    index = pd.MultiIndex.from_frame(frame[["planned_link_id", "estimated_time_bin"]].rename(columns={"planned_link_id": "link_id", "estimated_time_bin": "time_bin"}))
    frame[mean_column] = lookup[mean_column].reindex(index).to_numpy()
    frame[std_column] = lookup[std_column].reindex(index).to_numpy()
    frame[count_column] = lookup["count"].reindex(index).fillna(0).to_numpy(dtype=int)
    link = history.groupby("link_id")[["sum", "sum_sq", "count"]].sum()
    link_mean = link["sum"] / link["count"].clip(lower=1)
    link_std = np.sqrt(((link["sum_sq"] - link["sum"] ** 2 / link["count"].clip(lower=1)) / (link["count"] - 1).clip(lower=1)).clip(lower=0))
    missing = frame[mean_column].isna()
    frame.loc[missing, mean_column] = frame.loc[missing, "planned_link_id"].map(link_mean)
    frame.loc[missing, std_column] = frame.loc[missing, "planned_link_id"].map(link_std)
    frame.loc[missing, count_column] = frame.loc[missing, "planned_link_id"].map(link["count"]).fillna(0)
    if frame[mean_column].isna().any():
        total_count = history["count"].sum()
        total_sum = history["sum"].sum()
        total_sq = history["sum_sq"].sum()
        global_mean = total_sum / max(total_count, 1)
        global_std = np.sqrt(max(0, (total_sq - total_sum**2 / max(total_count, 1)) / max(total_count - 1, 1)))
        frame[mean_column] = frame[mean_column].fillna(global_mean)
        frame[std_column] = frame[std_column].fillna(global_std)
        frame[count_column] = frame[count_column].replace(0, total_count)
    return frame


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    dates = [part.strip() for part in args.dates.split(",") if part.strip()]
    link = pd.read_parquet(args.state_root / "link_state.parquet").rename(columns={
        "bin_time": "link_state_bin_time", "feature_timestamp": "link_state_feature_timestamp",
        "availability_timestamp": "link_state_availability_timestamp",
    })
    area = pd.read_parquet(args.state_root / "area_state.parquet").rename(columns={
        "bin_time": "area_state_bin_time", "feature_timestamp": "area_state_feature_timestamp",
        "availability_timestamp": "area_state_availability_timestamp",
    })
    network = pd.read_parquet(args.state_root / "network_state.parquet").rename(columns={
        "bin_time": "network_state_bin_time", "feature_timestamp": "network_state_feature_timestamp",
        "availability_timestamp": "network_state_availability_timestamp",
    })
    tolerance = pd.Timedelta(minutes=args.max_state_age_minutes)
    manifest_path = args.output_root / "planned_causal_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["dates"] = sorted(set(manifest.get("dates", [])) | set(dates))
    else:
        manifest = {"dates": dates, "days": {}, "strict_availability_rule": True}
    model_root = args.strict_target_root / "models" / "daily_stats"
    for date in dates:
        frame = pd.read_parquet(args.planned_route_root / f"day={date}.parquet")
        frame["estimated_link_entry_time"] = pd.to_datetime(frame["estimated_link_entry_time"], utc=True)
        local = frame["estimated_link_entry_time"].dt.tz_convert("Asia/Shanghai")
        frame["estimated_time_bin"] = (local.dt.hour * 2 + (local.dt.minute >= 30).astype(int)).astype("int16")
        frame = asof_by_group(frame, link, "planned_link_id", "link_id", "estimated_link_entry_time", "link_state_bin_time", tolerance)
        if "area_grid" in frame:
            frame = asof_by_group(frame, area, "area_grid", "area_grid", "estimated_link_entry_time", "area_state_bin_time", tolerance)
        network_copy = network
        frame = pd.merge_asof(
            frame.sort_values("estimated_link_entry_time"), network_copy.sort_values("network_state_bin_time"),
            left_on="estimated_link_entry_time", right_on="network_state_bin_time", direction="backward", tolerance=tolerance,
        )
        previous = [value for value in dates if value < date][-args.history_days:]
        for target in TARGETS:
            frame = add_profile(frame, combine_history(model_root, previous, target), target)
        availability_columns = [column for column in frame.columns if "availability_timestamp" in column]
        checks = []
        for column in availability_columns:
            values = pd.to_datetime(frame[column], utc=True, errors="coerce")
            checks.append(values.isna() | values.lt(frame["estimated_link_entry_time"]))
        frame["strict_availability_check"] = np.logical_and.reduce(checks) if checks else True
        target = args.output_root / f"day={date}.parquet"
        frame.to_parquet(target, index=False, compression="zstd")
        manifest["days"][date] = {
            "rows": len(frame), "history_dates": previous,
            "strict_availability_pass_ratio": float(frame["strict_availability_check"].mean()),
            "link_state_coverage": float(frame["link_recent_traversal_count_15m"].notna().mean()) if "link_recent_traversal_count_15m" in frame else 0.0,
        }
        print(f"causal planned day={date} {manifest['days'][date]}", flush=True)
    manifest.update({
        "route_source": "read from planned route product", "prediction_time": "estimated_link_entry_time",
        "state_join": "strict backward asof", "max_state_age_minutes": args.max_state_age_minutes,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
