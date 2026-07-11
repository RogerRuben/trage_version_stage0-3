"""Attach lagged state to actual-route rows using actual link entry time.

This is the oracle-time upper-bound counterpart to the estimated-entry causal
dataset.  The route is still the matched completed route proxy, but the traffic
state lookup is rejoined with actual link entry time.  The result is not a
deployable Stage3 input; it is only for estimating the cost of entry-time
uncertainty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["lcs", "iis", "rts", "pmis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-root", type=Path, default=Path("stage2/output/routes/actual_route_planned_time_oracle"))
    parser.add_argument("--state-root", type=Path, default=Path("stage2/output/lagged_state_store"))
    parser.add_argument("--strict-target-root", type=Path, default=Path("stage2/output/strict_targets"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2/output/actual_entry_oracle_causal_dataset"))
    parser.add_argument("--dates", default="20161009,20161010,20161011,20161012,20161013,20161014,20161015,20161016,20161017,20161018,20161019")
    parser.add_argument("--history-days", type=int, default=7)
    parser.add_argument("--max-state-age-minutes", type=int, default=60)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def parse_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def asof_by_group(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_group: str,
    right_group: str,
    left_time: str,
    right_time: str,
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    parts = []
    right = right[right[right_group].isin(left[left_group].dropna().unique())]
    right_groups = {
        key: group.sort_values(right_time)
        for key, group in right.groupby(right_group, observed=True, sort=False)
    }
    for key, group in left.groupby(left_group, observed=True, sort=False):
        state = right_groups.get(key)
        if state is None:
            parts.append(group)
            continue
        merged = pd.merge_asof(
            group.sort_values(left_time),
            state.drop(columns=[right_group]).sort_values(right_time),
            left_on=left_time,
            right_on=right_time,
            direction="backward",
            tolerance=tolerance,
        )
        parts.append(merged)
    return pd.concat(parts, ignore_index=True) if parts else left


def combine_history(model_root: Path, dates: list[str], target: str) -> pd.DataFrame:
    parts = []
    for date in dates:
        path = model_root / f"day={date}" / f"{target}.parquet"
        if path.exists():
            parts.append(pd.read_parquet(path))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).groupby(["link_id", "time_bin"], as_index=False)[["sum", "sum_sq", "count"]].sum()


def add_profile(frame: pd.DataFrame, history: pd.DataFrame, target: str, link_column: str, time_bin_column: str) -> pd.DataFrame:
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
    index = pd.MultiIndex.from_frame(frame[[link_column, time_bin_column]].rename(columns={link_column: "link_id", time_bin_column: "time_bin"}))
    frame[mean_column] = lookup[mean_column].reindex(index).to_numpy()
    frame[std_column] = lookup[std_column].reindex(index).to_numpy()
    frame[count_column] = lookup["count"].reindex(index).fillna(0).to_numpy(dtype=int)
    link = history.groupby("link_id")[["sum", "sum_sq", "count"]].sum()
    link_mean = link["sum"] / link["count"].clip(lower=1)
    link_std = np.sqrt(((link["sum_sq"] - link["sum"] ** 2 / link["count"].clip(lower=1)) / (link["count"] - 1).clip(lower=1)).clip(lower=0))
    missing = frame[mean_column].isna()
    frame.loc[missing, mean_column] = frame.loc[missing, link_column].map(link_mean)
    frame.loc[missing, std_column] = frame.loc[missing, link_column].map(link_std)
    frame.loc[missing, count_column] = frame.loc[missing, link_column].map(link["count"]).fillna(0)
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


def normalize_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = frame[column].dt.round("us")
    return frame


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    dates = [part.strip() for part in args.dates.split(",") if part.strip()]
    link = pd.read_parquet(args.state_root / "link_state.parquet").rename(columns={
        "bin_time": "link_state_bin_time",
        "feature_timestamp": "link_state_feature_timestamp",
        "availability_timestamp": "link_state_availability_timestamp",
    })
    area = pd.read_parquet(args.state_root / "area_state.parquet").rename(columns={
        "bin_time": "area_state_bin_time",
        "feature_timestamp": "area_state_feature_timestamp",
        "availability_timestamp": "area_state_availability_timestamp",
    })
    network = pd.read_parquet(args.state_root / "network_state.parquet").rename(columns={
        "bin_time": "network_state_bin_time",
        "feature_timestamp": "network_state_feature_timestamp",
        "availability_timestamp": "network_state_availability_timestamp",
    })
    tolerance = pd.Timedelta(minutes=args.max_state_age_minutes)
    model_root = args.strict_target_root / "models" / "daily_stats"
    manifest = {
        "route_source": "actual_matched_route_proxy",
        "time_mode": "actual_entry_oracle_upper_bound",
        "deployable": False,
        "state_join": "strict backward asof on actual_link_entry_time",
        "max_state_age_minutes": args.max_state_age_minutes,
        "dates": dates,
        "days": {},
    }
    for date in dates:
        output_path = args.output_root / f"day={date}.parquet"
        if args.skip_existing and output_path.exists():
            frame = pd.read_parquet(output_path, columns=["order_id", "actual_entry_strict_availability_check", "link_recent_traversal_count_15m"])
            manifest["days"][date] = {
                "rows": len(frame),
                "orders": int(frame["order_id"].nunique()),
                "strict_availability_pass_ratio": float(frame["actual_entry_strict_availability_check"].mean()),
                "link_state_coverage": float(frame["link_recent_traversal_count_15m"].notna().mean()),
                "skipped": True,
            }
            print(f"actual-entry oracle day={date} {manifest['days'][date]}", flush=True)
            continue
        frame = pd.read_parquet(args.route_root / f"day={date}.parquet")
        frame["planned_link_id"] = frame["planned_link_id"].astype(str)
        frame["actual_link_entry_time"] = parse_timestamp(frame["actual_link_entry_time"])
        frame["actual_link_exit_time"] = parse_timestamp(frame["actual_link_exit_time"])
        local = frame["actual_link_entry_time"].dt.tz_convert("Asia/Shanghai")
        frame["actual_time_bin"] = (local.dt.hour * 2 + (local.dt.minute >= 30).astype(int)).astype("int16")
        frame["prediction_link_entry_time"] = frame["actual_link_entry_time"]
        frame["prediction_time_bin"] = frame["actual_time_bin"]
        frame["prediction_hour"] = local.dt.hour.astype("int16")
        frame["prediction_weekday"] = local.dt.dayofweek.astype("int16")
        frame["prediction_is_weekend"] = local.dt.dayofweek.ge(5).astype("int8")
        frame = asof_by_group(frame, link, "planned_link_id", "link_id", "actual_link_entry_time", "link_state_bin_time", tolerance)
        if "area_grid" in frame:
            frame = asof_by_group(frame, area, "area_grid", "area_grid", "actual_link_entry_time", "area_state_bin_time", tolerance)
        frame = pd.merge_asof(
            frame.sort_values("actual_link_entry_time"),
            network.sort_values("network_state_bin_time"),
            left_on="actual_link_entry_time",
            right_on="network_state_bin_time",
            direction="backward",
            tolerance=tolerance,
        )
        previous = [value for value in dates if value < date][-args.history_days:]
        for target in TARGETS:
            frame = add_profile(frame, combine_history(model_root, previous, target), target, "planned_link_id", "actual_time_bin")
        availability_columns = [column for column in frame.columns if "availability_timestamp" in column]
        checks = []
        for column in availability_columns:
            values = pd.to_datetime(frame[column], utc=True, errors="coerce")
            checks.append(values.isna() | values.lt(frame["actual_link_entry_time"]))
        frame["actual_entry_strict_availability_check"] = np.logical_and.reduce(checks) if checks else True
        frame["strict_availability_check"] = frame["actual_entry_strict_availability_check"]
        frame["route_source"] = "actual_matched_route_proxy"
        frame["time_mode"] = "oracle_time_actual_entry"
        frame["route_conditioned_deployable"] = False
        frame = normalize_timestamps(frame)
        frame.to_parquet(output_path, index=False, compression="zstd")
        manifest["days"][date] = {
            "rows": len(frame),
            "orders": int(frame["order_id"].nunique()),
            "history_dates": previous,
            "strict_availability_pass_ratio": float(frame["actual_entry_strict_availability_check"].mean()),
            "link_state_coverage": float(frame["link_recent_traversal_count_15m"].notna().mean()) if "link_recent_traversal_count_15m" in frame else 0.0,
            "area_state_coverage": float(frame["area_recent_traversal_count_15m"].notna().mean()) if "area_recent_traversal_count_15m" in frame else 0.0,
            "network_state_coverage": float(frame["network_recent_traversal_count_15m"].notna().mean()) if "network_recent_traversal_count_15m" in frame else 0.0,
            "skipped": False,
        }
        print(f"actual-entry oracle day={date} {manifest['days'][date]}", flush=True)
    (args.output_root / "actual_entry_oracle_causal_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
