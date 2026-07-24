"""Build training-only idle-movement priors for Simulator v3.

Inputs are Stage0 OD tables for 2016-10-19 through 2016-10-22.  The test day
2016-10-23 is rejected explicitly.  Outputs support HV empirical idle
repositioning and AV demand-shortage rebalancing without future-demand use.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage4.simulator_v3.idle_management import _duration_bin, _haversine_m


TRAINING_DAYS = ("20161019", "20161020", "20161021", "20161022")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order-od-root", type=Path, default=Path("stage0/output/order_od"))
    parser.add_argument("--zone-system", type=Path, default=Path("stage4/data/decoupled_abm/operational_zone_system.json"))
    parser.add_argument("--output-root", type=Path, default=Path("stage4/data/decoupled_abm"))
    parser.add_argument("--training-days", nargs="+", default=list(TRAINING_DAYS))
    parser.add_argument("--maximum-idle-gap-min", type=float, default=120.0)
    parser.add_argument("--weekend-weight", type=float, default=0.7)
    return parser.parse_args()


def assign_zone(lon: pd.Series, lat: pd.Series, spec: dict) -> pd.Series:
    grid = float(spec["grid_size"])
    x = np.floor((lon.astype(float) - float(spec["min_lon"])) / grid).astype(int)
    y = np.floor((lat.astype(float) - float(spec["min_lat"])) / grid).astype(int)
    return pd.Series([f"z{ix}_{iy}" for ix, iy in zip(x, y)], index=lon.index)


def load_training_orders(args: argparse.Namespace, zone_spec: dict) -> pd.DataFrame:
    invalid = [day for day in args.training_days if str(day) >= "20161023"]
    if invalid:
        raise ValueError(f"Test/future days are forbidden in idle priors: {invalid}")
    frames = []
    for day in args.training_days:
        path = args.order_od_root / f"day={day}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=[
            "order_id", "driver_id", "origin_timestamp", "destination_timestamp",
            "origin_lon", "origin_lat", "destination_lon", "destination_lat",
        ])
        frame["source_date"] = str(day)
        frames.append(frame)
    orders = pd.concat(frames, ignore_index=True)
    orders = orders.dropna(subset=[
        "driver_id", "origin_timestamp", "destination_timestamp", "origin_lon",
        "origin_lat", "destination_lon", "destination_lat",
    ]).copy()
    orders = orders[orders["destination_timestamp"].astype(float) > orders["origin_timestamp"].astype(float)].copy()
    orders["origin_zone"] = assign_zone(orders["origin_lon"], orders["origin_lat"], zone_spec)
    orders["destination_zone"] = assign_zone(orders["destination_lon"], orders["destination_lat"], zone_spec)
    return orders.sort_values(["source_date", "driver_id", "origin_timestamp", "order_id"], kind="mergesort")


def build_hv_transitions(orders: pd.DataFrame, maximum_gap_sec: float) -> tuple[pd.DataFrame, dict]:
    grouped = orders.groupby(["source_date", "driver_id"], sort=False)
    nxt = grouped.shift(-1)
    chains = orders[[
        "source_date", "driver_id", "destination_timestamp", "destination_lon",
        "destination_lat", "destination_zone",
    ]].copy()
    chains["next_origin_timestamp"] = nxt["origin_timestamp"]
    chains["next_origin_lon"] = nxt["origin_lon"]
    chains["next_origin_lat"] = nxt["origin_lat"]
    chains["next_origin_zone"] = nxt["origin_zone"]
    chains["idle_duration_sec"] = chains["next_origin_timestamp"] - chains["destination_timestamp"]
    chains = chains[
        chains["idle_duration_sec"].gt(0)
        & chains["idle_duration_sec"].le(float(maximum_gap_sec))
        & chains["next_origin_zone"].notna()
    ].copy()
    chains["origin_zone"] = chains["destination_zone"].astype(str)
    chains["destination_zone"] = chains["next_origin_zone"].astype(str)
    dt = pd.to_datetime(chains["destination_timestamp"], unit="s", utc=True)
    chains["time_bin"] = dt.dt.hour * 2 + dt.dt.minute // 30
    chains["idle_duration_bin"] = chains["idle_duration_sec"].map(_duration_bin)
    chains["straight_distance_m"] = [
        _haversine_m((a, b), (c, d))
        for a, b, c, d in zip(
            chains["destination_lon"], chains["destination_lat"],
            chains["next_origin_lon"], chains["next_origin_lat"],
        )
    ]
    # A physical movement proxy, separate from the observed inter-trip gap.
    chains["move_distance_m"] = chains["straight_distance_m"] * 1.35
    chains["move_time_sec"] = chains["move_distance_m"] / 6.0
    keys = ["origin_zone", "destination_zone", "time_bin", "idle_duration_bin"]
    table = chains.groupby(keys, as_index=False).agg(
        sample_count=("driver_id", "size"),
        mean_move_distance_m=("move_distance_m", "mean"),
        mean_move_time_sec=("move_time_sec", "mean"),
        source_day_count=("source_date", "nunique"),
    )
    denominators = table.groupby(["origin_zone", "time_bin", "idle_duration_bin"])["sample_count"].transform("sum")
    table["transition_probability"] = table["sample_count"] / denominators
    table["training_source_dates"] = ",".join(sorted(orders["source_date"].unique()))
    table["uses_test_day_future_demand"] = False
    table["prior_version"] = "hv_idle_transition_training_20161019_22_v1"
    table = table.sort_values(keys, kind="mergesort").reset_index(drop=True)
    probability_error = float(
        (table.groupby(["origin_zone", "time_bin", "idle_duration_bin"])["transition_probability"].sum() - 1.0).abs().max()
    )
    source_dates = sorted(orders["source_date"].unique().tolist())
    audit_pass = (
        len(orders) > 0
        and len(chains) > 0
        and len(table) > 0
        and table["sample_count"].gt(0).all()
        and table["transition_probability"].notna().all()
        and table["transition_probability"].between(0.0, 1.0).all()
        and probability_error <= 1e-9
        and "20161023" not in source_dates
    )
    audit = {
        "status": "PASS" if audit_pass else "FAIL",
        "training_order_count": int(len(orders)),
        "eligible_consecutive_chain_count": int(len(chains)),
        "transition_row_count": int(len(table)),
        "origin_zone_count": int(table["origin_zone"].nunique()),
        "destination_zone_count": int(table["destination_zone"].nunique()),
        "probability_sum_max_abs_error": probability_error,
        "training_source_dates": source_dates,
        "uses_test_day_future_demand": False,
    }
    return table, audit


def build_av_demand_prior(orders: pd.DataFrame, weekend_weight: float) -> tuple[pd.DataFrame, dict]:
    if not 0 <= weekend_weight <= 1:
        raise ValueError("weekend_weight must be in [0, 1]")
    dt = pd.to_datetime(orders["origin_timestamp"], unit="s", utc=True)
    work = orders[["source_date", "origin_zone"]].copy()
    work["time_bin"] = dt.dt.hour * 2 + dt.dt.minute // 30
    counts = work.groupby(["source_date", "time_bin", "origin_zone"], as_index=False).size().rename(columns={"size": "demand_count"})
    zones = sorted(work["origin_zone"].unique())
    grid = pd.MultiIndex.from_product(
        [sorted(work["source_date"].unique()), range(48), zones],
        names=["source_date", "time_bin", "origin_zone"],
    ).to_frame(index=False)
    counts = grid.merge(counts, how="left", on=["source_date", "time_bin", "origin_zone"])
    counts["demand_count"] = counts["demand_count"].fillna(0.0)
    saturday = counts[counts["source_date"].eq("20161022")].rename(columns={"demand_count": "weekend_anchor_count"})
    weekdays = counts[counts["source_date"].isin(["20161019", "20161020", "20161021"])]
    weekday = weekdays.groupby(["time_bin", "origin_zone"], as_index=False)["demand_count"].median().rename(columns={"demand_count": "weekday_median_count"})
    prior = saturday[["time_bin", "origin_zone", "weekend_anchor_count"]].merge(weekday, how="outer", on=["time_bin", "origin_zone"])
    prior[["weekend_anchor_count", "weekday_median_count"]] = prior[["weekend_anchor_count", "weekday_median_count"]].fillna(0.0)
    prior["forecast_demand"] = weekend_weight * prior["weekend_anchor_count"] + (1.0 - weekend_weight) * prior["weekday_median_count"]
    prior = prior.rename(columns={"origin_zone": "zone"})
    prior["weekend_weight"] = float(weekend_weight)
    prior["training_source_dates"] = ",".join(sorted(orders["source_date"].unique()))
    prior["uses_test_day_future_demand"] = False
    prior["prior_version"] = "av_demand_prior_weekend_weighted_20161019_22_v1"
    prior = prior.sort_values(["time_bin", "zone"], kind="mergesort").reset_index(drop=True)
    source_dates = sorted(orders["source_date"].unique().tolist())
    audit_pass = (
        len(prior) > 0
        and prior["zone"].nunique() > 0
        and prior["time_bin"].nunique() == 48
        and prior["forecast_demand"].notna().all()
        and prior["forecast_demand"].ge(0).all()
        and float(prior["forecast_demand"].sum()) > 0
        and "20161023" not in source_dates
        and "20161022" in source_dates
    )
    audit = {
        "status": "PASS" if audit_pass else "FAIL",
        "prior_row_count": int(len(prior)),
        "zone_count": int(prior["zone"].nunique()),
        "time_bin_count": int(prior["time_bin"].nunique()),
        "weekend_weight": float(weekend_weight),
        "training_source_dates": source_dates,
        "uses_test_day_future_demand": False,
        "total_forecast_demand": float(prior["forecast_demand"].sum()),
    }
    return prior, audit


def main() -> None:
    args = parse_args()
    zone_spec = json.loads(args.zone_system.read_text(encoding="utf-8"))
    orders = load_training_orders(args, zone_spec)
    transitions, hv_audit = build_hv_transitions(orders, args.maximum_idle_gap_min * 60.0)
    demand_prior, av_audit = build_av_demand_prior(orders, args.weekend_weight)
    args.output_root.mkdir(parents=True, exist_ok=True)
    transitions.to_parquet(args.output_root / "hv_idle_zone_transition.parquet", index=False)
    demand_prior.to_parquet(args.output_root / "av_rebalancing_demand_prior.parquet", index=False)
    audit = {
        "status": "PASS" if hv_audit["status"] == "PASS" and av_audit["status"] == "PASS" else "FAIL",
        "hv_transition": hv_audit,
        "av_demand_prior": av_audit,
    }
    (args.output_root / "idle_management_prior_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
