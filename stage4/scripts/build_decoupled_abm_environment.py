"""Build demand-supply decoupled single-day ABM inputs for 2016-10-23.

This script deliberately separates:

* observed served-order demand on the test day;
* Stage3 condition-vector availability;
* training-day empirical HV supply generation;
* depot-based AV supply with a vehicle-hour share target;
* request-time scenario reconstruction;
* pickup ETA and pickup ODD conservative proxy inputs.

It does not infer true request times or true AV capability.  The generated
request times and pickup ODD matrix are scenario inputs for counterfactual
simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


EARTH_M = 6_371_000.0
SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class ZoneSpec:
    min_lon: float
    min_lat: float
    grid_size: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20161023")
    parser.add_argument("--train-dates", default="20161019,20161020,20161021,20161022")
    parser.add_argument("--weekend-anchor-date", default="20161022")
    parser.add_argument("--weekend-weight", type=float, default=0.7)
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--od-root", type=Path, default=Path("stage0/output/order_od_audited"))
    parser.add_argument("--order-base-root", type=Path, default=Path("stage0/output/order_base"))
    parser.add_argument("--stage3-inputs", type=Path, default=Path("stage3/output/full_day_20161023/stage4_inputs/stage4_inputs.parquet"))
    parser.add_argument("--stage2-route-conditioned", type=Path, default=Path("stage2/output/route_conditioned_dataset_full_20161023/estimated_time_daily/day=20161023.parquet"))
    parser.add_argument("--profiles", type=Path, default=Path("stage4/config/vehicle_capability_profiles.json"))
    parser.add_argument("--capability-reference", type=Path, default=Path("stage3/output/stage4_inputs_final/fold=3/stage4_inputs.parquet"))
    parser.add_argument("--capability-calibration-mode", choices=["none", "reference_quantile"], default="none")
    parser.add_argument("--output-root", type=Path, default=Path("stage4/output/decoupled_environment"))
    parser.add_argument("--data-root", type=Path, default=Path("stage4/data/decoupled_abm"))
    parser.add_argument("--results-dir", type=Path, default=Path("stage4/docs/results"))
    parser.add_argument("--grid-size", type=float, default=0.02)
    parser.add_argument("--target-av-vehicle-hour-share", type=float, default=0.05)
    parser.add_argument("--num-depots", type=int, default=8)
    parser.add_argument("--av-profile", default="moderate_av")
    parser.add_argument("--matching-response-sec", type=float, default=30.0)
    parser.add_argument("--minimum-pickup-sec", type=float, default=90.0)
    parser.add_argument("--max-request-lead-sec", type=float, default=1_800.0)
    parser.add_argument("--warmup-minutes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def read_day(root: Path, date: str) -> pd.DataFrame:
    return pd.read_parquet(root / f"day={date}.parquet")


def haversine_m(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_M * np.arcsin(np.sqrt(a))


def md5_frame(frame: pd.DataFrame, columns: list[str]) -> str:
    work = frame[columns].astype(str).sort_values(columns).to_csv(index=False).encode("utf-8")
    return hashlib.md5(work).hexdigest()


def make_zone_spec(frames: list[pd.DataFrame], grid_size: float) -> ZoneSpec:
    lon = pd.concat([f[["origin_lon", "destination_lon"]].stack() for f in frames], ignore_index=True)
    lat = pd.concat([f[["origin_lat", "destination_lat"]].stack() for f in frames], ignore_index=True)
    return ZoneSpec(
        min_lon=float(np.floor(lon.min() / grid_size) * grid_size),
        min_lat=float(np.floor(lat.min() / grid_size) * grid_size),
        grid_size=grid_size,
    )


def assign_zone(lon: pd.Series | np.ndarray, lat: pd.Series | np.ndarray, spec: ZoneSpec) -> pd.Series:
    lon_arr = pd.to_numeric(pd.Series(lon), errors="coerce")
    lat_arr = pd.to_numeric(pd.Series(lat), errors="coerce")
    gx = np.floor((lon_arr - spec.min_lon) / spec.grid_size).astype("Int64")
    gy = np.floor((lat_arr - spec.min_lat) / spec.grid_size).astype("Int64")
    return "z" + gx.astype(str) + "_" + gy.astype(str)


def time_bin_index(ts: pd.Series, minutes: int = 30) -> pd.Series:
    dt = pd.to_datetime(ts, utc=True, errors="coerce")
    return ((dt.dt.hour * 60 + dt.dt.minute) // minutes).astype("Int64")


def build_historical_orders(args: argparse.Namespace) -> pd.DataFrame:
    od = read_day(args.od_root, args.date)
    base = read_day(args.order_base_root, args.date)
    base_cols = [
        "order_id",
        "quality_flag",
        "quality_tier",
        "matching_success",
        "matched_route_length_m",
        "matched_link_count",
        "matching_confidence",
    ]
    base = base[[c for c in base_cols if c in base.columns]]
    orders = od.merge(base, on="order_id", how="left", validate="one_to_one")
    orders = orders.rename(columns={"driver_id": "historical_driver_id", "matched_route_length_m": "route_length_m"})
    orders["observed_boarding_time"] = pd.to_datetime(orders["origin_timestamp"], unit="s", utc=True, errors="coerce")
    orders["observed_dropoff_time"] = pd.to_datetime(orders["destination_timestamp"], unit="s", utc=True, errors="coerce")
    orders["stage0_valid_order"] = (
        orders["coordinate_valid"].fillna(False).astype(bool)
        & orders["duration_valid"].fillna(False).astype(bool)
    )
    orders["route_length_m"] = pd.to_numeric(orders.get("route_length_m"), errors="coerce")
    orders.loc[orders["route_length_m"].isna(), "route_length_m"] = haversine_m(
        orders.loc[orders["route_length_m"].isna(), "origin_lon"],
        orders.loc[orders["route_length_m"].isna(), "origin_lat"],
        orders.loc[orders["route_length_m"].isna(), "destination_lon"],
        orders.loc[orders["route_length_m"].isna(), "destination_lat"],
    )
    return orders


def build_train_orders(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for date in [d.strip() for d in args.train_dates.split(",") if d.strip()]:
        frame = read_day(args.od_root, date).copy()
        frame["observed_boarding_time"] = pd.to_datetime(frame["origin_timestamp"], unit="s", utc=True, errors="coerce")
        frame["observed_dropoff_time"] = pd.to_datetime(frame["destination_timestamp"], unit="s", utc=True, errors="coerce")
        out[date] = frame
    return out


def build_training_chain_stats(train: dict[str, pd.DataFrame], spec: ZoneSpec) -> tuple[pd.DataFrame, dict]:
    rows = []
    for date, frame in train.items():
        work = frame.dropna(
            subset=["driver_id", "observed_boarding_time", "observed_dropoff_time", "origin_lon", "origin_lat", "destination_lon", "destination_lat"]
        ).copy()
        work["origin_zone"] = assign_zone(work["origin_lon"], work["origin_lat"], spec).to_numpy()
        work["destination_zone"] = assign_zone(work["destination_lon"], work["destination_lat"], spec).to_numpy()
        work = work.sort_values(["driver_id", "observed_boarding_time"], kind="mergesort")
        work["prev_dropoff"] = work.groupby("driver_id")["observed_dropoff_time"].shift()
        work["prev_dest_lon"] = work.groupby("driver_id")["destination_lon"].shift()
        work["prev_dest_lat"] = work.groupby("driver_id")["destination_lat"].shift()
        work["prev_dest_zone"] = work.groupby("driver_id")["destination_zone"].shift()
        work["gap_sec"] = (work["observed_boarding_time"] - work["prev_dropoff"]).dt.total_seconds()
        work["empty_dist_m"] = haversine_m(work["prev_dest_lon"], work["prev_dest_lat"], work["origin_lon"], work["origin_lat"])
        work["date"] = date
        rows.append(work[["date", "driver_id", "observed_boarding_time", "observed_dropoff_time", "origin_zone", "destination_zone", "prev_dest_zone", "gap_sec", "empty_dist_m"]])
    chain = pd.concat(rows, ignore_index=True)
    feasible = chain[chain["gap_sec"].between(60, 7_200) & chain["empty_dist_m"].ge(0)].copy()
    speed = feasible["empty_dist_m"] / feasible["gap_sec"].replace(0, np.nan)
    stats = {
        "gap_p25_sec": float(feasible["gap_sec"].quantile(0.25)),
        "gap_p50_sec": float(feasible["gap_sec"].quantile(0.50)),
        "gap_p75_sec": float(feasible["gap_sec"].quantile(0.75)),
        "gap_p90_sec": float(feasible["gap_sec"].quantile(0.90)),
        "empty_speed_p50_mps": float(speed.replace([np.inf, -np.inf], np.nan).dropna().quantile(0.50)),
        "chain_rows": int(len(chain)),
        "feasible_chain_rows": int(len(feasible)),
    }
    return chain, stats


def build_eta_baseline(train: dict[str, pd.DataFrame], spec: ZoneSpec) -> tuple[pd.DataFrame, dict]:
    rows = []
    for date, frame in train.items():
        work = frame.copy()
        work["origin_zone"] = assign_zone(work["origin_lon"], work["origin_lat"], spec).to_numpy()
        work["destination_zone"] = assign_zone(work["destination_lon"], work["destination_lat"], spec).to_numpy()
        work["time_bin"] = time_bin_index(work["observed_boarding_time"]).to_numpy()
        work["duration_sec"] = pd.to_numeric(work["duration_sec"], errors="coerce")
        rows.append(work[["origin_zone", "destination_zone", "time_bin", "duration_sec"]])
    data = pd.concat(rows, ignore_index=True).dropna()
    zt = data.groupby(["origin_zone", "destination_zone", "time_bin"], as_index=False).agg(
        eta_sec=("duration_sec", "median"),
        sample_count=("duration_sec", "size"),
    )
    fallback_time = data.groupby("time_bin", as_index=False).agg(global_time_eta_sec=("duration_sec", "median"))
    global_eta = float(data["duration_sec"].median())
    return zt, {"global_eta_sec": global_eta, "fallback_by_time": fallback_time.to_dict("records")}


def load_stage2_eta(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["order_id", "stage2_predicted_service_time_sec"])
    frame = pd.read_parquet(path, columns=["order_id", "estimated_link_travel_time_sec"])
    frame["estimated_link_travel_time_sec"] = pd.to_numeric(frame["estimated_link_travel_time_sec"], errors="coerce").fillna(0)
    return frame.groupby("order_id", as_index=False).agg(stage2_predicted_service_time_sec=("estimated_link_travel_time_sec", "sum"))


def attach_eta(demand: pd.DataFrame, eta_table: pd.DataFrame, eta_meta: dict, spec: ZoneSpec, stage2_eta: pd.DataFrame) -> pd.DataFrame:
    demand = demand.copy()
    demand["origin_zone"] = assign_zone(demand["origin_lon"], demand["origin_lat"], spec).to_numpy()
    demand["destination_zone"] = assign_zone(demand["destination_lon"], demand["destination_lat"], spec).to_numpy()
    demand["time_bin"] = time_bin_index(demand["observed_boarding_time"]).to_numpy()
    demand = demand.merge(eta_table, on=["origin_zone", "destination_zone", "time_bin"], how="left")
    fallback_by_time = pd.DataFrame(eta_meta["fallback_by_time"])
    demand = demand.merge(fallback_by_time, on="time_bin", how="left")
    demand["predicted_service_time_sec"] = demand["eta_sec"]
    demand["eta_source"] = "train_od_time_eta_baseline"
    time_mask = demand["predicted_service_time_sec"].isna() & demand["global_time_eta_sec"].notna()
    demand.loc[time_mask, "predicted_service_time_sec"] = demand.loc[time_mask, "global_time_eta_sec"]
    demand.loc[time_mask, "eta_source"] = "train_timebin_eta_baseline"
    global_mask = demand["predicted_service_time_sec"].isna()
    demand.loc[global_mask, "predicted_service_time_sec"] = float(eta_meta["global_eta_sec"])
    demand.loc[global_mask, "eta_source"] = "train_global_eta_baseline"
    if not stage2_eta.empty:
        demand = demand.merge(stage2_eta, on="order_id", how="left")
        stage2_mask = demand["stage2_predicted_service_time_sec"].notna() & demand["condition_available"].fillna(False)
        demand.loc[stage2_mask, "predicted_service_time_sec"] = demand.loc[stage2_mask, "stage2_predicted_service_time_sec"]
        demand.loc[stage2_mask, "eta_source"] = "stage2_estimated_route_eta"
    else:
        demand["stage2_predicted_service_time_sec"] = np.nan
    demand["predicted_service_time_sec"] = demand["predicted_service_time_sec"].clip(lower=60, upper=7_200)
    demand["realized_service_time_sec"] = pd.to_numeric(demand["duration_sec"], errors="coerce").fillna(demand["predicted_service_time_sec"])
    demand["eta_available"] = demand["predicted_service_time_sec"].notna()
    demand["historical_duration_replay_mode"] = True
    return demand


def attach_stage3_conditions(demand: pd.DataFrame, stage3_path: Path) -> pd.DataFrame:
    stage3 = pd.read_parquet(stage3_path)
    keep = [
        "order_id",
        "route_length_m",
        "link_count",
        "route_prediction_confidence",
        "lcs_expected",
        "lcs_tail_probability",
        "pmis_expected",
        "pmis_tail_probability",
        "rts_expected",
        "rts_tail_probability",
        "core_overall_high_stress_probability",
        "extended_overall_high_stress_probability",
        "intersection_applicability",
        "intersection_severity",
        "intersection_tail_probability",
        "iis_availability",
        "overall_uncertainty",
        "model_version",
        "prediction_cutoff_time",
    ]
    keep = [c for c in keep if c in stage3.columns]
    merged = demand.merge(stage3[keep], on="order_id", how="left", suffixes=("", "_stage3"))
    merged["condition_available"] = merged["lcs_tail_probability"].notna() & merged["pmis_tail_probability"].notna() & merged["rts_tail_probability"].notna()
    merged["stress_surcharge_allowed"] = merged["condition_available"]
    merged["hv_stress_compensation_allowed"] = merged["condition_available"]
    merged.loc[~merged["condition_available"], ["lcs_tail_probability", "pmis_tail_probability", "rts_tail_probability", "overall_uncertainty"]] = np.nan
    return merged


def stable_uniform(order_id: pd.Series, salt: str) -> pd.Series:
    vals = []
    for value in order_id.astype(str):
        h = hashlib.md5(f"{salt}:{value}".encode("utf-8")).hexdigest()
        vals.append(int(h[:12], 16) / float(16 ** 12 - 1))
    return pd.Series(vals, index=order_id.index)


def attach_request_times(demand: pd.DataFrame, chain_stats: dict, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    out = {}
    lower = max(float(args.matching_response_sec + args.minimum_pickup_sec), 60.0)
    q25 = max(lower + 60, min(float(chain_stats["gap_p25_sec"]), args.max_request_lead_sec))
    q50 = max(lower + 60, min(float(chain_stats["gap_p50_sec"]), args.max_request_lead_sec))
    q75 = max(lower + 60, min(float(chain_stats["gap_p75_sec"]), args.max_request_lead_sec))
    scenario_fracs = {"RT-Low": 0.25, "RT-Base": 0.50, "RT-High": 0.75}
    scenario_caps = {"RT-Low": q25, "RT-Base": q50, "RT-High": q75}
    business_lower = pd.Timestamp(f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]} 00:00:00", tz="UTC") - pd.Timedelta(minutes=args.warmup_minutes)
    for scenario, frac in scenario_fracs.items():
        frame = demand.copy()
        # Order-level lower bound varies with a conservative minimum pickup
        # proxy.  The training chain supplies only scenario bounds, not a true
        # passenger request-time distribution.
        od_dist = haversine_m(frame["origin_lon"], frame["origin_lat"], frame["destination_lon"], frame["destination_lat"])
        order_lower = np.maximum(lower, args.matching_response_sec + np.minimum(600.0, np.maximum(args.minimum_pickup_sec, od_dist / 12.0)))
        seconds_since_boundary = (frame["observed_boarding_time"] - business_lower).dt.total_seconds().clip(lower=lower + 1)
        frame["request_time_lower_bound_sec"] = order_lower
        cap = scenario_caps[scenario]
        frame["request_time_upper_bound_sec"] = np.minimum(seconds_since_boundary - 1, np.maximum(cap, order_lower + 60))
        frame["request_time_upper_bound_sec"] = frame["request_time_upper_bound_sec"].clip(lower=order_lower + 1)
        u = stable_uniform(frame["order_id"], scenario)
        jitter = (u - 0.5) * 0.20
        position = np.clip(frac + jitter, 0.05, 0.95)
        target_lead = frame["request_time_lower_bound_sec"] + position * (frame["request_time_upper_bound_sec"] - frame["request_time_lower_bound_sec"])
        lead = target_lead.copy()
        lead = np.minimum(lead, cap)
        lead = np.maximum(lead, frame["request_time_lower_bound_sec"])
        frame["request_lead_clipped"] = lead.ne(target_lead)
        frame["latent_request_lead_sec"] = lead.clip(lower=lower)
        frame["simulated_request_time"] = (frame["observed_boarding_time"] - pd.to_timedelta(frame["latent_request_lead_sec"], unit="s")).dt.round("us")
        frame["request_time_scenario"] = scenario
        frame["request_time_source"] = "order_level_scenario_with_training_chain_bounds"
        frame["request_lead_bound_source"] = "training_driver_chain_feasible_gap_bounds_clipped_to_business_day"
        frame["request_time_identification_status"] = "latent_scenario"
        out[scenario] = frame
    return out


def build_sessions_for_day(frame: pd.DataFrame, spec: ZoneSpec, date: str, gap_threshold_min: int = 90) -> pd.DataFrame:
    work = frame.dropna(subset=["driver_id", "observed_boarding_time", "observed_dropoff_time", "origin_lon", "origin_lat"]).copy()
    work = work.sort_values(["driver_id", "observed_boarding_time"], kind="mergesort")
    gap = (work["observed_boarding_time"] - work.groupby("driver_id")["observed_dropoff_time"].shift()).dt.total_seconds() / 60
    work["new_session"] = gap.isna() | gap.gt(gap_threshold_min)
    work["session_seq"] = work["new_session"].groupby(work["driver_id"]).cumsum().astype(int)
    rows = []
    for (driver, seq), group in work.groupby(["driver_id", "session_seq"], sort=False):
        rows.append({
            "source_date": date,
            "source_driver_id": driver,
            "source_session_seq": int(seq),
            "relative_start_sec": int((group["observed_boarding_time"].iloc[0] - group["observed_boarding_time"].iloc[0].normalize()).total_seconds()),
            "relative_end_sec": int((group["observed_dropoff_time"].iloc[-1] - group["observed_dropoff_time"].iloc[-1].normalize()).total_seconds()),
            "duration_sec": max(60.0, float((group["observed_dropoff_time"].iloc[-1] - group["observed_boarding_time"].iloc[0]).total_seconds())),
            "initial_zone": assign_zone(group["origin_lon"].iloc[:1], group["origin_lat"].iloc[:1], spec).iloc[0],
            "initial_lon": float(group["origin_lon"].iloc[0]),
            "initial_lat": float(group["origin_lat"].iloc[0]),
            "historical_order_count": int(len(group)),
        })
    return pd.DataFrame(rows)


def active_curve(sessions: pd.DataFrame, bin_minutes: int = 30) -> pd.Series:
    bins = np.arange(0, SECONDS_PER_DAY, bin_minutes * 60)
    values = []
    for start in bins:
        end = start + bin_minutes * 60
        values.append(int(((sessions["relative_start_sec"] < end) & (sessions["relative_end_sec"] > start)).sum()))
    return pd.Series(values, index=np.arange(len(bins)), name="active")


def generate_decoupled_supply(train_sessions: dict[str, pd.DataFrame], args: argparse.Namespace, rng: np.random.Generator, replication: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weekday_dates = [d for d in train_sessions if d != args.weekend_anchor_date]
    curves = {d: active_curve(s) for d, s in train_sessions.items()}
    weekday_median = pd.concat([curves[d] for d in weekday_dates], axis=1).median(axis=1)
    weekend = curves[args.weekend_anchor_date]
    target = args.weekend_weight * weekend + (1 - args.weekend_weight) * weekday_median
    target = target.round().astype(int)
    pool = pd.concat(train_sessions.values(), ignore_index=True)
    rows = []
    generated = pd.Series(0, index=target.index)
    test_day_start = pd.Timestamp(f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]} 00:00:00", tz="UTC")
    session_id = 0
    for bin_idx, target_count in target.items():
        deficit = int(max(0, target_count - generated.loc[bin_idx]))
        if deficit <= 0:
            continue
        bin_pool = pool[(pool["relative_start_sec"] // 1800).eq(bin_idx)]
        if bin_pool.empty:
            bin_pool = pool
        pick_idx = rng.integers(0, len(bin_pool), size=deficit)
        chosen = bin_pool.iloc[pick_idx].reset_index(drop=True)
        for _, s in chosen.iterrows():
            start_sec = int(bin_idx * 1800 + rng.integers(0, 1800))
            duration = int(max(1_800, min(float(s["duration_sec"]), 14 * 3600)))
            end_sec = min(start_sec + duration, SECONDS_PER_DAY + args.warmup_minutes * 60)
            rows.append({
                "vehicle_id": f"DHV_R{replication}_{session_id:06d}",
                "synthetic_driver_id": f"SD_R{replication}_{session_id:06d}",
                "session_id": f"DHV_SESSION_R{replication}_{session_id:06d}",
                "online_start": test_day_start + pd.Timedelta(seconds=start_sec),
                "online_end": test_day_start + pd.Timedelta(seconds=end_sec),
                "initial_zone": s["initial_zone"],
                "initial_lon": float(s["initial_lon"]),
                "initial_lat": float(s["initial_lat"]),
                "historical_order_count_draw": int(s["historical_order_count"]),
                "generation_source_dates": args.train_dates,
                "generation_seed": int(args.seed),
                "replication_id": replication,
                "agent_source": "prior_day_empirical_supply",
            })
            session_id += 1
        sessions_tmp = pd.DataFrame(rows)
        if not sessions_tmp.empty:
            rel = pd.DataFrame({
                "relative_start_sec": (pd.to_datetime(sessions_tmp["online_start"], utc=True) - test_day_start).dt.total_seconds(),
                "relative_end_sec": (pd.to_datetime(sessions_tmp["online_end"], utc=True) - test_day_start).dt.total_seconds(),
            })
            generated = active_curve(rel.rename(columns={"relative_start_sec": "relative_start_sec", "relative_end_sec": "relative_end_sec"}))
    sessions = pd.DataFrame(rows)
    if not sessions.empty:
        rel_sessions = pd.DataFrame({
            "relative_start_sec": (pd.to_datetime(sessions["online_start"], utc=True) - test_day_start).dt.total_seconds(),
            "relative_end_sec": (pd.to_datetime(sessions["online_end"], utc=True) - test_day_start).dt.total_seconds(),
        })
        generated = active_curve(rel_sessions)
    audit = pd.DataFrame({
        "time_bin": target.index,
        "target_online_HV": target.values,
        "generated_online_HV": generated.reindex(target.index).fillna(0).astype(int).values,
    })
    audit["absolute_error"] = (audit["generated_online_HV"] - audit["target_online_HV"]).abs()
    audit["relative_error"] = audit["absolute_error"] / audit["target_online_HV"].replace(0, np.nan)
    agents = sessions.copy()
    agents["vehicle_type"] = "HV"
    agents["driver_id"] = agents["synthetic_driver_id"]
    agents["depot_id"] = pd.NA
    agents["vehicle_profile"] = "reference_hv"
    return agents, sessions, audit


def build_depots(train: dict[str, pd.DataFrame], args: argparse.Namespace, av_count: int, spec: ZoneSpec) -> pd.DataFrame:
    origins = pd.concat([f[["origin_lon", "origin_lat"]].dropna() for f in train.values()], ignore_index=True)
    n = max(1, min(args.num_depots, len(origins)))
    km = KMeans(n_clusters=n, random_state=args.seed, n_init=10)
    labels = km.fit_predict(origins[["origin_lon", "origin_lat"]].to_numpy(float))
    origins = origins.copy()
    origins["cluster"] = labels
    rows = []
    for cluster, group in origins.groupby("cluster"):
        center = km.cluster_centers_[int(cluster)]
        dist = haversine_m(group["origin_lon"].to_numpy(), group["origin_lat"].to_numpy(), center[0], center[1])
        medoid = group.iloc[int(np.argmin(dist))]
        rows.append({
            "depot_id": f"DEPOT_{int(cluster):02d}",
            "zone_id": f"depot_cluster_{int(cluster):02d}",
            "operational_zone": assign_zone(pd.Series([float(medoid["origin_lon"])]), pd.Series([float(medoid["origin_lat"])]), spec).iloc[0],
            "lon": float(medoid["origin_lon"]),
            "lat": float(medoid["origin_lat"]),
            "road_node_id": f"training_origin_medoid_{int(cluster):02d}",
            "capacity": int(math.ceil(av_count / n)) if n else 0,
            "training_date_range": ",".join(train.keys()),
            "depot_generation_method": "training_origin_kmeans_medoid_no_test_day_demand",
            "training_origin_count": int(len(group)),
        })
    depots = pd.DataFrame(rows).sort_values("depot_id").reset_index(drop=True)
    if av_count <= 0:
        depots["assigned_av_count"] = 0
        return depots
    weights = depots["training_origin_count"].to_numpy(float)
    weights = weights / weights.sum()
    counts = np.floor(weights * av_count).astype(int)
    remaining = av_count - int(counts.sum())
    if remaining > 0:
        order = np.argsort(-(weights * av_count - counts))
        counts[order[:remaining]] += 1
    depots["assigned_av_count"] = counts
    return depots


def build_av_agents(depots: pd.DataFrame, args: argparse.Namespace, replication: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for _, depot in depots.iterrows():
        for idx in range(int(depot["assigned_av_count"])):
            rows.append({
                "vehicle_id": f"AV_R{replication}_{depot['depot_id']}_{idx:04d}",
                "vehicle_type": "AV",
                "driver_id": pd.NA,
                "session_id": pd.NA,
                "depot_id": depot["depot_id"],
                "initial_zone": depot["operational_zone"],
                "initial_lon": float(depot["lon"]),
                "initial_lat": float(depot["lat"]),
                "online_start": start,
                "online_end": end,
                "vehicle_profile": args.av_profile,
                "replication_id": replication,
                "agent_source": "training_data_depot_vehicle_hour_share",
            })
    return pd.DataFrame(rows)


def build_pickup_calibration_and_odd(train: dict[str, pd.DataFrame], chain: pd.DataFrame, spec: ZoneSpec, results_dir: Path, order_base_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    all_orders = pd.concat(train.values(), ignore_index=True)
    all_orders["origin_zone"] = assign_zone(all_orders["origin_lon"], all_orders["origin_lat"], spec).to_numpy()
    all_orders["destination_zone"] = assign_zone(all_orders["destination_lon"], all_orders["destination_lat"], spec).to_numpy()
    all_orders["time_bin"] = time_bin_index(all_orders["observed_boarding_time"]).to_numpy()
    all_orders["od_dist_m"] = haversine_m(all_orders["origin_lon"], all_orders["origin_lat"], all_orders["destination_lon"], all_orders["destination_lat"])
    all_orders["loaded_speed_mps"] = (all_orders["od_dist_m"] / pd.to_numeric(all_orders["duration_sec"], errors="coerce")).replace([np.inf, -np.inf], np.nan)
    speed = all_orders.groupby(["origin_zone", "time_bin"], as_index=False).agg(
        loaded_speed_mps=("loaded_speed_mps", "median"),
        sample_count=("loaded_speed_mps", "size"),
    )
    speed["empty_speed_mps"] = (speed["loaded_speed_mps"] * 1.15).clip(lower=4.0, upper=14.0)
    global_speed = float((all_orders["loaded_speed_mps"].median() * 1.15))
    global_speed = min(max(global_speed, 4.0), 14.0)
    circ_parts = []
    for date in train:
        base_path = order_base_root / f"day={date}.parquet"
        if base_path.exists():
            base = pd.read_parquet(base_path, columns=["order_id", "matched_route_length_m"])
            tmp = train[date][["order_id", "origin_lon", "origin_lat", "destination_lon", "destination_lat"]].merge(base, on="order_id", how="left")
            tmp["origin_zone"] = assign_zone(tmp["origin_lon"], tmp["origin_lat"], spec).to_numpy()
            tmp["od_dist_m"] = haversine_m(tmp["origin_lon"], tmp["origin_lat"], tmp["destination_lon"], tmp["destination_lat"])
            tmp["circuity_factor"] = pd.to_numeric(tmp["matched_route_length_m"], errors="coerce") / tmp["od_dist_m"].replace(0, np.nan)
            circ_parts.append(tmp[["origin_zone", "circuity_factor"]])
    if circ_parts:
        circ = pd.concat(circ_parts, ignore_index=True)
        circuity = circ.groupby("origin_zone", as_index=False).agg(circuity_factor=("circuity_factor", "median"), sample_count=("circuity_factor", "count"))
        circuity["circuity_factor"] = circuity["circuity_factor"].clip(lower=1.05, upper=2.5)
    else:
        circuity = pd.DataFrame({"origin_zone": speed["origin_zone"].unique(), "circuity_factor": 1.35, "sample_count": 0})
    all_orders["stress_proxy"] = pd.to_numeric(all_orders.get("duration_sec"), errors="coerce") / all_orders["od_dist_m"].replace(0, np.nan)
    zone_pair = all_orders.groupby(["origin_zone", "destination_zone"], as_index=False).agg(
        train_order_count=("order_id", "size"),
        median_distance_m=("od_dist_m", "median"),
        median_duration_per_m=("stress_proxy", "median"),
    )
    # Conservative scenario proxy: very sparse or extremely long/high-duration pairs
    # are not considered pickup-ODD compatible for AVs.
    count_ok = zone_pair["train_order_count"].ge(5)
    dist_ok = zone_pair["median_distance_m"].le(zone_pair["median_distance_m"].quantile(0.95))
    stress_ok = zone_pair["median_duration_per_m"].le(zone_pair["median_duration_per_m"].quantile(0.90))
    zone_pair["pickup_odd_feasible"] = count_ok & dist_ok & stress_ok
    zone_pair["pickup_odd_proxy_source"] = "training_route_pressure_zone_pair_conservative_scenario_proxy"
    zone_pair["pickup_odd_proxy_version"] = "pickup_odd_proxy_v1"
    summary = {
        "pickup_odd_proxy_source": "training_route_pressure_zone_pair_conservative_scenario_proxy",
        "pickup_odd_proxy_version": "pickup_odd_proxy_v1",
        "zone_pair_count": int(len(zone_pair)),
        "zone_pair_feasible_share": float(zone_pair["pickup_odd_feasible"].mean()) if len(zone_pair) else 0.0,
        "empty_speed_global_mps": global_speed,
        "empty_speed_sample_count": int(speed["sample_count"].sum()) if len(speed) else 0,
        "circuity_factor_median": float(circuity["circuity_factor"].median()) if len(circuity) else 1.35,
    }
    pd.DataFrame([summary]).to_csv(results_dir / "pickup_eta_calibration_summary.csv", index=False)
    return speed, circuity, zone_pair, summary


def quantile_calibrate_to_reference(values: pd.Series, reference: pd.Series) -> pd.Series:
    """Map values to a reference empirical distribution by rank.

    This is used only for capability-scenario gating when the full-day
    inference distribution is on a different probability scale from the
    fold-level Stage3 export used to define scenario thresholds.  It does not
    change Stage3 outputs; it creates calibrated capability scores.
    """
    v = pd.to_numeric(values, errors="coerce")
    ref = pd.to_numeric(reference, errors="coerce").dropna().sort_values().to_numpy()
    if len(ref) == 0:
        return v
    ranks = v.rank(method="average", pct=True)
    q = ranks.to_numpy(dtype=float)
    mapped = np.full(len(v), np.nan, dtype=float)
    valid = np.isfinite(q)
    mapped[valid] = np.quantile(ref, np.clip(q[valid], 0.0, 1.0))
    return pd.Series(mapped, index=values.index)


def build_capability(demand: pd.DataFrame, profiles_path: Path, reference_path: Path, calibration_mode: str, output_root: Path, results_dir: Path) -> pd.DataFrame:
    doc = json.loads(profiles_path.read_text(encoding="utf-8"))
    profiles = doc["profiles"]
    reference = pd.read_parquet(reference_path) if reference_path.exists() else pd.DataFrame()
    rows = []
    binding_rows = []
    decomp_rows = []
    for profile_name, profile in profiles.items():
        if profile["vehicle_type"] != "AV":
            continue
        hard = profile["dimension_hard_threshold"]
        uncertainty_tolerance = float(profile["uncertainty_tolerance"])
        app_threshold = float(profile.get("iis_applicability_threshold", 0.5))
        lcs_raw = pd.to_numeric(demand["lcs_tail_probability"], errors="coerce")
        pmis_raw = pd.to_numeric(demand["pmis_tail_probability"], errors="coerce")
        rts_raw = pd.to_numeric(demand["rts_tail_probability"], errors="coerce")
        unc_raw = pd.to_numeric(demand["overall_uncertainty"], errors="coerce")
        if calibration_mode == "reference_quantile":
            lcs = quantile_calibrate_to_reference(lcs_raw, reference.get("lcs_tail_probability", lcs_raw))
            pmis = quantile_calibrate_to_reference(pmis_raw, reference.get("pmis_tail_probability", pmis_raw))
            rts = quantile_calibrate_to_reference(rts_raw, reference.get("rts_tail_probability", rts_raw))
            unc = quantile_calibrate_to_reference(unc_raw, reference.get("overall_uncertainty", unc_raw))
            scale_label = "fold3_reference_quantile_calibrated_deprecated"
        else:
            lcs, pmis, rts, unc = lcs_raw, pmis_raw, rts_raw, unc_raw
            scale_label = "raw_full_day_stage3_outputs_no_test_day_rank_remap"
        known = demand["condition_available"].fillna(False).astype(bool)
        lcs_v = known & lcs.gt(float(hard["lcs"]))
        pmis_v = known & pmis.gt(float(hard["pmis"]))
        rts_v = known & rts.gt(float(hard["rts"]))
        core_unc_v = known & unc.gt(uncertainty_tolerance)
        iis_available = demand.get("iis_availability", pd.Series(False, index=demand.index)).fillna(False).astype(bool)
        iis_app = pd.to_numeric(demand.get("intersection_applicability", pd.Series(np.nan, index=demand.index)), errors="coerce")
        iis_tail = pd.to_numeric(demand.get("intersection_tail_probability", pd.Series(np.nan, index=demand.index)), errors="coerce")
        iis_applied = known & iis_available & iis_app.ge(app_threshold)
        iis_v = iis_applied & iis_tail.gt(float(hard["iis"]))
        core_feasible = known & ~(lcs_v | pmis_v | rts_v | core_unc_v)
        service_feasible = core_feasible & ~iis_v
        margins = pd.DataFrame({
            "lcs": float(hard["lcs"]) - lcs,
            "pmis": float(hard["pmis"]) - pmis,
            "rts": float(hard["rts"]) - rts,
        })
        min_dim_margin = margins.min(axis=1, skipna=True)
        core_binding = margins.idxmin(axis=1, skipna=True).fillna("unknown_condition")
        core_unc_margin = uncertainty_tolerance - unc
        core_binding = core_binding.where(min_dim_margin.le(core_unc_margin), "uncertainty")
        core_binding = core_binding.where(known, "unknown_condition")
        threshold_text = []
        for idx in range(len(demand)):
            vals = []
            if bool(lcs_v.iloc[idx]): vals.append("lcs")
            if bool(pmis_v.iloc[idx]): vals.append("pmis")
            if bool(rts_v.iloc[idx]): vals.append("rts")
            if bool(core_unc_v.iloc[idx]): vals.append("uncertainty")
            if bool(iis_v.iloc[idx]): vals.append("iis")
            if not bool(known.iloc[idx]): vals.append("unknown_condition")
            threshold_text.append(",".join(vals))
        cap = pd.DataFrame({
            "fold": 3,
            "order_id": demand["order_id"].astype(str),
            "date": demand["date"].astype(str),
            "vehicle_profile": profile_name,
            "vehicle_type": "AV",
            "scenario_parameter_status": doc.get("parameter_status", "scenario_priors_not_empirical_av_estimates"),
            "condition_available": known,
            "capability_score_scale": scale_label,
            "raw_lcs_tail_probability": lcs_raw.astype("float32"),
            "raw_pmis_tail_probability": pmis_raw.astype("float32"),
            "raw_rts_tail_probability": rts_raw.astype("float32"),
            "raw_overall_uncertainty": unc_raw.astype("float32"),
            "calibrated_lcs_tail_probability": lcs.astype("float32"),
            "calibrated_pmis_tail_probability": pmis.astype("float32"),
            "calibrated_rts_tail_probability": rts.astype("float32"),
            "calibrated_overall_uncertainty": unc.astype("float32"),
            "core_service_feasible": core_feasible,
            "core_uncertainty_margin": core_unc_margin.astype("float32"),
            "core_binding_dimension": core_binding,
            "iis_constraint_applied": iis_applied,
            "iis_applicability_threshold": app_threshold,
            "conditional_iis_feasible": ~iis_v,
            "service_feasible": service_feasible,
            "feasible_with_extra_cost": service_feasible & (unc.gt(uncertainty_tolerance * 0.8).fillna(False)),
            "capability_cost": (
                (lcs.fillna(0) + pmis.fillna(0) + rts.fillna(0)) * 3.0
                + (~service_feasible & known).astype(float) * float(profile.get("fallback_cost_placeholder", 0.0))
            ).astype("float32"),
            "ODD_margin": pd.concat([min_dim_margin.rename("dimension"), core_unc_margin.rename("unc")], axis=1).min(axis=1).astype("float32"),
            "uncertainty_margin": core_unc_margin.astype("float32"),
            "minimum_dimension_margin": min_dim_margin.astype("float32"),
            "binding_dimension": core_binding,
            "threshold_violation_dimensions": threshold_text,
            "soft_threshold_violation_dimensions": "",
            "missing_modality_dimensions": np.where(iis_available, "", "iis"),
            "missing_modality_uncertainty_penalty": 0.0,
            "effective_uncertainty": unc.astype("float32"),
            "availability_adjusted_stress": ((lcs.fillna(0) + pmis.fillna(0) + rts.fillna(0)) / 3).astype("float32"),
            "uncertainty_adjusted_feasibility": service_feasible,
        })
        rows.append(cap)
        decomp_rows.append({
            "sample": "full_day_112165_condition_known" if profile_name == "moderate_av" else f"full_day_{profile_name}",
            "vehicle_profile": profile_name,
            "orders": int(len(cap)),
            "condition_known_orders": int(known.sum()),
            "core_feasible_share": float(core_feasible[known].mean()) if known.any() else 0.0,
            "service_feasible_share": float(service_feasible[known].mean()) if known.any() else 0.0,
            "lcs_hard_violation": int(lcs_v.sum()),
            "pmis_hard_violation": int(pmis_v.sum()),
            "rts_hard_violation": int(rts_v.sum()),
            "core_uncertainty_violation": int(core_unc_v.sum()),
            "iis_violation": int(iis_v.sum()),
            "missing_iis_count": int((~iis_available & known).sum()),
            "missing_iis_closes_av_count": 0,
            "multiple_violations": int(((pd.DataFrame({"lcs": lcs_v, "pmis": pmis_v, "rts": rts_v, "unc": core_unc_v, "iis": iis_v}).sum(axis=1)) > 1).sum()),
        })
        binding_rows.append(cap.groupby(["vehicle_profile", "binding_dimension"], dropna=False).size().reset_index(name="orders"))
    out = pd.concat(rows, ignore_index=True)
    output_root.mkdir(parents=True, exist_ok=True)
    fold_root = output_root / "fold=3"
    fold_root.mkdir(parents=True, exist_ok=True)
    out.to_parquet(fold_root / "vehicle_capability_mapping.parquet", index=False, compression="zstd")
    pd.DataFrame(decomp_rows).to_csv(results_dir / "full_day_odd_infeasibility_decomposition.csv", index=False)
    pd.concat(binding_rows, ignore_index=True).to_csv(results_dir / "full_day_odd_binding_dimension.csv", index=False)
    if not reference.empty:
        compare_rows = []
        for column in ["lcs_tail_probability", "pmis_tail_probability", "rts_tail_probability", "overall_uncertainty"]:
            full = pd.to_numeric(demand[column], errors="coerce").dropna()
            ref = pd.to_numeric(reference[column], errors="coerce").dropna()
            compare_rows.append({
                "field": column,
                "fold3_reference_mean": float(ref.mean()),
                "full_day_raw_mean": float(full.mean()),
                "fold3_reference_p50": float(ref.quantile(0.5)),
                "full_day_raw_p50": float(full.quantile(0.5)),
                "fold3_reference_p95": float(ref.quantile(0.95)),
                "full_day_raw_p95": float(full.quantile(0.95)),
                "scale_action": "raw_full_day_stage3_outputs_used_for_capability_gate" if calibration_mode == "none" else "capability_scores_quantile_calibrated_to_fold3_reference",
            })
        pd.DataFrame(compare_rows).to_csv(results_dir / "full_day_stage3_scale_drift_audit.csv", index=False)
    (output_root / "manifest.json").write_text(json.dumps({
        "status": "PASS",
        "rows": int(len(out)),
        "orders": int(demand["order_id"].nunique()),
        "profile_version": doc.get("profile_version"),
        "gate_version": "core_conditional_iis_v1",
        "capability_calibration_mode": calibration_mode,
    }, indent=2), encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.data_root.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    train = build_train_orders(args)
    historical = build_historical_orders(args)
    spec = make_zone_spec(list(train.values()), args.grid_size)
    zone_meta = {
        "zone_system": "fixed_lonlat_grid",
        "grid_size": args.grid_size,
        "min_lon": spec.min_lon,
        "min_lat": spec.min_lat,
        "uses_test_day_future_demand": False,
        "zone_boundary_source": "training_days_only",
        "version": "operational_zone_grid_v1",
    }
    (args.data_root / "operational_zone_system.json").write_text(json.dumps(zone_meta, indent=2), encoding="utf-8")
    chain, chain_stats = build_training_chain_stats(train, spec)
    demand = historical.copy()
    demand = attach_stage3_conditions(demand, args.stage3_inputs)
    eta_table, eta_meta = build_eta_baseline(train, spec)
    stage2_eta = load_stage2_eta(args.stage2_route_conditioned)
    demand = attach_eta(demand, eta_table, eta_meta, spec, stage2_eta)
    request_tables = attach_request_times(demand, chain_stats, args)
    for scenario, frame in request_tables.items():
        path = args.data_root / f"demand_{args.date}_{scenario}.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
    request_summary = []
    for scenario, frame in request_tables.items():
        request_summary.append({
            "scenario": scenario,
            "orders": int(len(frame)),
            "mean_lead_sec": float(frame["latent_request_lead_sec"].mean()),
            "p50_lead_sec": float(frame["latent_request_lead_sec"].quantile(0.5)),
            "p90_lead_sec": float(frame["latent_request_lead_sec"].quantile(0.9)),
            "clipped_share": float(frame["request_lead_clipped"].mean()),
            "condition_known": int(frame["condition_available"].sum()),
            "condition_unknown": int((~frame["condition_available"]).sum()),
        })
    pd.DataFrame(request_summary).to_csv(args.results_dir / "request_time_reconstruction_summary.csv", index=False)
    known = int(demand["condition_available"].sum())
    pd.DataFrame([{
        "raw_orders": int(len(historical)),
        "stage0_valid_orders": int(historical["stage0_valid_order"].sum()),
        "condition_known_orders": known,
        "condition_unknown_orders": int(len(historical) - known),
        "eta_available_orders": int(demand["eta_available"].sum()),
        "eta_stage2_estimated_route_orders": int(demand["eta_source"].eq("stage2_estimated_route_eta").sum()),
        "eta_train_od_time_baseline_orders": int(demand["eta_source"].eq("train_od_time_eta_baseline").sum()),
        "eta_train_timebin_baseline_orders": int(demand["eta_source"].eq("train_timebin_eta_baseline").sum()),
        "eta_train_global_baseline_orders": int(demand["eta_source"].eq("train_global_eta_baseline").sum()),
    }]).to_csv(args.results_dir / "full_day_order_coverage.csv", index=False)
    speed_table, circuity_table, pickup_odd, pickup_summary = build_pickup_calibration_and_odd(train, chain, spec, args.results_dir, args.order_base_root)
    speed_table.to_parquet(args.data_root / "pickup_empty_speed_by_zone_time.parquet", index=False, compression="zstd")
    circuity_table.to_parquet(args.data_root / "pickup_circuity_by_zone.parquet", index=False, compression="zstd")
    pickup_odd.to_parquet(args.data_root / "pickup_odd_zone_pair_proxy.parquet", index=False, compression="zstd")
    train_sessions = {date: build_sessions_for_day(frame, spec, date) for date, frame in train.items()}
    (args.data_root / "training_supply_sessions.parquet").parent.mkdir(parents=True, exist_ok=True)
    pd.concat(train_sessions.values(), ignore_index=True).to_parquet(args.data_root / "training_supply_sessions.parquet", index=False, compression="zstd")
    capability = build_capability(request_tables["RT-Base"], args.profiles, args.capability_reference, args.capability_calibration_mode, args.output_root / "capability_mapping", args.results_dir)
    env_manifest = []
    start = min(frame["simulated_request_time"].min() for frame in request_tables.values()).floor("30s")
    end = max(pd.to_datetime(historical["observed_dropoff_time"], utc=True, errors="coerce").max(), request_tables["RT-Base"]["simulated_request_time"].max()).ceil("30s")
    for replication in range(1, args.replications + 1):
        rng_supply = np.random.default_rng(args.seed + replication * 101)
        agents, sessions, supply_audit = generate_decoupled_supply(train_sessions, args, rng_supply, replication)
        hv_hours = float((pd.to_datetime(sessions["online_end"], utc=True) - pd.to_datetime(sessions["online_start"], utc=True)).dt.total_seconds().sum() / 3600)
        av_online_hours = max(1.0, (end - start).total_seconds() / 3600)
        av_count = int(math.floor((args.target_av_vehicle_hour_share / (1 - args.target_av_vehicle_hour_share)) * hv_hours / av_online_hours))
        depots = build_depots(train, args, av_count, spec)
        av_agents = build_av_agents(depots, args, replication, start, end)
        fleet = pd.concat([
            agents[["vehicle_id", "vehicle_type", "driver_id", "session_id", "depot_id", "initial_zone", "initial_lon", "initial_lat", "online_start", "online_end", "vehicle_profile", "replication_id", "agent_source"]],
            av_agents[["vehicle_id", "vehicle_type", "driver_id", "session_id", "depot_id", "initial_zone", "initial_lon", "initial_lat", "online_start", "online_end", "vehicle_profile", "replication_id", "agent_source"]],
        ], ignore_index=True)
        rep_root = args.output_root / f"replication={replication}"
        rep_root.mkdir(parents=True, exist_ok=True)
        sessions.to_parquet(rep_root / "decoupled_hv_sessions.parquet", index=False, compression="zstd")
        agents.to_parquet(rep_root / "decoupled_hv_agents.parquet", index=False, compression="zstd")
        depots.to_parquet(rep_root / "av_depots.parquet", index=False, compression="zstd")
        av_agents.to_parquet(rep_root / "av_agents.parquet", index=False, compression="zstd")
        fleet.to_parquet(rep_root / "simulation_fleet.parquet", index=False, compression="zstd")
        supply_audit.to_csv(rep_root / "supply_curve_audit.csv", index=False)
        pd.DataFrame([{
            "replication_id": replication,
            "hv_vehicle_hours": hv_hours,
            "av_count": av_count,
            "av_online_hours_per_vehicle": av_online_hours,
            "av_vehicle_hours": av_count * av_online_hours,
            "av_vehicle_hour_share": (av_count * av_online_hours) / (av_count * av_online_hours + hv_hours) if (av_count * av_online_hours + hv_hours) else 0.0,
            "unique_hv_agents": int(len(agents)),
            "peak_concurrent_hv": int(supply_audit["generated_online_HV"].max()),
            "av_to_unique_hv_ratio": av_count / len(agents) if len(agents) else 0.0,
            "av_to_peak_concurrent_hv_ratio": av_count / supply_audit["generated_online_HV"].max() if supply_audit["generated_online_HV"].max() else 0.0,
        }]).to_csv(rep_root / "av_vehicle_hour_summary.csv", index=False)
        env_manifest.append({
            "replication_id": replication,
            "fleet_hash": md5_frame(fleet, ["vehicle_id", "vehicle_type", "online_start", "online_end", "initial_lon", "initial_lat"]),
            "hv_vehicle_hours": hv_hours,
            "av_count": av_count,
            "supply_p90_relative_error": float(supply_audit["relative_error"].dropna().quantile(0.9)),
        })
    pd.concat([pd.read_csv(args.output_root / f"replication={r}" / "supply_curve_audit.csv").assign(replication_id=r) for r in range(1, args.replications + 1)], ignore_index=True).to_csv(args.results_dir / "decoupled_hv_supply_summary.csv", index=False)
    pd.concat([pd.read_csv(args.output_root / f"replication={r}" / "av_vehicle_hour_summary.csv") for r in range(1, args.replications + 1)], ignore_index=True).to_csv(args.results_dir / "av_vehicle_hour_summary.csv", index=False)
    pd.DataFrame([{
        "service_time_realization_mode": "stage2_eta_for_condition_known_else_train_eta_baseline; historical_duration_replay_realization",
        "orders": int(len(demand)),
        "mean_predicted_service_time_sec": float(demand["predicted_service_time_sec"].mean()),
        "mean_realized_service_time_sec": float(demand["realized_service_time_sec"].mean()),
        "eta_global_fallback_share": float(demand["eta_source"].eq("train_global_eta_baseline").mean()),
        "eta_stage2_share": float(demand["eta_source"].eq("stage2_estimated_route_eta").mean()),
    }]).to_csv(args.results_dir / "service_time_residual_summary.csv", index=False)
    (args.output_root / "manifest.json").write_text(json.dumps({
        "status": "PASS",
        "date": args.date,
        "train_dates": args.train_dates,
        "weekend_weight": args.weekend_weight,
        "orders": int(len(historical)),
        "condition_known_orders": known,
        "condition_unknown_orders": int(len(historical) - known),
        "replications": env_manifest,
        "zone_system": zone_meta,
        "pickup_odd_proxy": pickup_summary,
        "request_time_chain_stats": chain_stats,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "orders": int(len(historical)), "condition_known": known, "replications": len(env_manifest)}, indent=2))


if __name__ == "__main__":
    main()
