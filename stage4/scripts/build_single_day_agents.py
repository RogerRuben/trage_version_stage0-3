"""Build full-day demand, HV agents/sessions, AV depots, and simulation fleet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20161023")
    parser.add_argument("--train-dates", default="20161019,20161020,20161021,20161022")
    parser.add_argument("--od-root", type=Path, default=Path("stage0/output/order_od_audited"))
    parser.add_argument("--order-base-root", type=Path, default=Path("stage0/output/order_base"))
    parser.add_argument("--stage4-inputs", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("stage4/config/single_day_fleet_config.json"))
    parser.add_argument("--output-data-dir", type=Path, default=Path("stage4/data"))
    parser.add_argument("--results-dir", type=Path, default=Path("stage4/docs/results"))
    parser.add_argument("--av-ratio", type=float, default=None)
    parser.add_argument("--num-depots", type=int, default=None)
    return parser.parse_args()


def _read_day(root: Path, date: str) -> pd.DataFrame:
    return pd.read_parquet(root / f"day={date}.parquet")


def _haversine_m(lon1, lat1, lon2, lat2):
    r = 6371000.0
    lon1 = np.radians(lon1); lat1 = np.radians(lat1); lon2 = np.radians(lon2); lat2 = np.radians(lat2)
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build_historical_orders(args: argparse.Namespace) -> pd.DataFrame:
    od = _read_day(args.od_root, args.date)
    base = _read_day(args.order_base_root, args.date)[[
        "order_id", "quality_flag", "quality_tier", "matching_success", "matched_route_length_m",
        "matched_link_count", "matching_confidence",
    ]]
    orders = od.merge(base, on="order_id", how="left", validate="one_to_one")
    orders = orders.rename(columns={"driver_id": "historical_driver_id", "matched_route_length_m": "route_length_m"})
    orders["matched_route_available"] = orders["matching_success"].fillna(False).astype(bool)
    orders["stage0_valid_order"] = orders["coordinate_valid"].fillna(False).astype(bool) & orders["duration_valid"].fillna(False).astype(bool)
    keep = [
        "order_id", "historical_driver_id", "date", "origin_timestamp", "destination_timestamp",
        "origin_lon", "origin_lat", "destination_lon", "destination_lat", "duration_sec",
        "route_length_m", "matched_link_count", "quality_flag", "quality_tier", "matching_confidence",
        "matched_route_available", "od_route_eligible", "stage0_valid_order",
    ]
    return orders[[column for column in keep if column in orders.columns]].copy()


def build_hv_agents_sessions(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    work = orders.dropna(subset=["historical_driver_id", "origin_timestamp", "destination_timestamp", "origin_lon", "origin_lat"]).copy()
    work["origin_timestamp"] = pd.to_datetime(work["origin_timestamp"], unit="s", utc=True, errors="coerce")
    work["destination_timestamp"] = pd.to_datetime(work["destination_timestamp"], unit="s", utc=True, errors="coerce")
    work = work[work["origin_timestamp"].notna() & work["destination_timestamp"].notna()]
    work = work.sort_values(["historical_driver_id", "origin_timestamp"], kind="mergesort")
    work["prev_destination"] = work.groupby("historical_driver_id")["destination_timestamp"].shift()
    gap_min = (work["origin_timestamp"] - work["prev_destination"]).dt.total_seconds() / 60
    gap_stats = gap_min.dropna().quantile([0.5, 0.75, 0.90, 0.95]).to_dict()
    threshold = 90 if gap_stats.get(0.90, 0) > 60 else 60
    new_session = gap_min.isna() | gap_min.gt(threshold)
    work["session_seq"] = new_session.groupby(work["historical_driver_id"]).cumsum().astype(int)
    sessions = []
    for (driver, seq), group in work.groupby(["historical_driver_id", "session_seq"], sort=False):
        sessions.append({
            "driver_id": driver,
            "session_id": f"HV_{driver}_{seq}",
            "online_start": group["origin_timestamp"].iloc[0],
            "online_end": group["destination_timestamp"].iloc[-1],
            "initial_lon": float(group["origin_lon"].iloc[0]),
            "initial_lat": float(group["origin_lat"].iloc[0]),
            "historical_order_count": int(len(group)),
            "first_order_id": group["order_id"].iloc[0],
            "last_order_id": group["order_id"].iloc[-1],
            "session_gap_threshold_min": threshold,
        })
    sessions_df = pd.DataFrame(sessions)
    sessions_df["online_start"] = pd.to_datetime(sessions_df["online_start"], utc=True)
    sessions_df["online_end"] = pd.to_datetime(sessions_df["online_end"], utc=True)
    zero_or_negative = sessions_df["online_end"].le(sessions_df["online_start"])
    sessions_df.loc[zero_or_negative, "online_end"] = sessions_df.loc[zero_or_negative, "online_start"] + pd.Timedelta(seconds=1)
    agents = sessions_df.sort_values("online_start").groupby("driver_id", as_index=False).agg(
        session_count=("session_id", "count"),
        first_observed_time=("online_start", "min"),
        last_observed_time=("online_end", "max"),
        first_observed_lon=("initial_lon", "first"),
        first_observed_lat=("initial_lat", "first"),
        historical_order_count=("historical_order_count", "sum"),
    )
    agents["vehicle_id"] = "HV_" + agents["driver_id"].astype(str)
    agents["vehicle_type"] = "HV"
    agents["agent_source"] = "observed_full_day_driver"
    gap_summary = pd.DataFrame([{
        "gap_p50_min": gap_stats.get(0.5, np.nan),
        "gap_p75_min": gap_stats.get(0.75, np.nan),
        "gap_p90_min": gap_stats.get(0.90, np.nan),
        "gap_p95_min": gap_stats.get(0.95, np.nan),
        "gap_max_min": float(gap_min.max(skipna=True)),
        "session_gap_threshold_min": threshold,
    }])
    sessions_df["vehicle_id"] = "HV_" + sessions_df["driver_id"].astype(str)
    return agents, sessions_df, gap_summary, threshold


def _load_config(args: argparse.Namespace) -> dict:
    if args.config.exists():
        config = json.loads(args.config.read_text(encoding="utf-8"))
    else:
        config = {"test_day": args.date, "av_ratio_to_hv": 0.05, "maximum_av_ratio_to_hv": 0.05, "num_depots": 8, "av_profile": "moderate_av"}
    if args.av_ratio is not None:
        config["av_ratio_to_hv"] = args.av_ratio
    if args.num_depots is not None:
        config["num_depots"] = args.num_depots
    ratio = float(config.get("av_ratio_to_hv", 0.05))
    cap = float(config.get("maximum_av_ratio_to_hv", 0.05))
    if ratio < 0 or ratio > cap or cap > 0.05:
        raise ValueError(f"AV ratio must satisfy 0 <= ratio <= maximum <= 0.05; got ratio={ratio}, max={cap}")
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def build_depots(args: argparse.Namespace, config: dict, av_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dates = [part.strip() for part in args.train_dates.split(",") if part.strip()]
    origins = []
    for date in train_dates:
        frame = _read_day(args.od_root, date)
        origins.append(frame[["origin_lon", "origin_lat"]].dropna())
    train = pd.concat(origins, ignore_index=True)
    n_depots = int(config.get("num_depots", 8))
    coords = train[["origin_lon", "origin_lat"]].to_numpy(float)
    km = KMeans(n_clusters=n_depots, random_state=2026, n_init=10)
    labels = km.fit_predict(coords)
    train["cluster"] = labels
    depots = []
    for cluster, group in train.groupby("cluster"):
        center = km.cluster_centers_[int(cluster)]
        dist = _haversine_m(group["origin_lon"].to_numpy(), group["origin_lat"].to_numpy(), center[0], center[1])
        medoid = group.iloc[int(np.argmin(dist))]
        depots.append({
            "depot_id": f"DEPOT_{int(cluster):02d}",
            "zone_id": f"cluster_{int(cluster):02d}",
            "lon": float(medoid["origin_lon"]),
            "lat": float(medoid["origin_lat"]),
            "road_node_id": f"training_origin_medoid_{int(cluster):02d}",
            "capacity": int(np.ceil(av_count / max(n_depots, 1))),
            "training_date_range": ",".join(train_dates),
            "depot_generation_method": "training_origin_kmeans_medoid_no_test_day_demand",
            "training_origin_count": int(len(group)),
        })
    depot_df = pd.DataFrame(depots).sort_values("depot_id").reset_index(drop=True)
    counts = np.zeros(len(depot_df), dtype=int)
    weights = depot_df["training_origin_count"].to_numpy(float)
    weights = weights / weights.sum()
    base = np.floor(weights * av_count).astype(int)
    counts += base
    remaining = av_count - int(counts.sum())
    if remaining > 0:
        order = np.argsort(-(weights * av_count - base))
        counts[order[:remaining]] += 1
    depot_df["assigned_av_count"] = counts
    return depot_df, train


def build_av_agents(depots: pd.DataFrame, config: dict, simulation_start, simulation_end) -> pd.DataFrame:
    rows = []
    for _, depot in depots.iterrows():
        for index in range(int(depot["assigned_av_count"])):
            rows.append({
                "vehicle_id": f"AV_{depot['depot_id']}_{index:04d}",
                "vehicle_type": "AV",
                "depot_id": depot["depot_id"],
                "initial_lon": float(depot["lon"]),
                "initial_lat": float(depot["lat"]),
                "online_start": simulation_start,
                "online_end": simulation_end,
                "vehicle_profile": config.get("av_profile", "moderate_av"),
                "agent_source": "training_data_depot",
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_data_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(args)
    orders = build_historical_orders(args)
    orders.to_parquet(args.output_data_dir / f"test_day_{args.date}_historical_orders.parquet", index=False, compression="zstd")
    agents, sessions, gap_summary, threshold = build_hv_agents_sessions(orders)
    agents.to_parquet(args.output_data_dir / f"hv_agents_{args.date}.parquet", index=False, compression="zstd")
    sessions.to_parquet(args.output_data_dir / f"hv_agent_sessions_{args.date}.parquet", index=False, compression="zstd")
    hv_count = int(len(agents))
    ratio = float(config.get("av_ratio_to_hv", 0.05))
    av_count = int(np.floor(ratio * hv_count))
    depots, _ = build_depots(args, config, av_count)
    depots.to_parquet(args.output_data_dir / "av_depots.parquet", index=False, compression="zstd")
    simulation_start = pd.to_datetime(orders["origin_timestamp"], unit="s", utc=True, errors="coerce").min()
    simulation_end = pd.to_datetime(orders["destination_timestamp"], unit="s", utc=True, errors="coerce").max()
    av_agents = build_av_agents(depots, config, simulation_start, simulation_end)
    av_agents.to_parquet(args.output_data_dir / f"av_agents_{args.date}.parquet", index=False, compression="zstd")
    hv_fleet = sessions[["vehicle_id", "driver_id", "session_id", "online_start", "online_end", "initial_lon", "initial_lat"]].copy()
    hv_fleet["vehicle_type"] = "HV"
    hv_fleet["depot_id"] = pd.NA
    hv_fleet["vehicle_profile"] = "reference_hv"
    hv_fleet["agent_source"] = "observed_driver_session"
    av_fleet = av_agents.copy()
    av_fleet["driver_id"] = pd.NA
    av_fleet["session_id"] = pd.NA
    fleet = pd.concat([
        hv_fleet[["vehicle_id", "vehicle_type", "driver_id", "session_id", "depot_id", "initial_lon", "initial_lat", "online_start", "online_end", "vehicle_profile", "agent_source"]],
        av_fleet[["vehicle_id", "vehicle_type", "driver_id", "session_id", "depot_id", "initial_lon", "initial_lat", "online_start", "online_end", "vehicle_profile", "agent_source"]],
    ], ignore_index=True)
    fleet.to_parquet(args.output_data_dir / f"simulation_fleet_{args.date}.parquet", index=False, compression="zstd")
    manifest = {
        "date": args.date,
        "historical_orders": int(len(orders)),
        "hv_agent_count": hv_count,
        "hv_session_count": int(len(sessions)),
        "session_gap_threshold_min": int(threshold),
        "av_ratio_to_hv": ratio,
        "maximum_av_ratio_to_hv": float(config.get("maximum_av_ratio_to_hv", 0.05)),
        "av_count": av_count,
        "depot_count": int(len(depots)),
        "simulation_start": str(simulation_start),
        "simulation_end": str(simulation_end),
    }
    (args.output_data_dir / f"simulation_fleet_{args.date}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame([{"historical_orders": len(orders), "hv_agent_count": hv_count, "hv_session_count": len(sessions), "av_count": av_count, "depot_count": len(depots), "av_ratio_to_hv": ratio}]).to_csv(args.results_dir / "hv_agent_summary.csv", index=False)
    gap_summary.to_csv(args.results_dir / "hv_session_summary.csv", index=False)
    depots[["depot_id", "lon", "lat", "assigned_av_count", "training_origin_count", "training_date_range", "depot_generation_method"]].to_csv(args.results_dir / "av_depot_summary.csv", index=False)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
