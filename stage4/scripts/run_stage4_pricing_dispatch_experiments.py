"""ODD-constrained pricing and dynamic AV/HV dispatch experiments.

The simulator uses fixed dispatch windows, availability-aware capability
mapping, passenger/HV acceptance, explicit fares/payouts/profit, and
window-level Hungarian matching for GlobalMatch-style strategies.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


STATUS_OFFLINE = "OFFLINE"
STATUS_IDLE = "IDLE"
STATUS_BUSY = "BUSY"
STATUS_NEAR_FREE = "NEAR_FREE"


@dataclass
class VehicleState:
    vehicle_id: str
    vehicle_type: str
    current_lon: float
    current_lat: float
    current_zone: str
    available_time: pd.Timestamp
    release_lon: float
    release_lat: float
    shift_start: pd.Timestamp
    shift_end: pd.Timestamp
    status: str = STATUS_IDLE
    current_order_id: str | None = None
    cumulative_income: float = 0.0
    cumulative_service_time: float = 0.0
    cumulative_pickup_time: float = 0.0
    cumulative_stress_burden: float = 0.0
    pickup_distance_m: float = 0.0
    service_distance_m: float = 0.0
    order_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-input-root", type=Path, default=Path("stage3/output/stage4_inputs_core_v2"))
    parser.add_argument("--mapping-root", type=Path, default=Path("stage4/output/capability_mapping_v2"))
    parser.add_argument("--pricing-config", type=Path, default=Path("stage4/config/pricing_scenarios.json"))
    parser.add_argument("--stakeholder-config", type=Path, default=Path("stage4/config/stakeholder_parameters.json"))
    parser.add_argument("--dispatch-config", type=Path, default=Path("stage4/config/dispatch_scenarios.json"))
    parser.add_argument("--output-root", type=Path, default=Path("stage4/output/pricing_dispatch"))
    parser.add_argument("--docs-results-root", type=Path, default=Path("stage4/docs/results"))
    parser.add_argument("--max-orders-per-fold", type=int, default=0, help="0 means all orders.")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    r = 6371000.0
    lon1 = np.radians(lon1); lat1 = np.radians(lat1)
    lon2 = np.radians(np.asarray(lon2)); lat2 = np.radians(np.asarray(lat2))
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def zone_id(lon: float, lat: float, grid: float) -> str:
    if pd.isna(lon) or pd.isna(lat):
        return "unknown"
    return f"{math.floor(float(lon) / grid)}:{math.floor(float(lat) / grid)}"


def prepare_orders(frame: pd.DataFrame, grid: float, max_orders: int, seed: int) -> pd.DataFrame:
    orders = frame.copy()
    orders["decision_time"] = pd.to_datetime(orders["decision_time"], errors="coerce", utc=True)
    orders = orders.dropna(subset=["decision_time", "origin_lon", "origin_lat", "destination_lon", "destination_lat"]).copy()
    orders["zone"] = [zone_id(lon, lat, grid) for lon, lat in zip(orders["origin_lon"], orders["origin_lat"])]
    orders["stress_core"] = orders[["lcs_expected", "pmis_expected", "rts_expected"]].mean(axis=1)
    orders["peak_period"] = orders["decision_time"].dt.hour.isin([7, 8, 9, 17, 18, 19])
    if max_orders and len(orders) > max_orders:
        orders = orders.sample(max_orders, random_state=seed).sort_values("decision_time")
    return orders.sort_values("decision_time").reset_index(drop=True)


def build_experiments(config: dict) -> list[dict]:
    main = config["main_experiment"]
    experiments = []
    base = {
        "supply_scenario": "moderate",
        "av_penetration": 0.5,
        "odd_profile": "moderate_av",
        "pricing_scenario": "P3_shared_comp",
        "dispatch_strategy": "ODD-Gated Price-Aware Matching",
        "relocation_mode": "pending_demand",
    }
    for strategy in config["dispatch_strategies"]:
        experiments.append({**base, "experiment_family": "strategy", "dispatch_strategy": strategy})
    for supply in main["supply_scenarios"]:
        experiments.append({**base, "experiment_family": "supply", "supply_scenario": supply})
    for penetration in main["av_penetration_rates"]:
        experiments.append({**base, "experiment_family": "av_penetration", "av_penetration": penetration})
    for profile in main["odd_profiles"]:
        experiments.append({**base, "experiment_family": "odd_profile", "odd_profile": profile})
    for pricing in main["pricing_scenarios"]:
        experiments.append({**base, "experiment_family": "pricing", "pricing_scenario": pricing})
    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for exp in experiments:
        key = tuple(sorted((k, str(v)) for k, v in exp.items() if k != "experiment_family"))
        if key not in seen:
            seen.add(key)
            unique.append(exp)
    return unique


def initial_vehicles(orders: pd.DataFrame, total_fleet: int, av_penetration: float, grid: float, rng: np.random.Generator) -> list[VehicleState]:
    av_count = int(round(total_fleet * av_penetration))
    hv_count = total_fleet - av_count
    start = orders["decision_time"].min()
    end = orders["decision_time"].max() + pd.Timedelta(hours=3)
    first_window = orders[orders["decision_time"].le(start + pd.Timedelta(minutes=30))]
    seed_points = first_window[["origin_lon", "origin_lat"]].dropna()
    if seed_points.empty:
        seed_points = orders[["origin_lon", "origin_lat"]].dropna().head(1)
    sample = seed_points.sample(total_fleet, replace=True, random_state=int(rng.integers(1, 1_000_000)))
    vehicles = []
    for idx, (_, row) in enumerate(sample.iterrows()):
        vehicle_type = "AV" if idx < av_count else "HV"
        shift_start = start if vehicle_type == "AV" else start + pd.Timedelta(minutes=int(rng.integers(0, 120)))
        shift_end = end if vehicle_type == "AV" else end - pd.Timedelta(minutes=int(rng.integers(0, 120)))
        lon = float(row["origin_lon"]); lat = float(row["origin_lat"])
        vehicles.append(VehicleState(
            vehicle_id=f"{vehicle_type}_{idx}",
            vehicle_type=vehicle_type,
            current_lon=lon,
            current_lat=lat,
            current_zone=zone_id(lon, lat, grid),
            available_time=shift_start,
            release_lon=lon,
            release_lat=lat,
            shift_start=shift_start,
            shift_end=shift_end,
            status=STATUS_OFFLINE if shift_start > start else STATUS_IDLE,
        ))
    return vehicles


def update_vehicle_status(vehicles: list[VehicleState], now: pd.Timestamp, grid: float) -> None:
    for vehicle in vehicles:
        if now < vehicle.shift_start or now > vehicle.shift_end:
            vehicle.status = STATUS_OFFLINE
            continue
        if vehicle.available_time <= now:
            vehicle.status = STATUS_IDLE
            vehicle.current_order_id = None
            vehicle.current_zone = zone_id(vehicle.current_lon, vehicle.current_lat, grid)
        elif vehicle.available_time <= now + pd.Timedelta(minutes=5):
            vehicle.status = STATUS_NEAR_FREE
        else:
            vehicle.status = STATUS_BUSY


def edge_price(order: pd.Series, vehicle: VehicleState, pickup_m: float, supply_ratio: float, pricing: dict, pricing_root: dict, stakeholder: dict) -> dict:
    distance_km = float(order["route_length_m"]) / 1000.0
    service_min = max(distance_km / 0.48, 1.0)  # approx at 8 m/s
    base = pricing_root["base_fare"] + distance_km * pricing_root["distance_fare_per_km"] + service_min * pricing_root["time_fare_per_min"]
    surge = min(pricing_root["surge_cap"], max(pricing_root["surge_floor"], supply_ratio))
    vehicle_adj = pricing["av_vehicle_adjustment"] if vehicle.vehicle_type == "AV" else pricing["hv_vehicle_adjustment"]
    stress_surcharge = min(pricing_root["stress_surcharge_cap"], float(order["composite_expected"]) * pricing["stress_surcharge_multiplier"])
    hv_comp = 0.0
    if vehicle.vehicle_type == "HV":
        hv_comp = float(order["composite_expected"]) * pricing["hv_compensation_multiplier"]
    fare = max(0.0, base * surge + vehicle_adj + stress_surcharge * pricing["compensation_passenger_share"])
    passenger = stakeholder["passenger"]
    pickup_min = pickup_m / 7.0 / 60.0
    gc = fare + passenger["value_of_pick_time_per_min" if False else "value_of_pickup_time_per_min"] * pickup_min + passenger["value_of_wait_time_per_min"] * pickup_min
    gc += passenger["av_preference_adjustment"] if vehicle.vehicle_type == "AV" else passenger["hv_preference_adjustment"]
    cap = pricing.get("passenger_gc_cap", passenger["generalized_cost_threshold"])
    passenger_accept = gc <= cap
    driver_payout = np.nan
    driver_utility = np.nan
    driver_accept = True
    driver_reject_reason = ""
    platform_cost = 0.0
    if vehicle.vehicle_type == "HV":
        driver = stakeholder["driver"]
        driver_payout = driver["base_payout"] + service_min * driver["service_time_payout_per_min"] + pickup_m / 1000.0 * driver["pickup_compensation_per_km"] + hv_comp
        if bool(order["peak_period"]):
            driver_payout += driver["peak_scarcity_bonus"]
        driver_cost = pickup_m / 1000.0 * driver["pickup_cost_per_km"] + service_min * driver["service_cost_per_min"]
        stress_disutility = float(order["composite_expected"]) * driver["stress_disutility_multiplier"]
        driver_utility = driver_payout - driver_cost - stress_disutility
        threshold = pricing.get("minimum_driver_utility", driver["minimum_utility"])
        driver_accept = driver_utility >= threshold
        driver_reject_reason = "" if driver_accept else "driver_utility_below_threshold"
        platform_cost = driver_payout + stakeholder["platform"]["hv_variable_cost"] + hv_comp * pricing["compensation_platform_share"]
    else:
        platform = stakeholder["platform"]
        platform_cost = pickup_m / 1000.0 * platform["av_pickup_cost_per_km"] + distance_km * (platform["av_service_cost_per_km"] + platform["av_energy_cost_per_km"]) + service_min * platform["av_service_time_cost_per_min"]
    profit = fare - platform_cost
    return {
        "quoted_fare": fare,
        "surge_multiplier": surge,
        "stress_surcharge": stress_surcharge,
        "passenger_generalized_cost": gc,
        "passenger_accept": passenger_accept,
        "passenger_rejection_reason": "" if passenger_accept else "generalized_cost_above_threshold",
        "driver_payout": driver_payout,
        "driver_utility": driver_utility,
        "driver_accept": driver_accept,
        "driver_rejection_reason": driver_reject_reason,
        "hv_compensation": hv_comp,
        "compensation_passenger_share": pricing["compensation_passenger_share"],
        "compensation_platform_share": pricing["compensation_platform_share"],
        "platform_cost": platform_cost,
        "platform_profit": profit,
        "gross_booking_value": fare,
    }


def candidate_edges(orders: pd.DataFrame, vehicles: list[VehicleState], mapping: dict, now: pd.Timestamp, exp: dict, pricing: dict, pricing_root: dict, stakeholder: dict, config: dict) -> tuple[pd.DataFrame, dict]:
    available = [v for v in vehicles if v.status in {STATUS_IDLE, STATUS_NEAR_FREE} and v.available_time <= now]
    if not available:
        return pd.DataFrame(), {"mean_candidate_count": 0.0, "zero_candidate_share": 1.0, "cross_zone_candidate_share": 0.0}
    edges = []
    counts = []
    cross = 0
    supply_ratio = len(orders) / max(len(available), 1)
    for _, order in orders.iterrows():
        lon = np.array([v.current_lon for v in available])
        lat = np.array([v.current_lat for v in available])
        dist = haversine_m(float(order["origin_lon"]), float(order["origin_lat"]), lon, lat)
        feasible_idx = np.where(dist <= config["max_pickup_m"])[0]
        if len(feasible_idx) > config["max_candidates_per_order"]:
            feasible_idx = feasible_idx[np.argsort(dist[feasible_idx])[: config["max_candidates_per_order"]]]
        count = 0
        for pos in feasible_idx:
            vehicle = available[int(pos)]
            pickup_m = float(dist[int(pos)])
            odd_feasible = True
            odd_margin = np.nan
            binding = ""
            if vehicle.vehicle_type == "AV":
                cap = mapping.get((order["order_id"], exp["odd_profile"]))
                odd_feasible = bool(cap["service_feasible"]) if cap is not None else True
                odd_margin = float(cap["ODD_margin"]) if cap is not None else np.nan
                binding = cap.get("binding_dimension", "") if cap is not None else ""
            econ = edge_price(order, vehicle, pickup_m, supply_ratio, pricing, pricing_root, stakeholder)
            passenger_accept = bool(econ["passenger_accept"])
            driver_accept = bool(econ["driver_accept"])
            if exp["dispatch_strategy"] in {"ODD Gate Only", "ODD-Gated Price-Aware Matching", "Three-Stakeholder Balanced"} and vehicle.vehicle_type == "AV" and not odd_feasible:
                continue
            if not passenger_accept or not driver_accept:
                continue
            count += 1
            cross += int(vehicle.current_zone != order["zone"])
            edges.append({
                "order_id": order["order_id"],
                "vehicle_id": vehicle.vehicle_id,
                "vehicle_type": vehicle.vehicle_type,
                "pickup_m": pickup_m,
                "time_feasible": True,
                "space_feasible": True,
                "vehicle_available": True,
                "ODD_feasible": odd_feasible,
                "passenger_accepted": passenger_accept,
                "driver_accepted": driver_accept,
                "ODD_margin": odd_margin,
                "binding_ODD_dimension": binding,
                "stress_burden": float(order["composite_expected"]),
                **econ,
            })
        counts.append(count)
    stats = {
        "mean_candidate_count": float(np.mean(counts)) if counts else 0.0,
        "zero_candidate_share": float(np.mean([c == 0 for c in counts])) if counts else 1.0,
        "cross_zone_candidate_share": cross / max(sum(counts), 1),
    }
    return pd.DataFrame(edges), stats


def match_edges(edges: pd.DataFrame, strategy: str, rng: np.random.Generator) -> pd.DataFrame:
    if edges.empty:
        return edges
    if strategy == "Random":
        shuffled = edges.sample(frac=1, random_state=int(rng.integers(0, 1_000_000)))
        return shuffled.drop_duplicates("order_id").drop_duplicates("vehicle_id")
    if strategy == "Nearest":
        sorted_edges = edges.sort_values("pickup_m")
        return sorted_edges.drop_duplicates("order_id").drop_duplicates("vehicle_id")
    orders = list(edges["order_id"].unique())
    vehicles = list(edges["vehicle_id"].unique())
    order_idx = {o: i for i, o in enumerate(orders)}
    vehicle_idx = {v: i for i, v in enumerate(vehicles)}
    big = 1e9
    cost = np.full((len(orders), len(vehicles)), big, dtype="float64")
    for i, row in edges.iterrows():
        base = row["pickup_m"]
        if strategy == "GlobalMatch-MinOperatingCost":
            base = row["platform_cost"]
        elif strategy == "Cost-only":
            base = row["platform_cost"] - row["platform_profit"]
        elif strategy == "Simple Risk Penalty":
            base = row["pickup_m"] + (row["stress_burden"] * 2500.0 if row["vehicle_type"] == "AV" else 0.0)
        elif strategy == "ODD Gate Only":
            base = row["pickup_m"]
        elif strategy == "ODD-Gated Price-Aware Matching":
            base = -row["platform_profit"] + row["passenger_generalized_cost"] * 0.15
        elif strategy == "Three-Stakeholder Balanced":
            base = -row["platform_profit"] + row["passenger_generalized_cost"] * 0.2
            if row["vehicle_type"] == "HV":
                base -= max(row.get("driver_utility", 0.0), 0.0) * 0.5
            if row["vehicle_type"] == "AV":
                base += max(-row.get("ODD_margin", 0.0), 0.0) * 5000.0
        cost[order_idx[row["order_id"]], vehicle_idx[row["vehicle_id"]]] = min(cost[order_idx[row["order_id"]], vehicle_idx[row["vehicle_id"]]], base)
    r, c = linear_sum_assignment(cost)
    pairs = {(orders[ri], vehicles[ci]) for ri, ci in zip(r, c) if cost[ri, ci] < big / 2}
    return edges[edges.apply(lambda row: (row["order_id"], row["vehicle_id"]) in pairs, axis=1)].drop_duplicates(["order_id", "vehicle_id"])


def run_one(inputs: pd.DataFrame, mapping_frame: pd.DataFrame, exp: dict, configs: dict, rng: np.random.Generator, output_dir: Path) -> dict:
    dispatch = configs["dispatch"]
    pricing_root = configs["pricing"]
    pricing = pricing_root["pricing_scenarios"][exp["pricing_scenario"]]
    stakeholder = configs["stakeholder"]
    total_fleet = dispatch["supply_scenarios"][exp["supply_scenario"]]["total_fleet"]
    vehicles = initial_vehicles(inputs, total_fleet, float(exp["av_penetration"]), dispatch["candidate_zone_grid_size"], rng)
    cap_map = mapping_frame.set_index(["order_id", "vehicle_profile"]).to_dict("index")
    interval = pd.Timedelta(seconds=dispatch["dispatch_interval_seconds"])
    start = inputs["decision_time"].min().floor(f"{dispatch['dispatch_interval_seconds']}s")
    inputs = inputs.copy()
    inputs["dispatch_window"] = inputs["decision_time"].dt.floor(f"{dispatch['dispatch_interval_seconds']}s")
    pending: list[dict] = []
    order_logs = []
    window_logs = []
    windows = sorted(inputs["dispatch_window"].dropna().unique().tolist())
    if windows:
        windows.append(pd.Timestamp(windows[-1]) + pd.Timedelta(seconds=dispatch["passenger_patience_seconds"]) + interval)
    order_cursor = 0
    orders = inputs.sort_values("decision_time").reset_index(drop=True)
    for now in windows:
        update_vehicle_status(vehicles, now, dispatch["candidate_zone_grid_size"])
        new_orders = []
        while order_cursor < len(orders) and orders.loc[order_cursor, "decision_time"] <= now:
            item = orders.loc[order_cursor].to_dict()
            item["arrival_window"] = now
            pending.append(item)
            new_orders.append(item)
            order_cursor += 1
        still = []
        cancelled = []
        for item in pending:
            if (now - item["decision_time"]).total_seconds() > dispatch["passenger_patience_seconds"]:
                cancelled.append(item)
            else:
                still.append(item)
        pending = still
        for item in cancelled:
            order_logs.append({"order_id": item["order_id"], "fold": item["fold"], "decision_time": item["decision_time"], "zone": item["zone"], "served": False, "cancelled": True, "cancellation_reason": "passenger_patience_timeout", "pricing_mechanism": exp["pricing_scenario"], "dispatch_strategy": exp["dispatch_strategy"]})
        pending_frame = pd.DataFrame(pending)
        available_before = [v for v in vehicles if v.status in {STATUS_IDLE, STATUS_NEAR_FREE} and v.available_time <= now]
        edges, cand_stats = candidate_edges(pending_frame, vehicles, cap_map, now, exp, pricing, pricing_root, stakeholder, dispatch) if len(pending_frame) else (pd.DataFrame(), {"mean_candidate_count": 0.0, "zero_candidate_share": 0.0, "cross_zone_candidate_share": 0.0})
        selected = match_edges(edges, exp["dispatch_strategy"], rng)
        selected_orders = set(selected["order_id"]) if not selected.empty else set()
        for _, edge in selected.iterrows():
            order = pending_frame[pending_frame["order_id"].eq(edge["order_id"])].iloc[0]
            vehicle = next(v for v in vehicles if v.vehicle_id == edge["vehicle_id"])
            pickup_sec = float(edge["pickup_m"]) / dispatch["pickup_speed_mps"]
            service_sec = max(float(order["route_length_m"]) / dispatch["service_speed_mps"], 60.0)
            wait_sec = max((now - order["decision_time"]).total_seconds(), 0.0)
            vehicle.available_time = now + pd.Timedelta(seconds=pickup_sec + service_sec)
            vehicle.current_lon = float(order["destination_lon"])
            vehicle.current_lat = float(order["destination_lat"])
            vehicle.release_lon = vehicle.current_lon
            vehicle.release_lat = vehicle.current_lat
            vehicle.status = STATUS_BUSY
            vehicle.current_order_id = edge["order_id"]
            vehicle.cumulative_income += 0 if vehicle.vehicle_type == "AV" else float(edge["driver_payout"])
            vehicle.cumulative_service_time += service_sec
            vehicle.cumulative_pickup_time += pickup_sec
            vehicle.cumulative_stress_burden += float(order["composite_expected"])
            vehicle.pickup_distance_m += float(edge["pickup_m"])
            vehicle.service_distance_m += float(order["route_length_m"])
            vehicle.order_count += 1
            order_logs.append({
                "order_id": edge["order_id"], "fold": int(order["fold"]), "decision_time": order["decision_time"], "zone": order["zone"],
                "origin_lon": order["origin_lon"], "origin_lat": order["origin_lat"], "destination_lon": order["destination_lon"], "destination_lat": order["destination_lat"],
                "lcs_expected": order["lcs_expected"], "pmis_expected": order["pmis_expected"], "rts_expected": order["rts_expected"],
                "core_overall_high_stress_probability": order["core_overall_high_stress_probability"],
                "extended_overall_high_stress_probability": order.get("extended_overall_high_stress_probability", np.nan),
                "iis_availability": order["iis_availability"], "quoted_vehicle_type": vehicle.vehicle_type, "quoted_fare": edge["quoted_fare"],
                "accepted": True, "assigned_vehicle": vehicle.vehicle_id, "pickup_time_sec": pickup_sec, "waiting_time_sec": wait_sec,
                "service_time_sec": service_sec, "passenger_generalized_cost": edge["passenger_generalized_cost"], "driver_payout": edge["driver_payout"],
                "driver_utility": edge["driver_utility"], "platform_profit": edge["platform_profit"], "ODD_feasible": edge["ODD_feasible"],
                "ODD_margin": edge["ODD_margin"], "binding_ODD_dimension": edge["binding_ODD_dimension"], "served": True, "cancelled": False,
                "cancellation_reason": "", "pricing_mechanism": exp["pricing_scenario"], "dispatch_strategy": exp["dispatch_strategy"],
                "gross_booking_value": edge["gross_booking_value"], "platform_cost": edge["platform_cost"], "hv_compensation": edge["hv_compensation"],
                "compensation_passenger_share": edge["compensation_passenger_share"], "compensation_platform_share": edge["compensation_platform_share"],
                "vehicle_profile": exp["odd_profile"], "supply_scenario": exp["supply_scenario"], "av_penetration": exp["av_penetration"],
            })
        pending = [item for item in pending if item["order_id"] not in selected_orders]
        if exp["relocation_mode"] == "pending_demand" and pending:
            target = pending[0]
            idle_av = [v for v in vehicles if v.vehicle_type == "AV" and v.status == STATUS_IDLE]
            for vehicle in idle_av[: max(1, len(idle_av) // 20)]:
                vehicle.current_lon = 0.95 * vehicle.current_lon + 0.05 * float(target["origin_lon"])
                vehicle.current_lat = 0.95 * vehicle.current_lat + 0.05 * float(target["origin_lat"])
        available_after = [v for v in vehicles if v.status in {STATUS_IDLE, STATUS_NEAR_FREE}]
        window_logs.append({
            "window_time": now, "new_orders": len(new_orders), "pending_orders": len(pending), "matched_orders": len(selected), "cancelled_orders": len(cancelled),
            "available_AV": sum(v.vehicle_type == "AV" for v in available_after), "available_HV": sum(v.vehicle_type == "HV" for v in available_after),
            "busy_AV": sum(v.vehicle_type == "AV" and v.status == STATUS_BUSY for v in vehicles), "busy_HV": sum(v.vehicle_type == "HV" and v.status == STATUS_BUSY for v in vehicles),
            **cand_stats,
        })
    for item in pending:
        order_logs.append({"order_id": item["order_id"], "fold": item["fold"], "decision_time": item["decision_time"], "zone": item["zone"], "served": False, "cancelled": True, "cancellation_reason": "end_of_day_unserved", "pricing_mechanism": exp["pricing_scenario"], "dispatch_strategy": exp["dispatch_strategy"]})
    order_log = pd.DataFrame(order_logs)
    window_log = pd.DataFrame(window_logs)
    vehicle_log = pd.DataFrame([asdict(v) for v in vehicles])
    for frame in [order_log, window_log, vehicle_log]:
        for column in frame.columns:
            if pd.api.types.is_datetime64_any_dtype(frame[column]):
                frame[column] = frame[column].astype(str)
    output_dir.mkdir(parents=True, exist_ok=True)
    order_log.to_parquet(output_dir / "order_log.parquet", index=False, compression="zstd")
    window_log.to_parquet(output_dir / "window_log.parquet", index=False, compression="zstd")
    vehicle_log.to_parquet(output_dir / "vehicle_log.parquet", index=False, compression="zstd")
    served = order_log[order_log["served"].fillna(False)]
    hv = served[served["quoted_vehicle_type"].eq("HV")]
    av = served[served["quoted_vehicle_type"].eq("AV")]
    total_orders = len(inputs)
    return {
        **exp,
        "orders": total_orders,
        "served_orders": len(served),
        "match_rate": len(served) / max(total_orders, 1),
        "cancel_rate": float(order_log["cancelled"].fillna(False).mean()) if len(order_log) else 1.0,
        "passenger_acceptance_rate": len(served) / max(total_orders, 1),
        "mean_waiting_time_sec": float(served["waiting_time_sec"].mean()) if len(served) else np.nan,
        "mean_pickup_time_sec": float(served["pickup_time_sec"].mean()) if len(served) else np.nan,
        "mean_passenger_fare": float(served["quoted_fare"].mean()) if len(served) else np.nan,
        "mean_passenger_generalized_cost": float(served["passenger_generalized_cost"].mean()) if len(served) else np.nan,
        "platform_revenue": float(served["gross_booking_value"].sum()) if len(served) else 0.0,
        "platform_cost": float(served["platform_cost"].sum()) if len(served) else 0.0,
        "platform_profit": float(served["platform_profit"].sum()) if len(served) else 0.0,
        "profit_margin": float(served["platform_profit"].sum() / max(served["gross_booking_value"].sum(), 1e-9)) if len(served) else np.nan,
        "HV_payout": float(hv["driver_payout"].sum()) if len(hv) else 0.0,
        "HV_net_income": float(hv["driver_utility"].sum()) if len(hv) else 0.0,
        "HV_stress_burden": float(hv[["lcs_expected", "pmis_expected", "rts_expected"]].mean(axis=1).mean()) if len(hv) else np.nan,
        "HV_high_stress_assignment_share": float(hv["core_overall_high_stress_probability"].ge(0.5).mean()) if len(hv) else np.nan,
        "AV_utilization": float(vehicle_log.loc[vehicle_log["vehicle_type"].eq("AV"), "order_count"].gt(0).mean()) if (vehicle_log["vehicle_type"].eq("AV")).any() else 0.0,
        "AV_assignment_share": len(av) / max(len(served), 1),
        "AV_stress_exposure": float(av[["lcs_expected", "pmis_expected", "rts_expected"]].mean(axis=1).mean()) if len(av) else np.nan,
        "AV_ODD_violation_rate": float((~av["ODD_feasible"].fillna(True)).mean()) if len(av) else 0.0,
        "mean_candidate_count": float(window_log["mean_candidate_count"].mean()) if len(window_log) else 0.0,
        "zero_candidate_order_share": float(window_log["zero_candidate_share"].mean()) if len(window_log) else 0.0,
        "cross_zone_candidate_share": float(window_log["cross_zone_candidate_share"].mean()) if len(window_log) else 0.0,
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.docs_results_root.mkdir(parents=True, exist_ok=True)
    configs = {
        "pricing": json.loads(args.pricing_config.read_text(encoding="utf-8")),
        "stakeholder": json.loads(args.stakeholder_config.read_text(encoding="utf-8")),
        "dispatch": json.loads(args.dispatch_config.read_text(encoding="utf-8")),
    }
    experiments = build_experiments(configs["dispatch"])
    summaries = []
    rng = np.random.default_rng(args.seed)
    for fold_path in sorted(args.stage3_input_root.glob("fold=*/stage4_inputs.parquet")):
        fold = int(fold_path.parent.name.split("=", 1)[-1])
        raw = pd.read_parquet(fold_path)
        raw["fold"] = fold
        inputs = prepare_orders(raw, configs["dispatch"]["candidate_zone_grid_size"], args.max_orders_per_fold, args.seed + fold)
        mapping = pd.read_parquet(args.mapping_root / f"fold={fold}" / "vehicle_capability_mapping.parquet")
        for exp_id, exp in enumerate(experiments):
            out_dir = args.output_root / f"fold={fold}" / f"exp={exp_id:03d}_{exp['experiment_family']}"
            summary = run_one(inputs, mapping, exp, configs, rng, out_dir)
            summary["fold"] = fold
            summary["experiment_id"] = exp_id
            summaries.append(summary)
            print(f"fold={fold} exp={exp_id} {exp['experiment_family']} {exp['dispatch_strategy']} match={summary['match_rate']:.3f} profit={summary['platform_profit']:.1f}", flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_root / "scenario_summary.csv", index=False)
    # Small result files for git/docs.
    summary.to_csv(args.docs_results_root / "scenario_summary.csv", index=False)
    stakeholder_cols = ["fold", "experiment_family", "dispatch_strategy", "pricing_scenario", "supply_scenario", "av_penetration", "odd_profile", "mean_passenger_fare", "mean_passenger_generalized_cost", "platform_profit", "HV_net_income", "HV_stress_burden", "AV_stress_exposure"]
    summary[stakeholder_cols].to_csv(args.docs_results_root / "stakeholder_summary.csv", index=False)
    summary[["fold", "pricing_scenario", "dispatch_strategy", "mean_passenger_fare", "platform_profit", "HV_payout", "HV_net_income"]].to_csv(args.docs_results_root / "pricing_summary.csv", index=False)
    summary[summary["experiment_family"].eq("odd_profile")].to_csv(args.docs_results_root / "odd_sensitivity_summary.csv", index=False)
    summary[summary["experiment_family"].eq("av_penetration")].to_csv(args.docs_results_root / "av_penetration_summary.csv", index=False)
    manifest = {"status": "PASS", "experiments": len(summary), "folds": sorted(summary["fold"].unique().tolist()), "max_orders_per_fold": args.max_orders_per_fold}
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
