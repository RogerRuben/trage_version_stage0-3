"""Legacy prototype: run demand-supply decoupled single-day AV/HV ABM.

The simulator uses pre-generated CRN environment files from
``build_decoupled_abm_environment.py``.  Matching is performed on sparse
candidate edges only; the script never constructs an all-orders by all-vehicles
dense cost matrix.

This is a 30-second discrete-time sparse-matching simulator with event-state
updates at each decision epoch.  It is not a full priority-queue event-driven
simulator.

For the request-plan-execution separated simulator, use
``stage4/scripts/run_simulator_v3.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


EARTH_M = 6_371_000.0
SERVICE_CARDINALITY_BONUS = 1_000_000_000


@dataclass
class Vehicle:
    vehicle_id: str
    vehicle_type: str
    lon: float
    lat: float
    zone: str
    online_start: pd.Timestamp
    online_end: pd.Timestamp
    profile: str
    driver_id: str | None = None
    session_id: str | None = None
    depot_id: str | None = None
    status: str = "OFFLINE"
    available_time: pd.Timestamp | None = None
    current_order_id: str | None = None
    reserved_order_id: str | None = None
    busy_time_sec: float = 0.0
    stationary_idle_time_sec: float = 0.0
    repositioning_time_sec: float = 0.0
    repositioning_km: float = 0.0
    pickup_km: float = 0.0
    service_km: float = 0.0
    income: float = 0.0
    stress_burden: float = 0.0
    served_orders: int = 0
    last_state_time: pd.Timestamp | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["Safe GlobalMatch-MinPickup", "ODD-Gated Price-Aware", "Three-Stakeholder Balanced"], required=True)
    parser.add_argument("--replication", type=int, default=1)
    parser.add_argument("--request-time-scenario", default="RT-Base", choices=["RT-Low", "RT-Base", "RT-High"])
    parser.add_argument("--operation-setting", default="O0", choices=["O0", "O1", "O2", "O3"])
    parser.add_argument("--environment-root", type=Path, default=Path("stage4/output/decoupled_environment"))
    parser.add_argument("--data-root", type=Path, default=Path("stage4/data/decoupled_abm"))
    parser.add_argument("--capability", type=Path, default=Path("stage4/output/decoupled_environment/capability_mapping/fold=3/vehicle_capability_mapping.parquet"))
    parser.add_argument("--radius-config", type=Path, default=Path("stage4/config/search_radius_schedule.json"))
    parser.add_argument("--output-root", type=Path, default=Path("stage4/output/decoupled_abm"))
    parser.add_argument("--results-dir", type=Path, default=Path("stage4/docs/results"))
    parser.add_argument("--profile", default="moderate_av")
    parser.add_argument("--matching-epoch-sec", type=int, default=30)
    parser.add_argument("--max-candidates-per-order", type=int, default=80)
    parser.add_argument("--passenger-gc-cap", type=float, default=120.0)
    parser.add_argument("--driver-utility-min", type=float, default=-2.0)
    parser.add_argument("--hv-stress-budget-zone-epoch", type=float, default=60.0)
    parser.add_argument("--min-zone-service-bonus", type=float, default=1.0)
    parser.add_argument("--preassignment-horizon-sec", type=float, default=300.0)
    parser.add_argument("--enable-preassignment", action="store_true", help="Disabled by default until two-layer reservation state is fully validated.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def haversine_m(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_M * np.arcsin(np.sqrt(a))


def slug(value: str) -> str:
    return value.replace(" ", "_").replace("-", "_").replace("+", "plus")


def radius_for_wait(wait_sec: float, failure_stage: int, schedule: dict) -> tuple[int, float]:
    minutes = wait_sec / 60.0
    time_stage = len(schedule["stages"]) - 1
    for idx, stage in enumerate(schedule["stages"]):
        if minutes >= float(stage["wait_min"]) and minutes < float(stage["wait_max"]):
            time_stage = idx
            break
    stage_idx = min(len(schedule["stages"]) - 1, max(time_stage, failure_stage))
    return stage_idx, float(schedule["stages"][stage_idx]["radius_m"])


def load_vehicles(path: Path) -> dict[str, Vehicle]:
    frame = pd.read_parquet(path)
    vehicles: dict[str, Vehicle] = {}
    for row in frame.itertuples(index=False):
        vid = str(row.vehicle_id)
        vehicles[vid] = Vehicle(
            vehicle_id=vid,
            vehicle_type=str(row.vehicle_type),
            lon=float(row.initial_lon),
            lat=float(row.initial_lat),
            zone=str(getattr(row, "initial_zone", "")),
            online_start=pd.to_datetime(row.online_start, utc=True),
            online_end=pd.to_datetime(row.online_end, utc=True),
            profile=str(getattr(row, "vehicle_profile", "moderate_av")),
            driver_id=None if pd.isna(getattr(row, "driver_id", pd.NA)) else str(getattr(row, "driver_id")),
            session_id=None if pd.isna(getattr(row, "session_id", pd.NA)) else str(getattr(row, "session_id")),
            depot_id=None if pd.isna(getattr(row, "depot_id", pd.NA)) else str(getattr(row, "depot_id")),
            available_time=pd.to_datetime(row.online_start, utc=True),
            last_state_time=pd.to_datetime(row.online_start, utc=True),
        )
    return vehicles


def update_vehicle_status(vehicles: dict[str, Vehicle], now: pd.Timestamp) -> None:
    for vehicle in vehicles.values():
        if now < vehicle.online_start or now > vehicle.online_end:
            vehicle.status = "OFFLINE"
            vehicle.current_order_id = None if now > vehicle.online_end else vehicle.current_order_id
            continue
        if vehicle.current_order_id and vehicle.available_time is not None and now >= vehicle.available_time:
            vehicle.current_order_id = None
            vehicle.reserved_order_id = None
            vehicle.status = "IDLE_IN_FIELD" if vehicle.vehicle_type == "AV" else "IDLE"
        elif vehicle.current_order_id:
            remaining = (vehicle.available_time - now).total_seconds() if vehicle.available_time is not None else 999999
            vehicle.status = "NEAR_FREE_UNRESERVED" if remaining <= 300 else "BUSY_UNRESERVED"
        elif vehicle.status in {"OFFLINE", "REPOSITIONING"}:
            if vehicle.available_time is not None and now >= vehicle.available_time:
                vehicle.status = "IDLE_AT_DEPOT" if vehicle.vehicle_type == "AV" and vehicle.depot_id and vehicle.served_orders == 0 else ("IDLE_IN_FIELD" if vehicle.vehicle_type == "AV" else "IDLE")


def load_orders(args: argparse.Namespace) -> pd.DataFrame:
    path = args.data_root / f"demand_20161023_{args.request_time_scenario}.parquet"
    orders = pd.read_parquet(path)
    orders["simulated_request_time"] = pd.to_datetime(orders["simulated_request_time"], utc=True, errors="coerce")
    orders["observed_boarding_time"] = pd.to_datetime(orders["observed_boarding_time"], utc=True, errors="coerce")
    orders["predicted_service_time_sec"] = pd.to_numeric(orders["predicted_service_time_sec"], errors="coerce").clip(lower=60, upper=7_200)
    orders["realized_service_time_sec"] = pd.to_numeric(orders["realized_service_time_sec"], errors="coerce").fillna(orders["predicted_service_time_sec"]).clip(lower=60, upper=7_200)
    orders["order_id"] = orders["order_id"].astype(str)
    orders = orders.sort_values("simulated_request_time", kind="mergesort").reset_index(drop=True)
    return orders


def stress_value(order: pd.Series) -> float:
    if not bool(order.get("condition_available", False)):
        return 0.0
    vals = [order.get("lcs_tail_probability"), order.get("pmis_tail_probability"), order.get("rts_tail_probability")]
    vals = [float(v) for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if vals else 0.0


def load_maps(args: argparse.Namespace):
    cap = pd.read_parquet(args.capability)
    cap = cap[(cap["vehicle_profile"].eq(args.profile)) & cap["vehicle_type"].eq("AV")]
    cap_map = {str(row.order_id): row._asdict() for row in cap.itertuples(index=False)}
    pickup_odd = pd.read_parquet(args.data_root / "pickup_odd_zone_pair_proxy.parquet")
    pickup_map = {(str(r.origin_zone), str(r.destination_zone)): bool(r.pickup_odd_feasible) for r in pickup_odd.itertuples(index=False)}
    speed = pd.read_parquet(args.data_root / "pickup_empty_speed_by_zone_time.parquet")
    speed_map = {(str(r.origin_zone), int(r.time_bin)): float(r.empty_speed_mps) for r in speed.itertuples(index=False)}
    global_speed = float(speed["empty_speed_mps"].median()) if len(speed) else 6.0
    circ_path = args.data_root / "pickup_circuity_by_zone.parquet"
    if circ_path.exists():
        circ = pd.read_parquet(circ_path)
        circ_map = {str(r.origin_zone): float(r.circuity_factor) for r in circ.itertuples(index=False)}
        global_circuity = float(circ["circuity_factor"].median()) if len(circ) else 1.35
    else:
        circ_map = {}
        global_circuity = 1.35
    return cap_map, pickup_map, speed_map, global_speed, circ_map, global_circuity


def pickup_eta(vehicle: Vehicle, order: pd.Series, speed_map: dict, global_speed: float, circ_map: dict, global_circuity: float) -> tuple[float, float, str]:
    dist = float(haversine_m(vehicle.lon, vehicle.lat, float(order["origin_lon"]), float(order["origin_lat"])))
    circuity = circ_map.get(str(vehicle.zone), global_circuity)
    circuity = min(max(float(circuity), 1.05), 2.50)
    speed = speed_map.get((str(vehicle.zone), int(order["time_bin"])), global_speed)
    speed = min(max(float(speed), 3.0), 15.0)
    road_dist = dist * circuity
    speed_source = "zone_time_empty_speed_prior" if (str(vehicle.zone), int(order["time_bin"])) in speed_map else "global_time_prior"
    circuity_source = "zone_circuity_prior" if str(vehicle.zone) in circ_map else "global_circuity_prior"
    return road_dist, road_dist / speed, f"{speed_source}+{circuity_source}"


def maybe_idle_move(vehicles: dict[str, Vehicle], now: pd.Timestamp, operation_setting: str, rng: np.random.Generator) -> dict:
    if operation_setting not in {"O1", "O3"}:
        return {"idle_movement_count": 0, "idle_movement_km": 0.0, "idle_movement_cost": 0.0}
    moved = 0
    km = 0.0
    for vehicle in vehicles.values():
        if vehicle.status not in {"IDLE", "IDLE_AT_DEPOT", "IDLE_IN_FIELD"} or now < vehicle.online_start or now > vehicle.online_end:
            continue
        # Lightweight joint idle-management package: move a small reproducible
        # share of long-idle vehicles to an adjacent grid cell.  This is a
        # scenario proxy, not recovered true cruising.
        if rng.random() > 0.015:
            continue
        dlon = float(rng.choice([-0.01, 0.0, 0.01]))
        dlat = float(rng.choice([-0.01, 0.0, 0.01]))
        if dlon == 0 and dlat == 0:
            continue
        new_lon = vehicle.lon + dlon
        new_lat = vehicle.lat + dlat
        dist = float(haversine_m(vehicle.lon, vehicle.lat, new_lon, new_lat))
        move_time = dist / 6.0
        if now + pd.Timedelta(seconds=move_time) > vehicle.online_end:
            continue
        vehicle.lon = new_lon
        vehicle.lat = new_lat
        vehicle.available_time = now + pd.Timedelta(seconds=move_time)
        vehicle.status = "REPOSITIONING"
        vehicle.repositioning_km += dist / 1000.0
        vehicle.repositioning_time_sec += move_time
        moved += 1
        km += dist / 1000.0
    return {"idle_movement_count": moved, "idle_movement_km": km, "idle_movement_cost": km * 0.3}


def edge_economics(order: pd.Series, vehicle: Vehicle, pickup_dist_m: float, pickup_time_sec: float, wait_sec: float, cap_row: dict | None, pickup_odd_feasible: bool, strategy: str, args: argparse.Namespace) -> dict:
    route_km = float(order.get("route_length_m", 0.0) or 0.0) / 1000.0
    service_min = float(order["predicted_service_time_sec"]) / 60.0
    stress = stress_value(order)
    base_fare = 8.0 + 2.0 * route_km + 0.4 * service_min
    stress_surcharge = 0.0 if not bool(order.get("stress_surcharge_allowed", False)) else min(8.0, 4.0 * stress)
    fare = base_fare + (stress_surcharge if strategy != "Safe GlobalMatch-MinPickup" else 0.0)
    passenger_gc = fare + wait_sec / 60.0 * 0.35 + pickup_time_sec / 60.0 * 0.25
    passenger_ok = passenger_gc <= args.passenger_gc_cap
    service_cost = 0.45 * route_km
    pickup_cost = 0.30 * pickup_dist_m / 1000.0
    service_odd_feasible = True
    capability_cost = 0.0
    remote_cost = 0.0
    if vehicle.vehicle_type == "AV":
        if cap_row is None:
            service_odd_feasible = False
        else:
            service_odd_feasible = bool(cap_row.get("service_feasible", False))
            capability_cost = float(cap_row.get("capability_cost", 0.0))
            remote_cost = 5.0 if bool(cap_row.get("feasible_with_extra_cost", False)) else 0.0
        driver_payout = 0.0
        driver_utility = np.nan
        driver_ok = True
        odd_ok = service_odd_feasible and pickup_odd_feasible
        platform_profit = fare - (pickup_cost + service_cost + capability_cost + remote_cost)
    else:
        odd_ok = True
        gross_comp = 0.0 if (strategy == "Safe GlobalMatch-MinPickup" or not bool(order.get("hv_stress_compensation_allowed", False))) else min(8.0, 5.0 * stress)
        passenger_funded = 0.5 * gross_comp
        platform_funded = gross_comp - passenger_funded
        fare += passenger_funded
        base_payout = 5.0 + 1.15 * route_km + 0.2 * service_min
        pickup_comp = 0.25 * pickup_dist_m / 1000.0
        driver_payout = base_payout + pickup_comp + gross_comp
        driver_cost = 0.25 * pickup_dist_m / 1000.0 + 0.22 * route_km
        stress_disutility = 2.5 * stress
        driver_utility = driver_payout - driver_cost - stress_disutility
        driver_ok = driver_utility >= args.driver_utility_min
        platform_cost = base_payout + pickup_comp + platform_funded + 0.1 * route_km
        platform_profit = fare - platform_cost
    feasible = passenger_ok and driver_ok and odd_ok
    return {
        "feasible": feasible,
        "passenger_acceptable": passenger_ok,
        "driver_acceptable": driver_ok,
        "pickup_odd_feasible": pickup_odd_feasible,
        "service_odd_feasible": service_odd_feasible,
        "combined_odd_feasible": odd_ok,
        "fare": fare,
        "driver_payout": driver_payout,
        "driver_utility": driver_utility,
        "platform_profit": platform_profit,
        "operating_contribution": platform_profit,
        "passenger_gc": passenger_gc,
        "stress": stress,
        "capability_cost": capability_cost,
        "remote_assistance_cost": remote_cost,
        "empty_movement_cost": 0.0,
    }


def build_edges(
    pending: pd.DataFrame,
    vehicles: dict[str, Vehicle],
    now: pd.Timestamp,
    cap_map: dict,
    pickup_odd_map: dict,
    speed_map: dict,
    global_speed: float,
    circ_map: dict,
    global_circuity: float,
    failure_stage: dict[str, int],
    args: argparse.Namespace,
    schedule: dict,
) -> tuple[pd.DataFrame, dict]:
    available = []
    preassignment_enabled = bool(args.enable_preassignment and args.operation_setting in {"O2", "O3"})
    for v in vehicles.values():
        if now < v.online_start or now > v.online_end:
            continue
        if v.status in {"IDLE", "IDLE_AT_DEPOT", "IDLE_IN_FIELD"} and v.available_time <= now:
            available.append(v)
        elif preassignment_enabled and v.status in {"BUSY_UNRESERVED", "NEAR_FREE_UNRESERVED"} and v.available_time is not None and (v.available_time - now).total_seconds() <= args.preassignment_horizon_sec and v.reserved_order_id is None:
            available.append(v)
    if pending.empty or not available:
        return pd.DataFrame(), {
            "candidate_edges": 0,
            "candidate_truncation_rate": 0.0,
            "orders_hitting_candidate_cap": 0,
            "peak_candidate_edge_count": 0,
            "radius_violation_count": 0,
            "order_edge_counts": {},
            "order_feasible_counts": {},
        }
    coords = np.radians(np.array([[v.lat, v.lon] for v in available], dtype=float))
    tree = BallTree(coords, metric="haversine")
    order_coords = np.radians(pending[["origin_lat", "origin_lon"]].to_numpy(float))
    rows = []
    hit_cap = 0
    possible_total = 0
    radius_violations = 0
    order_edge_counts: dict[str, int] = {}
    order_feasible_counts: dict[str, int] = {}
    for oi, (_, order) in enumerate(pending.iterrows()):
        oid = str(order["order_id"])
        order_edge_counts[oid] = 0
        order_feasible_counts[oid] = 0
        wait_sec = max(0.0, (now - order["simulated_request_time"]).total_seconds())
        stage, radius = radius_for_wait(wait_sec, failure_stage.get(oid, 0), schedule)
        idx, dist = tree.query_radius(order_coords[oi:oi + 1], r=radius / EARTH_M, return_distance=True, sort_results=True)
        candidates = idx[0]
        dists = dist[0] * EARTH_M
        possible_total += len(candidates)
        if len(candidates) > args.max_candidates_per_order:
            hit_cap += 1
        for ci, vi in enumerate(candidates[: args.max_candidates_per_order]):
            vehicle = available[int(vi)]
            pickup_dist = float(dists[ci])
            if pickup_dist > radius + 1e-6:
                radius_violations += 1
                continue
            road_dist, pickup_time, pickup_eta_source = pickup_eta(vehicle, order, speed_map, global_speed, circ_map, global_circuity)
            start_available = max(now, vehicle.available_time or now)
            deadline = order["simulated_request_time"] + pd.Timedelta(seconds=int(schedule.get("passenger_patience_seconds", 480)))
            safe_release_time = start_available
            preassigned = start_available > now
            if preassigned:
                safe_release_time = start_available + pd.Timedelta(seconds=60)
            if safe_release_time + pd.Timedelta(seconds=pickup_time) > deadline:
                continue
            if safe_release_time + pd.Timedelta(seconds=pickup_time + float(order["predicted_service_time_sec"])) > vehicle.online_end:
                continue
            cap = cap_map.get(oid) if vehicle.vehicle_type == "AV" else None
            pickup_odd = True
            pickup_proxy_source = "not_applicable_hv"
            if vehicle.vehicle_type == "AV":
                pickup_odd = pickup_odd_map.get((str(vehicle.zone), str(order["origin_zone"])), False)
                pickup_proxy_source = "pickup_odd_proxy_v1_known_pair" if (str(vehicle.zone), str(order["origin_zone"])) in pickup_odd_map else "pickup_odd_proxy_v1_unknown_pair"
                if not bool(order.get("condition_available", False)):
                    pickup_odd = False
            econ = edge_economics(order, vehicle, road_dist, pickup_time, wait_sec, cap, pickup_odd, args.strategy, args)
            if args.strategy == "Three-Stakeholder Balanced" and vehicle.vehicle_type == "HV":
                # Local stress budget proxy.  This constraint makes Balanced
                # non-identical to price-aware matching without using fixed
                # scenario costs as edge objectives.
                if econ["stress"] > 0.85:
                    econ["feasible"] = False
            order_edge_counts[oid] += 1
            if bool(econ["feasible"]):
                order_feasible_counts[oid] += 1
            row = {
                "order_id": oid,
                "vehicle_id": vehicle.vehicle_id,
                "vehicle_type": vehicle.vehicle_type,
                "origin_zone": str(order["origin_zone"]),
                "search_stage": stage,
                "search_radius_m": radius,
                "wait_sec": wait_sec,
                "pickup_dist_m": road_dist,
                "pickup_time_sec": pickup_time,
                "pickup_eta_source": pickup_eta_source,
                "preassigned": preassigned,
                "safe_release_time": safe_release_time,
                "pickup_odd_proxy_source": pickup_proxy_source,
            }
            row.update(econ)
            rows.append(row)
    edges = pd.DataFrame(rows)
    trunc = 1 - (min(possible_total, len(pending) * args.max_candidates_per_order) / possible_total) if possible_total else 0.0
    return edges, {
        "candidate_edges": int(len(edges)),
        "candidate_truncation_rate": float(max(0.0, trunc)),
        "orders_hitting_candidate_cap": int(hit_cap),
        "peak_candidate_edge_count": int(len(edges)),
        "radius_violation_count": int(radius_violations),
        "order_edge_counts": order_edge_counts,
        "order_feasible_counts": order_feasible_counts,
    }


def solve_sparse_matching(edges: pd.DataFrame, strategy: str, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    feasible = edges[edges["feasible"]].copy()
    if feasible.empty:
        return feasible, {"matching_solver": "sparse_networkx_max_weight_matching", "matched_edges": 0, "matching_runtime_sec": 0.0}
    start = time.perf_counter()
    graph = nx.Graph()
    for row in feasible.itertuples(index=False):
        order_node = f"o:{row.order_id}"
        vehicle_node = f"v:{row.vehicle_id}"
        if strategy == "Safe GlobalMatch-MinPickup":
            secondary = -float(row.pickup_time_sec)
        elif strategy == "ODD-Gated Price-Aware":
            secondary = float(row.operating_contribution)
        else:
            # Balanced keeps the same marginal contribution objective but
            # applies stricter feasibility constraints and a zone-service
            # bonus for underserved origins.
            secondary = float(row.operating_contribution) + args.min_zone_service_bonus
        weight = SERVICE_CARDINALITY_BONUS + int(round(secondary * 1_000))
        graph.add_edge(order_node, vehicle_node, weight=weight)
    matching = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=True, weight="weight")
    pairs = set()
    for a, b in matching:
        if a.startswith("o:"):
            pairs.add((a[2:], b[2:]))
        else:
            pairs.add((b[2:], a[2:]))
    chosen = feasible[feasible.apply(lambda r: (str(r["order_id"]), str(r["vehicle_id"])) in pairs, axis=1)].copy()
    return chosen, {
        "matching_solver": "sparse_networkx_max_weight_matching",
        "matched_edges": int(len(chosen)),
        "matching_runtime_sec": float(time.perf_counter() - start),
    }


def run_simulation(args: argparse.Namespace) -> dict:
    rep_root = args.environment_root / f"replication={args.replication}"
    out = args.output_root / f"replication={args.replication}" / slug(args.operation_setting) / slug(args.strategy) / args.request_time_scenario
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    orders = load_orders(args)
    fleet = load_vehicles(rep_root / "simulation_fleet.parquet")
    cap_map, pickup_odd_map, speed_map, global_speed, circ_map, global_circuity = load_maps(args)
    schedule = json.loads(args.radius_config.read_text(encoding="utf-8"))
    patience = int(schedule.get("passenger_patience_seconds", 480))
    rng_idle = np.random.default_rng(5000 + args.replication)
    start = orders["simulated_request_time"].min().floor(f"{args.matching_epoch_sec}s")
    end = max(orders["observed_boarding_time"].max(), orders["simulated_request_time"].max()).ceil(f"{args.matching_epoch_sec}s") + pd.Timedelta(seconds=patience)
    order_lookup = orders.set_index("order_id", drop=False)
    next_order = 0
    pending: set[str] = set()
    final_status = {oid: "unseen" for oid in orders["order_id"].astype(str)}
    failure_stage: dict[str, int] = {}
    dispatch_round: dict[str, int] = {}
    order_logs = []
    window_logs = []
    all_epoch_stats = []
    now = start
    lost_demand_penalty = 12.0
    av_fixed_daily_cost = 80.0
    depot_daily_cost = 50.0
    preassignment_failure_cost = 3.0
    while now <= end:
        update_vehicle_status(fleet, now)
        while next_order < len(orders) and orders.loc[next_order, "simulated_request_time"] <= now:
            oid = str(orders.loc[next_order, "order_id"])
            pending.add(oid)
            final_status[oid] = "pending"
            failure_stage[oid] = 0
            dispatch_round[oid] = 0
            next_order += 1
        expired = []
        for oid in list(pending):
            wait = (now - order_lookup.loc[oid, "simulated_request_time"]).total_seconds()
            if wait > patience:
                expired.append(oid)
        for oid in expired:
            pending.discard(oid)
            final_status[oid] = "cancelled_patience_timeout"
            order = order_lookup.loc[oid]
            order_logs.append({
                "order_id": oid,
                "final_status": "cancelled_patience_timeout",
                "cancellation_reason": "patience_timeout",
                "strategy": args.strategy,
                "operation_setting": args.operation_setting,
                "replication_id": args.replication,
                "request_time_scenario": args.request_time_scenario,
                "simulated_request_time": str(order["simulated_request_time"]),
                "observed_boarding_time": str(order["observed_boarding_time"]),
                "condition_available": bool(order.get("condition_available", False)),
            })
        idle_stats = maybe_idle_move(fleet, now, args.operation_setting, rng_idle)
        pending_frame = order_lookup.loc[list(pending)].copy() if pending else orders.iloc[0:0].copy()
        edges, edge_stats = build_edges(
            pending_frame,
            fleet,
            now,
            cap_map,
            pickup_odd_map,
            speed_map,
            global_speed,
            circ_map,
            global_circuity,
            failure_stage,
            args,
            schedule,
        )
        chosen, match_stats = solve_sparse_matching(edges, args.strategy, args) if not edges.empty else (edges, {"matching_solver": "sparse_networkx_max_weight_matching", "matched_edges": 0, "matching_runtime_sec": 0.0})
        matched_orders = 0
        for row in chosen.itertuples(index=False):
            oid = str(row.order_id)
            if oid not in pending:
                continue
            order = order_lookup.loc[oid]
            vehicle = fleet[str(row.vehicle_id)]
            start_available = max(now, vehicle.available_time or now)
            total_time = float(row.pickup_time_sec) + float(order["realized_service_time_sec"])
            vehicle.current_order_id = oid
            vehicle.reserved_order_id = oid if bool(row.preassigned) else None
            vehicle.status = "BUSY_PREASSIGNED" if bool(row.preassigned) else "BUSY_UNRESERVED"
            vehicle.available_time = start_available + pd.Timedelta(seconds=total_time)
            vehicle.lon = float(order["destination_lon"])
            vehicle.lat = float(order["destination_lat"])
            vehicle.zone = str(order["destination_zone"])
            vehicle.served_orders += 1
            vehicle.pickup_km += float(row.pickup_dist_m) / 1000.0
            vehicle.service_km += float(order.get("route_length_m", 0.0) or 0.0) / 1000.0
            vehicle.busy_time_sec += total_time
            vehicle.income += float(row.driver_payout) if vehicle.vehicle_type == "HV" else 0.0
            vehicle.stress_burden += float(row.stress)
            pending.remove(oid)
            final_status[oid] = "served"
            matched_orders += 1
            order_logs.append({
                "order_id": oid,
                "historical_driver_id": order.get("historical_driver_id"),
                "final_status": "served",
                "cancellation_reason": "",
                "assigned_vehicle": row.vehicle_id,
                "vehicle_type": row.vehicle_type,
                "strategy": args.strategy,
                "operation_setting": args.operation_setting,
                "replication_id": args.replication,
                "request_time_scenario": args.request_time_scenario,
                "simulated_request_time": str(order["simulated_request_time"]),
                "observed_boarding_time": str(order["observed_boarding_time"]),
                "condition_available": bool(order.get("condition_available", False)),
                "waiting_time_sec": float(row.wait_sec),
                "dispatch_round": int(dispatch_round.get(oid, 0)),
                "search_stage": int(row.search_stage),
                "search_radius_m": float(row.search_radius_m),
                "pickup_time_sec": float(row.pickup_time_sec),
                "predicted_service_time_sec": float(order["predicted_service_time_sec"]),
                "realized_service_time_sec": float(order["realized_service_time_sec"]),
                "service_time_error_sec": float(order["realized_service_time_sec"] - order["predicted_service_time_sec"]),
                "pickup_odd_feasible": bool(row.pickup_odd_feasible),
                "service_odd_feasible": bool(row.service_odd_feasible),
                "combined_odd_feasible": bool(row.combined_odd_feasible),
                "fare": float(row.fare),
                "driver_payout": float(row.driver_payout),
                "driver_utility": float(row.driver_utility) if pd.notna(row.driver_utility) else np.nan,
                "operating_contribution": float(row.operating_contribution),
                "passenger_generalized_cost": float(row.passenger_gc),
                "stress": float(row.stress),
                "preassigned": bool(row.preassigned),
                "safe_release_time": str(row.safe_release_time),
                "pickup_eta_source": row.pickup_eta_source,
                "pickup_odd_proxy_source": row.pickup_odd_proxy_source,
            })
        order_edge_counts = edge_stats.pop("order_edge_counts", {})
        order_feasible_counts = edge_stats.pop("order_feasible_counts", {})
        matched_set = set(chosen["order_id"].astype(str)) if len(chosen) else set()
        for oid in list(pending):
            dispatch_round[oid] = dispatch_round.get(oid, 0) + 1
            if oid not in matched_set and (order_edge_counts.get(oid, 0) == 0 or order_feasible_counts.get(oid, 0) == 0):
                failure_stage[oid] = min(failure_stage.get(oid, 0) + 1, len(schedule["stages"]) - 1)
        epoch_stat = {
            "window_time": str(now),
            "new_orders_seen": int(next_order),
            "pending_orders": int(len(pending)),
            "matched_orders": int(matched_orders),
            "online_HV": sum(v.vehicle_type == "HV" and v.status != "OFFLINE" for v in fleet.values()),
            "available_HV": sum(v.vehicle_type == "HV" and v.status == "IDLE" for v in fleet.values()),
            "available_AV": sum(v.vehicle_type == "AV" and v.status in {"IDLE_AT_DEPOT", "IDLE_IN_FIELD"} for v in fleet.values()),
            **edge_stats,
            **match_stats,
            **idle_stats,
        }
        all_epoch_stats.append(epoch_stat)
        window_logs.append(epoch_stat)
        now += pd.Timedelta(seconds=args.matching_epoch_sec)
    for oid in list(pending):
        order = order_lookup.loc[oid]
        final_status[oid] = "cancelled_end_of_day"
        order_logs.append({
            "order_id": oid,
            "final_status": "cancelled_end_of_day",
            "cancellation_reason": "end_of_day",
            "strategy": args.strategy,
            "operation_setting": args.operation_setting,
            "replication_id": args.replication,
            "request_time_scenario": args.request_time_scenario,
            "simulated_request_time": str(order["simulated_request_time"]),
            "observed_boarding_time": str(order["observed_boarding_time"]),
            "condition_available": bool(order.get("condition_available", False)),
        })
    for oid, status in final_status.items():
        if status == "unseen":
            order = order_lookup.loc[oid]
            order_logs.append({
                "order_id": oid,
                "final_status": "cancelled_end_of_day",
                "cancellation_reason": "unseen_end_of_day",
                "strategy": args.strategy,
                "operation_setting": args.operation_setting,
                "replication_id": args.replication,
                "request_time_scenario": args.request_time_scenario,
                "simulated_request_time": str(order["simulated_request_time"]),
                "observed_boarding_time": str(order["observed_boarding_time"]),
                "condition_available": bool(order.get("condition_available", False)),
            })
    order_log = pd.DataFrame(order_logs)
    window_log = pd.DataFrame(window_logs)
    vehicle_log = pd.DataFrame([{
        "vehicle_id": v.vehicle_id,
        "vehicle_type": v.vehicle_type,
        "driver_id": v.driver_id,
        "session_id": v.session_id,
        "depot_id": v.depot_id,
        "online_time_sec": max(0.0, (v.online_end - v.online_start).total_seconds()),
        "busy_time_sec": v.busy_time_sec,
        "stationary_idle_time_sec": max(0.0, (v.online_end - v.online_start).total_seconds() - v.busy_time_sec - v.repositioning_time_sec),
        "repositioning_time_sec": v.repositioning_time_sec,
        "repositioning_km": v.repositioning_km,
        "empty_km": v.repositioning_km,
        "pickup_km": v.pickup_km,
        "service_km": v.service_km,
        "income": v.income,
        "stress_burden": v.stress_burden,
        "served_orders": v.served_orders,
    } for v in fleet.values()])
    order_log.to_parquet(out / "order_log.parquet", index=False, compression="zstd")
    vehicle_log.to_parquet(out / "vehicle_log.parquet", index=False, compression="zstd")
    window_log.to_parquet(out / "window_log.parquet", index=False, compression="zstd")
    served = order_log["final_status"].eq("served")
    served_log = order_log[served].copy()
    lost_demand_cost = float((~served).sum() * lost_demand_penalty)
    av_fixed_cost = float(vehicle_log["vehicle_type"].eq("AV").sum() * av_fixed_daily_cost)
    depot_cost = float(vehicle_log.loc[vehicle_log["vehicle_type"].eq("AV"), "depot_id"].nunique() * depot_daily_cost)
    empty_cost = float(vehicle_log["repositioning_km"].sum() * 0.30)
    operating = float(pd.to_numeric(served_log.get("operating_contribution", pd.Series(dtype=float)), errors="coerce").sum())
    net_profit = operating - lost_demand_cost - av_fixed_cost - depot_cost - empty_cost
    summary = {
        "replication_id": args.replication,
        "strategy": args.strategy,
        "operation_setting": args.operation_setting,
        "request_time_scenario": args.request_time_scenario,
        "orders": int(len(orders)),
        "served_orders": int(served.sum()),
        "cancelled_orders": int((~served).sum()),
        "match_rate": float(served.mean()),
        "cancel_rate": float((~served).mean()),
        "condition_known_orders": int(order_log["condition_available"].fillna(False).sum()),
        "condition_unknown_orders": int((~order_log["condition_available"].fillna(False)).sum()),
        "unknown_condition_served": int(served_log["condition_available"].fillna(False).eq(False).sum()) if len(served_log) else 0,
        "mean_waiting_time_sec": float(pd.to_numeric(served_log.get("waiting_time_sec"), errors="coerce").mean()) if len(served_log) else np.nan,
        "mean_pickup_time_sec": float(pd.to_numeric(served_log.get("pickup_time_sec"), errors="coerce").mean()) if len(served_log) else np.nan,
        "mean_passenger_gc": float(pd.to_numeric(served_log.get("passenger_generalized_cost"), errors="coerce").mean()) if len(served_log) else np.nan,
        "av_assignment_share": float(served_log["vehicle_type"].eq("AV").mean()) if len(served_log) else 0.0,
        "av_odd_violation": int((served_log[served_log["vehicle_type"].eq("AV")]["combined_odd_feasible"] == False).sum()) if len(served_log) and "combined_odd_feasible" in served_log else 0,
        "hv_income": float(vehicle_log.loc[vehicle_log["vehicle_type"].eq("HV"), "income"].sum()),
        "hv_stress_burden": float(vehicle_log.loc[vehicle_log["vehicle_type"].eq("HV"), "stress_burden"].sum()),
        "av_vehicle_hour_share": float(vehicle_log.loc[vehicle_log["vehicle_type"].eq("AV"), "online_time_sec"].sum() / vehicle_log["online_time_sec"].sum()) if vehicle_log["online_time_sec"].sum() else 0.0,
        "empty_vehicle_km": float(vehicle_log["repositioning_km"].sum()),
        "pickup_vehicle_km": float(vehicle_log["pickup_km"].sum()),
        "operating_contribution": operating,
        "lost_demand_cost": lost_demand_cost,
        "av_fixed_cost": av_fixed_cost,
        "depot_cost": depot_cost,
        "empty_movement_cost": empty_cost,
        "preassignment_failure_cost": 0.0,
        "scenario_net_profit": net_profit,
        "simulator_type": "30_second_discrete_time_sparse_matching",
        "preassignment_enabled": bool(args.enable_preassignment and args.operation_setting in {"O2", "O3"}),
        "balanced_constraints_status": "hv_stress_edge_filter_proxy_not_full_zone_budget" if args.strategy == "Three-Stakeholder Balanced" else "not_applicable",
        "matching_solver": "sparse_networkx_max_weight_matching",
        "max_matching_runtime_sec": float(window_log["matching_runtime_sec"].max()) if len(window_log) else 0.0,
        "peak_candidate_edge_count": int(window_log["peak_candidate_edge_count"].max()) if len(window_log) else 0,
        "orders_hitting_candidate_cap": int(window_log["orders_hitting_candidate_cap"].sum()) if len(window_log) else 0,
        "candidate_truncation_rate_mean": float(window_log["candidate_truncation_rate"].mean()) if len(window_log) else 0.0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_simulation(args)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.results_dir / "decoupled_dispatch_summary.csv"
    new = pd.DataFrame([summary])
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        mask = ~(
            old["replication_id"].eq(args.replication)
            & old["strategy"].eq(args.strategy)
            & old["operation_setting"].eq(args.operation_setting)
            & old["request_time_scenario"].eq(args.request_time_scenario)
        )
        new = pd.concat([old[mask], new], ignore_index=True)
    new.to_csv(summary_path, index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
