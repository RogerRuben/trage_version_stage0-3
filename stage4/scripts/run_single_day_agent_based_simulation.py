"""Legacy prototype: single-day full-order AV/HV dynamic dispatch simulation.

For the request-plan-execution separated simulator, use
``stage4/scripts/run_simulator_v3.py``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import BallTree


EARTH_M = 6371000.0
PICKUP_SPEED_MPS = 8.0
SERVICE_SPEED_MPS = 8.0
SERVICE_BONUS = 1_000_000.0


@dataclass
class Vehicle:
    vehicle_id: str
    vehicle_type: str
    lon: float
    lat: float
    online_start: pd.Timestamp
    online_end: pd.Timestamp
    profile: str
    driver_id: str | None = None
    session_id: str | None = None
    depot_id: str | None = None
    status: str = "OFFLINE"
    available_time: pd.Timestamp | None = None
    current_order_id: str | None = None
    served_orders: int = 0
    pickup_distance: float = 0.0
    service_distance: float = 0.0
    income: float = 0.0
    stress_burden: float = 0.0
    busy_time_sec: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["GlobalMatch-MinPickup", "ODD-Gated Price-Aware", "Three-Stakeholder Balanced"], required=True)
    parser.add_argument("--orders", type=Path, default=Path("stage3/output/full_day_20161023/stage4_inputs/stage4_inputs.parquet"))
    parser.add_argument("--historical-orders", type=Path, default=Path("stage4/data/test_day_20161023_historical_orders.parquet"))
    parser.add_argument("--fleet", type=Path, default=Path("stage4/data/simulation_fleet_20161023.parquet"))
    parser.add_argument("--capability", type=Path, default=Path("stage4/output/single_day_20161023/capability_mapping/fold=3/vehicle_capability_mapping.parquet"))
    parser.add_argument("--radius-config", type=Path, default=Path("stage4/config/search_radius_schedule.json"))
    parser.add_argument("--output-root", type=Path, default=Path("stage4/output/single_day_20161023"))
    parser.add_argument("--results-dir", type=Path, default=Path("stage4/docs/results"))
    parser.add_argument("--dispatch-interval-sec", type=int, default=120)
    parser.add_argument("--max-candidates-per-order", type=int, default=30)
    parser.add_argument("--profile", default="moderate_av")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def haversine_m(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1); lat1 = np.radians(lat1); lon2 = np.radians(lon2); lat2 = np.radians(lat2)
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_M * np.arcsin(np.sqrt(a))


def load_orders(args: argparse.Namespace) -> pd.DataFrame:
    orders = pd.read_parquet(args.orders)
    hist = pd.read_parquet(args.historical_orders, columns=["order_id", "historical_driver_id"])
    orders = orders.merge(hist, on="order_id", how="left", validate="one_to_one")
    orders["decision_time"] = pd.to_datetime(orders["decision_time"], utc=True, errors="coerce")
    orders = orders[orders["decision_time"].notna()].copy()
    orders = orders.sort_values("decision_time", kind="mergesort").reset_index(drop=True)
    orders["service_time_sec"] = np.maximum(pd.to_numeric(orders["route_length_m"], errors="coerce").fillna(0) / SERVICE_SPEED_MPS, 180)
    return orders


def load_fleet(path: Path) -> dict[str, Vehicle]:
    frame = pd.read_parquet(path)
    vehicles = {}
    for index, row in frame.iterrows():
        vid = str(row["vehicle_id"])
        # HV vehicle_id can repeat across sessions.  Use a simulation-unit id
        # that preserves the real driver/session boundary and avoids future
        # position leakage across offline gaps.
        if row["vehicle_type"] == "HV":
            vid = f"{vid}|{row['session_id']}"
        vehicles[vid] = Vehicle(
            vehicle_id=vid,
            vehicle_type=str(row["vehicle_type"]),
            lon=float(row["initial_lon"]),
            lat=float(row["initial_lat"]),
            online_start=pd.to_datetime(row["online_start"], utc=True),
            online_end=pd.to_datetime(row["online_end"], utc=True),
            profile=str(row.get("vehicle_profile", "moderate_av")),
            driver_id=None if pd.isna(row.get("driver_id")) else str(row.get("driver_id")),
            session_id=None if pd.isna(row.get("session_id")) else str(row.get("session_id")),
            depot_id=None if pd.isna(row.get("depot_id")) else str(row.get("depot_id")),
            status="OFFLINE",
            available_time=pd.to_datetime(row["online_start"], utc=True),
        )
    return vehicles


def radius_for_wait(wait_sec: float, schedule: dict) -> tuple[int, float]:
    minutes = wait_sec / 60.0
    for idx, stage in enumerate(schedule["stages"]):
        if minutes >= float(stage["wait_min"]) and minutes < float(stage["wait_max"]):
            return idx, float(stage["radius_m"])
    return len(schedule["stages"]) - 1, float(schedule["stages"][-1]["radius_m"])


def update_vehicle_status(vehicles: dict[str, Vehicle], now: pd.Timestamp) -> None:
    for v in vehicles.values():
        if now < v.online_start or now > v.online_end:
            v.status = "OFFLINE"
            continue
        if v.current_order_id is not None and v.available_time is not None and now >= v.available_time:
            v.current_order_id = None
            v.status = "IDLE_IN_FIELD" if v.vehicle_type == "AV" else "IDLE"
        elif v.current_order_id is not None:
            v.status = "BUSY"
        elif v.status == "OFFLINE":
            v.status = "IDLE_AT_DEPOT" if v.vehicle_type == "AV" else "IDLE"


def fare_and_edge(order, vehicle: Vehicle, pickup_dist: float, pickup_time: float, wait_sec: float, cap_row: dict | None, strategy: str) -> dict:
    route_km = float(order["route_length_m"]) / 1000.0
    service_min = float(order["service_time_sec"]) / 60.0
    stress = float(np.nanmean([order["lcs_tail_probability"], order["pmis_tail_probability"], order["rts_tail_probability"]]))
    base_fare = 8.0 + 2.0 * route_km + 0.4 * service_min
    stress_surcharge = 0.0 if strategy == "GlobalMatch-MinPickup" else min(8.0, 4.0 * stress)
    fare = base_fare + stress_surcharge
    wait_cost = wait_sec / 60.0 * 0.35
    pickup_cost = pickup_time / 60.0 * 0.25
    passenger_gc = fare + wait_cost + pickup_cost
    passenger_ok = passenger_gc <= 120.0
    service_cost = 0.45 * route_km
    pickup_operating = 0.30 * pickup_dist / 1000.0
    capability_cost = 0.0
    odd_feasible = True
    remote_cost = 0.0
    fallback_cost = 0.0
    if vehicle.vehicle_type == "AV":
        if cap_row is not None:
            odd_feasible = bool(cap_row.get("service_feasible", True))
            capability_cost = float(cap_row.get("capability_cost", 0.0))
            if bool(cap_row.get("feasible_with_extra_cost", False)):
                remote_cost = 5.0
            if not odd_feasible:
                fallback_cost = 25.0
        driver_payout = 0.0
        driver_utility = np.nan
        av_cost = pickup_operating + service_cost + capability_cost + remote_cost + (fallback_cost if strategy == "GlobalMatch-MinPickup" else 0.0)
        platform_profit = fare - av_cost
        driver_ok = True
    else:
        gross_comp = 0.0 if strategy == "GlobalMatch-MinPickup" else min(8.0, 5.0 * stress)
        passenger_funded = 0.5 * gross_comp if strategy != "GlobalMatch-MinPickup" else 0.0
        platform_funded = gross_comp - passenger_funded
        fare += passenger_funded
        base_payout = 5.0 + 1.15 * route_km + 0.2 * service_min
        pickup_comp = 0.25 * pickup_dist / 1000.0
        driver_payout = base_payout + pickup_comp + gross_comp
        driver_cost = 0.25 * pickup_dist / 1000.0 + 0.22 * route_km
        stress_disutility = 2.5 * stress
        driver_utility = driver_payout - driver_cost - stress_disutility
        driver_ok = driver_utility >= -2.0
        platform_cost = base_payout + pickup_comp + platform_funded + 0.1 * route_km
        platform_profit = fare - platform_cost
    if strategy != "GlobalMatch-MinPickup" and vehicle.vehicle_type == "AV" and not odd_feasible:
        feasible = False
    else:
        feasible = passenger_ok and driver_ok
    return {
        "feasible": feasible,
        "odd_feasible": odd_feasible,
        "passenger_acceptable": passenger_ok,
        "driver_acceptable": driver_ok,
        "fare": fare,
        "driver_payout": driver_payout,
        "driver_utility": driver_utility,
        "platform_profit": platform_profit,
        "passenger_gc": passenger_gc,
        "pickup_time": pickup_time,
        "pickup_dist": pickup_dist,
        "stress": stress,
        "capability_cost": capability_cost,
        "remote_assistance_cost": remote_cost,
        "fallback_cost": fallback_cost,
    }


def build_edges(pending: pd.DataFrame, vehicles: dict[str, Vehicle], now: pd.Timestamp, caps: dict[str, dict], schedule: dict, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    idle = [v for v in vehicles.values() if v.status in {"IDLE", "IDLE_AT_DEPOT", "IDLE_IN_FIELD"} and v.available_time <= now and now <= v.online_end]
    if not idle or pending.empty:
        return pd.DataFrame(), {"idle_vehicle_count": len(idle), "candidate_edges": 0, "radius_violation_count": 0}
    coords = np.radians(np.array([[v.lat, v.lon] for v in idle], dtype=float))
    tree = BallTree(coords, metric="haversine")
    order_coords = np.radians(pending[["origin_lat", "origin_lon"]].to_numpy(float))
    rows = []
    radius_violation = 0
    for oi, (_, order) in enumerate(pending.iterrows()):
        wait_sec = max(0.0, (now - order["decision_time"]).total_seconds())
        stage, radius = radius_for_wait(wait_sec, schedule)
        idx, dist = tree.query_radius(order_coords[oi:oi+1], r=radius / EARTH_M, return_distance=True, sort_results=True)
        candidates = idx[0][: args.max_candidates_per_order]
        dists = dist[0][: args.max_candidates_per_order] * EARTH_M
        for ci, vi in enumerate(candidates):
            vehicle = idle[int(vi)]
            pickup_dist = float(dists[ci])
            if pickup_dist > radius + 1e-6:
                radius_violation += 1
                continue
            pickup_time = pickup_dist / PICKUP_SPEED_MPS
            if now + pd.Timedelta(seconds=pickup_time + float(order["service_time_sec"])) > vehicle.online_end:
                continue
            cap = caps.get(str(order["order_id"])) if vehicle.vehicle_type == "AV" else None
            edge = fare_and_edge(order, vehicle, pickup_dist, pickup_time, wait_sec, cap, args.strategy)
            edge.update({
                "order_id": str(order["order_id"]),
                "vehicle_id": vehicle.vehicle_id,
                "vehicle_type": vehicle.vehicle_type,
                "search_stage": stage,
                "search_radius_m": radius,
                "wait_sec": wait_sec,
            })
            rows.append(edge)
    return pd.DataFrame(rows), {"idle_vehicle_count": len(idle), "candidate_edges": len(rows), "radius_violation_count": radius_violation}


def solve_matching(edges: pd.DataFrame, strategy: str) -> pd.DataFrame:
    feasible = edges[edges["feasible"]].copy()
    if feasible.empty:
        return feasible
    orders = feasible["order_id"].drop_duplicates().tolist()
    vehicles = feasible["vehicle_id"].drop_duplicates().tolist()
    order_index = {value: i for i, value in enumerate(orders)}
    vehicle_index = {value: i for i, value in enumerate(vehicles)}
    n, m = len(orders), len(vehicles)
    size = max(n, m)
    matrix = np.full((size, size), SERVICE_BONUS / 2, dtype=float)
    if strategy == "GlobalMatch-MinPickup":
        for row in feasible.itertuples():
            matrix[order_index[row.order_id], vehicle_index[row.vehicle_id]] = row.pickup_dist
    else:
        for row in feasible.itertuples():
            matrix[order_index[row.order_id], vehicle_index[row.vehicle_id]] = -SERVICE_BONUS - float(row.platform_profit)
    r, c = linear_sum_assignment(matrix)
    selected = []
    pair_set = set()
    for ri, ci in zip(r, c):
        if ri < n and ci < m and matrix[ri, ci] < SERVICE_BONUS / 4:
            pair_set.add((orders[ri], vehicles[ci]))
    if not pair_set:
        return feasible.iloc[0:0].copy()
    chosen = feasible[feasible.apply(lambda row: (row["order_id"], row["vehicle_id"]) in pair_set, axis=1)].copy()
    return chosen.drop_duplicates(["order_id", "vehicle_id"])


def main() -> None:
    args = parse_args()
    out = args.output_root / args.strategy.replace(" ", "_").replace("-", "_")
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    orders = load_orders(args)
    vehicles = load_fleet(args.fleet)
    cap_frame = pd.read_parquet(args.capability)
    cap_frame = cap_frame[(cap_frame["vehicle_profile"].eq(args.profile)) & (cap_frame["vehicle_type"].eq("AV"))]
    caps = {str(row.order_id): row._asdict() for row in cap_frame.itertuples(index=False)}
    schedule = json.loads(args.radius_config.read_text(encoding="utf-8"))
    patience = int(schedule.get("passenger_patience_seconds", 480))
    start = orders["decision_time"].min().floor("120s")
    end = orders["decision_time"].max().ceil("120s") + pd.Timedelta(seconds=patience)
    pending_ids: set[str] = set()
    next_order = 0
    order_status = {str(order_id): "unseen" for order_id in orders["order_id"].astype(str)}
    order_logs = []
    window_logs = []
    radius_violations = 0
    now = start
    order_lookup = orders.set_index(orders["order_id"].astype(str), drop=False)
    while now <= end:
        update_vehicle_status(vehicles, now)
        while next_order < len(orders) and orders.loc[next_order, "decision_time"] <= now:
            oid = str(orders.loc[next_order, "order_id"])
            pending_ids.add(oid)
            order_status[oid] = "pending"
            next_order += 1
        pending = order_lookup.loc[list(pending_ids)].copy() if pending_ids else orders.iloc[0:0].copy()
        if not pending.empty:
            wait = (now - pending["decision_time"]).dt.total_seconds()
            expired = pending.index[wait.gt(patience)].tolist()
            for oid in expired:
                pending_ids.discard(str(oid))
                order_status[str(oid)] = "cancelled_patience_timeout"
                order_logs.append({"order_id": str(oid), "final_status": "cancelled_patience_timeout", "decision_time": str(order_lookup.loc[oid, "decision_time"]), "strategy": args.strategy})
            pending = pending.drop(index=expired, errors="ignore")
        edges, edge_stats = build_edges(pending, vehicles, now, caps, schedule, args)
        radius_violations += edge_stats["radius_violation_count"]
        chosen = solve_matching(edges, args.strategy) if not edges.empty else edges
        matched_orders = 0
        for row in chosen.itertuples():
            oid = str(row.order_id)
            if oid not in pending_ids:
                continue
            order = order_lookup.loc[oid]
            vehicle = vehicles[row.vehicle_id]
            service_time = float(order["service_time_sec"])
            total_time = float(row.pickup_time) + service_time
            vehicle.current_order_id = oid
            vehicle.status = "BUSY"
            vehicle.available_time = now + pd.Timedelta(seconds=total_time)
            vehicle.lon = float(order["destination_lon"])
            vehicle.lat = float(order["destination_lat"])
            vehicle.served_orders += 1
            vehicle.pickup_distance += float(row.pickup_dist)
            vehicle.service_distance += float(order["route_length_m"])
            vehicle.income += float(row.driver_payout) if vehicle.vehicle_type == "HV" else 0.0
            vehicle.stress_burden += float(row.stress)
            vehicle.busy_time_sec += total_time
            pending_ids.remove(oid)
            order_status[oid] = "served"
            matched_orders += 1
            order_logs.append({
                "order_id": oid,
                "historical_driver_id": order.get("historical_driver_id"),
                "decision_time": str(order["decision_time"]),
                "final_status": "served",
                "assigned_vehicle": row.vehicle_id,
                "vehicle_type": row.vehicle_type,
                "waiting_time_sec": float(row.wait_sec),
                "search_stage": int(row.search_stage),
                "search_radius_m": float(row.search_radius_m),
                "radius_expansion_count": int(row.search_stage),
                "pickup_time_sec": float(row.pickup_time),
                "service_time_sec": service_time,
                "odd_feasible": bool(row.odd_feasible),
                "fare": float(row.fare),
                "driver_payout": float(row.driver_payout),
                "driver_utility": float(row.driver_utility) if pd.notna(row.driver_utility) else np.nan,
                "platform_profit": float(row.platform_profit),
                "passenger_generalized_cost": float(row.passenger_gc),
                "cancellation_reason": "",
                "strategy": args.strategy,
            })
        window_logs.append({
            "window_time": str(now),
            "new_orders_seen": next_order,
            "pending_orders": len(pending_ids),
            "matched_orders": matched_orders,
            "candidate_edges": edge_stats["candidate_edges"],
            "online_HV": sum(v.vehicle_type == "HV" and v.status != "OFFLINE" for v in vehicles.values()),
            "available_HV": sum(v.vehicle_type == "HV" and v.status == "IDLE" for v in vehicles.values()),
            "available_AV": sum(v.vehicle_type == "AV" and v.status in {"IDLE_AT_DEPOT", "IDLE_IN_FIELD"} for v in vehicles.values()),
        })
        now += pd.Timedelta(seconds=args.dispatch_interval_sec)
    for oid in list(pending_ids):
        order_status[oid] = "cancelled_end_of_day"
        order_logs.append({"order_id": str(oid), "final_status": "cancelled_end_of_day", "decision_time": str(order_lookup.loc[oid, "decision_time"]), "strategy": args.strategy})
    never = [oid for oid, status in order_status.items() if status == "unseen"]
    for oid in never:
        order_logs.append({"order_id": oid, "final_status": "cancelled_end_of_day", "decision_time": str(order_lookup.loc[oid, "decision_time"]), "strategy": args.strategy})
    order_log = pd.DataFrame(order_logs)
    vehicle_log = pd.DataFrame([{
        "vehicle_id": v.vehicle_id, "vehicle_type": v.vehicle_type, "driver_id": v.driver_id, "session_id": v.session_id,
        "depot_id": v.depot_id, "served_orders": v.served_orders, "pickup_distance_m": v.pickup_distance,
        "service_distance_m": v.service_distance, "income": v.income, "stress_burden": v.stress_burden,
        "busy_time_sec": v.busy_time_sec,
        "online_time_sec": max(0.0, (v.online_end - v.online_start).total_seconds()),
    } for v in vehicles.values()])
    vehicle_log["utilization"] = vehicle_log["busy_time_sec"] / vehicle_log["online_time_sec"].replace(0, np.nan)
    window_log = pd.DataFrame(window_logs)
    order_log.to_parquet(out / "order_log.parquet", index=False, compression="zstd")
    vehicle_log.to_parquet(out / "vehicle_log.parquet", index=False, compression="zstd")
    window_log.to_parquet(out / "window_log.parquet", index=False, compression="zstd")
    served = order_log["final_status"].eq("served")
    summary = {
        "strategy": args.strategy,
        "orders": int(len(orders)),
        "served_orders": int(served.sum()),
        "match_rate": float(served.mean()),
        "cancel_rate": float(1 - served.mean()),
        "mean_waiting_time_sec": float(pd.to_numeric(order_log.loc[served, "waiting_time_sec"], errors="coerce").mean()),
        "mean_pickup_time_sec": float(pd.to_numeric(order_log.loc[served, "pickup_time_sec"], errors="coerce").mean()),
        "platform_profit": float(pd.to_numeric(order_log.loc[served, "platform_profit"], errors="coerce").sum()),
        "mean_passenger_gc": float(pd.to_numeric(order_log.loc[served, "passenger_generalized_cost"], errors="coerce").mean()),
        "hv_income": float(vehicle_log.loc[vehicle_log["vehicle_type"].eq("HV"), "income"].sum()),
        "hv_stress_burden": float(vehicle_log.loc[vehicle_log["vehicle_type"].eq("HV"), "stress_burden"].sum()),
        "av_assignment_share": float(order_log.loc[served, "vehicle_type"].eq("AV").mean()) if served.any() else 0.0,
        "av_odd_violation": int((order_log.loc[served & order_log["vehicle_type"].eq("AV"), "odd_feasible"] == False).sum()) if "odd_feasible" in order_log else 0,
        "radius_violation_count": int(radius_violations),
        "served_plus_cancelled": int(len(order_log)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_path = args.results_dir / "single_day_dispatch_summary.csv"
    current = pd.DataFrame([summary])
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        old = old[old["strategy"].ne(args.strategy)]
        current = pd.concat([old, current], ignore_index=True)
    current.to_csv(summary_path, index=False)
    radius = order_log.groupby(["strategy", "search_stage", "search_radius_m"], dropna=False).size().reset_index(name="orders")
    radius.to_csv(args.results_dir / f"dynamic_radius_summary_{args.strategy.replace(' ', '_').replace('-', '_')}.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
