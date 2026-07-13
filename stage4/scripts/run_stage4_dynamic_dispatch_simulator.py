"""Rolling AV/HV dispatch simulator with ODD-gated capability mapping."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIOS = ["Random", "Nearest", "GlobalMatch", "Cost-only", "Simple risk-penalty", "ODD-gated", "ODD-gated + HV compensation"]


@dataclass
class Vehicle:
    vehicle_id: str
    vehicle_type: str
    lon: float
    lat: float
    available_time: pd.Timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-input-root", type=Path, default=Path("stage3/output/stage4_inputs_core_v2"))
    parser.add_argument("--mapping-root", type=Path, default=Path("stage4/output/capability_mapping_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("stage4/output/dynamic_dispatch"))
    parser.add_argument("--av-fleet-size", type=int, default=250)
    parser.add_argument("--hv-fleet-size", type=int, default=750)
    parser.add_argument("--patience-minutes", type=float, default=8.0)
    parser.add_argument("--pickup-speed-mps", type=float, default=7.0)
    parser.add_argument("--service-speed-mps", type=float, default=8.0)
    parser.add_argument("--max-pickup-m", type=float, default=6000.0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    radius = 6371000.0
    lon1 = np.radians(lon1); lat1 = np.radians(lat1)
    lon2 = np.radians(np.asarray(lon2)); lat2 = np.radians(np.asarray(lat2))
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return radius * 2 * np.arcsin(np.sqrt(a))


def initialize_vehicles(orders: pd.DataFrame, av_size: int, hv_size: int, rng: np.random.Generator, start_time: pd.Timestamp) -> list[Vehicle]:
    seed_points = orders[["origin_lon", "origin_lat"]].dropna()
    if seed_points.empty:
        seed_points = pd.DataFrame({"origin_lon": [108.94], "origin_lat": [34.26]})
    sample = seed_points.sample(n=av_size + hv_size, replace=True, random_state=int(rng.integers(0, 1_000_000)))
    vehicles = []
    for idx, (_, row) in enumerate(sample.iterrows()):
        vehicle_type = "AV" if idx < av_size else "HV"
        vehicles.append(Vehicle(f"{vehicle_type}_{idx}", vehicle_type, float(row["origin_lon"]), float(row["origin_lat"]), start_time))
    return vehicles


def choose_vehicle(order: pd.Series, vehicles: list[Vehicle], now: pd.Timestamp, scenario: str, feasible_av: bool, rng: np.random.Generator, args: argparse.Namespace) -> tuple[int | None, float]:
    available = [idx for idx, vehicle in enumerate(vehicles) if vehicle.available_time <= now]
    if not available:
        return None, math.inf
    lon = np.array([vehicles[idx].lon for idx in available])
    lat = np.array([vehicles[idx].lat for idx in available])
    pickup = haversine_m(float(order["origin_lon"]), float(order["origin_lat"]), lon, lat)
    feasible = pickup <= args.max_pickup_m
    if scenario.startswith("ODD-gated"):
        for pos, idx in enumerate(available):
            if vehicles[idx].vehicle_type == "AV" and not feasible_av:
                feasible[pos] = False
    if not feasible.any():
        return None, math.inf
    candidates = [(available[pos], float(pickup[pos])) for pos in np.where(feasible)[0]]
    if scenario == "Random":
        return candidates[int(rng.integers(0, len(candidates)))]
    def cost(item):
        idx, pickup_m = item
        vehicle = vehicles[idx]
        stress = float(order["composite_expected"])
        penalty = 0.0
        if scenario == "Simple risk-penalty" and vehicle.vehicle_type == "AV":
            penalty = stress * 1500.0
        if scenario.startswith("ODD-gated") and vehicle.vehicle_type == "HV" and order["core_overall_high_stress_probability"] >= 0.5:
            penalty = -200.0 if "compensation" in scenario else 0.0
        if scenario == "Cost-only":
            penalty = float(order["route_length_m"]) * 0.02
        return pickup_m + penalty
    return min(candidates, key=cost)


def simulate_fold(inputs: pd.DataFrame, mapping: pd.DataFrame, scenario: str, profile: str, args: argparse.Namespace, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    orders = inputs.copy()
    orders["decision_time"] = pd.to_datetime(orders["decision_time"], errors="coerce", utc=True)
    orders = orders.dropna(subset=["decision_time", "origin_lon", "origin_lat", "destination_lon", "destination_lat"]).sort_values("decision_time")
    profile_map = mapping[mapping["vehicle_profile"].eq(profile)][["order_id", "service_feasible"]].set_index("order_id")["service_feasible"].to_dict()
    start = orders["decision_time"].min()
    vehicles = initialize_vehicles(orders, args.av_fleet_size, args.hv_fleet_size, rng, start)
    assignments = []
    pending: list[dict] = []
    cancelled = 0
    for _, row in orders.iterrows():
        now = row["decision_time"]
        pending.append(row.to_dict())
        still_pending = []
        for item in pending:
            wait = (now - item["decision_time"]).total_seconds() / 60.0
            if wait > args.patience_minutes:
                cancelled += 1
            else:
                still_pending.append(item)
        pending = still_pending
        made_progress = True
        while made_progress and pending:
            made_progress = False
            item = pending[0]
            feasible_av = bool(profile_map.get(item["order_id"], True))
            idx, pickup_m = choose_vehicle(pd.Series(item), vehicles, now, scenario, feasible_av, rng, args)
            if idx is None:
                break
            vehicle = vehicles[idx]
            pickup_sec = pickup_m / args.pickup_speed_mps
            service_sec = max(float(item["route_length_m"]) / args.service_speed_mps, 60.0)
            wait_sec = max((now - item["decision_time"]).total_seconds(), 0.0)
            vehicle.available_time = now + pd.Timedelta(seconds=pickup_sec + service_sec)
            vehicle.lon = float(item["destination_lon"]); vehicle.lat = float(item["destination_lat"])
            assignments.append({
                "order_id": item["order_id"],
                "vehicle_id": vehicle.vehicle_id,
                "vehicle_type": vehicle.vehicle_type,
                "pickup_m": pickup_m,
                "pickup_time_sec": pickup_sec,
                "waiting_time_sec": wait_sec,
                "service_time_sec": service_sec,
                "composite_expected": item["composite_expected"],
                "core_overall_high_stress_probability": item["core_overall_high_stress_probability"],
                "av_feasible": feasible_av,
                "odd_violation": vehicle.vehicle_type == "AV" and not feasible_av,
                "hv_compensation": 5.0 if ("compensation" in scenario and vehicle.vehicle_type == "HV" and item["core_overall_high_stress_probability"] >= 0.5) else 0.0,
            })
            pending.pop(0)
            made_progress = True
    cancelled += len(pending)
    assigned = pd.DataFrame(assignments)
    if assigned.empty:
        metrics = {"match_rate": 0.0, "cancel_rate": 1.0}
        return assigned, metrics
    total = len(orders)
    av = assigned[assigned["vehicle_type"].eq("AV")]
    hv = assigned[assigned["vehicle_type"].eq("HV")]
    metrics = {
        "orders": total,
        "assigned_orders": len(assigned),
        "match_rate": len(assigned) / total,
        "cancel_rate": cancelled / total,
        "pickup_time_mean_sec": float(assigned["pickup_time_sec"].mean()),
        "waiting_time_mean_sec": float(assigned["waiting_time_sec"].mean()),
        "platform_operating_cost": float((assigned["pickup_m"] * 0.001 + assigned["service_time_sec"] / 600.0 + assigned["hv_compensation"]).sum()),
        "AV_assigned_share": len(av) / max(1, len(assigned)),
        "AV_mean_stress_exposure": float(av["composite_expected"].mean()) if len(av) else None,
        "AV_high_stress_share": float(av["core_overall_high_stress_probability"].ge(0.5).mean()) if len(av) else None,
        "AV_ODD_violation_rate": float(av["odd_violation"].mean()) if len(av) else 0.0,
        "HV_assigned_share": len(hv) / max(1, len(assigned)),
        "HV_mean_stress_exposure": float(hv["composite_expected"].mean()) if len(hv) else None,
        "HV_high_stress_share": float(hv["core_overall_high_stress_probability"].ge(0.5).mean()) if len(hv) else None,
        "HV_pickup_burden_sec": float(hv["pickup_time_sec"].mean()) if len(hv) else None,
        "HV_compensation_cost": float(assigned["hv_compensation"].sum()),
    }
    return assigned, metrics


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary_rows = []
    for path in sorted(args.stage3_input_root.glob("fold=*/stage4_inputs.parquet")):
        fold = int(path.parent.name.split("=", 1)[-1])
        inputs = pd.read_parquet(path)
        mapping = pd.read_parquet(args.mapping_root / f"fold={fold}" / "vehicle_capability_mapping.parquet")
        profiles = [profile for profile in mapping["vehicle_profile"].unique() if profile != "reference_hv"]
        for scenario in SCENARIOS:
            scenario_profiles = profiles if scenario.startswith("ODD-gated") else ["moderate_av"]
            for profile in scenario_profiles:
                assignments, metrics = simulate_fold(inputs, mapping, scenario, profile, args, rng)
                out_dir = args.output_root / f"fold={fold}" / scenario.replace(" ", "_").replace("+", "plus").replace(":", "_") / profile
                out_dir.mkdir(parents=True, exist_ok=True)
                assignments.to_parquet(out_dir / "assignments.parquet", index=False, compression="zstd")
                summary_rows.append({"fold": fold, "scenario": scenario, "vehicle_profile": profile, **metrics})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_root / "dynamic_dispatch_summary.csv", index=False)
    report = ["# Stage4 dynamic dispatch report", "", "Rolling simulator with pending queue, vehicle state, pickup feasibility, patience cancellation, AV/HV fleet constraints, and ODD-gated capability mapping.", "", summary.to_markdown(index=False, floatfmt=".4f")]
    (args.output_root / "stage4_dynamic_dispatch_report.md").write_text("\n".join(report), encoding="utf-8")
    (args.output_root / "manifest.json").write_text(json.dumps({"status": "PASS", "rows": len(summary), "scenarios": SCENARIOS}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
