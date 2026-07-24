"""Run the only Stage 4 execution allowed during canonical rebaseline."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--decision-epoch-sec", type=int, default=30)
    parser.add_argument("--patience-sec", type=int, default=480)
    parser.add_argument("--max-candidates", type=int, default=20)
    return parser.parse_args()


def epoch_seconds(series: pd.Series) -> np.ndarray:
    return (pd.to_datetime(series, utc=True).astype("int64") // 1_000_000_000).to_numpy(dtype=np.int64)


def xy(lon, lat, latitude=34.25):
    return np.column_stack([np.asarray(lon) * 111_000 * math.cos(math.radians(latitude)), np.asarray(lat) * 111_000])


def radius(wait: float) -> tuple[int, float]:
    if wait < 120: return 0, 2000.0
    if wait < 240: return 1, 3000.0
    if wait < 360: return 2, 4500.0
    return 3, 6000.0


def main() -> None:
    a = arguments(); profile = json.loads(a.profile.read_text(encoding="utf-8"))
    demand = pd.read_parquet(a.environment_root / "demand.parquet")
    fleet = pd.read_parquet(a.environment_root / "fleet.parquet")
    residuals = pd.read_parquet(a.environment_root / "service_residuals.parquet").set_index("order_id")
    demand = demand.sort_values(["request_time", "order_id"], kind="mergesort").reset_index(drop=True)
    request_seconds = epoch_seconds(demand.request_time)
    vehicle_ids = fleet.vehicle_id.astype(str).tolist(); vehicle_index = {value: index for index, value in enumerate(vehicle_ids)}
    vehicle_type = fleet.vehicle_type.astype(str).to_numpy()
    current_lon = fleet.initial_lon.to_numpy(dtype=float).copy(); current_lat = fleet.initial_lat.to_numpy(dtype=float).copy()
    idle = set(range(len(fleet))); busy: list[tuple[float, int, int]] = []
    pending: set[int] = set(); next_request = 0; current = int(request_seconds.min()); end_requests = int(request_seconds.max())
    final_status = np.full(len(demand), "UNREVEALED", dtype=object)
    assigned_vehicle = np.full(len(demand), "", dtype=object); assigned_type = np.full(len(demand), "", dtype=object)
    dispatch_time = np.full(len(demand), np.nan); pickup_eta = np.full(len(demand), np.nan)
    completion_time = np.full(len(demand), np.nan); search_stage = np.full(len(demand), -1, dtype=int)
    odd_pickup = np.zeros(len(demand), dtype=bool); odd_service = np.zeros(len(demand), dtype=bool)
    transitions = []; legs = []; events = []; epoch_logs = []; assignment_rows = []
    event_sequence = 0; peak_edges = 0; candidate_truncated = 0; total_candidate_queries = 0

    def transition(index: int, old: str, new: str, timestamp: float, trigger: str, vehicle: str = ""):
        nonlocal event_sequence
        transition_sequence = event_sequence + 1
        transitions.append({"transition_sequence": transition_sequence,
                            "order_id": demand.at[index, "order_id"], "old_status": old, "new_status": new,
                            "transition_time": timestamp, "trigger": trigger, "vehicle_id": vehicle})
        event_sequence += 1
        events.append({"event_sequence": event_sequence, "event_time": timestamp, "event_priority": 1,
                       "event_type": trigger, "entity_id": demand.at[index, "order_id"], "handled": True})

    while current <= end_requests + a.patience_sec or pending or busy:
        while busy and busy[0][0] <= current:
            finished, vehicle, order = heapq.heappop(busy)
            current_lon[vehicle] = float(demand.at[order, "destination_lon"])
            current_lat[vehicle] = float(demand.at[order, "destination_lat"])
            idle.add(vehicle); completion_time[order] = finished; final_status[order] = "COMPLETED"
            transition(order, "IN_SERVICE", "COMPLETED", finished, "SERVICE_COMPLETED", vehicle_ids[vehicle])
        newly = 0
        while next_request < len(demand) and request_seconds[next_request] <= current:
            pending.add(next_request); final_status[next_request] = "PENDING"
            transition(next_request, "UNREVEALED", "PENDING", request_seconds[next_request], "REQUEST_REVEALED")
            next_request += 1; newly += 1
        cancelled_now = []
        for order in pending:
            if current - request_seconds[order] > a.patience_sec:
                cancelled_now.append(order)
        for order in cancelled_now:
            pending.remove(order); final_status[order] = "CANCELLED"
            transition(order, "PENDING", "CANCELLED", current, "PATIENCE_TIMEOUT")

        matched_now = 0; edge_count = 0; solver_started = time.perf_counter()
        if pending and idle:
            idle_list = sorted(idle); locations = xy(current_lon[idle_list], current_lat[idle_list]); tree = cKDTree(locations)
            graph = nx.Graph()
            for order in sorted(pending):
                wait = current - request_seconds[order]; stage, maximum = radius(wait); search_stage[order] = stage
                query_k = min(a.max_candidates + 1, len(idle_list))
                distances, local_indexes = tree.query(
                    xy([demand.at[order, "origin_lon"]], [demand.at[order, "origin_lat"]])[0],
                    k=query_k, distance_upper_bound=maximum,
                )
                distances = np.atleast_1d(distances); local_indexes = np.atleast_1d(local_indexes)
                valid_pairs = [(float(distance), idle_list[int(local)]) for distance, local in zip(distances, local_indexes)
                               if np.isfinite(distance) and int(local) < len(idle_list)]
                total_candidate_queries += 1
                if len(valid_pairs) > a.max_candidates:
                    candidate_truncated += 1; valid_pairs = valid_pairs[:a.max_candidates]
                for distance, vehicle in valid_pairs:
                    pickup_feasible = distance <= float(profile["pickup_distance_max_m"])
                    service_feasible = bool(demand.at[order, "condition_available"])
                    service_feasible &= bool(demand.at[order, "route_direction_valid"])
                    for target in ("lcs", "pmis", "rts"):
                        service_feasible &= float(demand.at[order, f"{target}_expected"]) <= float(profile[f"{target}_expected_max"])
                        service_feasible &= float(demand.at[order, f"{target}_uncertainty"]) <= float(profile["uncertainty_max"])
                    if vehicle_type[vehicle] == "AV" and not (pickup_feasible and service_feasible):
                        continue
                    eta = distance * 1.3 / 8.0
                    graph.add_edge(f"o:{order}", f"v:{vehicle}", weight=1_000_000.0 - eta,
                                   pickup_distance=distance, pickup_eta=eta,
                                   pickup_odd=pickup_feasible, service_odd=service_feasible)
                    edge_count += 1
            peak_edges = max(peak_edges, edge_count)
            matching = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=True, weight="weight")
            for left, right in matching:
                if left.startswith("v:"): left, right = right, left
                if not left.startswith("o:") or not right.startswith("v:"): continue
                order = int(left.split(":", 1)[1]); vehicle = int(right.split(":", 1)[1])
                data = graph.get_edge_data(left, right); eta = float(data["pickup_eta"]); distance = float(data["pickup_distance"])
                pending.remove(order); idle.remove(vehicle); assigned_vehicle[order] = vehicle_ids[vehicle]
                assigned_type[order] = vehicle_type[vehicle]; dispatch_time[order] = current; pickup_eta[order] = eta
                predicted = float(demand.at[order, "predicted_service_time_sec"])
                residual = float(residuals.at[demand.at[order, "order_id"], "service_time_residual_sec"])
                realized = max(30.0, predicted + residual); pickup_end = current + eta; complete = pickup_end + realized
                heapq.heappush(busy, (complete, vehicle, order)); final_status[order] = "IN_SERVICE"
                pickup_ok = bool(data["pickup_odd"]); service_ok = bool(data["service_odd"])
                odd_pickup[order] = pickup_ok; odd_service[order] = service_ok
                transition(order, "PENDING", "ASSIGNED", current, "GLOBAL_MATCH", vehicle_ids[vehicle])
                transition(order, "ASSIGNED", "PICKUP_STARTED", current, "PICKUP_STARTED", vehicle_ids[vehicle])
                transition(order, "PICKUP_STARTED", "BOARDED", pickup_end, "PASSENGER_BOARDED", vehicle_ids[vehicle])
                transition(order, "BOARDED", "IN_SERVICE", pickup_end, "SERVICE_STARTED", vehicle_ids[vehicle])
                legs.extend([
                    {"vehicle_id": vehicle_ids[vehicle], "order_id": demand.at[order, "order_id"], "leg_type": "PICKUP",
                     "start_time": current, "end_time": pickup_end, "start_lon": current_lon[vehicle], "start_lat": current_lat[vehicle],
                     "end_lon": demand.at[order, "origin_lon"], "end_lat": demand.at[order, "origin_lat"],
                     "distance_m": distance, "expected_time_sec": eta, "realized_time_sec": eta},
                    {"vehicle_id": vehicle_ids[vehicle], "order_id": demand.at[order, "order_id"], "leg_type": "SERVICE",
                     "start_time": pickup_end, "end_time": complete, "start_lon": demand.at[order, "origin_lon"],
                     "start_lat": demand.at[order, "origin_lat"], "end_lon": demand.at[order, "destination_lon"],
                     "end_lat": demand.at[order, "destination_lat"], "distance_m": demand.at[order, "route_length_m"],
                     "expected_time_sec": predicted, "realized_time_sec": realized},
                ])
                fare = 10.0 + 2.0 * float(demand.at[order, "route_length_m"]) / 1000.0
                driver_payout = fare * 0.70 if vehicle_type[vehicle] == "HV" else 0.0
                pickup_cost = distance / 1000.0 * 0.6; service_cost = float(demand.at[order, "route_length_m"]) / 1000.0 * 0.5
                assignment_rows.append({
                    "order_id": demand.at[order, "order_id"], "vehicle_id": vehicle_ids[vehicle], "vehicle_type": vehicle_type[vehicle],
                    "pickup_odd_feasible": pickup_ok, "service_odd_feasible": service_ok,
                    "combined_odd_feasible": pickup_ok and service_ok, "condition_available": demand.at[order, "condition_available"],
                    "capability_profile": profile["profile_id"], "capability_mapping_version": "canonical_smoke_direct_v1",
                    "fare_revenue": fare, "driver_payout": driver_payout, "pickup_cost": pickup_cost,
                    "service_cost": service_cost, "marginal_operating_contribution": fare-driver_payout-pickup_cost-service_cost,
                })
                matched_now += 1
        runtime = time.perf_counter() - solver_started
        epoch_logs.append({"epoch_time": current, "new_orders": newly, "pending_orders": len(pending),
                           "matched_orders": matched_now, "cancelled_orders": len(cancelled_now),
                           "idle_vehicles": len(idle), "busy_vehicles": len(busy), "candidate_edges": edge_count,
                           "matching_runtime_sec": runtime})
        current += a.decision_epoch_sec

    request_log = demand[["order_id", "condition_available", "request_time"]].copy()
    request_log["final_status"] = final_status; request_log["assigned_vehicle"] = assigned_vehicle
    request_log["vehicle_type"] = assigned_type; request_log["dispatch_time"] = dispatch_time
    request_log["pickup_time_sec"] = pickup_eta; request_log["completion_time"] = completion_time
    request_log["waiting_time_sec"] = dispatch_time - request_seconds + pickup_eta
    request_log["search_stage"] = search_stage; request_log["pickup_odd_feasible"] = odd_pickup
    request_log["service_odd_feasible"] = odd_service; request_log["combined_odd_feasible"] = odd_pickup & odd_service
    request_log["capability_profile"] = profile["profile_id"]
    request_log["capability_mapping_version"] = "canonical_smoke_direct_v1"
    request_log["historical_realized_duration_read"] = False
    a.output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "request_log": request_log,
        "request_transition_log": pd.DataFrame(transitions).sort_values(["transition_time", "transition_sequence"]),
        "vehicle_leg_log": pd.DataFrame(legs).sort_values(["vehicle_id", "start_time"]),
        "event_execution_log": pd.DataFrame(events).sort_values(["event_time", "event_sequence"]),
        "system_epoch_log": pd.DataFrame(epoch_logs),
        "assignment_ledger": pd.DataFrame(assignment_rows),
    }
    files = []
    for role, frame in outputs.items():
        path = a.output_root / f"{role}.parquet"; frame.to_parquet(path, index=False, compression="zstd")
        files.append({"role": role, "path": path.as_posix()})
    ledger = outputs["assignment_ledger"]
    completed = int(request_log.final_status.eq("COMPLETED").sum()); cancelled = int(request_log.final_status.eq("CANCELLED").sum())
    economy = pd.DataFrame([{
        "fare_revenue": float(ledger.fare_revenue.sum()), "driver_payout": float(ledger.driver_payout.sum()),
        "pickup_cost": float(ledger.pickup_cost.sum()), "service_cost": float(ledger.service_cost.sum()),
        "operating_contribution": float(ledger.marginal_operating_contribution.sum()),
        "lost_demand_cost": float(cancelled * 5.0), "av_fixed_cost": float((fleet.vehicle_type == "AV").sum() * 50.0),
        "depot_cost": 500.0, "scenario_net_profit": float(ledger.marginal_operating_contribution.sum() - cancelled*5.0 - (fleet.vehicle_type == "AV").sum()*50.0 - 500.0),
    }])
    economy_path = a.output_root / "economy_ledger.csv"; economy.to_csv(economy_path, index=False)
    files.append({"role": "economy_ledger", "path": economy_path.as_posix()})
    summary = {
        "status": "COMPLETED", "strategy": "Safe GlobalMatch-MinPickup", "operation": "O0",
        "preassignment": False, "idle_movement": "Stay", "replication": 1,
        "demand_orders": int(len(demand)), "completed_orders": completed, "cancelled_orders": cancelled,
        "match_rate": completed / len(demand), "av_assignments": int(request_log.vehicle_type.eq("AV").sum()),
        "av_assignment_share": float(request_log.vehicle_type.eq("AV").sum() / max(completed, 1)),
        "av_odd_violations": int(((request_log.vehicle_type == "AV") & ~request_log.combined_odd_feasible).sum()),
        "realized_duration_reads": int(request_log.historical_realized_duration_read.sum()),
        "candidate_truncation_rate": candidate_truncated / max(total_candidate_queries, 1),
        "peak_candidate_edge_count": peak_edges,
        "mean_matching_runtime_sec": float(outputs["system_epoch_log"].matching_runtime_sec.mean()),
        "profile_id": profile["profile_id"], "profile_threshold_source": profile["threshold_source"],
        "files": files,
    }
    summary_path = a.output_root / "summary.json"; summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
