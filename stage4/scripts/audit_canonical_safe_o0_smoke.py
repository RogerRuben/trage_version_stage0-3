"""Compute Stage 4 Safe/O0 smoke audits from emitted logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ALLOWED = {
    ("UNREVEALED", "PENDING"), ("PENDING", "ASSIGNED"), ("ASSIGNED", "PICKUP_STARTED"),
    ("PICKUP_STARTED", "BOARDED"), ("BOARDED", "IN_SERVICE"), ("IN_SERVICE", "COMPLETED"),
    ("PENDING", "CANCELLED"),
}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True); args = parser.parse_args()
    request = pd.read_parquet(args.run_root / "request_log.parquet")
    transitions = pd.read_parquet(args.run_root / "request_transition_log.parquet")
    legs = pd.read_parquet(args.run_root / "vehicle_leg_log.parquet")
    events = pd.read_parquet(args.run_root / "event_execution_log.parquet")
    ledger = pd.read_csv(args.run_root / "economy_ledger.csv").iloc[0]
    assignments = pd.read_parquet(args.run_root / "assignment_ledger.parquet")
    illegal = int(sum((old, new) not in ALLOWED for old, new in zip(transitions.old_status, transitions.new_status)))
    terminal_reopened = 0; chain_mismatch = 0; duplicate_completed = 0; duplicate_cancelled = 0
    for _, group in transitions.sort_values(["transition_time", "transition_sequence"]).groupby("order_id"):
        terminal = False
        previous_new = None
        for row in group.itertuples():
            if terminal: terminal_reopened += 1
            if previous_new is not None and row.old_status != previous_new: chain_mismatch += 1
            if row.new_status in {"COMPLETED", "CANCELLED"}: terminal = True
            previous_new = row.new_status
        duplicate_completed += max(0, int(group.new_status.eq("COMPLETED").sum()) - 1)
        duplicate_cancelled += max(0, int(group.new_status.eq("CANCELLED").sum()) - 1)
    overlap = 0; discontinuity = 0
    for _, group in legs.sort_values("start_time").groupby("vehicle_id"):
        previous = None
        for row in group.itertuples():
            if previous is not None:
                if row.start_time < previous.end_time - 1: overlap += 1
                if abs(row.start_lon - previous.end_lon) > 1e-6 or abs(row.start_lat - previous.end_lat) > 1e-6: discontinuity += 1
            previous = row
    av = assignments.vehicle_type.eq("AV")
    unknown_av = int((av & ~assignments.condition_available.astype(bool)).sum())
    pickup_violation = int((av & ~assignments.pickup_odd_feasible.astype(bool)).sum())
    service_violation = int((av & ~assignments.service_odd_feasible.astype(bool)).sum())
    combined_violation = int((av & ~assignments.combined_odd_feasible.astype(bool)).sum())
    recomputed_contribution = float((assignments.fare_revenue - assignments.driver_payout - assignments.pickup_cost - assignments.service_cost).sum())
    recomputed_net = recomputed_contribution - float(ledger.lost_demand_cost) - float(ledger.av_fixed_cost) - float(ledger.depot_cost)
    checks = {
        "demand_balance": int(request.final_status.isin(["COMPLETED", "CANCELLED"]).sum()) == len(request) == 1000,
        "illegal_request_transitions": illegal == 0,
        "terminal_state_reopened": terminal_reopened == 0,
        "request_transition_chain": chain_mismatch == 0,
        "duplicate_completed": duplicate_completed == 0,
        "duplicate_cancelled": duplicate_cancelled == 0,
        "vehicle_leg_overlap": overlap == 0,
        "vehicle_leg_spatial_discontinuity": discontinuity == 0,
        "event_time_monotonic": bool(events.event_time.is_monotonic_increasing),
        "unknown_condition_av_assignment": unknown_av == 0,
        "pickup_odd_violation": pickup_violation == 0,
        "service_odd_violation": service_violation == 0,
        "combined_odd_violation": combined_violation == 0,
        "historical_realized_duration_reads": int(request.historical_realized_duration_read.sum()) == 0,
        "operating_contribution_recomputes": abs(recomputed_contribution - float(ledger.operating_contribution)) < 1e-6,
        "scenario_net_profit_recomputes": abs(recomputed_net - float(ledger.scenario_net_profit)) < 1e-6,
        "profile_not_test_calibrated": not assignments.capability_profile.astype(str).str.contains("full_day_calibrated").any(),
    }
    audit = {
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "counts": {"orders": int(len(request)), "completed": int(request.final_status.eq("COMPLETED").sum()),
                   "cancelled": int(request.final_status.eq("CANCELLED").sum()), "av_assignments": int(av.sum()),
                   "illegal_transitions": illegal, "terminal_reopened": terminal_reopened,
                   "transition_chain_mismatches": chain_mismatch, "duplicate_completed": duplicate_completed,
                   "duplicate_cancelled": duplicate_cancelled,
                   "leg_overlaps": overlap, "spatial_discontinuities": discontinuity,
                   "unknown_condition_av_assignments": unknown_av, "pickup_odd_violations": pickup_violation,
                   "service_odd_violations": service_violation, "combined_odd_violations": combined_violation},
        "economy": {"operating_contribution": float(ledger.operating_contribution),
                    "scenario_net_profit": float(ledger.scenario_net_profit),
                    "recalculation_error": float(recomputed_net - float(ledger.scenario_net_profit))},
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()
