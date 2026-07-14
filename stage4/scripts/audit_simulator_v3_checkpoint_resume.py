"""Checkpoint/resume audit for Simulator v3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage4.scripts.run_simulator_v3 import build_engine, parse_args as parse_run_args
from stage4.simulator_v3.logging.request_logger import request_to_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("stage4/docs/results/simulator_v3"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("stage4/output/simulator_v3/checkpoint_resume_audit/checkpoint.pkl"))
    return parser.parse_args()


def make_run_args(extra: list[str]):
    old = sys.argv
    try:
        sys.argv = ["run_simulator_v3.py"] + extra
        return parse_run_args()
    finally:
        sys.argv = old


def request_frame(engine) -> pd.DataFrame:
    return pd.DataFrame([request_to_record(r) for r in engine.state.requests.values()]).sort_values("order_id").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    run_args = make_run_args([
        "--strategy", "Safe GlobalMatch-MinPickup",
        "--operation", "O0",
        "--request-time-scenario", "RT-Base",
        "--replication", "1",
        "--min-request-time", "2016-10-23T00:00:00Z",
        "--max-orders", "1000",
        "--max-vehicles", "2000",
        "--candidate-maximum", "80",
        "--overwrite",
    ])
    full_engine, end = build_engine(run_args)
    full_engine.run(end, finalize=True)
    full_req = request_frame(full_engine)

    resumed_engine, resumed_end = build_engine(run_args)
    midpoint = min(r.request_time for r in resumed_engine.state.requests.values()) + (
        max(r.request_time for r in resumed_engine.state.requests.values()) - min(r.request_time for r in resumed_engine.state.requests.values())
    ) / 2
    resumed_engine.run(midpoint, finalize=False)
    resumed_engine.save_checkpoint(args.checkpoint_path)
    loaded = resumed_engine.load_checkpoint(args.checkpoint_path)
    loaded.run(resumed_end, finalize=True)
    resumed_req = request_frame(loaded)

    compare_cols = ["order_id", "final_status", "assigned_vehicle_id", "dropoff_time"]
    merged = full_req[compare_cols].merge(resumed_req[compare_cols], on="order_id", suffixes=("_full", "_resumed"))
    mismatch = merged[
        (merged["final_status_full"] != merged["final_status_resumed"])
        | (merged["assigned_vehicle_id_full"].fillna("") != merged["assigned_vehicle_id_resumed"].fillna(""))
        | (merged["dropoff_time_full"].fillna("") != merged["dropoff_time_resumed"].fillna(""))
    ]
    full_distance = sum(v.cumulative_pickup_distance_m + v.cumulative_service_distance_m + v.cumulative_reposition_distance_m + v.cumulative_rebalancing_distance_m for v in full_engine.state.vehicles.values())
    resumed_distance = sum(v.cumulative_pickup_distance_m + v.cumulative_service_distance_m + v.cumulative_reposition_distance_m + v.cumulative_rebalancing_distance_m for v in loaded.state.vehicles.values())
    audit = {
        "completed_full": int(full_req["final_status"].eq("COMPLETED").sum()),
        "completed_resumed": int(resumed_req["final_status"].eq("COMPLETED").sum()),
        "cancelled_full": int(full_req["final_status"].eq("CANCELLED").sum()),
        "cancelled_resumed": int(resumed_req["final_status"].eq("CANCELLED").sum()),
        "request_assignment_dropoff_mismatch_count": int(len(mismatch)),
        "vehicle_total_distance_abs_diff_m": float(abs(full_distance - resumed_distance)),
    }
    audit["checkpoint_resume_pass"] = (
        "PASS"
        if audit["completed_full"] == audit["completed_resumed"]
        and audit["cancelled_full"] == audit["cancelled_resumed"]
        and audit["request_assignment_dropoff_mismatch_count"] == 0
        and audit["vehicle_total_distance_abs_diff_m"] <= 1e-6
        else "FAIL"
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "simulator_v3_checkpoint_resume_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

