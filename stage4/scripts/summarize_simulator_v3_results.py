"""Summarize completed Simulator v3 full-day runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("stage4/output/simulator_v3"))
    parser.add_argument("--results-dir", type=Path, default=Path("stage4/docs/results/simulator_v3"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for summary_path in sorted(args.output_root.glob("replication=*/strategy=*/operation=*/RT-Base/summary.json")):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(data.get("orders", 0)) != 114356:
            continue
        run_dir = summary_path.parent
        req = pd.read_parquet(run_dir / "request_log.parquet")
        transitions = pd.read_parquet(run_dir / "request_transition_log.parquet") if (run_dir / "request_transition_log.parquet").exists() else pd.DataFrame()
        rows.append({
            "replication_id": data.get("replication_id"),
            "strategy": data.get("strategy"),
            "operation": data.get("operation"),
            "orders": data.get("orders"),
            "completed_orders": data.get("completed_orders"),
            "cancelled_orders": data.get("cancelled_orders"),
            "match_rate": data.get("match_rate"),
            "av_assignment_share": data.get("av_assignment_share"),
            "av_completed_orders": int((req["assigned_vehicle_id"].astype(str).str.startswith("AV_") & req["final_status"].eq("COMPLETED")).sum()),
            "unknown_condition_av_assignment_count": int((~req["condition_available"].fillna(False).astype(bool) & req["assigned_vehicle_id"].astype(str).str.startswith("AV_")).sum()),
            "reserved_transition_count": int(transitions["new_status"].astype(str).eq("RESERVED").sum()) if not transitions.empty else 0,
            "routing_query_count": data.get("routing_query_count"),
            "routing_cache_hit_rate": data.get("routing_cache_hit_rate"),
            "candidate_maximum": data.get("candidate_maximum"),
            "run_dir": str(run_dir),
        })
    out = pd.DataFrame(rows).sort_values(["strategy", "operation"], kind="mergesort")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.results_dir / "simulator_v3_full_day_matrix_summary.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
