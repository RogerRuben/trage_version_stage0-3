"""Summarize and audit demand-supply decoupled ABM outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("stage4/output/decoupled_abm"))
    parser.add_argument("--environment-root", type=Path, default=Path("stage4/output/decoupled_environment"))
    parser.add_argument("--results-dir", type=Path, default=Path("stage4/docs/results"))
    return parser.parse_args()


def collect_summaries(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.glob("replication=*/**/summary.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["summary_path"] = str(path)
        rows.append(doc)
    return pd.DataFrame(rows)


def read_order_logs(root: Path) -> pd.DataFrame:
    parts = []
    for path in root.glob("replication=*/**/order_log.parquet"):
        frame = pd.read_parquet(path)
        frame["log_path"] = str(path)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def read_window_logs(root: Path) -> pd.DataFrame:
    parts = []
    for path in root.glob("replication=*/**/window_log.parquet"):
        frame = pd.read_parquet(path)
        frame["log_path"] = str(path)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary = collect_summaries(args.output_root)
    if summary.empty:
        raise SystemExit("No decoupled ABM summary.json files found.")
    summary.to_csv(args.results_dir / "decoupled_dispatch_summary.csv", index=False)
    op = summary[summary["strategy"].eq("ODD-Gated Price-Aware") & summary["request_time_scenario"].eq("RT-Base")].copy()
    op.to_csv(args.results_dir / "operational_module_comparison.csv", index=False)
    rt = summary[summary["strategy"].eq("ODD-Gated Price-Aware") & summary["operation_setting"].eq("O3")].copy()
    rt.to_csv(args.results_dir / "request_time_sensitivity_summary.csv", index=False)
    idle = op[[
        "replication_id",
        "operation_setting",
        "match_rate",
        "mean_waiting_time_sec",
        "mean_pickup_time_sec",
        "av_assignment_share",
        "empty_vehicle_km",
        "scenario_net_profit",
    ]].copy()
    idle["idle_movement_interpretation"] = np.where(idle["operation_setting"].isin(["O1", "O3"]), "joint_hv_idle_proxy_plus_av_rebalancing_package", "stay")
    idle.to_csv(args.results_dir / "idle_movement_summary.csv", index=False)
    pre = op[[
        "replication_id",
        "operation_setting",
        "match_rate",
        "mean_waiting_time_sec",
        "mean_pickup_time_sec",
        "scenario_net_profit",
    ]].copy()
    pre["preassignment_enabled"] = pre["operation_setting"].isin(["O2", "O3"])
    pre["preassignment_implementation"] = "safe_release_proxy_on_busy_or_near_free_vehicle"
    pre.to_csv(args.results_dir / "preassignment_summary.csv", index=False)
    orders = read_order_logs(args.output_root)
    windows = read_window_logs(args.output_root)
    audit = {}
    if not orders.empty:
        grouped = orders.groupby(["replication_id", "strategy", "operation_setting", "request_time_scenario"])
        order_counts = grouped["order_id"].nunique()
        served_cancelled = grouped.size()
        duplicate_order_rows = int((grouped["order_id"].value_counts() > 1).sum())
        audit["demand_order_count_pass"] = status(bool(order_counts.eq(114356).all()))
        audit["served_plus_cancelled_pass"] = status(bool(served_cancelled.eq(114356).all()))
        audit["duplicate_order_rows"] = duplicate_order_rows
        audit["unknown_condition_total"] = int((~orders["condition_available"].fillna(False)).sum() / max(1, len(grouped)))
        av_served = orders[orders["final_status"].eq("served") & orders.get("vehicle_type", pd.Series("", index=orders.index)).eq("AV")]
        audit["gated_av_odd_violation_count"] = int((av_served.get("combined_odd_feasible", pd.Series(True, index=av_served.index)) == False).sum()) if len(av_served) else 0
        audit["gated_av_odd_pass"] = status(audit["gated_av_odd_violation_count"] == 0)
        unknown_av = orders[(~orders["condition_available"].fillna(False)) & orders.get("vehicle_type", pd.Series("", index=orders.index)).eq("AV")]
        audit["unknown_condition_av_assignment_count"] = int(len(unknown_av))
        audit["unknown_condition_av_assignment_pass"] = status(len(unknown_av) == 0)
        rt = pd.to_datetime(orders["simulated_request_time"], utc=True, errors="coerce")
        bt = pd.to_datetime(orders["observed_boarding_time"], utc=True, errors="coerce")
        audit["request_before_boarding_pass"] = status(bool((rt < bt).dropna().all()))
    if not windows.empty:
        audit["sparse_matching_solver_pass"] = status(windows.get("matching_solver", pd.Series("", index=windows.index)).astype(str).str.contains("sparse").all())
        audit["peak_candidate_edge_count"] = int(pd.to_numeric(windows.get("peak_candidate_edge_count", pd.Series(0)), errors="coerce").max())
        audit["max_matching_runtime_sec"] = float(pd.to_numeric(windows.get("matching_runtime_sec", pd.Series(0)), errors="coerce").max())
        audit["mean_candidate_truncation_rate"] = float(pd.to_numeric(windows.get("candidate_truncation_rate", pd.Series(0)), errors="coerce").mean())
    env_manifest = json.loads((args.environment_root / "manifest.json").read_text(encoding="utf-8"))
    audit["environment_manifest_status"] = env_manifest.get("status")
    audit["weekend_weight"] = env_manifest.get("weekend_weight")
    audit["condition_known_orders"] = env_manifest.get("condition_known_orders")
    audit["condition_unknown_orders"] = env_manifest.get("condition_unknown_orders")
    audit["crn_replications_available"] = len(env_manifest.get("replications", []))
    audit["crn_environment_files_pass"] = status(all((args.environment_root / f"replication={i}" / "simulation_fleet.parquet").exists() for i in range(1, 4)))
    audit["balanced_differentiation_pass"] = status("Three-Stakeholder Balanced" in set(summary["strategy"]) and "ODD-Gated Price-Aware" in set(summary["strategy"]))
    audit["overall_status"] = status(all(str(v) == "PASS" for k, v in audit.items() if k.endswith("_pass")))
    (args.results_dir / "decoupled_abm_audit_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    # Supply/demand endogeneity summary: keep it compact and explicit.
    supply = pd.read_csv(args.results_dir / "decoupled_hv_supply_summary.csv")
    demand = pd.read_parquet("stage4/data/decoupled_abm/demand_20161023_RT-Base.parquet")
    demand["time_bin"] = pd.to_numeric(demand["time_bin"], errors="coerce")
    demand_curve = demand.groupby("time_bin").size().rename("demand_orders").reset_index()
    corr = demand_curve.merge(supply[supply["replication_id"].eq(1)], on="time_bin", how="inner")
    pd.DataFrame([{
        "environment": "E1_demand_supply_decoupled",
        "observed_order_stream": "observed_served_order_arrival_stream_not_total_latent_demand",
        "time_bin_demand_supply_corr": float(corr["demand_orders"].corr(corr["generated_online_HV"])),
        "peak_supply_demand_ratio": float(corr["generated_online_HV"].max() / corr["demand_orders"].max()),
        "supply_source": "20161022_weekend_anchor_0.7_plus_20161019_21_weekday_median_0.3",
    }]).to_csv(args.results_dir / "supply_demand_endogeneity_diagnostics.csv", index=False)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
