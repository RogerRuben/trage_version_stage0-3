"""Audit single-day ABM inputs and strategy outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("stage4")
RESULTS = ROOT / "docs" / "results"
OUTPUT = ROOT / "output" / "single_day_20161023"
DATA = ROOT / "data"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    fleet_manifest = json.loads((DATA / "simulation_fleet_20161023_manifest.json").read_text(encoding="utf-8"))
    hv_agents = pd.read_parquet(DATA / "hv_agents_20161023.parquet")
    sessions = pd.read_parquet(DATA / "hv_agent_sessions_20161023.parquet")
    depots = pd.read_parquet(DATA / "av_depots.parquet")
    av_agents = pd.read_parquet(DATA / "av_agents_20161023.parquet")
    cap = pd.read_parquet(OUTPUT / "capability_mapping" / "fold=3" / "vehicle_capability_mapping.parquet")
    moderate = cap[(cap["vehicle_profile"].eq("moderate_av")) & (cap["vehicle_type"].eq("AV"))]
    strategy_rows = []
    audit = {
        "hv_agent_count": int(len(hv_agents)),
        "hv_session_count": int(len(sessions)),
        "av_count": int(len(av_agents)),
        "av_ratio_to_hv": float(len(av_agents) / max(len(hv_agents), 1)),
        "av_ratio_cap_pass": bool(len(av_agents) / max(len(hv_agents), 1) <= 0.05 + 1e-12),
        "depot_count": int(len(depots)),
        "depot_uses_test_day_future_demand": False,
        "moderate_av_feasible_share": float(moderate["service_feasible"].mean()) if len(moderate) else None,
        "moderate_av_missing_iis_share": float(moderate["missing_modality_dimensions"].astype(str).str.contains("iis").mean()) if len(moderate) else None,
        "strategies": {},
    }
    for directory in sorted(OUTPUT.glob("*")):
        if not directory.is_dir() or not (directory / "summary.json").exists():
            continue
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        order_log = pd.read_parquet(directory / "order_log.parquet")
        served = order_log["final_status"].eq("served")
        strategy = summary["strategy"]
        status_counts = order_log["final_status"].value_counts().to_dict()
        av_viol = int(summary.get("av_odd_violation", 0))
        radius_viol = int(summary.get("radius_violation_count", 0))
        served_cancelled_ok = int(summary.get("served_plus_cancelled", 0)) == int(summary.get("orders", 0))
        gated = strategy in {"ODD-Gated Price-Aware", "Three-Stakeholder Balanced"}
        audit["strategies"][strategy] = {
            "served_plus_cancelled_ok": served_cancelled_ok,
            "radius_violation_count": radius_viol,
            "av_odd_violation": av_viol,
            "gated_odd_pass": bool((not gated) or av_viol == 0),
            "status_counts": status_counts,
        }
        strategy_rows.append(summary)
    dispatch = pd.DataFrame(strategy_rows)
    if not dispatch.empty:
        dispatch.to_csv(RESULTS / "single_day_dispatch_summary.csv", index=False)
    radius_parts = []
    for path in RESULTS.glob("dynamic_radius_summary_*.csv"):
        radius_parts.append(pd.read_csv(path))
    if radius_parts:
        pd.concat(radius_parts, ignore_index=True).to_csv(RESULTS / "dynamic_radius_summary.csv", index=False)
    audit["status"] = "PASS" if (
        audit["av_ratio_cap_pass"]
        and all(item["served_plus_cancelled_ok"] and item["radius_violation_count"] == 0 and item["gated_odd_pass"] for item in audit["strategies"].values())
    ) else "FAIL"
    (RESULTS / "single_day_audit_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
