"""Audit full-day HV agent/session reconstruction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    out = Path("stage4/docs/results")
    out.mkdir(parents=True, exist_ok=True)
    agents = pd.read_parquet("stage4/data/hv_agents_20161023.parquet")
    sessions = pd.read_parquet("stage4/data/hv_agent_sessions_20161023.parquet")
    sessions["online_start"] = pd.to_datetime(sessions["online_start"], utc=True)
    sessions["online_end"] = pd.to_datetime(sessions["online_end"], utc=True)
    overlap = 0
    for _, group in sessions.sort_values(["driver_id", "online_start"]).groupby("driver_id"):
        overlap += int(group["online_start"].lt(group["online_end"].shift()).fillna(False).sum())
    audit = {
        "driver_id_unique_in_agents": bool(agents["driver_id"].is_unique),
        "session_online_start_before_end": bool((sessions["online_start"] < sessions["online_end"]).all()),
        "session_overlap_count": int(overlap),
        "initial_location_missing_count": int(sessions[["initial_lon", "initial_lat"]].isna().any(axis=1).sum()),
        "agent_source_values": sorted(agents["agent_source"].astype(str).unique().tolist()),
    }
    audit["status"] = "PASS" if audit["driver_id_unique_in_agents"] and audit["session_online_start_before_end"] and overlap == 0 and audit["initial_location_missing_count"] == 0 else "FAIL"
    (out / "hv_agent_session_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
