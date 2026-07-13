"""Audit AV depot initialization and AV/HV ratio cap."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    out = Path("stage4/docs/results")
    out.mkdir(parents=True, exist_ok=True)
    depots = pd.read_parquet("stage4/data/av_depots.parquet")
    av = pd.read_parquet("stage4/data/av_agents_20161023.parquet")
    hv = pd.read_parquet("stage4/data/hv_agents_20161023.parquet")
    merged = av.merge(depots[["depot_id", "lon", "lat", "training_date_range", "depot_generation_method"]], on="depot_id", how="left", validate="many_to_one")
    same_location = (merged["initial_lon"].round(7).eq(merged["lon"].round(7)) & merged["initial_lat"].round(7).eq(merged["lat"].round(7))).all()
    ratio = len(av) / max(len(hv), 1)
    uses_test_day = depots["training_date_range"].astype(str).str.contains("20161023").any()
    audit = {
        "hv_agent_count": int(len(hv)),
        "av_count": int(len(av)),
        "av_ratio_to_hv": float(ratio),
        "av_ratio_cap_pass": bool(ratio <= 0.05 + 1e-12),
        "depot_count": int(len(depots)),
        "av_initial_position_equals_depot": bool(same_location),
        "uses_test_day_future_demand": bool(uses_test_day),
        "depot_methods": sorted(depots["depot_generation_method"].astype(str).unique().tolist()),
    }
    audit["status"] = "PASS" if audit["av_ratio_cap_pass"] and audit["av_initial_position_equals_depot"] and not audit["uses_test_day_future_demand"] else "FAIL"
    (out / "av_depot_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
