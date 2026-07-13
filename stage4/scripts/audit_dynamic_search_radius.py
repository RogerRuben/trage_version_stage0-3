"""Audit dynamic radius behavior in single-day strategy logs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path("stage4/output/single_day_20161023")
    out = Path("stage4/docs/results")
    out.mkdir(parents=True, exist_ok=True)
    strategy_audits = {}
    for path in root.glob("*/order_log.parquet"):
        strategy = path.parent.name
        log = pd.read_parquet(path)
        if "search_radius_m" not in log.columns or "waiting_time_sec" not in log.columns:
            continue
        served = log[log["final_status"].eq("served")].copy()
        if served.empty:
            strategy_audits[strategy] = {"status": "WARN_EMPTY"}
            continue
        wait = pd.to_numeric(served["waiting_time_sec"], errors="coerce")
        radius = pd.to_numeric(served["search_radius_m"], errors="coerce")
        expected_ok = (
            ((wait < 120) & radius.eq(2000))
            | ((wait >= 120) & (wait < 240) & radius.eq(3000))
            | ((wait >= 240) & (wait < 360) & radius.eq(4500))
            | ((wait >= 360) & radius.eq(6000))
        )
        audit = {
            "served_rows": int(len(served)),
            "first_stage_rows": int(radius.eq(2000).sum()),
            "radius_stage_mismatch_count": int((~expected_ok).sum()),
            "over_patience_served_count": int(wait.gt(480).sum()),
            "max_radius_m": float(radius.max()),
        }
        audit["status"] = "PASS" if audit["radius_stage_mismatch_count"] == 0 and audit["over_patience_served_count"] == 0 and audit["max_radius_m"] <= 6000 else "FAIL"
        strategy_audits[strategy] = audit
    result = {"strategies": strategy_audits, "status": "PASS" if all(v.get("status") == "PASS" for v in strategy_audits.values()) else "FAIL"}
    (out / "dynamic_radius_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
