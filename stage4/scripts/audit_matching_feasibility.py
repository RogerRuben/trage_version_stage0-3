"""Audit Stage4 dynamic matching feasibility constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ODD_STRATEGIES = {"ODD Gate Only", "ODD-Gated Price-Aware Matching", "Three-Stakeholder Balanced"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("stage4/output/pricing_dispatch"))
    parser.add_argument("--output-root", type=Path, default=Path("stage4/output/pricing_dispatch/audits"))
    parser.add_argument("--max-pickup-m", type=float, default=6000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    errors: list[str] = []
    for path in sorted(args.run_root.glob("fold=*/exp=*/order_log.parquet")):
        fold = path.parts[-3].split("=", 1)[-1]
        exp = path.parts[-2]
        frame = pd.read_parquet(path)
        served = frame[frame.get("served", False).fillna(False)].copy()
        duplicate_served = int(served["order_id"].duplicated().sum()) if "order_id" in served else len(served)
        av_odd_violations = 0
        strategy = str(served["dispatch_strategy"].dropna().iloc[0]) if len(served) and "dispatch_strategy" in served else ""
        if len(served) and strategy in ODD_STRATEGIES and "quoted_vehicle_type" in served:
            av = served[served["quoted_vehicle_type"].eq("AV")]
            av_odd_violations = int((~av["ODD_feasible"].fillna(False)).sum()) if "ODD_feasible" in av else len(av)
        pickup_violations = int(served["pickup_m"].gt(args.max_pickup_m + 1e-6).sum()) if "pickup_m" in served else len(served)
        negative_pickup_time = int(served["pickup_time_sec"].lt(0).sum()) if "pickup_time_sec" in served else 0
        passenger_reject_served = int((~served.get("accepted", pd.Series(True, index=served.index)).fillna(False)).sum())
        hv_negative_utility = 0
        if len(served) and "quoted_vehicle_type" in served:
            hv = served[served["quoted_vehicle_type"].eq("HV")]
            hv_negative_utility = int(hv["driver_utility"].lt(-1e-6).sum()) if "driver_utility" in hv else 0
        rows.append({
            "fold": fold,
            "experiment": exp,
            "served_orders": len(served),
            "duplicate_served_orders": duplicate_served,
            "av_odd_violations": av_odd_violations,
            "pickup_distance_violations": pickup_violations,
            "pickup_time_violations": negative_pickup_time,
            "passenger_reject_served": passenger_reject_served,
            "hv_negative_utility_served": hv_negative_utility,
        })
        if duplicate_served:
            errors.append(f"{fold}/{exp}: duplicate served order count={duplicate_served}")
        if av_odd_violations:
            errors.append(f"{fold}/{exp}: AV hard ODD violation count={av_odd_violations}")
        if passenger_reject_served:
            errors.append(f"{fold}/{exp}: served passenger rejection count={passenger_reject_served}")
        if pickup_violations:
            errors.append(f"{fold}/{exp}: pickup distance violation count={pickup_violations}")
        if negative_pickup_time:
            errors.append(f"{fold}/{exp}: pickup time violation count={negative_pickup_time}")
        if hv_negative_utility:
            errors.append(f"{fold}/{exp}: HV negative utility served count={hv_negative_utility}")
    pd.DataFrame(rows).to_csv(args.output_root / "matching_feasibility_audit.csv", index=False)
    result = {"status": "PASS" if not errors else "FAIL", "errors": errors, "rows": rows}
    (args.output_root / "matching_feasibility_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": errors[:10]}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
