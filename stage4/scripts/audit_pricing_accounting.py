"""Audit Stage4 pricing, payout, and platform-profit accounting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("stage4/output/pricing_dispatch"))
    parser.add_argument("--output-root", type=Path, default=Path("stage4/output/pricing_dispatch/audits"))
    parser.add_argument("--tolerance", type=float, default=1e-4)
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
        if served.empty:
            rows.append({"fold": fold, "experiment": exp, "served_orders": 0, "max_profit_identity_error": 0.0, "negative_fare_count": 0, "bad_compensation_share_count": 0, "bad_compensation_identity_count": 0, "bad_driver_payout_identity_count": 0, "bad_platform_cost_identity_count": 0, "bad_fare_identity_count": 0})
            continue
        identity = (served["gross_booking_value"] - served["platform_cost"] - served["platform_profit"]).abs()
        negative_fare = int(served["gross_booking_value"].lt(-args.tolerance).sum())
        share_sum = served.get("compensation_passenger_share", 0).fillna(0) + served.get("compensation_platform_share", 0).fillna(0)
        bad_share = int(share_sum.sub(1.0).abs().gt(args.tolerance).sum())
        comp_identity = (served["gross_stress_compensation"].fillna(0) - served["passenger_funded_compensation"].fillna(0) - served["platform_funded_compensation"].fillna(0)).abs()
        bad_comp_identity = int(comp_identity.gt(args.tolerance).sum())
        hv = served[served["quoted_vehicle_type"].eq("HV")].copy()
        bad_driver_identity = 0
        bad_platform_identity = 0
        if len(hv):
            driver_expected = hv[["base_driver_payout", "service_time_payout", "pickup_compensation", "scarcity_bonus", "gross_stress_compensation"]].fillna(0).sum(axis=1)
            bad_driver_identity = int((driver_expected - hv["driver_total_payout"]).abs().gt(args.tolerance).sum())
            platform_expected = hv[["base_driver_payout", "service_time_payout", "pickup_compensation", "scarcity_bonus", "platform_funded_compensation", "platform_hv_variable_cost"]].fillna(0).sum(axis=1)
            bad_platform_identity = int((platform_expected - hv["platform_cost"]).abs().gt(args.tolerance).sum())
        fare_expected = served[["base_fare_component", "surge_component", "vehicle_adjustment", "passenger_funded_compensation"]].fillna(0).sum(axis=1)
        bad_fare_identity = int((fare_expected - served["gross_booking_value"]).abs().gt(args.tolerance).sum())
        p0_bad_comp = 0
        if "pricing_mechanism" in served:
            p0 = served[served["pricing_mechanism"].eq("P0_uniform")]
            p0_bad_comp = int(p0["gross_stress_compensation"].fillna(0).abs().gt(args.tolerance).sum()) if len(p0) else 0
        max_error = float(identity.max())
        rows.append({
            "fold": fold,
            "experiment": exp,
            "served_orders": len(served),
            "max_profit_identity_error": max_error,
            "negative_fare_count": negative_fare,
            "bad_compensation_share_count": bad_share,
            "bad_compensation_identity_count": bad_comp_identity,
            "bad_driver_payout_identity_count": bad_driver_identity,
            "bad_platform_cost_identity_count": bad_platform_identity,
            "bad_fare_identity_count": bad_fare_identity,
            "p0_nonzero_compensation_count": p0_bad_comp,
        })
        if max_error > args.tolerance:
            errors.append(f"{fold}/{exp}: max profit identity error={max_error:.6f}")
        if negative_fare:
            errors.append(f"{fold}/{exp}: negative fare count={negative_fare}")
        if bad_share:
            errors.append(f"{fold}/{exp}: compensation share sum error count={bad_share}")
        if bad_comp_identity:
            errors.append(f"{fold}/{exp}: compensation funding identity error count={bad_comp_identity}")
        if bad_driver_identity:
            errors.append(f"{fold}/{exp}: driver payout identity error count={bad_driver_identity}")
        if bad_platform_identity:
            errors.append(f"{fold}/{exp}: platform cost identity error count={bad_platform_identity}")
        if bad_fare_identity:
            errors.append(f"{fold}/{exp}: fare identity error count={bad_fare_identity}")
        if p0_bad_comp:
            errors.append(f"{fold}/{exp}: P0 nonzero compensation count={p0_bad_comp}")
    pd.DataFrame(rows).to_csv(args.output_root / "pricing_accounting_audit.csv", index=False)
    result = {"status": "PASS" if not errors else "FAIL", "errors": errors, "rows": rows}
    (args.output_root / "pricing_accounting_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": errors[:10]}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
