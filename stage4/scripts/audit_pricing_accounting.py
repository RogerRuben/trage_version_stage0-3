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
            rows.append({"fold": fold, "experiment": exp, "served_orders": 0, "max_profit_identity_error": 0.0, "negative_fare_count": 0, "bad_compensation_share_count": 0})
            continue
        identity = (served["gross_booking_value"] - served["platform_cost"] - served["platform_profit"]).abs()
        negative_fare = int(served["gross_booking_value"].lt(-args.tolerance).sum())
        share_sum = served.get("compensation_passenger_share", 0).fillna(0) + served.get("compensation_platform_share", 0).fillna(0)
        bad_share = int(share_sum.sub(1.0).abs().gt(args.tolerance).sum())
        max_error = float(identity.max())
        rows.append({
            "fold": fold,
            "experiment": exp,
            "served_orders": len(served),
            "max_profit_identity_error": max_error,
            "negative_fare_count": negative_fare,
            "bad_compensation_share_count": bad_share,
        })
        if max_error > args.tolerance:
            errors.append(f"{fold}/{exp}: max profit identity error={max_error:.6f}")
        if negative_fare:
            errors.append(f"{fold}/{exp}: negative fare count={negative_fare}")
        if bad_share:
            errors.append(f"{fold}/{exp}: compensation share sum error count={bad_share}")
    pd.DataFrame(rows).to_csv(args.output_root / "pricing_accounting_audit.csv", index=False)
    result = {"status": "PASS" if not errors else "FAIL", "errors": errors, "rows": rows}
    (args.output_root / "pricing_accounting_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": errors[:10]}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
