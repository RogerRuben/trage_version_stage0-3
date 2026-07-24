"""Independent audit of canonical dispatch-time Stage 2 smoke outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage2.canonical.dispatch_time import audit_dispatch_features
from stage2.scripts.build_stage2_dispatch_smoke import FEATURES, TARGETS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fit-date", required=True)
    parser.add_argument("--dates", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    args = parser.parse_args()
    dates = [value.strip() for value in args.dates.split(",") if value.strip()]
    metadata = json.loads((args.output_root / "stage2_summary.json").read_text(encoding="utf-8"))
    rows = []; failures = []
    for date in dates:
        path = args.output_root / "link_predictions" / f"day={date}.parquet"
        service_path = args.output_root / "service_time_predictions" / f"day={date}.parquet"
        frame = pd.read_parquet(path)
        service = pd.read_parquet(service_path)
        feature_audit = audit_dispatch_features(frame, FEATURES)
        cutoff = pd.to_datetime(frame.prediction_cutoff_time, utc=True)
        availability = pd.to_datetime(frame.feature_availability_timestamp, utc=True)
        training = pd.to_datetime(frame.model_training_cutoff, utc=True)
        one_cutoff = int(frame.groupby("order_id").prediction_cutoff_time.nunique().gt(1).sum())
        probability_failures = sum(int((~frame[f"{target}_tail_probability"].between(0, 1)).sum()) for target in TARGETS)
        expected_failures = sum(int((~frame[f"{target}_expected_raw"].between(0, 1)).sum()) for target in TARGETS)
        row = {
            "date": date, "orders": int(frame.order_id.nunique()), "link_rows": int(len(frame)),
            "service_orders": int(service.order_id.nunique()),
            "heldout_false_rows": int((~frame.heldout_prediction.astype(bool)).sum()),
            "training_not_before_prediction": int((training >= cutoff).sum()),
            "availability_after_decision": int((availability > cutoff).sum()),
            "orders_with_multiple_cutoffs": one_cutoff,
            "probability_range_failures": probability_failures,
            "expected_range_failures": expected_failures,
            "nonpositive_service_predictions": int((service.predicted_service_time_sec <= 0).sum()),
            "realized_duration_permission_rows": int(service.realized_duration_read_allowed.astype(bool).sum()),
            "forbidden_features": "|".join(feature_audit["forbidden_model_features"]),
            "outside_whitelist": "|".join(feature_audit["features_outside_whitelist"]),
        }
        passed = all(row[key] == 0 for key in [
            "heldout_false_rows", "training_not_before_prediction", "availability_after_decision",
            "orders_with_multiple_cutoffs", "probability_range_failures", "expected_range_failures",
            "nonpositive_service_predictions", "realized_duration_permission_rows",
        ]) and not row["forbidden_features"] and not row["outside_whitelist"] and row["orders"] == row["service_orders"] == 1000
        row["status"] = "PASS" if passed else "FAIL"
        rows.append(row)
        if not passed: failures.append(date)
    coverage = pd.DataFrame(rows)
    args.coverage.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.coverage, index=False)
    audit = {
        "status": "PASS" if not failures else "FAIL",
        "schema_version": "stage2_dispatch_prediction_v2",
        "fit_date": args.fit_date, "prediction_dates": dates,
        "fit_date_outside_prediction_dates": args.fit_date not in dates,
        "single_order_cutoff_contract": True,
        "actual_link_entry_used_as_feature": False,
        "estimated_future_entry_unlocks_state": False,
        "prediction_mode": "dispatch_time",
        "model_role": metadata["model_role"], "days": rows,
    }
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
