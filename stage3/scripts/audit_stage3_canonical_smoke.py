"""Independent leakage, calibration, and semantic audit for Stage 3 smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


TARGETS = ("lcs", "pmis", "rts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-date", required=True)
    parser.add_argument("--validation-date", required=True)
    parser.add_argument("--test-date", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.output_root / "stage3_summary.json").read_text(encoding="utf-8"))
    dates = [args.train_date, args.validation_date, args.test_date]
    frames = {date: pd.read_parquet(args.output_root / "condition_vectors" / f"day={date}.parquet") for date in dates}
    rows = []; failures = []
    for date, frame in frames.items():
        cutoff = pd.to_datetime(frame.prediction_cutoff_time, utc=True)
        decision = pd.to_datetime(frame.decision_time, utc=True)
        probability_failures = sum(int((~frame[f"{target}_tail_probability"].between(0, 1)).sum()) for target in TARGETS)
        expected_failures = sum(int((~frame[f"{target}_expected"].between(0, 1)).sum()) for target in TARGETS)
        missing_iis_non_na = int(frame.loc[~frame.iis_availability.astype(bool), [
            "iis_applicability_probability", "iis_conditional_severity", "iis_tail_probability"
        ]].notna().sum().sum())
        row = {
            "date": date, "orders": int(frame.order_id.nunique()), "duplicate_orders": int(frame.order_id.duplicated().sum()),
            "cutoff_after_decision": int((cutoff > decision).sum()),
            "probability_range_failures": probability_failures,
            "expected_range_failures": expected_failures,
            "missing_iis_non_na_values": missing_iis_non_na,
            "realized_stage1_feature_count": int(frame.realized_stage1_feature_count.sum()),
            "extended_probability_nonmissing": int(frame.extended_overall_high_stress_probability.notna().sum()),
            "condition_unavailable": int((~frame.condition_available.astype(bool)).sum()),
        }
        passed = row["orders"] == 1000 and all(row[key] == 0 for key in row if key not in {"date", "orders"})
        row["status"] = "PASS" if passed else "FAIL"; rows.append(row)
        if not passed: failures.append(date)
    drift = []
    for target in TARGETS:
        train = frames[args.train_date][f"{target}_expected"]
        test = frames[args.test_date][f"{target}_expected"]
        statistic, pvalue = ks_2samp(train, test)
        drift.append({"field": f"{target}_expected", "ks_statistic": float(statistic), "p_value": float(pvalue),
                      "train_mean": float(train.mean()), "test_mean": float(test.mean()),
                      "status": "PASS" if statistic < 0.50 else "FAIL"})
        if statistic >= 0.50: failures.append(f"drift:{target}")
    coverage = pd.DataFrame(rows); args.coverage.parent.mkdir(parents=True, exist_ok=True); coverage.to_csv(args.coverage, index=False)
    audit = {
        "status": "PASS" if not failures else "FAIL", "schema_version": "stage3_condition_vector_v2",
        "temporal_order": f"{args.train_date} < {args.validation_date} < {args.test_date}",
        "calibration_fit_date": summary["calibration_fit_date"],
        "calibration_validation_only": summary["calibration_fit_date"] == args.validation_date,
        "stage1_realized_features": summary["stage1_realized_feature_columns"],
        "expected_semantics": "continuous_regression_expectation_not_q90",
        "extended_probability_semantics": "unavailable_not_max_proxy",
        "iis_missing_policy": summary["iis_policy"], "days": rows, "distribution_drift": drift,
    }
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
