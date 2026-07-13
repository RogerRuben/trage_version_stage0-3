"""Audit full-day Stage2/Stage3/Stage4 inference coverage and leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20161023")
    parser.add_argument("--order-base", type=Path, default=Path("stage0/output/order_base/day=20161023.parquet"))
    parser.add_argument("--od-path", type=Path, default=Path("stage0/output/order_od_audited/day=20161023.parquet"))
    parser.add_argument("--route-conditioned", type=Path, required=True)
    parser.add_argument("--stage2-predictions", type=Path, required=True)
    parser.add_argument("--stage4-inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("stage3/docs/results"))
    parser.add_argument("--report", type=Path, default=Path("stage3/docs/stage3_full_day_20161023_inference_report.md"))
    return parser.parse_args()


def _orders(path: Path) -> set[str]:
    return set(pd.read_parquet(path, columns=["order_id"])["order_id"].astype(str))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    raw = _orders(args.order_base)
    od = _orders(args.od_path)
    route = _orders(args.route_conditioned)
    stage2 = _orders(args.stage2_predictions)
    stage4 = pd.read_parquet(args.stage4_inputs)
    stage4["order_id"] = stage4["order_id"].astype(str)
    stage4_orders = set(stage4["order_id"])

    coverage_rows = [
        {"stage": "raw_stage0_order_base", "orders": len(raw), "share_of_raw": 1.0},
        {"stage": "stage0_od_valid", "orders": len(od), "share_of_raw": len(od) / max(len(raw), 1)},
        {"stage": "route_conditioned_ready", "orders": len(route), "share_of_raw": len(route) / max(len(raw), 1)},
        {"stage": "stage2_rc_mstnet_predicted", "orders": len(stage2), "share_of_raw": len(stage2) / max(len(raw), 1)},
        {"stage": "stage3_stage4_exported", "orders": len(stage4_orders), "share_of_raw": len(stage4_orders) / max(len(raw), 1)},
        {"stage": "not_route_conditioned", "orders": len(raw - route), "share_of_raw": len(raw - route) / max(len(raw), 1)},
    ]
    coverage = pd.DataFrame(coverage_rows)
    coverage_path = args.output_dir / "full_day_inference_coverage.csv"
    coverage.to_csv(coverage_path, index=False)

    core_prob_cols = [
        column for column in [
            "lcs_tail_probability",
            "pmis_tail_probability",
            "rts_tail_probability",
            "core_overall_high_stress_probability",
            "extended_overall_high_stress_probability",
        ] if column in stage4.columns
    ]
    prob_cols = [column for column in stage4.columns if column.endswith("_probability")]
    expected_cols = [column for column in ["lcs_expected", "pmis_expected", "rts_expected"] if column in stage4.columns]
    decision = pd.to_datetime(stage4["decision_time"], utc=True, errors="coerce")
    cutoff = pd.to_datetime(stage4["prediction_cutoff_time"], utc=True, errors="coerce")
    origin = pd.to_datetime(stage4.get("origin_timestamp"), utc=True, errors="coerce") if "origin_timestamp" in stage4 else pd.Series(pd.NaT, index=stage4.index)
    audit = {
        "date": args.date,
        "raw_orders": len(raw),
        "od_valid_orders": len(od),
        "route_conditioned_ready_orders": len(route),
        "stage2_predicted_orders": len(stage2),
        "stage4_exported_orders": len(stage4_orders),
        "not_route_conditioned_orders": len(raw - route),
        "stage2_matches_route_orders": stage2 == route,
        "stage4_matches_stage2_orders": stage4_orders == stage2,
        "order_id_unique": int(stage4["order_id"].nunique()) == len(stage4),
        "core_condition_missing_rows": int(stage4[expected_cols + core_prob_cols].isna().any(axis=1).sum()) if expected_cols and core_prob_cols else None,
        "probability_range_violations": int(sum(((pd.to_numeric(stage4[col], errors="coerce") < 0) | (pd.to_numeric(stage4[col], errors="coerce") > 1)).sum() for col in core_prob_cols)),
        "expected_range_violations": int(sum(((pd.to_numeric(stage4[col], errors="coerce") < 0) | (pd.to_numeric(stage4[col], errors="coerce") > 1)).sum() for col in expected_cols)),
        "decision_time_parse_success_rate": float(decision.notna().mean()),
        "prediction_cutoff_after_decision_count": int((cutoff > decision).fillna(False).sum()),
        "origin_timestamp_fallback_count": int(stage4.get("decision_time_source", pd.Series("", index=stage4.index)).astype(str).eq("origin_timestamp_fallback").sum()),
        "iis_available_orders": int(stage4.get("iis_availability", pd.Series(False, index=stage4.index)).fillna(False).astype(bool).sum()),
        "fallback_condition_columns_present": [column for column in stage4.columns if "fallback" in column.lower() or "condition_source" in column.lower()],
        "status": "PASS",
    }
    audit["status"] = "PASS" if (
        audit["stage2_matches_route_orders"]
        and audit["stage4_matches_stage2_orders"]
        and audit["order_id_unique"]
        and audit["core_condition_missing_rows"] == 0
        and audit["probability_range_violations"] == 0
        and audit["expected_range_violations"] == 0
        and audit["prediction_cutoff_after_decision_count"] == 0
        and not audit["fallback_condition_columns_present"]
    ) else "FAIL"
    audit_path = args.output_dir / "full_day_inference_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = [
        "# Full-day 2016-10-23 Stage3 inference audit",
        "",
        coverage.to_markdown(index=False),
        "",
        f"Audit status: **{audit['status']}**",
        "",
        "- Missing condition vectors were not imputed.",
        "- Orders outside the exported universe failed before route-conditioned inference and are reported as `not_route_conditioned`.",
        "- IIS full-day movement predictions are unavailable in this run; IIS is represented by `iis_availability=false`, not by zero stress.",
    ]
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
