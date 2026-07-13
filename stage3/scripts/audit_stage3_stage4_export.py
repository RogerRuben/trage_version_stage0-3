"""Audit Stage3-to-Stage4 export schema and leakage boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FORBIDDEN = ("actual", "target", "realized", "post_trip", "future")
REQUIRED = [
    "order_id",
    "date",
    "decision_time",
    "decision_time_source",
    "origin_lon",
    "origin_lat",
    "destination_lon",
    "destination_lat",
    "route_id",
    "lcs_expected",
    "lcs_tail_probability",
    "lcs_uncertainty",
    "pmis_expected",
    "pmis_tail_probability",
    "pmis_uncertainty",
    "rts_expected",
    "rts_tail_probability",
    "rts_uncertainty",
    "core_overall_high_stress_probability",
    "extended_overall_high_stress_probability",
    "iis_applicability",
    "iis_severity",
    "iis_tail_probability",
    "iis_availability",
    "overall_uncertainty",
    "modality_coverage_score",
    "route_prediction_confidence",
    "model_version",
    "prediction_cutoff_time",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("stage3/output/stage4_inputs"))
    parser.add_argument("--output-root", type=Path, default=Path("stage3/output/stage4_inputs/audit"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    for path in sorted(args.input_root.glob("fold=*/stage4_inputs.parquet")):
        fold = path.parent.name.split("=", 1)[-1]
        frame = pd.read_parquet(path)
        missing = [column for column in REQUIRED if column not in frame.columns]
        forbidden = [column for column in frame.columns if any(token in column.lower() for token in FORBIDDEN)]
        duplicate = int(frame["order_id"].duplicated().sum())
        decision_time = pd.to_datetime(frame.get("decision_time"), errors="coerce")
        cutoff_time = pd.to_datetime(frame.get("prediction_cutoff_time"), errors="coerce")
        origin_missing = int(frame[["origin_lon", "origin_lat"]].isna().any(axis=1).sum()) if {"origin_lon", "origin_lat"}.issubset(frame.columns) else len(frame)
        destination_missing = int(frame[["destination_lon", "destination_lat"]].isna().any(axis=1).sum()) if {"destination_lon", "destination_lat"}.issubset(frame.columns) else len(frame)
        invalid_coordinates = 0
        if {"origin_lon", "origin_lat", "destination_lon", "destination_lat"}.issubset(frame.columns):
            coord_mask = (
                frame["origin_lon"].between(-180, 180)
                & frame["destination_lon"].between(-180, 180)
                & frame["origin_lat"].between(-90, 90)
                & frame["destination_lat"].between(-90, 90)
            )
            invalid_coordinates = int((~coord_mask.fillna(False)).sum())
        decision_missing = int(decision_time.isna().sum())
        cutoff_missing = int(cutoff_time.isna().sum())
        decision_before_cutoff = int((decision_time.notna() & cutoff_time.notna() & (decision_time < cutoff_time)).sum())
        if missing:
            errors.append(f"fold={fold} missing {missing}")
        if forbidden:
            errors.append(f"fold={fold} forbidden columns {forbidden}")
        if duplicate:
            errors.append(f"fold={fold} duplicate order_id count={duplicate}")
        if decision_missing:
            errors.append(f"fold={fold} decision_time missing count={decision_missing}")
        if origin_missing or destination_missing or invalid_coordinates:
            errors.append(f"fold={fold} coordinate issue origin_missing={origin_missing} destination_missing={destination_missing} invalid={invalid_coordinates}")
        if decision_before_cutoff:
            errors.append(f"fold={fold} decision_time before prediction_cutoff_time count={decision_before_cutoff}")
        rows.append({
            "fold": fold,
            "rows": len(frame),
            "orders": frame["order_id"].nunique(),
            "missing_required": len(missing),
            "forbidden_columns": len(forbidden),
            "duplicate_orders": duplicate,
            "decision_time_parse_success_rate": float(decision_time.notna().mean()),
            "decision_time_missing": decision_missing,
            "prediction_cutoff_missing": cutoff_missing,
            "origin_destination_missing": origin_missing + destination_missing,
            "invalid_coordinate_count": invalid_coordinates,
            "decision_time_before_prediction_cutoff_count": decision_before_cutoff,
            "exported_order_coverage": float(frame["order_id"].nunique() / max(len(frame), 1)),
        })
    pd.DataFrame(rows).to_csv(args.output_root / "stage4_export_audit.csv", index=False)
    result = {"status": "PASS" if not errors else "FAIL", "errors": errors, "rows": rows}
    (args.output_root / "stage4_export_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": errors[:10]}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
