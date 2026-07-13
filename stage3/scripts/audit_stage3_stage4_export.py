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
    "pred_stop_go_stress",
    "pred_poi_mediated_stress",
    "pred_reliability_stress",
    "pred_composite_operational_stress",
    "overall_high_stress_probability",
    "overall_uncertainty",
    "modality_coverage_score",
    "route_prediction_confidence",
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
        if missing:
            errors.append(f"fold={fold} missing {missing}")
        if forbidden:
            errors.append(f"fold={fold} forbidden columns {forbidden}")
        if duplicate:
            errors.append(f"fold={fold} duplicate order_id count={duplicate}")
        rows.append({"fold": fold, "rows": len(frame), "orders": frame["order_id"].nunique(), "missing_required": len(missing), "forbidden_columns": len(forbidden), "duplicate_orders": duplicate})
    pd.DataFrame(rows).to_csv(args.output_root / "stage4_export_audit.csv", index=False)
    result = {"status": "PASS" if not errors else "FAIL", "errors": errors, "rows": rows}
    (args.output_root / "stage4_export_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": errors[:10]}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
