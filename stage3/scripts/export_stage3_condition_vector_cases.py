"""Export interpretable example orders for Stage3 condition-vector dimensions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DIMENSIONS = {
    "LCS": ("lcs_tail_probability", "lcs_expected", "lcs_uncertainty", "order_lcs_raw"),
    "PMIS": ("pmis_tail_probability", "pmis_expected", "pmis_uncertainty", "order_pmis_raw"),
    "RTS": ("rts_tail_probability", "rts_expected", "rts_uncertainty", "order_rts_raw"),
    "IIS": ("intersection_tail_probability", "intersection_severity", "overall_uncertainty", "order_iis_severity_q90"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage4-input-root", type=Path, default=Path("stage3/output/stage4_inputs_core_v2"))
    parser.add_argument("--target-root", type=Path, default=Path("stage3/output/rolling_order_targets"))
    parser.add_argument("--output-root", type=Path, default=Path("stage3/output/condition_vector_cases"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.stage4_input_root.glob("fold=*/stage4_inputs.parquet")):
        fold = int(path.parent.name.split("=", 1)[-1])
        inputs = pd.read_parquet(path)
        targets = pd.read_parquet(args.target_root / f"fold={fold}" / "split=test" / "order_targets.parquet")
        data = inputs.merge(targets, on="order_id", suffixes=("", "_target"), validate="one_to_one")
        for dim, (score_col, expected_col, uncertainty_col, realized_col) in DIMENSIONS.items():
            valid = data[score_col].notna()
            part = data.loc[valid].copy()
            if part.empty:
                continue
            selections = {
                "high_score_typical": part.nlargest(1, score_col),
                "low_score_typical": part.nsmallest(1, score_col),
                "high_confidence": part.nsmallest(1, uncertainty_col),
                "low_confidence": part.nlargest(1, uncertainty_col),
            }
            for case_type, selected in selections.items():
                row = selected.iloc[0]
                rows.append({
                    "fold": fold,
                    "dimension": dim,
                    "case_type": case_type,
                    "order_id": row["order_id"],
                    "date": row["date"],
                    "decision_time": row.get("decision_time"),
                    "route_length_m": row.get("route_length_m"),
                    "link_count": row.get("link_count"),
                    "movement_count": row.get("movement_count"),
                    "predicted_score": row.get(score_col),
                    "predicted_expected": row.get(expected_col),
                    "uncertainty": row.get(uncertainty_col),
                    "realized_pressure": row.get(realized_col),
                    "poi_stress_proxy": row.get("pmis_tail_probability"),
                    "iis_applicability": row.get("intersection_applicability"),
                    "iis_tail_probability": row.get("intersection_tail_probability"),
                    "route_id": row.get("route_id"),
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_root / "condition_vector_case_index.csv", index=False)
    report = ["# Stage3 condition-vector cases", "", "Cases are selected to illustrate environment/stress interpretability, not AV safety.", "", frame.to_markdown(index=False, floatfmt=".4f")]
    (args.output_root / "condition_vector_cases_report.md").write_text("\n".join(report), encoding="utf-8")
    (args.output_root / "manifest.json").write_text(json.dumps({"status": "PASS", "cases": len(frame)}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "cases": len(frame)}, indent=2))


if __name__ == "__main__":
    main()
