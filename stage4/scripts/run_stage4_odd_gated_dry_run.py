"""ODD-gated assignment dry run using Stage3 condition vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-input-root", type=Path, default=Path("stage3/output/stage4_inputs_core"))
    parser.add_argument("--mapping-root", type=Path, default=Path("stage4/output/capability_mapping"))
    parser.add_argument("--output-root", type=Path, default=Path("stage4/output/odd_gated_dry_run"))
    return parser.parse_args()


def scenario_rows(inputs: pd.DataFrame, mapping: pd.DataFrame, fold: int) -> list[dict]:
    rows = []
    stress = inputs.set_index("order_id")["pred_composite_operational_stress"]
    for scenario in ["Nearest", "GlobalMatch", "Cost-only assignment", "Risk-penalty assignment"]:
        rows.append({
            "fold": fold,
            "scenario": scenario,
            "match_rate": 1.0,
            "cancel_unserved_rate": 0.0,
            "AV_assigned_order_count": 0,
            "AV_ODD_violation_count": 0,
            "AV_mean_stress_exposure": None,
            "HV_mean_stress_exposure": float(stress.mean()),
            "HV_high_stress_share": float(inputs["overall_high_stress_probability"].ge(0.5).mean()),
            "compensation_cost": 0.0,
            "platform_objective": float((1 - stress).mean()),
        })
    for profile, part in mapping[mapping["vehicle_type"].eq("AV")].groupby("vehicle_profile"):
        feasible_orders = set(part.loc[part["service_feasible"], "order_id"])
        av_mask = inputs["order_id"].isin(feasible_orders)
        hv_mask = ~av_mask
        high_hv = inputs.loc[hv_mask, "overall_high_stress_probability"].ge(0.5)
        base = {
            "fold": fold,
            "match_rate": 1.0,
            "cancel_unserved_rate": 0.0,
            "AV_assigned_order_count": int(av_mask.sum()),
            "AV_ODD_violation_count": 0,
            "AV_mean_stress_exposure": float(inputs.loc[av_mask, "pred_composite_operational_stress"].mean()) if av_mask.any() else None,
            "HV_mean_stress_exposure": float(inputs.loc[hv_mask, "pred_composite_operational_stress"].mean()) if hv_mask.any() else None,
            "HV_high_stress_share": float(high_hv.mean()) if hv_mask.any() else None,
        }
        rows.append({**base, "scenario": f"ODD-gated assignment:{profile}", "compensation_cost": float(high_hv.sum() * 2.0), "platform_objective": float((1 - inputs["pred_composite_operational_stress"]).mean() - high_hv.sum() * 2.0 / max(1, len(inputs)))})
        rows.append({**base, "scenario": f"ODD-gated + HV compensation:{profile}", "compensation_cost": float(high_hv.sum() * 5.0), "platform_objective": float((1 - inputs["pred_composite_operational_stress"]).mean() - high_hv.sum() * 5.0 / max(1, len(inputs)))})
    return rows


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.stage3_input_root.glob("fold=*/stage4_inputs.parquet")):
        fold = int(path.parent.name.split("=", 1)[-1])
        inputs = pd.read_parquet(path)
        mapping = pd.read_parquet(args.mapping_root / f"fold={fold}" / "vehicle_capability_mapping.parquet")
        rows.extend(scenario_rows(inputs, mapping, fold))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_root / "odd_gated_dry_run_summary.csv", index=False)
    report = ["# Stage4 ODD-gated dry run", "", "Scenario smoke test; realized labels are not used as decision inputs.", "", frame.to_markdown(index=False, floatfmt=".4f")]
    (args.output_root / "odd_gated_dry_run_report.md").write_text("\n".join(report), encoding="utf-8")
    result = {"status": "PASS" if len(frame) else "FAIL", "rows": len(frame)}
    (args.output_root / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))
    if not len(frame):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
