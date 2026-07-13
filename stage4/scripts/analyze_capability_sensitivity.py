"""Summarize scenario capability sensitivity across AV/HV profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-input-root", type=Path, default=Path("stage3/output/stage4_inputs_core_v2"))
    parser.add_argument("--mapping-root", type=Path, default=Path("stage4/output/capability_mapping_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("stage4/output/capability_sensitivity"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.mapping_root.glob("fold=*/vehicle_capability_mapping.parquet")):
        fold = int(path.parent.name.split("=", 1)[-1])
        mapping = pd.read_parquet(path)
        inputs = pd.read_parquet(args.stage3_input_root / f"fold={fold}" / "stage4_inputs.parquet")
        stress = inputs.set_index("order_id")["composite_expected"]
        for profile, group in mapping.groupby("vehicle_profile"):
            feasible = group["service_feasible"]
            av_stress = stress.reindex(group.loc[feasible, "order_id"])
            hv_stress = stress.reindex(group.loc[~feasible, "order_id"])
            rows.append({
                "fold": fold,
                "vehicle_profile": profile,
                "vehicle_type": group["vehicle_type"].iloc[0],
                "AV_feasible_order_share": float(feasible.mean()),
                "AV_mean_stress_exposure": float(av_stress.mean()) if len(av_stress) else None,
                "HV_residual_stress_burden": float(hv_stress.mean()) if len(hv_stress) else None,
                "ODD_margin_mean": float(group["ODD_margin"].mean()),
                "ODD_margin_p10": float(group["ODD_margin"].quantile(0.10)),
                "missing_modality_penalty_mean": float(group["missing_modality_uncertainty_penalty"].mean()),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_root / "capability_sensitivity_by_fold.csv", index=False)
    summary = frame.groupby("vehicle_profile", as_index=False)[["AV_feasible_order_share", "AV_mean_stress_exposure", "HV_residual_stress_burden", "ODD_margin_mean", "ODD_margin_p10"]].mean()
    summary.to_csv(args.output_root / "capability_sensitivity_summary.csv", index=False)
    report = ["# Stage4 capability scenario sensitivity", "", "All profiles are scenario priors, not empirical AV capability estimates.", "", summary.to_markdown(index=False, floatfmt=".4f")]
    (args.output_root / "stage4_capability_scenario_report.md").write_text("\n".join(report), encoding="utf-8")
    (args.output_root / "manifest.json").write_text(json.dumps({"status": "PASS"}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
