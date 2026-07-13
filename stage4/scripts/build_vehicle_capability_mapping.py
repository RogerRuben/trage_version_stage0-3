"""Map Stage3 condition vectors to dimension-specific vehicle capability outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DIMENSIONS = {
    "lcs": "lcs_tail_probability",
    "pmis": "pmis_tail_probability",
    "rts": "rts_tail_probability",
    "iis": "intersection_tail_probability",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-input-root", type=Path, default=Path("stage3/output/stage4_inputs_core_v2"))
    parser.add_argument("--profiles", type=Path, default=Path("stage4/config/vehicle_capability_profiles.json"))
    parser.add_argument("--output-root", type=Path, default=Path("stage4/output/capability_mapping_v2"))
    return parser.parse_args()


def dimension_available(frame: pd.DataFrame, dimension: str) -> pd.Series:
    if dimension == "iis":
        return frame.get("iis_availability", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    return pd.Series(True, index=frame.index)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    profile_doc = json.loads(args.profiles.read_text(encoding="utf-8"))
    profiles = profile_doc["profiles"]
    manifest = []
    for path in sorted(args.stage3_input_root.glob("fold=*/stage4_inputs.parquet")):
        fold = path.parent.name.split("=", 1)[-1]
        frame = pd.read_parquet(path)
        outputs = []
        for name, profile in profiles.items():
            sensitivity = profile["dimension_sensitivity"]
            soft = profile["dimension_soft_threshold"]
            hard = profile["dimension_hard_threshold"]
            missing_penalty = float(profile.get("missing_modality_penalty", 0.0))
            weighted_sum = pd.Series(0.0, index=frame.index)
            weight_sum = pd.Series(0.0, index=frame.index)
            hard_violations = []
            soft_violations = []
            missing_dimensions = []
            dimension_margins = {}
            for dim, column in DIMENSIONS.items():
                if column not in frame:
                    available = pd.Series(False, index=frame.index)
                    value = pd.Series(np.nan, index=frame.index)
                else:
                    available = dimension_available(frame, dim)
                    value = pd.to_numeric(frame[column], errors="coerce")
                weight = float(sensitivity.get(dim, 1.0))
                weighted_sum = weighted_sum + value.fillna(0.0).where(available, 0.0) * weight
                weight_sum = weight_sum + available.astype(float) * weight
                hard_violations.append((value.gt(float(hard[dim])) & available).map(lambda flag, d=dim: d if flag else ""))
                soft_violations.append((value.gt(float(soft[dim])) & available).map(lambda flag, d=dim: d if flag else ""))
                missing_dimensions.append((~available).map(lambda flag, d=dim: d if flag else ""))
                dimension_margins[dim] = (float(hard[dim]) - value).where(available, np.nan)
            stress = weighted_sum / weight_sum.replace(0, np.nan)
            missing_count = pd.concat(missing_dimensions, axis=1).ne("").sum(axis=1)
            uncertainty = pd.to_numeric(frame["overall_uncertainty"], errors="coerce").fillna(0.0)
            effective_uncertainty = uncertainty + missing_count * missing_penalty
            adjusted_stress = stress.fillna(0.0) + effective_uncertainty * 0.25
            hard_text = pd.concat(hard_violations, axis=1).agg(lambda row: ",".join([value for value in row if value]), axis=1)
            soft_text = pd.concat(soft_violations, axis=1).agg(lambda row: ",".join([value for value in row if value]), axis=1)
            missing_text = pd.concat(missing_dimensions, axis=1).agg(lambda row: ",".join([value for value in row if value]), axis=1)
            margin_frame = pd.DataFrame(dimension_margins)
            minimum_dimension_margin = margin_frame.min(axis=1, skipna=True)
            binding_dimension = margin_frame.idxmin(axis=1, skipna=True).fillna("none")
            uncertainty_margin = float(profile["uncertainty_tolerance"]) - effective_uncertainty
            odd_margin = pd.concat([minimum_dimension_margin.rename("minimum_dimension_margin"), uncertainty_margin.rename("uncertainty_margin")], axis=1).min(axis=1, skipna=True)
            binding_dimension = binding_dimension.where(minimum_dimension_margin.le(uncertainty_margin), "uncertainty")
            hard_broken = hard_text.ne("")
            feasible = (~hard_broken) & effective_uncertainty.le(float(profile["uncertainty_tolerance"]))
            soft_only = soft_text.ne("") & feasible
            capability_cost = (
                adjusted_stress * 10
                + soft_only.astype(float) * float(profile["remote_assistance_cost_placeholder"])
                + (~feasible).astype(float) * float(profile["fallback_cost_placeholder"])
            ).astype("float32")
            outputs.append(pd.DataFrame({
                "fold": int(fold),
                "order_id": frame["order_id"],
                "date": frame["date"],
                "vehicle_profile": name,
                "vehicle_type": profile["vehicle_type"],
                "scenario_parameter_status": profile_doc["parameter_status"],
                "service_feasible": feasible,
                "feasible_with_extra_cost": soft_only,
                "capability_cost": capability_cost,
                "ODD_margin": odd_margin.astype("float32"),
                "uncertainty_margin": uncertainty_margin.astype("float32"),
                "minimum_dimension_margin": minimum_dimension_margin.astype("float32"),
                "binding_dimension": binding_dimension,
                "threshold_violation_dimensions": hard_text,
                "soft_threshold_violation_dimensions": soft_text,
                "missing_modality_dimensions": missing_text,
                "missing_modality_uncertainty_penalty": (missing_count * missing_penalty).astype("float32"),
                "effective_uncertainty": effective_uncertainty.astype("float32"),
                "availability_adjusted_stress": adjusted_stress.astype("float32"),
                "uncertainty_adjusted_feasibility": feasible,
            }))
        output = pd.concat(outputs, ignore_index=True)
        fold_root = args.output_root / f"fold={fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        output.to_parquet(fold_root / "vehicle_capability_mapping.parquet", index=False, compression="zstd")
        manifest.append({"fold": int(fold), "rows": len(output), "orders": int(frame["order_id"].nunique()), "profiles": len(profiles)})
    result = {"status": "PASS" if manifest else "FAIL", "folds": manifest, "profile_version": profile_doc["profile_version"]}
    (args.output_root / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not manifest:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
