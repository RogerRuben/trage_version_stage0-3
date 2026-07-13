"""Map Stage3 condition vectors to scenario vehicle capability outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DIMENSIONS = {
    "stop_go": "pred_stop_go_stress",
    "poi_mediated": "pred_poi_mediated_stress",
    "reliability": "pred_reliability_stress",
    "intersection": "pred_intersection_stress",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-input-root", type=Path, default=Path("stage3/output/stage4_inputs_core"))
    parser.add_argument("--profiles", type=Path, default=Path("stage4/config/vehicle_capability_profiles.json"))
    parser.add_argument("--output-root", type=Path, default=Path("stage4/output/capability_mapping"))
    return parser.parse_args()


def weighted_stress(frame: pd.DataFrame, sensitivity: dict[str, float]) -> pd.Series:
    total = 0.0
    denom = 0.0
    for key, column in DIMENSIONS.items():
        if column in frame:
            weight = float(sensitivity.get(key, 1.0))
            total = total + frame[column].fillna(0.0) * weight
            denom += weight
    return total / max(denom, 1e-9)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    profiles = json.loads(args.profiles.read_text(encoding="utf-8"))["profiles"]
    manifest = []
    for path in sorted(args.stage3_input_root.glob("fold=*/stage4_inputs.parquet")):
        fold = path.parent.name.split("=", 1)[-1]
        frame = pd.read_parquet(path)
        outputs = []
        for name, profile in profiles.items():
            stress = weighted_stress(frame, profile["stress_sensitivity"])
            uncertainty = frame["overall_uncertainty"].fillna(0.0)
            odd_margin = float(profile["odd_hard_threshold"]) - (stress + uncertainty * 0.25)
            feasible = odd_margin.ge(0) & uncertainty.le(float(profile["uncertainty_tolerance"]))
            violations = []
            for key, column in DIMENSIONS.items():
                if column in frame:
                    violations.append(frame[column].fillna(0).mul(profile["stress_sensitivity"].get(key, 1.0)).gt(profile["odd_hard_threshold"]).map(lambda flag, k=key: k if flag else ""))
            violation_text = pd.concat(violations, axis=1).agg(lambda row: ",".join([value for value in row if value]), axis=1) if violations else ""
            outputs.append(pd.DataFrame({
                "fold": int(fold),
                "order_id": frame["order_id"],
                "date": frame["date"],
                "vehicle_profile": name,
                "vehicle_type": profile["vehicle_type"],
                "service_feasible": feasible,
                "capability_cost": (stress * 10 + uncertainty * 5 + (~feasible).astype(float) * profile["fallback_cost_placeholder"]).astype("float32"),
                "ODD_margin": odd_margin.astype("float32"),
                "threshold_violation_dimensions": violation_text,
                "uncertainty_adjusted_feasibility": feasible,
            }))
        output = pd.concat(outputs, ignore_index=True)
        fold_root = args.output_root / f"fold={fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        output.to_parquet(fold_root / "vehicle_capability_mapping.parquet", index=False, compression="zstd")
        manifest.append({"fold": int(fold), "rows": len(output), "orders": int(frame["order_id"].nunique())})
    result = {"status": "PASS" if manifest else "FAIL", "folds": manifest}
    (args.output_root / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not manifest:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
