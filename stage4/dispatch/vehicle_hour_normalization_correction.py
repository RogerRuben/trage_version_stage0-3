"""Close S5A.1 exact vehicle-hour semantics with two bounded neutral replays."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage4.fleetpy_adapter.upstream import FLEETPY_COMMIT, FleetPyCompatibilityError

from . import odd_aware_runner as s4_runner
from . import rolling_or_runner as s3_runner
from .fleet_normalization import exact_baseline_vehicle_hours
from .parameterization_diagnostics import (
    FAMILIES,
    build_neutral_exposure_path,
    fleet_vehicle_hour_scenarios,
    gamma_reference_regimes,
    load_dispatch_ready_exposures,
)

OUTPUT_REL = Path("stage4/output/vehicle_hour_normalization_correction")
DOC_REL = Path("stage4/docs/vehicle_hour_normalization_correction")
BASE_COMMIT = "2be7ffe10a371f505385c18cf70f334df55a5105"
OLD_S4_FINGERPRINT = "a90f1285813cfe5fc9fedeeb6514ed5b204ad5de7a5e230316639d8e1ff2c961"
CORRECTION_CONFIG_REL = Path("stage4/config/vehicle_hour_normalization_correction.json")


def _write_json(value: Any, path: Path) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def _s3_aggregate(summary: dict[str, Any], runtime_output: Path) -> dict[str, Any]:
    outcomes = pd.read_parquet(runtime_output / "request_outcomes.parquet")
    epochs = pd.read_parquet(runtime_output / "candidate_epoch_stats.parquet")
    matched = outcomes["matched"].astype(bool)
    waits = pd.to_numeric(
        outcomes.loc[outcomes["pickup_time"].notna(), "total_request_to_pickup_wait_s"],
        errors="coerce",
    )
    return {
        "requests": len(outcomes),
        "matched": int(matched.sum()),
        "completed": int(outcomes["completed"].sum()),
        "patience_expired": int(outcomes["patience_expired"].sum()),
        "first_window_matched": int(((outcomes["attempt_count"] == 1) & matched).sum()),
        "carry_over_recovered": int(
            (outcomes["entered_carry_over"].astype(bool) & matched).sum()
        ),
        "critical_matched": int(epochs["critical_matched"].sum()),
        "HV_assignments": int((outcomes["vehicle_type"] == "HV").sum()),
        "AV_assignments": int((outcomes["vehicle_type"] == "AV").sum()),
        "pickup_wait_p50_s": float(waits.quantile(0.50)),
        "pickup_wait_p90_s": float(waits.quantile(0.90)),
        "pickup_wait_p95_s": float(waits.quantile(0.95)),
        "runtime_s": float(summary["computation"]["total_runtime_s"]),
        "fleet": summary["fleet"],
    }


def _comparison_fields(value: dict[str, Any]) -> dict[str, int]:
    names = (
        "requests",
        "matched",
        "completed",
        "patience_expired",
        "first_window_matched",
        "carry_over_recovered",
        "critical_matched",
    )
    return {name: int(value[name]) for name in names}


def _write_report(
    root: Path,
    fleet: pd.DataFrame,
    s3: dict[str, Any],
    s4: dict[str, Any],
    gamma: dict[str, Any],
    equality: bool,
) -> None:
    h_exact = float(fleet.iloc[0]["H_base_exact"])
    h_bin = float(fleet.iloc[0]["H_base_15min_equivalent"])
    lines = [
        "# Stage4 S5A.1 Vehicle-Hour Normalization Correction",
        "",
        "Recommendation: `GO_S5B_EXPERIMENTAL_DESIGN`"
        if equality
        else "Recommendation: `REVISE_VEHICLE_HOUR_NORMALIZATION`",
        "",
        "## Root cause",
        "",
        "S0 15-minute overlap-based active supply was integrated as if it were exact vehicle-hours, while HV availability was measured using exact continuous session duration.",
        "",
        "## Corrected semantics",
        "",
        f"- `H_base_exact = {h_exact:.6f}` vehicle-hours: total exact continuous duration of the frozen effective HV sessions; this is the q_A denominator.",
        f"- `H_base_15min_equivalent = {h_bin:.6f}` vehicle-hours: temporal supply-profile bin equivalent; it is not the q_A denominator.",
        "",
        "## Corrected fleet accounting",
        "",
        "| requested q_A | achieved q_A | AV count | raw HV residual h | target HV h | achieved HV h | HV error % | HV sessions |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in fleet.itertuples(index=False):
        lines.append(
            f"| {row.requested_q_A:.2f} | {row.achieved_q_A:.6f} | {row.AV_vehicle_count} | {row.raw_HV_residual_vehicle_hours:.6f} | {row.target_HV_vehicle_hours:.6f} | {row.achieved_HV_vehicle_hours:.6f} | {row.HV_vehicle_hour_error_pct:.6f} | {row.selected_HV_session_count} |"
        )
    lines.extend(
        [
            "",
            "## Corrected q_A=0.25 neutral replay",
            "",
            f"- S3 requests/matched/completed/expired: {s3['requests']}/{s3['matched']}/{s3['completed']}/{s3['patience_expired']}.",
            f"- S3 HV/AV assignments: {s3['HV_assignments']}/{s3['AV_assignments']}; runtime {s3['runtime_s']:.3f}s.",
            f"- S4 requests/matched/completed/expired: {s4['requests']}/{s4['matched']}/{s4['completed']}/{s4['patience_expired']}.",
            f"- S4 first-window/carry-recovered/critical-matched: {s4['first_window_matched']}/{s4['carry_over_recovered']}/{s4['critical_matched']}; runtime {s4['runtime_s']:.3f}s.",
            f"- Corrected S3/S4 aggregate equality: `{equality}`.",
            f"- Corrected S4 fingerprint: `{s4['corrected_canonical_outcome_fingerprint_sha256']}`.",
            "",
            "## Refreshed Gamma references",
            "",
            "Gamma is a cumulative reference-envelope exposure budget, not a safety threshold.",
            "",
            "| family | ZERO | MEAN | PATH | UNCONSTRAINED | PATH-MEAN |",
            "| --- | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for family in FAMILIES:
        row = gamma["families"][family]
        lines.append(
            f"| {family} | 0 | {row['NEUTRAL_FINAL_MEAN']:.6f} | {row['NEUTRAL_PATH_ENVELOPE']:.6f} | null | {row['PATH_minus_MEAN']:.6f} |"
        )
    dynamic = gamma["families"]["dynamic"]
    lines.extend(
        [
            "",
            f"Dynamic PATH maximum occurs at AV assignment rank {dynamic['path_max_assignment_rank']} ({dynamic['path_max_assignment_time']}).",
            "",
            "## Canonical status",
            "",
            "The rolling matcher and ODD-aware kernel remain valid. The previous q_A=0.25 canonical fleet composition used an inconsistent vehicle-hour denominator and is superseded for scientific penetration comparisons.",
            "",
            f"The previous fingerprint `{OLD_S4_FINGERPRINT}` is engineering lineage only. This correction is the canonical base for S5B.",
            "",
        ]
    )
    path = root / DOC_REL / "stage4_s5a1_vehicle_hour_normalization_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_vehicle_hour_normalization_correction(
    root: str | Path, fleetpy_root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    output = root / OUTPUT_REL
    output.mkdir(parents=True, exist_ok=True)
    h_exact = exact_baseline_vehicle_hours(root)
    if not np.isclose(h_exact, 12279.336388888889, rtol=0.0, atol=1e-6):
        raise FleetPyCompatibilityError(f"unexpected exact baseline hours: {h_exact}")
    fleet = fleet_vehicle_hour_scenarios(root)
    if not bool((fleet["HV_vehicle_hour_error_pct"] <= 2.0).all()):
        raise FleetPyCompatibilityError("a corrected HV target exceeds 2% tolerance")
    fleet.to_csv(output / "corrected_fleet_vehicle_hour_scenarios.csv", index=False)
    with tempfile.TemporaryDirectory(
        prefix="_bounded_runtime_", dir=output
    ) as temp_name:
        temp = Path(temp_name)
        old_s3_output, old_s3_doc = s3_runner.OUTPUT_REL, s3_runner.DOC_REL
        try:
            s3_runner.OUTPUT_REL = temp / "s3"
            s3_runner.DOC_REL = temp / "s3_docs"
            s3_summary = s3_runner.run_rolling_or_baseline(root, fleetpy_root)
        finally:
            s3_runner.OUTPUT_REL, s3_runner.DOC_REL = old_s3_output, old_s3_doc
        s3 = _s3_aggregate(s3_summary, temp / "s3")
        expected = _comparison_fields(s3)
        _write_json(s3, output / "corrected_q025_s3_summary.json")
        old_s4_output, old_s4_doc = s4_runner.OUTPUT_REL, s4_runner.DOC_REL
        try:
            s4_runner.OUTPUT_REL = temp / "s4"
            s4_runner.DOC_REL = temp / "s4_docs"
            s4_summary = s4_runner.run_odd_aware_decision_kernel(
                root,
                fleetpy_root,
                config_path=root / CORRECTION_CONFIG_REL,
                expected=expected,
            )
        finally:
            s4_runner.OUTPUT_REL, s4_runner.DOC_REL = old_s4_output, old_s4_doc
        s4 = {
            **s4_summary["canonical_reproduction"],
            "runtime_s": float(s4_summary["computation"]["total_runtime_s"]),
            "corrected_canonical_outcome_fingerprint_sha256": s4_summary[
                "canonical_outcome_fingerprint_sha256"
            ],
            "HV_assignments": int(
                pd.read_parquet(temp / "s4" / "canonical_request_outcomes.parquet")[
                    "vehicle_type"
                ]
                .eq("HV")
                .sum()
            ),
            "AV_assignments": int(
                pd.read_parquet(temp / "s4" / "canonical_request_outcomes.parquet")[
                    "vehicle_type"
                ]
                .eq("AV")
                .sum()
            ),
        }
        equality = _comparison_fields(s4) == expected
        if not equality:
            raise FleetPyCompatibilityError("corrected neutral S4 disagrees with S3")
        corrected_assignment = output / "corrected_q025_s4_assignment_log.parquet"
        shutil.copy2(
            temp / "s4" / "canonical_assignment_log.parquet", corrected_assignment
        )
        exposure = load_dispatch_ready_exposures(root)
        path = build_neutral_exposure_path(
            root,
            exposure,
            assignment_path=corrected_assignment,
            exposure_state_path=temp / "s4" / "canonical_exposure_state.parquet",
        )
        gamma = gamma_reference_regimes(path)
    path.to_parquet(output / "corrected_neutral_exposure_path.parquet", index=False)
    _write_json(s4, output / "corrected_q025_s4_summary.json")
    _write_json(gamma, output / "corrected_gamma_reference_regimes.json")
    _write_report(root, fleet, s3, s4, gamma, equality)
    summary = {
        "phase_status": "STAGE4_S5A1_VEHICLE_HOUR_NORMALIZATION_CORRECTED",
        "recommendation": "GO_S5B_EXPERIMENTAL_DESIGN",
        "base_commit": BASE_COMMIT,
        "fleetpy_commit": FLEETPY_COMMIT,
        "H_base_exact": h_exact,
        "H_base_15min_equivalent": float(fleet.iloc[0]["H_base_15min_equivalent"]),
        "corrected_s3_s4_equality": equality,
        "corrected_canonical_outcome_fingerprint_sha256": s4[
            "corrected_canonical_outcome_fingerprint_sha256"
        ],
    }
    _write_json(summary, output / "correction_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--fleetpy-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_vehicle_hour_normalization_correction(args.root, args.fleetpy_root),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
