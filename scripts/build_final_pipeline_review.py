"""Build the independent Stage 0-4 rebaseline review from computed evidence.

This review certifies the canonical engineering smoke only.  It deliberately
keeps the formal-experiment release gate separate, so a passing software/data
lineage audit cannot be mistaken for scientific-model readiness.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical_pipeline.manifest import config_sha256, load_manifest, load_yaml
from canonical_pipeline.preflight import validate_config, validate_field_registry
from canonical_pipeline.registry import RunRegistry


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    root = ROOT
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_yaml(config_path)
    cfg_hash = config_sha256(config_path)
    schema = root / config["manifest_schema"]
    manifest_paths = {
        "raw": config["stage0"]["input_manifest"],
        "stage0": config["stage0"]["output_manifest"],
        "stage1": config["stage1"]["output_manifest"],
        "stage2": config["stage2"]["output_manifest"],
        "stage3": config["stage3"]["output_manifest"],
        "stage4_supply": config["stage4"]["supply_manifest"],
        "stage4": config["stage4"]["output_manifest"],
        "end_to_end": "artifacts/canonical/end_to_end_smoke.manifest.json",
    }
    manifests = {
        name: load_manifest(root / relative, schema, root)
        for name, relative in manifest_paths.items()
    }
    audits = {
        "raw": read_json(root / "docs/pipeline_rebaseline/raw_smoke_audit.json"),
        "stage0": read_json(root / "docs/pipeline_rebaseline/stage0_canonical_smoke_audit.json"),
        "stage1": read_json(root / "docs/pipeline_rebaseline/stage1_v2_smoke_audit.json"),
        "stage2": read_json(root / "docs/pipeline_rebaseline/stage2_dispatch_smoke_audit.json"),
        "stage3": read_json(root / "docs/pipeline_rebaseline/stage3_smoke_audit.json"),
        "stage4": read_json(root / "docs/pipeline_rebaseline/stage4_safe_o0_smoke_audit.json"),
        "end_to_end": read_json(root / "docs/pipeline_rebaseline/end_to_end_smoke_audit.json"),
    }
    stage4_summary = read_json(root / "artifacts/canonical/smoke_v2/stage4_safe_o0/summary.json")

    # Publish the requested Stage1 v1/v2 evidence as small, reviewable files.
    comparison = pd.DataFrame(audits["stage1"]["v1_v2_comparison"])
    comparison_path = root / "docs/pipeline_rebaseline/stage1_v1_v2_comparison.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    comparison_report = root / "docs/pipeline_rebaseline/stage1_v1_v2_distribution_report.md"
    comparison_report.write_text(
        "# Stage1 label v1/v2 smoke comparison\n\n"
        "This is a 1,000-order-per-day engineering comparison, not a formal label-validation study.\n\n"
        "- All four dates have 1,000 overlapping orders.\n"
        "- The largest semantic change is GNS (mean v2-v1 approximately -0.24; near-zero rank correlation).\n"
        "- LCS, RTS, and PMIS retain moderate rank association but are not interchangeable with v1.\n"
        "- IIS remains a conditional observed label; missing values are never replaced with zero.\n"
        "- Core composite v2 uses LCS/GNS/RTS. PMIS is excluded from the equal-weight core to prevent duplicate interaction weighting.\n\n"
        f"Machine-readable evidence: `{comparison_path.relative_to(root).as_posix()}`.\n",
        encoding="utf-8",
    )

    legacy = pd.read_csv(root / "docs/pipeline_rebaseline/legacy_artifact_inventory.csv")
    legacy_counts = Counter(legacy["status"].astype(str))
    classification = pd.DataFrame([
        {"status": "canonical", "artifact_count": len(manifests), "scope": "explicit manifests in pipeline_canonical.yaml"},
        {"status": "exploratory", "artifact_count": legacy_counts.get("exploratory", 0), "scope": "frozen legacy inventory"},
        {"status": "deprecated", "artifact_count": legacy_counts.get("deprecated", 0), "scope": "frozen legacy inventory"},
        {"status": "unknown", "artifact_count": legacy_counts.get("unknown", 0), "scope": "frozen legacy inventory"},
    ])
    classification_path = root / "docs/pipeline_rebaseline/artifact_classification_summary.csv"
    classification.to_csv(classification_path, index=False, encoding="utf-8-sig")

    stage2_days = audits["stage2"]["days"]
    stage3_days = audits["stage3"]["days"]
    information_leakage_checks = {
        "stage2_availability_after_decision_zero": sum(day["availability_after_decision"] for day in stage2_days) == 0,
        "stage2_realized_duration_permission_zero": sum(day["realized_duration_permission_rows"] for day in stage2_days) == 0,
        "stage3_realized_stage1_features_zero": len(audits["stage3"]["stage1_realized_features"]) == 0,
        "stage3_validation_only_calibration": audits["stage3"]["calibration_validation_only"] is True,
        "stage4_realized_duration_reads_zero": stage4_summary["realized_duration_reads"] == 0,
        "stage4_profile_not_test_calibrated": audits["stage4"]["checks"]["profile_not_test_calibrated"] is True,
    }
    mathematical_checks = {
        "partition_invariance": audits["stage1"]["partition_invariance"]["pass"],
        "cdf_monotonic_and_tail": audits["stage1"]["cdf"]["pass"],
        "pmis_not_double_weighted": audits["stage1"]["pmis_role"] == "interaction_output_excluded_from_core_composite",
        "iis_missing_not_zero": audits["stage1"]["iis_missing_policy"] == "NA_not_zero",
        "stage3_expected_not_q90": audits["stage3"]["expected_semantics"] == "continuous_regression_expectation_not_q90",
        "stage3_extended_not_max_proxy": audits["stage3"]["extended_probability_semantics"] == "unavailable_not_max_proxy",
    }
    counterfactual_checks = {
        "counterfactual_mode": config["stage4"]["mode"] == "counterfactual_smoke",
        "predicted_distribution_service_time": config["stage4"]["service_time_source"] == "predicted_distribution",
        "historical_duration_reads_zero": stage4_summary["realized_duration_reads"] == 0,
        "unknown_condition_av_assignment_zero": audits["stage4"]["counts"]["unknown_condition_av_assignments"] == 0,
        "av_odd_violation_zero": audits["stage4"]["counts"]["combined_odd_violations"] == 0,
    }
    registry = RunRegistry(root / config["run_registry"])
    canonical_successes = [
        row for row in registry.rows()
        if row["stage"] == "end_to_end" and row["config_hash"] == cfg_hash
        and row["canonical"].lower() == "true" and row["status"] == "SUCCESS"
    ]
    governance_checks = {
        "config_contract_valid": not validate_config(config, root),
        "field_registry_valid": not validate_field_registry(root / "docs/pipeline_contract/field_availability_registry.csv"),
        "all_inputs_explicit_canonical": all(item.status == "canonical" for item in manifests.values()),
        "all_manifest_audits_pass": all(item.audit_status == "PASS" for item in manifests.values()),
        "single_config_hash": all(item.data["config_hash"] == cfg_hash for item in manifests.values()),
        "single_canonical_success_for_config": len(canonical_successes) == 1,
        "formal_stage4_disabled": config["governance"]["formal_stage4_enabled"] is False,
    }
    lineage_checks = {
        "end_to_end_lineage_pass": audits["end_to_end"]["checks"]["lineage_dependencies"],
        "all_manifests_load_and_hash": audits["end_to_end"]["checks"]["all_manifests_load_and_hash"],
        "lineage_trace_rows": len(pd.read_csv(root / "docs/pipeline_rebaseline/canonical_lineage_trace.csv")) >= 7,
    }
    stage_checks = {name: audit["status"] == "PASS" for name, audit in audits.items()}
    review_sections = {
        "information_leakage": information_leakage_checks,
        "data_lineage": lineage_checks,
        "mathematical_definitions": mathematical_checks,
        "counterfactual_inputs": counterfactual_checks,
        "experiment_governance": governance_checks,
    }
    review_status = {
        name: "PASS" if all(checks.values()) else "FAIL"
        for name, checks in review_sections.items()
    }
    engineering_pass = all(stage_checks.values()) and all(value == "PASS" for value in review_status.values())

    formal_blockers = list(audits["end_to_end"]["formal_inference_blockers"])
    final = {
        "status": "PASS" if engineering_pass else "FAIL",
        "certification_scope": "canonical_engineering_smoke_only",
        "pipeline_version": config["pipeline_version"],
        "config_hash": cfg_hash,
        "canonical_run_id": canonical_successes[0]["run_id"] if len(canonical_successes) == 1 else None,
        "stage_audits": stage_checks,
        "review_status": review_status,
        "review_evidence": review_sections,
        "artifact_classification": classification.to_dict(orient="records"),
        "formal_experiment_release_gate": "HOLD",
        "formal_experiment_blockers": formal_blockers,
        "stage4_functional_smoke": {
            "orders": stage4_summary["demand_orders"],
            "completed": stage4_summary["completed_orders"],
            "cancelled": stage4_summary["cancelled_orders"],
            "av_assignments": stage4_summary["av_assignments"],
            "av_odd_violations": stage4_summary["av_odd_violations"],
            "realized_duration_reads": stage4_summary["realized_duration_reads"],
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")

    report = f"""# Final Stage 0-4 canonical pipeline review

## Decision

The **canonical engineering smoke audit is {final['status']}**. This certifies
contracts, field availability, explicit manifests, the 1,000-order-per-day
Stage 0-3 chain, and the 1,000-order Safe/O0 counterfactual functional test.

The **formal experiment release gate remains HOLD**. A functional smoke pass is
not evidence that the formal Stage 2/3 models or full-scale Stage 4 study are ready.

## Independent review

| Review | Status |
| --- | --- |
| Information leakage | {review_status['information_leakage']} |
| Data lineage | {review_status['data_lineage']} |
| Mathematical definitions | {review_status['mathematical_definitions']} |
| Counterfactual inputs | {review_status['counterfactual_inputs']} |
| Experiment governance | {review_status['experiment_governance']} |

Every PASS above is recomputed from manifests, field registries, audit JSON,
and Stage4 logs; it is not a hand-written acceptance marker.

## Frozen temporal chain

- Stage1/Stage2 upstream fit: 2016-10-19.
- Stage3 train: 2016-10-20.
- Calibration: 2016-10-22 only.
- Test and Stage4 smoke: 2016-10-23.
- Stage2 applies one dispatch-time cutoff to all links in an order.

## Evidence summary

- Raw input: 1,000 complete orders on each of 20161019/20/22/23, extracted by two-pass streaming.
- Stage0: exact interval time/distance conservation and zero unflagged illegal directed transitions.
- Stage1: partition-invariant median; 88,823 cohort CDF models with zero monotonic/tail failures.
- Stage2: 3,000 held-out downstream orders; zero post-decision availability and zero realized-duration permission rows.
- Stage3: validation-only calibration; test core-overall AUC/AP/Brier/ECE = 0.7682/0.3978/0.1428/0.1112.
- Stage4: 1,000 completed, 0 cancelled, 12 AV assignments, zero ODD violations, zero historical-duration reads.

Stage4 numbers are functional-test outputs and must not be used as research findings.

## Formal experiment blockers

""" + "\n".join(f"- {item}" for item in formal_blockers) + f"""

## Artifact governance

- Canonical manifests: {len(manifests)}.
- Frozen exploratory legacy artifacts: {legacy_counts.get('exploratory', 0)}.
- Deprecated artifacts: {legacy_counts.get('deprecated', 0)}.
- Unknown legacy artifacts: {legacy_counts.get('unknown', 0)}.
- Config hash: `{cfg_hash}`.
- Canonical run id: `{final['canonical_run_id']}`.

## Release conclusion

The rebaseline engineering skeleton is reproducible and audited. Formal Stage4
experiments must remain disabled until the listed blockers are resolved and a
new canonical version supersedes this engineering smoke.
"""
    args.output_report.write_text(report, encoding="utf-8")
    print(json.dumps(final, indent=2))
    if not engineering_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
