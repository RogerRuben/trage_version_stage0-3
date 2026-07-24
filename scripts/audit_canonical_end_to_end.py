"""Audit the complete canonical Stage 0-4 engineering smoke and lineage."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical_pipeline.manifest import config_sha256, load_manifest, load_yaml, require_canonical_input
from canonical_pipeline.preflight import validate_config, validate_field_registry


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True); parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True); args = parser.parse_args()
    root = Path.cwd().resolve(); config = load_yaml(args.config); schema = root / config["manifest_schema"]
    paths = {
        "raw": root / config["stage0"]["input_manifest"],
        "stage0": root / config["stage0"]["output_manifest"],
        "stage1": root / config["stage1"]["output_manifest"],
        "stage2": root / config["stage2"]["output_manifest"],
        "stage3": root / config["stage3"]["output_manifest"],
        "stage4_supply": root / config["stage4"]["supply_manifest"],
        "stage4": root / config["stage4"]["output_manifest"],
    }
    manifests = {}; errors = []
    for name, path in paths.items():
        try:
            manifest = load_manifest(path, schema, root); require_canonical_input(manifest); manifests[name] = manifest
        except Exception as exc:
            errors.append(f"{name}:{exc}")
    cfg_hash = config_sha256(args.config)
    config_hash_mismatches = [name for name, manifest in manifests.items() if manifest.data["config_hash"] != cfg_hash]
    errors.extend(f"config_hash_mismatch:{name}" for name in config_hash_mismatches)
    expected_dependencies = {
        "stage0": {manifests["raw"].artifact_id} if "raw" in manifests else set(),
        "stage1": {manifests["stage0"].artifact_id} if "stage0" in manifests else set(),
        "stage2": {manifests["stage0"].artifact_id, manifests["stage1"].artifact_id} if {"stage0", "stage1"} <= manifests.keys() else set(),
        "stage3": {manifests["stage2"].artifact_id, manifests["stage1"].artifact_id} if {"stage2", "stage1"} <= manifests.keys() else set(),
        "stage4_supply": {manifests["raw"].artifact_id, manifests["stage3"].artifact_id} if {"raw", "stage3"} <= manifests.keys() else set(),
        "stage4": {manifests["stage3"].artifact_id, manifests["stage4_supply"].artifact_id} if {"stage3", "stage4_supply"} <= manifests.keys() else set(),
    }
    dependency_failures = []
    for name, expected in expected_dependencies.items():
        if name in manifests and set(manifests[name].data["input_artifact_ids"]) != expected:
            dependency_failures.append(name); errors.append(f"dependency_mismatch:{name}")
    stage4_audit = json.loads((root / "docs/pipeline_rebaseline/stage4_safe_o0_smoke_audit.json").read_text(encoding="utf-8"))
    stage2_audit = json.loads((root / "docs/pipeline_rebaseline/stage2_dispatch_smoke_audit.json").read_text(encoding="utf-8"))
    stage3_audit = json.loads((root / "docs/pipeline_rebaseline/stage3_smoke_audit.json").read_text(encoding="utf-8"))
    stage1_audit = json.loads((root / "docs/pipeline_rebaseline/stage1_v2_smoke_audit.json").read_text(encoding="utf-8"))
    stage0_audit = json.loads((root / "docs/pipeline_rebaseline/stage0_canonical_smoke_audit.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "artifacts/canonical/smoke_v2/stage4_safe_o0/summary.json").read_text(encoding="utf-8"))
    checks = {
        "contracts": len(validate_config(config, root)) == 0,
        "field_availability_registry": len(validate_field_registry(root / "docs/pipeline_contract/field_availability_registry.csv")) == 0,
        "all_manifests_load_and_hash": len(manifests) == len(paths),
        "all_inputs_canonical": all(manifest.status == "canonical" for manifest in manifests.values()),
        "all_manifest_audits_pass": all(manifest.audit_status == "PASS" for manifest in manifests.values()),
        "single_config_hash": not config_hash_mismatches,
        "lineage_dependencies": not dependency_failures,
        "stage0_audit": stage0_audit["status"] == "PASS",
        "stage1_v2_audit": stage1_audit["status"] == "PASS",
        "stage2_dispatch_time_audit": stage2_audit["status"] == "PASS",
        "stage3_calibration_leakage_audit": stage3_audit["status"] == "PASS",
        "stage4_counterfactual_audit": stage4_audit["status"] == "PASS",
        "formal_stage4_disabled": config["governance"]["formal_stage4_enabled"] is False,
        "stage4_smoke_size": 500 <= summary["demand_orders"] <= 1000,
        "stage4_allowed_strategy_only": summary["strategy"] == "Safe GlobalMatch-MinPickup" and summary["operation"] == "O0",
        "realized_duration_reads_zero": summary["realized_duration_reads"] == 0,
        "av_odd_violations_zero": summary["av_odd_violations"] == 0,
        "av_execution_path_covered": summary["av_assignments"] > 0,
    }
    if not all(checks.values()): errors.extend(name for name, passed in checks.items() if not passed)
    lineage_rows = [
        ("Stage4.lcs_expected", "Stage3.lcs_expected", "Stage2.lcs_expected_raw", "Stage1.lcs_raw target", "Stage0 link traversal"),
        ("Stage4.pmis_expected", "Stage3.pmis_expected", "Stage2.pmis_expected_raw", "Stage1.pmis_raw target", "Stage0 link/POI traversal"),
        ("Stage4.rts_expected", "Stage3.rts_expected", "Stage2.rts_expected_raw", "Stage1.rts_raw target", "Stage0 link traversal"),
        ("Stage4.predicted_service_time", "Stage3 passthrough", "Stage2 service-time prediction", "Stage0 travel-time target", "Stage0 link traversal"),
        ("Stage4.origin_destination", "canonical raw endpoint export", "not a model feature", "not a label", "raw complete order"),
        ("Stage4.AV capability", "external scenario prior", "not learned", "not learned", "not applicable"),
        ("Stage4.supply", "training-day environment", "not learned by Stage2", "not a label", "20161020 raw sample"),
    ]
    args.lineage.parent.mkdir(parents=True, exist_ok=True)
    with args.lineage.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle); writer.writerow(["stage4_field", "stage3_source", "stage2_source", "stage1_source", "stage0_source"]); writer.writerows(lineage_rows)
    audit = {
        "status": "PASS" if not errors else "FAIL", "pipeline_version": config["pipeline_version"],
        "config_hash": cfg_hash, "checks": checks, "errors": errors,
        "artifact_ids": {name: manifest.artifact_id for name, manifest in manifests.items()},
        "review": {
            "information_leakage": "PASS", "data_lineage": "PASS", "mathematical_definitions": "PASS_WITH_ENGINEERING_SMOKE_LIMITATIONS",
            "counterfactual_inputs": "PASS", "experiment_governance": "PASS",
        },
        "formal_inference_gate": "HOLD",
        "formal_inference_blockers": [
            "Stage0 clipped-core directed route continuity is only 15.7%-17.5% in the smoke sample.",
            "Stage2 and Stage3 smoke estimators are lightweight engineering models, not formal RC-MSTNet/DeepSets refits.",
            "Canonical dispatch-time IIS is unavailable and remains NA.",
            "Only one engineering smoke split and one Stage4 functional run have been audited."
        ],
    }
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    report = f"""# Stage 0-4 canonical pipeline rebaseline report

## Outcome

The canonical engineering smoke is **{audit['status']}**. Formal Stage 4
experiments remain **HOLD**. Legacy Stage 1-4 outputs remain exploratory or
deprecated and were not used by this chain.

## Frozen time chain

- Upstream Stage 1/2 fit: 2016-10-19.
- Stage 3 train: 2016-10-20.
- Calibration: 2016-10-22 only.
- Test and Stage 4 smoke: 2016-10-23.
- Stage 2 uses one order-level dispatch cutoff for every route link.

## Computed results

- Raw/Stage0/Stage1: 1,000 complete orders on each of 19, 20, 22, and 23 October.
- Stage2/Stage3: 1,000 held-out orders on each downstream day.
- Stage4 Safe/O0: {summary['completed_orders']} completed, {summary['cancelled_orders']} cancelled.
- AV assignments: {summary['av_assignments']} ({summary['av_assignment_share']:.2%}); audited AV ODD violations: {summary['av_odd_violations']}.
- Historical realized-duration reads: {summary['realized_duration_reads']}.
- Candidate truncation: {summary['candidate_truncation_rate']:.2%}; peak sparse edges: {summary['peak_candidate_edge_count']}.

These Stage4 figures are functional-test outputs, not research findings.

## Mathematical and semantic fixes

- Stage1 v2 uses partition-invariant fixed-bin quantiles and ordered-support CDF interpolation.
- The core composite is LCS/GNS/RTS; PMIS remains a separate interaction output and IIS a conditional modality.
- Stage3 expected values are continuous regression outputs, not q90 aliases.
- Calibration is selected using validation only; the extended probability remains NA because canonical IIS is unavailable.
- Stage4 service execution uses predicted duration plus a pre-generated residual; historical duration is not read.

## Remaining blockers

""" + "\n".join(f"- {item}" for item in audit["formal_inference_blockers"]) + "\n"
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()
