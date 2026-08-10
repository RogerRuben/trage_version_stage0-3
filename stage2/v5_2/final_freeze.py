"""Freeze the completed Stage 2 v5.2 research line for Stage 3 consumption.

This module is deliberately closure-only.  It reads and verifies already
produced artifacts; it never trains, evaluates, re-runs inference, or selects
new hyperparameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .b1_freeze import verify_b1_evidence_bundle
from .contracts import Stage2V52ContractError
from .phase_c_report import verify_phase_c_evidence_bundle
from .sparsity_diagnostic import verify_evidence_bundle as verify_sparsity_evidence
from .upstream_sampling_audit import verify_evidence_bundle as verify_upstream_evidence
from .verification import sha256_file


FINAL_STATUS = "STAGE2_FINAL_FROZEN"
FINAL_MODEL_ID = "M3"
FINAL_MODEL_ROLE = "structured_representation"
FINAL_CHECKPOINT = "stage2/output_v5_2/development/M3/epoch_004.pt"
FINAL_TRAINING_MANIFEST = "stage2/output_v5_2/development/M3/model_manifest.json"
FINAL_DEVELOPMENT_EVALUATION = (
    "stage2/output_v5_2/development/M3/evaluation/evaluation_manifest.json"
)
FINAL_VALIDATION_EVALUATION = (
    "stage2/output_v5_2/development/M3/validation_evaluation/evaluation_manifest.json"
)
FINAL_TRANSFER_MANIFEST = (
    "stage2/output_v5_2/transfer_shards/protocol=development/transfer_manifest.json"
)

RELEASE_SCHEMA = "stage2_v5_2_final_release_manifest.1"
CONTRACT_SCHEMA = "stage2_v5_2_to_stage3_contract.2"
CLOSURE_REPORT_SCHEMA = "stage2_v5_2_final_closure_report.1"
CLOSURE_TEST_EVIDENCE_SCHEMA = "stage2_v5_2_final_closure_test_evidence.1"
CLOSURE_TEST_EVIDENCE = (
    "stage2/docs/v5_2/stage2_v5_2_final_closure_test_evidence.json"
)

DEPLOYABLE_OUTPUTS = (
    "travel_time_p50",
    "pace_p50",
    "crawl",
    "stop",
    "speed_cv",
    "acceleration_rms",
)
CORE_MICRO_TARGETS = ("crawl", "stop", "speed_cv", "acceleration_rms")

EVIDENCE_PATHS = {
    "phase_b1": "stage2/docs/v5_2/stage2_v5_2_phase_b1_evidence_bundle.json",
    "phase_c": "stage2/docs/v5_2/stage2_v5_2_phase_c_evidence_bundle.json",
    "spatiotemporal_sparsity": (
        "stage2/docs/v5_2/sparsity_diagnostic/"
        "stage2_v5_2_spatiotemporal_sparsity_evidence_bundle.json"
    ),
    "upstream_sampling_representativeness": (
        "stage2/docs/v5_2/upstream_sampling_audit/"
        "stage0_stage1_upstream_representativeness_evidence_bundle.json"
    ),
}

EXPECTED_EVIDENCE_SCHEMAS = {
    "phase_b1": "stage2_v5_2_phase_b1_evidence_bundle.1",
    "phase_c": "stage2_v5_2_phase_c_evidence_bundle.2",
    "spatiotemporal_sparsity": (
        "stage2_v5_2_spatiotemporal_sparsity_evidence_bundle.1"
    ),
    "upstream_sampling_representativeness": (
        "stage0_stage1_upstream_representativeness_evidence.1"
    ),
}

SCIENTIFIC_REPORT_PATHS = {
    "phase_b1": "stage2/docs/v5_2/stage2_v5_2_phase_b1_transfer_tuning_report.json",
    "phase_c": "stage2/docs/v5_2/stage2_v5_2_phase_c_report.json",
    "spatiotemporal_sparsity": (
        "stage2/docs/v5_2/sparsity_diagnostic/"
        "stage2_v5_2_spatiotemporal_sparsity_report.json"
    ),
    "upstream_sampling_representativeness": (
        "stage2/docs/v5_2/upstream_sampling_audit/"
        "stage0_stage1_upstream_representativeness_report.json"
    ),
}

EXPERIMENT_DECISIONS = {
    "M3": {
        "status": "SELECTED_FINAL",
        "role": FINAL_MODEL_ROLE,
    },
    "M4": {
        "status": "REJECTED_ABLATION",
        "reason": "no stable incremental benefit over M3 under the frozen Phase C gate",
    },
    "M5": {
        "status": "CANCELLED_NOT_RUN",
        "reason": "Phase C did not authorize continuation",
    },
    "M6": {
        "status": "CANCELLED_NOT_RUN",
        "reason": "Phase C did not authorize continuation",
    },
    "Phase_D": {
        "status": "CANCELLED",
        "reason": "support-aware spatial expansion was rejected",
    },
    "Transfer_v2": {
        "status": "CANCELLED",
        "reason": "diagnostic evidence did not support a new transfer experiment",
    },
}

ALL_AUTHORIZATIONS_FALSE = {
    "stage2_training": False,
    "stage2_inference_rerun": False,
    "checkpoint_reselection": False,
    "tau_reselection": False,
    "phase_d": False,
    "transfer_v2": False,
    "stage3": False,
    "stage4": False,
    **{f"stage3_s{number}": False for number in range(1, 9)},
}


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    body = deepcopy(dict(payload))
    body.pop("artifact_sha256", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Stage2V52ContractError(f"missing final-freeze artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Stage2V52ContractError(f"final-freeze artifact is not an object: {path}")
    return payload


def _repo_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise Stage2V52ContractError(f"artifact escapes repository root: {value}") from error
    return path


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise Stage2V52ContractError(f"artifact escapes repository root: {path}") from error


def _descriptor(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Stage2V52ContractError(f"cannot freeze missing artifact: {path}")
    return {
        "path": _relative(path, root),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def _verify_descriptor(value: Mapping[str, Any], root: Path) -> None:
    if set(("path", "sha256", "size_bytes")) - set(value):
        raise Stage2V52ContractError("final release descriptor is incomplete")
    path = _repo_path(root, str(value["path"]))
    if (
        not path.is_file()
        or sha256_file(path) != value["sha256"]
        or int(path.stat().st_size) != int(value["size_bytes"])
    ):
        raise Stage2V52ContractError(
            f"final release descriptor does not resolve: {value.get('path')}"
        )


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, encoding="utf-8"
    ).strip()


def _deep_verify_scientific_evidence(
    evidence: Mapping[str, Mapping[str, Any]], *, repo_root: Path
) -> dict[str, Any]:
    return {
        "phase_b1": verify_b1_evidence_bundle(
            evidence["phase_b1"], repo_root=repo_root
        ),
        "phase_c": verify_phase_c_evidence_bundle(
            evidence["phase_c"], repo_root=repo_root
        ),
        "spatiotemporal_sparsity": verify_sparsity_evidence(
            evidence["spatiotemporal_sparsity"], repo_root=repo_root
        ),
        "upstream_sampling_representativeness": verify_upstream_evidence(
            evidence["upstream_sampling_representativeness"], repo_root=repo_root
        ),
    }


def _validate_frozen_decisions(
    evidence: Mapping[str, Mapping[str, Any]], *, repo_root: Path
) -> None:
    b1 = evidence["phase_b1"]
    phase_c = evidence["phase_c"]
    sparsity = evidence["spatiotemporal_sparsity"]
    upstream = evidence["upstream_sampling_representativeness"]

    b1_report_path = b1.get("reports", {}).get("json", {}).get("path")
    if b1_report_path is None:
        # B1 v1 binds the report under the top-level prepared/report inventory.
        b1_report_path = "stage2/docs/v5_2/stage2_v5_2_phase_b1_transfer_tuning_report.json"
    b1_report = _read_json(_repo_path(repo_root, str(b1_report_path)))
    if b1_report.get("scientific_conclusion", {}).get("classification") != "CASE_C":
        raise Stage2V52ContractError("B1 scientific conclusion is not frozen CASE_C")

    if (
        phase_c.get("phase_c_direction") != "FAIL"
        or phase_c.get("phase_d_authorized") is not False
    ):
        raise Stage2V52ContractError("Phase C does not support final closure")

    diagnostic_path = sparsity.get("reports", {}).get("json", {}).get("path")
    diagnostic = _read_json(_repo_path(repo_root, str(diagnostic_path)))
    classification = diagnostic.get("diagnostic_classification", {})
    if (
        classification.get("classification") != "DIAG-B"
        or classification.get("evidence", {}).get("structured_transfer_positive") is not True
        or classification.get("evidence", {}).get("m4_gate_increment_supported") is not False
    ):
        raise Stage2V52ContractError("sparsity diagnostic does not support M3 closure")

    upstream_report_path = upstream.get("outputs", {}).get("reports", {}).get("json", {}).get("path")
    upstream_report = _read_json(_repo_path(repo_root, str(upstream_report_path)))
    if upstream_report.get("final_classification", {}).get("classification") != "UP-D":
        raise Stage2V52ContractError("upstream audit classification is not frozen UP-D")


def _contract_markdown(
    *, checkpoint: Mapping[str, Any], training: Mapping[str, Any]
) -> str:
    return f"""# Stage 2 v5.2 → Stage 3 final contract

Contract schema: `{CONTRACT_SCHEMA}`

Stage 2 status: `{FINAL_STATUS}`

Stage 3 authorization: `NO`

## Frozen predictor

- Model: `M3` (`structured_representation`).
- Checkpoint: `{checkpoint['path']}`.
- Checkpoint SHA-256: `{checkpoint['sha256']}`.
- Training manifest SHA-256: `{training['sha256']}`.
- Stage 3 must not retrain Stage 2, reselect a checkpoint, reselect tau, or alter
  the frozen feature construction.

## Deployable outputs

Stage 3 may consume only the decision-time predictions `travel_time_p50`,
`pace_p50`, `crawl`, `stop`, `speed_cv`, and `acceleration_rms`, plus the frozen
identity, support, provenance, distance, and completeness fields required to
interpret them. `RTS` remains diagnostic-only and must not enter ODD–TOD
compatibility, AV feasibility, fallback selection, or Stage 4.

These dynamic outputs are **predicted operational-condition proxies**. They are
not AV accident, safety, takeover, or failure probabilities.

## Route and inference boundary

- The historical Test31 route is the fixed HV route and the first AV candidate.
- Original-route products retain ordered directed-edge/traversal identity.
- A future Stage 3 candidate-route adapter may tokenize a candidate directed-edge
  sequence, construct decision-time entry/horizon features, and run this exact
  frozen M3 checkpoint.
- Candidate predictions must never be copied from the historical route and may
  not be manually imputed. If the frozen input contract cannot be satisfied,
  the candidate dynamic state is `UNKNOWN`.
- Stage 3 owns any later bounded fallback search and returns one fixed AV route
  or no feasible route. Stage 2 performs no path search.

## Downstream decision boundary

Stage 4 receives fixed HV/AV routes, feasibility, service time, distance,
exposure, and unknown state. **Stage 4 has no route decision variable.** It must
not receive candidate-route search or Stage 3 threshold fitting.

## Frozen scientific interpretation

- B1: no convincing support-aware transfer evidence (`CASE_C`).
- Phase C: M4 failed the pre-registered continuation rule.
- Spatiotemporal diagnostic: structured M3 signal is positive while the M4 gate
  has no stable incremental support (`DIAG-B`).
- Upstream audit: demand concentration, Stage 0 selection, and Stage 1 label
  availability compound sparse-context attrition (`UP-D`).

Accordingly M3 is the final engineering predictor; M4 is a rejected ablation;
M5/M6, Phase D, and Transfer-v2 are cancelled.

## Authorization

This contract freezes the interface only. S1–S8, Stage 3 execution, and Stage 4
remain unauthorized until the user explicitly approves the next phase.
"""


def build_final_release(
    *, repo_root: str | Path, contract_path: str | Path, deep_verify: bool = True
) -> dict[str, Any]:
    """Build a closure manifest from immutable existing Stage 2 artifacts."""

    root = Path(repo_root).resolve()
    evidence = {
        name: _read_json(_repo_path(root, path))
        for name, path in EVIDENCE_PATHS.items()
    }
    for name, payload in evidence.items():
        if (
            payload.get("schema_version") != EXPECTED_EVIDENCE_SCHEMAS[name]
            or payload.get("status") != "PASS"
        ):
            raise Stage2V52ContractError(f"invalid frozen evidence identity: {name}")

    # Validate both the local evidence graph and the decision summaries before
    # publishing the final selection.  No model computation occurs here.
    if not deep_verify:
        raise Stage2V52ContractError(
            "a final release requires deep scientific evidence verification"
        )
    verifications = _deep_verify_scientific_evidence(evidence, repo_root=root)

    checkpoint_path = _repo_path(root, FINAL_CHECKPOINT)
    training_path = _repo_path(root, FINAL_TRAINING_MANIFEST)
    development_evaluation_path = _repo_path(root, FINAL_DEVELOPMENT_EVALUATION)
    validation_evaluation_path = _repo_path(root, FINAL_VALIDATION_EVALUATION)
    transfer_manifest_path = _repo_path(root, FINAL_TRANSFER_MANIFEST)
    checkpoint = _descriptor(checkpoint_path, root)
    training_descriptor = _descriptor(training_path, root)
    development_evaluation = _descriptor(development_evaluation_path, root)
    validation_evaluation = _descriptor(validation_evaluation_path, root)
    transfer_manifest = _descriptor(transfer_manifest_path, root)
    test_evidence_path = _repo_path(root, CLOSURE_TEST_EVIDENCE)
    test_evidence = _read_json(test_evidence_path)
    checks = test_evidence.get("checks", {})
    if (
        test_evidence.get("schema_version") != CLOSURE_TEST_EVIDENCE_SCHEMA
        or test_evidence.get("status") != "PASS"
        or test_evidence.get("training_or_inference_run") is not False
        or test_evidence.get("next_phase_authorized") is not False
        or set(checks) != {
            "targeted_final_freeze",
            "full_stage2_regression",
            "compileall",
            "scientific_evidence_deep_verification",
        }
        or any(check.get("status") != "PASS" for check in checks.values())
        or int(checks["targeted_final_freeze"].get("failed", -1)) != 0
        or int(checks["full_stage2_regression"].get("failed", -1)) != 0
        or int(checks["scientific_evidence_deep_verification"].get("failed", -1)) != 0
    ):
        raise Stage2V52ContractError("S0 closure test evidence is not passing")

    training = _read_json(training_path)
    evaluation = _read_json(development_evaluation_path)
    if (
        training.get("status") != "PASS"
        or training.get("model_id") != FINAL_MODEL_ID
        or training.get("selected_checkpoint_path") != FINAL_CHECKPOINT
        or training.get("selected_checkpoint_sha256") != checkpoint["sha256"]
        or training.get("constructor", {}).get("spatial_mode") != "concat"
        or evaluation.get("status") != "PASS"
        or evaluation.get("model_id") != FINAL_MODEL_ID
        or evaluation.get("checkpoint_sha256") != checkpoint["sha256"]
        or tuple(evaluation.get("adoption_targets", ())) != CORE_MICRO_TARGETS
        or evaluation.get("rts_stage3_deployable") is not False
    ):
        raise Stage2V52ContractError("M3 artifacts do not satisfy the final predictor contract")

    # Decision validation uses reports referenced from the evidence graph.
    # It is kept separate from deep file hashing so semantic drift also fails.
    _validate_frozen_decisions(evidence, repo_root=root)

    contract_file = (
        _repo_path(root, str(contract_path))
        if not Path(contract_path).is_absolute()
        else Path(contract_path).resolve()
    )
    _relative(contract_file, root)
    _atomic_text(
        contract_file,
        _contract_markdown(checkpoint=checkpoint, training=training_descriptor),
    )

    evidence_descriptors: dict[str, Any] = {}
    for name, path in EVIDENCE_PATHS.items():
        evidence_descriptors[name] = {
            **_descriptor(_repo_path(root, path), root),
            "schema_version": EXPECTED_EVIDENCE_SCHEMAS[name],
            "verification": verifications[name],
        }

    manifest: dict[str, Any] = {
        "schema_version": RELEASE_SCHEMA,
        "status": FINAL_STATUS,
        "phase": "S0_STAGE2_FINAL_CLOSURE",
        "closure_base_commit": _git_head(root),
        "stage3_authorized": False,
        "next_phase_authorized": False,
        "authorizations": dict(ALL_AUTHORIZATIONS_FALSE),
        "closure_artifacts": {
            "implementation": _descriptor(root / "stage2/v5_2/final_freeze.py", root),
            "tests": _descriptor(root / "stage2/tests_v5_2/test_final_freeze.py", root),
            "test_evidence": {
                **_descriptor(test_evidence_path, root),
                "schema_version": CLOSURE_TEST_EVIDENCE_SCHEMA,
            },
            "frozen_experiment_config": _descriptor(
                root / "stage2/config/stage2_v5_2.json", root
            ),
        },
        "upstream_frozen_releases": {
            "stage0": _descriptor(
                root / "stage0/docs/stage0_v6_freeze_manifest.json", root
            ),
            "stage1": _descriptor(
                root / "stage1/docs/stage1_v3_release_manifest.json", root
            ),
        },
        "final_predictor": {
            "model_id": FINAL_MODEL_ID,
            "model_role": FINAL_MODEL_ROLE,
            "selection_status": "FINAL_FROZEN",
            "checkpoint": checkpoint,
            "training_manifest": training_descriptor,
            "development_evaluation": development_evaluation,
            "validation_evaluation": validation_evaluation,
            "transfer_input_manifest": transfer_manifest,
            "inference_policy": {
                "retraining_allowed": False,
                "checkpoint_reselection_allowed": False,
                "tau_reselection_allowed": False,
                "feature_contract_changes_allowed": False,
                "candidate_route_adapter_must_use_frozen_m3": True,
            },
        },
        "target_roles": {
            "deployable_outputs": list(DEPLOYABLE_OUTPUTS),
            "core_micro_targets": list(CORE_MICRO_TARGETS),
            "diagnostic_only": ["rts"],
            "rts_stage3_deployable": False,
            "claim_boundary": "predicted_operational_condition_proxies_not_AV_risk_probabilities",
        },
        "experiment_decisions": deepcopy(EXPERIMENT_DECISIONS),
        "scientific_evidence": evidence_descriptors,
        "scientific_reports": {
            "phase_b1": {
                **_descriptor(_repo_path(root, SCIENTIFIC_REPORT_PATHS["phase_b1"]), root),
                "classification": "CASE_C",
            },
            "phase_c": {
                **_descriptor(_repo_path(root, SCIENTIFIC_REPORT_PATHS["phase_c"]), root),
                "direction": "FAIL",
            },
            "spatiotemporal_sparsity": {
                **_descriptor(
                    _repo_path(root, SCIENTIFIC_REPORT_PATHS["spatiotemporal_sparsity"]),
                    root,
                ),
                "classification": "DIAG-B",
            },
            "upstream_sampling_representativeness": {
                **_descriptor(
                    _repo_path(
                        root,
                        SCIENTIFIC_REPORT_PATHS[
                            "upstream_sampling_representativeness"
                        ],
                    ),
                    root,
                ),
                "classification": "UP-D",
            },
        },
        "scientific_conclusion": {
            "final_model": "M3 structured representation",
            "support_aware_gate": "REJECTED",
            "sparsity_diagnostic": "DIAG-B",
            "upstream_sampling_audit": "UP-D",
            "statement": (
                "Structured representation has a repeatable positive signal, while "
                "support-aware spatial gating has no reliable incremental benefit; "
                "freeze M3 and stop Stage 2 expansion."
            ),
        },
        "stage2_to_stage3_contract": {
            **_descriptor(contract_file, root),
            "schema_version": CONTRACT_SCHEMA,
        },
        "artifact_storage": {
            "checkpoint_repository_tracked": False,
            "checkpoint_required_for_stage3_inference": True,
            "checkpoint_integrity_is_sha256_bound": True,
        },
    }
    manifest["artifact_sha256"] = _canonical_hash(manifest)
    return manifest


def verify_final_release_manifest(
    payload: Mapping[str, Any], *, repo_root: str | Path, deep_verify: bool = False
) -> dict[str, Any]:
    """Verify final status, artifact bindings, and all execution boundaries."""

    root = Path(repo_root).resolve()
    if (
        payload.get("schema_version") != RELEASE_SCHEMA
        or payload.get("status") != FINAL_STATUS
        or payload.get("phase") != "S0_STAGE2_FINAL_CLOSURE"
        or payload.get("stage3_authorized") is not False
        or payload.get("next_phase_authorized") is not False
        or payload.get("artifact_sha256") != _canonical_hash(payload)
    ):
        raise Stage2V52ContractError("invalid Stage 2 final release identity")

    authorizations = payload.get("authorizations", {})
    if authorizations != ALL_AUTHORIZATIONS_FALSE or any(authorizations.values()):
        raise Stage2V52ContractError("a downstream or Stage 2 execution phase is authorized")

    predictor = payload.get("final_predictor", {})
    if (
        predictor.get("model_id") != FINAL_MODEL_ID
        or predictor.get("model_role") != FINAL_MODEL_ROLE
        or predictor.get("selection_status") != "FINAL_FROZEN"
        or predictor.get("checkpoint", {}).get("path") != FINAL_CHECKPOINT
        or any(predictor.get("inference_policy", {}).get(key) is not False for key in (
            "retraining_allowed",
            "checkpoint_reselection_allowed",
            "tau_reselection_allowed",
            "feature_contract_changes_allowed",
        ))
        or predictor.get("inference_policy", {}).get(
            "candidate_route_adapter_must_use_frozen_m3"
        ) is not True
    ):
        raise Stage2V52ContractError("final M3 predictor policy changed")

    target_roles = payload.get("target_roles", {})
    if (
        tuple(target_roles.get("deployable_outputs", ())) != DEPLOYABLE_OUTPUTS
        or tuple(target_roles.get("core_micro_targets", ())) != CORE_MICRO_TARGETS
        or target_roles.get("diagnostic_only") != ["rts"]
        or target_roles.get("rts_stage3_deployable") is not False
    ):
        raise Stage2V52ContractError("Stage 2 target roles changed")

    if payload.get("experiment_decisions") != EXPERIMENT_DECISIONS:
        raise Stage2V52ContractError("Stage 2 final experiment decisions changed")

    descriptors = [
        predictor["checkpoint"],
        predictor["training_manifest"],
        predictor["development_evaluation"],
        predictor["validation_evaluation"],
        predictor["transfer_input_manifest"],
        payload["upstream_frozen_releases"]["stage0"],
        payload["upstream_frozen_releases"]["stage1"],
        payload["stage2_to_stage3_contract"],
    ]
    evidence_payloads: dict[str, dict[str, Any]] = {}
    scientific = payload.get("scientific_evidence", {})
    if set(scientific) != set(EVIDENCE_PATHS):
        raise Stage2V52ContractError("scientific evidence inventory is incomplete")
    for name, value in scientific.items():
        if (
            value.get("path") != EVIDENCE_PATHS[name]
            or value.get("schema_version") != EXPECTED_EVIDENCE_SCHEMAS[name]
            or value.get("verification", {}).get("status") != "PASS"
        ):
            raise Stage2V52ContractError(f"scientific evidence identity changed: {name}")
        descriptors.append(value)
        evidence_payloads[name] = _read_json(_repo_path(root, str(value["path"])))
    for descriptor in descriptors:
        _verify_descriptor(descriptor, root)

    reports = payload.get("scientific_reports", {})
    if set(reports) != set(SCIENTIFIC_REPORT_PATHS):
        raise Stage2V52ContractError("scientific report inventory is incomplete")
    for name, descriptor in reports.items():
        if descriptor.get("path") != SCIENTIFIC_REPORT_PATHS[name]:
            raise Stage2V52ContractError(f"scientific report identity changed: {name}")
        _verify_descriptor(descriptor, root)
    b1_report = _read_json(_repo_path(root, reports["phase_b1"]["path"]))
    phase_c_report = _read_json(_repo_path(root, reports["phase_c"]["path"]))
    sparsity_report = _read_json(
        _repo_path(root, reports["spatiotemporal_sparsity"]["path"])
    )
    upstream_report = _read_json(
        _repo_path(root, reports["upstream_sampling_representativeness"]["path"])
    )
    if (
        reports["phase_b1"].get("classification") != "CASE_C"
        or b1_report.get("scientific_conclusion", {}).get("classification") != "CASE_C"
        or reports["phase_c"].get("direction") != "FAIL"
        or phase_c_report.get("phase_c_direction") != "FAIL"
        or reports["spatiotemporal_sparsity"].get("classification") != "DIAG-B"
        or sparsity_report.get("diagnostic_classification", {}).get("classification")
        != "DIAG-B"
        or reports["upstream_sampling_representativeness"].get("classification")
        != "UP-D"
        or upstream_report.get("final_classification", {}).get("classification")
        != "UP-D"
    ):
        raise Stage2V52ContractError("scientific report decision changed")

    closure_artifacts = payload.get("closure_artifacts", {})
    if set(closure_artifacts) != {
        "implementation", "tests", "test_evidence", "frozen_experiment_config"
    }:
        raise Stage2V52ContractError("S0 closure artifact inventory is incomplete")
    for descriptor in closure_artifacts.values():
        _verify_descriptor(descriptor, root)
    test_evidence = _read_json(
        _repo_path(root, closure_artifacts["test_evidence"]["path"])
    )
    if (
        closure_artifacts["test_evidence"].get("schema_version")
        != CLOSURE_TEST_EVIDENCE_SCHEMA
        or test_evidence.get("status") != "PASS"
        or test_evidence.get("training_or_inference_run") is not False
        or test_evidence.get("next_phase_authorized") is not False
        or any(
            check.get("status") != "PASS"
            for check in test_evidence.get("checks", {}).values()
        )
    ):
        raise Stage2V52ContractError("S0 closure test evidence changed")

    checkpoint = predictor["checkpoint"]
    training = _read_json(_repo_path(root, predictor["training_manifest"]["path"]))
    if (
        training.get("model_id") != FINAL_MODEL_ID
        or training.get("selected_checkpoint_path") != checkpoint["path"]
        or training.get("selected_checkpoint_sha256") != checkpoint["sha256"]
    ):
        raise Stage2V52ContractError("M3 training/checkpoint binding changed")

    if deep_verify:
        results = _deep_verify_scientific_evidence(
            evidence_payloads, repo_root=root
        )
        if any(result.get("status") != "PASS" for result in results.values()):
            raise Stage2V52ContractError("deep scientific evidence verification failed")

    contract = _repo_path(root, payload["stage2_to_stage3_contract"]["path"])
    contract_text = contract.read_text(encoding="utf-8")
    for required in (
        "Stage 3 authorization: `NO`",
        "frozen M3 checkpoint",
        "RTS` remains diagnostic-only",
        "Stage 4 has no route decision variable",
        "S1–S8, Stage 3 execution, and Stage 4",
    ):
        if required not in contract_text:
            raise Stage2V52ContractError(f"Stage 2→3 contract misses: {required}")

    return {
        "schema_version": "stage2_v5_2_final_release_verification.1",
        "status": "PASS",
        "stage2_status": FINAL_STATUS,
        "final_model": FINAL_MODEL_ID,
        "resolved_descriptor_count": (
            len(descriptors) + len(closure_artifacts) + len(reports)
        ),
        "scientific_evidence_count": len(scientific),
        "stage3_authorized": False,
        "next_phase_authorized": False,
    }


def write_final_closure(
    *,
    repo_root: str | Path,
    manifest_path: str | Path,
    contract_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest = build_final_release(
        repo_root=root, contract_path=contract_path, deep_verify=True
    )
    manifest_file = _repo_path(root, str(manifest_path))
    report_file = _repo_path(root, str(report_path))
    _atomic_json(manifest_file, manifest)
    verification = verify_final_release_manifest(
        manifest, repo_root=root, deep_verify=True
    )
    report = {
        "schema_version": CLOSURE_REPORT_SCHEMA,
        "status": "PASS",
        "phase_status": FINAL_STATUS,
        "base_commit": manifest["closure_base_commit"],
        "final_predictor": FINAL_MODEL_ID,
        "manifest": _descriptor(manifest_file, root),
        "contract": manifest["stage2_to_stage3_contract"],
        "verification": verification,
        "limitations": [
            "The M3 checkpoint is hash-bound local production state and is ignored by Git.",
            "No Stage 3 static inventory, calibration, inference, or experiment was run in S0.",
        ],
        "blockers": [],
        "next_phase_authorized": False,
    }
    report["artifact_sha256"] = _canonical_hash(report)
    _atomic_json(report_file, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--manifest",
        default="stage2/docs/v5_2/stage2_v5_2_final_release_manifest.json",
    )
    parser.add_argument(
        "--contract",
        default="stage2/docs/v5_2/stage2_v5_2_to_stage3_contract.md",
    )
    parser.add_argument(
        "--report",
        default="stage2/docs/v5_2/stage2_v5_2_final_closure_report.json",
    )
    args = parser.parse_args(argv)
    report = write_final_closure(
        repo_root=args.repo_root,
        manifest_path=args.manifest,
        contract_path=args.contract,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
