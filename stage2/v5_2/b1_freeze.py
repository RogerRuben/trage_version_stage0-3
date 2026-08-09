"""Read-only Phase B1 result extraction and hash-bound evidence verification."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import CORE_TRANSFER_TARGETS, Stage2V52ContractError
from .protocols import get_protocol
from .support_transfer import validate_embedded_hash
from .verification import sha256_file


B1_EVIDENCE_SCHEMA_VERSION = "stage2_v5_2_phase_b1_evidence_bundle.1"
B1_SUMMARY_SCHEMA_VERSION = "stage2_v5_2_phase_b1_transfer_tuning_report.2"
B1_EXECUTION_COMMIT = "876f41ea0f879d57ef4e7d8e2e09113bcb855a54"

MODEL_PATHS = {
    "M0": {
        "schema": "stage2/output_v5_2/transfer_tuning/M0/train_matrix_manifest.json",
        "training": "stage2/output_v5_2/transfer_tuning/M0/model/model_manifest.json",
        "checkpoint": "stage2/output_v5_2/transfer_tuning/M0/model/m0_micro_tree.joblib",
        "evaluation": "stage2/output_v5_2/transfer_tuning/M0/validation_evaluation.json",
    },
    "M1": {
        "training": "stage2/output_v5_2/transfer_tuning/M1/model_manifest.json",
        "evaluation": "stage2/output_v5_2/transfer_tuning/M1/validation_evaluation.json/evaluation_manifest.json",
    },
    "M2": {
        "training": "stage2/output_v5_2/transfer_tuning/M2/model_manifest.json",
        "evaluation": "stage2/output_v5_2/transfer_tuning/M2/validation_evaluation/evaluation_manifest.json",
    },
    "M3": {
        "training": "stage2/output_v5_2/transfer_tuning/M3/model_manifest.json",
        "evaluation": "stage2/output_v5_2/transfer_tuning/M3/validation_evaluation/evaluation_manifest.json",
    },
    "M4_p25": {
        "training": "stage2/output_v5_2/transfer_tuning/M4_p25/model_manifest.json",
        "evaluation": "stage2/output_v5_2/transfer_tuning/M4_p25/validation_evaluation/evaluation_manifest.json",
    },
    "M4_p50": {
        "training": "stage2/output_v5_2/transfer_tuning/M4_p50/model_manifest.json",
        "evaluation": "stage2/output_v5_2/transfer_tuning/M4_p50/validation_evaluation/evaluation_manifest.json",
    },
    "M4_p75": {
        "training": "stage2/output_v5_2/transfer_tuning/M4_p75/model_manifest.json",
        "evaluation": "stage2/output_v5_2/transfer_tuning/M4_p75/validation_evaluation/evaluation_manifest.json",
    },
}


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Stage2V52ContractError(f"expected JSON object: {path}")
    return payload


def _payload_hash(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("artifact_sha256", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _display(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _record(
    root: Path,
    value: str | Path,
    *,
    schema_version: str | None = None,
    expected_schema: str | None = None,
    expected_payload_status: str | None = None,
) -> dict[str, Any]:
    path = _resolve(root, value)
    if not path.is_file():
        raise Stage2V52ContractError(f"B1 evidence artifact is missing: {path}")
    payload: dict[str, Any] | None = None
    if path.suffix.lower() == ".json":
        payload = _json(path)
        actual_schema = payload.get("schema_version")
        if expected_schema is not None and actual_schema != expected_schema:
            raise Stage2V52ContractError(
                f"B1 evidence schema mismatch for {path}: {actual_schema!r} != {expected_schema!r}"
            )
        if expected_payload_status is not None and payload.get("status") != expected_payload_status:
            raise Stage2V52ContractError(
                f"B1 evidence status mismatch for {path}: {payload.get('status')!r}"
            )
        schema_version = str(actual_schema or schema_version or "json.unspecified")
    if not schema_version:
        raise Stage2V52ContractError(f"B1 evidence record has no schema identity: {path}")
    result: dict[str, Any] = {
        "path": _display(root, path),
        "sha256": sha256_file(path),
        "schema_version": schema_version,
        "status": "PASS",
    }
    if payload is not None and "status" in payload:
        result["payload_status"] = payload["status"]
    return result


def verify_existing_tau_freeze(
    freeze_path: str | Path, *, repo_root: str | Path, expected_file_sha256: str
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    path = _resolve(root, freeze_path)
    if not path.is_file():
        raise Stage2V52ContractError("frozen tau artifact is missing")
    actual_sha = sha256_file(path)
    if actual_sha != expected_file_sha256:
        raise Stage2V52ContractError("frozen tau artifact differs from the config-bound file hash")
    payload = _json(path)
    if (
        payload.get("schema_version") != "stage2_v5_2_tau_freeze.1"
        or payload.get("status") != "PASS"
        or payload.get("selection_protocol") != "transfer_tuning"
        or payload.get("rolling_reselection_allowed") is not False
        or payload.get("selected_candidate") != "p25"
        or float(payload.get("selected_tau", np.nan)) != 3.0
    ):
        raise Stage2V52ContractError("tau freeze payload violates the frozen B1 decision")
    validate_embedded_hash(payload, name="B1 tau freeze")
    return {
        "schema_version": "stage2_v5_2_existing_tau_freeze_verification.1",
        "status": "PASS",
        "path": _display(root, path),
        "sha256": actual_sha,
        "selected_candidate": "p25",
        "selected_tau": 3.0,
        "write_once": True,
    }


def _relative_improvement(baseline: Mapping[str, float], candidate: Mapping[str, float]) -> dict[str, float]:
    return {
        target: (float(baseline[target]) - float(candidate[target])) / float(baseline[target])
        for target in CORE_TRANSFER_TARGETS
    }


def _scan_existing_temporal_tensors(tensor_root: Path) -> dict[str, Any]:
    audited = fallback = 0
    strict_min = fallback_min = float("inf")
    fallback_row_ids: set[str] = set()
    shard_count = 0
    for path in sorted(tensor_root.rglob("*.npz")):
        shard_count += 1
        with np.load(path, allow_pickle=False) as payload:
            pad = np.asarray(payload["pad_mask"], dtype=bool)
            ages = np.asarray(payload["feature_age_s"], dtype=np.float64)
            sentinel = np.asarray(payload["no_history_temporal_fallback"], dtype=bool)
            valid = ~pad
            if np.any(~np.isfinite(ages[valid])) or np.any(ages[valid] <= 0):
                raise Stage2V52ContractError(f"non-positive or non-finite feature age in {path}")
            strict = valid & ~sentinel
            no_history = valid & sentinel
            audited += int(valid.sum())
            fallback += int(no_history.sum())
            if np.any(strict):
                strict_min = min(strict_min, float(ages[strict].min()))
            if np.any(no_history):
                fallback_min = min(fallback_min, float(ages[no_history].min()))
                row_ids = np.asarray(payload["row_id"])
                fallback_row_ids.update(str(value) for value in row_ids[no_history].tolist())
    if shard_count == 0 or not np.isfinite(strict_min) or not np.isfinite(fallback_min):
        raise Stage2V52ContractError("existing temporal tensors do not contain both strict and NO_HISTORY tokens")
    return {
        "shard_count": shard_count,
        "audited_token_count": audited,
        "strict_observed_history_token_count": audited - fallback,
        "no_history_temporal_fallback_token_count": fallback,
        "unique_physical_no_history_row_count": len(fallback_row_ids),
        "fallback_share": fallback / audited,
        "minimum_strict_observed_feature_age_s": strict_min,
        "minimum_no_history_fallback_feature_age_s": fallback_min,
    }


def build_b1_results_summary(*, repo_root: str | Path) -> dict[str, Any]:
    """Extract frozen B1 metrics without training or inference."""
    root = Path(repo_root).resolve()
    evaluations = {
        model: _json(_resolve(root, paths["evaluation"]))
        for model, paths in MODEL_PATHS.items()
    }
    for model, payload in evaluations.items():
        expected_schema = "stage2_v5_2_m0_evaluation.1" if model == "M0" else "stage2_v5_2_evaluation.2"
        if (
            payload.get("schema_version") != expected_schema
            or payload.get("status") != "PASS"
            or payload.get("protocol_id") != "transfer_tuning"
            or payload.get("evaluation_dates") != ["20161019", "20161020"]
        ):
            raise Stage2V52ContractError(f"{model} is not a frozen passing B1 evaluation")
    reference = evaluations["M1"]
    reference_counts = {
        group: {
            target: int(reference["metrics_by_support"][group][target]["count"])
            for target in CORE_TRANSFER_TARGETS
        }
        for group in ("overall", "low", "unseen")
    }
    for model, payload in evaluations.items():
        for group in ("overall", "low", "unseen"):
            for target in CORE_TRANSFER_TARGETS:
                count = int(payload["metrics_by_support"][group][target]["count"])
                if count != reference_counts[group][target] or count <= 0:
                    raise Stage2V52ContractError(f"{model} has inconsistent/empty {group}/{target} support")
    metrics = {
        model: {
            group: {
                target: float(payload["metrics_by_support"][group][target]["mae"])
                for target in CORE_TRANSFER_TARGETS
            }
            for group in ("overall", "low", "unseen")
        }
        for model, payload in evaluations.items()
    }
    low_vs_m1 = {
        model: _relative_improvement(metrics["M1"]["low"], metrics[model]["low"])
        for model in ("M2", "M3", "M4_p25", "M4_p50", "M4_p75")
    }
    m3_to_m4 = {
        group: _relative_improvement(metrics["M3"][group], metrics["M4_p25"][group])
        for group in ("overall", "low", "unseen")
    }
    transfer_manifest_path = _resolve(
        root, "stage2/output_v5_2/transfer_shards/protocol=transfer_tuning/transfer_manifest.json"
    )
    transfer_manifest = _json(transfer_manifest_path)
    audit_path = transfer_manifest_path.parent / str(transfer_manifest["temporal_audit_path"])
    audit = _json(audit_path)
    temporal = _scan_existing_temporal_tensors(transfer_manifest_path.parent)
    if (
        temporal["audited_token_count"] != int(audit["audited_token_count"])
        or temporal["no_history_temporal_fallback_token_count"]
        != int(audit["no_history_temporal_fallback_token_count"])
        or audit.get("temporal_leakage_count") != 0
    ):
        raise Stage2V52ContractError("existing tensor scan differs from the frozen temporal audit")
    temporal.update({
        "temporal_leakage_count": 0,
        "interpretation": (
            "explicit no-history sentinel represented by a positive audit age; "
            "it carries no target-day future observation"
        ),
        "not_a_real_observation": True,
    })
    result: dict[str, Any] = {
        "schema_version": B1_SUMMARY_SCHEMA_VERSION,
        "status": "PASS",
        "scope": "PHASE_B1_ONLY",
        "protocol_id": "transfer_tuning",
        "protocol_hash": get_protocol("transfer_tuning").digest,
        "train_dates": list(get_protocol("transfer_tuning").train_dates),
        "validation_dates": list(get_protocol("transfer_tuning").validation_dates),
        "overall_unique_traversal_count": int(reference["unique_traversal_count"]),
        "target_eligible_unique_traversal_counts": reference_counts,
        "metrics": metrics,
        "low_support_relative_improvement_vs_m1": low_vs_m1,
        "selected_m4_p25_comparisons": {
            "low_vs_m1": low_vs_m1["M4_p25"],
            "low_vs_m3": _relative_improvement(metrics["M3"]["low"], metrics["M4_p25"]["low"]),
            "unseen_vs_m2": _relative_improvement(metrics["M2"]["unseen"], metrics["M4_p25"]["unseen"]),
        },
        "m3_to_m4_p25_relative_improvement": m3_to_m4,
        "pace_p50_mae": {
            model: (None if model == "M0" else float(payload["pace_p50_mae"]))
            for model, payload in evaluations.items()
        },
        "tau_freeze": {
            "selected_candidate": "p25",
            "selected_tau": 3.0,
            "interpretation": (
                "selected by the preregistered metric; p25/p50/p75 overall differences are small "
                "and do not establish statistical superiority"
            ),
        },
        "no_history_temporal_fallback": temporal,
        "scientific_conclusion": {
            "classification": "CASE_C",
            "statement": (
                "B1 shows no convincing support-aware transfer evidence; keep the protocol frozen "
                "and test temporal robustness before any adoption claim"
            ),
            "rationale": [
                "M4-p25 has only small mixed low-support changes relative to M1",
                "M4-p25 does not stably improve over M3 across low-support and unseen targets",
                "unseen differences between M4 and M2 may include shared-backbone fine-tuning and cannot be attributed to the edge gate alone",
            ],
        },
        "phase_c_authorized": False,
    }
    result["artifact_sha256"] = _payload_hash(result)
    return result


def _model_evidence(root: Path, model: str) -> dict[str, Any]:
    paths = MODEL_PATHS[model]
    training_path = _resolve(root, paths["training"])
    training = _json(training_path)
    result: dict[str, Any] = {}
    if model == "M0":
        result["training_schema_artifact"] = _record(
            root, paths["schema"], expected_schema="stage2_v5_2_m0_matrix.2"
        )
        result["training_manifest"] = _record(
            root, training_path, expected_schema="stage2_v5_2_m0_training.2", expected_payload_status="PASS"
        )
        checkpoint_path = _resolve(root, paths["checkpoint"])
        if sha256_file(checkpoint_path) != training.get("model_sha256"):
            raise Stage2V52ContractError("M0 model binary differs from its training manifest")
        result["selected_checkpoint"] = _record(root, checkpoint_path, schema_version="joblib.micro_tree.1")
        result["evaluation_manifest"] = _record(
            root, paths["evaluation"], expected_schema="stage2_v5_2_m0_evaluation.1", expected_payload_status="PASS"
        )
        return result
    result["training_manifest"] = _record(
        root, training_path, expected_schema="stage2_v5_2_training.2", expected_payload_status="PASS"
    )
    checkpoint_path = _resolve(root, str(training["selected_checkpoint_path"]))
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != training.get("selected_checkpoint_sha256"):
        raise Stage2V52ContractError(f"{model} selected checkpoint differs from its training manifest")
    result["selected_checkpoint"] = _record(root, checkpoint_path, schema_version="torch.checkpoint.1")
    evaluation_path = _resolve(root, paths["evaluation"])
    evaluation = _json(evaluation_path)
    if evaluation.get("checkpoint_sha256") != checkpoint_sha:
        raise Stage2V52ContractError(f"{model} evaluation does not bind its selected checkpoint")
    result["evaluation_manifest"] = _record(
        root, evaluation_path, expected_schema="stage2_v5_2_evaluation.2", expected_payload_status="PASS"
    )
    return result


def build_b1_evidence_bundle(
    *, repo_root: str | Path, test_evidence_path: str | Path,
    b1_execution_commit: str = B1_EXECUTION_COMMIT,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if len(b1_execution_commit) != 40:
        raise Stage2V52ContractError("B1 execution commit must be a full Git SHA")
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{b1_execution_commit}^{{commit}}"], cwd=root,
        check=False, capture_output=True,
    )
    if completed.returncode != 0:
        raise Stage2V52ContractError("B1 execution commit is not present in this repository")
    transfer_path = _resolve(
        root, "stage2/output_v5_2/transfer_shards/protocol=transfer_tuning/transfer_manifest.json"
    )
    transfer = _json(transfer_path)
    temporal_path = transfer_path.parent / str(transfer["temporal_audit_path"])
    result: dict[str, Any] = {
        "schema_version": B1_EVIDENCE_SCHEMA_VERSION,
        "status": "PASS",
        "scope": "PHASE_B1_ONLY",
        "b1_execution_commit": b1_execution_commit,
        "phase_c_authorized": False,
        "protocol": {
            "protocol_id": "transfer_tuning",
            "protocol_hash": get_protocol("transfer_tuning").digest,
            "definition": _record(root, "stage2/v5_2/protocols.py", schema_version="stage2_v5_2_protocols.python"),
        },
        "config": _record(root, "stage2/config/stage2_v5_2.json", expected_schema="stage2_v5_2_config.3"),
        "upstream": {
            "stage1_release_manifest": _record(
                root, "stage1/docs/stage1_v3_release_manifest.json", expected_schema="stage1_v3_release_manifest.1"
            ),
            "v5_1_source_checkpoint": _record(
                root, "stage2/output_v5_1/fold_1/deep_model/best_model.pt", schema_version="torch.checkpoint.1"
            ),
            "v5_1_source_model_manifest": _record(
                root, "stage2/output_v5_1/fold_1/deep_model/model_manifest.json",
                schema_version="stage2_v5_1_model_manifest.json",
            ),
            "v5_1_feature_artifact": _record(
                root, "stage2/output_v5/protocols/fold_1/tensor_shards/feature_artifacts.json",
                schema_version="stage2_v5_feature_artifacts.json",
            ),
        },
        "prepared_artifacts": {
            "train_support": _record(
                root, "stage2/output_v5_2/b0/transfer_tuning_support.json",
                expected_schema="stage2_v5_2_train_support.1",
            ),
            "static_structure": _record(
                root, "stage2/output_v5_2/b0/transfer_tuning_static.json",
                expected_schema="stage2_v5_2_static_structure.1",
            ),
            "transfer_manifest": _record(
                root, transfer_path, expected_schema="stage2_v5_2_transfer_manifest.2", expected_payload_status="PASS"
            ),
            "temporal_audit": _record(
                root, temporal_path, expected_schema="stage2_v5_2_transfer_temporal_audit.1", expected_payload_status="PASS"
            ),
        },
        "models": {model: _model_evidence(root, model) for model in MODEL_PATHS},
        "tau": {
            "metrics": _record(
                root, "stage2/output_v5_2/transfer_tuning/tau_metrics.json",
                expected_schema="stage2_v5_2_tau_evaluation.2", expected_payload_status="PASS",
            ),
            "selection": _record(
                root, "stage2/output_v5_2/transfer_tuning/tau_selection.json",
                expected_schema="stage2_v5_2_tau_selection.2", expected_payload_status="PASS",
            ),
            "freeze": _record(
                root, "stage2/output_v5_2/transfer_tuning/stage2_v5_2_tau_freeze.json",
                expected_schema="stage2_v5_2_tau_freeze.1", expected_payload_status="PASS",
            ),
        },
        "gates": {
            "b0_smoke": _record(
                root, "stage2/docs/v5_2/stage2_v5_2_phase_b0_smoke.json",
                expected_schema="stage2_v5_2_phase_b0_smoke.1", expected_payload_status="PASS",
            ),
            "metadata_audit": _record(
                root, "stage2/docs/v5_2/stage2_v5_2_input_metadata_audit.json",
                expected_schema="stage2_v5_2_phase_b0_metadata_audit.1", expected_payload_status="PASS",
            ),
            "performance": _record(
                root, "stage2/docs/v5_2/stage2_v5_2_performance.json",
                expected_schema="stage2_v5_2_performance.2", expected_payload_status="PASS",
            ),
        },
        "test_evidence": _record(
            root, test_evidence_path,
            expected_schema="stage2_v5_2_phase_b1_test_evidence.1", expected_payload_status="PASS",
        ),
    }
    result["artifact_sha256"] = _payload_hash(result)
    return result


def verify_b1_evidence_bundle(payload: Mapping[str, Any], *, repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if (
        payload.get("schema_version") != B1_EVIDENCE_SCHEMA_VERSION
        or payload.get("status") != "PASS"
        or payload.get("scope") != "PHASE_B1_ONLY"
        or payload.get("phase_c_authorized") is not False
        or payload.get("protocol", {}).get("protocol_id") != "transfer_tuning"
        or payload.get("protocol", {}).get("protocol_hash") != get_protocol("transfer_tuning").digest
        or payload.get("artifact_sha256") != _payload_hash(payload)
    ):
        raise Stage2V52ContractError("invalid Phase B1 evidence bundle identity")
    required_models = set(MODEL_PATHS)
    if set(payload.get("models", {})) != required_models:
        raise Stage2V52ContractError("Phase B1 evidence model inventory is incomplete")
    commit = str(payload.get("b1_execution_commit", ""))
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=root,
        check=False, capture_output=True,
    )
    if len(commit) != 40 or completed.returncode != 0:
        raise Stage2V52ContractError("Phase B1 execution commit does not resolve")
    records: list[Mapping[str, Any]] = []
    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            if {"path", "sha256", "schema_version", "status"}.issubset(value):
                records.append(value)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    collect(payload)
    if len(records) < 35:
        raise Stage2V52ContractError("Phase B1 evidence bundle has too few bound artifacts")
    for record in records:
        if record.get("status") != "PASS":
            raise Stage2V52ContractError(f"non-passing B1 evidence record: {record.get('path')}")
        path = _resolve(root, str(record["path"]))
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise Stage2V52ContractError(f"B1 evidence hash does not resolve: {record.get('path')}")
    test_path = _resolve(root, str(payload["test_evidence"]["path"]))
    tests = _json(test_path)
    if int(tests.get("base", {}).get("passed", 0)) < 111 or int(tests.get("gpu_v5_2", {}).get("passed", 0)) < 64:
        raise Stage2V52ContractError("B1 test evidence is below the frozen minimum")
    config_path = _resolve(root, str(payload["config"]["path"]))
    config = _json(config_path)
    freeze_sha = payload["tau"]["freeze"]["sha256"]
    if config.get("tau_freeze", {}).get("expected_file_sha256") != freeze_sha:
        raise Stage2V52ContractError("B1 evidence tau hash differs from frozen config")
    return {
        "schema_version": "stage2_v5_2_phase_b1_evidence_verification.1",
        "status": "PASS",
        "resolved_artifact_count": len(records),
        "phase_c_authorized": False,
    }
