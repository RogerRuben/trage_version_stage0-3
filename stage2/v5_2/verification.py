"""Fail-closed v5.2 preflight, artifact, release, and final-gate verification."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    CONFIG_SCHEMA_VERSION, Stage2V52ContractError, TOKEN_REQUIRED_COLUMNS, require_columns,
)
from .feature_binding import bind_v51_source_model
from .protocols import get_protocol, protocol_role_dates, validate_protocols


FINAL_REQUIRED_GATES = (
    "phase_b0_metadata_schema_audit",
    "source_model_binding",
    "transfer_shard_integrity",
    "temporal_leakage_zero",
    "unique_traversal_evaluation",
    "m0_baseline_complete",
    "m1_baseline_complete",
    "spatial_adoption",
    "temporal_adoption",
    "pace_guard",
    "rolling_origin_complete",
    "legacy_benchmark_complete",
    "product_schema",
    "stage3_contract",
    "performance",
    "reproducibility",
)
FINAL_GATE_SPECS: dict[str, dict[str, Any]] = {
    "phase_b0_metadata_schema_audit": {"schemas": ("stage2_v5_2_phase_b0_metadata_audit.1",), "protocol": "release", "models": (None,), "dates": "protocol_scope"},
    "source_model_binding": {"schemas": ("stage2_v5_2_preflight.2",), "protocol": "release", "models": (None,), "dates": "none"},
    "transfer_shard_integrity": {"schemas": ("stage2_v5_2_transfer_manifest.2",), "protocol": "release", "models": (None,), "dates": "none"},
    "temporal_leakage_zero": {"schemas": ("stage2_v5_2_transfer_temporal_audit.1",), "protocol": "release", "models": (None,), "dates": "none"},
    "unique_traversal_evaluation": {"schemas": ("stage2_v5_2_evaluation.2",), "protocol": "release", "models": ("M1", "M4", "M5"), "dates": "canonical_role"},
    "m0_baseline_complete": {"schemas": ("stage2_v5_2_m0_evaluation.1",), "protocol": "release", "models": ("M0",), "dates": "canonical_role"},
    "m1_baseline_complete": {"schemas": ("stage2_v5_2_evaluation.2",), "protocol": "release", "models": ("M1",), "dates": "canonical_role"},
    "spatial_adoption": {"schemas": ("stage2_v5_2_rolling_spatial_adoption.1",), "protocol": "rolling", "models": (None,), "dates": "rolling_evaluation"},
    "temporal_adoption": {"schemas": ("stage2_v5_2_adoption.2",), "protocol": "rolling", "models": (None,), "dates": "rolling_evaluation"},
    "pace_guard": {"schemas": ("stage2_v5_2_pace_guard.1",), "protocol": "rolling", "models": (None,), "dates": "rolling_evaluation"},
    "rolling_origin_complete": {"schemas": ("stage2_v5_2_rolling_complete.1",), "protocol": "rolling", "models": (None,), "dates": "rolling_evaluation"},
    "legacy_benchmark_complete": {"schemas": ("stage2_v5_2_evaluation.2",), "protocol": "legacy", "models": ("M1", "M4", "M5"), "dates": "legacy"},
    "product_schema": {"schemas": ("stage2_v5_2_product_schema_verification.1",), "protocol": "release", "models": (None,), "dates": "protocol_scope"},
    "stage3_contract": {"schemas": ("stage2_v5_2_stage3_contract_verification.1",), "protocol": "global", "models": (None,), "dates": "none"},
    "performance": {"schemas": ("stage2_v5_2_performance.2",), "protocol": "global", "models": (None,), "dates": "none"},
    "reproducibility": {"schemas": ("stage2_v5_2_reproducibility.1",), "protocol": "release", "models": (None,), "dates": "protocol_scope"},
}
REQUIRED_RELEASE_OUTPUTS = frozenset({
    "micro_condition_tokens_manifest", "original_route_micro_conditions_manifest",
    "static_route_complexity_manifest", "rolling_results", "performance_report",
    "final_verification", "stage3_contract",
})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def temporal_leakage_audit(tokens: pd.DataFrame) -> dict[str, Any]:
    require_columns(tokens.columns, ("decision_time", "feature_cutoff_time", "feature_age_s"), product="temporal leakage audit")
    decision = pd.to_numeric(tokens["decision_time"], errors="coerce").to_numpy(float)
    cutoff = pd.to_numeric(tokens["feature_cutoff_time"], errors="coerce").to_numpy(float)
    reported_age = pd.to_numeric(tokens["feature_age_s"], errors="coerce").to_numpy(float)
    age = decision - cutoff
    invalid = ~np.isfinite(age) | (age <= 0) | ~np.isclose(age, reported_age, atol=1.0e-6, rtol=0)
    valid_age = age[~invalid]
    quantiles = np.quantile(valid_age, (0.01, 0.50, 0.99)) if len(valid_age) else (np.nan,) * 3
    return {
        "status": "PASS" if not invalid.any() and len(valid_age) else "FAIL",
        "temporal_leakage_count": int(invalid.sum()),
        "audited_row_count": int(len(age)),
        "minimum_feature_age_s": float(np.min(valid_age)) if len(valid_age) else None,
        "p01_feature_age_s": float(quantiles[0]) if len(valid_age) else None,
        "p50_feature_age_s": float(quantiles[1]) if len(valid_age) else None,
        "p99_feature_age_s": float(quantiles[2]) if len(valid_age) else None,
    }


def verify_artifact_payload(payload: Mapping[str, Any], *, artifact_type: str) -> None:
    rules = {
        "support": {"fit_scope": "train_only", "evaluation_support_used": False},
        "static_structure": {"fit_scope": "train_only", "evaluation_rows_used": 0},
        "micro_cdf": {"fit_split": "train", "evaluation_rows_used": 0},
        "m0_matrix": {"fit_scope": "train_only", "evaluation_rows_used": 0},
    }
    if artifact_type not in rules:
        raise Stage2V52ContractError(f"unknown artifact type: {artifact_type}")
    mismatches = {key: (payload.get(key), expected) for key, expected in rules[artifact_type].items() if payload.get(key) != expected}
    if mismatches:
        raise Stage2V52ContractError(f"{artifact_type} provenance violation: {mismatches}")
    if artifact_type == "support":
        if payload.get("schema_version") != "stage2_v5_2_train_support.1":
            raise Stage2V52ContractError("support artifact schema is invalid")
        quantiles = payload.get("positive_quantiles", {})
        candidates = payload.get("tau_candidates", ())
        candidate_table = payload.get("tau_candidate_table", {})
        if (
            not isinstance(quantiles, Mapping)
            or any(name not in quantiles for name in ("p25", "p50", "p75"))
            or [float(value) for value in candidates]
            != [float(quantiles[name]) for name in ("p25", "p50", "p75")]
            or not isinstance(candidate_table, Mapping)
            or list(candidate_table) != ["p25", "p50", "p75"]
            or [float(candidate_table[name]) for name in ("p25", "p50", "p75")]
            != [float(quantiles[name]) for name in ("p25", "p50", "p75")]
        ):
            raise Stage2V52ContractError("support artifact tau candidates are not P25/P50/P75")
        canonical = dict(payload)
        observed_hash = str(canonical.pop("artifact_sha256", ""))
        expected_hash = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if observed_hash != expected_hash:
            raise Stage2V52ContractError("support artifact canonical hash mismatch")


def preflight(
    *, config_path: str | Path, protocol_id: str, source_checkpoint_path: str | Path,
    feature_artifact_path: str | Path, source_model_manifest_path: str | Path,
    source_config_path: str | Path,
) -> dict[str, Any]:
    validate_protocols()
    protocol = get_protocol(protocol_id)
    required = [Path(config_path), Path(source_checkpoint_path), Path(feature_artifact_path), Path(source_model_manifest_path), Path(source_config_path)]
    missing = [path.as_posix() for path in required if not path.is_file()]
    if missing:
        return {"schema_version": "stage2_v5_2_preflight.2", "status": "FAIL", "protocol_id": protocol_id, "missing_files": missing, "experiments_run": False}
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Stage2V52ContractError("v5.2 config schema is not the frozen Phase A.2 schema")
    source, feature = bind_v51_source_model(
        protocol_id=protocol_id, feature_artifact_path=feature_artifact_path,
        source_checkpoint_path=source_checkpoint_path,
        source_model_manifest_path=source_model_manifest_path,
        source_config_path=source_config_path, backbone_kwargs=config["backbone"],
    )
    return {
        "schema_version": "stage2_v5_2_preflight.2", "status": "PASS",
        "protocol_id": protocol_id, "protocol_hash": protocol.digest,
        "source_model_binding": source.to_payload(), "feature_binding": feature.to_payload(),
        "missing_files": [], "file_hashes": {path.as_posix(): sha256_file(path) for path in required},
        "experiments_run": False,
    }


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def build_release_manifest(
    *, repo_root: str | Path, config_path: str | Path, protocol_id: str,
    source_checkpoint_path: str | Path, feature_artifact_path: str | Path,
    source_model_manifest_path: str | Path, source_config_path: str | Path,
    support_artifact_path: str | Path, static_artifact_path: str | Path,
    tau_freeze_artifact_path: str | Path, micro_cdf_path: str | Path,
    transfer_manifest_path: str | Path, training_manifest_path: str | Path,
    selected_checkpoint_path: str | Path, evaluation_manifest_path: str | Path,
    stage1_release_manifest_path: str | Path, output_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    stage1_release_path = Path(stage1_release_manifest_path)
    if not stage1_release_path.is_file():
        raise Stage2V52ContractError("Stage 1 release manifest path does not exist")
    stage1_release = json.loads(stage1_release_path.read_text(encoding="utf-8"))
    if not isinstance(stage1_release, dict):
        raise Stage2V52ContractError("Stage 1 release manifest must be a JSON object")
    stage1_release_sha = sha256_file(stage1_release_path)
    stage0_release = stage1_release.get("stage0_release", {})
    stage0_commit = stage0_release.get("git_commit", stage0_release.get("commit"))
    if (
        stage1_release.get("schema_version") != "stage1_v3_release_manifest.1"
        or not stage1_release.get("release_tag")
        or ("engineering_status" in stage1_release and stage1_release.get("engineering_status") != "PASS")
        or not stage0_release.get("tag") or not stage0_commit
    ):
        raise Stage2V52ContractError(
            "release manifest requires the Stage 1 release tag and Stage 0 tag+commit"
        )
    artifacts = {
        "config": config_path, "v5_1_source_checkpoint": source_checkpoint_path,
        "v5_1_feature_artifact": feature_artifact_path,
        "v5_1_source_model_manifest": source_model_manifest_path,
        "v5_1_source_config": source_config_path, "support_artifact": support_artifact_path,
        "static_structure_artifact": static_artifact_path, "tau_freeze": tau_freeze_artifact_path,
        "micro_cdf_artifact": micro_cdf_path, "transfer_manifest": transfer_manifest_path,
        "training_manifest": training_manifest_path, "selected_checkpoint": selected_checkpoint_path,
        "evaluation_manifest": evaluation_manifest_path,
        "stage1_release_manifest": stage1_release_path,
    }
    missing = [str(path) for path in artifacts.values() if not Path(path).is_file()]
    if missing:
        raise Stage2V52ContractError(f"release artifacts are missing: {missing}")
    missing_outputs = sorted(REQUIRED_RELEASE_OUTPUTS - set(output_paths))
    if missing_outputs:
        raise Stage2V52ContractError(f"release outputs manifest is incomplete: {missing_outputs}")
    missing_output_files = sorted(str(path) for path in output_paths.values() if not Path(path).is_file())
    if missing_output_files:
        raise Stage2V52ContractError(f"release output files are missing: {missing_output_files}")
    source_manifest = json.loads(Path(source_model_manifest_path).read_text(encoding="utf-8"))
    transfer = json.loads(Path(transfer_manifest_path).read_text(encoding="utf-8"))
    training = json.loads(Path(training_manifest_path).read_text(encoding="utf-8"))
    evaluation = json.loads(Path(evaluation_manifest_path).read_text(encoding="utf-8"))
    tau = json.loads(Path(tau_freeze_artifact_path).read_text(encoding="utf-8"))
    protocol = get_protocol(protocol_id)
    selected_sha = sha256_file(selected_checkpoint_path)
    training_sha = sha256_file(training_manifest_path)
    transfer_sha = sha256_file(transfer_manifest_path)
    source_checkpoint_sha = sha256_file(source_checkpoint_path)
    tau_freeze_sha = sha256_file(tau_freeze_artifact_path)
    relation_checks = {
        "source_manifest_schema_status": source_manifest.get("schema_version") == "stage2_v5_rc_mstnet.1" and source_manifest.get("status") == "PASS",
        "source_checkpoint_bound": source_manifest.get("checkpoint_sha256") == source_checkpoint_sha,
        "transfer_protocol_bound": transfer.get("protocol_id") == protocol_id and transfer.get("protocol_hash") == protocol.digest,
        "transfer_stage1_release_bound": transfer.get("stage1_release_manifest_sha256") == stage1_release_sha,
        "training_protocol_bound": training.get("schema_version") == "stage2_v5_2_training.2" and training.get("status") == "PASS" and training.get("protocol_id") == protocol_id and training.get("protocol_hash") == protocol.digest,
        "training_selected_checkpoint_bound": training.get("selected_checkpoint_sha256") == selected_sha,
        "training_transfer_bound": training.get("tensor_manifest_sha256") == transfer_sha,
        "training_source_bound": training.get("source", {}).get("source_checkpoint_sha256") == source_checkpoint_sha and training.get("source", {}).get("feature_artifact_sha256") == sha256_file(feature_artifact_path),
        "training_tau_freeze_bound": training.get("constructor", {}).get("support_tau_provenance", {}).get("tau_freeze_artifact_sha256") == tau_freeze_sha,
        "evaluation_protocol_bound": evaluation.get("schema_version") == "stage2_v5_2_evaluation.2" and evaluation.get("status") == "PASS" and evaluation.get("protocol_id") == protocol_id,
        "evaluation_checkpoint_bound": evaluation.get("checkpoint_sha256") == selected_sha,
        "evaluation_training_bound": evaluation.get("training_manifest_sha256") == training_sha,
        "evaluation_transfer_bound": evaluation.get("tensor_manifest_sha256") == transfer_sha,
        "tau_frozen_transfer_tuning": tau.get("schema_version") == "stage2_v5_2_tau_freeze.1" and tau.get("status") == "PASS" and tau.get("selection_protocol") == "transfer_tuning" and tau.get("rolling_reselection_allowed") is False,
    }
    if not all(relation_checks.values()):
        failed = sorted(name for name, passed in relation_checks.items() if not passed)
        raise Stage2V52ContractError(f"release artifact relationship checks failed: {failed}")
    return {
        "schema_version": "stage2_v5_2_release_manifest.3", "git_commit": _git_head(root),
        "protocol_id": protocol_id, "protocol_sha256": protocol.digest,
        "artifact_hashes": {name: sha256_file(path) for name, path in artifacts.items()},
        "artifact_paths": {name: Path(path).as_posix() for name, path in artifacts.items()},
        "stage1_release_manifest_sha256": stage1_release_sha,
        "stage1_release": dict(stage1_release), "stage0_release": dict(stage0_release),
        "output_hashes": {name: sha256_file(path) for name, path in output_paths.items()},
        "relationship_verification": {"status": "PASS", "checks": relation_checks},
    }


def _report_dates(report: Mapping[str, Any]) -> list[str]:
    return [str(value) for value in report.get("evaluation_dates", report.get("dates", [])) or []]


def _gate_policy_matches(
    name: str, report: Mapping[str, Any], release: Mapping[str, Any], release_context: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    spec = FINAL_GATE_SPECS[name]
    failures: list[str] = []
    if report.get("schema_version") not in spec["schemas"]:
        failures.append("schema_version")
    protocol_policy = spec["protocol"]
    protocol_id = report.get("protocol_id")
    release_protocol = str(release["protocol_id"])
    if protocol_policy == "release" and protocol_id != release_protocol:
        failures.append("protocol_id")
    elif protocol_policy == "rolling" and protocol_id != "rolling_origin_fold_1_2_3":
        failures.append("protocol_id")
    elif protocol_policy == "legacy" and protocol_id != "legacy_31":
        failures.append("protocol_id")
    elif protocol_policy == "global" and protocol_id not in (None, "global"):
        failures.append("protocol_id")
    if report.get("model_id") not in spec["models"]:
        failures.append("model_id")
    dates = _report_dates(report)
    if spec["dates"] == "protocol_scope":
        expected = sorted({date for values in protocol_role_dates(release_protocol).values() for date in values})
        if sorted(dates) != expected:
            failures.append("evaluation_dates")
    elif spec["dates"] == "canonical_role":
        role = str(report.get("role", ""))
        expected = protocol_role_dates(str(protocol_id)).get(role)
        if expected is None or dates != list(expected):
            failures.append("evaluation_dates")
    elif spec["dates"] == "rolling_evaluation":
        expected = sorted({date for index in range(1, 4) for date in get_protocol(f"fold_{index}").evaluation_dates})
        if sorted(dates) != expected:
            failures.append("evaluation_dates")
    elif spec["dates"] == "legacy" and dates != ["20161031"]:
        failures.append("evaluation_dates")
    elif spec["dates"] == "none" and dates:
        failures.append("evaluation_dates")
    context_fields = {
        "protocol_hash": release_context.get("protocol_sha256"),
        "stage1_release_manifest_sha256": release_context.get("stage1_release_manifest_sha256"),
        "git_commit": release_context.get("git_commit"),
    }
    for field, expected in context_fields.items():
        if field in report and report.get(field) != expected:
            failures.append(field)
    return not failures, sorted(set(failures))


def verify_final_gate_bundle(payload: Mapping[str, Any]) -> dict[str, Any]:
    release_path = Path(str(payload.get("release_manifest_path", "")))
    release_sha = payload.get("release_manifest_sha256")
    release_context = payload.get("release_context")
    if not release_path.is_file() or sha256_file(release_path) != release_sha or not isinstance(release_context, Mapping):
        return {"status": "FAIL", "stage2_status": "NOT_READY", "reason": "release_manifest_or_context_unbound"}
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("schema_version") != "stage2_v5_2_release_manifest.3":
        return {"status": "FAIL", "stage2_status": "NOT_READY", "reason": "release_manifest_schema_invalid"}
    expected_context = {
        "git_commit": release.get("git_commit"),
        "protocol_id": release.get("protocol_id"),
        "protocol_sha256": release.get("protocol_sha256"),
        "stage1_release_manifest_sha256": release.get("stage1_release_manifest_sha256"),
        "tau_freeze_sha256": release.get("artifact_hashes", {}).get("tau_freeze"),
        "transfer_manifest_sha256": release.get("artifact_hashes", {}).get("transfer_manifest"),
        "selected_checkpoint_sha256": release.get("artifact_hashes", {}).get("selected_checkpoint"),
    }
    if dict(release_context) != expected_context:
        return {"status": "FAIL", "stage2_status": "NOT_READY", "reason": "release_context_mismatch"}
    gates = payload.get("required_gates")
    if not isinstance(gates, Mapping) or set(gates) != set(FINAL_REQUIRED_GATES):
        missing = sorted(set(FINAL_REQUIRED_GATES) - set(gates or {}))
        extra = sorted(set(gates or {}) - set(FINAL_REQUIRED_GATES))
        return {"status": "FAIL", "stage2_status": "NOT_READY", "missing_gates": missing, "extra_gates": extra}
    required_reference_fields = {"report_path", "report_sha256"}
    results: dict[str, Any] = {}
    reports: dict[str, dict[str, Any]] = {}
    for name in FINAL_REQUIRED_GATES:
        reference = gates[name]
        if not isinstance(reference, Mapping):
            results[name] = {"status": "FAIL", "reason": "gate_is_not_a_report_reference"}
            continue
        missing_reference = sorted(required_reference_fields - set(reference))
        path = Path(str(reference.get("report_path", "")))
        if missing_reference or not path.is_file():
            results[name] = {"status": "FAIL", "missing_reference_fields": missing_reference, "report_exists": path.is_file()}
            continue
        if sha256_file(path) != reference.get("report_sha256"):
            results[name] = {"status": "FAIL", "reason": "report_hash_mismatch"}
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            results[name] = {"status": "FAIL", "reason": "report_not_object"}
            continue
        reports[name] = report
        metadata_matches, policy_failures = _gate_policy_matches(name, report, release, release_context)
        report_status = report.get("verification_status", report.get("status"))
        passed = metadata_matches and report_status == "PASS"
        results[name] = {
            "status": "PASS" if passed else "FAIL",
            "metadata_matches": metadata_matches,
            "policy_failures": policy_failures,
            "derived_report_status": report_status,
        }
    spatial = reports.get("spatial_adoption")
    temporal_reference = gates.get("temporal_adoption")
    if isinstance(spatial, Mapping) and isinstance(temporal_reference, Mapping):
        if not isinstance(spatial.get("adopt"), bool):
            results["spatial_adoption"] = {
                "status": "FAIL", "reason": "spatial_decision_has_no_boolean_adopt",
            }
            results["temporal_adoption"] = {
                "status": "FAIL", "reason": "temporal_branch_cannot_be_derived",
            }
        elif spatial.get("adopt") is False:
            stop_rule_ok = (
                temporal_reference.get("status") == "NOT_APPLICABLE_BY_FROZEN_STOP_RULE"
                and temporal_reference.get("report_sha256") == gates["spatial_adoption"].get("report_sha256")
            )
            results["temporal_adoption"] = {
                "status": "PASS" if stop_rule_ok else "FAIL",
                "derived_report_status": "NOT_APPLICABLE_BY_FROZEN_STOP_RULE" if stop_rule_ok else "INVALID_STOP_RULE",
            }
        elif spatial.get("adopt") is True and temporal_reference.get("status") == "NOT_APPLICABLE_BY_FROZEN_STOP_RULE":
            results["temporal_adoption"] = {"status": "FAIL", "reason": "temporal_gate_required_after_spatial_adoption"}
    passed = all(results.get(name, {}).get("status") == "PASS" for name in FINAL_REQUIRED_GATES)
    return {
        "status": "PASS" if passed else "FAIL",
        "stage2_status": "READY_FOR_AV_ROUTE_SUITABILITY_STAGE" if passed else "NOT_READY",
        "required_gate_names": list(FINAL_REQUIRED_GATES),
        "gate_results": results,
        "negative_transfer_stop_rule_supported": True,
        "release_manifest_sha256": release_sha,
        "release_context": dict(release_context),
    }


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def verify_phase_b(tokens: pd.DataFrame) -> dict[str, Any]:
    require_columns(tokens.columns, TOKEN_REQUIRED_COLUMNS, product="Phase B token product")
    temporal = temporal_leakage_audit(tokens)
    return {"schema_version": "stage2_v5_2_phase_b_verification.2", "status": temporal["status"], "temporal": temporal, "phase_allowed_after_pass": "B0_B1_ONLY", "full_rolling_allowed": False}


def verify_one_train_one_validation_bucket(
    *, train_traversal_path: str | Path, train_label_path: str | Path,
    validation_traversal_path: str | Path, validation_label_path: str | Path,
) -> dict[str, Any]:
    """Bounded Phase B0 data-contract PoC; never scans more than two explicit buckets."""
    pairs = {
        "train": (Path(train_traversal_path), Path(train_label_path)),
        "validation": (Path(validation_traversal_path), Path(validation_label_path)),
    }
    required = {"order_id", "traversal_id"}
    results: dict[str, Any] = {}
    for role, (traversal_path, label_path) in pairs.items():
        if not traversal_path.is_file() or not label_path.is_file():
            raise Stage2V52ContractError(f"Phase B0 {role} bucket files are missing")
        traversal = pd.read_parquet(traversal_path)
        labels = pd.read_parquet(label_path)
        if not required <= set(traversal) or not required <= set(labels):
            raise Stage2V52ContractError(f"Phase B0 {role} bucket lacks traversal identity")
        if traversal.empty or labels.empty:
            raise Stage2V52ContractError(f"Phase B0 {role} bucket is empty")
        if traversal.duplicated(list(required)).any() or labels.duplicated(list(required)).any():
            raise Stage2V52ContractError(f"Phase B0 {role} bucket identity is not unique")
        joined = traversal.loc[:, list(required)].merge(
            labels.loc[:, list(required)], on=list(required), how="outer", indicator=True,
        )
        mismatch = int(joined["_merge"].ne("both").sum())
        if mismatch:
            raise Stage2V52ContractError(f"Phase B0 {role} traversal/label reconciliation failed")
        results[role] = {
            "traversal_path": traversal_path.as_posix(), "traversal_sha256": sha256_file(traversal_path),
            "label_path": label_path.as_posix(), "label_sha256": sha256_file(label_path),
            "unique_traversal_count": int(len(joined)), "reconciliation_mismatch_count": mismatch,
        }
    return {
        "schema_version": "stage2_v5_2_phase_b0_one_bucket_correctness.1",
        "status": "PASS", "scope": "one_explicit_train_bucket_plus_one_explicit_validation_bucket",
        "full_protocol_scanned": False, "buckets": results,
    }
