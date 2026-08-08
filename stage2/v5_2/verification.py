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
from .protocols import get_protocol, validate_protocols


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
    tau_artifact_path: str | Path, micro_cdf_path: str | Path,
    transfer_manifest_path: str | Path, training_manifest_path: str | Path,
    selected_checkpoint_path: str | Path, evaluation_manifest_path: str | Path,
    stage1_release: Mapping[str, Any], output_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    stage0_release = stage1_release.get("stage0_release", {})
    if (
        stage1_release.get("schema_version") != "stage1_v3_release_manifest.1"
        or not stage1_release.get("release_tag") or not stage0_release.get("tag")
    ):
        raise Stage2V52ContractError("release manifest requires actual Stage 1 release_tag and Stage 0 tag")
    artifacts = {
        "config": config_path, "v5_1_source_checkpoint": source_checkpoint_path,
        "v5_1_feature_artifact": feature_artifact_path,
        "v5_1_source_model_manifest": source_model_manifest_path,
        "v5_1_source_config": source_config_path, "support_artifact": support_artifact_path,
        "static_structure_artifact": static_artifact_path, "tau_selection": tau_artifact_path,
        "micro_cdf_artifact": micro_cdf_path, "transfer_manifest": transfer_manifest_path,
        "training_manifest": training_manifest_path, "selected_checkpoint": selected_checkpoint_path,
        "evaluation_manifest": evaluation_manifest_path,
    }
    missing = [str(path) for path in artifacts.values() if not Path(path).is_file()]
    if missing:
        raise Stage2V52ContractError(f"release artifacts are missing: {missing}")
    missing_outputs = sorted(REQUIRED_RELEASE_OUTPUTS - set(output_paths))
    if missing_outputs:
        raise Stage2V52ContractError(f"release outputs manifest is incomplete: {missing_outputs}")
    return {
        "schema_version": "stage2_v5_2_release_manifest.3", "git_commit": _git_head(root),
        "protocol_id": protocol_id, "protocol_sha256": get_protocol(protocol_id).digest,
        "artifact_hashes": {name: sha256_file(path) for name, path in artifacts.items()},
        "artifact_paths": {name: Path(path).as_posix() for name, path in artifacts.items()},
        "stage1_release": dict(stage1_release), "stage0_release": dict(stage0_release),
        "output_hashes": {name: sha256_file(path) for name, path in output_paths.items()},
    }


def verify_final_gate_bundle(payload: Mapping[str, Any]) -> dict[str, Any]:
    gates = payload.get("required_gates")
    if not isinstance(gates, Mapping) or set(gates) != set(FINAL_REQUIRED_GATES):
        missing = sorted(set(FINAL_REQUIRED_GATES) - set(gates or {}))
        extra = sorted(set(gates or {}) - set(FINAL_REQUIRED_GATES))
        return {"status": "FAIL", "stage2_status": "NOT_READY", "missing_gates": missing, "extra_gates": extra}
    passed = all(gates[name] == "PASS" for name in FINAL_REQUIRED_GATES)
    return {"status": "PASS" if passed else "FAIL", "stage2_status": "READY_FOR_AV_ROUTE_SUITABILITY_STAGE" if passed else "NOT_READY", "required_gate_names": list(FINAL_REQUIRED_GATES)}


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
