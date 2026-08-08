"""Fail-closed v5.2 preflight, temporal, artifact, and release verification."""

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

from .contracts import Stage2V52ContractError, TOKEN_REQUIRED_COLUMNS, require_columns
from .protocols import get_protocol, validate_protocols


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def temporal_leakage_audit(tokens: pd.DataFrame) -> dict[str, Any]:
    require_columns(
        tokens.columns,
        ("decision_time", "feature_cutoff_time", "feature_age_s"),
        product="temporal leakage audit",
    )
    decision = pd.to_numeric(tokens["decision_time"], errors="coerce").to_numpy(float)
    cutoff = pd.to_numeric(tokens["feature_cutoff_time"], errors="coerce").to_numpy(float)
    reported_age = pd.to_numeric(tokens["feature_age_s"], errors="coerce").to_numpy(float)
    age = decision - cutoff
    invalid = ~np.isfinite(age) | (age <= 0) | ~np.isclose(age, reported_age, atol=1.0e-6, rtol=0)
    valid_age = age[~invalid]
    quantiles = np.quantile(valid_age, (0.01, 0.50, 0.99)) if len(valid_age) else (np.nan, np.nan, np.nan)
    return {
        "status": "PASS" if not invalid.any() else "FAIL",
        "temporal_leakage_count": int(invalid.sum()),
        "minimum_feature_age_s": float(np.min(valid_age)) if len(valid_age) else None,
        "p01_feature_age_s": float(quantiles[0]) if len(valid_age) else None,
        "p50_feature_age_s": float(quantiles[1]) if len(valid_age) else None,
        "p99_feature_age_s": float(quantiles[2]) if len(valid_age) else None,
    }


def verify_artifact_payload(payload: Mapping[str, Any], *, artifact_type: str) -> None:
    if payload.get("evaluation_rows_used") != 0:
        raise Stage2V52ContractError(f"{artifact_type} used evaluation rows")
    if artifact_type in {"support", "static_structure"} and payload.get("fit_scope", "train_only") != "train_only":
        raise Stage2V52ContractError(f"{artifact_type} is not Train-only")
    if artifact_type == "micro_cdf" and payload.get("fit_split") != "train":
        raise Stage2V52ContractError("micro CDF is not Train-only")


def preflight(
    *,
    config_path: str | Path,
    protocol_id: str,
    source_checkpoint_path: str | Path,
    feature_artifact_path: str | Path,
) -> dict[str, Any]:
    validate_protocols()
    protocol = get_protocol(protocol_id)
    required = [Path(config_path), Path(source_checkpoint_path), Path(feature_artifact_path)]
    missing = [path.as_posix() for path in required if not path.is_file()]
    return {
        "schema_version": "stage2_v5_2_preflight.1",
        "status": "PASS" if not missing else "FAIL",
        "protocol_id": protocol_id,
        "protocol_hash": protocol.digest,
        "missing_files": missing,
        "file_hashes": {path.as_posix(): sha256_file(path) for path in required if path.is_file()},
        "experiments_run": False,
    }


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def build_release_manifest(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    protocol_id: str,
    source_checkpoint_path: str | Path,
    feature_artifact_path: str | Path,
    support_artifact_path: str | Path,
    static_artifact_path: str | Path,
    tau_artifact_path: str | Path,
    micro_cdf_path: str | Path,
    stage1_release: Mapping[str, Any],
    output_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    bound = {
        "config_sha256": sha256_file(config_path),
        "v5_1_source_checkpoint_sha256": sha256_file(source_checkpoint_path),
        "feature_schema_sha256": sha256_file(feature_artifact_path),
        "support_artifact_sha256": sha256_file(support_artifact_path),
        "static_structure_artifact_sha256": sha256_file(static_artifact_path),
        "tau_selection_sha256": sha256_file(tau_artifact_path),
        "micro_cdf_artifact_sha256": sha256_file(micro_cdf_path),
        "protocol_sha256": get_protocol(protocol_id).digest,
    }
    stage0_release = stage1_release.get("stage0_release", {})
    if not stage1_release.get("tag") or not stage0_release.get("tag"):
        raise Stage2V52ContractError("release manifest requires frozen Stage 1 and Stage 0 identities")
    return {
        "schema_version": "stage2_v5_2_release_manifest.2",
        "git_commit": _git_head(root),
        "protocol_id": protocol_id,
        **bound,
        "v5_1_source_checkpoint_path": Path(source_checkpoint_path).as_posix(),
        "stage1_release": dict(stage1_release),
        "stage0_release": dict(stage0_release),
        "output_hashes": {name: sha256_file(path) for name, path in output_paths.items()},
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
    return {
        "schema_version": "stage2_v5_2_phase_b_verification.1",
        "status": temporal["status"],
        "temporal": temporal,
        "phase_allowed_after_pass": "B0_B1_ONLY",
        "full_rolling_allowed": False,
    }
