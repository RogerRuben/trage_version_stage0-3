"""Development and final release freezes for the rolling-origin v5 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def freeze_development(*, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    docs = root / "stage2/docs/v5"
    if any((docs / "protocols" / fold / "protocol_summary.json").exists() for fold in ("fold_1", "fold_2", "fold_3")):
        raise RuntimeError("refusing to freeze development after rolling evaluation exists")
    development = _json(docs / "protocols/development/protocol_summary.json")
    ablation = development.get("development_ablation")
    if development.get("status") != "PASS" or not isinstance(ablation, dict) or ablation.get("status") != "PASS":
        raise RuntimeError("development model and ablations must finish before freeze")
    split = _json(docs / "stage2_v5_split_freeze.json")
    result = {
        "schema_version": "stage2_v5_development_freeze.2",
        "freeze_status": "FROZEN_BEFORE_ROLLING_ORIGIN",
        "implementation_commit": _git_head(root),
        "base_stage2_tag": "stage2-v4-final",
        "base_stage2_commit": "70cb70265cbb95e5fc9981024a554de28ee2be85",
        "config_sha256": _sha256(root / "stage2/config/stage2_v5.json"),
        "split_freeze_sha256": _sha256(docs / "stage2_v5_split_freeze.json"),
        "development_model_id": development["model_id"],
        "selected_history_mode": ablation["selected_history_mode"],
        "rolling_fold_definition": split["rolling_folds"],
        "percentile_targets_used_for_selection": False,
        "upstream_rebuild_performed": False,
        "post_rolling_tuning_allowed": False,
    }
    (docs / "stage2_v5_development_freeze.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def freeze_release(*, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    docs = root / "stage2/docs/v5"
    verification = _json(docs / "stage2_v5_final_verification.json")
    if verification.get("stage3_admission_status") != "READY_FOR_STAGE3":
        raise RuntimeError("final verification does not admit Stage 3")
    development = _json(docs / "stage2_v5_development_freeze.json")
    rolling = _json(docs / "stage2_v5_rolling_origin_summary.json")
    legacy = _json(docs / "protocols/legacy/protocol_summary.json")
    service = _json(docs / "stage2_v5_service_time_target_audit.json")
    result = {
        "schema_version": "stage2_v5_release_manifest.2",
        "release_status": "FROZEN",
        "base_stage2_tag": "stage2-v4-final",
        "base_stage2_commit": "70cb70265cbb95e5fc9981024a554de28ee2be85",
        "v5_implementation_commit": development["implementation_commit"],
        "release_source_commit": _git_head(root),
        "v5_config_sha": development["config_sha256"],
        "split_freeze_sha": development["split_freeze_sha256"],
        "service_time_target_contract": service["target_contract"],
        "performance_gate_status": verification["performance_gate_status"],
        "performance_benchmark_sha": _sha256(docs / "stage2_v5_performance_benchmarks.json"),
        "profile_sha": _sha256(docs / "performance_profile_hotspots.txt"),
        "development_model_id": development["development_model_id"],
        "rolling_model_ids": rolling["protocol_model_ids"],
        "legacy_model_id": legacy["model_id"],
        "rolling_fold_definition": development["rolling_fold_definition"],
        "legacy_test_identity": {"dates": ["20161031"], "role": "frozen_legacy_benchmark", "untouched": False},
        "stage3_admission_status": verification["stage3_admission_status"],
        "engineering_status": verification["engineering_status"],
        "temporal_contract_status": verification["temporal_contract_status"],
        "rolling_origin_status": verification["rolling_origin_status"],
        "post_rolling_tuning_count": 0,
        "upstream_rebuild_performed": False,
    }
    (docs / "stage2_v5_release_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze-development", "freeze-release"))
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    result = freeze_development(repo_root=args.repo_root) if args.command == "freeze-development" else freeze_release(repo_root=args.repo_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
