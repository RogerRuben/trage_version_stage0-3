"""Development freeze and final release manifest generation for Stage 2 v5."""

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
    if (docs / "stage2_v5_final_test_results.json").exists():
        raise RuntimeError("refusing to freeze development after final-test results exist")
    verification = _json(docs / "stage2_v5_final_verification.json")
    if verification.get("stage3_admission_status") != "READY_FOR_ROUTE_SCENARIO_PROTOTYPE":
        raise RuntimeError("development verification is not prototype-ready")
    model = _json(root / "stage2/output_v5/deep_model/model_manifest.json")
    scenario = _json(docs / "stage2_v5_scenario_selection.json")
    split = _json(docs / "stage2_v5_split_freeze.json")
    upstream = _json(docs / "stage2_v5_final_upstream_plan.json")
    result = {
        "schema_version": "stage2_v5_development_freeze.1",
        "freeze_status": "FROZEN_BEFORE_FINAL_TEST",
        "implementation_commit": _git_head(root),
        "base_stage2_tag": "stage2-v4-final",
        "base_stage2_commit": "70cb70265cbb95e5fc9981024a554de28ee2be85",
        "config_sha256": _sha256(root / "stage2/config/stage2_v5.json"),
        "split_freeze_sha256": _sha256(docs / "stage2_v5_split_freeze.json"),
        "split": split["split"],
        "selection_model_id": model["model_id"],
        "distribution_model_id": model["model_id"],
        "checkpoint_sha256": model["checkpoint_sha256"],
        "selected_history_mode": "horizon_gate",
        "scenario_generator_id": scenario["generator_id"],
        "scenario_model": scenario["selected_model"],
        "scenario_seed": scenario["scenario_seed"],
        "scenario_count": scenario["scenario_count"],
        "route_time_scale": scenario["route_time_scale"],
        "route_dispersion_multiplier": scenario["route_dispersion_multiplier"],
        "route_offset_s": scenario["route_offset_s"],
        "admission_thresholds": _json(root / "stage2/config/stage2_v5.json")["admission"],
        "final_upstream_plan_sha256": _sha256(docs / "stage2_v5_final_upstream_plan.json"),
        "final_stage0_config_sha256": upstream["final_stage0_config_sha256"],
        "performance_gate_status": verification["performance_gate_status"],
        "development_scientific_status": verification["development_scientific_status"],
        "new_final_test_rows_read": 0,
        "post_final_test_tuning_allowed": False,
    }
    (docs / "stage2_v5_development_freeze.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def freeze_release(*, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root).resolve()
    docs = root / "stage2/docs/v5"
    verification = _json(docs / "stage2_v5_final_verification.json")
    final = _json(docs / "stage2_v5_final_test_results.json")
    development = _json(docs / "stage2_v5_development_freeze.json")
    if verification.get("stage3_admission_status") != "READY_FOR_STAGE3":
        raise RuntimeError("final verification does not admit Stage 3")
    if final.get("post_test_tuning_count") != 0 or final.get("protocol") != "one_shot_preregistered":
        raise RuntimeError("final test protocol was not one-shot and tuning-free")
    scenario = _json(docs / "stage2_v5_scenario_selection.json")
    split = _json(docs / "stage2_v5_split_freeze.json")
    result = {
        "schema_version": "stage2_v5_release_manifest.1",
        "release_status": "FROZEN",
        "base_stage2_tag": "stage2-v4-final",
        "base_stage2_commit": "70cb70265cbb95e5fc9981024a554de28ee2be85",
        "v5_implementation_commit": development["implementation_commit"],
        "release_source_commit": _git_head(root),
        "v5_config_sha": development["config_sha256"],
        "split_freeze_sha": development["split_freeze_sha256"],
        "service_time_target_contract": _json(docs / "stage2_v5_service_time_target_audit.json")["target_contract"],
        "performance_gate_status": verification["performance_gate_status"],
        "performance_benchmark_sha": _sha256(docs / "stage2_v5_performance_benchmarks.json"),
        "profile_sha": _sha256(docs / "performance_profile_hotspots.txt"),
        "baseline_comparison_status": verification["development_scientific_status"],
        "selection_model_id": development["selection_model_id"],
        "distribution_model_id": development["distribution_model_id"],
        "scenario_generator_id": scenario["generator_id"],
        "scenario_seed": scenario["scenario_seed"],
        "rolling_fold_definition": split.get("rolling_fold_definition"),
        "legacy_test_identity": {
            "dates": split["split"]["legacy_benchmark_dates"],
            "role": "frozen_legacy_benchmark_only",
            "untouched": False,
        },
        "new_test_identity": {
            "dates": final["dates"],
            "protocol": final["protocol"],
            "result_sha256": _sha256(docs / "stage2_v5_final_test_results.json"),
            "upstream_plan_sha256": development["final_upstream_plan_sha256"],
        },
        "stage3_admission_status": verification["stage3_admission_status"],
        "engineering_status": verification["engineering_status"],
        "temporal_contract_status": verification["temporal_contract_status"],
        "scientific_status": (
            "PREDICTIVE_BASELINE_VALIDATED"
            if final["aggregate_mae_better_than_strong_baseline"] and final["paired_bootstrap_ci_below_zero"]
            else "BASELINE_COMPETITIVE"
        ),
        "post_test_tuning_count": final["post_test_tuning_count"],
    }
    (docs / "stage2_v5_release_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze-development", "freeze-release"))
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    result = (
        freeze_development(repo_root=args.repo_root)
        if args.command == "freeze-development"
        else freeze_release(repo_root=args.repo_root)
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
