"""Execute one frozen Stage 2 v5 temporal protocol without upstream rebuilds."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import joblib

from .ablations import evaluate_ablations
from .baselines import evaluate_service_baselines, fit_service_baselines
from .config import load_config
from .evaluate import evaluate
from .prediction import merge_all
from .scenario_pipeline import run as run_scenarios
from .shards import build_shards


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def protocol_payload(base: dict[str, Any], protocol: str) -> dict[str, Any]:
    payload = copy.deepcopy(base)
    if protocol == "development":
        return payload
    if protocol.startswith("fold_"):
        fold = next((item for item in payload["rolling_folds"] if item["fold_id"] == protocol), None)
        if fold is None:
            raise ValueError(f"unknown rolling protocol: {protocol}")
        payload["split"] = {
            "protocol_name": protocol,
            "train_dates": fold["train_dates"],
            "validation_model_dates": fold["validation_model_dates"],
            "calibration_dates": fold["calibration_dates"],
            "evaluation_dates": fold["evaluation_dates"],
            "legacy_test_dates": [],
        }
        return payload
    if protocol == "legacy":
        legacy = payload["legacy_benchmark_fit"]
        payload["split"] = {
            "protocol_name": "legacy_frozen_benchmark",
            "train_dates": legacy["train_dates"],
            "validation_model_dates": legacy["validation_model_dates"],
            "calibration_dates": legacy["calibration_dates"],
            "evaluation_dates": [],
            "legacy_test_dates": legacy["benchmark_dates"],
        }
        return payload
    raise ValueError(f"unknown protocol: {protocol}")


def materialize_protocol_config(base_path: Path, output_path: Path, protocol: str) -> dict[str, Any]:
    payload = protocol_payload(json.loads(base_path.read_text(encoding="utf-8")), protocol)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    validated = load_config(output_path)
    return {
        "path": output_path.as_posix(),
        "sha256": validated.digest,
        "protocol_name": validated.section("split")["protocol_name"],
    }


def run_protocol(
    protocol: str,
    *,
    repo_root: str | Path = ".",
    resume: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    protocol_root = root / "stage2/output_v5/protocols" / protocol
    report_root = root / "stage2/docs/v5/protocols" / protocol
    config_path = protocol_root / "config.json"
    identity = materialize_protocol_config(root / "stage2/config/stage2_v5.json", config_path, protocol)
    config = load_config(config_path)
    baseline_root = protocol_root / "baselines"
    baseline_path = baseline_root / "service_time_baselines.joblib"
    if resume and baseline_path.is_file():
        baseline = joblib.load(baseline_path)
        if baseline.get("fit_dates") != config.section("split")["train_dates"]:
            raise ValueError(f"non-resumable baseline bundle for {protocol}")
    else:
        baseline = fit_service_baselines(config, repo_root=root)
        baseline_root.mkdir(parents=True, exist_ok=True)
        joblib.dump(baseline, baseline_path)
    baseline_metrics, baseline_bootstrap, baseline_summary = evaluate_service_baselines(baseline, config, repo_root=root)
    report_root.mkdir(parents=True, exist_ok=True)
    baseline_metrics.to_csv(report_root / "baseline_comparison.csv", index=False)
    baseline_bootstrap.to_csv(report_root / "baseline_paired_error_bootstrap.csv", index=False)
    (report_root / "baseline_summary.json").write_text(json.dumps(baseline_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    tensor_root = protocol_root / "tensor_shards"
    shard_summary = build_shards(config, repo_root=root, output_root=tensor_root, resume=resume)
    model_root = protocol_root / "deep_model"
    chunk_prediction_root = protocol_root / "deep_predictions"
    command = [
        str(config.section("deep")["python_executable"]), "-m", "stage2.v5.train_worker",
        "--config", str(config_path), "--tensor-root", str(tensor_root),
        "--output", str(model_root), "--prediction-root", str(chunk_prediction_root),
        "--history-mode", "gate",
    ]
    if resume:
        partial_model = (model_root / "best_model.pt").exists() != (model_root / "model_manifest.json").exists()
        command.append("--force" if partial_model else "--resume")
    subprocess.run(command, cwd=root, check=True)
    prediction_root = protocol_root / "predictions"
    prediction_summary = merge_all(chunk_prediction_root, prediction_root)
    model_summary = evaluate(
        repo_root=root,
        config_path=config_path,
        baseline_root=baseline_root,
        prediction_root=prediction_root,
        report_root=report_root,
    )
    ablation_summary = None
    if protocol == "development":
        ablation_roots: dict[str, Path] = {"horizon_gate": prediction_root}
        for history_mode in ("ordinary_concatenation", "without_recent", "without_profile"):
            local_root = protocol_root / "ablations" / history_mode
            local_chunks = local_root / "deep_predictions"
            local_model = local_root / "deep_model"
            local_command = [
                str(config.section("deep")["python_executable"]), "-m", "stage2.v5.train_worker",
                "--config", str(config_path), "--tensor-root", str(tensor_root),
                "--output", str(local_model), "--prediction-root", str(local_chunks),
                "--history-mode", history_mode,
            ]
            if resume:
                partial_model = (local_model / "best_model.pt").exists() != (local_model / "model_manifest.json").exists()
                local_command.append("--force" if partial_model else "--resume")
            subprocess.run(local_command, cwd=root, check=True)
            merged_root = local_root / "predictions"
            merge_all(local_chunks, merged_root)
            ablation_roots[history_mode] = merged_root
        ablation_summary = evaluate_ablations(
            repo_root=root,
            config_path=config_path,
            model_roots=ablation_roots,
            report_root=report_root,
        )
    scenario_summary = run_scenarios(
        repo_root=root,
        config_path=config_path,
        prediction_root=prediction_root,
        output_root=protocol_root / "route_scenarios",
        report_root=report_root,
        model_root=model_root,
    )
    model_manifest = json.loads((model_root / "model_manifest.json").read_text(encoding="utf-8"))
    result = {
        "schema_version": "stage2_v5_protocol_run.1",
        "status": "PASS",
        "protocol": protocol,
        "protocol_config": identity,
        "train_dates": config.section("split")["train_dates"],
        "validation_model_dates": config.section("split")["validation_model_dates"],
        "calibration_dates": config.section("split")["calibration_dates"],
        "evaluation_dates": config.section("split")["evaluation_dates"],
        "legacy_benchmark_dates": config.section("split")["legacy_test_dates"],
        "feature_artifact_sha256": hashlib.sha256((tensor_root / "feature_artifacts.json").read_bytes()).hexdigest(),
        "feature_fit_dates": json.loads((tensor_root / "feature_artifacts.json").read_text(encoding="utf-8"))["fit_dates"],
        "baseline_fit_dates": baseline["fit_dates"],
        "model_fit_dates": model_manifest["fit_dates"],
        "model_id": model_manifest["model_id"],
        "shard_day_count": shard_summary["day_count"],
        "prediction_day_count": prediction_summary["day_count"],
        "model_evaluation": model_summary,
        "scenario_selection": scenario_summary,
        "development_ablation": ablation_summary,
        "upstream_rebuild_performed": False,
    }
    (report_root / "protocol_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", choices=("development", "fold_1", "fold_2", "fold_3", "legacy"))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result = run_protocol(args.protocol, repo_root=args.repo_root, resume=not args.no_resume)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
