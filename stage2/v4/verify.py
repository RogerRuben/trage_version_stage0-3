"""Final engineering verification and freeze manifest for Stage 2 v4."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .config import Stage2V4Config
from .contracts import FORBIDDEN_FORMAL_TARGETS
from .io import (
    atomic_write_json,
    sha256_file,
    stage2_v4_code_identity,
)


VERIFY_SCHEMA_VERSION = "stage2_v4_verification.1"
FREEZE_SCHEMA_VERSION = "stage2_v4_release_manifest.1"


def _read_json(path: Path, failures: list[str], label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"cannot read {label}: {path}: {exc}")
        return {}
    if not isinstance(value, dict):
        failures.append(f"{label} is not an object: {path}")
        return {}
    return value


def _expect(
    failures: list[str],
    actual: Any,
    expected: Any,
    label: str,
) -> None:
    if actual != expected:
        failures.append(f"{label}: expected {expected!r}, got {actual!r}")


def verify_stage2_v4(
    output_root: str | Path,
    config: Stage2V4Config,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(output_root)
    failures: list[str] = []
    preflight = _read_json(
        Path("stage2/docs/v4/stage2_v4_preflight.json"),
        failures,
        "preflight",
    )
    history = _read_json(
        root / "causal_history_store/history_store_manifest.json",
        failures,
        "history manifest",
    )
    dataset = _read_json(
        root / "route_conditioned_dataset/dataset_manifest.json",
        failures,
        "dataset manifest",
    )
    shards = _read_json(
        root / "tensor_shards/tensor_manifest.json",
        failures,
        "tensor manifest",
    )
    baseline = _read_json(
        root / "models/baselines/baseline_manifest.json",
        failures,
        "baseline manifest",
    )
    deep = _read_json(
        root / "models/rc_mstnet_v4/model_manifest.json",
        failures,
        "deep model manifest",
    )
    calibration = _read_json(
        root / "models/calibration/calibration_manifest.json",
        failures,
        "calibration manifest",
    )
    evaluation = _read_json(
        root / "evaluation/final_test/evaluation_report.json",
        failures,
        "evaluation report",
    )
    manifests = {
        "preflight": preflight,
        "history": history,
        "dataset": dataset,
        "shards": shards,
        "baseline": baseline,
        "deep": deep,
        "calibration": calibration,
        "evaluation": evaluation,
    }
    for name, manifest in manifests.items():
        _expect(
            failures,
            manifest.get("engineering_status"),
            "PASS",
            f"{name} status",
        )
        _expect(
            failures,
            manifest.get("stage2_config_sha256"),
            config.digest,
            f"{name} config SHA",
        )

    expected = config.section("stage1_release")
    _expect(failures, preflight.get("bucket_count"), 196, "preflight bucket count")
    _expect(
        failures,
        preflight.get("counters", {}).get("order_count"),
        220000,
        "preflight order count",
    )
    for counter in (
        "orphan_traversal_label_count",
        "decision_time_missing_count",
        "self_order_history_candidate_count",
        "route_token_conservation_error",
        "label_row_conservation_error",
    ):
        _expect(
            failures,
            preflight.get("counters", {}).get(counter),
            0,
            f"preflight {counter}",
        )
    _expect(
        failures,
        history.get("event_count"),
        expected["traversal_label_count"],
        "history event count",
    )
    for track in ("revealed_route_proxy", "oracle_timing"):
        _expect(
            failures,
            dataset.get("tracks", {}).get(track, {}).get("row_count"),
            expected["route_sequence_count"],
            f"dataset {track} row count",
        )
    _expect(
        failures,
        shards.get("route_token_count"),
        expected["route_sequence_count"],
        "tensor route token count",
    )
    _expect(failures, shards.get("order_count"), 220000, "tensor order count")
    _expect(failures, baseline.get("test_rows_read"), 0, "baseline Test reads")
    _expect(failures, deep.get("test_rows_read"), 0, "deep training Test reads")
    _expect(
        failures,
        calibration.get("fit_dates"),
        ["20161027"],
        "calibration fit dates",
    )
    _expect(
        failures,
        calibration.get("test_rows_read"),
        0,
        "calibration Test reads",
    )
    _expect(
        failures,
        evaluation.get("test_dates"),
        ["20161031"],
        "evaluation Test date",
    )
    _expect(
        failures,
        evaluation.get("test_tuning_violation_count"),
        0,
        "Test tuning violations",
    )

    dataset_root = root / "route_conditioned_dataset/revealed_route_proxy"
    mask_nan_violations = 0
    time_leakage_violations = 0
    route_duplicate_count = 0
    dataset_rows = 0
    target_masks = {
        "crawl_time_share": "crawl_target_valid",
        "stop_time_share": "stop_target_valid",
        "speed_cv_bounded": "speed_cv_target_valid",
        "acceleration_rms_bounded": "acceleration_rms_target_valid",
        "lcs_raw": "lcs_target_valid",
        "rts_raw": "rts_target_valid",
    }
    for path in sorted(dataset_root.glob("day=*.parquet")):
        schema = set(pq.read_schema(path).names)
        forbidden = schema & FORBIDDEN_FORMAL_TARGETS
        if forbidden:
            failures.append(f"forbidden formal targets in {path}: {sorted(forbidden)}")
        columns = [
            "split",
            "date",
            "order_id",
            "route_sequence",
            "decision_time",
            "availability_timestamp",
            *target_masks,
            *target_masks.values(),
        ]
        frame = pd.read_parquet(path, columns=list(dict.fromkeys(columns)))
        dataset_rows += len(frame)
        route_duplicate_count += int(
            frame.duplicated(["split", "date", "order_id", "route_sequence"]).sum()
        )
        available = pd.to_numeric(frame["availability_timestamp"], errors="coerce")
        decision = pd.to_numeric(frame["decision_time"], errors="coerce")
        time_leakage_violations += int((available.notna() & available.ge(decision)).sum())
        for target, mask in target_masks.items():
            valid = frame[mask].fillna(False).astype(bool)
            value = pd.to_numeric(frame[target], errors="coerce")
            mask_nan_violations += int((valid & value.isna()).sum())
            mask_nan_violations += int((~valid & value.notna()).sum())
    _expect(
        failures,
        dataset_rows,
        expected["route_sequence_count"],
        "physical dataset rows",
    )
    _expect(failures, route_duplicate_count, 0, "physical route duplicates")
    _expect(failures, time_leakage_violations, 0, "physical time leakage")
    _expect(failures, mask_nan_violations, 0, "physical mask/NaN violations")

    report = {
        "schema_version": VERIFY_SCHEMA_VERSION,
        "engineering_status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "stage2_config_sha256": config.digest,
        "stage2_code_sha": stage2_v4_code_identity(),
        "stage1_release_tag": expected["release_tag"],
        "stage1_release_commit": expected["release_commit"],
        "stage1_model_id": expected["model_id"],
        "counters": {
            "order_count": preflight.get("counters", {}).get("order_count"),
            "route_token_count": dataset_rows,
            "history_event_count": history.get("event_count"),
            "route_duplicate_count": route_duplicate_count,
            "time_leakage_violation_count": time_leakage_violations,
            "test_tuning_violation_count": evaluation.get(
                "test_tuning_violation_count"
            ),
            "mask_nan_violation_count": mask_nan_violations,
        },
        "runtime_s": time.perf_counter() - started,
    }
    return report


def _git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def freeze_stage2_v4(
    output_root: str | Path,
    verification_path: str | Path,
    release_path: str | Path,
    config: Stage2V4Config,
) -> dict[str, Any]:
    verification = json.loads(Path(verification_path).read_text(encoding="utf-8"))
    if verification.get("engineering_status") != "PASS":
        raise RuntimeError("Stage 2 v4 cannot freeze without PASS verification")
    root = Path(output_root)
    model = json.loads(
        (root / "models/rc_mstnet_v4/model_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    calibration = json.loads(
        (root / "models/calibration/calibration_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    evaluation = json.loads(
        (root / "evaluation/final_test/evaluation_report.json").read_text(
            encoding="utf-8"
        )
    )
    expected = config.section("stage1_release")
    release = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "engineering_status": "ENGINEERING_PASS",
        "temporal_status": "TEMPORAL_CONTRACT_PASS",
        "scientific_status": "PREDICTIVE_BASELINE_VALIDATED",
        "stage3_admission_status": "REQUIRES_EVALUATION_THRESHOLD_REVIEW",
        "git_source_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "stage2_config_sha256": config.digest,
        "stage2_code_sha": verification["stage2_code_sha"],
        "stage1_release": {
            "tag": expected["release_tag"],
            "commit": expected["release_commit"],
            "model_id": expected["model_id"],
            "config_sha256": expected["config_sha256"],
        },
        "stage0_release": expected["stage0_release"],
        "split": config.section("split"),
        "feature_cutoff_rule": {
            "decision_time_source": "stage0_order_departure_time",
            "history_event_predicate": "availability_timestamp < decision_time",
            "self_order_history_excluded": True,
        },
        "target_contract": {
            "supervision_unit": "physical_traversal",
            "formal_continuous_targets": [
                "crawl_time_share",
                "stop_time_share",
                "speed_cv_bounded",
                "acceleration_rms_bounded",
                "lcs_raw",
                "rts_raw",
            ],
            "derived_outputs": ["lcs_pct", "rts_pct"],
            "excluded_targets": sorted(FORBIDDEN_FORMAL_TARGETS),
            "missing_target_policy": "nullable_value_plus_explicit_validity_mask",
        },
        "history_manifest_sha256": sha256_file(
            root / "causal_history_store/history_store_manifest.json"
        ),
        "dataset_manifest_sha256": sha256_file(
            root / "route_conditioned_dataset/dataset_manifest.json"
        ),
        "tensor_manifest_sha256": sha256_file(
            root / "tensor_shards/tensor_manifest.json"
        ),
        "baseline_manifest_sha256": sha256_file(
            root / "models/baselines/baseline_manifest.json"
        ),
        "deep_model_manifest_sha256": sha256_file(
            root / "models/rc_mstnet_v4/model_manifest.json"
        ),
        "deep_checkpoint_sha256": model["checkpoint_sha256"],
        "deep_model_id": model["model_id"],
        "calibration_manifest_sha256": sha256_file(
            root / "models/calibration/calibration_manifest.json"
        ),
        "calibration_bundle_sha256": calibration["bundle_sha256"],
        "calibration_model_id": calibration["calibration_model_id"],
        "evaluation_report_sha256": sha256_file(
            root / "evaluation/final_test/evaluation_report.json"
        ),
        "prediction_file_sha256": evaluation["prediction_file_sha256"],
        "order_route_prediction_sha256": sha256_file(
            root / "evaluation/final_test/order_route_predictions.parquet"
        ),
        "report_attachment_sha256": {
            name: sha256_file(root / "reports" / name)
            for name in (
                "metrics_by_target.csv",
                "metrics_by_subgroup.csv",
                "bootstrap_intervals.csv",
                "calibration_metrics.csv",
                "entry_time_gap_audit.csv",
                "ablation_results.csv",
                "order_aggregation_metrics.csv",
            )
        },
        "test_evaluation_identity": {
            "test_dates": evaluation["test_dates"],
            "deep_model_id": evaluation["deep_model_id"],
            "calibration_model_id": evaluation["calibration_model_id"],
            "stage1_cdf_model_id": evaluation["stage1_cdf_model_id"],
            "test_tuning_violation_count": evaluation[
                "test_tuning_violation_count"
            ],
        },
        "verification_sha256": sha256_file(verification_path),
        "test_metrics": {
            "continuous": evaluation["continuous_metrics"],
            "tail": evaluation["tail_metrics"],
            "order": evaluation["order_metrics"],
        },
        "limitations": [
            "predictions are associative, not causal effects",
            "tail probabilities are pressure-label probabilities, not accident probabilities",
            "oracle timing track is diagnostic only and cannot enter Stage 3",
        ],
    }
    atomic_write_json(release_path, release)
    return release
