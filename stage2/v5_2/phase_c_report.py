"""Build the immutable Phase C development report from completed artifacts.

This module never trains or evaluates a model.  It validates and summarizes
the already-produced M0--M4 development artifacts, then emits a compact report
and a hash-bound evidence bundle for remote review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .contracts import CORE_TRANSFER_TARGETS, Stage2V52ContractError
from .feature_binding import sha256_path
from .protocols import get_protocol


REPORT_SCHEMA_VERSION = "stage2_v5_2_phase_c_report.2"
EVIDENCE_SCHEMA_VERSION = "stage2_v5_2_phase_c_evidence_bundle.2"
POST_C_STATUS_SCHEMA_VERSION = "stage2_v5_2_status_manifest.2"
MODELS = ("M0", "M1", "M2", "M3", "M4")
PHASE_C_AUTHORIZATION_COMMIT = "7a95a685042d41ded7143c4cac015fc04761af90"
PHASE_C_EXECUTION_COMMIT = "13196fb7d42ee0054d2dc3550c597c7c2f881790"


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Stage2V52ContractError(f"missing Phase C artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Stage2V52ContractError(f"Phase C artifact is not an object: {path}")
    return payload


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _descriptor(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Stage2V52ContractError(f"cannot bind missing Phase C artifact: {path}")
    return {
        "path": _relative(path, root),
        "sha256": sha256_path(path),
        "size_bytes": int(path.stat().st_size),
    }


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, encoding="utf-8",
    ).strip()


def _git_file_descriptor(root: Path, *, commit: str, path: str) -> dict[str, Any]:
    try:
        content = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=root)
    except subprocess.CalledProcessError as error:
        raise Stage2V52ContractError(
            f"cannot resolve Phase C Git provenance: {commit}:{path}"
        ) from error
    return {
        "commit": commit,
        "path": path,
        "git_file_sha256": hashlib.sha256(content).hexdigest(),
        "git_file_size_bytes": len(content),
    }


def _validate_git_file_descriptor(
    payload: Mapping[str, Any], *, repo_root: Path, name: str,
) -> None:
    commit, path = payload.get("commit"), payload.get("path")
    if not isinstance(commit, str) or not isinstance(path, str):
        raise Stage2V52ContractError(f"{name} Git provenance is incomplete")
    if dict(payload) != _git_file_descriptor(repo_root, commit=commit, path=path):
        raise Stage2V52ContractError(f"{name} Git provenance does not resolve")


def _relative_improvement(reference: float, candidate: float) -> float:
    if reference <= 0:
        raise Stage2V52ContractError("Phase C relative comparison requires positive reference MAE")
    return (reference - candidate) / reference


def _evaluation_paths(root: Path) -> dict[str, Path]:
    base = root / "stage2/output_v5_2/development"
    return {
        "M0": base / "M0/evaluation.json",
        "M1": base / "M1/evaluation.json/evaluation_manifest.json",
        "M2": base / "M2/evaluation/evaluation_manifest.json",
        "M3": base / "M3/evaluation/evaluation_manifest.json",
        "M4": base / "M4/evaluation/evaluation_manifest.json",
    }


def _validation_paths(root: Path) -> dict[str, Path]:
    base = root / "stage2/output_v5_2/development"
    return {
        "M0": base / "M0/validation_evaluation.json",
        "M1": base / "M1/validation_evaluation.json/evaluation_manifest.json",
        "M2": base / "M2/validation_evaluation/evaluation_manifest.json",
        "M3": base / "M3/validation_evaluation/evaluation_manifest.json",
        "M4": base / "M4/validation_evaluation/evaluation_manifest.json",
    }


def _training_paths(root: Path) -> dict[str, Path]:
    base = root / "stage2/output_v5_2/development"
    return {
        "M0": base / "M0/model/model_manifest.json",
        **{model: base / model / "model_manifest.json" for model in MODELS[1:]},
    }


def _checkpoint_paths(root: Path, training: Mapping[str, Mapping[str, Any]]) -> dict[str, Path]:
    return {
        "M0": root / "stage2/output_v5_2/development/M0/model/m0_micro_tree.joblib",
        **{
            model: root / str(training[model]["selected_checkpoint_path"])
            for model in MODELS[1:]
        },
    }


def _validate_evaluations(
    evaluations: Mapping[str, Mapping[str, Any]], *, role: str, expected_dates: tuple[str, ...],
) -> None:
    protocol = get_protocol("development")
    for model, payload in evaluations.items():
        if (
            payload.get("status") != "PASS"
            or payload.get("protocol_id") != "development"
            or payload.get("protocol_hash") != protocol.digest
            or payload.get("role") != role
            or tuple(payload.get("evaluation_dates", ())) != expected_dates
            or payload.get("model_id") != model
        ):
            raise Stage2V52ContractError(f"Phase C {role} evaluation identity differs for {model}")
        for group in ("core_mae", "low_support_core_mae", "unseen_core_mae"):
            values = payload.get(group, {})
            if set(values) != set(CORE_TRANSFER_TARGETS) or any(values[target] is None for target in values):
                raise Stage2V52ContractError(f"Phase C {model} has incomplete {group}")
        if set(payload.get("metrics_by_date", {})) != set(expected_dates):
            raise Stage2V52ContractError(f"Phase C {model} has incomplete per-date metrics")


def _group_metrics(payload: Mapping[str, Any], group: str) -> dict[str, float]:
    field = {
        "overall": "core_mae", "low": "low_support_core_mae",
        "unseen": "unseen_core_mae",
    }[group]
    return {target: float(payload[field][target]) for target in CORE_TRANSFER_TARGETS}


def _group_counts(payload: Mapping[str, Any], group: str) -> dict[str, int]:
    metrics = payload.get("metrics_by_support", {}).get(group, {})
    return {target: int(metrics[target]["count"]) for target in CORE_TRANSFER_TARGETS}


def _daily_metrics(payload: Mapping[str, Any], date: str) -> dict[str, Any]:
    day = payload["metrics_by_date"][date]
    return {
        "overall": {
            target: float(day["groups"]["overall"][target]["mae"])
            for target in CORE_TRANSFER_TARGETS
        },
        "low": {
            target: float(day["groups"]["low"][target]["mae"])
            for target in CORE_TRANSFER_TARGETS
        },
        "unseen": {
            target: float(day["groups"]["unseen"][target]["mae"])
            for target in CORE_TRANSFER_TARGETS
        },
        "pace_p50_mae": (
            float(day["pace_p50"]["mae"])
            if isinstance(day.get("pace_p50"), Mapping) and day["pace_p50"].get("mae") is not None
            else None
        ),
    }


def _comparison(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in ("overall", "low", "unseen"):
        base, value = _group_metrics(reference, group), _group_metrics(candidate, group)
        result[group] = {
            target: _relative_improvement(base[target], value[target])
            for target in CORE_TRANSFER_TARGETS
        }
        result[f"{group}_four_target_mean_relative_improvement"] = float(
            sum(result[group].values()) / len(CORE_TRANSFER_TARGETS)
        )
    base_pace, value_pace = reference.get("pace_p50_mae"), candidate.get("pace_p50_mae")
    result["pace_p50_relative_improvement"] = (
        _relative_improvement(float(base_pace), float(value_pace))
        if base_pace is not None and value_pace is not None else None
    )
    return result


def classify_phase_c_direction(decision: Mapping[str, Any], *, pace_guard_pass: bool) -> str:
    """Map the pre-registered gate to PASS/WEAK/FAIL without manual relabeling."""
    if bool(decision.get("adopt")) and pace_guard_pass:
        return "PASS"
    wins = int(decision.get("low_support_target_wins", 0))
    mean_improvement = float(decision.get("low_support_mean_relative_improvement", 0.0))
    overall_stable = bool(decision.get("overall_no_target_degrades_over_2pct"))
    unseen_stable = bool(decision.get("unseen_not_worse_than_structure_only"))
    if (
        wins >= 3
        and mean_improvement > 0
        and overall_stable
        and unseen_stable
        and pace_guard_pass
    ):
        return "WEAK"
    return "FAIL"


def _validate_m4_tau_binding(
    *, training: Mapping[str, Any], evaluation: Mapping[str, Any],
    tau_freeze: Mapping[str, Any], tau_freeze_sha256: str,
    current_support_sha256: str,
) -> dict[str, Any]:
    constructor = training.get("constructor", {})
    provenance = constructor.get("support_tau_provenance", {})
    expected_source_support = tau_freeze.get("transfer_tuning_support_sha256")
    if (
        tau_freeze.get("schema_version") != "stage2_v5_2_tau_freeze.1"
        or tau_freeze.get("status") != "PASS"
        or tau_freeze.get("selected_candidate") != "p25"
        or float(tau_freeze.get("selected_tau", float("nan"))) != 3.0
        or constructor.get("spatial_mode") != "support_aware"
        or float(constructor.get("support_tau", float("nan"))) != 3.0
        or provenance.get("kind") != "frozen_transfer_tuning_selection"
        or provenance.get("support_tau_candidate") != "p25"
        or float(provenance.get("support_tau_value", float("nan"))) != 3.0
        or provenance.get("tau_freeze_artifact_sha256") != tau_freeze_sha256
        or provenance.get("support_tau_source_support_sha256") != expected_source_support
        or provenance.get("current_protocol_support_artifact_sha256") != current_support_sha256
        or float(evaluation.get("support_tau", float("nan"))) != 3.0
        or evaluation.get("support_tau_candidate") != "p25"
        or evaluation.get("support_tau_source_support_sha256") != expected_source_support
        or evaluation.get("support_artifact_sha256") != current_support_sha256
    ):
        raise Stage2V52ContractError("M4 training/evaluation does not consume the frozen p25 tau binding")
    return {
        "status": "PASS", "selected_candidate": "p25", "selected_tau": 3.0,
        "tau_freeze_file_sha256": tau_freeze_sha256,
        "transfer_tuning_support_sha256": expected_source_support,
        "development_support_sha256": current_support_sha256,
        "training_spatial_mode": "support_aware",
    }


def _validate_artifact_hash(payload: Mapping[str, Any], *, name: str) -> None:
    expected = payload.get("artifact_sha256")
    body = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    if not isinstance(expected, str) or expected != _canonical_hash(body):
        raise Stage2V52ContractError(f"{name} embedded artifact hash does not resolve")


def verify_phase_c_evidence_bundle(
    payload: Mapping[str, Any], *, repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if (
        payload.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or payload.get("status") != "PASS"
        or payload.get("phase_c_direction") not in {"PASS", "WEAK", "FAIL"}
        or payload.get("phase_d_authorized") is not False
    ):
        raise Stage2V52ContractError("invalid Phase C evidence identity")
    _validate_artifact_hash(payload, name="Phase C evidence bundle")
    if (
        payload.get("phase_c_authorization_commit") != PHASE_C_AUTHORIZATION_COMMIT
        or payload.get("execution_base_commit") != PHASE_C_AUTHORIZATION_COMMIT
        or payload.get("phase_c_execution_commit") != PHASE_C_EXECUTION_COMMIT
    ):
        raise Stage2V52ContractError("Phase C execution commits are not frozen")
    _validate_git_file_descriptor(
        payload["frozen_bindings"]["phase_c_execution_config_git"],
        repo_root=root,
        name="Phase C execution config",
    )
    resolved = 0

    def visit(value: Any) -> None:
        nonlocal resolved
        if isinstance(value, Mapping):
            if set(("path", "sha256", "size_bytes")) <= set(value):
                path = root / str(value["path"])
                if (
                    not path.is_file()
                    or sha256_path(path) != value["sha256"]
                    or int(path.stat().st_size) != int(value["size_bytes"])
                ):
                    raise Stage2V52ContractError(f"Phase C evidence descriptor does not resolve: {path}")
                resolved += 1
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    if resolved < 30 or set(payload.get("models", {})) != set(MODELS):
        raise Stage2V52ContractError("Phase C evidence bundle is incomplete")
    report_descriptor = payload["reports"]["json"]
    report = _read_json(root / str(report_descriptor["path"]))
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("phase_c_direction") != payload.get("phase_c_direction")
        or report.get("phase_d_authorized") is not False
    ):
        raise Stage2V52ContractError("Phase C report and evidence direction differ")
    _validate_artifact_hash(report, name="Phase C report")
    tau_descriptor = payload["frozen_bindings"]["tau_freeze"]
    support_descriptor = payload["train_only_artifacts"]["support"]
    m4_descriptors = payload["models"]["M4"]
    tau_binding = _validate_m4_tau_binding(
        training=_read_json(root / str(m4_descriptors["training_manifest"]["path"])),
        evaluation=_read_json(root / str(m4_descriptors["development_evaluation"]["path"])),
        tau_freeze=_read_json(root / str(tau_descriptor["path"])),
        tau_freeze_sha256=str(tau_descriptor["sha256"]),
        current_support_sha256=str(support_descriptor["sha256"]),
    )
    if payload.get("m4_tau_consumption") != tau_binding or report.get("m4_tau_consumption") != tau_binding:
        raise Stage2V52ContractError("Phase C M4 tau relationship is not self-contained")
    return {
        "status": "PASS", "resolved_artifact_count": resolved,
        "phase_c_direction": payload["phase_c_direction"], "phase_d_authorized": False,
        "m4_tau_consumption": "PASS",
    }


def build_phase_c_report(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    protocol = get_protocol("development")
    evaluation_paths, validation_paths, training_paths = (
        _evaluation_paths(root), _validation_paths(root), _training_paths(root)
    )
    evaluations = {model: _read_json(path) for model, path in evaluation_paths.items()}
    validations = {model: _read_json(path) for model, path in validation_paths.items()}
    training = {model: _read_json(path) for model, path in training_paths.items()}
    _validate_evaluations(
        evaluations, role="evaluation", expected_dates=tuple(protocol.evaluation_dates),
    )
    _validate_evaluations(
        validations, role="validation", expected_dates=tuple(protocol.validation_dates),
    )
    for model, payload in training.items():
        if payload.get("status") != "PASS" or payload.get("protocol_id") != "development":
            raise Stage2V52ContractError(f"Phase C training manifest differs for {model}")
    checkpoints = _checkpoint_paths(root, training)
    checkpoint_descriptors = {model: _descriptor(path, root) for model, path in checkpoints.items()}
    for model in MODELS[1:]:
        expected = evaluations[model].get("checkpoint_sha256")
        if checkpoint_descriptors[model]["sha256"] != expected:
            raise Stage2V52ContractError(f"Phase C selected checkpoint hash differs for {model}")

    decision_path = root / "stage2/output_v5_2/development/spatial_direction_decision.json"
    decision = _read_json(decision_path)
    if decision.get("status") != "PASS" or tuple(decision.get("evaluation_dates", ())) != tuple(protocol.evaluation_dates):
        raise Stage2V52ContractError("Phase C directional decision identity differs")
    m1_pace, m4_pace = float(evaluations["M1"]["pace_p50_mae"]), float(evaluations["M4"]["pace_p50_mae"])
    pace_degradation = (m4_pace - m1_pace) / m1_pace
    pace_guard = {"status": "PASS" if pace_degradation <= 0.02 else "FAIL", "relative_degradation": pace_degradation, "maximum": 0.02}
    direction = classify_phase_c_direction(decision, pace_guard_pass=pace_guard["status"] == "PASS")
    tau_path = root / "stage2/output_v5_2/transfer_tuning/stage2_v5_2_tau_freeze.json"
    support_path = root / "stage2/output_v5_2/development/artifacts/support.json"
    tau_descriptor, support_descriptor = _descriptor(tau_path, root), _descriptor(support_path, root)
    tau_freeze = _read_json(tau_path)
    m4_tau_consumption = _validate_m4_tau_binding(
        training=training["M4"], evaluation=evaluations["M4"], tau_freeze=tau_freeze,
        tau_freeze_sha256=tau_descriptor["sha256"],
        current_support_sha256=support_descriptor["sha256"],
    )
    validation_tau_consumption = _validate_m4_tau_binding(
        training=training["M4"], evaluation=validations["M4"], tau_freeze=tau_freeze,
        tau_freeze_sha256=tau_descriptor["sha256"],
        current_support_sha256=support_descriptor["sha256"],
    )
    if validation_tau_consumption != m4_tau_consumption:
        raise Stage2V52ContractError("M4 validation and evaluation tau bindings differ")

    model_metrics = {
        model: {
            "overall": _group_metrics(payload, "overall"),
            "low": _group_metrics(payload, "low"),
            "unseen": _group_metrics(payload, "unseen"),
            "counts": {
                group: _group_counts(payload, group) for group in ("overall", "low", "unseen")
            },
            "pace_p50_mae": payload.get("pace_p50_mae"),
            "per_date": {
                date: _daily_metrics(payload, date) for date in protocol.evaluation_dates
            },
        }
        for model, payload in evaluations.items()
    }
    temporal_leakage = {
        model: int(training[model].get("temporal_leakage_count", 0)) for model in MODELS[1:]
    }
    if any(temporal_leakage.values()):
        raise Stage2V52ContractError("Phase C temporal leakage is nonzero")
    test_evidence_path = root / "stage2/docs/v5_2/stage2_v5_2_phase_c_test_evidence.json"
    test_evidence = _read_json(test_evidence_path)
    if (
        test_evidence.get("status") != "PASS"
        or int(test_evidence.get("base", {}).get("passed", 0)) < 111
        or int(test_evidence.get("gpu_v5_2", {}).get("passed", 0)) < 64
        or int(test_evidence.get("base", {}).get("failed", -1)) != 0
        or int(test_evidence.get("gpu_v5_2", {}).get("failed", -1)) != 0
        or test_evidence.get("phase_c_direction") != direction
        or test_evidence.get("phase_d_authorized") is not False
    ):
        raise Stage2V52ContractError("Phase C full test evidence is incomplete")
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "PASS",
        "phase_c_direction": direction,
        "phase_c_formal_spatial_adopt": bool(decision["adopt"]),
        "phase_d_authorized": False,
        "m5_authorized": False,
        "execution_base_commit": _git_head(root),
        "protocol": {
            "id": "development", "sha256": protocol.digest,
            "train_dates": list(protocol.train_dates),
            "validation_dates": list(protocol.validation_dates),
            "calibration_dates": list(protocol.calibration_dates),
            "evaluation_dates": list(protocol.evaluation_dates),
        },
        "tau_freeze": tau_descriptor,
        "selected_tau": m4_tau_consumption["selected_tau"],
        "selected_candidate": m4_tau_consumption["selected_candidate"],
        "m4_tau_consumption": m4_tau_consumption,
        "selected_checkpoints": checkpoint_descriptors,
        "metrics": model_metrics,
        "comparisons": {
            "M4_vs_M1": _comparison(evaluations["M1"], evaluations["M4"]),
            "M4_vs_M3": _comparison(evaluations["M3"], evaluations["M4"]),
            "M4_vs_M2": _comparison(evaluations["M2"], evaluations["M4"]),
        },
        "directional_gate": decision,
        "pace_guard_M4_vs_M1": pace_guard,
        "temporal_leakage_count": temporal_leakage,
        "test_evidence": {
            "artifact": _descriptor(test_evidence_path, root),
            "base_passed": int(test_evidence["base"]["passed"]),
            "gpu_v5_2_passed": int(test_evidence["gpu_v5_2"]["passed"]),
            "compileall_status": test_evidence["compileall"]["status"],
        },
        "runtime": {
            "measurement": "observed_shell_wall_clock_seconds",
            "coverage": "instrumented_commands_including_reconnected_M0_to_M4_execution",
            "seconds": {
                "transfer_shards": 670.0, "M0_build_train_matrix": 346.4,
                "M0_transform_validation": 56.7, "M0_transform_evaluation": 45.7,
                "M0_train": 107.9, "M0_evaluate_validation_and_evaluation": 54.6,
                "M1_train": 27.3, "M1_evaluate_validation_and_evaluation": 56.8,
                "M2_train": 559.8, "M2_evaluate_validation_and_evaluation": 53.8,
                "M3_train": 467.5, "M3_evaluate_validation_and_evaluation": 57.0,
                "M4_train": 463.5, "M4_evaluate_validation_and_evaluation": 56.9,
            },
            "peak_rss_mb": None,
            "peak_rss_note": "not available: host-wide CIM memory query was permission denied; no estimate substituted",
        },
        "interpretation": (
            "Phase C fails the pre-registered support-aware spatial continuation rule. "
            "Structured representation shows modest value, but M4 misses the low-support strength gate, "
            "fails overall stability, is systematically worse than M2 on unseen targets, and adds no "
            "stable increment over M3. Stop support-aware spatial transfer expansion; this does not "
            "invalidate the Stage 2 source model or the completed experiment."
        ),
    }
    report["artifact_sha256"] = _canonical_hash(report)
    return report


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    targets = tuple(CORE_TRANSFER_TARGETS)
    lines = [
        "# Stage 2 v5.2 Phase C development report", "",
        f"- Direction: `{report['phase_c_direction']}`",
        f"- Formal spatial adopt: `{str(report['phase_c_formal_spatial_adopt']).upper()}`",
        "- Phase D authorized: `NO`", "- M5 authorized: `NO`", "",
        "## Frozen protocol", "",
        f"- Protocol hash: `{report['protocol']['sha256']}`",
        f"- Train: `{report['protocol']['train_dates'][0]}-{report['protocol']['train_dates'][-1]}`",
        f"- Validation: `{report['protocol']['validation_dates'][0]}-{report['protocol']['validation_dates'][-1]}`",
        f"- Calibration: `{report['protocol']['calibration_dates'][0]}`",
        f"- Evaluation: `{report['protocol']['evaluation_dates'][0]}-{report['protocol']['evaluation_dates'][-1]}`",
        f"- Frozen tau: `{report['selected_candidate']} / {report['selected_tau']}`", "",
        "## M4 frozen tau consumption", "",
        f"- Relationship audit: `{report['m4_tau_consumption']['status']}`",
        f"- Tau freeze file SHA: `{report['m4_tau_consumption']['tau_freeze_file_sha256']}`",
        f"- Transfer-tuning support SHA: `{report['m4_tau_consumption']['transfer_tuning_support_sha256']}`",
        f"- Development support SHA: `{report['m4_tau_consumption']['development_support_sha256']}`", "",
        "## Aggregate 20161025-20161027", "",
        "| Model | Acc RMS | Crawl | Speed CV | Stop | Pace P50 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        metric = report["metrics"][model]
        lines.append(
            f"| {model} | {_fmt(metric['overall']['acceleration_rms'])} | "
            f"{_fmt(metric['overall']['crawl'])} | {_fmt(metric['overall']['speed_cv'])} | "
            f"{_fmt(metric['overall']['stop'])} | {_fmt(metric['pace_p50_mae'])} |"
        )
    lines.extend(["", "## Selected checkpoints", "", "| Model | Path | SHA-256 |", "|---|---|---|"])
    for model in MODELS:
        checkpoint = report["selected_checkpoints"][model]
        lines.append(f"| {model} | `{checkpoint['path']}` | `{checkpoint['sha256']}` |")
    for group, title in (("low", "Low-support"), ("unseen", "Unseen")):
        lines.extend(["", f"## {title} MAE", "", "| Model | Acc RMS | Crawl | Speed CV | Stop |", "|---|---:|---:|---:|---:|"])
        for model in MODELS:
            metric = report["metrics"][model][group]
            lines.append(
                f"| {model} | {_fmt(metric['acceleration_rms'])} | {_fmt(metric['crawl'])} | "
                f"{_fmt(metric['speed_cv'])} | {_fmt(metric['stop'])} |"
            )
    lines.extend(["", "## Evaluation support counts", "", "Counts are target-valid unique traversals and are identical across paired models.", "", "| Group | Acc RMS | Crawl | Speed CV | Stop |", "|---|---:|---:|---:|---:|"])
    for group in ("overall", "low", "unseen"):
        counts = report["metrics"]["M4"]["counts"][group]
        lines.append(
            f"| {group} | {counts['acceleration_rms']} | {counts['crawl']} | "
            f"{counts['speed_cv']} | {counts['stop']} |"
        )
    lines.extend(["", "## Per-date overall MAE", ""])
    for date in report["protocol"]["evaluation_dates"]:
        lines.extend([f"### {date}", "", "| Model | Acc RMS | Crawl | Speed CV | Stop | Pace P50 |", "|---|---:|---:|---:|---:|---:|"])
        for model in MODELS:
            day = report["metrics"][model]["per_date"][date]
            lines.append(
                f"| {model} | {_fmt(day['overall']['acceleration_rms'])} | {_fmt(day['overall']['crawl'])} | "
                f"{_fmt(day['overall']['speed_cv'])} | {_fmt(day['overall']['stop'])} | {_fmt(day['pace_p50_mae'])} |"
            )
        lines.append("")
    lines.extend(["## Frozen relative comparisons", "", "Positive percentages favor M4.", "", "| Comparison | Group | Acc RMS | Crawl | Speed CV | Stop | Four-target mean |", "|---|---|---:|---:|---:|---:|---:|"])
    for comparison, groups in (
        ("M4 vs M1", ("overall", "low", "unseen")),
        ("M4 vs M3", ("overall", "low", "unseen")),
        ("M4 vs M2", ("unseen",)),
    ):
        values = report["comparisons"][comparison.replace(" ", "_")]
        for group in groups:
            metric = values[group]
            lines.append(
                f"| {comparison} | {group} | {100 * metric['acceleration_rms']:.3f}% | "
                f"{100 * metric['crawl']:.3f}% | {100 * metric['speed_cv']:.3f}% | "
                f"{100 * metric['stop']:.3f}% | "
                f"{100 * values[f'{group}_four_target_mean_relative_improvement']:.3f}% |"
            )
    lines.append("")
    gate = report["directional_gate"]
    lines.extend([
        "## Directional gate", "",
        f"- Low-support wins vs M1: `{gate['low_support_target_wins']}/4`",
        f"- Low-support mean relative improvement: `{100.0 * gate['low_support_mean_relative_improvement']:.3f}%`",
        f"- Overall no target degrades over 2%: `{str(gate['overall_no_target_degrades_over_2pct']).upper()}`",
        f"- Unseen no worse than M2: `{str(gate['unseen_not_worse_than_structure_only']).upper()}`",
        f"- Pace guard: `{report['pace_guard_M4_vs_M1']['status']}`",
        "- Temporal leakage: `0`", "",
        "The pre-registered continuation gate does not pass. The result is classified as "
        f"`C-{report['phase_c_direction']}`. Structured representation retains modest value, but the "
        "support-aware spatial transfer path has direct counter-evidence from the overall-stability, "
        "unseen-versus-M2, and M3-to-M4 comparisons. Spatial transfer expansion must stop; no retuning "
        "or rerun is authorized.", "",
        "## Runtime and memory", "",
        f"- Instrumented wall-clock total: `{sum(report['runtime']['seconds'].values()):.1f} s`",
        "- Peak RSS: `not available` (the host CIM query was permission denied; no estimate was substituted).", "",
        "## Verification", "",
        f"- Base combined suite: `{report['test_evidence']['base_passed']} passed`",
        f"- GPU v5.2 suite: `{report['test_evidence']['gpu_v5_2_passed']} passed`",
        f"- compileall: `{report['test_evidence']['compileall_status']}`", "",
        "Phase D remains unauthorized. No M5/M6, rolling folds, tau reselection, or 20161028-30 data were run.", "",
    ])
    return "\n".join(lines)


def build_evidence_bundle(
    repo_root: str | Path, *, report_json_path: Path, report_markdown_path: Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    training_paths, validation_paths, evaluation_paths = (
        _training_paths(root), _validation_paths(root), _evaluation_paths(root)
    )
    training = {model: _read_json(path) for model, path in training_paths.items()}
    checkpoints = _checkpoint_paths(root, training)
    base = root / "stage2/output_v5_2/development"
    report = _read_json(report_json_path)
    bundle: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "PASS",
        "phase_c_direction": report["phase_c_direction"],
        "phase_c_authorization_commit": PHASE_C_AUTHORIZATION_COMMIT,
        "execution_base_commit": PHASE_C_AUTHORIZATION_COMMIT,
        "phase_c_execution_commit": PHASE_C_EXECUTION_COMMIT,
        "phase_d_authorized": False,
        "frozen_bindings": {
            "phase_c_authorization": _descriptor(
                root / "stage2/docs/v5_2/stage2_v5_2_phase_c_authorization.json", root,
            ),
            "b1_evidence": _descriptor(
                root / "stage2/docs/v5_2/stage2_v5_2_phase_b1_evidence_bundle.json", root,
            ),
            "tau_freeze": _descriptor(
                root / "stage2/output_v5_2/transfer_tuning/stage2_v5_2_tau_freeze.json", root,
            ),
            "phase_c_execution_config_git": _git_file_descriptor(
                root,
                commit=PHASE_C_AUTHORIZATION_COMMIT,
                path="stage2/config/stage2_v5_2.json",
            ),
            "post_c_frozen_config": _descriptor(root / "stage2/config/stage2_v5_2.json", root),
            "source_config": _descriptor(root / "stage2/config/stage2_v5_1_development.json", root),
        },
        "train_only_artifacts": {
            "support": _descriptor(base / "artifacts/support.json", root),
            "static": _descriptor(base / "artifacts/static.json", root),
            "transfer_manifest": _descriptor(
                root / "stage2/output_v5_2/transfer_shards/protocol=development/transfer_manifest.json", root,
            ),
            "temporal_audit": _descriptor(
                root / "stage2/output_v5_2/transfer_shards/protocol=development/temporal_audit.json", root,
            ),
        },
        "models": {
            model: {
                "training_manifest": _descriptor(training_paths[model], root),
                "selected_checkpoint": _descriptor(checkpoints[model], root),
                "validation_evaluation": _descriptor(validation_paths[model], root),
                "development_evaluation": _descriptor(evaluation_paths[model], root),
            }
            for model in MODELS
        },
        "directional_decision": _descriptor(base / "spatial_direction_decision.json", root),
        "m4_tau_consumption": report["m4_tau_consumption"],
        "reports": {
            "json": _descriptor(report_json_path, root),
            "markdown": _descriptor(report_markdown_path, root),
        },
        "test_evidence": _descriptor(
            root / "stage2/docs/v5_2/stage2_v5_2_phase_c_test_evidence.json", root,
        ),
        "code": {
            name: _descriptor(root / path, root)
            for name, path in {
                "m0_features": "stage2/v5_2/m0_features.py",
                "training": "stage2/v5_2/training.py",
                "evaluation": "stage2/v5_2/evaluation.py",
                "phase_c_report": "stage2/v5_2/phase_c_report.py",
            }.items()
        },
    }
    bundle["artifact_sha256"] = _canonical_hash(bundle)
    return bundle


def build_post_c_status_manifest(
    repo_root: str | Path, *, report_path: Path, evidence_path: Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config_path = root / "stage2/config/stage2_v5_2.json"
    config = _read_json(config_path)
    report = _read_json(report_path)
    evidence = _read_json(evidence_path)
    if (
        config.get("phase") != "PHASE_C_COMPLETE_FROZEN"
        or config.get("execution_authorization") != "NONE_POST_C"
        or config.get("phase_b1_complete") is not True
        or config.get("phase_c_complete") is not True
        or config.get("phase_c_direction") != "FAIL"
        or config.get("phase_d_authorized") is not False
        or config.get("m5_authorized") is not False
        or config.get("current_status") != "PHASE_C_FAIL_FROZEN"
        or report.get("phase_c_direction") != "FAIL"
        or evidence.get("phase_c_direction") != "FAIL"
    ):
        raise Stage2V52ContractError("post-Phase-C freeze state is inconsistent")
    status: dict[str, Any] = {
        "schema_version": POST_C_STATUS_SCHEMA_VERSION,
        "status": "PHASE_C_FAIL_FROZEN",
        "current_status": "PHASE_C_FAIL_FROZEN",
        "branch": "codex/stage2-v5-micro-transfer",
        "phase": "PHASE_C_COMPLETE_FROZEN",
        "phase_c_execution_commit": PHASE_C_EXECUTION_COMMIT,
        "freeze_correction_base_commit": _git_head(root),
        "config": _descriptor(config_path, root),
        "execution_authorization": "NONE_POST_C",
        "phase_b1_complete": True,
        "phase_c_authorized": False,
        "phase_c_was_authorized": True,
        "phase_c_complete": True,
        "phase_c_direction": "FAIL",
        "phase_d_authorized": False,
        "m5_authorized": False,
        "formal_spatial_adopt": False,
        "rerun_training_required": False,
        "rerun_inference_required": False,
        "tau_reselection_allowed": False,
        "tau_freeze": report["tau_freeze"],
        "selected_candidate": report["selected_candidate"],
        "selected_tau": report["selected_tau"],
        "m4_tau_consumption": report["m4_tau_consumption"],
        "phase_c_report": _descriptor(report_path, root),
        "phase_c_evidence_bundle": _descriptor(evidence_path, root),
        "phase_c_test_evidence": report["test_evidence"]["artifact"],
        "b1_evidence_bundle": _descriptor(
            root / "stage2/docs/v5_2/stage2_v5_2_phase_b1_evidence_bundle.json", root,
        ),
        "stage2_final_ready": False,
        "formal_release_available": False,
        "formal_release_requires": "PHASE_D_COMPLETE",
        "scientific_conclusion": (
            "Structured representation has modest value, but fixed n/(n+tau) support-aware "
            "spatial gating has no reliable incremental benefit; stop spatial transfer expansion."
        ),
    }
    status["artifact_sha256"] = _canonical_hash(status)
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output-json", default="stage2/docs/v5_2/stage2_v5_2_phase_c_report.json",
    )
    parser.add_argument(
        "--output-markdown", default="stage2/docs/v5_2/stage2_v5_2_phase_c_report.md",
    )
    parser.add_argument(
        "--output-evidence", default="stage2/docs/v5_2/stage2_v5_2_phase_c_evidence_bundle.json",
    )
    parser.add_argument(
        "--output-status", default="stage2/docs/v5_2/stage2_v5_2_status_manifest.json",
    )
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    json_path, markdown_path, evidence_path, status_path = (
        root / args.output_json, root / args.output_markdown, root / args.output_evidence,
        root / args.output_status,
    )
    report = build_phase_c_report(root)
    _atomic_json(json_path, report)
    _atomic_text(markdown_path, render_markdown(report))
    evidence = build_evidence_bundle(
        root, report_json_path=json_path, report_markdown_path=markdown_path,
    )
    _atomic_json(evidence_path, evidence)
    verification = verify_phase_c_evidence_bundle(evidence, repo_root=root)
    status = build_post_c_status_manifest(
        root, report_path=json_path, evidence_path=evidence_path,
    )
    _atomic_json(status_path, status)
    print(json.dumps({
        "status": "PASS", "phase_c_direction": report["phase_c_direction"],
        "report_sha256": sha256_path(json_path), "evidence_sha256": sha256_path(evidence_path),
        "resolved_artifact_count": verification["resolved_artifact_count"],
        "status_manifest_sha256": sha256_path(status_path),
        "phase_d_authorized": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
