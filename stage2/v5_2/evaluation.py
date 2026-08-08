"""Formal checkpoint inference, unique-traversal evaluation, and adoption manifests."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import CORE_TRANSFER_TARGETS, Stage2V52ContractError
from .feature_binding import sha256_path
from .micro_metrics import (
    decide_spatial_transfer, decide_temporal_adapter, pace_stability,
    relative_error_improvement,
)
from .protocols import get_protocol, protocol_role_dates
from .training import (
    CORE_METRIC_DEFINITION,
    PACE_METRIC_DEFINITION,
    TRAINING_SCHEMA_VERSION,
    _protocol_root,
    _protocol_shards,
    collect_unique_predictions,
    initialized_transfer_model,
    unique_traversal_metrics,
)


EVALUATION_SCHEMA_VERSION = "stage2_v5_2_evaluation.2"
TAU_METRICS_SCHEMA_VERSION = "stage2_v5_2_tau_evaluation.2"
ADOPTION_SCHEMA_VERSION = "stage2_v5_2_adoption.2"
ROLLING_SPATIAL_ADOPTION_SCHEMA_VERSION = "stage2_v5_2_rolling_spatial_adoption.1"


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage2V52ContractError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _code_sha256() -> str:
    return sha256_path(Path(__file__))


def predict_checkpoint(
    *, protocol_id: str, model_id: str, role: str, tensor_root: str | Path,
    checkpoint_path: str | Path, training_manifest_path: str | Path, batch_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one selected checkpoint and merge overlap predictions by physical traversal."""
    import torch

    protocol = get_protocol(protocol_id)
    training = _json(training_manifest_path)
    if (
        training.get("schema_version") != TRAINING_SCHEMA_VERSION
        or training.get("status") != "PASS"
        or training.get("protocol_id") != protocol_id
        or training.get("model_id") != model_id
        or training.get("protocol_hash") != protocol.digest
    ):
        raise Stage2V52ContractError("training manifest is not the selected protocol/model run")
    checkpoint_sha = sha256_path(checkpoint_path)
    if training.get("selected_checkpoint_sha256") != checkpoint_sha:
        raise Stage2V52ContractError("evaluation checkpoint hash differs from training selection")
    source = training.get("source", {})
    constructor = training.get("constructor", {})
    model, source_binding = initialized_transfer_model(
        protocol_id=protocol_id,
        feature_artifact_path=source["feature_artifact_path"],
        checkpoint_path=source["source_checkpoint_path"],
        source_model_manifest_path=source["source_model_manifest_path"],
        source_config_path=source["source_config_path"],
        static_feature_count=int(constructor["static_feature_count"]),
        support_tau=float(constructor["support_tau"]),
        spatial_mode=str(constructor["spatial_mode"]),
        temporal_mode=str(constructor["temporal_mode"]),
        backbone_kwargs=dict(constructor["backbone_kwargs"]),
    )
    if model_id != "M1":
        saved = torch.load(checkpoint_path, map_location="cpu")
        state = saved.get("model_state_dict", saved)
        model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tensor_manifest_path = _protocol_root(Path(tensor_root), protocol_id) / "transfer_manifest.json"
    if sha256_path(tensor_manifest_path) != training.get("tensor_manifest_sha256"):
        raise Stage2V52ContractError("evaluation tensor manifest differs from training")
    paths = _protocol_shards(Path(tensor_root), protocol_id, role)
    component_weights = training.get("loss_policy", {}).get("component_weights", {})
    predictions, diagnostics = collect_unique_predictions(
        model, paths, batch_size, device, component_weights=component_weights,
    )
    expected_dates = protocol_role_dates(protocol_id)[role]
    if tuple(sorted(predictions["date"].astype(str).unique())) != tuple(sorted(expected_dates)):
        raise Stage2V52ContractError("prediction dates differ from frozen protocol role")
    return predictions, {
        **diagnostics,
        "protocol_id": protocol_id, "protocol_hash": protocol.digest,
        "model_id": model_id, "role": role, "evaluation_dates": list(expected_dates),
        "checkpoint_sha256": checkpoint_sha,
        "training_manifest_sha256": sha256_path(training_manifest_path),
        "training_manifest_path": Path(training_manifest_path).as_posix(),
        "tensor_manifest_sha256": sha256_path(tensor_manifest_path),
        "source_checkpoint_sha256": source_binding["source_checkpoint_sha256"],
        "feature_artifact_sha256": source_binding["feature_artifact_sha256"],
        "static_artifact_sha256": training["static_artifact_sha256"],
        "support_artifact_sha256": training["support_artifact_sha256"],
        "support_tau": float(constructor["support_tau"]),
    }


def evaluate_checkpoint(
    *, protocol_id: str, model_id: str, role: str, tensor_root: str | Path,
    checkpoint_path: str | Path, training_manifest_path: str | Path,
    output_root: str | Path, batch_size: int,
) -> dict[str, Any]:
    predictions, provenance = predict_checkpoint(
        protocol_id=protocol_id, model_id=model_id, role=role, tensor_root=tensor_root,
        checkpoint_path=checkpoint_path, training_manifest_path=training_manifest_path,
        batch_size=batch_size,
    )
    metrics = unique_traversal_metrics(predictions)
    by_date = {
        str(date): unique_traversal_metrics(day)
        for date, day in predictions.groupby("date", sort=True, observed=True)
    }
    output = Path(output_root)
    prediction_path = output / "unique_traversal_predictions.parquet"
    _atomic_parquet(prediction_path, predictions)
    core_mae = {
        target: metrics["groups"]["overall"][target]["mae"] for target in CORE_TRANSFER_TARGETS
    }
    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION, "status": "PASS",
        **provenance,
        "core_metric_definition": CORE_METRIC_DEFINITION,
        "pace_metric_definition": PACE_METRIC_DEFINITION,
        "core_mae": core_mae,
        "low_support_core_mae": {
            target: metrics["groups"]["low"][target]["mae"] for target in CORE_TRANSFER_TARGETS
        },
        "unseen_core_mae": {
            target: metrics["groups"]["unseen"][target]["mae"] for target in CORE_TRANSFER_TARGETS
        },
        "pace_p50_mae": metrics["pace_p50"]["mae"],
        "metrics_by_support": metrics["groups"],
        "metrics_by_date": by_date,
        "pace_guard_input": metrics["pace_p50"],
        "rts_role": "legacy_descriptive_diagnostic_not_stage3_deployable",
        "rts_stage3_deployable": False,
        "adoption_targets": list(CORE_TRANSFER_TARGETS),
        "prediction_path": prediction_path.as_posix(),
        "prediction_sha256": sha256_path(prediction_path),
        "evaluation_code_sha256": _code_sha256(),
        "evaluation_schema": "unique_traversal_overlap_mean_then_stratified_mae.1",
    }
    required_group_values = [
        report[name][target]
        for name in ("core_mae", "low_support_core_mae", "unseen_core_mae")
        for target in CORE_TRANSFER_TARGETS
    ]
    if any(value is None for value in required_group_values) or report["pace_p50_mae"] is None:
        report["status"] = "FAIL_INSUFFICIENT_SUPPORT"
    report_path = output / "evaluation_manifest.json"
    _atomic_json(report_path, report)
    return report


def validate_evaluation_payload(payload: Mapping[str, Any], *, protocol_id: str) -> None:
    protocol = get_protocol(protocol_id)
    if payload.get("schema_version") != EVALUATION_SCHEMA_VERSION or payload.get("status") != "PASS":
        raise Stage2V52ContractError("adoption requires a successful formal evaluator manifest")
    if payload.get("protocol_id") != protocol_id or payload.get("protocol_hash") != protocol.digest:
        raise Stage2V52ContractError("evaluation protocol provenance mismatch")
    role = str(payload.get("role", ""))
    if role not in protocol_role_dates(protocol_id):
        raise Stage2V52ContractError("evaluation manifest has an invalid canonical role")
    expected_dates = protocol_role_dates(protocol_id)[role]
    if tuple(payload.get("evaluation_dates", ())) != expected_dates:
        raise Stage2V52ContractError("evaluation dates differ from frozen protocol role")
    if (
        payload.get("rts_role") != "legacy_descriptive_diagnostic_not_stage3_deployable"
        or payload.get("rts_stage3_deployable") is not False
    ):
        raise Stage2V52ContractError("RTS must remain a non-deployable frozen-reference diagnostic")
    if set(payload.get("adoption_targets", ())) != set(CORE_TRANSFER_TARGETS):
        raise Stage2V52ContractError("adoption target set must exclude RTS and pace")
    for key in (
        "checkpoint_sha256", "training_manifest_sha256", "tensor_manifest_sha256",
        "source_checkpoint_sha256", "feature_artifact_sha256", "prediction_sha256",
        "evaluation_code_sha256", "support_artifact_sha256", "static_artifact_sha256",
    ):
        if not isinstance(payload.get(key), str) or len(str(payload[key])) != 64:
            raise Stage2V52ContractError(f"evaluation manifest lacks hash provenance: {key}")


def build_tau_metrics_manifest(
    *, m1_evaluation_path: str | Path, m4_evaluation_paths: Sequence[str | Path],
    support_artifact_path: str | Path, feature_artifact_path: str | Path,
) -> dict[str, Any]:
    protocol = get_protocol("transfer_tuning")
    m1 = _json(m1_evaluation_path)
    validate_evaluation_payload(m1, protocol_id="transfer_tuning")
    if m1.get("model_id") != "M1" or m1.get("role") != "validation":
        raise Stage2V52ContractError("tau M1 provenance must be transfer-tuning validation")
    candidates: dict[str, Any] = {}
    for path in m4_evaluation_paths:
        payload = _json(path)
        validate_evaluation_payload(payload, protocol_id="transfer_tuning")
        if payload.get("model_id") != "M4" or payload.get("role") != "validation":
            raise Stage2V52ContractError("tau candidates must be formal M4 validation evaluations")
        tau = payload.get("support_tau")
        if tau is None:
            raise Stage2V52ContractError("M4 evaluation manifest does not bind support_tau")
        key = str(float(tau))
        if key in candidates:
            raise Stage2V52ContractError("duplicate tau evaluation candidate")
        candidates[key] = {
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "unique_traversal_count": payload["unique_traversal_count"],
            "core_mae": payload["core_mae"],
            "evaluation_manifest_sha256": sha256_path(path),
        }
    support = _json(support_artifact_path)
    expected = {float(value) for value in support.get("tau_candidates", ())}
    observed = {float(value) for value in candidates}
    if observed != expected or len(expected) != 3:
        raise Stage2V52ContractError("tau evaluations must cover exactly support P25/P50/P75")
    support_sha = sha256_path(support_artifact_path)
    feature_sha = sha256_path(feature_artifact_path)
    for path in m4_evaluation_paths:
        payload = _json(path)
        if payload.get("feature_artifact_sha256") != feature_sha:
            raise Stage2V52ContractError("tau candidate feature artifact differs from frozen source")
        if payload.get("support_artifact_sha256") != support_sha:
            raise Stage2V52ContractError("tau candidate support artifact differs from frozen source")
    if m1.get("feature_artifact_sha256") != feature_sha:
        raise Stage2V52ContractError("tau M1 feature artifact differs from frozen source")
    if m1.get("support_artifact_sha256") != support_sha:
        raise Stage2V52ContractError("tau M1 support artifact differs from frozen source")
    return {
        "schema_version": TAU_METRICS_SCHEMA_VERSION,
        "status": "PASS", "protocol_id": "transfer_tuning", "protocol_hash": protocol.digest,
        "train_dates": list(protocol.train_dates), "validation_dates": list(protocol.validation_dates),
        "m1_source_checkpoint_sha256": m1["source_checkpoint_sha256"],
        "m1_checkpoint_sha256": m1["checkpoint_sha256"],
        "m1_unique_traversal_count": m1["unique_traversal_count"],
        "m1_core_mae": m1["core_mae"],
        "m1_evaluation_manifest_sha256": sha256_path(m1_evaluation_path),
        "m4_candidates": candidates,
        "support_artifact_sha256": support_sha,
        "support_artifact_embedded_sha256": support.get("artifact_sha256"),
        "feature_artifact_sha256": feature_sha,
        "core_metric_definition": CORE_METRIC_DEFINITION,
        "evaluation_code_sha256": _code_sha256(),
        "evaluation_schema": "unique_traversal_overlap_mean_then_stratified_mae.1",
    }


def evaluate_spatial_adoption(
    *, m1: Mapping[str, Any], m2: Mapping[str, Any], m4: Mapping[str, Any],
) -> dict[str, Any]:
    for payload in (m1, m2, m4):
        validate_evaluation_payload(payload, protocol_id=str(m4["protocol_id"]))
    if (m1.get("model_id"), m2.get("model_id"), m4.get("model_id")) != ("M1", "M2", "M4"):
        raise Stage2V52ContractError("spatial adoption requires formal M1/M2/M4 evaluations")
    paired_fields = (
        "protocol_id", "protocol_hash", "role", "evaluation_dates",
        "tensor_manifest_sha256", "feature_artifact_sha256", "support_artifact_sha256",
        "static_artifact_sha256", "source_checkpoint_sha256",
    )
    mismatches = [field for field in paired_fields if m1.get(field) != m2.get(field) or m1.get(field) != m4.get(field)]
    if mismatches:
        raise Stage2V52ContractError(f"spatial adoption is not a paired comparison: {mismatches}")
    for payload in (m1, m2, m4):
        for field in ("core_mae", "low_support_core_mae", "unseen_core_mae"):
            if any(payload.get(field, {}).get(target) is None for target in CORE_TRANSFER_TARGETS):
                raise Stage2V52ContractError("spatial adoption has an insufficient-support metric group")
    low = {
        target: relative_error_improvement(
            float(m1["low_support_core_mae"][target]), float(m4["low_support_core_mae"][target])
        )
        for target in CORE_TRANSFER_TARGETS
    }
    overall = {
        target: relative_error_improvement(
            float(m1["core_mae"][target]), float(m4["core_mae"][target])
        )
        for target in CORE_TRANSFER_TARGETS
    }
    for payload in (m2, m4):
        if any(not np.isfinite(float(payload["unseen_core_mae"][target])) for target in CORE_TRANSFER_TARGETS):
            raise Stage2V52ContractError("spatial adoption unseen MAE must be finite")
    result = decide_spatial_transfer(
        low_support_improvement_by_target=low, overall_improvement_by_target=overall,
        unseen_candidate_error=m4["unseen_core_mae"],
        unseen_structure_only_error=m2["unseen_core_mae"],
    )
    return {
        "schema_version": ADOPTION_SCHEMA_VERSION,
        "status": "PASS", "verification_status": "PASS", "protocol_id": m4["protocol_id"],
        "role": m4["role"], "evaluation_dates": list(m4["evaluation_dates"]),
        "source_evaluation_hashes": {
            "M1": _canonical_manifest_hash(m1), "M2": _canonical_manifest_hash(m2),
            "M4": _canonical_manifest_hash(m4),
        },
        **result,
    }


def evaluate_rolling_spatial_adoption(
    *,
    m1_evaluations: Sequence[Mapping[str, Any]],
    m2_evaluations: Sequence[Mapping[str, Any]],
    m4_evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the formal M4 adoption gate only after three paired rolling folds."""
    if not (len(m1_evaluations) == len(m2_evaluations) == len(m4_evaluations) == 3):
        raise Stage2V52ContractError("rolling spatial adoption requires three M1/M2/M4 folds")
    by_model = {
        model: {str(payload.get("protocol_id", "")): payload for payload in values}
        for model, values in (("M1", m1_evaluations), ("M2", m2_evaluations), ("M4", m4_evaluations))
    }
    expected_folds = {"fold_1", "fold_2", "fold_3"}
    if any(set(values) != expected_folds for values in by_model.values()):
        raise Stage2V52ContractError("rolling spatial adoption requires exactly fold_1/fold_2/fold_3")
    values: dict[str, dict[str, list[float]]] = {
        model: {group: [] for group in ("overall", "low", "unseen")}
        for model in ("M1", "M2", "M4")
    }
    provenance: dict[str, dict[str, str]] = {}
    evaluation_dates: list[str] = []
    selected_m4: dict[str, str] = {}
    for fold in sorted(expected_folds):
        m1, m2, m4 = (by_model[model][fold] for model in ("M1", "M2", "M4"))
        single = evaluate_spatial_adoption(m1=m1, m2=m2, m4=m4)
        if m4.get("role") != "evaluation":
            raise Stage2V52ContractError("formal rolling spatial adoption requires evaluation role")
        provenance[fold] = dict(single["source_evaluation_hashes"])
        selected_m4[fold] = str(m4["checkpoint_sha256"])
        for date in m4["evaluation_dates"]:
            if date in evaluation_dates:
                raise Stage2V52ContractError("rolling spatial evaluation dates overlap")
            evaluation_dates.append(str(date))
            for model, payload in (("M1", m1), ("M2", m2), ("M4", m4)):
                day = payload.get("metrics_by_date", {}).get(date, {}).get("groups", {})
                for group in ("overall", "low", "unseen"):
                    for target in CORE_TRANSFER_TARGETS:
                        mae = day.get(group, {}).get(target, {}).get("mae")
                        if mae is None or not np.isfinite(float(mae)):
                            raise Stage2V52ContractError("rolling spatial daily MAE is insufficient")
                        values[model][group].append((target, float(mae)))
    if len(evaluation_dates) != 6:
        raise Stage2V52ContractError("rolling spatial adoption requires six distinct evaluation dates")

    def means(model: str, group: str) -> dict[str, float]:
        return {
            target: float(np.mean([value for name, value in values[model][group] if name == target]))
            for target in CORE_TRANSFER_TARGETS
        }

    m1_low, m1_overall = means("M1", "low"), means("M1", "overall")
    m4_low, m4_overall = means("M4", "low"), means("M4", "overall")
    result = decide_spatial_transfer(
        low_support_improvement_by_target={
            target: relative_error_improvement(m1_low[target], m4_low[target])
            for target in CORE_TRANSFER_TARGETS
        },
        overall_improvement_by_target={
            target: relative_error_improvement(m1_overall[target], m4_overall[target])
            for target in CORE_TRANSFER_TARGETS
        },
        unseen_candidate_error=means("M4", "unseen"),
        unseen_structure_only_error=means("M2", "unseen"),
    )
    return {
        "schema_version": ROLLING_SPATIAL_ADOPTION_SCHEMA_VERSION,
        "status": "PASS", "verification_status": "PASS",
        "protocol_id": "rolling_origin_fold_1_2_3",
        "decision_scope": "rolling_origin_three_fold_six_dates",
        "role": "evaluation", "evaluation_dates": sorted(evaluation_dates),
        "source_evaluation_hashes": provenance,
        "selected_m4_checkpoint_sha256_by_protocol": selected_m4,
        **result,
    }


def _canonical_manifest_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def evaluate_temporal_adoption(
    *, m4_evaluations: Sequence[Mapping[str, Any]], m5_evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Decide M5 only from paired formal rolling-origin evaluation manifests."""
    if len(m4_evaluations) != 3 or len(m5_evaluations) != 3:
        raise Stage2V52ContractError("temporal adoption requires M4/M5 results for exactly three rolling folds")
    daily: dict[str, list[float]] = {}
    target_values: dict[str, list[float]] = {target: [] for target in CORE_TRANSFER_TARGETS}
    provenance: dict[str, dict[str, str]] = {}
    for m4, m5 in zip(m4_evaluations, m5_evaluations):
        protocol_id = str(m4.get("protocol_id", ""))
        if protocol_id not in {"fold_1", "fold_2", "fold_3"} or m5.get("protocol_id") != protocol_id:
            raise Stage2V52ContractError("temporal adoption fold pairing is invalid")
        validate_evaluation_payload(m4, protocol_id=protocol_id)
        validate_evaluation_payload(m5, protocol_id=protocol_id)
        if m4.get("model_id") != "M4" or m5.get("model_id") != "M5" or m4.get("role") != "evaluation" or m5.get("role") != "evaluation":
            raise Stage2V52ContractError("temporal adoption requires formal M4/M5 evaluation roles")
        if set(m4.get("metrics_by_date", {})) != set(m5.get("metrics_by_date", {})):
            raise Stage2V52ContractError("M4/M5 daily evaluation support differs")
        provenance[protocol_id] = {"M4": _canonical_manifest_hash(m4), "M5": _canonical_manifest_hash(m5)}
        for date in sorted(m4["metrics_by_date"]):
            improvements: list[float] = []
            for target in CORE_TRANSFER_TARGETS:
                baseline = m4["metrics_by_date"][date]["groups"]["overall"][target]["mae"]
                candidate = m5["metrics_by_date"][date]["groups"]["overall"][target]["mae"]
                if baseline is None or candidate is None or float(baseline) <= 0:
                    raise Stage2V52ContractError("temporal adoption daily metric is insufficient")
                value = (float(baseline) - float(candidate)) / float(baseline)
                improvements.append(value); target_values[target].append(value)
            daily[date] = improvements
    if len(daily) != 6:
        raise Stage2V52ContractError("temporal adoption requires exactly six distinct rolling evaluation dates")
    daily_mean = {date: float(np.mean(daily[date])) for date in sorted(daily)}
    target_mean = {target: float(np.mean(values)) for target, values in target_values.items()}
    result = decide_temporal_adapter(daily_mean, target_mean)
    return {
        "schema_version": ADOPTION_SCHEMA_VERSION, "status": "PASS", "verification_status": "PASS",
        "protocol_id": "rolling_origin_fold_1_2_3", "source_evaluation_hashes": provenance,
        "evaluation_dates": sorted(daily), "daily_mean_improvements": daily_mean,
        "target_mean_improvements": target_mean, **result,
    }


def evaluate_pace_guard(payload: Mapping[str, Any]) -> dict[str, Any]:
    return pace_stability(float(payload["candidate_pace_p50_mae"]), float(payload["v5_1_pace_p50_mae"]))
