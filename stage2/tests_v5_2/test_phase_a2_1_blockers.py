from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stage2.v5_2.cli import _resolve_training_tau, main
from stage2.v5_2.contracts import CORE_TRANSFER_TARGETS, Stage2V52ContractError, validate_stage3_fields
from stage2.v5_2.evaluation import (
    evaluate_rolling_spatial_adoption, evaluate_spatial_adoption, evaluate_temporal_adoption,
)
from stage2.v5_2.feature_binding import (
    SOURCE_BACKBONE_KEYS, V51_SOURCE_PROTOCOL_NAMES, bind_v51_source_model,
)
from stage2.v5_2.performance import benchmark_kernel_devices
from stage2.v5_2.protocols import get_protocol
from stage2.v5_2.support_transfer import fit_train_support
from stage2.v5_2.training import (
    LOSS_COMPONENT_KEYS, validate_loss_weight_schema, validate_m5_m4_adoption,
)
from stage2.v5_2.verification import FINAL_REQUIRED_GATES, verify_artifact_payload, verify_final_gate_bundle
from stage2.v5_2.verification import sha256_file


def _hash(char: str) -> str:
    return char * 64


def _evaluation(protocol_id: str, model_id: str, error: float) -> dict[str, object]:
    protocol = get_protocol(protocol_id)
    by_date = {
        date: {
            "groups": {
                group: {
                    target: {"count": 10, "mae": error}
                    for target in CORE_TRANSFER_TARGETS
                }
                for group in ("overall", "low", "unseen")
            }
        }
        for date in protocol.evaluation_dates
    }
    return {
        "schema_version": "stage2_v5_2_evaluation.2", "status": "PASS",
        "protocol_id": protocol_id, "protocol_hash": protocol.digest,
        "model_id": model_id, "role": "evaluation",
        "evaluation_dates": list(protocol.evaluation_dates),
        "checkpoint_sha256": _hash(model_id[-1].lower()),
        "training_manifest_sha256": _hash("t"), "tensor_manifest_sha256": _hash("x"),
        "source_checkpoint_sha256": _hash("s"), "feature_artifact_sha256": _hash("f"),
        "prediction_sha256": _hash("p"), "evaluation_code_sha256": _hash("e"),
        "support_artifact_sha256": _hash("u"), "static_artifact_sha256": _hash("a"),
        "rts_role": "legacy_descriptive_diagnostic_not_stage3_deployable",
        "rts_stage3_deployable": False,
        "adoption_targets": list(CORE_TRANSFER_TARGETS),
        "core_mae": {target: error for target in CORE_TRANSFER_TARGETS},
        "low_support_core_mae": {target: error for target in CORE_TRANSFER_TARGETS},
        "unseen_core_mae": {target: error for target in CORE_TRANSFER_TARGETS},
        "metrics_by_date": by_date,
    }


def test_real_v51_fold_protocol_name_mapping() -> None:
    assert V51_SOURCE_PROTOCOL_NAMES == {
        "transfer_tuning": "v5_1_rolling_fold_1_diagnostic",
        "development": "development_temporal_evaluation",
        "fold_1": "v5_1_rolling_fold_1_diagnostic",
        "fold_2": "v5_1_rolling_fold_2_diagnostic",
        "fold_3": "v5_1_rolling_fold_3_diagnostic",
        "legacy_31": "v5_1_legacy_frozen_benchmark",
    }


def test_real_v51_source_manifest_without_protocol_field(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from stage2.v5.models.rc_mstnet_v5 import RCMSTNetV5
    from stage2.v5.shards import CATEGORY_NAMES, RESERVED_TOKENS

    sizes = tuple(6 for _ in CATEGORY_NAMES)
    vocabularies = {}
    for name in CATEGORY_NAMES:
        tokens = {token: index for index, token in enumerate(RESERVED_TOKENS)}
        tokens.update({f"{name}_{index}": index for index in range(len(tokens), 6)})
        vocabularies[name] = {"token_to_index": tokens, "seen_tokens": []}
    protocol = get_protocol("fold_1")
    feature = tmp_path / "feature.json"
    feature.write_text(json.dumps({
        "fit_dates": list(protocol.train_dates), "numeric_features": ["x", "y"],
        "vocabularies": vocabularies,
    }), encoding="utf-8")
    backbone = {
        "hidden_dim": 16, "categorical_embedding_dim": 4, "transformer_layers": 1,
        "attention_heads": 4, "dropout": 0.0, "minimum_log_scale": -5.0,
        "maximum_log_scale": 2.0, "distribution_family": "monotonic_quantiles",
        "maximum_log_p50": 1.0, "maximum_log_p90_p50_ratio": 2.0,
        "maximum_log_p95_p90_ratio": 1.0, "history_mode": "ordinary_concatenation",
    }
    model = RCMSTNetV5(numeric_feature_count=2, categorical_sizes=sizes, **backbone)
    checkpoint = tmp_path / "model.pt"
    checkpoint_config = {**backbone, "numeric_feature_count": 2}
    torch.save({"model_state_dict": model.state_dict(), "model_config": checkpoint_config}, checkpoint)
    from stage2.v5_2.feature_binding import sha256_path
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "stage2_v5_rc_mstnet.1", "status": "PASS",
        "model_id": "real-frozen-fixture", "checkpoint_sha256": sha256_path(checkpoint),
        "fit_dates": list(protocol.train_dates), "validation_dates": list(protocol.validation_dates),
        "history_mode": backbone["history_mode"],
        "distribution_family": backbone["distribution_family"],
    }), encoding="utf-8")
    source_config = tmp_path / "source.json"
    source_config.write_text(json.dumps({
        "split": {
            "protocol_name": V51_SOURCE_PROTOCOL_NAMES["fold_1"],
            "train_dates": list(protocol.train_dates),
            "validation_model_dates": list(protocol.validation_dates),
        },
        "distribution": {"family": backbone["distribution_family"]},
        "deep": {"dropout": backbone["dropout"]},
    }), encoding="utf-8")
    binding, _ = bind_v51_source_model(
        protocol_id="fold_1", feature_artifact_path=feature,
        source_checkpoint_path=checkpoint, source_model_manifest_path=manifest,
        source_config_path=source_config,
        backbone_kwargs={key: backbone[key] for key in SOURCE_BACKBONE_KEYS},
    )
    assert binding.source_protocol_name == "v5_1_rolling_fold_1_diagnostic"


def test_transfer_tuning_m4_requires_p25_p50_p75_enum() -> None:
    args = argparse.Namespace(
        model="M4", protocol="transfer_tuning", tau_candidate=None,
        tau_artifact=None, support_artifact=None,
    )
    with pytest.raises(Stage2V52ContractError):
        _resolve_training_tau(args, {})


def test_non_tuning_m4_requires_frozen_tau() -> None:
    args = argparse.Namespace(
        model="M4", protocol="fold_1", tau_candidate=None,
        tau_artifact=None, support_artifact=None,
    )
    with pytest.raises(Stage2V52ContractError):
        _resolve_training_tau(args, {})


def test_loss_weight_schema_matches_components_and_unknown_fails() -> None:
    weights = {name: 0.0 for name in LOSS_COMPONENT_KEYS}
    assert set(validate_loss_weight_schema(weights)) == set(LOSS_COMPONENT_KEYS)
    with pytest.raises(Stage2V52ContractError):
        validate_loss_weight_schema({**weights, "pace": 1.0})
    incomplete = dict(weights); incomplete.pop("stop_positive")
    with pytest.raises(Stage2V52ContractError):
        validate_loss_weight_schema(incomplete)


def test_spatial_adoption_role_date_artifact_mismatch_fails() -> None:
    m1, m2, m4 = (_evaluation("fold_1", model, error) for model, error in (("M1", 1.0), ("M2", 1.0), ("M4", 0.9)))
    m1["role"] = "validation"
    m1["evaluation_dates"] = list(get_protocol("fold_1").validation_dates)
    with pytest.raises(Stage2V52ContractError):
        evaluate_spatial_adoption(m1=m1, m2=m2, m4=m4)


def test_adopted_m4_can_initialize_m5_only_from_rolling_decision() -> None:
    m1 = [_evaluation(f"fold_{index}", "M1", 1.0) for index in range(1, 4)]
    m2 = [_evaluation(f"fold_{index}", "M2", 1.0) for index in range(1, 4)]
    m4 = [_evaluation(f"fold_{index}", "M4", 0.9) for index in range(1, 4)]
    decision = evaluate_rolling_spatial_adoption(
        m1_evaluations=m1, m2_evaluations=m2, m4_evaluations=m4,
    )
    assert decision["status"] == "PASS" and decision["adopt"] is True
    validate_m5_m4_adoption(
        decision, protocol_id="fold_1", m4_checkpoint_sha256=_hash("4"),
    )


def test_temporal_adoption_accepts_real_three_fold_six_date_mapping() -> None:
    m4 = [_evaluation(f"fold_{index}", "M4", 1.0) for index in range(1, 4)]
    m5 = [_evaluation(f"fold_{index}", "M5", 0.8) for index in range(1, 4)]
    selected = {f"fold_{index}": m4[index - 1]["checkpoint_sha256"] for index in range(1, 4)}
    adoption_sha = _hash("d")
    adoption = {
        "schema_version": "stage2_v5_2_rolling_spatial_adoption.1", "status": "PASS",
        "verification_status": "PASS", "protocol_id": "rolling_origin_fold_1_2_3",
        "decision_scope": "rolling_origin_three_fold_six_dates", "adopt": True,
        "selected_m4_checkpoint_sha256_by_protocol": selected,
    }
    for baseline, candidate in zip(m4, m5):
        candidate.update({
            "parent_m4_checkpoint_sha256": baseline["checkpoint_sha256"],
            "parent_m4_adoption_sha256": adoption_sha,
        })
    decision = evaluate_temporal_adoption(
        m4_evaluations=m4, m5_evaluations=m5,
        m4_adoption_manifest=adoption, m4_adoption_manifest_sha256=adoption_sha,
    )
    assert isinstance(decision["daily_mean_improvements"], dict)
    assert len(decision["daily_mean_improvements"]) == 6


def test_real_support_payload_passes_verifier() -> None:
    payload = fit_train_support(["a", "a", "b"], fit_dates=["20161009"]).to_payload()
    verify_artifact_payload(payload, artifact_type="support")


def test_m0_validation_transform_uses_train_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from stage2.v5_2 import m0_features

    def fake_batches(_root: Path, date: str, **_: object):
        yield pd.DataFrame({
            "date": [date], "order_id": [f"o-{date}"], "traversal_id": [1],
            "observed_directed_edge_uid": ["edge-a"], "feature_a": [np.nan],
            "crawl_time_share": [0.1], "crawl_target_valid": [True],
            "stop_time_share": [0.2], "stop_target_valid": [True],
            "speed_cv_bounded": [0.3], "speed_cv_target_valid": [True],
            "acceleration_rms_bounded": [0.4], "acceleration_rms_target_valid": [True],
            "rts_raw": [0.5], "rts_target_valid": [True],
        })

    monkeypatch.setattr(m0_features, "_iter_m0_day_batches", fake_batches)
    monkeypatch.setattr(m0_features, "_source_hashes", lambda *args: {"fixture": True})
    monkeypatch.setattr(m0_features, "_projected_route_rows", lambda _root, dates: len(tuple(dates)))
    schema = {
        "feature_names": ["feature_a"], "median": {"feature_a": 7.0},
        "missing_policy": "Train_partition_median_fitted_on_exact_protocol_dates", "dtype": "float32",
    }
    train_manifest = tmp_path / "train.json"
    train_manifest.write_text(json.dumps({
        "schema_version": "stage2_v5_2_m0_matrix.2", "protocol_id": "transfer_tuning",
        "fit_scope": "train_only", "feature_schema": schema, "feature_schema_hash": "schema-hash",
    }), encoding="utf-8")
    support = fit_train_support(["edge-a"], fit_dates=["20161009"]).to_payload()
    support["protocol_id"] = "transfer_tuning"
    support_path = tmp_path / "support.json"
    support_path.write_text(json.dumps(support), encoding="utf-8")
    matrix, manifest = tmp_path / "validation.npz", tmp_path / "validation.json"
    result = m0_features.transform_m0_feature_matrix(
        protocol_id="transfer_tuning", role="validation", repo_root=tmp_path,
        route_feature_root=tmp_path, train_matrix_manifest_path=train_manifest,
        support_artifact_path=support_path, output_matrix_path=matrix,
        output_manifest_path=manifest,
    )
    with np.load(matrix, allow_pickle=False) as archive:
        assert archive["features"].tolist() == [[7.0], [7.0]]
        assert archive["split"].dtype.kind == "U"
        assert archive["date"].dtype.kind == "U"
        assert archive["order_id"].dtype.kind == "U"
    assert result["train_feature_schema_hash"] == "schema-hash"
    assert result["evaluation_dates"] == ["20161019", "20161020"]


def test_final_verifier_rejects_naked_pass_strings() -> None:
    result = verify_final_gate_bundle({
        "required_gates": {name: "PASS" for name in FINAL_REQUIRED_GATES}
    })
    assert result["status"] == "FAIL"


def test_final_verifier_requires_release_bound_hard_specs(tmp_path: Path) -> None:
    gates = {}
    for name in FINAL_REQUIRED_GATES:
        report = {
            "schema_version": f"fixture_{name}.1", "status": "PASS",
            "protocol_id": "rolling_origin_fold_1_2_3", "model_id": None,
            "evaluation_dates": [],
        }
        if name == "spatial_adoption":
            report.update({"adopt": False, "decision_status": "RETAIN_V5_1_NEGATIVE_OR_INSUFFICIENT_TRANSFER"})
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        gates[name] = {
            "report_path": str(path), "report_sha256": sha256_file(path),
            "schema_version": report["schema_version"], "protocol_id": report["protocol_id"],
            "model_id": None, "evaluation_dates": [],
        }
    spatial = gates["spatial_adoption"]
    gates["temporal_adoption"] = {
        **spatial, "status": "NOT_APPLICABLE_BY_FROZEN_STOP_RULE",
    }
    result = verify_final_gate_bundle({"required_gates": gates})
    assert result["status"] == "FAIL"
    assert result["reason"] == "release_manifest_or_context_unbound"


def test_none_phase_a2_blocks_workloads(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "schema_version": "stage2_v5_2_config.3",
        "execution_authorization": "NONE_PHASE_A_2",
    }), encoding="utf-8")
    with pytest.raises(Stage2V52ContractError):
        main(["--config", str(config), "benchmark", "--output-csv", str(tmp_path / "x.csv"), "--output-json", str(tmp_path / "x.json")])


def test_cpu_only_benchmark_not_labeled_cuda() -> None:
    assert benchmark_kernel_devices(torch_kernel=False, cuda_available=True) == ("cpu",)
    assert benchmark_kernel_devices(torch_kernel=True, cuda_available=True) == ("cpu", "cuda")


def test_stage3_reader_rejects_evaluation_and_rts_diagnostics() -> None:
    with pytest.raises(Stage2V52ContractError):
        validate_stage3_fields(["order_id", "truth"])
    with pytest.raises(Stage2V52ContractError):
        validate_stage3_fields(["order_id", "pred_rts_raw"])
