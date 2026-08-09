from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stage2.v5_2.contracts import Stage2V52ContractError
from stage2.v5_2.micro_products import DIMENSIONS, aggregate_original_route_micro_conditions
from stage2.v5_2.support_transfer import fit_train_support
from stage2.v5_2.verification import (
    FINAL_REQUIRED_GATES, sha256_file, verify_artifact_payload, verify_final_gate_bundle,
)


def test_artifact_verifier_uses_type_specific_evaluation_fields() -> None:
    support = fit_train_support(["a", "a", "b"], fit_dates=["20161009"]).to_payload()
    verify_artifact_payload(support, artifact_type="support")
    verify_artifact_payload(
        {"fit_scope": "train_only", "evaluation_rows_used": 0}, artifact_type="static_structure"
    )
    verify_artifact_payload(
        {"fit_split": "train", "evaluation_rows_used": 0}, artifact_type="micro_cdf"
    )
    with pytest.raises(Stage2V52ContractError):
        verify_artifact_payload(
            {"fit_scope": "train_only", "evaluation_rows_used": 0}, artifact_type="support"
        )


def test_final_verifier_rejects_missing_or_invented_gates() -> None:
    exact = {name: "PASS" for name in FINAL_REQUIRED_GATES}
    assert verify_final_gate_bundle({"required_gates": exact})["status"] == "FAIL"
    incomplete = dict(exact); incomplete.pop(FINAL_REQUIRED_GATES[0])
    assert verify_final_gate_bundle({"required_gates": incomplete})["status"] == "FAIL"
    invented = {**exact, "manual_override": "PASS"}
    assert verify_final_gate_bundle({"required_gates": invented})["status"] == "FAIL"


def test_service_time_complete_is_near_one_coverage_only() -> None:
    frame = pd.DataFrame({
        "split": ["evaluation", "evaluation"], "date": ["20161025"] * 2,
        "order_id": ["o"] * 2, "route_sequence": [0, 1],
        "estimated_travel_time_p50_s": [10.0, float("nan")],
        "allocated_distance_m": [999.0, 1.0], "edge_train_support": [2, 2],
        "support_group": ["low", "low"], "protocol_id": ["development"] * 2,
        "model_id": ["M4"] * 2, "prediction_source": ["fixture"] * 2,
        "route_track": ["historical_original_service_route"] * 2,
        "route_source": ["frozen_stage1_route_parts"] * 2,
        "route_product_version": ["stage1_v3_route_sequence_context.1"] * 2,
    })
    for column in DIMENSIONS.values():
        frame[column] = 0.5
    cdf = {
        "fit_split": "train", "evaluation_rows_used": 0, "protocol_id": "development",
        "model_id": "M4", "prediction_source": "fixture",
        "thresholds": {name: 0.8 for name in DIMENSIONS},
    }
    result = aggregate_original_route_micro_conditions(
        frame, cdf, minimum_coverage=0.8, service_time_complete_threshold=0.999
    ).iloc[0]
    assert bool(result["service_time_complete_flag"])
    assert result["service_time_status"] == "complete"


def test_config_schema_and_phase_execution_is_gate_bound() -> None:
    config = json.loads(open("stage2/config/stage2_v5_2.json", encoding="utf-8").read())
    assert config["schema_version"] == "stage2_v5_2_config.3"
    authorization = config["execution_authorization"]
    assert authorization in {"NONE_PHASE_A_2", "PHASE_B0", "PHASE_B1", "NONE_POST_B1"}
    if authorization == "PHASE_B1":
        gate = config["phase_b0_gate"]
        report_path = Path(gate["report_path"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert sha256_file(report_path) == gate["report_sha256"]
        assert report["schema_version"] == "stage2_v5_2_phase_b0_smoke.1"
        assert report["status"] == "PASS"
        assert report["authorizes_phase_b1"] is True
    if authorization == "NONE_POST_B1":
        assert config["phase"] == "B1_COMPLETE_FROZEN"
        assert config["phase_b1_complete"] is True
        assert config["phase_c_authorized"] is False
    assert config["performance"]["warmup_runs"] == 2
    assert config["performance"]["repeat_runs"] == 3
    assert config["training"]["loss_weights"]["rts"] == 0.0
