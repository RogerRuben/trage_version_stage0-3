from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from stage2.v5_2.contracts import Stage2V52ContractError
from stage2.v5_2.final_freeze import (
    ALL_AUTHORIZATIONS_FALSE,
    DEPLOYABLE_OUTPUTS,
    EVIDENCE_PATHS,
    EXPERIMENT_DECISIONS,
    FINAL_CHECKPOINT,
    FINAL_STATUS,
    _canonical_hash,
    verify_final_release_manifest,
)


REPO = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO / "stage2/docs/v5_2/stage2_v5_2_final_release_manifest.json"
CONTRACT_PATH = REPO / "stage2/docs/v5_2/stage2_v5_2_to_stage3_contract.md"


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_stage2_final_release_manifest_resolves() -> None:
    result = verify_final_release_manifest(_manifest(), repo_root=REPO)
    assert result == {
        "schema_version": "stage2_v5_2_final_release_verification.1",
        "status": "PASS",
        "stage2_status": FINAL_STATUS,
        "final_model": "M3",
        "resolved_descriptor_count": 20,
        "scientific_evidence_count": 4,
        "stage3_authorized": False,
        "next_phase_authorized": False,
    }


def test_candidate_route_stage2_uses_frozen_M3() -> None:
    predictor = _manifest()["final_predictor"]
    assert predictor["model_id"] == "M3"
    assert predictor["model_role"] == "structured_representation"
    assert predictor["checkpoint"]["path"] == FINAL_CHECKPOINT
    assert predictor["inference_policy"]["candidate_route_adapter_must_use_frozen_m3"]


def test_no_stage2_retraining() -> None:
    manifest = _manifest()
    assert manifest["status"] == FINAL_STATUS
    assert manifest["authorizations"] == ALL_AUTHORIZATIONS_FALSE
    assert not any(manifest["authorizations"].values())
    policy = manifest["final_predictor"]["inference_policy"]
    assert policy["retraining_allowed"] is False
    assert policy["checkpoint_reselection_allowed"] is False
    assert policy["tau_reselection_allowed"] is False
    assert policy["feature_contract_changes_allowed"] is False


def test_stage2_target_roles_are_frozen() -> None:
    target_roles = _manifest()["target_roles"]
    assert tuple(target_roles["deployable_outputs"]) == DEPLOYABLE_OUTPUTS
    assert target_roles["diagnostic_only"] == ["rts"]
    assert target_roles["rts_stage3_deployable"] is False


def test_stage2_scientific_evidence_is_complete_and_bound() -> None:
    manifest = _manifest()
    scientific = manifest["scientific_evidence"]
    assert set(scientific) == set(EVIDENCE_PATHS)
    assert all(item["verification"]["status"] == "PASS" for item in scientific.values())
    assert manifest["scientific_reports"]["phase_b1"]["classification"] == "CASE_C"
    assert manifest["scientific_reports"]["phase_c"]["direction"] == "FAIL"
    assert manifest["scientific_conclusion"]["sparsity_diagnostic"] == "DIAG-B"
    assert manifest["scientific_conclusion"]["upstream_sampling_audit"] == "UP-D"


def test_experiment_expansion_is_rejected_or_cancelled() -> None:
    decisions = _manifest()["experiment_decisions"]
    assert decisions == EXPERIMENT_DECISIONS
    assert decisions["M4"]["status"] == "REJECTED_ABLATION"
    assert decisions["M5"]["status"] == decisions["M6"]["status"] == "CANCELLED_NOT_RUN"
    assert decisions["Phase_D"]["status"] == decisions["Transfer_v2"]["status"] == "CANCELLED"


def test_stage4_has_no_route_decision() -> None:
    contract = CONTRACT_PATH.read_text(encoding="utf-8")
    assert "Stage 4 has no route decision variable" in contract
    assert "Stage 3 authorization: `NO`" in contract
    assert "S1–S8, Stage 3 execution, and Stage 4" in contract


def test_stage3_is_not_authorized_by_final_closure() -> None:
    manifest = _manifest()
    assert manifest["stage3_authorized"] is False
    assert manifest["next_phase_authorized"] is False
    assert all(manifest["authorizations"][f"stage3_s{number}"] is False for number in range(1, 9))


def test_final_release_descriptor_tamper_fails_closed() -> None:
    tampered = deepcopy(_manifest())
    tampered["final_predictor"]["checkpoint"]["sha256"] = "0" * 64
    tampered["artifact_sha256"] = _canonical_hash(tampered)
    with pytest.raises(Stage2V52ContractError, match="descriptor does not resolve"):
        verify_final_release_manifest(tampered, repo_root=REPO)


def test_final_release_policy_tamper_fails_closed() -> None:
    tampered = deepcopy(_manifest())
    tampered["authorizations"]["stage3_s1"] = True
    tampered["artifact_sha256"] = _canonical_hash(tampered)
    with pytest.raises(Stage2V52ContractError, match="execution phase is authorized"):
        verify_final_release_manifest(tampered, repo_root=REPO)
