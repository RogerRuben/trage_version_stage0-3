from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from stage2.v5_2.b1_freeze import (
    classify_b1_scientific_conclusion, sha256_git_file,
    verify_b1_evidence_bundle, verify_existing_tau_freeze,
)
from stage2.v5_2.cli import (
    COMMAND_AUTHORIZATIONS, _freeze_tau, _require_execution, _require_protocol_execution,
    _verify_phase_c_authorization,
)
from stage2.v5_2.contracts import Stage2V52ContractError
from stage2.v5_2.feature_binding import sha256_path


REPO = Path(__file__).resolve().parents[2]


def _config() -> dict[str, object]:
    return json.loads((REPO / "stage2/config/stage2_v5_2.json").read_text(encoding="utf-8"))


def test_phase_c_freeze_retains_reviewed_authorization_provenance_but_blocks_execution() -> None:
    config = _config()
    assert config["execution_authorization"] == "NONE_POST_C"
    assert config["phase"] == "PHASE_C_COMPLETE_FROZEN"
    assert config["phase_c_complete"] is True
    assert config["phase_c_direction"] == "FAIL"
    assert config["phase_d_authorized"] is False
    historical = dict(config)
    historical.update({
        "phase": "PHASE_C", "execution_authorization": "PHASE_C",
        "phase_c_authorized": True, "current_status": "PHASE_C_AUTHORIZED",
    })
    assert _verify_phase_c_authorization(historical, repo_root=REPO)["status"] == "PASS"
    with pytest.raises(Stage2V52ContractError):
        _require_execution(config, COMMAND_AUTHORIZATIONS["train-model"])
    blocked = {
        "build-tau-metrics", "tune-tau", "freeze-tau", "build-release-manifest",
    }
    for command in blocked:
        with pytest.raises(Stage2V52ContractError):
            _require_execution(config, COMMAND_AUTHORIZATIONS[command])
    _require_execution(config, COMMAND_AUTHORIZATIONS["verify-b1-evidence"])
    _require_execution(config, COMMAND_AUTHORIZATIONS["verify-existing-tau-freeze"])


def test_post_phase_c_freeze_blocks_all_development_execution_commands() -> None:
    config = _config()
    for command in (
        "fit-support", "fit-static-artifact", "fit-train-cdf", "build-transfer-shards",
        "build-m0-feature-matrix", "transform-m0-feature-matrix", "evaluate-m0",
        "train-tree-baseline", "train-model", "evaluate-model", "decide-spatial-adoption",
    ):
        with pytest.raises(Stage2V52ContractError, match="NONE_POST_C"):
            _require_execution(config, COMMAND_AUTHORIZATIONS[command])


def test_post_phase_c_status_and_execution_config_provenance_are_synchronized() -> None:
    status_path = REPO / "stage2/docs/v5_2/stage2_v5_2_status_manifest.json"
    evidence_path = REPO / "stage2/docs/v5_2/stage2_v5_2_phase_c_evidence_bundle.json"
    config_path = REPO / "stage2/config/stage2_v5_2.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert status["status"] == status["current_status"] == "PHASE_C_FAIL_FROZEN"
    assert status["phase"] == "PHASE_C_COMPLETE_FROZEN"
    assert status["phase_c_authorized"] is False
    assert status["phase_d_authorized"] is False
    assert status["config"]["sha256"] == sha256_path(config_path)
    assert evidence["execution_base_commit"] == evidence["phase_c_authorization_commit"]
    execution_config = evidence["frozen_bindings"]["phase_c_execution_config_git"]
    assert execution_config["git_file_sha256"] == sha256_git_file(
        REPO, execution_config["commit"], execution_config["path"],
    )
    assert evidence["frozen_bindings"]["post_c_frozen_config"]["sha256"] == sha256_path(config_path)


def test_phase_c_binds_b1_execution_config_hash() -> None:
    binding = _config()["phase_c_authorization"]
    assert sha256_git_file(
        REPO, binding["b1_execution_commit"], binding["b1_execution_config_path"]
    ) == binding["b1_execution_config_sha256"]


def test_transfer_tuning_cannot_reopen_under_a_later_authorization() -> None:
    args = argparse.Namespace(command="train-model", protocol="transfer_tuning")
    with pytest.raises(Stage2V52ContractError, match="cannot be reopened"):
        _require_protocol_execution({"execution_authorization": "PHASE_C"}, args)


def test_phase_c_blocks_rolling_and_m5() -> None:
    with pytest.raises(Stage2V52ContractError, match="only the frozen development"):
        _require_protocol_execution(
            {"execution_authorization": "PHASE_C"},
            argparse.Namespace(command="train-model", protocol="fold_1", model="M4"),
        )
    with pytest.raises(Stage2V52ContractError, match="forbids M5"):
        _require_protocol_execution(
            {"execution_authorization": "PHASE_C"},
            argparse.Namespace(command="train-model", protocol="development", model="M5"),
        )


def test_existing_tau_freeze_is_write_once(tmp_path: Path) -> None:
    output = tmp_path / "stage2_v5_2_tau_freeze.json"
    output.write_text("{}\n", encoding="utf-8")
    args = argparse.Namespace(
        output=str(output), selection="missing", metrics="missing", support_artifact="missing"
    )
    with pytest.raises(Stage2V52ContractError, match="write-once"):
        _freeze_tau(args)


def test_existing_tau_freeze_hash_verifies() -> None:
    config = _config()
    result = verify_existing_tau_freeze(
        config["tau_freeze"]["artifact_path"], repo_root=REPO,
        expected_file_sha256=config["tau_freeze"]["expected_file_sha256"],
    )
    assert result["status"] == "PASS"
    assert result["write_once"] is True


def test_b1_evidence_bundle_hashes_resolve() -> None:
    path = REPO / "stage2/docs/v5_2/stage2_v5_2_phase_b1_evidence_bundle.json"
    bundle = json.loads(path.read_text(encoding="utf-8"))
    result = verify_b1_evidence_bundle(bundle, repo_root=REPO)
    assert result["status"] == "PASS"
    assert result["resolved_artifact_count"] >= 35
    assert result["phase_c_authorized"] is False


def test_phase_b1_verification_is_not_final_ready() -> None:
    payload = json.loads(
        (REPO / "stage2/docs/v5_2/stage2_v5_2_phase_b1_verification.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "PASS"
    assert payload["scope"] == "PHASE_B1_ONLY"
    assert payload["stage2_final_ready"] is False
    assert payload["phase_c_authorized"] is False


def test_b1_low_unseen_and_no_history_report_is_frozen() -> None:
    payload = json.loads(
        (REPO / "stage2/docs/v5_2/stage2_v5_2_phase_b1_transfer_tuning_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["schema_version"] == "stage2_v5_2_phase_b1_transfer_tuning_report.2"
    assert payload["scientific_conclusion"]["classification"] == "CASE_C"
    assert payload["phase_c_authorized"] is False
    for group in ("low", "unseen"):
        counts = payload["target_eligible_unique_traversal_counts"][group]
        assert all(value > 0 for value in counts.values())
    temporal = payload["no_history_temporal_fallback"]
    assert temporal["audited_token_count"] == 9954428
    assert temporal["no_history_temporal_fallback_token_count"] == 384
    assert temporal["unique_physical_no_history_row_count"] == 278
    assert temporal["not_a_real_observation"] is True
    automatic = classify_b1_scientific_conclusion(payload["metrics"])
    assert automatic["classification"] == "CASE_C"
    assert automatic["rule"]["mean_relative_improvement_threshold"] == 0.02
