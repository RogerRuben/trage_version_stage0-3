from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from stage2.v5_2.b1_freeze import verify_b1_evidence_bundle, verify_existing_tau_freeze
from stage2.v5_2.cli import (
    COMMAND_AUTHORIZATIONS, _freeze_tau, _require_execution, _require_protocol_execution,
)
from stage2.v5_2.contracts import Stage2V52ContractError


REPO = Path(__file__).resolve().parents[2]


def _config() -> dict[str, object]:
    return json.loads((REPO / "stage2/config/stage2_v5_2.json").read_text(encoding="utf-8"))


def test_b1_complete_blocks_training() -> None:
    config = _config()
    assert config["execution_authorization"] == "NONE_POST_B1"
    blocked = {
        "train-model", "train-tree-baseline", "evaluate-model", "build-tau-metrics",
        "tune-tau", "freeze-tau", "build-transfer-shards", "build-m0-feature-matrix",
    }
    for command in blocked:
        with pytest.raises(Stage2V52ContractError):
            _require_execution(config, COMMAND_AUTHORIZATIONS[command])


def test_transfer_tuning_cannot_reopen_under_a_later_authorization() -> None:
    args = argparse.Namespace(command="train-model", protocol="transfer_tuning")
    with pytest.raises(Stage2V52ContractError, match="cannot be reopened"):
        _require_protocol_execution({"execution_authorization": "PHASE_C"}, args)


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
