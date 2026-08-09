from __future__ import annotations

import pytest

from stage2.v5_2.contracts import Stage2V52ContractError
from stage2.v5_2.phase_c_report import _validate_m4_tau_binding, classify_phase_c_direction


def _decision(
    *, adopt: bool, wins: int, mean: float,
    overall_stable: bool = True, unseen_stable: bool = True,
) -> dict[str, object]:
    return {
        "adopt": adopt,
        "low_support_target_wins": wins,
        "low_support_mean_relative_improvement": mean,
        "overall_no_target_degrades_over_2pct": overall_stable,
        "unseen_not_worse_than_structure_only": unseen_stable,
    }


def test_phase_c_pass_requires_frozen_adoption_gate() -> None:
    assert classify_phase_c_direction(
        _decision(adopt=True, wins=3, mean=0.021), pace_guard_pass=True,
    ) == "PASS"


def test_phase_c_weak_is_metric_derived_not_hardcoded() -> None:
    assert classify_phase_c_direction(
        _decision(adopt=False, wins=3, mean=0.006), pace_guard_pass=True,
    ) == "WEAK"


def test_phase_c_overall_or_unseen_failure_cannot_be_weak() -> None:
    assert classify_phase_c_direction(
        _decision(adopt=False, wins=3, mean=0.006, overall_stable=False),
        pace_guard_pass=True,
    ) == "FAIL"
    assert classify_phase_c_direction(
        _decision(adopt=False, wins=3, mean=0.006, unseen_stable=False),
        pace_guard_pass=True,
    ) == "FAIL"


def test_phase_c_fail_for_negative_or_unstable_direction() -> None:
    assert classify_phase_c_direction(
        _decision(adopt=False, wins=2, mean=-0.01), pace_guard_pass=True,
    ) == "FAIL"
    assert classify_phase_c_direction(
        _decision(adopt=False, wins=3, mean=0.01), pace_guard_pass=False,
    ) == "FAIL"


def _tau_payloads() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    freeze = {
        "schema_version": "stage2_v5_2_tau_freeze.1", "status": "PASS",
        "selected_candidate": "p25", "selected_tau": 3.0,
        "transfer_tuning_support_sha256": "source-support",
    }
    provenance = {
        "kind": "frozen_transfer_tuning_selection", "support_tau_candidate": "p25",
        "support_tau_value": 3.0, "tau_freeze_artifact_sha256": "freeze-file",
        "support_tau_source_support_sha256": "source-support",
        "current_protocol_support_artifact_sha256": "development-support",
    }
    training = {
        "constructor": {
            "spatial_mode": "support_aware", "support_tau": 3.0,
            "support_tau_provenance": provenance,
        }
    }
    evaluation = {
        "support_tau": 3.0, "support_tau_candidate": "p25",
        "support_tau_source_support_sha256": "source-support",
        "support_artifact_sha256": "development-support",
    }
    return training, evaluation, freeze


def test_m4_tau_consumption_is_relationship_verified() -> None:
    training, evaluation, freeze = _tau_payloads()
    result = _validate_m4_tau_binding(
        training=training, evaluation=evaluation, tau_freeze=freeze,
        tau_freeze_sha256="freeze-file", current_support_sha256="development-support",
    )
    assert result["status"] == "PASS"
    assert result["selected_tau"] == 3.0


def test_m4_tau_relationship_rejects_unbound_training() -> None:
    training, evaluation, freeze = _tau_payloads()
    training["constructor"]["support_tau_provenance"]["tau_freeze_artifact_sha256"] = "wrong"
    with pytest.raises(Stage2V52ContractError, match="does not consume"):
        _validate_m4_tau_binding(
            training=training, evaluation=evaluation, tau_freeze=freeze,
            tau_freeze_sha256="freeze-file", current_support_sha256="development-support",
        )
