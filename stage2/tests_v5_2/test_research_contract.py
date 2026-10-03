from __future__ import annotations

from pathlib import Path

from stage2.v5_2.contracts import (
    CORE_TRANSFER_TARGETS,
    DIAGNOSTIC_TARGETS,
    FORBIDDEN_STAGE3_CONCEPTS,
    PREDICTED_MICRO_TARGETS,
    RESEARCH_CONTRACT,
    STAGE3_DEPLOYABLE_MICRO_TARGETS,
    validate_research_contract,
)


def test_micro_target_roles_are_frozen_and_travel_time_is_not_the_only_target() -> None:
    validate_research_contract(RESEARCH_CONTRACT)
    assert tuple(RESEARCH_CONTRACT["predicted_micro_targets"]) == PREDICTED_MICRO_TARGETS
    assert tuple(RESEARCH_CONTRACT["core_transfer_targets"]) == CORE_TRANSFER_TARGETS
    assert tuple(RESEARCH_CONTRACT["stage3_deployable_micro_targets"]) == STAGE3_DEPLOYABLE_MICRO_TARGETS
    assert tuple(RESEARCH_CONTRACT["diagnostic_targets"]) == DIAGNOSTIC_TARGETS
    assert RESEARCH_CONTRACT["travel_time_is_only_prediction_target"] is False


def test_stage3_contract_has_no_path_planning_decision_variable() -> None:
    path = Path("stage2/docs/v5_2/stage2_v5_2_to_stage3_contract.md")
    text = path.read_text(encoding="utf-8").lower()
    for concept in FORBIDDEN_STAGE3_CONCEPTS:
        assert f"`{concept}`" not in text
    assert "no route decision variable" in text
