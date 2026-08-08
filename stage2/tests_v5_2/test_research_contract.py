from __future__ import annotations

from pathlib import Path

from stage2.v5_2.contracts import (
    FORBIDDEN_STAGE3_CONCEPTS,
    MICRO_TARGETS,
    RESEARCH_CONTRACT,
    validate_research_contract,
)


def test_formal_micro_targets_are_frozen_and_travel_time_is_not_the_only_target() -> None:
    validate_research_contract(RESEARCH_CONTRACT)
    assert tuple(RESEARCH_CONTRACT["formal_micro_targets"]) == MICRO_TARGETS
    assert RESEARCH_CONTRACT["travel_time_is_only_formal_target"] is False


def test_stage3_contract_has_no_path_planning_decision_variable() -> None:
    path = Path("stage2/docs/v5_2/stage2_v5_2_to_stage3_contract.md")
    text = path.read_text(encoding="utf-8").lower()
    for concept in FORBIDDEN_STAGE3_CONCEPTS:
        assert f"`{concept}`" not in text
    assert "no route decision variable" in text
