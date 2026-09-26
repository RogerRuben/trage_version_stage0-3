from __future__ import annotations

from pathlib import Path

from stage4.dispatch.deterministic_repositioning_runner import (
    BASES,
    CONTROL_IDS,
    Q50_REPEAT_IDS,
    TREATMENT_IDS,
    _execution_config,
    _protocol_config,
    prepare_registry,
)


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_mode_is_arc_level_single_source_matrix() -> None:
    config = _protocol_config(ROOT)
    assert config["routing_mode"] == "SINGLE_SOURCE_MATRIX"
    assert config["assignment_matrix_representation"] == (
        "ARC_LEVEL_1X1_MATRIX_WITH_SPARSE_CANDIDATES"
    )
    assert config["gpu_usage"] == "NONE_CPU_ONLY"
    assert config["gamma_frontier_authorized"] is False


def test_control_and_treatment_use_identical_routing_mode() -> None:
    control = _execution_config(ROOT, enabled=False)
    treatment = _execution_config(ROOT, enabled=True)
    assert control["routing_mode"] == treatment["routing_mode"] == "SINGLE_SOURCE_MATRIX"
    assert control["max_parallel_scenarios"] == treatment["max_parallel_scenarios"] == 1
    assert control["repositioning_enabled"] is False
    assert treatment["repositioning_enabled"] is True


def test_registry_contains_only_preregistered_deterministic_scenarios() -> None:
    registry = prepare_registry(ROOT)
    expected = set(Q50_REPEAT_IDS) | set(CONTROL_IDS.values()) | set(TREATMENT_IDS.values())
    actual = set(
        registry.loc[
            registry["workstream"].eq("ROUTING_DETERMINISM_REPOSITIONING"), "run_id"
        ]
    )
    assert actual == expected
    assert set(CONTROL_IDS) == set(TREATMENT_IDS) == set(BASES)
