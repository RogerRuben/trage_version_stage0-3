from __future__ import annotations

from pathlib import Path

from stage4.dispatch.efficient_repositioning_runner import (
    RUNS,
    execution_config,
    prepare_registry,
    protocol_config,
)


ROOT = Path(__file__).resolve().parents[2]


def test_efficient_protocol_freezes_five_scalar_runs_only() -> None:
    config = protocol_config(ROOT)
    assert config["routing_mode"] == "SCALAR_ROUTE"
    assert config["max_required_full_day_runs"] == len(RUNS) == 5
    assert not config["q25_authorized"]
    assert not config["all_av_authorized"]
    assert not config["gamma_frontier_authorized"]


def test_control_and_treatment_share_routing_and_train_reference() -> None:
    control = execution_config(ROOT, False)
    treatment = execution_config(ROOT, True)
    assert control["routing_mode"] == treatment["routing_mode"] == "SCALAR_ROUTE"
    assert control["max_parallel_scenarios"] == treatment["max_parallel_scenarios"] == 1
    assert control["repositioning_reference_sha256"] == treatment["repositioning_reference_sha256"]
    assert not control["repositioning_enabled"]
    assert treatment["repositioning_enabled"]


def test_registry_has_exactly_the_five_efficient_runs() -> None:
    registry = prepare_registry(ROOT)
    actual = set(registry.loc[registry["workstream"].eq("EFFICIENT_REPOSITIONING_R0.5C"), "run_id"])
    assert actual == set(RUNS)
