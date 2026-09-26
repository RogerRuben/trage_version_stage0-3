import json
from pathlib import Path

from stage4.dispatch.acceptance import passenger_acceptance, stable_acceptance_uniform
from stage4.dispatch.exposure import CumulativeExposureState
from stage4.dispatch.final_experiment_aggregation import _resolved_rows
from stage4.dispatch.final_experiment_runner import (
    OUTPUT_FILES,
    _atomic_json,
    batch_partition,
    load_execution_config,
    load_registry,
    phase_rows,
    required_outputs_exist,
    resume_action,
    scenario_config_sha256,
    scenario_dir,
    scientific_configuration,
    unique_rows,
)
from stage4.dispatch.solver import AssignmentArc, solve_lexicographic

ROOT = Path(__file__).resolve().parents[2]


def _config(tmp_path: Path):
    frozen = load_execution_config(ROOT)
    return {**frozen, "output_root": "out"}


def _row():
    return load_registry(ROOT)[0]


def test_reuse_row_is_never_selected_for_duplicate_execution():
    rows = load_registry(ROOT)
    assert len(rows) == 42
    assert len(unique_rows(rows)) == 41
    reuse = next(row for row in rows if row["reuse_source_scenario_id"])
    assert reuse not in phase_rows(rows, "ALL")


def test_completed_scenario_is_skipped_on_resume(tmp_path):
    row = _row()
    config = _config(tmp_path)
    directory = scenario_dir(tmp_path, row["scenario_id"], config)
    directory.mkdir(parents=True)
    for name in OUTPUT_FILES:
        (directory / name).write_text("{}", encoding="utf-8")
    _atomic_json(
        {
            "status": "COMPLETED",
            "execution_commit": "abc",
            "registry_commit": config["registry_commit"],
            "scenario_config_sha256": scenario_config_sha256(row),
        },
        directory / "run_status.json",
    )
    assert required_outputs_exist(directory)
    assert resume_action(tmp_path, row, "abc", config=config) == "SKIP"


def test_failed_scenario_isolated_while_other_scenario_remains_runnable(tmp_path):
    rows = load_registry(ROOT)[:2]
    config = _config(tmp_path)
    failed_dir = scenario_dir(tmp_path, rows[0]["scenario_id"], config)
    failed_dir.mkdir(parents=True)
    _atomic_json({"status": "FAILED"}, failed_dir / "run_status.json")
    partition = batch_partition(
        tmp_path,
        rows,
        "abc",
        retry_failed=False,
        config=config,
    )
    assert [row["scenario_id"] for row in partition["FAILED"]] == [
        rows[0]["scenario_id"]
    ]
    assert [row["scenario_id"] for row in partition["RUN"]] == [rows[1]["scenario_id"]]


def test_scenario_config_hash_is_stable_and_ignores_volatile_fields():
    row = _row()
    reordered = dict(reversed(list(row.items())))
    reordered["volatile_timestamp"] = "tomorrow"
    assert scientific_configuration(row) == scientific_configuration(reordered)
    assert scenario_config_sha256(row) == scenario_config_sha256(reordered)


def test_scenario_output_directories_are_isolated(tmp_path):
    config = _config(tmp_path)
    first = scenario_dir(tmp_path, "A", config)
    second = scenario_dir(tmp_path, "B", config)
    assert first != second
    assert first.parent == second.parent
    assert first.name == "A" and second.name == "B"


def test_aggregate_reader_resolves_reuse_without_copying_raw_outputs():
    unique = [{"scenario_id": "SOURCE", "request_count": 30000}]
    registry = [
        {
            "scenario_id": "SOURCE",
            "experiment_block": "MAIN_STRUCTURAL",
            "scientific_role": "source",
            "reuse_source_scenario_id": "",
        },
        {
            "scenario_id": "REUSED",
            "experiment_block": "ODD_POLICY",
            "scientific_role": "reused",
            "reuse_source_scenario_id": "SOURCE",
        },
    ]
    resolved = _resolved_rows(unique, registry)
    assert [row["scenario_id"] for row in resolved] == ["SOURCE", "REUSED"]
    assert resolved[1]["request_count"] == 30000
    assert resolved[1]["reuse_source_scenario_id"] == "SOURCE"


def test_disabled_gamma_and_cost_add_no_optional_solver_layers():
    arcs = [AssignmentArc(1, 10, 5.0, False, False)]
    result = solve_lexicographic(
        arcs,
        exposure_state=CumulativeExposureState(),
        gammas={"static": None, "dynamic": None, "speed": None},
        cost_level_enabled=False,
    )
    assert result.enabled_gamma_constraint_count == 0
    assert result.cost_level_solved is False


def test_acceptance_crn_matches_frozen_s5b_design():
    design = json.loads(
        (ROOT / "stage4/config/experimental_design/acceptance_design.json").read_text(
            encoding="utf-8"
        )
    )
    seed = design["acceptance_seed"]
    assert seed == 20260827
    order = "frozen-order"
    draw = stable_acceptance_uniform(order, seed)
    for probability in design["probability_levels"]:
        decision = passenger_acceptance(order, probability, seed)
        assert decision.passenger_accepts_av == (draw <= probability)
