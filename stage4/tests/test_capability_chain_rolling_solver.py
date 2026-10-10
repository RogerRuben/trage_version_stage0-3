from copy import deepcopy
from itertools import product
import json

import pytest

from stage4.dispatch import capability_chain_rolling_solver as solver


def _task(job_id, release, deadline, service=0, source="FORECAST", profiles=("C",)):
    return dict(job_id=job_id, release_s=release, deadline_s=deadline, service_time_s=service,
                compatible_profiles=list(profiles), source_kind=source)


def _edge(origin, target, travel=0, distance=0, supported=True):
    return dict(origin_id=origin, target_job_id=target, supported=supported, travel_time_s=travel,
                empty_distance_m=distance, compatible_profiles=["C"])


def _action(action_id, kind, ready, location, job_id=None, critical=False, carry=False,
            resource="car", distance=0, now=0, pickup=None):
    action = dict(action_id=action_id, resource_id=resource, kind=kind, ready_s=ready, location_id=location,
                  job_id=job_id, empty_distance_m=distance, critical=critical, carry_over=carry)
    if kind == "SERVE":
        pickup = now if pickup is None else pickup
        action.update(commit_s=now, pickup_s=pickup, travel_time_s=pickup-now, origin_id="start")
    return action


def _scenario(tasks, edges, scenario_id="history", weight=1):
    return dict(scenario_id=scenario_id, weight=weight, tasks=tasks, connections=edges)


def _problem(actions, scenarios, resources=None, now=0):
    return dict(now_s=now, step_s=30,
                resources=resources or [dict(resource_id="car", profile_id="C", admission_end_s=300)],
                actions=actions, scenarios=scenarios)


def test_critical_protection_and_ordinary_defer_benefit_with_original_deadline():
    tasks = [_task("ordinary", 0, 150, 120, source="ACTUAL_PENDING"),
             _task("future_a", 30, 30, 30), _task("future_b", 60, 60, 30)]
    edges = [_edge("start", "future_a"), _edge("future_a", "future_b"), _edge("future_b", "ordinary")]
    actions = [_action("serve", "SERVE", 120, "ordinary", "ordinary", carry=True),
               _action("wait", "WAIT", 30, "start")]
    problem = _problem(actions, [_scenario(tasks, edges)])
    preserving = solver.solve_epoch(problem, "SERVICE_PRESERVING")
    deferred = solver.solve_epoch(problem, "CHAIN_DEFER")
    assert preserving["selected_actions"][0]["kind"] == "SERVE"
    assert preserving["current_served"] == preserving["expected_served"] == 1
    assert deferred["selected_actions"][0]["kind"] == "WAIT"
    assert deferred["current_served"] == 0 and deferred["expected_served"] == 3
    visits = deferred["recourse_paths"][0]["jobs"]
    assert [visit["job_id"] for visit in visits] == ["future_a", "future_b", "ordinary"]
    assert visits[-1]["commit_s"] == 90 and visits[-1]["deadline_s"] == 150
    critical = deepcopy(problem)
    critical["actions"][0]["critical"] = True
    protected = solver.solve_epoch(critical, "CHAIN_DEFER")
    assert protected["critical_now"] == protected["current_served"] == 1
    assert protected["selected_actions"][0]["kind"] == "SERVE"
    corrupt = deepcopy(deferred)
    corrupt["recourse_paths"][0]["jobs"][-1]["commit_s"] = 60
    with pytest.raises(ValueError, match="commit_s"):
        solver.validate_epoch_solution(problem, corrupt)
    assert solver.validate_epoch_solution(problem, deferred)["valid"]


def test_two_scenarios_share_one_first_action_and_forecast_cannot_be_first_serve():
    actions = [_action("layout_a", "LAYOUT", 0, "a"), _action("layout_b", "LAYOUT", 0, "b")]
    scenarios = [_scenario([_task("forecast_a", 30, 30)], [_edge("a", "forecast_a")], "history_a", 0.5),
                 _scenario([_task("forecast_b", 30, 30)], [_edge("b", "forecast_b")], "history_b", 0.5)]
    problem = _problem(actions, scenarios)
    result = solver.solve_epoch(problem, "CHAIN_DEFER")
    assert result["expected_served"] == 0.5
    assert result["selected_actions"][0]["action_id"] == "layout_a"
    assert {path["first_action_id"] for path in result["recourse_paths"]} == {"layout_a"}
    assert result["current_served"] == 0
    corrupt = deepcopy(result)
    corrupt["recourse_paths"][1]["first_action_id"] = "layout_b"
    with pytest.raises(ValueError, match="common first action"):
        solver.validate_epoch_solution(problem, corrupt)
    invalid = deepcopy(problem)
    invalid["actions"].append(_action("fake", "SERVE", 30, "forecast_a", "forecast_a", pickup=30))
    with pytest.raises(ValueError, match="ACTUAL_PENDING"):
        solver.solve_epoch(invalid, "SERVICE_PRESERVING")
    # Even two resources with the same currently feasible job may serve it once.
    pending = _task("pending", 0, 30, 30, source="ACTUAL_PENDING")
    duplicated = _problem(
        [_action("serve_1", "SERVE", 30, "pending", "pending", resource="car_1"),
         _action("wait_1", "WAIT", 30, "start", resource="car_1"),
         _action("serve_2", "SERVE", 30, "pending", "pending", resource="car_2"),
         _action("wait_2", "WAIT", 30, "start", resource="car_2")],
        [_scenario([pending], [])],
        resources=[dict(resource_id=car, profile_id="C", admission_end_s=300) for car in ("car_1", "car_2")])
    assert solver.solve_epoch(duplicated, "SERVICE_PRESERVING")["current_served"] == 1
    json.dumps(result, allow_nan=False)
    # Unequal action counts: mixed-radix codes 0..23 have exactly the same
    # order as all (rank_a, rank_m, rank_z) tuples. Input order is reversed to
    # verify the existing stable resource/action sorting remains authoritative.
    counts = {"car_z": 4, "car_a": 2, "car_m": 3}
    rank_actions = [_action(f"{car}_choice_{rank}", "WAIT", 30, "start", resource=car)
                    for car, count in counts.items() for rank in reversed(range(count))]
    ranked = _problem(rank_actions, [_scenario([], [])],
                      resources=[dict(resource_id=car, profile_id="C", admission_end_s=300) for car in counts])
    ranked_solution = solver.solve_epoch(ranked, "CHAIN_DEFER")
    encoding = ranked_solution["model"]["stable_first_actions_encoding"]
    assert encoding["resource_order"] == ["car_a", "car_m", "car_z"]
    assert encoding["radices"] == [2, 3, 4] and encoding["multipliers"] == [12, 4, 1]
    rank_tuples = list(product(*(range(radix) for radix in encoding["radices"])))
    codes = [sum(rank * multiplier for rank, multiplier in zip(ranks, encoding["multipliers"]))
             for ranks in rank_tuples]
    assert codes == list(range(24)) and encoding["maximum_code"] == 23 < 2**53
    assert [action["action_id"] for action in ranked_solution["selected_actions"]] == [
        "car_a_choice_0", "car_m_choice_0", "car_z_choice_0"]
    tie_stages = [stage for stage in ranked_solution["stages"] if stage["stage"].startswith("stable_first")]
    assert len(tie_stages) == 1 and tie_stages[0]["stage"] == "stable_first_actions:mixed_radix"
    assert tie_stages[0]["tie_encoding"]["tie_only"]


def test_relocate_only_moves_location_sparse_master_and_hard_limits(monkeypatch):
    problem = _problem(
        [_action("relocate", "RELOCATE", 30, "target", distance=100), _action("wait", "WAIT", 30, "start")],
        [_scenario([_task("forecast", 30, 30, 200)], [_edge("target", "forecast", distance=5)])],
        resources=[dict(resource_id="car", profile_id="C", admission_end_s=40)])
    original_milp = solver.milp

    def sparse_milp(*args, **kwargs):
        from scipy.sparse import issparse
        assert issparse(kwargs["constraints"].A)
        assert kwargs["options"]["mip_rel_gap"] == 0
        return original_milp(*args, **kwargs)

    monkeypatch.setattr(solver, "milp", sparse_milp)
    result = solver.solve_epoch(problem, "CHAIN_DEFER")
    assert result["selected_actions"][0]["kind"] == "RELOCATE"
    assert result["current_served"] == 0 and result["expected_served"] == 1
    assert result["actual_first_empty_distance_m"] == 100
    assert result["expected_empty_distance_m"] == 105
    assert result["recourse_paths"][0]["jobs"][0]["finish_s"] == 230 > 40
    assert result["kind"] == solver.KIND and result["model"]["sparse"]
    assert all(stage["status"] == 0 for stage in result["stages"])
    assert solver.validate_epoch_solution(problem, result)["valid"]
    with monkeypatch.context() as limited:
        limited.setattr(solver, "MAX_CHAINS_PER_RESOURCE_SCENARIO", 1)
        with pytest.raises(solver.RollingModelLimitError, match="not truncated"):
            solver.solve_epoch(problem, "CHAIN_DEFER")
    with monkeypatch.context() as limited:
        limited.setattr(solver, "MAX_MASTER_VARIABLES", 2)
        with pytest.raises(solver.RollingModelLimitError, match="variable limit"):
            solver.solve_epoch(problem, "CHAIN_DEFER")
    with monkeypatch.context() as limited:
        limited.setattr(solver, "MAX_MASTER_NONZEROS", 1)
        with pytest.raises(solver.RollingModelLimitError, match="nonzero limit"):
            solver.solve_epoch(problem, "CHAIN_DEFER")
    with monkeypatch.context() as limited:
        limited.setattr(solver, "DECISION_TIME_LIMIT_S", 0)
        with pytest.raises(solver.DecisionTimeout, match="budget expired"):
            solver.solve_epoch(problem, "CHAIN_DEFER")


def test_exact_compression_preserves_decision_masks_and_pareto_tradeoffs(monkeypatch):
    # A,B,C reaches C early at cost 10; B,A,C reaches it late at cost 1.
    # Both labels are necessary: only the early one can add D, while the late
    # one is the best three-job terminal column. A,C is earlier and cheaper
    # still, but has a DIFFERENT served mask and cannot dominate either label.
    tasks = [_task(job, 0, 300) for job in ("a", "b", "c")] + [_task("d", 60, 60)]
    edges = [_edge("start", "a"), _edge("start", "b"),
             _edge("a", "b", travel=60, distance=10), _edge("b", "a", travel=120, distance=1),
             _edge("a", "c"), _edge("b", "c"), _edge("c", "d")]
    problem = _problem([_action("layout", "LAYOUT", 0, "start"),
                        _action("wait", "WAIT", 30, "start")], [_scenario(tasks, edges)])
    compressed = deepcopy(problem)
    compressed["exact_chain_compression"] = True
    fields = ("critical_now", "current_served", "carry_over_current_served", "expected_served",
              "expected_empty_distance_m", "actual_first_empty_distance_m")
    for policy in solver.POLICIES:
        full = solver.solve_epoch(problem, policy)
        reduced = solver.solve_epoch(compressed, policy)
        assert {field: full[field] for field in fields} == {field: reduced[field] for field in fields}
        assert full["selected_actions"] == reduced["selected_actions"]
        assert reduced["expected_served"] == 4 and reduced["expected_empty_distance_m"] == 10
        assert reduced["model"]["compression_statistics"]["max_pareto_labels_per_state"] == 2
        assert reduced["model"]["complete_finite_domain"]
        assert not reduced["model"]["all_task_sequences_enumerated"]
        assert solver.validate_epoch_solution(compressed, reduced)["valid"]
    for candidate in (problem, compressed):
        cheaper = deepcopy(candidate)
        cheaper["scenarios"][0]["connections"][-1]["supported"] = False
        result = solver.solve_epoch(cheaper, "CHAIN_DEFER")
        assert result["expected_served"] == 3 and result["expected_empty_distance_m"] == 1
    dominated = deepcopy(compressed)
    dominated["scenarios"][0]["connections"][3].update(travel_time_s=60, empty_distance_m=20)
    dominated_result = solver.solve_epoch(dominated, "CHAIN_DEFER")
    assert dominated_result["model"]["compression_statistics"]["dominated_prefixes"] > 0
    # The expensive A,B,C branch was expanded first. Discovering the cheaper
    # and earlier B,A,C branch later must replace its terminal descendants.
    discovered_later = deepcopy(compressed)
    discovered_later["scenarios"][0]["connections"][3]["travel_time_s"] = 0
    replaced = solver.solve_epoch(discovered_later, "CHAIN_DEFER")
    assert replaced["expected_served"] == 4 and replaced["expected_empty_distance_m"] == 1
    assert replaced["model"]["compression_statistics"]["labels_removed_by_new_dominance"] > 0
    # >4,000 raw permutations, but exactly 128 equivalent final job-mask
    # columns. The retained-column limit must not apply to raw sequences.
    jobs = [f"job_{index}" for index in range(7)]
    dense_edges = [_edge("start", job) for job in jobs]
    dense_edges += [_edge(origin, target) for origin in jobs for target in jobs if origin != target]
    dense = _problem([_action("layout", "LAYOUT", 0, "start")],
                     [_scenario([_task(job, 0, 300) for job in jobs], dense_edges)])
    with pytest.raises(solver.RollingModelLimitError, match="chain limit"):
        solver.solve_epoch(dense, "CHAIN_DEFER")
    dense["exact_chain_compression"] = True
    exact = solver.solve_epoch(dense, "CHAIN_DEFER")
    assert exact["expected_served"] == 7
    assert exact["model"]["scenario_chain_variables"] == 128
    stats = exact["model"]["compression_statistics"]
    assert stats["emitted_columns"] > stats["retained_columns"] and stats["dominated_prefixes"] > 0
    with monkeypatch.context() as limited:
        limited.setattr(solver, "MAX_PREFIX_LABELS_TOTAL", 2)
        with pytest.raises(solver.RollingModelLimitError, match="prefix label limit"):
            solver.solve_epoch(dense, "CHAIN_DEFER")


def test_explicit_task_cap_up_to_64_preserves_default_20():
    problem = _problem([_action("wait", "WAIT", 30, "start")],
                       [_scenario([_task(f"forecast_{index}", 60, 60) for index in range(21)], [])])
    with pytest.raises(solver.RollingModelLimitError, match="maximum 20"):
        solver.solve_epoch(problem, "CHAIN_DEFER")
    problem["task_limit_per_scenario"] = 21
    result = solver.solve_epoch(problem, "CHAIN_DEFER")
    assert result["model"]["task_limit_per_scenario"] == 21
    assert not result["model"]["exact_chain_compression"]
    assert result["model"]["default_task_limit_per_scenario"] == 20
    problem["task_limit_per_scenario"] = 64
    problem["exact_chain_compression"] = True
    problem["scenarios"][0]["tasks"] = [_task(f"forecast_{index}", 60, 60) for index in range(64)]
    assert solver.solve_epoch(problem, "CHAIN_DEFER")["model"]["hard_task_limit_per_scenario"] == 64
    for value in (0, -1, True, 21.0, "21", 65):
        invalid = deepcopy(problem)
        invalid["task_limit_per_scenario"] = value
        with pytest.raises(ValueError, match="task_limit_per_scenario"):
            solver.solve_epoch(invalid, "CHAIN_DEFER")
    problem["scenarios"][0]["tasks"].append(_task("too_many", 60, 60))
    with pytest.raises(solver.RollingModelLimitError, match="maximum 64"):
        solver.solve_epoch(problem, "CHAIN_DEFER")
