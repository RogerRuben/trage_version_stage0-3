from copy import deepcopy
import json

import pytest

from stage4.dispatch import capability_chain_rolling_solver as solver
from stage4.dispatch.generic_defer import PROXY_PROVENANCE, project_post_service_geometry


def _task(job_id, release=0, deadline=300, service=0, source="FORECAST", profiles=("C",)):
    return dict(job_id=job_id, release_s=release, deadline_s=deadline, service_time_s=service,
                source_kind=source, compatible_profiles=list(profiles))


def _edge(origin, target, travel=0, distance=0, supported=True, profiles=("C",)):
    return dict(origin_id=origin, target_job_id=target, travel_time_s=travel,
                empty_distance_m=distance, supported=supported, compatible_profiles=list(profiles))


def _action(action_id, kind, ready, location, job=None, critical=False, distance=0):
    action = dict(action_id=action_id, resource_id="car", kind=kind, ready_s=ready, location_id=location,
                  job_id=job, empty_distance_m=distance, critical=critical, carry_over=False)
    if kind == "SERVE":
        action.update(commit_s=0, pickup_s=0, travel_time_s=0, origin_id="start")
    return action


def _problem(tasks, edges, actions):
    return dict(now_s=0, step_s=30,
                resources=[dict(resource_id="car", profile_id="C", admission_end_s=300)],
                actions=actions,
                scenarios=[dict(scenario_id="history", weight=1, tasks=tasks, connections=edges)])


def _reference(travel=0, distance=0):
    return dict(travel_time_s=travel, empty_distance_m=distance, history_dates=["20161010", "20161017"],
                kind="EARLIER_HISTORY_TYPICAL_SERVICEABLE_PICKUP")


def test_projection_deep_copy_and_only_post_service_layer_changes():
    tasks = [_task("paid", service=120, source="ACTUAL_PENDING"),
             _task("forecast", release=31, deadline=150, profiles=("HV",))]
    known_edges = [_edge("start", "forecast", travel=20, distance=80),
                   _edge("busy_end", "forecast", travel=45, distance=250, profiles=("HV",)),
                   _edge("relocate_end", "paid", travel=30, distance=100)]
    problem = _problem(tasks, known_edges + [_edge("paid", "forecast", travel=None, distance=None,
                                                  supported=False)],
                       [_action("serve", "SERVE", 120, "paid", "paid"),
                        _action("wait", "WAIT", 30, "start"),
                        _action("busy", "BUSY", 45, "busy_end"),
                        _action("relocate", "RELOCATE", 60, "relocate_end", distance=200)])
    problem.update(task_limit_per_scenario=64, exact_chain_compression=True)
    problem["first_actions"] = {"serve": deepcopy(problem["actions"][0])}
    original = deepcopy(problem)
    reference = _reference(35, 175)
    projected = project_post_service_geometry(problem, reference)
    assert problem == original
    assert projected["actions"] == problem["actions"] and projected["first_actions"] == problem["first_actions"]
    assert projected["resources"] == problem["resources"]
    assert projected["scenarios"][0]["tasks"] == tasks
    assert projected["task_limit_per_scenario"] == 64 and projected["exact_chain_compression"]
    connections = projected["scenarios"][0]["connections"]
    assert connections[:len(known_edges)] == known_edges
    proxies = {edge["origin_id"]: edge for edge in connections[len(known_edges):]}
    assert set(proxies) == {"paid", "forecast"}
    assert proxies["paid"]["compatible_profiles"] == ["HV"]
    assert proxies["forecast"]["compatible_profiles"] == ["C"]
    assert all(edge["supported"] and edge["travel_time_s"] == 35 and edge["empty_distance_m"] == 175
               and edge["provenance"] == PROXY_PROVENANCE for edge in proxies.values())
    info = projected["valuation_model"]
    assert info["name"] == "TIME_TYPE_DEFER" and info["proxy_connections_are_not_physical_certificates"]
    assert info["per_scenario_projection"][0]["replaced_post_service_connections"] == 1
    assert info["per_scenario_projection"][0]["added_post_service_connections"] == 1
    json.dumps(projected, allow_nan=False)
    projected["actions"][0]["ready_s"] = 999
    projected["scenarios"][0]["tasks"][0]["compatible_profiles"].append("changed")
    info["reference"]["history_dates"].append("changed")
    assert problem == original and reference["history_dates"] == ["20161010", "20161017"]
    for invalid_time in (-1, float("nan"), float("inf"), True, 301):
        with pytest.raises(ValueError, match="reference.travel_time_s"):
            project_post_service_geometry(problem, _reference(invalid_time))
    with pytest.raises(ValueError, match="reference.empty_distance_m"):
        project_post_service_geometry(problem, _reference(distance=-1))


def test_same_kernel_generic_can_defer_but_protects_critical_and_keeps_limits():
    tasks = [_task("ordinary", deadline=150, service=120, source="ACTUAL_PENDING"),
             _task("future_a", release=30, deadline=30, service=30),
             _task("future_b", release=60, deadline=60, service=30)]
    # Only this known current-state edge is a genuine route. Subsequent proxy
    # geometry lets the same rolling kernel plan A,B,ordinary after WAIT.
    problem = _problem(tasks, [_edge("start", "future_a")],
                       [_action("serve", "SERVE", 120, "ordinary", "ordinary"),
                        _action("wait", "WAIT", 30, "start")])
    projected = project_post_service_geometry(problem, _reference())
    result = solver.solve_epoch(projected, "CHAIN_DEFER")
    assert result["selected_actions"][0]["kind"] == "WAIT"
    assert result["current_served"] == 0 and result["expected_served"] == 3
    assert [visit["job_id"] for visit in result["recourse_paths"][0]["jobs"]] == ["future_a", "future_b", "ordinary"]
    assert solver.validate_epoch_solution(projected, result)["valid"]
    critical = deepcopy(problem)
    critical["actions"][0]["critical"] = True
    protected = solver.solve_epoch(project_post_service_geometry(critical, _reference()), "CHAIN_DEFER")
    assert protected["critical_now"] == protected["current_served"] == 1
    assert protected["selected_actions"][0]["kind"] == "SERVE"
    invalid = deepcopy(projected)
    invalid["actions"].append(_action("fake", "SERVE", 60, "future_a", "future_a"))
    with pytest.raises(ValueError, match="ACTUAL_PENDING"):
        solver.solve_epoch(invalid, "CHAIN_DEFER")
    # Completing the task-origin proxy layer can explode raw permutations.
    # Existing limits still fail explicitly; exact compression remains an
    # existing kernel option rather than a new parameter or relaxed budget.
    jobs = [f"future_{index}" for index in range(7)]
    dense = _problem([_task(job) for job in jobs], [_edge("start", job) for job in jobs],
                     [_action("layout", "LAYOUT", 0, "start")])
    dense = project_post_service_geometry(dense, _reference())
    with pytest.raises(solver.RollingModelLimitError, match="chain limit"):
        solver.solve_epoch(dense, "CHAIN_DEFER")
    dense["exact_chain_compression"] = True
    compressed = solver.solve_epoch(dense, "CHAIN_DEFER")
    assert compressed["expected_served"] == 7
    assert compressed["model"]["scenario_chain_variables"] == 128
    assert compressed["model"]["columns"] <= 20_000 and compressed["model"]["nonzeros"] <= 150_000
    assert compressed["runtime_s"] < 10
