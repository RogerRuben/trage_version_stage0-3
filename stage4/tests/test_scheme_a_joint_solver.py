"""Four focused synthetic checks; no real-data replay or experiment gate."""
from copy import deepcopy
from fractions import Fraction
from itertools import permutations, product
import math

import numpy as np
import pytest

from stage4.dispatch import scheme_a_joint_solver as joint
from stage4.dispatch.capability_chain_rolling_solver import validate_epoch_solution


def _action(aid, kind, ready, location, *, job=None, distance=0, critical=False, carry=False, resource="C1"):
    return dict(action_id=aid, resource_id=resource, kind=kind, ready_s=ready, location_id=location,
                job_id=job, empty_distance_m=distance, critical=critical, carry_over=carry)


def _task(job, release, deadline, service, source="FORECAST"):
    return dict(job_id=job, release_s=release, deadline_s=deadline, service_time_s=service,
                source_kind=source, compatible_profiles=["C", "HV"])


def _edge(origin, target, travel=0, distance=0, profiles=("C", "HV")):
    return dict(origin_id=origin, target_job_id=target, travel_time_s=travel,
                empty_distance_m=distance, supported=True, compatible_profiles=list(profiles))


def _layout_problem():
    scenes = []
    for sid, weight in (("S1", .6), ("S2", .4)):
        scenes.append(dict(scenario_id=sid, weight=weight,
                           tasks=[_task("P", 0, 30, 30, "ACTUAL_PENDING"),
                                  _task("F" + sid, 60, 90, 30), _task("G" + sid, 120, 150, 30)],
                           connections=[_edge("BAD", "P", profiles=("HV",)),
                                        _edge("GOOD", "P", 10, 10),
                                        _edge("P", "F" + sid, 10, 10, profiles=("C",)),
                                        _edge("F" + sid, "G" + sid, 10, 10, profiles=("C",)),
                                        _edge("H_HOME", "P", profiles=("HV",))]))
    return dict(now_s=0, step_s=30, task_limit_per_scenario=64, exact_chain_compression=True,
                resources=[dict(resource_id="C1", profile_id="C", admission_end_s=1800),
                           dict(resource_id="H1", profile_id="HV", admission_end_s=1800)],
                actions=[_action("L_BAD", "LAYOUT", 0, "BAD"), _action("L_GOOD", "LAYOUT", 0, "GOOD"),
                         _action("H_WAIT", "WAIT", 30, "H_HOME", resource="H1")],
                scenarios=scenes)


def _dispatch_problem(critical=False):
    scene = dict(tasks=[_task("P", 0, 90, 570, "ACTUAL_PENDING"),
                        _task("F", 90, 120, 30), _task("G", 180, 210, 30)],
                 connections=[_edge("HUB", "F", 5, 50), _edge("F", "G", 5, 50)])
    return dict(now_s=30, step_s=30, task_limit_per_scenario=64,
                resources=[dict(resource_id="C1", profile_id="C", admission_end_s=1800)],
                actions=[_action("SERVE_P", "SERVE", 600, "P", job="P", critical=critical, carry=True),
                         _action("WAIT", "WAIT", 60, "HOME"),
                         _action("RELOCATE", "RELOCATE", 90, "HUB", distance=100)],
                scenarios=[dict(scenario_id=sid, weight=.5, **deepcopy(scene)) for sid in ("S1", "S2")])


def _oracle(problem, policy):
    """Independent all-action/all-task-permutation oracle for tiny inputs.

    No production preparer, enumerator, master, objective builder, or validator
    is used. It enumerates action tuples then scenario resource permutations,
    rejects shared customer identities, and compares exact rational objectives.
    """
    resources = sorted(problem["resources"], key=lambda row: row["resource_id"])
    action_groups = [[a for a in sorted(problem["actions"], key=lambda row: row["action_id"])
                      if a["resource_id"] == r["resource_id"]] for r in resources]
    now, step = Fraction(str(problem["now_s"])), Fraction(30)
    weights = [Fraction(str(scene["weight"])) for scene in problem["scenarios"]]
    weights = [weight / sum(weights) for weight in weights]
    best = None
    best_actions = None
    for actions in product(*action_groups):
        first_jobs = [a["job_id"] for a in actions if a["kind"] == "SERVE"]
        if len(first_jobs) != len(set(first_jobs)):
            continue
        expected = distance = Fraction(0)
        valid = True
        for scene, weight in zip(problem["scenarios"], weights):
            edge_map = {(e["origin_id"], e["target_job_id"]): e for e in scene["connections"]}
            tasks = {t["job_id"]: t for t in scene["tasks"]}
            choices = []
            for r, a in zip(resources, actions):
                first = a["job_id"] if a["kind"] == "SERVE" else None
                eligible = [jid for jid, task in tasks.items() if jid != first
                            and r["profile_id"] in task["compatible_profiles"]]
                columns = []
                for length in range(len(eligible) + 1):
                    for seq in permutations(eligible, length):
                        origin = a["location_id"]
                        ready = Fraction(str(a["ready_s"]))
                        empty = Fraction(str(a["empty_distance_m"]))
                        for jid in seq:
                            task, edge = tasks[jid], edge_map.get((origin, jid))
                            if edge is None or not edge["supported"] or r["profile_id"] not in edge["compatible_profiles"]:
                                break
                            released = Fraction(str(task["release_s"]))
                            commit = math.ceil(max(ready, released) / step) * step
                            pickup = commit + Fraction(str(edge["travel_time_s"]))
                            if commit >= Fraction(str(r["admission_end_s"])) or pickup > Fraction(str(task["deadline_s"])):
                                break
                            ready = pickup + Fraction(str(task["service_time_s"]))
                            origin = jid
                            empty += Fraction(str(edge["empty_distance_m"]))
                        else:
                            served = ((first,) if first else ()) + seq
                            columns.append((served, empty))
                choices.append(columns)
            candidate = None
            for combination in product(*choices):
                ids = [jid for jobs, _ in combination for jid in jobs]
                if len(ids) != len(set(ids)):
                    continue
                value = (len(ids), -sum((dist for _, dist in combination), Fraction(0)))
                if candidate is None or value > candidate:
                    candidate = value
            if candidate is None:
                valid = False
                break
            expected += weight * candidate[0]
            distance -= weight * candidate[1]
        if not valid:
            continue
        critical = sum(a["kind"] == "SERVE" and a["critical"] for a in actions)
        current = len(first_jobs)
        carry = sum(a["kind"] == "SERVE" and a["carry_over"] for a in actions)
        tiers = ((critical, current, carry, expected, -distance) if policy == "SERVICE_PRESERVING" else
                 (critical, expected, current, carry, -distance))
        # Exact sequential per-resource action rank tie; only after all science.
        tie = tuple(-group.index(action) for group, action in zip(action_groups, actions))
        value = tiers + tie
        if best is None or value > best:
            best, best_actions = value, tuple(a["action_id"] for a in actions)
    return best, best_actions


def _assert_oracle(problem, policy, solution):
    best, actions = _oracle(problem, policy)
    assert tuple(a["action_id"] for a in solution["selected_actions"]) == actions
    values = (solution["critical_now"], solution["current_served"], solution["carry_over_current_served"],
              solution["expected_served"], -solution["expected_empty_distance_m"])
    if policy == "CHAIN_DEFER":
        values = (values[0], values[3], values[1], values[2], values[4])
    assert np.asarray(values) == pytest.approx([float(value) for value in best[:5]])
    assert validate_epoch_solution(problem, solution)["valid"]
    assert solution["runtime_s"] < 10


def test_joint_layout_complete_multiservice_profile_links_and_pending_identity():
    problem = _layout_problem()
    solution = joint.solve_joint_epoch(problem)
    _assert_oracle(problem, "CHAIN_DEFER", solution)
    assert solution["selected_actions"][0]["action_id"] == "L_GOOD"
    assert solution["expected_served"] == 3
    c_paths = [path for path in solution["recourse_paths"] if path["resource_id"] == "C1"]
    assert all(len(path["jobs"]) == 3 for path in c_paths)
    assert all(path["jobs"][0]["source_kind"] == "ACTUAL_PENDING" for path in c_paths)
    assert all(not path["served_job_ids"] for path in solution["recourse_paths"] if path["resource_id"] == "H1")
    quality = solution["quality_certificate"]
    assert quality["LP_upper_bound"] == pytest.approx(3)
    assert quality["conditioned_tiers"] == {"critical_now": 0}
    assert quality["exact_dual_arithmetic"]
    assert solution["model"]["sparse"] and solution["model"]["complete_finite_domain"]


def test_common_relocation_vs_wait_and_service_preserving_oracle():
    problem = _dispatch_problem()
    for policy, action, expected in (("CHAIN_DEFER", "RELOCATE", 2), ("SERVICE_PRESERVING", "SERVE_P", 1)):
        solution = joint.solve_joint_epoch(problem, policy)
        _assert_oracle(problem, policy, solution)
        assert solution["selected_actions"][0]["action_id"] == action
        assert all(path["first_action_id"] == action for path in solution["recourse_paths"])
        assert solution["quality_certificate"]["LP_upper_bound"] == pytest.approx(expected)
    assert set(solution["quality_certificate"]["conditioned_tiers"]) == {
        "critical_now", "current_served", "carry_over_current_served"}
    assert not solution["model_domain"]["independent_relocation_between_future_customers_represented"]


def test_high_precision_empty_distance_is_final_no_floating_tie_face():
    problem = _layout_problem()
    problem["actions"] = [_action("A_LONG", "LAYOUT", 0, "LONG"), _action("B_SHORT", "LAYOUT", 0, "SHORT"),
                          _action("H_WAIT", "WAIT", 30, "H_HOME", resource="H1")]
    for scene in problem["scenarios"]:
        scene["connections"] = [_edge("LONG", "P", 10, 11.123456789123457),
                                _edge("SHORT", "P", 10, 10.123456789123456),
                                *scene["connections"][2:]]
    solution = joint.solve_joint_epoch(problem, include_bound=False)
    _assert_oracle(problem, "CHAIN_DEFER", solution)
    assert solution["selected_actions"][0]["action_id"] == "B_SHORT"
    distance = next(stage for stage in solution["stages"] if stage["stage"] == "expected_empty_distance_m")
    assert not distance["exact_objective_lock"]
    assert not solution["model"]["stable_first_actions_encoding"]["floating_distance_lock"]
    assert solution["quality_certificate"]["status"] == "NOT_REQUESTED"


def test_critical_conditional_bound_and_strict_integer_contract(monkeypatch):
    problem = _dispatch_problem(critical=True)
    solution = joint.solve_joint_epoch(problem)
    _assert_oracle(problem, "CHAIN_DEFER", solution)
    assert solution["critical_now"] == 1
    assert solution["quality_certificate"]["LP_upper_bound"] == pytest.approx(1)
    assert solution["quality_certificate"]["conditioned_tiers"] == {"critical_now": 1}
    real_milp = joint.milp

    def poison(*args, **kwargs):
        result = real_milp(*args, **kwargs)
        result.x = np.asarray(result.x).copy()
        result.x[0] += 1e-7
        return result

    monkeypatch.setattr(joint, "milp", poison)
    with pytest.raises(joint.JointNumericalContractError, match="raw numerical contract failed"):
        joint.solve_joint_epoch(problem)
