from copy import deepcopy
import json

import pytest

from stage4.analysis import capability_chain_instance_solver as solver


def _task(job_id, release=0, deadline=300, service=30, profiles=("HV", "C")):
    return dict(job_id=job_id, release_s=release, deadline_s=deadline,
                service_time_s=service, compatible_profiles=list(profiles))


def _resource(resource_id, profile="C", sites=("hot", "linked"), fixed="hot", start=0, end=300):
    return dict(resource_id=resource_id, profile_id=profile, start_s=start, end_s=end,
                site_ids=list(sites), fixed_site_id=fixed)


def _edge(origin, target, travel=0, distance=1, profiles=("HV", "C"), supported=True):
    return dict(origin_id=origin, target_job_id=target, travel_time_s=travel,
                empty_distance_m=distance, supported=supported, compatible_profiles=list(profiles))


def _instance(tasks, resources, connections, sites=("hot", "linked")):
    return dict(schema_version="capability_chain_instance_v1", execution_step_s=30,
                tasks=tasks, resources=resources, sites=[dict(site_id=site) for site in sites],
                connections=connections)


def test_joint_layout_chain_selection_global_uniqueness_and_stable_distance_tie():
    instance = _instance(
        [_task("a"), _task("b"), _task("c")],
        [_resource("car_1"), _resource("car_2")],
        [_edge("hot", "a", distance=100), _edge("linked", "a", distance=1),
         _edge("linked", "b", distance=1), _edge("a", "b", distance=1), _edge("b", "c", distance=1)])
    fixed = solver.solve_instance(instance, optimize_layout=False)
    joint = solver.solve_instance(instance)
    assert fixed["served"] == joint["served"] == 3
    assert joint["empty_distance_m"] == 3 < fixed["empty_distance_m"]
    jobs = [job for row in joint["selected"] for job in row["job_ids"]]
    assert len(jobs) == len(set(jobs)) == 3
    assert all(row["site_id"] == "linked" for row in joint["selected"] if row["job_ids"])
    assert joint["selected"][0]["job_ids"] == []  # Stable tie selects the first car idle.
    assert joint["selected"][1]["job_ids"] == ["a", "b", "c"]
    assert joint["kind"] == solver.KIND
    assert joint["enumeration"]["idle_columns_per_resource"] == 1
    assert sum(row["chain_count"] for row in joint["enumeration"]["per_resource_site"]) == sum(
        joint["enumeration"]["chains_per_resource"].values())
    assert solver.validate_solution(instance, joint)["valid"]
    json.dumps(joint, allow_nan=False)
    reordered = deepcopy(instance)
    for name in ("tasks", "sites", "resources", "connections"):
        reordered[name].reverse()
    assert solver.solve_instance(reordered)["selected"] == joint["selected"]
    corrupted = deepcopy(joint)
    corrupted["selected"][0] = deepcopy(corrupted["selected"][1])
    corrupted["selected"][0]["resource_id"] = "car_1"
    with pytest.raises(ValueError, match="repeated served job"):
        solver.validate_solution(instance, corrupted)


def test_original_deadline_deferred_release_grid_compatibility_and_cross_shift_finish():
    instance = _instance(
        [_task("first", release=31, deadline=60, service=17),
         _task("waiting", release=1, deadline=95, service=50),
         _task("expired", release=1, deadline=89, service=10),
         _task("future", release=121, deadline=160),
         _task("wrong_task", profiles=("HV",)), _task("wrong_edge"), _task("unsupported")],
        [_resource("car", sites=("site",), fixed="site", end=100)],
        [_edge("site", "first"), _edge("first", "waiting", travel=5),
         _edge("first", "expired", travel=0), _edge("waiting", "future"),
         _edge("site", "wrong_task"), _edge("site", "wrong_edge", profiles=("HV",)),
         _edge("site", "unsupported", supported=False)], sites=("site",))
    result = solver.solve_instance(instance)
    assert result["served"] == 2
    first, waiting = result["selected"][0]["jobs"]
    assert first["job_id"] == "first" and first["commit_s"] == 60 and first["finish_s"] == 77
    assert waiting["job_id"] == "waiting" and waiting["commit_s"] == 90
    assert waiting["pickup_s"] == 95 and waiting["finish_s"] == 145 > 100
    assert solver.validate_solution(instance, result)["checked_jobs"] == 2
    corrupt = deepcopy(result)
    corrupt["selected"][0]["jobs"][0]["commit_s"] = 30
    with pytest.raises(ValueError, match="commit_s"):
        solver.validate_solution(instance, corrupt)
    # A resource whose end falls exactly on the next grid tick cannot commit.
    boundary = _instance([_task("late", release=31, deadline=100)],
                         [_resource("car", sites=("site",), fixed="site", end=60)],
                         [_edge("site", "late")], sites=("site",))
    assert solver.solve_instance(boundary)["served"] == 0
    # Relative clock times may be negative when the resource began before the
    # chosen snapshot origin; durations and distances still must be nonnegative.
    before_origin = _instance([_task("before", release=-31, deadline=-30)],
                              [_resource("car", sites=("site",), fixed="site", start=-60, end=0)],
                              [_edge("site", "before")], sites=("site",))
    assert solver.solve_instance(before_origin)["selected"][0]["jobs"][0]["commit_s"] == -30


def test_complete_sparse_lp_gap_exact_integer_solution_and_hard_limits(monkeypatch):
    # Timed chains are C:{a,b}/{c,d}, HV:{a,c}/{b,d}. Each second task
    # finishes at 60 and no third commitment fits the end=60 resource window.
    tasks = [_task(job, deadline=30) for job in ("a", "b", "c", "d")]
    resources = [_resource("c_car", sites=("site",), fixed="site", end=60),
                 _resource("h_car", profile="HV", sites=("site",), fixed="site", end=60)]
    edges = [_edge("site", job) for job in ("a", "b", "c", "d")]
    edges += [_edge("a", "b", profiles=("C",)), _edge("c", "d", profiles=("C",)),
              _edge("a", "c", profiles=("HV",)), _edge("b", "d", profiles=("HV",))]
    instance = _instance(tasks, resources, edges, sites=("site",))
    original_linprog = solver.linprog

    def sparse_linprog(*args, **kwargs):
        from scipy.sparse import issparse
        assert issparse(kwargs["A_ub"]) and issparse(kwargs["A_eq"])
        return original_linprog(*args, **kwargs)

    monkeypatch.setattr(solver, "linprog", sparse_linprog)
    result = solver.solve_instance(instance)
    assert result["served"] == 3
    assert result["lp"]["served_upper_bound"] == pytest.approx(4)
    assert result["lp"]["gap_jobs"] == pytest.approx(1)
    assert result["lp"]["rows"] == 6
    assert result["lp"]["columns"] == 14 == sum(result["enumeration"]["chains_per_resource"].values())
    assert result["lp"]["nonzeros"] == 30
    assert result["lp"]["certificate"]["checked_columns"] == 14
    assert result["lp"]["certificate"]["primal_dual_gap"] < 1e-8
    assert result["validation"]["valid"]
    no_routes = deepcopy(instance)
    no_routes["connections"] = [_edge("site", "a", travel=None, distance=None, supported=False)]
    idle_only = solver.solve_instance(no_routes)
    assert idle_only["served"] == idle_only["lp"]["served_upper_bound"] == 0
    assert idle_only["lp"]["columns"] == idle_only["lp"]["nonzeros"] == 2
    assert all(not row["jobs"] for row in idle_only["selected"])
    assert solver.validate_solution(no_routes, idle_only)["valid"]
    bad_supported = deepcopy(no_routes)
    bad_supported["connections"][0]["supported"] = True
    with pytest.raises(ValueError, match="connection.travel_time_s"):
        solver.solve_instance(bad_supported)
    oversized = deepcopy(instance)
    oversized["tasks"] = [_task(f"job_{index}") for index in range(11)]
    with pytest.raises(solver.InstanceLimitError, match="tasks limit"):
        solver.solve_instance(oversized)
    with monkeypatch.context() as limited:
        limited.setattr(solver, "MAX_CHAINS_PER_RESOURCE", 3)
        with pytest.raises(solver.InstanceLimitError, match="not truncated"):
            solver.solve_instance(instance)
    with monkeypatch.context() as limited:
        limited.setattr(solver, "MAX_DP_STATES", 2)
        with pytest.raises(solver.InstanceLimitError, match="DP state limit"):
            solver.solve_instance(instance)
