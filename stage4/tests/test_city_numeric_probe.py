from dataclasses import replace
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult
from scipy.sparse import csr_matrix, load_npz

from stage4.analysis.city_numeric_probe import LegacyNumericProbe, compare_saved_problem, load_problem, save_problem
from stage4.dispatch import city_defer_native as fixed
from stage4.dispatch import flexibility_model as base


def _problem():
    return base.Problem(0., (base.Vehicle(1357911, "HV", "PRIVATE_COORDINATE", -0., 1000., False),),
        (base.Request(2468022, 0., 300., 30., "PRIVATE_DROPOFF", frozenset(("C",))),),
        (base.CurrentPickup(1357911, 2468022, 5.),), (),
        base.ModelLimits(solver_time_limit_s=10., recourse_mode="FLOW_RELAXED"))


def _legacy():
    legacy = NS()
    def milp(c, **kwargs):
        x = np.asarray([5e-7, 1-5e-7])
        return OptimizeResult(x=x, status=0, message="Optimal", success=True,
                              fun=float(np.asarray(c) @ x), mip_dual_bound=-1., mip_gap=0., mip_node_count=0)
    legacy.base = NS(milp=milp)
    def component(problem, waiting, options, by_vehicle, by_request, option_ids, records, budget):
        result = legacy.base.milp(np.asarray([0., -1.]), integrality=np.ones(2, dtype=np.int8),
            bounds=Bounds(np.zeros(2), np.ones(2)),
            constraints=LinearConstraint(csr_matrix([[1., 1.], [0., 1.]]), [1., -np.inf], [1., 1.]),
            options=dict(time_limit=10., mip_rel_gap=0.))
        if np.max(np.abs(result.x - np.rint(result.x))) > 1e-7:
            raise RuntimeError("city component returned a noninteger common first action")
    legacy._solve_component = component
    def solve(problem):
        vehicles, requests = base._validate(problem)
        options = base._options(problem, vehicles, requests)
        return legacy._solve_component(problem, requests, options, {1357911: [0, 1]},
                                       {2468022: [1]}, [0, 1], [], None)
    legacy.solve_city_defer = solve
    return legacy


def test_problem_serialization_roundtrips_exact_float_tuple_frozenset(tmp_path):
    problem = _problem()
    scene = base.Scenario("HISTORY", 1/3, -7*86400., (), ())
    problem = replace(problem, scenarios=(scene,))
    path = save_problem(problem, tmp_path / "problem.json")
    restored = load_problem(path)
    assert restored == problem and isinstance(restored.vehicles, tuple)
    assert isinstance(restored.waiting_requests[0].compatible_profiles, frozenset)
    assert restored.vehicles[0].ready_time_s.hex() == (-0.).hex()
    assert restored.scenarios[0].probability.hex() == (1/3).hex()


def test_probe_records_actual_binary_mask_stage_and_sparse_failure_then_restores(tmp_path):
    legacy, problem = _legacy(), _problem()
    original_component, original_milp = legacy._solve_component, legacy.base.milp
    probe = LegacyNumericProbe(legacy, tmp_path)
    probe.observe_problem(problem, 0)
    probe.install()
    try:
        with pytest.raises(RuntimeError, match="noninteger"):
            legacy.solve_city_defer(problem)
    finally:
        probe.close()
    assert legacy._solve_component is original_component and legacy.base.milp is original_milp
    assert probe.last_failure_path == tmp_path / "numerical_failure.json"
    failure = json.loads(probe.last_failure_path.read_text())
    stage = failure["last_stage"]
    assert stage["legacy_stage_ordinal"] == 2 and stage["stage"] == "expected_total_service"
    assert stage["skipped_constant_level_ordinals"] == [1, 4]
    assert stage["max_binary_deviation"] == pytest.approx(5e-7)
    assert stage["max_raw_row_violation"] == stage["max_raw_bound_violation"] == 0
    assert "1357911" not in json.dumps(failure) and "PRIVATE_COORDINATE" not in json.dumps(failure)
    matrix = load_npz(tmp_path / "last_milp_matrix.npz")
    assert matrix.shape == (2, 2) and matrix.nnz == 3
    arrays = np.load(tmp_path / "last_milp_arrays.npz")
    assert np.array_equal(arrays["integrality"], [1, 1]) and arrays["x"][0] == 5e-7
    assert load_problem(tmp_path / "problem.json") == problem


def test_saved_same_problem_comparison_retains_original_failure_and_reports_maxdev(tmp_path):
    problem, legacy = _problem(), _legacy()
    path = save_problem(problem, tmp_path / "problem.json")
    failure = tmp_path / "numerical_failure.json"
    failure.write_text('{"original":true}', encoding="utf-8")
    saved_bytes = path.read_bytes()
    original_milp = legacy.base.milp
    summary = compare_saved_problem(path, legacy, fixed)
    assert summary["legacy"]["same_integer_postcheck_failure"]
    assert summary["legacy"]["max_binary_deviation"] == pytest.approx(5e-7)
    assert summary["fixed"]["status"] == "SUCCESS" and summary["fixed"]["objective"]["current_service"] == 1
    assert summary["fixed"]["numerical_contract"]["max_certified_row_violation"] == 0
    assert legacy.base.milp is original_milp
    assert path.read_bytes() == saved_bytes and failure.read_text() == '{"original":true}'
    assert not summary["original_missing_vector_recovered"]
