from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import numpy as np

from stage4.dispatch.city_defer_native import (
    CityDeferNativeAdapter, build_city_future_problem, solve_city_defer,
)
from stage4.dispatch.flexibility_model import (
    CurrentPickup, FuturePickup, ModelLimits, Problem, Request, Scenario, Vehicle, solve_dispatch,
)
from stage4.dispatch.flexibility_native import position_key
from stage4.dispatch.solver import AssignmentArc


def _defer_case():
    vehicles = (Vehicle(1, "C", "V1", 0, 600, False), Vehicle(2, "HV", "V2", 60, 100, False),
                Vehicle(3, "HV", "V3", 0, 600, False))
    waiting = (Request(10, 0, 90, 1000, "D10", frozenset(("C",))),)
    future = (Request(20, 30, 30, 30, "D20", frozenset(("C",))), Request(21, 30, 30, 30, "D21"))
    scene = Scenario("HISTORY", 1., -1000, future,
        (FuturePickup(1, None, 20, 0, "V1"), FuturePickup(2, None, 10, 0, "V2"),
         FuturePickup(3, None, 21, 0, "V3")))
    return Problem(0, vehicles, waiting, (CurrentPickup(1, 10, 0),), (scene,),
                   ModelLimits(solver_time_limit_s=10, recourse_mode="FLOW_RELAXED"))


def _vector(problem, decision):
    requests = {request.request_id: request for request in problem.waiting_requests}
    etas = {(pickup.vehicle_id, pickup.request_id): pickup.pickup_eta_s for pickup in problem.current_pickups}
    return (sum(requests[rid].critical for _, rid in decision.selected_pairs),
            decision.expected_total_service_count, decision.immediate_service_count,
            sum(requests[rid].carry_over for _, rid in decision.selected_pairs),
            -sum(etas[pair] for pair in decision.selected_pairs))


def test_decomposed_lookahead_can_defer_and_matches_joint_one_next_model():
    problem = _defer_case()
    actual = solve_city_defer(problem)
    assert actual.selected_pairs == () and actual.expected_total_service_count == 3
    assert actual.mip_component_count == actual.pure_future_component_count == 1
    assert _vector(problem, actual) == pytest.approx(_vector(problem, solve_dispatch(problem, "LOOKAHEAD")))
    critical = replace(problem, waiting_requests=(replace(problem.waiting_requests[0], critical=True),))
    protected = solve_city_defer(critical)
    assert protected.selected_pairs == ((1, 10),) and protected.critical_matched == 1
    assert _vector(critical, protected) == pytest.approx(_vector(critical, solve_dispatch(critical, "LOOKAHEAD")))
    rejected = replace(problem, scenarios=(replace(problem.scenarios[0],
        new_requests=(replace(problem.scenarios[0].new_requests[0], passenger_accepts_av=False),
                      problem.scenarios[0].new_requests[1])),))
    assert solve_city_defer(rejected).selected_pairs == ((1, 10),)
    assert not actual.provenance["complete_chain_equivalence"] and not actual.provenance["resource_fallback"]
    with pytest.raises(ValueError, match="resource cap"):
        solve_city_defer(replace(problem, limits=replace(problem.limits, max_variables=1)))


def test_future_controls_share_known_states_and_generic_only_changes_after_task():
    here = position_key(108.93, 34.24)
    far = position_key(109.03, 34.24)
    request = Request(10, 0, 300, 30, far, frozenset(("C",)))
    future = Request(20, 60, 300, 30, here, frozenset(("C",)))
    problem = Problem(0, (Vehicle(1, "C", here, 0, 600, False),), (request,),
                      (CurrentPickup(1, 10, 0),), (), ModelLimits(solver_time_limit_s=10))
    scenes = [(Scenario("HISTORY", 1., -100, (future,), ()), {20: here})]
    forecast = NS(global_pace=.1, pace_by_slot={})
    cfg = dict(future_top_k_vehicles=5, future_search_radius_m=2000,
               forecast_train_dates=["20161010", "20161017", "20161024"])
    physical, _ = build_city_future_problem(problem, scenes, {10: here}, forecast, cfg, "LOCATION_AWARE_DEFER")
    generic, diagnostics = build_city_future_problem(problem, scenes, {10: here}, forecast, cfg,
        "TIME_TYPE_DEFER", generic_reference=dict(travel_time_s=129, empty_distance_m=762))
    assert physical.current_pickups == generic.current_pickups == problem.current_pickups
    known = lambda p: tuple(arc for arc in p.scenarios[0].pickups if arc.after_request_id is None)
    assert known(physical) == known(generic)
    assert not any(arc.after_request_id == 10 for arc in physical.scenarios[0].pickups)
    proxies = [arc for arc in generic.scenarios[0].pickups if arc.after_request_id == 10]
    assert len(proxies) == 1 and proxies[0].pickup_eta_s == 129
    assert proxies[0].origin_position == far and diagnostics["generic_reference"]["empty_distance_m"] == 762
    assert diagnostics["candidate_rule_is_an_additional_top_k_approximation"]
    assert diagnostics["no_physical_recourse_certificate"]


def test_native_filter_replacement_keeps_original_execution_indices_and_no_fallback(monkeypatch):
    cfg = dict(solver_time_limit_s=10)
    replacement = AssignmentArc(1, 10, 5., True, False, payload={"validated": True})
    original = [AssignmentArc(9, 99, 100., False, False, payload={"wrong": True}),
                AssignmentArc(1, 10, 100., True, False, payload={"wrong": True})]
    def physical_filter(c, arcs, now):
        return [replacement]
    physical_filter.last_diagnostics = dict(input_arc_count=2, validated_arc_count=1,
        original_indices=[1], request_ids=[99, 10], counts={"av_pickup_arcs_retained": 1},
        rejection_reason_counts={"UNSUPPORTED": 1}, routing_budget_is_separate_from_solver=True)
    adapter = CityDeferNativeAdapter("LOCATION_AWARE_DEFER", None, cfg, current_arc_filter=physical_filter)
    problem = _defer_case()
    problem = replace(problem, waiting_requests=(replace(problem.waiting_requests[0], critical=True),),
                      current_pickups=(CurrentPickup(1, 10, 5.),))
    monkeypatch.setattr(adapter, "_problem", lambda c, arcs, waiting, now: problem)
    result = adapter.solve(NS(), original, [10], 0)
    assert result.selected_indices == (1,) and original[1] is replacement
    assert original[result.selected_indices[0]].payload == {"validated": True}
    assert result.pickup_eta_optimum_s == 5 and adapter.rows[-1]["physical_validation_time_s"] >= 0
    assert adapter.rows[-1]["resource_fallback"] is None
    summary = adapter.rows[-1]["first_arc_filter_diagnostics"]
    assert summary["input_arc_count"] == 2 and summary["validated_arc_count"] == 1
    assert summary["counts"] == {"av_pickup_arcs_retained": 1}
    assert "original_indices" not in summary and "request_ids" not in summary
    capped = replace(problem, limits=replace(problem.limits, max_variables=1))
    monkeypatch.setattr(adapter, "_problem", lambda *args: capped)
    with pytest.raises(ValueError, match="resource cap"):
        adapter.solve(NS(), original, [10], 0)


def test_generic_equal_time_resources_are_distributed_across_future_requests():
    here, far = position_key(108.93, 34.24), position_key(109.03, 34.24)
    vehicles = tuple(Vehicle(vid, "C", here, 0, 600, False) for vid in range(1, 11))
    waiting = tuple(Request(100 + vid, 0, 300, 30, far, frozenset(("C",))) for vid in range(1, 11))
    pickups = tuple(CurrentPickup(vid, 100 + vid, 0) for vid in range(1, 11))
    problem = Problem(0, vehicles, waiting, pickups, (), ModelLimits(solver_time_limit_s=10))
    future = tuple(Request(rid, 60, 300, 30, here, frozenset(("C",))) for rid in range(200, 212))
    scenes = [(Scenario("HISTORY", 1., -100, future, ()), {request.request_id: here for request in future})]
    cfg = dict(future_top_k_vehicles=5, future_search_radius_m=2000, forecast_seed=20261004,
               forecast_train_dates=["20161010", "20161017", "20161024"])
    forecast = NS(global_pace=.1, pace_by_slot={})
    positions = {request.request_id: here for request in waiting}
    arguments = (problem, scenes, positions, forecast, cfg, "TIME_TYPE_DEFER")
    projected, _ = build_city_future_problem(*arguments,
        generic_reference=dict(travel_time_s=129, empty_distance_m=762))
    repeated, _ = build_city_future_problem(*arguments,
        generic_reference=dict(travel_time_s=129, empty_distance_m=762))
    assert projected.scenarios == repeated.scenarios
    future_arcs = [arc for arc in projected.scenarios[0].pickups
                   if arc.after_request_id is not None and arc.request_id >= 200]
    for request in future:
        assert len({arc.vehicle_id for arc in future_arcs if arc.request_id == request.request_id}) == 5
    assert len({arc.vehicle_id for arc in future_arcs}) > 5


def test_tight_numeric_contract_exact_thirds_locks_and_rejects_real_fraction(monkeypatch):
    from stage4.dispatch import city_defer_native as city
    problem = _defer_case()
    problem = replace(problem, scenarios=tuple(replace(problem.scenarios[0], scenario_id=f"H{i}", probability=1/3)
                                               for i in range(3)))
    real_milp = city.milp
    magnitude = [city.MIP_FEASIBILITY_TOLERANCE / 8]
    seen_options = []

    def perturbed_milp(*args, **kwargs):
        assert kwargs["options"]["mip_feasibility_tolerance"] == 1e-9
        assert kwargs["options"]["primal_feasibility_tolerance"] == 1e-9
        seen_options.append(kwargs["options"])
        result = real_milp(*args, **kwargs)
        integer_indices = np.flatnonzero(kwargs["integrality"])
        selected = next(index for index in integer_indices if result.x[index] > .5)
        idle = next(index for index in integer_indices if result.x[index] < .5)
        # Numerical perturbation is certified only within the explicit solver
        # tolerance, with exact capacity/previous-grid locks rechecked after
        # binary projection and integer matching reconstruction.
        result.x[selected] -= magnitude[0]
        result.x[idle] += magnitude[0]
        objective = args[0] if args else kwargs["c"]
        result.fun = float(np.asarray(objective) @ result.x)
        return result

    monkeypatch.setattr(city, "milp", perturbed_milp)
    decision = city.solve_city_defer(problem)
    assert seen_options and decision.selected_pairs == () and decision.expected_total_service_count == 3
    contract = decision.numerical_contract
    assert contract["expected_service_lattice_denominator"] == 3
    assert 0 < contract["max_binary_deviation"] <= 1e-9 < 1e-7
    assert contract["max_certified_row_violation"] == 0
    assert contract["solver_options_receipt"]["bundled_options_manager_and_pass_options_verified"]
    tiers = [stage for stage in decision.stages if stage["stage"] == "expected_total_service"]
    assert all(stage["objective_lattice_denominator"] == 3 and stage["exact_objective_lock"]
               and type(stage["grid_units"]) is int for stage in tiers)
    magnitude[0] = 1e-4
    with pytest.raises(city.CityNumericalContractError, match="binary deviation") as failure:
        city.solve_city_defer(problem)
    assert failure.value.diagnostics["max_binary_deviation"] > 1e-7
    assert "request_ids" not in failure.value.diagnostics
