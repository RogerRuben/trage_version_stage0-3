"""Few mathematical integration tests, not a field-by-field evidence framework."""
from dataclasses import replace
from itertools import product

import pytest

from stage3.odd_tod.research_compatibility import (
    ResearchMovement, compatible_profiles, evaluate_research_compatibility,
)
from stage4.analysis.flexibility_small_instances import NC, POLICIES, small_case
from stage4.dispatch.flexibility_model import (
    CurrentPickup, FuturePickup, ModelLimits, Problem, Request, Scenario, Vehicle,
    score_fixed_action, solve_dispatch,
)


def brute_expected_value(problem):
    """Independent enumeration of all tiny first actions and next matchings."""
    current = {r.request_id: r for r in problem.waiting_requests}
    choices = []
    for vehicle in problem.vehicles:
        choices.append([None] + [a for a in problem.current_pickups
                                 if a.vehicle_id == vehicle.vehicle_id])
    best = -1
    for action in product(*choices):
        chosen = [a.request_id for a in action if a is not None]
        if len(set(chosen)) != len(chosen):
            continue
        states = []
        feasible = True
        for vehicle, arc in zip(problem.vehicles, action):
            if arc is None:
                states.append((vehicle, None, max(problem.now_s, vehicle.ready_time_s)))
                continue
            request = current[arc.request_id]
            completion = problem.now_s + arc.pickup_eta_s + request.pickup_overhead_s + request.predicted_service_time_s
            if (vehicle.ready_time_s > problem.now_s
                    or problem.now_s + arc.pickup_eta_s > request.pickup_deadline_s
                    or completion > vehicle.availability_end_s
                    or (vehicle.profile_id != "HV" and
                        (vehicle.profile_id not in request.compatible_profiles or not request.passenger_accepts_av))):
                feasible = False
                break
            states.append((vehicle, request.request_id, completion))
        if not feasible:
            continue
        value = float(len(chosen))
        for scenario in problem.scenarios:
            all_requests = {**current, **{r.request_id: r for r in scenario.new_requests}}
            later_choices = []
            for vehicle, after, ready in states:
                eligible = [None]
                for arc in scenario.pickups:
                    if arc.vehicle_id != vehicle.vehicle_id or arc.after_request_id != after or arc.request_id in chosen:
                        continue
                    request = all_requests[arc.request_id]
                    arrival = max(problem.now_s + problem.limits.rolling_step_s,
                                  ready, request.release_time_s) + arc.pickup_eta_s
                    if (arrival <= request.pickup_deadline_s and
                            arrival + request.pickup_overhead_s + request.predicted_service_time_s <= vehicle.availability_end_s and
                            (vehicle.profile_id == "HV" or
                             (request.passenger_accepts_av and vehicle.profile_id in request.compatible_profiles))):
                        eligible.append(arc.request_id)
                later_choices.append(eligible)
            max_later = 0
            for matching in product(*later_choices):
                served = [r for r in matching if r is not None]
                if len(served) == len(set(served)):
                    max_later = max(max_later, len(served))
            value += scenario.probability * max_later
        best = max(best, value)
    return best


def test_nested_profiles_and_explicit_control_assumptions():
    for maneuver in ("STRAIGHT", "RIGHT", "LEFT", "ROUNDABOUT", "UTURN"):
        for signalized in (False, True):
            movements = (ResearchMovement(maneuver, signalized, "SYNTHETIC"),)
            allowed = compatible_profiles(movements, NC)
            assert ("C" not in allowed or "M" in allowed) and ("M" not in allowed or "A" in allowed)
    assert compatible_profiles((ResearchMovement("LEFT", False, "SYNTHETIC"),), NC) == {"M", "A"}
    assert compatible_profiles((ResearchMovement("UTURN", True, "SYNTHETIC"),), NC) == {"A"}
    assert compatible_profiles((ResearchMovement("LEFT", True, "SYNTHETIC", True),), NC) == set()
    with pytest.raises(ValueError, match="cannot prove absence"):
        ResearchMovement("LEFT", False, "POSITIVE_EVIDENCE")
    with pytest.raises(ValueError, match="control must be explicit"):
        ResearchMovement("LEFT", None, "SYNTHETIC")


def test_traffic_budget_and_unknown_not_extra_gate():
    boundary = {**NC, "NC": .5, "SC": .05, "U": .45}
    for profile in ("C", "M", "A"):
        result = evaluate_research_compatibility(profile, (), boundary)
        assert result.compatible and result.unknown_traffic_share == .45
    beyond = {**boundary, "SC": .051, "NC": .499}
    assert not evaluate_research_compatibility("M", (), beyond).compatible
    assert evaluate_research_compatibility("A", (), beyond).compatible
    mixed = {**NC, "NC": .9, "MR": .1}
    assert compatible_profiles((), mixed) == {"M", "A"}


@pytest.mark.parametrize("name,expected", [
    ("flexibility_reservation", (1, 2, 2)),
    ("location_counterexample", (2, 1, 2)),
    ("forecast_error_counterexample", (2, 1, 1)),
])
def test_contrasts_and_brute_force_optimum(name, expected):
    problem, realized = small_case(name)
    scores = []
    for policy in POLICIES:
        decision = solve_dispatch(problem, policy)
        score = score_fixed_action(problem, decision.selected_pairs, realized)
        scores.append(score.expected_total_service_count)
        assert score.selected_pairs == decision.selected_pairs
        assert decision.variable_count <= 10 and decision.sparse_matrix_bytes < 2048
    assert tuple(scores) == expected
    lookahead = solve_dispatch(problem, "LOOKAHEAD")
    assert lookahead.expected_total_service_count == pytest.approx(brute_expected_value(problem))


def test_nonanticipativity_and_probability_weighting():
    reserve, _ = small_case("flexibility_reservation")
    location, _ = small_case("location_counterexample")
    scenarios = (replace(reserve.scenarios[0], probability=.75),
                 replace(location.scenarios[0], probability=.25))
    problem = replace(reserve, scenarios=scenarios)
    result = solve_dispatch(problem, "LOOKAHEAD")
    assert result.selected_pairs == ((2, 10),)  # ONE common first action, not one per scenario.
    assert result.expected_total_service_count == pytest.approx(1.75)
    assert result.expected_total_service_count == pytest.approx(brute_expected_value(problem))
    swapped = replace(problem, scenarios=(replace(scenarios[0], probability=.25),
                                         replace(scenarios[1], probability=.75)))
    assert solve_dispatch(swapped, "LOOKAHEAD").selected_pairs == ((1, 10),)


def test_time_chain_and_busy_vehicle_recourse():
    # Arrival deadline and service completion/session end are different constraints.
    request = Request(1, 0, 10, 100, "END", pickup_overhead_s=20)
    problem = Problem(0, (Vehicle(1, "HV", "START", 0, 130),), (request,), (CurrentPickup(1, 1, 10),))
    for policy in POLICIES:
        assert solve_dispatch(problem, policy).selected_pairs == ((1, 1),)
        shorter = replace(problem, vehicles=(replace(problem.vehicles[0], availability_end_s=129),))
        assert solve_dispatch(shorter, policy).selected_pairs == ()
    future = Request(2, 60, 80, 10, "END")
    scenario = Scenario("F", 1, 0, (future,), (FuturePickup(1, None, 2, 10, "START"),))
    busy = replace(problem, vehicles=(Vehicle(1, "HV", "START", 50, 130),), scenarios=(scenario,))
    result = solve_dispatch(busy, "LOOKAHEAD")
    assert result.selected_pairs == () and result.expected_next_service_count == 1


def test_waiting_request_is_not_counted_twice():
    request = Request(1, 0, 300, 10, "END")
    scenario = Scenario("F", 1, 0, (), (
        FuturePickup(1, None, 1, 0, "START"), FuturePickup(1, 1, 1, 0, "END"),
    ))
    problem = Problem(0, (Vehicle(1, "HV", "START", 0, 1000),), (request,),
                      (CurrentPickup(1, 1, 0),), (scenario,))
    result = solve_dispatch(problem, "LOOKAHEAD")
    assert result.expected_total_service_count == 1 and result.expected_next_service_count == 0


def test_no_future_information_and_no_forecast_baseline_equivalence():
    problem, _ = small_case("flexibility_reservation")
    late = replace(problem, scenarios=(replace(problem.scenarios[0], information_available_at_s=1),))
    for policy in POLICIES:
        with pytest.raises(ValueError, match="future information"):
            solve_dispatch(late, policy)
    absent = replace(problem, scenarios=())
    assert solve_dispatch(absent, "LOOKAHEAD").selected_pairs == solve_dispatch(absent, "MYOPIC").selected_pairs


def test_sparse_resource_limits_and_post_service_position():
    problem, _ = small_case("flexibility_reservation")
    with pytest.raises(ValueError, match="variable resource cap"):
        solve_dispatch(replace(problem, limits=ModelLimits(max_variables=2)), "LOOKAHEAD")
    with pytest.raises(ValueError, match="sparse model resource cap"):
        solve_dispatch(replace(problem, limits=ModelLimits(max_nonzeros=2)), "LOOKAHEAD")
    scenario = problem.scenarios[0]
    broken = replace(scenario.pickups[0], origin_position="WRONG_PLACE")
    with pytest.raises(ValueError, match="post-service vehicle position"):
        solve_dispatch(replace(problem, scenarios=(replace(scenario, pickups=(broken,)),)), "LOOKAHEAD")
