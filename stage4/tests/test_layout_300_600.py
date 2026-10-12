"""Small time/competition/placement checks; no Actor or simulation."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from stage4.dispatch.coarse_layout import compatible_head_weights, integer_quotas, solve_placement
from stage4.dispatch.scheme_a_city_graph import ChainTask, RouteState, build_restricted_chains
from stage4.dispatch.scheme_a_match_deadline_value import MatchDeadlineSceneDAG


def point(meters):
    return (108.95 + meters / (111320 * np.cos(np.deg2rad(34.25))), 34.25)


def task(rid, release, meters=0, duration=60):
    return ChainTask(rid, release, release + 300, duration, point(meters), point(meters),
        frozenset({"C", "HV"}), True, "20161010", str(rid))


def model(tasks):
    return MatchDeadlineSceneDAG(tasks, [dict(position=point(0)), dict(position=point(4000))],
        0, 1800, NS(global_pace=.1, pace_by_slot={0: .1}), {}, lambda: None)


def test_old_graph_opt_in_matches_dispatch_deadline_and_independent_pickup_limit():
    forecast = task(-1, 100)
    start = RouteState(point(0), 120, 300, "C")
    cfg = dict(decision_time_s=0, planning_horizon_s=1800)
    eta = [500.]
    connector = lambda *args: dict(supported=True, compatible_profiles={"C", "HV"},
        travel_time_s=eta[0], empty_distance_m=1000.)
    old, _ = build_restricted_chains("old", 1, "s", start, [forecast], [], connector, cfg, lambda: None)
    assert old == []
    cfg.update(patience_mode="MATCH_DEADLINE_AND_PICKUP_CAP", max_match_wait_s=300, max_pickup_eta_s=600)
    new, _ = build_restricted_chains("new", 1, "s", start, [forecast], [], connector, cfg, lambda: None)
    assert new[0].request_ids == (-1,)
    assert new[0].payload["finish_s"] == 680. > start.admission_end_s
    eta[0] = 601.
    assert not build_restricted_chains("late_pickup", 1, "s", start, [forecast], [], connector, cfg, lambda: None)[0]
    eta[0] = 0.
    boundary = RouteState(point(0), 300, 600, "C")
    expired = task(-1, 0)
    assert not build_restricted_chains("late_match", 1, "s", boundary, [expired], [], connector, cfg, lambda: None)[0]


def test_shared_customer_capacity_and_primary_face_tie_keep_simple_baseline():
    models = {"one": model([task(-1, 1)])}
    counts, info = solve_placement(models, {"one": 1.}, {("HV", 0, 0, 6): 1},
        {(0, 6): 1}, ["a", "b"], {}, lambda: None, tie_targets={(0, 6): [0, 1]})
    assert info["lp_expected_services"] == pytest.approx(1.)
    assert counts[(0, 6)].tolist() == [0, 1]
    assert info["shared_customer_capacity"] and info["shared_scenario_placement"]
    assert info["integer_service_optimality_claimed"] is False
    assert info["tie_status"] == "COMPATIBLE_HOTSPOT_L1_ON_RESTRICTED_PRIMARY_FACE"


def test_one_birth_placement_is_shared_across_conflicting_history_scenarios():
    models = {"left": model([task(-1, 1)]), "right": model([task(-1, 1, 4000)])}
    counts, info = solve_placement(models, {"left": .5, "right": .5}, {}, {(0, 6): 1},
        ["a", "b"], {}, lambda: None, tie_targets={(0, 6): [1, 0]})
    assert info["lp_expected_services"] == pytest.approx(.5)
    assert counts[(0, 6)].tolist() == [1, 0]


def test_continuation_and_HV_competition_can_differ_from_compatible_head_counts():
    models = {"one": model([task(-1, 1), task(-2, 301), task(-3, 1, 4000, 900)])}
    weights = compatible_head_weights(models, {"one": 1.}, (0, 6), 2)
    assert weights.tolist() == [2., 1.]
    simple = integer_quotas(weights, 1, ["a", "b"])
    counts, info = solve_placement(models, {"one": 1.}, {("HV", 0, 0, 6): 1},
        {(0, 6): 1}, ["a", "b"], {}, lambda: None, tie_targets={(0, 6): simple})
    assert simple.tolist() == [1, 0]
    assert counts[(0, 6)].tolist() == [0, 1]
    assert info["lp_expected_services"] == pytest.approx(3.)
    assert info["integer_placement_supply_preserved"]
