"""Small independent current-action oracle; no replay or future data."""
from itertools import product
import random

import pytest

from stage4.dispatch.scheme_a_city_master import CurrentAction
from stage4.dispatch.scheme_a_current_flow import solve_current_flow


def action(vid, kind, rid=None, distance=0, critical=False, carry=False, label=""):
    return CurrentAction(f"{vid}_{kind}_{rid}_{label}", vid, kind, rid,
        critical, carry, float(distance), {"relocation_slot_s":900} if kind == "RELOCATE" else None)


def vector(chosen, values, policy):
    critical = sum(a.critical for a in chosen)
    served = sum(a.kind == "SERVE" for a in chosen)
    carry = sum(a.carry_over for a in chosen)
    expected = 3*served + sum(values[a.action_id] for a in chosen)
    distance = -sum(round(a.empty_distance_m) for a in chosen)
    return ((critical,served,carry,expected,distance) if policy == "SERVICE_PRESERVING"
            else (critical,expected,served,carry,distance))


def oracle(actions, values, policy, cap):
    groups = [[a for a in actions if a.vehicle_id == vid] for vid in sorted({a.vehicle_id for a in actions})]
    scores = []
    for chosen in product(*groups):
        ids = [a.request_id for a in chosen if a.kind == "SERVE"]
        if len(ids) == len(set(ids)) and sum(a.kind == "RELOCATE" for a in chosen) <= cap:
            scores.append(vector(chosen,values,policy))
    return max(scores)


def test_sparse_current_flow_matches_exhaustive_lexicographic_oracle():
    for seed in range(12):
        rng = random.Random(seed)
        actions, values = [], {}
        for vid in range(4):
            group = [action(vid,"WAIT"), action(vid,"RELOCATE",distance=20+vid)]
            for rid in range(3):
                if rng.random() < .7:
                    group.append(action(vid,"SERVE",rid, distance=1+rng.randrange(20),
                        critical=rid == 0,carry=rid != 2))
            actions.extend(group)
            values.update({a.action_id:0 if a.kind == "WAIT" else rng.randrange(-9,10) for a in group})
        for cap in (0,1,2):
            for policy in ("SERVICE_PRESERVING","CHAIN_DEFER"):
                result = solve_current_flow(actions,values,policy=policy,move_cap=cap)
                assert result["score_units"] == oracle(actions,values,policy,cap)[:4]
                reference = solve_current_flow(actions,values,policy=policy,move_cap=cap,backend="INTEGER_REFERENCE")
                assert reference["score_units"] == result["score_units"]
                assert result["opt_status"] == "OPTIMAL_PRIMARY_CURRENT_FLOW_APPROXIMATE_FUTURE"
                assert result["distance_global_optimality_claimed"] is False
                assert result["future_chain_variables"] == result["routing_queries"] == 0
                assert result["dense_matrix"] is False
                assert result["current_moves"] <= cap


def test_policy_tradeoff_keeps_critical_service_but_allows_ordinary_deferral():
    wait, serve = action(1,"WAIT"), action(1,"SERVE",7,distance=10)
    values = {wait.action_id:0, serve.action_id:-9}
    assert solve_current_flow([wait,serve],values)["current_served"] == 1
    assert solve_current_flow([wait,serve],values,policy="CHAIN_DEFER")["current_served"] == 0
    serve = action(1,"SERVE",7,distance=10,critical=True)
    values[serve.action_id] = -9
    assert solve_current_flow([wait,serve],values,policy="CHAIN_DEFER")["critical_now"] == 1


def test_limits_return_feasible_named_backup_not_fake_optimal_or_exception(monkeypatch):
    from stage4.dispatch import scheme_a_current_flow as kernel
    actions = [action(v,"WAIT") for v in range(3)]
    actions += [action(v,"SERVE",v,distance=2,critical=v == 0) for v in range(3)]
    values = {a.action_id:0 for a in actions}
    cap = solve_current_flow(actions,values,maximum_actions=1)
    expired = solve_current_flow(actions,values,deadline_s=0.)
    assert cap["opt_status"] == expired["opt_status"] == "FEASIBLE_BACKUP_SERVICE_FIRST"
    assert cap["current_served"] == expired["current_served"] == 3
    ticks = iter([0.,0.,0.,100.,100.])
    monkeypatch.setattr(kernel,"perf_counter",lambda:next(ticks,100.))
    timed = solve_current_flow(actions,values,backend="INTEGER_REFERENCE")
    assert timed["opt_status"] == "FEASIBLE_TIME_LIMIT"
    assert timed["current_served"] == 3 and timed["critical_now"] == 1


def test_current_schema_refuses_future_or_committed_resources_and_missing_wait():
    wait = action(1,"WAIT")
    for invalid in (action(1,"SERVE",-1), action(1,"BUSY"), action(1,"LAYOUT")):
        with pytest.raises(ValueError):
            solve_current_flow([wait,invalid],{a.action_id:0 for a in (wait,invalid)})
    serve = action(1,"SERVE",7)
    with pytest.raises(ValueError):
        solve_current_flow([serve],{serve.action_id:0})
