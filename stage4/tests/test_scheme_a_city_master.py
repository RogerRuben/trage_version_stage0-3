"""Four focused synthetic checks for the restricted city master."""
from collections import Counter
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from stage4.dispatch import scheme_a_city_master as city


def test_effective_offline_budget_is_30_and_realtime_is_still_hard_capped_10(monkeypatch):
    clock={"now":0.}
    monkeypatch.setattr(city,"perf_counter",lambda:clock["now"])
    realtime=city._Budget(30.)
    offline=city._Budget(30.,offline_layout=True)
    assert realtime.seconds == 10. and offline.seconds == 30.
    assert city._Budget(90.,offline_layout=True).seconds == 30.
    clock["now"]=11.
    assert offline.remaining("offline") == 19.
    with pytest.raises(city.CityMasterTimeout):
        realtime.remaining("realtime")
    action=city.CurrentAction("a",1,"WAIT",None)
    result=city.solve_city_master([action],[],{"h":1.},time_limit_s=30.,offline_layout=True)
    assert result["effective_integer_master_budget_s"] == 30.
    result=city.solve_city_master([action],[],{"h":1.},time_limit_s=30.)
    assert result["effective_integer_master_budget_s"] == 10.


def _action(aid, vid, kind="WAIT", rid=None, distance=0., critical=False, carry=False, slot=0):
    payload = {"relocation_slot_s": slot} if kind == "RELOCATE" else None
    return city.CurrentAction(aid, vid, kind, rid, critical, carry, distance, payload)


def _chain(cid, sid, vid, aid, requests, distance=0., slots=()):
    return city.ServiceChain(cid, sid, vid, aid, tuple(requests), distance, None, tuple(slots))


def _oracle(actions, chains, weights, policy, cap=50):
    """Independent action/optional-chain exhaustive enumeration for tiny data."""
    groups = [[a for a in actions if a.vehicle_id == vid] for vid in sorted({a.vehicle_id for a in actions})]
    rational = {sid: Fraction(str(weight)) for sid, weight in weights.items()}
    total = sum(rational.values())
    rational = {sid: weight / total for sid, weight in rational.items()}
    best = None
    best_ids = None
    for chosen_actions in product(*groups):
        first = [a.request_id for a in chosen_actions if a.kind == "SERVE"]
        if len(first) != len(set(first)):
            continue
        current_moves = Counter(a.payload["relocation_slot_s"] for a in chosen_actions if a.kind == "RELOCATE")
        if sum(current_moves.values()) > cap:
            continue
        options = [[None, *(c for c in chains if c.action_id == a.action_id and c.scenario_id == sid)]
                   for a in chosen_actions for sid in sorted(weights)]
        for continuation in product(*options):
            chosen_chains = [chain for chain in continuation if chain is not None]
            feasible = True
            for sid in weights:
                requests = first + [rid for c in chosen_chains if c.scenario_id == sid for rid in c.request_ids]
                moves = Counter(current_moves)
                for chain in chosen_chains:
                    if chain.scenario_id == sid:
                        moves.update(chain.relocation_slots)
                if len(requests) != len(set(requests)) or any(count > cap for count in moves.values()):
                    feasible = False
            if not feasible:
                continue
            critical = sum(a.kind == "SERVE" and a.critical for a in chosen_actions)
            carry = sum(a.kind == "SERVE" and a.carry_over for a in chosen_actions)
            current = len(first)
            expected = current + sum((rational[c.scenario_id] * len(c.request_ids) for c in chosen_chains), Fraction(0))
            distance = sum((Fraction(str(a.empty_distance_m)) for a in chosen_actions), Fraction(0))
            distance += sum((rational[c.scenario_id] * Fraction(str(c.empty_distance_m)) for c in chosen_chains), Fraction(0))
            score = ((critical, expected, current, carry, -distance) if policy == "CHAIN_DEFER" else
                     (critical, current, carry, expected, -distance))
            if best is None or score > best:
                best, best_ids = score, (chosen_actions, chosen_chains)
    return best, best_ids


def _check(actions, chains, weights, policy="CHAIN_DEFER", cap=50):
    solution = city.solve_city_master(actions, chains, weights, policy=policy, relocation_cap=cap)
    expected, _ = _oracle(actions, chains, weights, policy, cap)
    actual = (solution["critical_now"], solution["expected_served"], solution["current_served"],
              solution["carry_over_current_served"], -solution["expected_empty_distance_m"])
    if policy == "SERVICE_PRESERVING":
        actual = (actual[0], actual[2], actual[3], actual[1], actual[4])
    assert actual == pytest.approx([float(value) for value in expected])
    assert solution["runtime_s"] < 10
    assert solution["model"]["sparse"]
    assert not solution["model"]["global_or_citywide_optimality_bound_claimed"]
    return solution


def _shared_pending():
    actions = [_action("V1_WAIT", 1), _action("V1_SERVE7", 1, "SERVE", 7),
               _action("V2_WAIT", 2), _action("V2_SERVE8", 2, "SERVE", 8)]
    chains = []
    for sid in ("S1", "S2"):
        chains += [_chain(sid + "_W1", sid, 1, "V1_WAIT", (7, 9)),
                   _chain(sid + "_W2", sid, 2, "V2_WAIT", (7, 10)),
                   _chain(sid + "_A1", sid, 1, "V1_SERVE7", (9,)),
                   _chain(sid + "_A2", sid, 2, "V2_SERVE8", (7, 10))]
    return actions, chains, {"S1": .5, "S2": .5}


def test_multijob_continuation_vs_current_only_and_selectable_layout():
    actions = [_action("SERVE_P", 1, "SERVE", 100, carry=True), _action("WAIT", 1)]
    weights = {"S1": .6, "S2": .4}
    chains = [_chain("C_" + sid, sid, 1, "WAIT", (-1, -2, -3), 100) for sid in weights]
    defer = _check(actions, chains, weights)
    preserve = _check(actions, chains, weights, "SERVICE_PRESERVING")
    assert defer["selected_action_ids"] == ("WAIT",) and defer["expected_served"] == 3
    assert preserve["selected_action_ids"] == ("SERVE_P",) and preserve["current_served"] == 1
    layouts = [_action("LAYOUT_BAD", 1, "LAYOUT"), _action("LAYOUT_GOOD", 1, "LAYOUT")]
    layout_chains = [_chain("L_" + sid, sid, 1, "LAYOUT_GOOD", (-1, -2, -3)) for sid in weights]
    layout = _check(layouts, layout_chains, weights)
    assert layout["selected_action_ids"] == ("LAYOUT_GOOD",)
    assert layout["model"]["independent_single_vehicle_components"] == 1
    assert layout["model"]["optional_chain_variables"] == 2


def test_shared_real_pending_identity_is_global_against_first_serve_and_chains():
    actions, chains, weights = _shared_pending()
    for policy in city.POLICIES:
        solution = _check(actions, chains, weights, policy)
        chosen_actions = [a for a in actions if a.action_id in solution["selected_action_ids"]]
        chosen_chains = [c for c in chains if c.chain_id in solution["selected_chain_ids"]]
        first = [a.request_id for a in chosen_actions if a.kind == "SERVE"]
        for sid in weights:
            requests = first + [rid for c in chosen_chains if c.scenario_id == sid for rid in c.request_ids]
            assert len(requests) == len(set(requests))
        assert solution["model"]["coupled_milp_components"] == 1
        assert solution["expected_served"] == 3


def test_current_and_future_relocation_slots_keep_global_cap_coupling():
    actions = []
    chains = []
    weights = {"S1": .5, "S2": .5}
    for vid in (1, 2, 3):
        actions += [_action(f"W{vid}", vid), _action(f"R{vid}", vid, "RELOCATE", distance=1)]
        for sid in weights:
            chains += [_chain(f"{sid}R{vid}", sid, vid, f"R{vid}", (-vid * 10, -vid * 10 - 1), 1, (0, 900)),
                       _chain(f"{sid}W{vid}", sid, vid, f"W{vid}", (-vid * 100,), 100, (0,))]
    solution = _check(actions, chains, weights, cap=2)
    assert solution["expected_served"] == 2
    assert sum(a.kind == "RELOCATE" for a in actions if a.action_id in solution["selected_action_ids"]) == 1
    assert solution["model"]["coupled_milp_components"] == 1
    assert solution["model"]["global_relocation_coupling_preserved"]
    assert solution["model"]["future_relocation_slot_rows"] == 4
    assert all(count <= 2 for counts in solution["model"]["constrained_relocations_by_scenario_slot"].values() for count in counts.values())
    assert _check(actions, chains, weights, cap=1)["expected_served"] == 1
    assert _check(actions, chains, weights, cap=0)["expected_served"] == 0


def test_strict_numerical_and_sparse_size_limits_without_fallback(monkeypatch):
    actions, chains, weights = _shared_pending()
    with pytest.raises(city.CityMasterLimitError, match="variable limit"):
        city.solve_city_master(actions, chains, weights, max_variables=3)
    with pytest.raises(city.CityMasterLimitError, match="nonzero budget"):
        city.solve_city_master(actions, chains, weights, max_nonzeros=5)
    with pytest.raises(ValueError, match="repeats customer"):
        city.solve_city_master(actions, [_chain("DUP", "S1", 1, "V1_WAIT", (7, 7))], weights)
    real_milp = city.contract.milp

    def poisoned(*args, **kwargs):
        result = real_milp(*args, **kwargs)
        result.x = np.asarray(result.x).copy()
        result.x[0] += 1e-7
        return result

    monkeypatch.setattr(city.contract, "milp", poisoned)
    with pytest.raises(city.contract.JointNumericalContractError, match="raw numerical contract failed"):
        city.solve_city_master(actions, chains, weights)
