"""Opt-in lazy oracle separation on a fixed sparse current-action graph.

Default EAGER behavior remains available. The caller supplies conservative
action bounds and a batch validator; no route or native state is fabricated.
Only validated actions can be returned. Certificates concern this graph and
the unchanged lexicographic goals, not future/day service.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter

from .scheme_a_city_master import CurrentAction
from .scheme_a_current_flow import solve_current_flow


@dataclass(frozen=True)
class CheckedAction:
    action: CurrentAction
    continuation_units: int


def _score(result, actions):
    by_id = {action.action_id: action for action in actions}
    selected = [by_id[aid] for aid in result["selected_action_ids"]]
    # Same rounding as current_flow_secondary; never change physical distance.
    return (*map(int, result["score_units"]),
            -sum(int(round(action.empty_distance_m * 1000.)) for action in selected))


def select_with_validation(known, known_units, unknown_bounds, upper_units, validate_many,
                           *, mode="EAGER", policy="SERVICE_PRESERVING", time_limit_s=10.,
                           move_cap=50, maximum_actions=20000, backend="AUTO",
                           check=lambda: None):
    """An independent switch, not automatically enabled in the city adapter.

For an admissible unknown arc, its true units must be <= upper_units and its
true millimetre distance >= the bound's millimetre distance. Rejection must be
a definite result under the fixed research validator, never a budget expiry.
The callback returns {action_id: CheckedAction | None} for every queried ID.
"""
    if mode not in ("EAGER", "LAZY_CERTIFIED"):
        raise ValueError("unknown validation switch")
    if policy not in ("SERVICE_PRESERVING", "CHAIN_DEFER") or not 0 < time_limit_s <= 10:
        raise ValueError("unchanged current policy and at most ten seconds are required")
    started = perf_counter()
    deadline = started+time_limit_s
    admitted = {action.action_id: action for action in known}
    pending = {action.action_id: action for action in unknown_bounds}
    units = dict(known_units)
    if (len(admitted) != len(known) or len(pending) != len(unknown_bounds)
            or admitted.keys() & pending.keys() or set(units) != set(admitted)
            or set(upper_units) != set(pending)):
        raise ValueError("duplicate/missing action bounds")
    vehicles = {action.vehicle_id for action in known if action.kind == "WAIT"}
    if any(action.kind != "SERVE" or action.vehicle_id not in vehicles for action in unknown_bounds):
        raise ValueError("unknowns must be current service arcs with known WAITs")
    if any(type(value) is not int or not -9 <= value <= 9 for value in (*units.values(), *upper_units.values())):
        raise ValueError("current continuation bounds must retain integer range")
    calls, queried, rejected = 0, [], []
    solve_time, oracle_time = 0., 0.
    trace = []

    def solve(actions, values):
        nonlocal solve_time
        check()
        remaining = deadline-perf_counter()
        if remaining <= 0:
            return None
        stamp = perf_counter()
        result = solve_current_flow(actions, values, policy=policy, move_cap=move_cap,
            time_limit_s=min(10., remaining), deadline_s=deadline, maximum_actions=maximum_actions,
            backend=backend, empty_distance_mode="PRIMARY_FACE_MM")
        solve_time += perf_counter()-stamp
        return result

    def query(ids):
        nonlocal calls, oracle_time
        if not ids:
            return
        stamp = perf_counter()
        answer = validate_many(tuple(ids))
        oracle_time += perf_counter()-stamp
        calls += 1
        if set(answer) != set(ids):
            raise ValueError("validator omitted or introduced candidate identities")
        for aid in ids:
            bound, value = pending.pop(aid), answer[aid]
            queried.append(aid)
            if value is None:
                rejected.append(aid)
                continue
            if not isinstance(value, CheckedAction):
                raise TypeError("unresolved oracle answers cannot become admissibility results")
            actual = value.action
            if (actual.action_id != aid or actual.vehicle_id != bound.vehicle_id
                    or actual.request_id != bound.request_id or actual.kind != "SERVE"
                    or actual.critical != bound.critical or actual.carry_over != bound.carry_over):
                raise ValueError("validator changed the graph/policy identity")
            if (type(value.continuation_units) is not int
                    or not -9 <= value.continuation_units <= upper_units[aid]
                    or not isfinite(actual.empty_distance_m) or actual.empty_distance_m < 0
                    or int(round(actual.empty_distance_m*1000.)) <
                       int(round(bound.empty_distance_m*1000.))):
                raise ValueError("optimistic candidate bound violated; certificate refused")
            admitted[aid], units[aid] = actual, value.continuation_units

    # A cheap validated incumbent is always selected before expensive queries.
    best = solve(list(admitted.values()), units)
    if best is None:
        raise TimeoutError("no time to obtain a validated incumbent")
    best_score = _score(best, admitted.values())
    upper_score, certified = None, False
    if mode == "EAGER" and perf_counter() < deadline:
        query(sorted(pending))
        final = solve(list(admitted.values()), units)
        if final is not None and _score(final, admitted.values()) >= best_score:
            best, best_score = final, _score(final, admitted.values())
        certified = (final is not None and not pending
                     and final.get("primary_optimality_proved", False)
                     and final.get("secondary_optimality_proved", False))
        upper_score = best_score if certified else None
    elif mode == "LAZY_CERTIFIED":
        while perf_counter() < deadline:
            optimistic = [*admitted.values(), *pending.values()]
            optimistic_units = {**units, **{aid: upper_units[aid] for aid in pending}}
            upper = solve(optimistic, optimistic_units)
            if upper is None:
                break
            if not (upper.get("primary_optimality_proved", False)
                    and upper.get("secondary_optimality_proved", False)):
                # An unproved heuristic result is not an upper certificate.
                break
            upper_score = _score(upper, optimistic)
            if upper_score < best_score:
                raise RuntimeError("optimistic optimum is below its validated incumbent")
            ids = sorted(set(upper["selected_action_ids"]) & pending.keys())
            trace.append(dict(unknown_remaining=len(pending), selected_unknown=len(ids),
                              lower=list(best_score), upper=list(upper_score)))
            if upper_score == best_score:
                certified = True
                break
            if not ids:
                best, best_score, certified = upper, upper_score, True
                break
            query(ids)
            candidate = solve(list(admitted.values()), units)
            if candidate is not None and _score(candidate, admitted.values()) >= best_score:
                best, best_score = candidate, _score(candidate, admitted.values())
    check()
    if set(best["selected_action_ids"]) & pending.keys():
        raise RuntimeError("an unvalidated action was selected for execution")
    elapsed = perf_counter()-started
    return dict(selected=[admitted[aid] for aid in best["selected_action_ids"]],
        score=best_score, selected_units={aid: units[aid] for aid in best["selected_action_ids"]},
        mode=mode, full_current_objective_certified=certified and elapsed <= time_limit_s,
        mathematical_score_bracket_closed=certified, upper_score=upper_score,
        queried_action_ids=tuple(queried), rejected_action_ids=tuple(rejected),
        query_batches=calls, oracle_calls=len(queried), unqueried_actions=len(pending),
        solve_time_s=solve_time, oracle_time_s=oracle_time, total_time_s=elapsed,
        wall_overrun=elapsed>time_limit_s, trace=trace,
        all_selected_actions_validated=True, whole_day_guarantee=False)
