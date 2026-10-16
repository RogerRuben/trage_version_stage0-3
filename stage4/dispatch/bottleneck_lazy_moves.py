"""Select, certify, then replan optional moves within ONE native deadline.

Relaxed moves are planning hints only. They can never reach FleetPy, nor can
their optimistic continuation paths become the retained execution state.
No independent-vehicle value filter is used, so neutral joint moves survive.
"""
from collections import Counter
from time import perf_counter


def _combine(result, attempts, counts):
    info = dict(result.get("joint_control", {}))
    for name in ("planning_labels", "accepted_coalitions", "future_only_accepted_coalitions",
                 "neighborhoods_evaluated", "generated_continuation_columns", "actual_query_calls",
                 "batch_root_queries", "batch_query_reuses", "joint_wall_time_s",
                 "warm_exact_reused_continuations", "warm_retime_calls"):
        info[name] = sum(a.get("joint_control", {}).get(name, 0) for a in attempts)
    for name in ("phase_timings_s", "search_work", "topology_work"):
        total = Counter()
        for attempt in attempts:
            total.update(attempt.get("joint_control", {}).get(name, {}))
        info[name] = dict(total)
    for name in ("max_joint_variables", "max_joint_nonzeros", "peak_process_rss_mib"):
        info[name] = max((a.get("joint_control", {}).get(name, 0) for a in attempts), default=0)
    info["lazy_moves"] = dict(counts)
    info["lazy_moves"]["relaxed_moves_executed"] = 0
    info["lazy_moves"]["budget_scope"] = "ONE_DEADLINE_SHARED_BY_PROBE_ROUTING_AND_REFINEMENT"
    info["probe_uses_uncertified_move_hints"] = True
    info["executed_first_actions_all_certified"] = True
    info["probe_optimality_is_not_physical_route_certificate"] = True
    return dict(result, joint_control=info)


def solve_lazy_moves(actions, roots, baseline, *, planner, certify, deadline, memory,
                     maximum_labels=50000):
    """Route only selected moves, separating definite failures before replan.

    planner(actions, roots, baseline, deadline, memory, label_cap) must keep the
    supplied feasible baseline on timeout. certify returns an exact action/root
    pair or None. Both callbacks retain the caller's physical/profile limits.
    A failed/late cell never deletes other certified candidates. A refinement
    timeout keeps the executable no-move baseline, not an optimistic plan.
    """
    base = [a for a in actions if a.kind != "RELOCATE"]
    proposals = {a.action_id: a for a in actions if a.kind == "RELOCATE"}
    if any(a in proposals for a in baseline["selected_action_ids"]):
        raise ValueError("the lazy incumbent must not contain an unvalidated MOVE")
    counts = Counter(optimistic_proposals=len(proposals), probe_selected=0, route_calls=0,
                     certified_moves=0, unsupported_moves=0, budget_skipped_selected=0,
                     route_cap_skipped_selected=0, planning_rounds=0, refinement_fallback=0,
                     rejection_replans=0)
    started = perf_counter()
    remaining = max(0., deadline-started)
    if not proposals or not remaining:
        return baseline, base, {a.action_id: roots[a.action_id] for a in base}, counts
    # Explicit finite-budget algorithm, not an extension of the ten-second cap.
    probe_deadline = started + .4*remaining
    certification_deadline = started + .7*remaining
    working_memory = memory.fork()
    active = {a.action_id:a for a in actions}; active_roots = dict(roots)
    certified, attempts, used = {}, [], 0
    while proposals and perf_counter()<probe_deadline and counts["route_calls"]<50:
        probe_memory = working_memory.fork()
        probe = planner(list(active.values()), active_roots, baseline, probe_deadline,
                        probe_memory, max(0,maximum_labels-used))
        attempts.append(probe); counts["planning_rounds"] += 1
        used += probe.get("joint_control", {}).get("planning_labels",0)
        selected = sorted(set(probe["selected_action_ids"]) & proposals.keys())
        counts["probe_selected"] += len(selected)
        if not selected:
            if not probe.get("joint_control", {}).get("resource_fallback", True):
                memory.adopt(probe_memory)
            exact_actions = [*base,*certified.values()]
            exact_roots = {a.action_id:active_roots[a.action_id] for a in exact_actions}
            return _combine(probe, attempts, counts), exact_actions, exact_roots, counts
        working_memory = probe_memory
        rejected = False
        for index, aid in enumerate(selected):
            if counts["route_calls"]>=50:
                counts["route_cap_skipped_selected"] += len(selected)-index
                break
            if perf_counter() >= certification_deadline:
                counts["budget_skipped_selected"] += len(selected)-index
                break
            counts["route_calls"] += 1
            exact = certify(proposals.pop(aid))
            if exact is None:
                counts["unsupported_moves"] += 1; rejected = True
                del active[aid]; del active_roots[aid]
                continue
            action, root = exact
            if (action.action_id != aid or action.kind != "RELOCATE"
                    or not isinstance(action.payload, dict)
                    or action.payload.get("optimistic_move")
                    or action.payload.get("move", {}).get("routed") is None):
                raise ValueError("lazy certification did not provide an executable current MOVE")
            certified[aid] = action; active[aid] = action; active_roots[aid] = root
        counts["certified_moves"] = len(certified)
        if not rejected:
            break
        # A rejected cell is not evidence against the other sites/vehicles.
        # Re-price the remaining hints, within the SAME probe/label budget.
        counts["rejection_replans"] += 1
    exact_actions = [*base, *certified.values()]
    exact_roots = {a.action_id:active_roots[a.action_id] for a in exact_actions}
    if not certified or perf_counter() >= deadline:
        counts["refinement_fallback"] = 1
        result = dict(baseline, joint_control=dict(status="LAZY_MOVE_KEEP_PHYSICAL_BASELINE",
            resource_fallback=True, reason="NO_CERTIFIED_MOVE_OR_SHARED_DEADLINE_EXHAUSTED"))
        return _combine(result, attempts, counts), exact_actions, exact_roots, counts
    # Retiming/replanning is mandatory: no zero-time probe path is trusted.
    final_memory = working_memory.fork()
    final = planner(exact_actions, exact_roots, baseline, deadline, final_memory,
                    max(0, maximum_labels-used))
    attempts.append(final); counts["planning_rounds"] += 1
    allowed = {a.action_id for a in exact_actions}
    if not set(final["selected_action_ids"]) <= allowed:
        raise ValueError("relaxed MOVE escaped the exact refinement domain")
    if not final.get("joint_control", {}).get("resource_fallback", True):
        memory.adopt(final_memory)
    else:
        counts["refinement_fallback"] = 1
    return _combine(final, attempts, counts), exact_actions, exact_roots, counts
