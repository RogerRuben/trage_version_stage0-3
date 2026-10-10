"""Light two-scale kernel: exact CURRENT actions, approximate future values.

Only already-admitted SERVE / WAIT / RELOCATE actions enter this module. It
does not route, predict, enumerate future customers, or override capability.
The caller supplies a coarse continuation VALUE DIFFERENCE relative to WAIT,
in thirds of a service opportunity, not a promised count of future services.

Integer lexicographic costs and sparse min-cost flow replace repeated MILPs.
A reference-path deadline returns a feasible, explicitly nonoptimal decision,
not an error that discards a day. The native CPU solver has no interrupt API:
late returns are labelled WALL_OVERRUN, never hidden as in-budget execution.
Schema/feasibility errors still fail visibly.
This is a new algorithm, NOT an equivalent implementation of the city chains.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush
import importlib
from math import isfinite
from pathlib import Path
import sys
from time import perf_counter

from .scheme_a_city_master import CurrentAction


@dataclass
class _Edge:
    target: int
    reverse: int
    capacity: int
    cost: int


def _add(graph, source, target, capacity, cost):
    index = len(graph[source])
    reverse = len(graph[target])
    graph[source].append(_Edge(target, reverse, capacity, cost))
    graph[target].append(_Edge(source, index, 0, -cost))
    return source, index


def _features(action, units, policy):
    served = int(action.kind == "SERVE")
    critical, carry = int(action.critical), int(action.carry_over)
    anticipated = 3 * served + units
    return ((critical, served, carry, anticipated) if policy == "SERVICE_PRESERVING"
            else (critical, anticipated, served, carry))


def _totals(selected, features):
    return tuple(sum(features[action.action_id][i] for action in selected) for i in range(4))


@lru_cache(maxsize=1)
def _compiled_flow_class():
    """Optional pinned CPU runtime, without modifying the scientific env."""
    directory = Path(__file__).resolve().parents[1]/"runtime"/"ortools_current_flow"
    added = False
    try:
        if directory.is_dir():
            sys.path.insert(0,str(directory))
            added = True
        package = importlib.import_module("ortools")
        if package.__version__ != "9.15.6755":
            return None,"ORTOOLS_VERSION_NOT_PINNED"
        module = importlib.import_module("ortools.graph.python.min_cost_flow")
        return module.SimpleMinCostFlow,None
    except ImportError as error:
        return None,f"ORTOOLS_UNAVAILABLE:{type(error).__name__}"
    finally:
        if added:
            sys.path.remove(str(directory))


def _compiled_solve(graph, references, source, sink, resource_count):
    factory,reason = _compiled_flow_class()
    if factory is None:
        return False,reason
    forward = [(node,index,edge) for node,edges in enumerate(graph)
               for index,edge in enumerate(edges) if edge.capacity]
    maximum = max((abs(edge.cost) for _,_,edge in forward),default=0)
    # Account for native cost scaling and total objective accumulation, not
    # just casting individual costs to int64. Never approximate an oversized
    # lexicographic cost: the arbitrary-integer reference remains available.
    if maximum * max(1,len(graph)+1,resource_count) >= ((1<<63)-1)//16:
        return False,"EXACT_INTEGER_COST_EXCEEDS_NATIVE_SAFE_RANGE"
    import numpy as np
    engine = factory()
    arc_ids = engine.add_arcs_with_capacity_and_unit_cost(
        np.asarray([node for node,_,_ in forward],dtype=np.int32),
        np.asarray([edge.target for _,_,edge in forward],dtype=np.int32),
        np.asarray([edge.capacity for _,_,edge in forward],dtype=np.int64),
        np.asarray([edge.cost for _,_,edge in forward],dtype=np.int64))
    supply = np.zeros(len(graph),dtype=np.int64)
    supply[source],supply[sink] = resource_count,-resource_count
    engine.set_nodes_supplies(np.arange(len(graph),dtype=np.int32),supply)
    status = engine.solve()
    if status == engine.BAD_COST_RANGE:
        return False,"NATIVE_BAD_COST_RANGE_USE_EXACT_REFERENCE"
    if status != engine.OPTIMAL:
        raise RuntimeError(f"current-flow native solver rejected the feasible WAIT network: {status}")
    for (node,index,edge),flow in zip(forward,engine.flows(arc_ids)):
        if (node,index) in references and int(flow):
            edge.capacity -= int(flow)
    return True,None


def _backup(actions, waits):
    """Deterministic admitted-service backup; no maximum-count claim."""
    chosen, requests = {}, set()
    for action in sorted((a for a in actions if a.kind == "SERVE"), key=lambda a:
            (not a.critical, not a.carry_over, a.empty_distance_m,
             a.vehicle_id, a.request_id, a.action_id)):
        if action.vehicle_id not in chosen and action.request_id not in requests:
            chosen[action.vehicle_id] = action
            requests.add(action.request_id)
    return [chosen.get(vid, waits[vid]) for vid in sorted(waits)]


def _check_selection(selected, vehicles, move_cap):
    if {a.vehicle_id for a in selected} != set(vehicles) or len(selected) != len(vehicles):
        raise RuntimeError("current flow lost or duplicated a resource")
    requests = [a.request_id for a in selected if a.kind == "SERVE"]
    if len(requests) != len(set(requests)) or sum(a.kind == "RELOCATE" for a in selected) > move_cap:
        raise RuntimeError("current flow violated request uniqueness or shared MOVE capacity")


def solve_current_flow(actions, continuation_units, *, policy="SERVICE_PRESERVING",
                       move_cap=50, time_limit_s=10., deadline_s=None,
                       maximum_actions=20_000, backend="AUTO"):
    """Select one feasible current action per available vehicle.

Input actions must ALREADY satisfy road/direction, profile, acceptance,
deadline, availability and commitment checks. Every available vehicle needs
one WAIT. BUSY / unborn resources and negative synthetic forecast IDs are not
current actions. Current MOVE choices share one present-time capacity pool.

Continuation deltas are bounded integer units [-9,9]: three equal historical
scenarios, up to three service opportunities each. Future competition is only
approximated by the value provider; never sum these as realized capacity.
    Global secondary empty-distance optimality is deliberately removed. Equal
    primary-value actions prefer shorter local distance in stable input order;
    this is NOT a minimum-total-distance guarantee. Physical distances, paths
    and time windows are never rounded or modified.
"""
    started = perf_counter()
    if policy not in ("SERVICE_PRESERVING", "CHAIN_DEFER"):
        raise ValueError("unknown current-flow policy")
    if type(move_cap) is not int or not 0 <= move_cap <= 50:
        raise ValueError("current MOVE pool must retain the original cap of at most 50")
    if not isfinite(time_limit_s) or not 0 < time_limit_s <= 10:
        raise ValueError("current-flow time limit must be in (0,10]")
    if backend not in ("AUTO","INTEGER_REFERENCE"):
        raise ValueError("current-flow backend must be AUTO or INTEGER_REFERENCE")
    if type(maximum_actions) is not int or maximum_actions <= 0:
        raise ValueError("maximum_actions must be positive")
    deadline = started + time_limit_s
    if deadline_s is not None:
        if not isfinite(deadline_s):
            raise ValueError("deadline must be a finite monotonic timestamp")
        deadline = min(deadline, float(deadline_s))
    actions = sorted(actions, key=lambda a: (a.vehicle_id, a.action_id))
    waits, groups, features, seen, slots = {}, defaultdict(list), {}, set(), set()
    for action in actions:
        if not isinstance(action, CurrentAction) or action.kind not in ("SERVE", "WAIT", "RELOCATE"):
            raise ValueError("only admitted current actions enter current flow")
        if (type(action.vehicle_id) is not int or not action.action_id or action.action_id in seen
                or type(action.critical) is not bool or type(action.carry_over) is not bool):
            raise ValueError("invalid or duplicated current action")
        seen.add(action.action_id)
        if action.kind == "SERVE":
            if type(action.request_id) is not int or action.request_id < 0:
                raise ValueError("synthetic future requests cannot enter physical current flow")
        elif action.request_id is not None or action.critical or action.carry_over:
            raise ValueError("non-service action cannot claim a current customer")
        if not isfinite(action.empty_distance_m) or action.empty_distance_m < 0:
            raise ValueError("invalid physical empty distance")
        units = continuation_units.get(action.action_id)
        if type(units) is not int or not -9 <= units <= 9:
            raise ValueError("every action requires a bounded integer continuation difference")
        if action.kind == "WAIT":
            if action.vehicle_id in waits or units != 0 or action.empty_distance_m != 0:
                raise ValueError("one zero-baseline WAIT is required per available resource")
            waits[action.vehicle_id] = action
        elif action.kind == "RELOCATE":
            slot = action.payload.get("relocation_slot_s") if isinstance(action.payload, dict) else None
            if type(slot) is not int or slot < 0 or slot % 900:
                raise ValueError("current MOVE retains its original 900-second slot")
            slots.add(slot)
        groups[action.vehicle_id].append(action)
        features[action.action_id] = _features(action, units, policy)
    if set(groups) != set(waits) or len(slots) > 1:
        raise ValueError("current-only flow requires WAITs and one current relocation slot")
    backup = _backup(actions, waits)

    def result(selected, status, reason=None, nodes=0, edges=0, augmentations=0,
               native=False, native_reason=None):
        _check_selection(selected, groups, move_cap)
        return dict(selected_action_ids=tuple(sorted(a.action_id for a in selected)),
            critical_now=sum(a.critical for a in selected),
            current_served=sum(a.kind == "SERVE" for a in selected),
            carry_over_current_served=sum(a.carry_over for a in selected),
            current_moves=sum(a.kind == "RELOCATE" for a in selected),
            score_units=_totals(selected, features), runtime_s=perf_counter()-started,
            opt_status=status, fallback_reason=reason,
            backend="ORTOOLS_CPU_INTEGER_CURRENT_FLOW" if native else "PYTHON_INTEGER_CURRENT_FLOW", nodes=nodes,
            residual_edges=edges, augmentations=augmentations,
            future_chain_variables=0, routing_queries=0, dense_matrix=False,
            current_empty_distance_m=sum(a.empty_distance_m for a in selected),
            distance_global_optimality_claimed=False,
            native_backend_fallback_reason=native_reason,
            wall_deadline_exceeded=perf_counter() > deadline,
            primary_optimality_proved=native or status.startswith("OPTIMAL_PRIMARY"),
            native_interrupt_api_available=False,
            future_global_optimality_claimed=False, whole_day_optimality_claimed=False)

    if len(actions) > maximum_actions or perf_counter() >= deadline:
        return result(backup, "FEASIBLE_BACKUP_SERVICE_FIRST",
                      "CURRENT_ACTION_LIMIT" if len(actions) > maximum_actions else "CURRENT_WALL_DEADLINE")
    # Domination multipliers are exact Python integers. Each earlier unit
    # outweighs the TOTAL possible range of all later priorities. This is
    # lexicographic encoding, not an arbitrary weighted risk/utility average.
    ranges = [sum(max(features[a.action_id][i] for a in group)
                  - min(features[a.action_id][i] for a in group)
                  for group in groups.values()) for i in range(4)]
    multipliers = [0] * 4
    for i in range(3, -1, -1):
        multipliers[i] = 1 + sum(multipliers[j] * ranges[j] for j in range(i+1, 4))
    encoded = {aid:sum(c*x for c,x in zip(multipliers, value)) for aid,value in features.items()}
    vehicles = sorted(groups)
    request_ids = sorted({a.request_id for a in actions if a.kind == "SERVE"})
    source, sink, pool = 0, 1, 2
    vehicle_nodes = {vid:3+i for i,vid in enumerate(vehicles)}
    request_nodes = {rid:3+len(vehicles)+i for i,rid in enumerate(request_ids)}
    graph = [[] for _ in range(3+len(vehicles)+len(request_ids))]
    references = {}
    for vid in vehicles:
        _add(graph, source, vehicle_nodes[vid], 1, 0)
    for rid in request_ids:
        _add(graph, request_nodes[rid], sink, 1, 0)
    _add(graph, pool, sink, move_cap, 0)
    potential = [0] * len(graph)
    incoming = defaultdict(list)
    for vid in vehicles:
        # All MOVE choices use one identical global capacity resource and no
        # site capacity. Only the best admitted current MOVE for this vehicle
        # is needed; this elimination is exact for the CURRENT model.
        moves = [a for a in groups[vid] if a.kind == "RELOCATE"]
        moves = [a for a in moves if encoded[a.action_id] > 0]  # WAIT dominates the rest.
        best_move = min(moves,key=lambda a:(-encoded[a.action_id],a.empty_distance_m,a.action_id)) if moves else None
        for action in sorted(groups[vid],key=lambda a:(-encoded[a.action_id],a.empty_distance_m,a.action_id)):
            if action.kind == "RELOCATE" and action is not best_move:
                continue
            target = (request_nodes[action.request_id] if action.kind == "SERVE"
                      else pool if action.kind == "RELOCATE" else sink)
            cost = -encoded[action.action_id]
            references[action.action_id] = _add(graph, vehicle_nodes[vid], target, 1, cost)
            incoming[target].append(cost)
    by_id = {a.action_id:a for a in actions}

    def chosen_from_flow():
        selected = dict(waits)
        for aid,(node,index) in references.items():
            if graph[node][index].capacity == 0:
                action = by_id[aid]
                selected[action.vehicle_id] = action
        return [selected[vid] for vid in vehicles]

    if perf_counter() >= deadline:
        return result(backup,"FEASIBLE_BACKUP_SERVICE_FIRST","CURRENT_WALL_DEADLINE")
    native_reason = None
    if backend == "AUTO":
        used,native_reason = _compiled_solve(graph,set(references.values()),source,sink,len(vehicles))
        if used:
            late = perf_counter() >= deadline
            return result(chosen_from_flow(),
                "FEASIBLE_NATIVE_WALL_OVERRUN" if late else "OPTIMAL_PRIMARY_CURRENT_FLOW_APPROXIMATE_FUTURE",
                reason="NATIVE_NONINTERRUPTIBLE_CALL_OVERRAN_DEADLINE" if late else None,
                nodes=len(graph),edges=sum(map(len,graph)),augmentations=None,native=True)
    # The initial residual graph is a DAG S->vehicle->request/MOVE/T->T.
    # Its shortest distances give feasible reduced costs without Bellman-Ford.
    for node in (*request_nodes.values(), pool):
        potential[node] = min(incoming[node], default=0)
    potential[sink] = min(0, *(potential[n] for n in (*request_nodes.values(),pool)))
    augmentations, timed_out = 0, False
    while True:
        if perf_counter() >= deadline:
            timed_out = True
            break
        distance = [None] * len(graph)
        parent = [None] * len(graph)
        distance[source] = 0
        queue, popped = [(0,source)], 0
        while queue:
            value,node = heappop(queue)
            if value != distance[node]:
                continue
            popped += 1
            if popped % 64 == 0 and perf_counter() >= deadline:
                timed_out = True
                break
            for index,edge in enumerate(graph[node]):
                if not edge.capacity:
                    continue
                reduced = edge.cost + potential[node] - potential[edge.target]
                if reduced < 0:
                    raise RuntimeError("current flow lost its integer reduced-cost invariant")
                candidate = value + reduced
                old = distance[edge.target]
                if old is None or candidate < old:
                    distance[edge.target] = candidate
                    parent[edge.target] = node,index
                    heappush(queue,(candidate,edge.target))
        if timed_out:
            break  # No partial path is applied: the last flow is still feasible.
        if distance[sink] is None or distance[sink] + potential[sink] - potential[source] >= 0:
            break  # Remaining zero-valued WAITs complete the optimal decision.
        for node,value in enumerate(distance):
            if value is not None:
                potential[node] += value
        node = sink
        while node != source:
            previous,index = parent[node]
            edge = graph[previous][index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = previous
        augmentations += 1
    chosen = chosen_from_flow()
    if timed_out and (_totals(backup,features),-sum(a.empty_distance_m for a in backup)) > (
            _totals(chosen,features),-sum(a.empty_distance_m for a in chosen)):
        chosen = backup
    return result(chosen, "FEASIBLE_TIME_LIMIT" if timed_out else "OPTIMAL_PRIMARY_CURRENT_FLOW_APPROXIMATE_FUTURE",
                  "CURRENT_WALL_DEADLINE" if timed_out else None, len(graph),
                  sum(map(len,graph)), augmentations,native_reason=native_reason)
