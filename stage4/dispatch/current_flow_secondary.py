"""Distance minimization on an exact primary-optimal sparse flow face.

Complementary slackness fixes positive reduced-cost arcs at zero and negative
ones at capacity. Only zero reduced-cost arcs vary, avoiding huge combined
primary/distance weights. Integer distance costs have millimetre precision;
physical distances remain unmodified, with a reported rounding-gap bound.
"""
from collections import deque
from heapq import heappop, heappush
from time import perf_counter


def _potentials(graph, deadline):
    price = [0] * len(graph)
    queue = deque(range(len(graph)))
    queued = [True] * len(graph)
    pops = 0
    while queue:
        node = queue.popleft()
        queued[node] = False
        pops += 1
        if pops % 32 == 0 and perf_counter() >= deadline:
            raise TimeoutError("SECONDARY_PRIMARY_FACE_TIME_LIMIT")
        for edge in graph[node]:
            candidate = price[node] + edge.cost
            if edge.capacity and candidate < price[edge.target]:
                price[edge.target] = candidate
                if not queued[edge.target]:
                    queue.append(edge.target)
                    queued[edge.target] = True
    if any(e.capacity and e.cost + price[u] - price[e.target] < 0
           for u, edges in enumerate(graph) for e in edges):
        raise RuntimeError("primary optimum has no valid residual price certificate")
    return price


def _variable_flow(arcs, supplies, deadline, backend):
    from .scheme_a_current_flow import _compiled_flow_class, _add
    count, amount = len(supplies), sum(max(0, value) for value in supplies)
    if not amount:
        return [0] * len(arcs), "TRIVIAL_PRIMARY_FACE"
    if perf_counter() >= deadline:
        raise TimeoutError("SECONDARY_DISTANCE_TIME_LIMIT")
    if backend == "AUTO":
        factory, _ = _compiled_flow_class()
        safe = max((abs(a[3]) for a in arcs), default=0) * max(1, count + 1, amount) < ((1 << 63) - 1) // 16
        if factory is not None and safe:
            import numpy as np
            solver = factory()
            ids = solver.add_arcs_with_capacity_and_unit_cost(
                np.asarray([a[0] for a in arcs], dtype=np.int32),
                np.asarray([a[1] for a in arcs], dtype=np.int32),
                np.asarray([a[2] for a in arcs], dtype=np.int64),
                np.asarray([a[3] for a in arcs], dtype=np.int64))
            solver.set_nodes_supplies(np.arange(count, dtype=np.int32), np.asarray(supplies, dtype=np.int64))
            status = solver.solve()
            if status == solver.OPTIMAL:
                return list(map(int, solver.flows(ids))), "ORTOOLS_ZERO_REDUCED_COST_FACE"
            if status != solver.BAD_COST_RANGE:
                raise RuntimeError("secondary primary-optimal face lost its feasible flow")
    source, sink = count, count + 1
    graph = [[] for _ in range(count + 2)]
    refs = [_add(graph, u, v, capacity, cost) for u, v, capacity, cost in arcs]
    for node, value in enumerate(supplies):
        if value > 0:
            _add(graph, source, node, value, 0)
        elif value < 0:
            _add(graph, node, sink, -value, 0)
    potential, sent = [0] * len(graph), 0
    while sent < amount:
        if perf_counter() >= deadline:
            raise TimeoutError("SECONDARY_DISTANCE_TIME_LIMIT")
        distance, parent = [None] * len(graph), [None] * len(graph)
        distance[source] = 0
        queue, pops = [(0, source)], 0
        while queue:
            value, node = heappop(queue)
            if distance[node] != value:
                continue
            pops += 1
            if pops % 64 == 0 and perf_counter() >= deadline:
                raise TimeoutError("SECONDARY_DISTANCE_TIME_LIMIT")
            for index, edge in enumerate(graph[node]):
                if not edge.capacity:
                    continue
                reduced = edge.cost + potential[node] - potential[edge.target]
                if reduced < 0:
                    raise RuntimeError("secondary reduced-cost invariant lost")
                candidate = value + reduced
                if distance[edge.target] is None or candidate < distance[edge.target]:
                    distance[edge.target], parent[edge.target] = candidate, (node, index)
                    heappush(queue, (candidate, edge.target))
        if distance[sink] is None:
            raise RuntimeError("secondary primary face is infeasible")
        maximum = max(v for v in distance if v is not None)
        for node, value in enumerate(distance):
            potential[node] += maximum if value is None else value
        step, node = amount - sent, sink
        while node != source:
            u, index = parent[node]
            step = min(step, graph[u][index].capacity)
            node = u
        node = sink
        while node != source:
            u, index = parent[node]
            edge = graph[u][index]
            edge.capacity -= step
            graph[node][edge.reverse].capacity += step
            node = u
        sent += step
    return [capacity - graph[u][index].capacity for (u, index), (_, _, capacity, _) in zip(refs, arcs)], "PYTHON_ZERO_REDUCED_COST_FACE"


def minimize_empty_face(original, references, selected, vehicle_nodes, request_nodes,
                        source, sink, pool, node_count, deadline, backend):
    from .scheme_a_current_flow import _add
    started = perf_counter()
    old_ids = tuple(sorted(a.action_id for a in selected))
    info = dict(secondary_status="PRIMARY_RETAINED", secondary_backend=None,
        secondary_optimality_proved=False, secondary_scope="ADMITTED_PRIMARY_OPTIMAL_FACE_MM",
        secondary_cost_resolution_m=.001, secondary_distance_gap_bound_m=None,
        secondary_variable_arcs=0, secondary_time_s=0.)
    if perf_counter() >= deadline:
        return old_ids, dict(info, secondary_status="SECONDARY_BUDGET_EXHAUSTED")
    by_ref = {ref: aid for aid, ref in references.items()}
    by_id = {a.action_id: a for a in selected}
    selected_requests = {a.request_id for a in selected if a.kind == "SERVE"}
    moves = sum(a.kind == "RELOCATE" for a in selected)
    request_by_node = {node: rid for rid, node in request_nodes.items()}
    residual = [[] for _ in range(node_count)]
    costs_mm, initial = {}, []
    for u, index, v, capacity, primary_cost, distance_m in original:
        key = (u, index)
        if key in by_ref:
            flow = int(by_ref[key] in by_id)
        elif u == source:
            flow = 1
        elif u == pool:
            flow = moves
        elif u in request_by_node and v == sink:
            flow = int(request_by_node[u] in selected_requests)
        else:
            raise RuntimeError("unknown current-flow topology in secondary face")
        if not 0 <= flow <= capacity:
            raise RuntimeError("primary selection is not a feasible complete flow")
        ru, ri = _add(residual, u, v, capacity - flow, primary_cost)
        residual[v][residual[ru][ri].reverse].capacity = flow
        costs_mm[key] = int(round(distance_m * 1000.))
        initial.append(flow)
    try:
        price = _potentials(residual, deadline)
        supplies = [0] * node_count
        supplies[source], supplies[sink] = len(selected), -len(selected)
        variable, variable_keys, fixed = [], [], set()
        for arc, flow in zip(original, initial):
            u, index, v, capacity, cost, _ = arc
            reduced, key = cost + price[u] - price[v], (u, index)
            if reduced == 0:
                variable.append((u, v, capacity, costs_mm[key]))
                variable_keys.append(key)
            else:
                expected = capacity if reduced < 0 else 0
                if flow != expected:
                    raise RuntimeError("primary flow violates complementary slackness")
                supplies[u] -= flow
                supplies[v] += flow
                if flow and key in by_ref:
                    fixed.add(by_ref[key])
        flows, solver = _variable_flow(variable, supplies, deadline, backend)
        chosen = fixed | {by_ref[key] for key, value in zip(variable_keys, flows) if value and key in by_ref}
        if len(chosen) != len(selected):
            raise RuntimeError("secondary face did not select one action per resource")
        old_cost = sum(costs_mm[references[aid]] for aid in old_ids)
        new_cost = sum(costs_mm[references[aid]] for aid in chosen)
        if new_cost > old_cost:
            raise RuntimeError("secondary optimum is worse than its feasible primary input")
        if new_cost == old_cost:
            chosen = set(old_ids)
        return tuple(sorted(chosen)), dict(info, secondary_status="OPTIMAL_PRIMARY_FACE_MM",
            secondary_backend=solver, secondary_optimality_proved=True,
            secondary_distance_gap_bound_m=len(selected) / 1000.,
            secondary_empty_cost_mm=new_cost, secondary_variable_arcs=len(variable),
            secondary_time_s=perf_counter() - started)
    except TimeoutError as error:
        return old_ids, dict(info, secondary_status=str(error), secondary_time_s=perf_counter() - started)
