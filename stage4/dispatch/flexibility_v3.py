"""Exact decision-relevant decomposition of the ONE-next-service model.

No regional partition or approximate candidate removal. Components come from
all current choices and all scenario request capacities. Fixed-idle vehicles
do not couple scenarios. Pure-future components are sparse matchings, not MIPs.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from math import isfinite
import time

import numpy as np
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.csgraph import maximum_bipartite_matching

from . import flexibility_model as base
from .solver import AssignmentArc, solve_lexicographic


@dataclass(frozen=True)
class DecisionV3(base.Decision):
    component_count: int = 0
    mip_component_count: int = 0
    pure_future_component_count: int = 0
    pure_future_edges: int = 0
    pure_future_expected_count: float = 0.0


class CSRRows:
    """Compile COO once; append only the few lexicographic lock rows."""
    def __init__(self, limits, count):
        self.limits, self.count = limits, count
        self.lower, self.upper = [], []
        self.row, self.col, self.data = [], [], []
        self.cached = None
        self.nnz = 0

    def add_sparse(self, columns, values=None, lower=-np.inf, upper=np.inf):
        columns = list(columns)
        values = [1.] * len(columns) if values is None else list(values)
        if not columns:
            if lower > 1e-7 or upper < -1e-7:
                raise RuntimeError("infeasible constant row in direct CSR compiler")
            return
        if self.nnz + len(columns) > self.limits.max_nonzeros or len(self.lower) >= self.limits.max_rows:
            raise ValueError("sparse model resource cap exceeded")
        self.nnz += len(columns)
        i = len(self.lower)
        if self.cached is None:
            self.row.extend([i] * len(columns))
            self.col.extend(columns)
            self.data.extend(values)
        else:
            extra = csr_matrix((values, ([0] * len(columns), columns)), shape=(1, self.count))
            self.cached = vstack((self.cached, extra), format="csr")
        self.lower.append(lower)
        self.upper.append(upper)

    def add(self, coefficients, lower=-np.inf, upper=np.inf):
        # Only objective lock rows use dictionaries, not every model row.
        pairs = [(j, v) for j, v in coefficients.items() if v]
        self.add_sparse((j for j, _ in pairs), (v for _, v in pairs), lower, upper)

    def matrix(self, columns):
        if columns != self.count:
            raise ValueError("CSR column identity changed")
        if self.cached is None:
            self.cached = csr_matrix((self.data, (self.row, self.col)),
                                     shape=(len(self.lower), columns), dtype=float)
            self.row.clear(); self.col.clear(); self.data.clear()
        return self.cached


class _Components:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, key):
        if key not in self.parent:
            self.parent[key], self.rank[key] = key, 0
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while key != root:
            previous = self.parent[key]
            self.parent[key] = root
            key = previous
        return root

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        self.rank[a] += int(self.rank[a] == self.rank[b])


def _compile(problem):
    vehicles, waiting = base._validate(problem)
    options = base._options(problem, vehicles, waiting)
    keys = {(o.vehicle_id, o.request_id): j for j, o in enumerate(options)}
    by_vehicle = defaultdict(list)
    by_request = defaultdict(list)
    for j, o in enumerate(options):
        by_vehicle[o.vehicle_id].append(j)
        if o.request_id is not None:
            by_request[o.request_id].append(j)
    recourse = []
    for scene in problem.scenarios:
        requests = {**waiting, **{r.request_id: r for r in scene.new_requests}}
        seen = set()
        for pickup in scene.pickups:
            key = pickup.vehicle_id, pickup.after_request_id, pickup.request_id
            if key in seen:
                raise ValueError("duplicate future sparse pickup")
            seen.add(key)
            if pickup.vehicle_id not in vehicles or pickup.request_id not in requests:
                raise ValueError("unknown future pickup identity")
            if pickup.after_request_id is not None and pickup.after_request_id not in waiting:
                raise ValueError("unknown prior current request")
            if not isfinite(pickup.pickup_eta_s) or pickup.pickup_eta_s < 0:
                raise ValueError("invalid future pickup ETA")
            j = keys.get((pickup.vehicle_id, pickup.after_request_id))
            if j is None:
                continue
            option = options[j]
            if pickup.origin_position != option.position:
                raise ValueError("future ETA origin disagrees with post-service vehicle position")
            request, vehicle = requests[pickup.request_id], vehicles[pickup.vehicle_id]
            departure = max(problem.now_s + problem.limits.rolling_step_s,
                            option.ready_time_s, request.release_time_s)
            arrival = departure + pickup.pickup_eta_s
            completion = arrival + request.pickup_overhead_s + request.predicted_service_time_s
            if (base._allowed(vehicle, request) and arrival <= request.pickup_deadline_s + 1e-7
                    and completion <= vehicle.availability_end_s + 1e-7):
                recourse.append((scene, pickup, j))
    # Preserve full logical model caps, BEFORE decomposing or substituting x=1.
    states = {(s.scenario_id, j) for s, _, j in recourse}
    future_v = {(s.scenario_id, p.vehicle_id) for s, p, _ in recourse}
    future_r = {(s.scenario_id, p.request_id) for s, p, _ in recourse}
    full_nnz = (len(options) + sum(len(v) for v in by_request.values())
        + 3 * len(recourse) + len(states)
        + sum(len(by_request.get(r, ())) for _, r in future_r))
    full_rows = len(by_vehicle) + len(waiting) + len(states) + len(future_v) + len(future_r)
    if full_nnz > problem.limits.max_nonzeros or full_rows > problem.limits.max_rows:
        raise ValueError("sparse model resource cap exceeded before decomposition")
    return vehicles, waiting, options, by_vehicle, by_request, recourse


def _matching(records):
    pairs, expected, bytes_used = [], 0., 0
    by_scene = defaultdict(list)
    for s, p, _ in records:
        by_scene[s.scenario_id].append((s, p))
    for scene_records in by_scene.values():
        scene = scene_records[0][0]
        edges = sorted({(p.vehicle_id, p.request_id) for _, p in scene_records})
        if not edges:
            continue
        vs = {v: i for i, v in enumerate(sorted({v for v, _ in edges}))}
        rs = {r: i for i, r in enumerate(sorted({r for _, r in edges}))}
        graph = csr_matrix((np.ones(len(edges), dtype=np.int8),
            ([vs[v] for v, _ in edges], [rs[r] for _, r in edges])), shape=(len(vs), len(rs)))
        bytes_used += graph.data.nbytes + graph.indices.nbytes + graph.indptr.nbytes
        match = maximum_bipartite_matching(graph, perm_type="column")
        rids = list(rs)
        for vid, i in vs.items():
            if match[i] >= 0:
                pairs.append((scene.scenario_id, vid, rids[match[i]]))
                expected += scene.probability
    return tuple(pairs), expected, bytes_used


def _solve_component(problem, vehicles, waiting, options, by_vehicle, by_request,
                     option_ids, records, face, remaining):
    started = time.perf_counter()
    free_ids = [j for j in option_ids if len(by_vehicle[options[j].vehicle_id]) > 1]
    col = {j: i for i, j in enumerate(free_ids)}
    count = len(free_ids) + len(records)
    limits = replace(problem.limits, solver_time_limit_s=remaining,
                     compress_fixed_states=False, trim_flow_rows=False)
    rows = CSRRows(limits, count)
    local_vehicles = {options[j].vehicle_id for j in free_ids}
    local_requests = {options[j].request_id for j in free_ids if options[j].request_id is not None}
    for vid in sorted(local_vehicles):
        rows.add_sparse((col[j] for j in by_vehicle[vid]), lower=1., upper=1.)
    for rid in sorted(local_requests):
        rows.add_sparse((col[j] for j in by_request[rid]), upper=1.)
    per_state, per_v, per_r = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, (s, p, j) in enumerate(records, len(free_ids)):
        per_state[s.scenario_id, j].append(i)
        per_v[s.scenario_id, p.vehicle_id].append(i)
        per_r[s.scenario_id, p.request_id].append(i)
    for (_, j), ys in per_state.items():
        if j in col:
            rows.add_sparse((*ys, col[j]), (*([1.] * len(ys)), -1.), upper=0.)
        else:
            rows.add_sparse(ys, upper=1.)  # Fixed idle state substituted here.
    # Retain useful implied capacity rows: fewer rows is not always faster.
    for ys in per_v.values():
        rows.add_sparse(ys, upper=1.)
    for (_, rid), ys in per_r.items():
        rows.add_sparse((*ys, *(col[j] for j in by_request.get(rid, ()))), upper=1.)
    critical, immediate, carry, future, eta = (np.zeros(count) for _ in range(5))
    for j, k in col.items():
        o = options[j]
        if o.request_id is not None:
            r = waiting[o.request_id]
            critical[k], immediate[k], carry[k], eta[k] = r.critical, 1., r.carry_over, o.pickup_eta_s
    for i, (s, _, _) in enumerate(records, len(free_ids)):
        future[i] = s.probability
    for objective, optimum in zip((critical, immediate, carry), face):
        nonzero = np.flatnonzero(objective)
        if len(nonzero):
            rows.add_sparse(nonzero, objective[nonzero], optimum - 1e-7, optimum + 1e-7)
    integrality = np.zeros(count, dtype=np.int8)
    integrality[:len(free_ids)] = 1
    build_s = time.perf_counter() - started
    partial, optimize_s = base._run_levels(rows, count, [(future, True), (eta, False)], limits, integrality)
    selected = tuple(sorted((options[j].vehicle_id, options[j].request_id) for j, k in col.items()
                            if options[j].request_id is not None and partial[k] > .5))
    full = np.ones(len(options))
    for j, k in col.items():
        full[j] = partial[k]
    recovery_started = time.perf_counter()
    recovered, q = base._recover_recourse(problem, records, full, selected)
    if abs(q - float(future @ partial)) > 1e-6:
        raise RuntimeError("v3 recourse failed integral matching recovery")
    recovery_s = time.perf_counter() - recovery_started
    matrix = rows.matrix(count)
    return selected, recovered, q, count, len(free_ids), matrix.nnz, (
        matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes), build_s, optimize_s, recovery_s


def solve_dispatch_v3(problem, policy, *, current_face=None, current_selection=None):
    """Strict SP-lookahead only; unchanged other policies use the old backend."""
    if policy != "SERVICE_PRESERVING_LOOKAHEAD":
        return base.solve_dispatch(problem, policy, current_face=current_face)
    if problem.limits.recourse_mode != "FLOW_RELAXED":
        raise ValueError("v3 decomposition requires the frozen one-next-service FLOW model")
    started = time.perf_counter()
    vehicles, waiting, options, by_v, by_r, recourse = _compile(problem)
    arcs = [AssignmentArc(o.vehicle_id, o.request_id, o.pickup_eta_s,
                         waiting[o.request_id].critical, waiting[o.request_id].carry_over)
            for o in options if o.request_id is not None]
    signature = tuple(sorted((a.vehicle_id, a.request_id, a.pickup_eta_s, a.critical, a.carry_over) for a in arcs))
    if current_face is not None and signature != current_face.signature:
        raise ValueError("reused current-service face belongs to a different sparse graph")
    if current_selection is not None and current_face is None:
        raise ValueError("a supplied current selection requires its optimal face certificate")
    if current_selection is None:
        oracle = solve_lexicographic(arcs)
        current_selection = tuple((arcs[i].vehicle_id, arcs[i].request_id) for i in oracle.selected_indices)
        if current_face is None:
            current_face = base.CurrentServiceFace.from_arcs(arcs, oracle)
    selection = set(current_selection)
    if (len(selection) != len(current_selection) or len({v for v, _ in selection}) != len(selection)
            or len({r for _, r in selection}) != len(selection)
            or not selection <= {(a.vehicle_id, a.request_id) for a in arcs}):
        raise ValueError("invalid current oracle selection")
    observed_face = (sum(waiting[r].critical for _, r in selection), len(selection),
                     sum(waiting[r].carry_over for _, r in selection))
    if current_face is not None and observed_face != (
            current_face.critical_matched, current_face.total_matched, current_face.carry_over_matched):
        raise ValueError("current oracle selection does not lie on the supplied face")
    graph = _Components()
    free_v = {v for v, ids in by_v.items() if len(ids) > 1}
    for vid in sorted(free_v):
        graph.find(("V", vid))
    for o in options:
        if o.request_id is not None:
            graph.join(("V", o.vehicle_id), ("C", o.request_id))
    edge_nodes = []
    for s, p, _ in recourse:
        v = ("V", p.vehicle_id) if p.vehicle_id in free_v else ("FV", s.scenario_id, p.vehicle_id)
        r = ("R", s.scenario_id, p.request_id)
        graph.join(v, r)
        if by_r.get(p.request_id):
            graph.join(r, ("C", p.request_id))
        edge_nodes.append(v)
    groups = {}
    for vid in sorted(free_v):
        groups.setdefault(graph.find(("V", vid)), {"vehicles": [], "records": []})["vehicles"].append(vid)
    for record, node in zip(recourse, edge_nodes):
        groups.setdefault(graph.find(node), {"vehicles": [], "records": []})["records"].append(record)
    build_s = time.perf_counter() - started
    selected, recovered = [], []
    expected = variables = integer_count = nnz = matrix_bytes = 0
    optimize_s = recovery_s = pure_expected = 0.
    pure_count = pure_edges = mip_count = 0
    for group in groups.values():
        records = group["records"]
        if not group["vehicles"]:
            t = time.perf_counter()
            pairs, q, memory = _matching(records)
            recovery_s += time.perf_counter() - t
            recovered.extend(pairs); expected += q; pure_expected += q
            pure_count += 1; pure_edges += len(records); matrix_bytes += memory
            continue
        vids = set(group["vehicles"])
        fixed_ids = {j for _, _, j in records if options[j].vehicle_id not in vids}
        option_ids = [j for j, o in enumerate(options) if o.vehicle_id in vids or j in fixed_ids]
        local_choice = [(v, r) for v, r in selection if v in vids]
        face = (sum(waiting[r].critical for _, r in local_choice), len(local_choice),
                sum(waiting[r].carry_over for _, r in local_choice))
        remaining = problem.limits.solver_time_limit_s - optimize_s
        if remaining <= 0:
            raise RuntimeError("sparse model solver timeout")
        result = _solve_component(problem, vehicles, waiting, options, by_v, by_r,
                                  option_ids, records, face, remaining)
        choice, pairs, q, n, ni, nz, memory, b, o, recovery = result
        selected.extend(choice); recovered.extend(pairs); expected += q
        variables += n; integer_count += ni; nnz += nz; matrix_bytes += memory
        build_s += b; optimize_s += o; recovery_s += recovery; mip_count += 1
    if (sum(waiting[r].critical for _, r in selected), len(selected),
            sum(waiting[r].carry_over for _, r in selected)) != observed_face:
        raise RuntimeError("decomposition violated the current lexicographic face")
    for s in problem.scenarios:
        pairs = [(v, r) for sid, v, r in recovered if sid == s.scenario_id]
        if (len({v for v, _ in pairs}) != len(pairs) or len({r for _, r in pairs}) != len(pairs)
                or {r for _, r in pairs} & {r for _, r in selected}):
            raise RuntimeError("decomposed recourse has duplicate capacity allocation")
    return DecisionV3(policy, tuple(sorted(selected)), len(selected), expected, len(selected) + expected,
        optimize_s, variables, nnz, matrix_bytes, tuple(recovered), integer_count, build_s, recovery_s,
        problem.limits.solver_backend, "FLOW_RELAXED", len(options) - integer_count,
        len(groups), mip_count, pure_count, pure_edges, pure_expected)
