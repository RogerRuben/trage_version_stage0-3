"""CPU-only sparse two-stage next-service assignment research prototype.

Only the first action is implemented. Recourse is at most ONE additional
service per vehicle/scenario, not a complete stochastic vehicle-routing model.
Inputs are sparse pickup estimates, never request-by-vehicle dense matrices.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from math import isfinite

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from .solver import AssignmentArc, solve_lexicographic


@dataclass(frozen=True)
class Vehicle:
    vehicle_id: int
    profile_id: str
    ready_position: str
    ready_time_s: float
    availability_end_s: float


@dataclass(frozen=True)
class Request:
    request_id: int
    release_time_s: float
    pickup_deadline_s: float
    predicted_service_time_s: float
    dropoff_position: str
    compatible_profiles: frozenset[str] = field(default_factory=lambda: frozenset(("C", "M", "A")))
    passenger_accepts_av: bool = True
    pickup_overhead_s: float = 0.0
    critical: bool = False
    carry_over: bool = False


@dataclass(frozen=True)
class CurrentPickup:
    vehicle_id: int
    request_id: int
    pickup_eta_s: float


@dataclass(frozen=True)
class FuturePickup:
    vehicle_id: int
    after_request_id: int | None
    request_id: int
    pickup_eta_s: float
    origin_position: str


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    probability: float
    information_available_at_s: float
    new_requests: tuple[Request, ...]
    pickups: tuple[FuturePickup, ...]


@dataclass(frozen=True)
class ModelLimits:
    horizon_s: float = 600.0
    rolling_step_s: float = 30.0
    max_scenarios: int = 8
    max_variables: int = 20_000
    max_nonzeros: int = 150_000
    max_rows: int = 50_000
    solver_time_limit_s: float = 30.0
    recourse_mode: str = "BINARY"
    solver_backend: str = "SCIPY"
    highspy_runtime_dir: str | None = None
    lock_current_face: bool = False


@dataclass(frozen=True)
class Problem:
    now_s: float
    vehicles: tuple[Vehicle, ...]
    waiting_requests: tuple[Request, ...]
    current_pickups: tuple[CurrentPickup, ...]
    scenarios: tuple[Scenario, ...] = ()
    limits: ModelLimits = field(default_factory=ModelLimits)


@dataclass(frozen=True)
class Decision:
    policy: str
    selected_pairs: tuple[tuple[int, int], ...]
    immediate_service_count: int
    expected_next_service_count: float
    expected_total_service_count: float
    solve_time_s: float
    variable_count: int
    constraint_nonzeros: int
    sparse_matrix_bytes: int
    recourse_pairs: tuple[tuple[str, int, int], ...] = ()
    integer_variable_count: int = 0
    model_build_time_s: float = 0.0
    recourse_recovery_time_s: float = 0.0
    solver_backend: str = "SCIPY"
    recourse_mode: str = "BINARY"


@dataclass(frozen=True)
class _Option:
    vehicle_id: int
    request_id: int | None
    ready_time_s: float
    position: str
    pickup_eta_s: float = 0.0


class _SparseRows:
    def __init__(self, limits: ModelLimits):
        self.limits = limits
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.values: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []

    def add(self, coefficients: dict[int, float], lower=-np.inf, upper=np.inf):
        nonzero = [(j, x) for j, x in coefficients.items() if x != 0]
        if (len(self.values) + len(nonzero) > self.limits.max_nonzeros
                or len(self.lower) + 1 > self.limits.max_rows):
            raise ValueError("sparse model resource cap exceeded; reduce candidate inputs")
        row = len(self.lower)
        for column, value in nonzero:
            self.rows.append(row)
            self.cols.append(column)
            self.values.append(value)
        self.lower.append(lower)
        self.upper.append(upper)

    def matrix(self, columns):
        return csr_matrix((self.values, (self.rows, self.cols)),
                          shape=(len(self.lower), columns), dtype=float)


def _validate_request(request):
    numbers = (request.release_time_s, request.pickup_deadline_s,
               request.predicted_service_time_s, request.pickup_overhead_s)
    if (not all(isfinite(x) for x in numbers)
            or request.pickup_deadline_s < request.release_time_s
            or request.predicted_service_time_s <= 0 or request.pickup_overhead_s < 0
            or not request.compatible_profiles <= {"C", "M", "A"}
            or type(request.passenger_accepts_av) is not bool):
        raise ValueError("invalid request time/compatibility input")


def _allowed(vehicle, request):
    return vehicle.profile_id == "HV" or (
        request.passenger_accepts_av and vehicle.profile_id in request.compatible_profiles)


def _validate(problem, *, fixed_action=False):
    limits = problem.limits
    if type(limits.lock_current_face) is not bool:
        raise ValueError("current face lock must be an explicit boolean")
    if limits.recourse_mode not in ("BINARY", "FLOW_RELAXED") or limits.solver_backend not in ("SCIPY", "HIGHS_PERSISTENT"):
        raise ValueError("unrecognized acceleration mode")
    if (not isfinite(problem.now_s) or not isfinite(limits.horizon_s)
            or limits.horizon_s <= 0 or not isfinite(limits.rolling_step_s)
            or not 0 < limits.rolling_step_s <= limits.horizon_s
            or not isfinite(limits.solver_time_limit_s) or limits.solver_time_limit_s <= 0
            or min(limits.max_scenarios, limits.max_variables,
                   limits.max_nonzeros, limits.max_rows) <= 0):
        raise ValueError("invalid model limits")
    if len(problem.scenarios) > limits.max_scenarios:
        raise ValueError("scenario resource cap exceeded")
    # Reject oversized sparse inputs BEFORE any solver vector/matrix allocation.
    upper_columns = (len(problem.vehicles) + len(problem.current_pickups)
                     + sum(len(s.pickups) for s in problem.scenarios))
    if upper_columns > limits.max_variables:
        raise ValueError("variable resource cap exceeded; reduce candidate inputs")
    vehicles = {v.vehicle_id: v for v in problem.vehicles}
    requests = {r.request_id: r for r in problem.waiting_requests}
    if len(vehicles) != len(problem.vehicles) or len(requests) != len(problem.waiting_requests):
        raise ValueError("duplicate vehicle/request identity")
    for vehicle in vehicles.values():
        if (vehicle.profile_id not in ("HV", "C", "M", "A")
                or not all(isfinite(x) for x in (vehicle.ready_time_s, vehicle.availability_end_s))
                or vehicle.availability_end_s < vehicle.ready_time_s):
            raise ValueError("invalid vehicle state")
    for request in requests.values():
        _validate_request(request)
        if request.release_time_s > problem.now_s:
            raise ValueError("unreleased request in current waiting set")
    if len({s.scenario_id for s in problem.scenarios}) != len(problem.scenarios):
        raise ValueError("duplicate scenario identity")
    if problem.scenarios and abs(sum(s.probability for s in problem.scenarios) - 1) > 1e-8:
        raise ValueError("scenario probabilities must sum to one")
    for scenario in problem.scenarios:
        if (not isfinite(scenario.probability) or scenario.probability <= 0
                or not isfinite(scenario.information_available_at_s)):
            raise ValueError("invalid scenario probability/provenance")
        if not fixed_action and scenario.information_available_at_s > problem.now_s:
            raise ValueError("future information cannot enter the dispatch decision")
        ids = set(requests)
        for request in scenario.new_requests:
            _validate_request(request)
            if request.request_id in ids:
                raise ValueError("new forecast request duplicates a known request")
            ids.add(request.request_id)
            if not problem.now_s < request.release_time_s <= problem.now_s + limits.horizon_s:
                raise ValueError("forecast release outside the next-service horizon")
    return vehicles, requests


def _options(problem, vehicles, requests):
    options = [_Option(v.vehicle_id, None, max(problem.now_s, v.ready_time_s), v.ready_position)
               for v in problem.vehicles]
    seen = set()
    for pickup in problem.current_pickups:
        key = (pickup.vehicle_id, pickup.request_id)
        if key in seen:
            raise ValueError("duplicate current sparse pickup")
        seen.add(key)
        if pickup.vehicle_id not in vehicles or pickup.request_id not in requests:
            raise ValueError("unknown current pickup identity")
        if not isfinite(pickup.pickup_eta_s) or pickup.pickup_eta_s < 0:
            raise ValueError("invalid current pickup ETA")
        vehicle, request = vehicles[pickup.vehicle_id], requests[pickup.request_id]
        arrival = problem.now_s + pickup.pickup_eta_s
        completion = arrival + request.pickup_overhead_s + request.predicted_service_time_s
        if (vehicle.ready_time_s <= problem.now_s and _allowed(vehicle, request)
                and arrival <= request.pickup_deadline_s + 1e-7
                and completion <= vehicle.availability_end_s + 1e-7):
            options.append(_Option(vehicle.vehicle_id, request.request_id, completion,
                                   request.dropoff_position, pickup.pickup_eta_s))
    return options


def _run_levels(rows, count, levels, limits, integrality=None):
    if count == 0:
        return np.zeros(0), 0.0
    if limits.solver_backend == "HIGHS_PERSISTENT":
        from .persistent_highs import run_levels
        return run_levels(rows, count, levels, limits, integrality)
    started = time.perf_counter()
    solution = None
    for objective, maximize in levels:
        if not np.any(objective):
            continue
        remaining = limits.solver_time_limit_s - (time.perf_counter() - started)
        if remaining <= 0:
            raise RuntimeError("sparse model solver timeout")
        result = milp(-objective if maximize else objective,
                      integrality=np.ones(count, dtype=np.int8) if integrality is None else integrality,
                      bounds=Bounds(np.zeros(count), np.ones(count)),
                      constraints=LinearConstraint(rows.matrix(count), rows.lower, rows.upper),
                      options={"presolve": True, "time_limit": remaining, "mip_rel_gap": 0.0})
        if not result.success or result.x is None:
            raise RuntimeError(f"sparse model not proven optimal: {result.message}")
        solution = result.x
        optimum = float(objective @ solution)
        nonzero = np.flatnonzero(objective)
        rows.add({int(j): float(objective[j]) for j in nonzero},
                 optimum - 1e-7, optimum + 1e-7)
    if solution is None:  # Idle-only model still has vehicle-state equalities.
        objective = np.ones(count)
        remaining = limits.solver_time_limit_s - (time.perf_counter() - started)
        result = milp(objective, integrality=np.ones(count, dtype=np.int8) if integrality is None else integrality,
                      bounds=Bounds(np.zeros(count), np.ones(count)),
                      constraints=LinearConstraint(rows.matrix(count), rows.lower, rows.upper),
                      options={"time_limit": max(remaining, 0.001), "mip_rel_gap": 0.0})
        if not result.success or result.x is None:
            raise RuntimeError(f"idle-only model failed: {result.message}")
        solution = result.x
    return solution, time.perf_counter() - started


def _recover_recourse(problem, recourse, solution, selected):
    """Integral maximum matchings for the selected first-stage vehicle states.

    This applies to the ONE-next-service model only. Current served requests
    have zero residual capacity; each remaining vehicle/request has capacity 1.
    No dense assignment matrix or rounding of fractional future edges is used.
    """
    served_now = {request for _, request in selected}
    recovered = []
    expected = 0.0
    for scenario in problem.scenarios:
        edges = sorted({(pickup.vehicle_id, pickup.request_id)
                        for s, pickup, option_j in recourse
                        if s.scenario_id == scenario.scenario_id
                        and solution[option_j] > .5 and pickup.request_id not in served_now})
        if not edges:
            continue
        vehicles = {v: i for i, v in enumerate(sorted({v for v, _ in edges}))}
        requests = {r: i for i, r in enumerate(sorted({r for _, r in edges}))}
        graph = csr_matrix((np.ones(len(edges), dtype=np.int8),
                            ([vehicles[v] for v, _ in edges], [requests[r] for _, r in edges])),
                           shape=(len(vehicles), len(requests)))
        matching = maximum_bipartite_matching(graph, perm_type="column")
        request_ids = list(requests)
        for vehicle, row in vehicles.items():
            if matching[row] >= 0:
                recovered.append((scenario.scenario_id, vehicle, request_ids[matching[row]]))
                expected += scenario.probability
    return tuple(recovered), expected


def _solve(problem, policy, fixed_pairs=None):
    build_started = time.perf_counter()
    vehicles, requests = _validate(problem, fixed_action=fixed_pairs is not None)
    options = _options(problem, vehicles, requests)
    option_keys = {(o.vehicle_id, o.request_id): j for j, o in enumerate(options)}
    recourse = []
    if policy in ("LOOKAHEAD", "SERVICE_PRESERVING_LOOKAHEAD") or fixed_pairs is not None:
        for scenario in problem.scenarios:
            future_requests = {**requests, **{r.request_id: r for r in scenario.new_requests}}
            seen = set()
            for pickup in scenario.pickups:
                key = (pickup.vehicle_id, pickup.after_request_id, pickup.request_id)
                if key in seen:
                    raise ValueError("duplicate future sparse pickup")
                seen.add(key)
                if pickup.vehicle_id not in vehicles or pickup.request_id not in future_requests:
                    raise ValueError("unknown future pickup identity")
                if pickup.after_request_id is not None and pickup.after_request_id not in requests:
                    raise ValueError("unknown prior current request")
                if not isfinite(pickup.pickup_eta_s) or pickup.pickup_eta_s < 0:
                    raise ValueError("invalid future pickup ETA")
                j = option_keys.get((pickup.vehicle_id, pickup.after_request_id))
                if j is None:
                    continue  # Its current state option was physically infeasible.
                option = options[j]
                if pickup.origin_position != option.position:
                    raise ValueError("future ETA origin disagrees with post-service vehicle position")
                request = future_requests[pickup.request_id]
                vehicle = vehicles[pickup.vehicle_id]
                departure = max(problem.now_s + problem.limits.rolling_step_s,
                                option.ready_time_s, request.release_time_s)
                arrival = departure + pickup.pickup_eta_s
                completion = arrival + request.pickup_overhead_s + request.predicted_service_time_s
                if (_allowed(vehicle, request)
                        and arrival <= request.pickup_deadline_s + 1e-7
                        and completion <= vehicle.availability_end_s + 1e-7):
                    recourse.append((scenario, pickup, j))
    count = len(options) + len(recourse)
    rows = _SparseRows(problem.limits)
    by_vehicle = {v: {} for v in vehicles}
    by_current_request = {r: {} for r in requests}
    for j, option in enumerate(options):
        by_vehicle[option.vehicle_id][j] = 1.0
        if option.request_id is not None:
            by_current_request[option.request_id][j] = 1.0
    for coefficients in by_vehicle.values():
        rows.add(coefficients, 1.0, 1.0)
    for coefficients in by_current_request.values():
        rows.add(coefficients, upper=1.0)
    if fixed_pairs is not None:
        if len(set(fixed_pairs)) != len(fixed_pairs):
            raise ValueError("duplicate fixed assignment")
        fixed = dict(fixed_pairs)
        if len(fixed) != len(fixed_pairs) or any(v not in vehicles for v in fixed):
            raise ValueError("invalid fixed-action vehicle identity")
        for vehicle in vehicles:
            key = (vehicle, fixed.get(vehicle))
            if key not in option_keys:
                raise ValueError("fixed action is not a feasible current assignment")
            rows.add({option_keys[key]: 1.0}, 1.0, 1.0)
    per_future_vehicle = {}
    per_future_request = {}
    per_future_option = {}
    for i, (scenario, pickup, option_j) in enumerate(recourse, start=len(options)):
        if problem.limits.recourse_mode == "FLOW_RELAXED":
            per_future_option.setdefault((scenario.scenario_id, option_j), {})[i] = 1.0
        else:
            rows.add({i: 1.0, option_j: -1.0}, upper=0.0)
        per_future_vehicle.setdefault((scenario.scenario_id, pickup.vehicle_id), {})[i] = 1.0
        per_future_request.setdefault((scenario.scenario_id, pickup.request_id), {})[i] = 1.0
    # Capacity on a chosen state, not merely one independent gate per edge.
    # Equivalent for binary x, and stronger/fewer rows in the LP relaxation.
    for (_, option_j), coefficients in per_future_option.items():
        rows.add({**coefficients, option_j: -1.0}, upper=0.0)
    for coefficients in per_future_vehicle.values():
        rows.add(coefficients, upper=1.0)
    for (_, request_id), coefficients in per_future_request.items():
        # Known waiting requests can be served now OR later, never twice.
        rows.add({**coefficients, **by_current_request.get(request_id, {})}, upper=1.0)
    immediate = np.zeros(count)
    critical = np.zeros(count)
    carry = np.zeros(count)
    av = np.zeros(count)
    eta = np.zeros(count)
    future_value = np.zeros(count)
    for j, option in enumerate(options):
        if option.request_id is not None:
            request = requests[option.request_id]
            immediate[j] = 1.0
            critical[j], carry[j] = float(request.critical), float(request.carry_over)
            av[j] = float(vehicles[option.vehicle_id].profile_id != "HV")
            eta[j] = option.pickup_eta_s
    for i, (scenario, _, _) in enumerate(recourse, start=len(options)):
        future_value[i] = scenario.probability
    if fixed_pairs is not None:
        levels = [(future_value, True)]
    elif policy == "LOOKAHEAD":
        levels = [(critical, True), (immediate + future_value, True),
                  (immediate, True), (carry, True), (eta, False)]
    elif policy == "SERVICE_PRESERVING_LOOKAHEAD":
        # Optimize future flexibility ONLY on the same current-service face as
        # MYOPIC. This protects epoch counts, not full-window service dominance.
        if problem.limits.lock_current_face:
            # y=0 is feasible for every feasible first action. The first three
            # optima therefore come from the current sparse graph alone.
            current_arcs = [AssignmentArc(o.vehicle_id, o.request_id, o.pickup_eta_s,
                requests[o.request_id].critical, requests[o.request_id].carry_over,
                vehicle_type="HV" if vehicles[o.vehicle_id].profile_id == "HV" else "AV")
                for o in options if o.request_id is not None]
            face = solve_lexicographic(current_arcs)
            for objective, optimum in ((critical, face.critical_matched),
                                       (immediate, face.total_matched),
                                       (carry, face.carry_over_matched)):
                nonzero = np.flatnonzero(objective)
                if len(nonzero):
                    rows.add({int(j): float(objective[j]) for j in nonzero},
                             optimum - 1e-7, optimum + 1e-7)
            levels = [(future_value, True), (eta, False)]
        else:
            levels = [(critical, True), (immediate, True), (carry, True),
                      (future_value, True), (eta, False)]
    else:  # Simple AV-first control: differs ONLY in the pre-ETA tie-break.
        levels = [(critical, True), (immediate, True), (carry, True), (av, True), (eta, False)]
    integrality = None
    if problem.limits.recourse_mode == "FLOW_RELAXED":
        integrality = np.zeros(count, dtype=np.int8)
        integrality[:len(options)] = 1
    build_time = time.perf_counter() - build_started
    solution, runtime = _run_levels(rows, count, levels, problem.limits, integrality)
    selected = tuple(sorted((o.vehicle_id, o.request_id) for j, o in enumerate(options)
                            if o.request_id is not None and solution[j] > 0.5))
    selected_future = tuple((s.scenario_id, p.vehicle_id, p.request_id)
                            for i, (s, p, _) in enumerate(recourse, start=len(options))
                            if solution[i] > 0.5)
    matrix = rows.matrix(count)
    future = float(future_value @ solution)
    recovery_time = 0.0
    if problem.limits.recourse_mode == "FLOW_RELAXED":
        recovery_started = time.perf_counter()
        selected_future, recovered_value = _recover_recourse(problem, recourse, solution, selected)
        if abs(recovered_value - future) > 1e-6:
            raise RuntimeError("continuous recourse failed integral matching recovery")
        future = recovered_value
        recovery_time = time.perf_counter() - recovery_started
    return Decision(policy, selected, len(selected), future, len(selected) + future,
                    runtime, count, matrix.nnz,
                    matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes,
                    selected_future, integer_variable_count=count if integrality is None else int(integrality.sum()),
                    model_build_time_s=build_time, recourse_recovery_time_s=recovery_time,
                    solver_backend=problem.limits.solver_backend, recourse_mode=problem.limits.recourse_mode)


def solve_dispatch(problem: Problem, policy: str) -> Decision:
    """MYOPIC reuses the unchanged frozen solver; other policies are opt-in."""
    if policy not in ("MYOPIC", "AV_FIRST", "LOOKAHEAD", "SERVICE_PRESERVING_LOOKAHEAD"):
        raise ValueError("unrecognized research dispatch policy")
    if policy != "MYOPIC":
        return _solve(problem, policy)
    vehicles, requests = _validate(problem)
    options = _options(problem, vehicles, requests)
    arcs = [AssignmentArc(o.vehicle_id, o.request_id, o.pickup_eta_s,
                          requests[o.request_id].critical, requests[o.request_id].carry_over,
                          vehicle_type="HV" if vehicles[o.vehicle_id].profile_id == "HV" else "AV")
            for o in options if o.request_id is not None]
    result = solve_lexicographic(arcs)
    selected = tuple(sorted((arcs[i].vehicle_id, arcs[i].request_id) for i in result.selected_indices))
    return Decision(policy, selected, len(selected), 0.0, float(len(selected)),
                    result.solve_time_s, len(arcs), 2 * len(arcs), 0)


def score_fixed_action(problem: Problem, selected_pairs, realized_scenario: Scenario) -> Decision:
    """Ex-post one-next-service score; cannot change the already selected action.

    All policies receive the SAME optimistic continuation model. This isolates
    first-action quality; it is not a native FleetPy full-window service result.
    """
    from dataclasses import replace
    if abs(realized_scenario.probability - 1.0) > 1e-8:
        raise ValueError("realized continuation must be a single probability-one state")
    scored = replace(problem, scenarios=(realized_scenario,))
    return _solve(scored, "FIXED_ACTION_EVALUATION", tuple(selected_pairs))
