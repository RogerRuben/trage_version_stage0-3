"""City sparse ONE-next-service defer controls, not complete chain planning.

The first stage uses the same caller-validated native arcs for both controls.
Historical M3 chord pace predicts known-state spatial pickups. Only the
TIME_TYPE_DEFER after-current-task layer substitutes a frozen typical ETA and
distance, with time-layer/request-identity-hash Top-K candidates. Both controls
may wait.
No execution truth, routes, data loaders, or automatic fallback are used here.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
from heapq import merge
import hashlib
from math import ceil, isfinite, lcm
from time import perf_counter
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.spatial import cKDTree

from . import flexibility_model as base
from . import flexibility_v3 as decomposed
from .flexibility_native import NativeFlexibilityAdapter, cached_xy, position_key
from .generic_defer import PROXY_PROVENANCE
from .solver import LexicographicResult


POLICIES = ("LOCATION_AWARE_DEFER", "TIME_TYPE_DEFER")
KIND = "CITY_TWO_STAGE_ONE_NEXT_SERVICE_APPROXIMATION_NOT_COMPLETE_CHAIN_OPTIMUM"
MAX_VARIABLES = 20_000
MAX_NONZEROS = 150_000
DECISION_TIME_LIMIT_S = 10.0
MIP_FEASIBILITY_TOLERANCE = 1e-9
PRIMAL_FEASIBILITY_TOLERANCE = 1e-9


class CityNumericalContractError(RuntimeError):
    """Numerical failure with aggregate diagnostics, never raw request IDs."""
    def __init__(self, message, diagnostics):
        self.diagnostics = deepcopy(diagnostics)
        super().__init__(f"{message}; numerical_diagnostics={self.diagnostics!r}")


@lru_cache(maxsize=1)
def _solver_options_receipt():
    # This is the same bundled core/OptionsManager/passOptions path used by
    # SciPy's _highs_wrapper. Never claim a tolerance merely from a warning.
    from scipy.optimize._highspy import _core, _highs_options
    probe, options = _core._Highs(), _core.HighsOptions()
    manager = _highs_options.HighsOptionsManager()
    options.output_flag = False
    requested = dict(mip_feasibility_tolerance=MIP_FEASIBILITY_TOLERANCE,
                     primal_feasibility_tolerance=PRIMAL_FEASIBILITY_TOLERANCE,
                     dual_feasibility_tolerance=1e-9, mip_abs_gap=0.0)
    for key, value in requested.items():
        if manager.get_option_type(key) == -1:
            raise CityNumericalContractError("bundled HiGHS does not recognize a required option", dict(option=key))
        setattr(options, key, value)
    if probe.passOptions(options) != _core.HighsStatus.kOk:
        raise CityNumericalContractError("bundled HiGHS rejected explicit numerical options", {})
    for key, value in requested.items():
        status, actual = probe.getOptionValue(key)
        if status != _core.HighsStatus.kOk or actual != value:
            raise CityNumericalContractError("bundled HiGHS numerical option readback disagrees", dict(option=key))
    return dict(highs_version=probe.version(), options=requested,
                bundled_options_manager_and_pass_options_verified=True,
                scipy_forwarding_path="MILP_IV_OPTIONS_UPDATE_THEN_HIGHS_OPTIONS_MANAGER_PASSOPTIONS")


def _scenario_lattice(problem):
    probabilities = [scene.probability for scene in problem.scenarios]
    if not probabilities:
        fractions = []
    elif all(value == float(Fraction(1, len(probabilities))) for value in probabilities):
        # Configured equally weighted history days are exactly 1/N; preserve
        # their float input records while avoiding binary 1/3 objective locks.
        fractions = [Fraction(1, len(probabilities))] * len(probabilities)
    else:
        fractions = [Fraction(str(value)) for value in probabilities]
        if sum(fractions) != 1:
            raise CityNumericalContractError("nonuniform scenario weights have no exact unit-sum decimal lattice", {})
    denominator = lcm(*(value.denominator for value in fractions)) if fractions else 1
    numerators = {scene.scenario_id: int(value * denominator) for scene, value in zip(problem.scenarios, fractions)}
    return denominator, numerators


def _maximum_row_violation(rows, count, vector):
    lhs = rows.matrix(count) @ vector
    return max(0.0, float(np.max(np.asarray(rows.lower) - lhs, initial=0.0)),
               float(np.max(lhs - np.asarray(rows.upper), initial=0.0)))


def _run_city_levels(rows, count, objectives, integrality, certify, budget):
    receipt = _solver_options_receipt()
    stages, solution = [], None
    optimize_s = 0.0
    for name, objective, maximize, divisor in objectives:
        budget.remaining(name)
        if not np.any(objective):
            stages.append(dict(stage=name, value=0.0, opt_status="CONSTANT_EXACT", objective_lattice_denominator=divisor))
            continue
        if divisor is not None and float(np.sum(np.abs(objective))) >= 2**53:
            raise CityNumericalContractError("integer objective exceeds exact double lattice range", dict(stage=name))
        options = dict(receipt["options"], time_limit=budget.remaining(name), mip_rel_gap=0.0, presolve=True)
        started = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = milp(-objective if maximize else objective, integrality=integrality,
                          bounds=Bounds(np.zeros(count), np.ones(count)),
                          constraints=LinearConstraint(rows.matrix(count), rows.lower, rows.upper), options=options)
        optimize_s += perf_counter() - started
        # SciPy's advertised forwarding warning is expected. A wrapper-level
        # rejection/ignored-value warning is fatal, not silently suppressed.
        for warning in caught:
            message = str(warning.message)
            if not ("Unrecognized options detected" in message and "passed to HiGHS verbatim" in message):
                raise CityNumericalContractError("solver warned that numerical options may not apply", dict(stage=name, warning=message))
        if result.status == 1:
            raise CityDecisionTimeout(f"{name}: sparse model not proven optimal: {result.message}")
        if result.status != 0 or not result.success or result.x is None:
            raise RuntimeError(f"{name}: sparse model not proven optimal: {result.message}")
        budget.remaining(name)
        solution, diagnostics = certify(result.x, name)
        certified_value = float(objective @ solution)
        native_value = float(-result.fun if maximize else result.fun)
        # A feasible integral matching, not rounding of a floating objective,
        # determines the exact lattice value. Bound numerical objective drift
        # by explicitly declared variable/row tolerances and finite coefficients.
        coefficient_sum = float(np.sum(np.abs(objective)))
        largest_coefficient = float(np.max(np.abs(objective), initial=0.0))
        error_bound = (MIP_FEASIBILITY_TOLERANCE * coefficient_sum
                       + PRIMAL_FEASIBILITY_TOLERANCE * len(rows.lower) * largest_coefficient
                       + 64 * np.finfo(float).eps * max(1.0, coefficient_sum))
        diagnostics.update(native_objective_value=native_value, certified_objective_value=certified_value,
                           objective_numeric_error_bound=error_bound)
        if abs(native_value - certified_value) > error_bound:
            raise CityNumericalContractError("matching certificate changed the tier objective beyond numerical bounds", diagnostics)
        grid_units = None
        if divisor is not None:
            if error_bound >= .25 or certified_value != int(certified_value):
                raise CityNumericalContractError("tier has no uniquely certified integer lattice value", diagnostics)
            grid_units = int(certified_value)
            bound = getattr(result, "mip_dual_bound", None)
            min_value = -grid_units if maximize else grid_units
            if bound is None or not isfinite(bound) or ceil(float(bound) - error_bound) < min_value:
                raise CityNumericalContractError("optimal solver status does not certify the integer objective grid", diagnostics)
            indices = np.flatnonzero(objective)
            rows.add_sparse(indices, objective[indices], lower=grid_units, upper=grid_units)
        # Pickup ETA is last: do not manufacture a floating +/-epsilon face.
        if _maximum_row_violation(rows, count, solution) != 0.0:
            raise CityNumericalContractError("integral certificate violates an exact tier lock", diagnostics)
        stages.append(dict(stage=name, value=certified_value / divisor if divisor else certified_value,
                           opt_status="OPTIMAL", status=int(result.status),
                           objective_lattice_denominator=divisor, grid_units=grid_units,
                           exact_objective_lock=divisor is not None, numerical_diagnostics=diagnostics))
    if solution is None:
        raise RuntimeError("city component has no decision-relevant objective")
    return solution, optimize_s, tuple(stages), receipt


class CityDecisionTimeout(TimeoutError):
    pass


class _Budget:
    def __init__(self, seconds):
        if not isfinite(seconds) or seconds <= 0:
            raise ValueError("city OR decision budget must be positive and finite")
        self.seconds = min(float(seconds), DECISION_TIME_LIMIT_S)
        self.started = perf_counter()

    def remaining(self, stage):
        remaining = self.seconds - (perf_counter() - self.started)
        if remaining <= 0:
            raise CityDecisionTimeout(f"{stage}: city total {self.seconds:g}s OR decision budget expired")
        return remaining


@dataclass(frozen=True)
class CityDeferDecision(decomposed.DecisionV3):
    decision_time_s: float = 0.0
    logical_variable_count: int = 0
    logical_nonzeros: int = 0
    logical_rows: int = 0
    critical_matched: int = 0
    carry_over_matched: int = 0
    stages: tuple = ()
    provenance: dict | None = None
    numerical_contract: dict | None = None


def _bounded_problem(problem, remaining):
    limits = replace(problem.limits, max_variables=min(problem.limits.max_variables, MAX_VARIABLES),
                     max_nonzeros=min(problem.limits.max_nonzeros, MAX_NONZEROS),
                     solver_time_limit_s=remaining, recourse_mode="FLOW_RELAXED",
                     lock_current_face=False, compress_fixed_states=False, trim_flow_rows=False)
    return replace(problem, limits=limits)


def _solve_component(problem, waiting, options, by_vehicle, by_request, option_ids, records, budget, lattice):
    started = perf_counter()
    free_ids = [j for j in option_ids if len(by_vehicle[options[j].vehicle_id]) > 1]
    columns = {j: index for index, j in enumerate(free_ids)}
    count = len(free_ids) + len(records)
    rows = decomposed.CSRRows(problem.limits, count)
    for vid in sorted({options[j].vehicle_id for j in free_ids}):
        rows.add_sparse((columns[j] for j in by_vehicle[vid]), lower=1., upper=1.)
    for rid in sorted({options[j].request_id for j in free_ids if options[j].request_id is not None}):
        rows.add_sparse((columns[j] for j in by_request[rid]), upper=1.)
    per_state, per_vehicle, per_request = defaultdict(list), defaultdict(list), defaultdict(list)
    for index, (scene, pickup, j) in enumerate(records, len(free_ids)):
        per_state[scene.scenario_id, j].append(index)
        per_vehicle[scene.scenario_id, pickup.vehicle_id].append(index)
        per_request[scene.scenario_id, pickup.request_id].append(index)
    for (_, j), future_columns in per_state.items():
        if j in columns:
            rows.add_sparse((*future_columns, columns[j]), (*([1.] * len(future_columns)), -1.), upper=0.)
        else:
            rows.add_sparse(future_columns, upper=1.)
    for future_columns in per_vehicle.values():
        rows.add_sparse(future_columns, upper=1.)
    for (_, rid), future_columns in per_request.items():
        rows.add_sparse((*future_columns, *(columns[j] for j in by_request.get(rid, ()))), upper=1.)
    critical, immediate, carry, future, eta = (np.zeros(count) for _ in range(5))
    for j, index in columns.items():
        option = options[j]
        if option.request_id is not None:
            request = waiting[option.request_id]
            critical[index], immediate[index], carry[index] = request.critical, 1., request.carry_over
            eta[index] = option.pickup_eta_s
    denominator, numerators = lattice
    for index, (scene, _, _) in enumerate(records, len(free_ids)):
        future[index] = numerators[scene.scenario_id]
    objectives = (("critical_now", critical, True),
                  ("expected_total_service", denominator * immediate + future, True),
                  ("current_service", immediate, True), ("current_carry_over", carry, True),
                  ("current_pickup_eta_s", eta, False))
    objectives = tuple((name, objective, maximize,
                        None if name == "current_pickup_eta_s" else denominator if name == "expected_total_service" else 1)
                       for name, objective, maximize in objectives)
    integer = np.zeros(count, dtype=np.int8)
    integer[:len(free_ids)] = 1
    build_s = perf_counter() - started
    if problem.limits.solver_backend != "SCIPY":
        raise ValueError("city certified numerical contract requires the declared SCIPY backend")
    recovery_s = 0.0
    last_pairs = ()

    def certify(raw, stage):
        nonlocal recovery_s, last_pairs
        budget.remaining("tier integer/matching certification")
        diagnostics = dict(stage=stage, binary_tolerance=MIP_FEASIBILITY_TOLERANCE,
                           primal_row_tolerance=PRIMAL_FEASIBILITY_TOLERANCE)
        raw = np.asarray(raw)
        if raw.shape != (count,) or not np.all(np.isfinite(raw)):
            raise CityNumericalContractError("invalid numerical solution vector", diagnostics)
        binary = np.rint(raw[:len(free_ids)])
        deviation = float(np.max(np.abs(raw[:len(free_ids)] - binary), initial=0.0))
        diagnostics["max_binary_deviation"] = deviation
        if deviation > MIP_FEASIBILITY_TOLERANCE or np.any((binary != 0) & (binary != 1)):
            raise CityNumericalContractError("binary deviation exceeds the explicit solver tolerance", diagnostics)
        raw_violation = _maximum_row_violation(rows, count, raw)
        diagnostics["max_raw_row_violation"] = raw_violation
        bound_violation = max(0.0, float(-raw.min()), float(raw.max() - 1))
        diagnostics["max_raw_bound_violation"] = bound_violation
        if raw_violation > PRIMAL_FEASIBILITY_TOLERANCE or bound_violation > MIP_FEASIBILITY_TOLERANCE:
            raise CityNumericalContractError("raw solution violates declared row/bound tolerances", diagnostics)
        certified = np.zeros(count)
        certified[:len(free_ids)] = binary
        full = np.ones(len(options))
        for j, index in columns.items():
            full[j] = binary[index]
        selected_now = tuple(sorted((options[j].vehicle_id, options[j].request_id) for j, index in columns.items()
                                    if options[j].request_id is not None and binary[index] == 1))
        recovery_started = perf_counter()
        pairs, _ = base._recover_recourse(problem, records, full, selected_now)
        recovery_s += perf_counter() - recovery_started
        active_edges = {(scene.scenario_id, pickup.vehicle_id, pickup.request_id): index
                        for index, (scene, pickup, j) in enumerate(records, len(free_ids)) if full[j] == 1}
        for pair in pairs:
            if pair not in active_edges:
                raise CityNumericalContractError("matching used an inactive state edge", diagnostics)
            certified[active_edges[pair]] = 1
        violation = _maximum_row_violation(rows, count, certified)
        diagnostics.update(max_certified_row_violation=violation,
                           binary_projection_within_explicit_tolerance=True,
                           matching_reconstructed_integral_recourse=True)
        if violation != 0.0:
            raise CityNumericalContractError("projected binary/matching certificate violates a capacity or earlier tier lock", diagnostics)
        last_pairs = pairs
        budget.remaining("tier full-row and objective-lock verification")
        return certified, diagnostics

    partial, optimize_s, stages, receipt = _run_city_levels(rows, count, objectives, integer, certify, budget)
    budget.remaining("component optimality completion")
    selected = tuple(sorted((options[j].vehicle_id, options[j].request_id) for j, index in columns.items()
                            if options[j].request_id is not None and partial[index] > .5))
    pairs = last_pairs
    q_units = sum(numerators[sid] for sid, _, _ in pairs)
    if int(future @ partial) != q_units:
        raise RuntimeError("city matching objective disagrees with its exact scenario lattice")
    q = q_units / denominator
    matrix = rows.matrix(count)
    return dict(selected=selected, pairs=pairs, expected=q, variables=count, integer_count=len(free_ids),
                nonzeros=int(matrix.nnz), matrix_bytes=int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes),
                build_s=build_s, optimize_s=optimize_s, recovery_s=recovery_s, stages=stages, solver_receipt=receipt)


def solve_city_defer(problem, policy="LOCATION_AWARE_DEFER", *, _budget=None):
    """Exact component decomposition for the finite ONE-next-service model.

    Objectives are critical, current plus weighted next service, current,
    carry-over current, then current pickup ETA. Unlike the SP v3 controller,
    no immediate-service face is locked before expected service is optimized.
    The decomposition is exact for this model, not for complete future chains.
    """
    if policy not in POLICIES:
        raise ValueError(f"city defer policy must be one of {POLICIES!r}")
    budget = _budget or _Budget(problem.limits.solver_time_limit_s)
    started = perf_counter()
    problem = _bounded_problem(problem, budget.remaining("sparse model compilation"))
    vehicles, waiting, options, by_v, by_r, recourse = decomposed._compile(problem)
    lattice = _scenario_lattice(problem)
    denominator, numerators = lattice
    states = {(scene.scenario_id, j) for scene, _, j in recourse}
    future_v = {(scene.scenario_id, pickup.vehicle_id) for scene, pickup, _ in recourse}
    future_r = {(scene.scenario_id, pickup.request_id) for scene, pickup, _ in recourse}
    logical_nnz = (len(options) + sum(len(ids) for ids in by_r.values()) + 3 * len(recourse)
                   + len(states) + sum(len(by_r.get(rid, ())) for _, rid in future_r))
    logical_rows = len(by_v) + len(waiting) + len(states) + len(future_v) + len(future_r)
    graph = decomposed._Components()
    free_v = {vid for vid, ids in by_v.items() if len(ids) > 1}
    for vid in sorted(free_v):
        graph.find(("V", vid))
    for option in options:
        if option.request_id is not None:
            graph.join(("V", option.vehicle_id), ("C", option.request_id))
    edge_nodes = []
    for scene, pickup, _ in recourse:
        budget.remaining("exact connectivity decomposition")
        node = ("V", pickup.vehicle_id) if pickup.vehicle_id in free_v else ("FV", scene.scenario_id, pickup.vehicle_id)
        target = ("R", scene.scenario_id, pickup.request_id)
        graph.join(node, target)
        if by_r.get(pickup.request_id):
            graph.join(target, ("C", pickup.request_id))
        edge_nodes.append(node)
    groups = {}
    for vid in sorted(free_v):
        groups.setdefault(graph.find(("V", vid)), dict(vehicles=[], records=[]))["vehicles"].append(vid)
    for record, node in zip(recourse, edge_nodes):
        groups.setdefault(graph.find(node), dict(vehicles=[], records=[]))["records"].append(record)
    build_s = perf_counter() - started
    selected, recovered, stages = [], [], []
    expected = optimize_s = recovery_s = pure_expected = 0.0
    pure_expected_units = 0
    variables = integer_count = nnz = matrix_bytes = pure_count = pure_edges = mip_count = 0
    for component_id, group in enumerate(groups.values()):
        budget.remaining("component allocation")
        records = group["records"]
        if not group["vehicles"]:
            recovery_started = perf_counter()
            pairs, q, memory = decomposed._matching(records)
            budget.remaining("pure-future sparse matching")
            recovery_s += perf_counter() - recovery_started
            recovered.extend(pairs)
            expected += q
            pure_expected += q
            pure_expected_units += sum(numerators[sid] for sid, _, _ in pairs)
            pure_count += 1
            pure_edges += len(records)
            matrix_bytes += memory
            continue
        vids = set(group["vehicles"])
        fixed_ids = {j for _, _, j in records if options[j].vehicle_id not in vids}
        option_ids = [j for j, option in enumerate(options) if option.vehicle_id in vids or j in fixed_ids]
        result = _solve_component(problem, waiting, options, by_v, by_r, option_ids, records, budget, lattice)
        selected.extend(result["selected"])
        recovered.extend(result["pairs"])
        expected += result["expected"]
        variables += result["variables"]
        integer_count += result["integer_count"]
        nnz += result["nonzeros"]
        if nnz > problem.limits.max_nonzeros:
            raise ValueError("city sparse model resource cap exceeded including objective lock rows")
        matrix_bytes += result["matrix_bytes"]
        build_s += result["build_s"]
        optimize_s += result["optimize_s"]
        recovery_s += result["recovery_s"]
        mip_count += 1
        stages.extend(dict(component=component_id, **stage) for stage in result["stages"])
    if len({vid for vid, _ in selected}) != len(selected) or len({rid for _, rid in selected}) != len(selected):
        raise RuntimeError("city common first action duplicates vehicle/request capacity")
    for scene in problem.scenarios:
        pairs = [(vid, rid) for sid, vid, rid in recovered if sid == scene.scenario_id]
        if (len({vid for vid, _ in pairs}) != len(pairs) or len({rid for _, rid in pairs}) != len(pairs)
                or {rid for _, rid in pairs} & {rid for _, rid in selected}):
            raise RuntimeError("city integral recourse duplicated capacity or a current service")
    budget.remaining("city result validation")
    expected = sum(numerators[sid] for sid, _, _ in recovered) / denominator
    pure_expected = pure_expected_units / denominator
    diagnostics = [stage["numerical_diagnostics"] for stage in stages if "numerical_diagnostics" in stage]
    numerical_contract = dict(
        mip_feasibility_tolerance=MIP_FEASIBILITY_TOLERANCE,
        primal_feasibility_tolerance=PRIMAL_FEASIBILITY_TOLERANCE,
        expected_service_lattice_denominator=denominator,
        max_binary_deviation=max((entry["max_binary_deviation"] for entry in diagnostics), default=0.0),
        max_raw_row_violation=max((entry["max_raw_row_violation"] for entry in diagnostics), default=0.0),
        max_certified_row_violation=max((entry["max_certified_row_violation"] for entry in diagnostics), default=0.0),
        integer_tier_locks_are_exact_equalities=True, pickup_eta_is_final_without_float_face_lock=True,
        no_projection_outside_explicit_solver_tolerance=True,
        all_matching_capacity_and_prior_objective_rows_rechecked=True,
        solver_options_receipt=deepcopy(_solver_options_receipt()) if diagnostics else None)
    return CityDeferDecision(
        policy=policy, selected_pairs=tuple(sorted(selected)), immediate_service_count=len(selected),
        expected_next_service_count=expected, expected_total_service_count=len(selected) + expected,
        solve_time_s=optimize_s, variable_count=variables, constraint_nonzeros=nnz,
        sparse_matrix_bytes=matrix_bytes, recourse_pairs=tuple(recovered), integer_variable_count=integer_count,
        model_build_time_s=build_s, recourse_recovery_time_s=recovery_s,
        solver_backend=problem.limits.solver_backend, recourse_mode="FLOW_RELAXED",
        eliminated_fixed_variables=len(options) - integer_count, component_count=len(groups),
        mip_component_count=mip_count, pure_future_component_count=pure_count,
        pure_future_edges=pure_edges, pure_future_expected_count=pure_expected,
        decision_time_s=perf_counter() - budget.started, logical_variable_count=len(options) + len(recourse),
        logical_nonzeros=logical_nnz, logical_rows=logical_rows,
        critical_matched=sum(waiting[rid].critical for _, rid in selected),
        carry_over_matched=sum(waiting[rid].carry_over for _, rid in selected), stages=tuple(stages),
        numerical_contract=numerical_contract,
        provenance=dict(kind=KIND, complete_chain_equivalence=False,
            exact_decomposition_scope="COMPLETE_SUPPLIED_SPARSE_ONE_NEXT_SERVICE_MODEL_ONLY",
            current_service_face_locked=False,
            objective_order=["CRITICAL_NOW", "CURRENT_PLUS_WEIGHTED_NEXT_SERVICE", "CURRENT_SERVICE",
                             "CURRENT_CARRY_OVER", "CURRENT_PICKUP_ETA"],
            future_is_historical_value_proxy_not_execution=True, only_first_action_executed=True,
            next_service_per_vehicle_per_scenario=1, recourse_physical_certificate=False,
            resource_caps_checked_before_decomposition=True, resource_fallback=False,
            total_or_budget_s=budget.seconds, max_variables=MAX_VARIABLES, max_nonzeros=MAX_NONZEROS))


def _checked_reference(reference, cfg):
    if not isinstance(reference, dict):
        raise ValueError("TIME_TYPE_DEFER needs a frozen historical reference object")
    result = deepcopy(reference)
    for field in ("travel_time_s", "empty_distance_m"):
        value = result.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
            raise ValueError(f"generic historical reference {field} must be finite and nonnegative")
    if result["travel_time_s"] > 300:
        raise ValueError("generic historical reference exceeds original 300s serviceable pickup cap")
    result.setdefault("history_dates", list(cfg.get("forecast_train_dates", ())))
    if not result["history_dates"] or any(str(date).replace("-", "") > "20161024" for date in result["history_dates"]):
        raise ValueError("generic reference must be frozen earlier-history information")
    return result


def build_city_future_problem(problem, scenario_inputs, pickup_positions, forecast, cfg, policy,
                              *, generic_reference=None, _budget=None):
    """Spatial known states plus spatial or time/type post-current states.

    Top-K is per state family, not a complete all-state/all-customer graph.
    Both controls share exactly the known WAIT/BUSY spatial layer. Generic
    post-current resource selection ignores endpoint position/direction and is
    stable by earliest effective departure, then a request-specific frozen
    identity hash within the same time layer. No Cartesian distance matrix.
    """
    if policy not in POLICIES:
        raise ValueError("unknown city valuation policy")
    budget = _budget or _Budget(problem.limits.solver_time_limit_s)
    problem = _bounded_problem(problem, budget.remaining("future graph initialization"))
    vehicles, waiting = base._validate(problem)
    options = base._options(problem, vehicles, waiting)
    top_k = cfg["future_top_k_vehicles"]
    radius = cfg["future_search_radius_m"]
    if type(top_k) is not int or top_k <= 0 or not isfinite(radius) or radius <= 0:
        raise ValueError("invalid frozen future sparse-neighborhood settings")
    reference = _checked_reference(generic_reference, cfg) if policy == "TIME_TYPE_DEFER" else None
    known = [option for option in options if option.request_id is None]
    paid = [option for option in options if option.request_id is not None
            and (vehicles[option.vehicle_id].strict_completion_deadline
                 or option.ready_time_s < vehicles[option.vehicle_id].availability_end_s)]

    def geometry(states):
        points = np.stack([cached_xy(option.position) for option in states]) if states else np.empty((0, 2))
        return points, cKDTree(points) if states else None

    known_points, known_tree = geometry(known)
    paid_points, paid_tree = geometry(paid) if policy == "LOCATION_AWARE_DEFER" else (None, None)
    by_profile, paid_by_vehicle = defaultdict(list), defaultdict(list)
    for option in paid:
        by_profile[vehicles[option.vehicle_id].profile_id].append(option)
        paid_by_vehicle[option.vehicle_id].append(option)
    option_order = lambda option: (option.ready_time_s, option.vehicle_id, option.request_id)
    for states in by_profile.values():
        states.sort(key=option_order)
    for states in paid_by_vehicle.values():
        states.sort(key=option_order)
    counters = dict(known_state_arcs=0, after_current_task_arcs=0, generic_proxy_arcs=0,
                    known_state_count=len(known), after_current_task_state_count=len(paid),
                    candidate_top_k_per_state_family=top_k, search_radius_m=radius,
                    generic_candidate_rule="EARLIEST_EFFECTIVE_DEPARTURE_THEN_FROZEN_REQUEST_VEHICLE_HASH_TOP_K",
                    generic_hash_seed=cfg.get("forecast_seed", 0),
                    candidate_rule_is_an_additional_top_k_approximation=True)

    def feasible(option, request, eta):
        vehicle = vehicles[option.vehicle_id]
        if option.request_id == request.request_id or not base._allowed(vehicle, request):
            return False
        departure = max(problem.now_s + problem.limits.rolling_step_s,
                        option.ready_time_s, request.release_time_s)
        completion = departure + eta + request.pickup_overhead_s + request.predicted_service_time_s
        return (departure + eta <= request.pickup_deadline_s
                and base._admission_feasible(vehicle, departure, completion, completion_tolerance_s=0))

    def spatial(request, target_position, states, points, tree):
        if tree is None:
            return ()
        point = cached_xy(target_position)
        candidates = []
        for index in tree.query_ball_point(point, radius):
            budget.remaining("spatial sparse future candidate construction")
            option = states[index]
            distance = float(np.linalg.norm(points[index] - point))
            departure = max(problem.now_s + problem.limits.rolling_step_s, option.ready_time_s, request.release_time_s)
            pace = forecast.pace_by_slot.get(int(departure // 1800), forecast.global_pace)
            if not isfinite(pace) or pace <= 0:
                raise ValueError("historical M3 chord pace unavailable")
            eta = distance * pace
            if feasible(option, request, eta):
                candidates.append((distance, option.vehicle_id, -1 if option.request_id is None else option.request_id,
                                   option, eta))
        candidates.sort(key=lambda row: row[:3])
        chosen = set()
        for _, vid, _, _, _ in candidates:
            if len(chosen) < top_k:
                chosen.add(vid)
        return tuple(base.FuturePickup(option.vehicle_id, option.request_id, request.request_id, eta, option.position)
                     for _, vid, _, option, eta in candidates if vid in chosen)

    def generic(request, scenario_id):
        eta = reference["travel_time_s"]
        eligible = [states for profile, states in by_profile.items()
                    if profile == "HV" or (request.passenger_accepts_av and profile in request.compatible_profiles)]
        candidates = {}
        for option in merge(*eligible, key=option_order):
            budget.remaining("generic time/type Top-K resource selection")
            if option.ready_time_s > request.pickup_deadline_s - eta:
                break
            if option.vehicle_id in candidates or not feasible(option, request, eta):
                continue
            departure = max(problem.now_s + problem.limits.rolling_step_s,
                            option.ready_time_s, request.release_time_s)
            token = f'{cfg.get("forecast_seed", 0)}|{scenario_id}|{request.request_id}|{option.vehicle_id}'
            identity_rank = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little")
            candidates[option.vehicle_id] = (departure, identity_rank, option.vehicle_id)
        # Effective departure, not raw ready time, is the relevant time class:
        # all states ready before release can wait to the same departure. A
        # frozen per-request hash distributes these interchangeable resources
        # instead of giving every future request the same five vehicle IDs.
        chosen = [vid for _, _, vid in sorted(candidates.values())[:top_k]]
        result = []
        for vid in chosen:
            for option in paid_by_vehicle[vid]:
                budget.remaining("generic conditional post-current state construction")
                if feasible(option, request, eta):
                    result.append(base.FuturePickup(vid, option.request_id, request.request_id, eta, option.position))
        return tuple(result)

    scenarios, total_future = [], 0
    for scene, forecast_positions in scenario_inputs:
        arcs = []
        positions = {**pickup_positions, **forecast_positions}
        for request in (*problem.waiting_requests, *scene.new_requests):
            budget.remaining("scenario sparse future graph")
            known_arcs = spatial(request, positions[request.request_id], known, known_points, known_tree)
            paid_arcs = (generic(request, scene.scenario_id) if reference is not None else
                         spatial(request, positions[request.request_id], paid, paid_points, paid_tree))
            arcs.extend(known_arcs)
            arcs.extend(paid_arcs)
            counters["known_state_arcs"] += len(known_arcs)
            counters["after_current_task_arcs"] += len(paid_arcs)
            counters["generic_proxy_arcs"] += len(paid_arcs) if reference is not None else 0
            if len(options) + total_future + len(arcs) > problem.limits.max_variables:
                raise ValueError("city variable resource cap exceeded during sparse future graph construction")
        total_future += len(arcs)
        scenarios.append(replace(scene, pickups=tuple(arcs)))
    budget.remaining("future graph completion")
    counters.update(provenance=PROXY_PROVENANCE if reference is not None else "HISTORICAL_M3_CHORD_PACE_SPATIAL_VALUE_PROXY",
                    generic_reference=deepcopy(reference), no_physical_recourse_certificate=True,
                    known_wait_busy_locations_and_estimates_shared=True)
    return replace(problem, scenarios=tuple(scenarios)), counters


def _first_arc_filter_summary(callback):
    """Allowlisted aggregate physical checks; never copy identities/indices."""
    diagnostics = getattr(callback, "last_diagnostics", None)
    if diagnostics is None:
        diagnostics = getattr(getattr(callback, "__self__", None), "last_diagnostics", None)
    if not isinstance(diagnostics, Mapping):
        return {}
    summary = {}
    for field in ("input_arc_count", "validated_arc_count", "pickup_support_time_s", "pickup_routing_time_s",
                  "pickup_seam_evidence_time_s", "routing_budget_s", "routing_budget_is_separate_from_solver",
                  "geometry_timestamp", "decision_timestamp", "eta_beta", "route_cache_entries",
                  "retained_pickup_path_count"):
        if field in diagnostics and isinstance(diagnostics[field], (str, bool, int, float)):
            summary[field] = diagnostics[field]
    for field in ("counts", "rejection_reason_counts"):
        values = diagnostics.get(field)
        if isinstance(values, Mapping):
            summary[field] = {key: int(value) for key, value in values.items()
                              if isinstance(key, str) and isinstance(value, (int, float)) and isfinite(value)}
    return summary


class CityDeferNativeAdapter(NativeFlexibilityAdapter):
    """Native return-index adapter with explicit failures and no fallback.

    current_arc_filter(control, original_arcs, now) returns allowed AssignmentArc
    objects, possibly replacements with validated ETA/payload. Replacements are
    written back to original_arcs at their original pair indices so native
    _assign receives the validated payload. Physical filtering is separately
    timed; the 10s OR budget begins after it finishes.
    """
    def __init__(self, policy, forecast, cfg, *, generic_reference=None, current_arc_filter=None, remaining_model=None):
        if policy not in POLICIES:
            raise ValueError("unknown city defer control")
        super().__init__(policy, forecast, deepcopy(cfg), remaining_model)
        self.generic_reference = (_checked_reference(generic_reference, cfg) if policy == "TIME_TYPE_DEFER" else None)
        self.current_arc_filter = current_arc_filter
        self.last_decision = None
        self.future_diagnostics = {}

    def _problem(self, c, arcs, waiting_ids, now):
        # The custom policy names intentionally make the frozen parent build
        # only its common observable current state, not its old future graph.
        current = super()._problem(c, arcs, waiting_ids, now)
        sampling_started = perf_counter()
        scenarios = self.forecast.scenarios(now, c.config["profile_id"], c.acceptance_rate, c.acceptance_seed)
        self.stage_timings["forecast_sampling_time_s"] = perf_counter() - sampling_started
        self._or_budget.remaining("historical forecast sampling")
        pickup_positions = {request.request_id: position_key(c.request_by_rid[request.request_id].pickup_lon_wgs84,
                            c.request_by_rid[request.request_id].pickup_lat_wgs84) for request in current.waiting_requests}
        graph_started = perf_counter()
        problem, self.future_diagnostics = build_city_future_problem(current, scenarios, pickup_positions,
            self.forecast, self.cfg, self.policy, generic_reference=self.generic_reference, _budget=self._or_budget)
        self.stage_timings["future_graph_time_s"] = perf_counter() - graph_started
        return problem

    def solve(self, c, original_arcs, waiting_ids, now):
        physical_started = perf_counter()
        original_indices = {(arc.vehicle_id, arc.request_id): index for index, arc in enumerate(original_arcs)}
        if len(original_indices) != len(original_arcs):
            raise ValueError("duplicate native current arc identity")
        filtered = list(original_arcs if self.current_arc_filter is None else self.current_arc_filter(c, original_arcs, now))
        seen = set()
        for arc in filtered:
            key = arc.vehicle_id, arc.request_id
            if key not in original_indices or key in seen:
                raise ValueError("physical first-arc filter introduced an unknown or duplicate native pair")
            seen.add(key)
            if self.current_arc_filter is not None:
                original_arcs[original_indices[key]] = arc
        physical_s = perf_counter() - physical_started
        physical_summary = _first_arc_filter_summary(self.current_arc_filter)
        self._or_budget = _Budget(self.cfg["solver_time_limit_s"])
        self.stage_timings = dict(forecast_sampling_time_s=0., future_graph_time_s=0., problem_setup_time_s=0.)
        self.future_diagnostics = {}
        self.busy_diagnostics = {}
        preparation_started = perf_counter()
        problem = self._problem(c, filtered, waiting_ids, now)
        self.stage_timings["problem_setup_time_s"] = max(0., perf_counter() - preparation_started
            - self.stage_timings["forecast_sampling_time_s"] - self.stage_timings["future_graph_time_s"])
        decision = solve_city_defer(problem, self.policy, _budget=self._or_budget)
        selected = set(decision.selected_pairs)
        if not selected <= seen:
            raise RuntimeError("city solver returned an unvalidated native first arc")
        indices = tuple(sorted(original_indices[key] for key in selected))
        chosen = [original_arcs[index] for index in indices]
        self._or_budget.remaining("native first-action index mapping")
        self.last_decision = decision
        result = LexicographicResult(indices, perf_counter() - self._or_budget.started,
            decision.critical_matched, len(chosen), decision.carry_over_matched,
            backend="CITY_SPARSE_ONE_NEXT_SERVICE_DEFER_NO_FALLBACK",
            pickup_eta_optimum_s=sum(arc.pickup_eta_s for arc in chosen))
        self.rows.append(dict(simulation_time_s=now, policy=self.policy,
            current_arc_count=len(original_arcs), validated_current_arc_count=len(filtered),
            current_selected=len(chosen), physical_validation_time_s=physical_s,
            first_arc_filter_diagnostics=physical_summary,
            physical_first_arc_validation_source=("CURRENT_ARC_FILTER" if self.current_arc_filter is not None
                                                  else "CALLER_SUPPLIED_CURRENT_ARCS"),
            solver_time_s=result.solve_time_s, resource_fallback=None, opt_status="OPTIMAL",
            variable_count=decision.logical_variable_count, solver_variable_count=decision.variable_count,
            nonzeros=decision.logical_nonzeros, solver_nonzeros=decision.constraint_nonzeros,
            expected_next_service=decision.expected_next_service_count,
            expected_total_service=decision.expected_total_service_count,
            integer_variable_count=decision.integer_variable_count,
            eliminated_fixed_variables=decision.eliminated_fixed_variables,
            decomposed_component_count=decision.component_count,
            pure_future_component_count=decision.pure_future_component_count,
            pure_future_edges=decision.pure_future_edges, pure_future_expected_count=decision.pure_future_expected_count,
            model_build_time_s=decision.model_build_time_s, optimization_time_s=decision.solve_time_s,
            recourse_recovery_time_s=decision.recourse_recovery_time_s,
            numerical_contract=decision.numerical_contract,
            provenance=decision.provenance, future_graph=self.future_diagnostics,
            **self.stage_timings, **self.busy_diagnostics))
        return result
