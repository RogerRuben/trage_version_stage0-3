"""Finite Scheme-A joint layout/current-action and multi-service-chain model.

The complete supplied constant-time chain domain is reused from the earlier
prototype. This is an additive solver, not a city-scale rollout, new prediction
model, or new capability policy. Historical scenario recourse values the common
first action; recourse service is never counted as physical realized service.

LAYOUT and RELOCATE are common first actions. Independent relocations between
future customer services are NOT in this finite recourse domain.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
import math
from time import perf_counter
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from . import capability_chain_rolling_solver as domain


KIND = "SCHEME_A_COMPLETE_SUPPLIED_FINITE_CHAINS_COMMON_FIRST_ACTION"
POLICIES = domain.POLICIES
NUMERICAL_TOLERANCE = 1e-9
_EXACT_DOUBLE_INTEGER_LIMIT = 2**53


class JointNumericalContractError(RuntimeError):
    """A numerical solve did not support its advertised finite-model result."""


@lru_cache(maxsize=1)
def _solver_options_receipt():
    # Verify the actual bundled OptionsManager/passOptions/readback path, not
    # merely the presence of an option in a Python dictionary.
    from scipy.optimize._highspy import _core, _highs_options

    probe, options = _core._Highs(), _core.HighsOptions()
    manager = _highs_options.HighsOptionsManager()
    options.output_flag = False
    requested = dict(mip_feasibility_tolerance=NUMERICAL_TOLERANCE,
                     primal_feasibility_tolerance=NUMERICAL_TOLERANCE,
                     dual_feasibility_tolerance=NUMERICAL_TOLERANCE,
                     mip_abs_gap=0.0)
    for key, value in requested.items():
        if manager.get_option_type(key) == -1:
            raise JointNumericalContractError(f"bundled HiGHS does not recognize {key}")
        setattr(options, key, value)
    if probe.passOptions(options) != _core.HighsStatus.kOk:
        raise JointNumericalContractError("bundled HiGHS rejected the numerical options")
    for key, value in requested.items():
        status, actual = probe.getOptionValue(key)
        if status != _core.HighsStatus.kOk or actual != value:
            raise JointNumericalContractError(f"bundled HiGHS option readback disagrees: {key}")
    return dict(highs_version=probe.version(), options=requested,
                bundled_option_readback_verified=True)


def _lattice(values, rows):
    """Return a primitive rational objective grid only when safely certified."""
    nonzero = [value for value in values if value]
    if not nonzero:
        return np.zeros(len(values)), Fraction(1), 0.0
    denominator = math.lcm(*(value.denominator for value in nonzero))
    integers = [int(value * denominator) for value in values]
    divisor = math.gcd(*(abs(value) for value in integers if value))
    integers = [value // divisor for value in integers]
    coefficient_sum = sum(abs(value) for value in integers)
    maximum = max(abs(value) for value in integers)
    # Both floating objective error and integrality/row tolerances must be too
    # small to hide a distinct integer objective value.
    error = (NUMERICAL_TOLERANCE * coefficient_sum
             + NUMERICAL_TOLERANCE * len(rows.lower) * maximum
             + 64 * np.finfo(float).eps * max(1, coefficient_sum))
    if coefficient_sum >= _EXACT_DOUBLE_INTEGER_LIMIT or error >= .25:
        return None
    return np.asarray(integers, dtype=float), Fraction(denominator, divisor), error


def _violation(matrix, rows, vector):
    lhs = matrix @ vector
    return max(0.0, float(np.max(np.asarray(rows.lower) - lhs, initial=0.0)),
               float(np.max(lhs - np.asarray(rows.upper), initial=0.0)))


def _run_milp(objective, rows, budget, receipt, name):
    matrix = rows.matrix()
    options = dict(receipt["options"], time_limit=budget.remaining(name),
                   mip_rel_gap=0.0, presolve=True)
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = milp(objective, integrality=np.ones(rows.columns, dtype=np.int8),
                      bounds=Bounds(np.zeros(rows.columns), np.ones(rows.columns)),
                      constraints=LinearConstraint(matrix, rows.lower, rows.upper), options=options)
    for warning in caught:
        message = str(warning.message)
        if not ("Unrecognized options detected" in message and "passed to HiGHS verbatim" in message):
            raise JointNumericalContractError(f"{name}: solver options warning: {message}")
    elapsed = perf_counter() - started
    if result.status == 1:
        error = domain.DecisionTimeout(f"{name}: not proven optimal within the common decision budget")
        def finite_result(field):
            value = getattr(result, field, None)
            return float(value) if value is not None and math.isfinite(float(value)) else None
        error.solver_diagnostics = dict(stage=name, solver_status=int(result.status),
            solver_message=str(result.message), effective_total_budget_s=getattr(budget, "seconds", domain.DECISION_TIME_LIMIT_S),
            stage_time_limit_s=options["time_limit"], stage_elapsed_s=elapsed,
            columns=rows.columns, nonzeros=int(matrix.nnz), incumbent_returned=result.x is not None,
            incumbent_objective=finite_result("fun"), dual_bound=finite_result("mip_dual_bound"),
            mip_gap=finite_result("mip_gap"))
        raise error
    if result.status != 0 or not result.success or result.x is None:
        raise RuntimeError(f"{name}: finite sparse MILP did not reach optimality: {result.message}")
    budget.remaining(name)
    raw = np.asarray(result.x)
    if raw.shape != (rows.columns,) or not np.all(np.isfinite(raw)):
        raise JointNumericalContractError(f"{name}: invalid numerical solution vector")
    binary = np.rint(raw)
    deviation = float(np.max(np.abs(raw - binary), initial=0.0))
    raw_violation = _violation(matrix, rows, raw)
    bound_violation = max(0.0, float(-raw.min()), float(raw.max() - 1))
    if (deviation > NUMERICAL_TOLERANCE or raw_violation > NUMERICAL_TOLERANCE
            or bound_violation > NUMERICAL_TOLERANCE or np.any((binary != 0) & (binary != 1))):
        raise JointNumericalContractError(
            f"{name}: raw numerical contract failed: binary={deviation:g}, "
            f"row={raw_violation:g}, bound={bound_violation:g}")
    # All constraint coefficients, including service objective locks, are
    # exact integers within the double sum range. No epsilon face is used.
    certified_violation = _violation(matrix, rows, binary)
    if certified_violation != 0:
        raise JointNumericalContractError(f"{name}: projected integer certificate violates an exact row")
    return result, binary, dict(runtime_s=elapsed, max_binary_deviation=deviation,
                               max_raw_row_violation=raw_violation,
                               max_raw_bound_violation=bound_violation,
                               max_certified_row_violation=certified_violation,
                               rounding_tolerance=NUMERICAL_TOLERANCE)


def _action_tie(model, action_columns, size):
    groups = [[action for action in model.actions if action.resource_id == resource.resource_id]
              for resource in model.resources]
    radices = [len(group) for group in groups]
    multipliers = [math.prod(radices[index + 1:]) for index in range(len(radices))]
    maximum = math.prod(radices) - 1
    if maximum >= _EXACT_DOUBLE_INTEGER_LIMIT:
        raise domain.RollingModelLimitError("stable first-action code exceeds exact integer range")
    values = [Fraction(0)] * size
    for group, multiplier in zip(groups, multipliers):
        for rank, action in enumerate(group):
            values[action_columns[action.action_id]] = Fraction(rank * multiplier)
    return values, dict(kind="EXACT_MIXED_RADIX_FIRST_ACTION_RANKS", tie_only=True,
                        resource_order=[resource.resource_id for resource in model.resources],
                        action_order=[[action.action_id for action in group] for group in groups],
                        radices=radices, multipliers=multipliers, maximum_code=maximum)


def _selection(model, paths, rows, action_columns, policy, budget, receipt):
    coefficients = domain._objectives(model, paths)
    order = (["critical_now", "current_served", "carry_over_current_served", "expected_served"]
             if policy == "SERVICE_PRESERVING" else
             ["critical_now", "expected_served", "current_served", "carry_over_current_served"])
    stages, fixed = [], []
    selected = None
    bound_rows = None
    value_by_name = {}

    def stage(name, values, maximize, *, final_float=False, force=False):
        nonlocal selected
        budget.remaining(name)
        if not any(values) and not force:
            value_by_name[name] = Fraction(0)
            stages.append(dict(stage=name, value=0.0, value_exact="0", direction="MAX" if maximize else "MIN",
                               opt_status="CONSTANT_EXACT", runtime_s=0.0, exact_objective_lock=False))
            return
        grid = _lattice(values, rows)
        if grid is None and not final_float:
            raise JointNumericalContractError(f"{name}: objective has no safely certifiable integer grid")
        if grid is None:
            objective = np.asarray([float(value) for value in values])
            scale = None
            coefficient_sum = float(np.sum(np.abs(objective)))
            maximum = float(np.max(np.abs(objective), initial=0.0))
            error = (NUMERICAL_TOLERANCE * coefficient_sum
                     + NUMERICAL_TOLERANCE * len(rows.lower) * maximum
                     + 64 * np.finfo(float).eps * max(1., coefficient_sum))
        else:
            objective, scale, error = grid
        result, binary, diagnostics = _run_milp(-objective if maximize else objective,
                                              rows, budget, receipt, name)
        selected = tuple(int(index) for index in np.flatnonzero(binary))
        for previous_name, previous_values, previous_value in fixed:
            actual = sum((previous_values[index] for index in selected), Fraction(0))
            if actual != previous_value:
                raise JointNumericalContractError(f"{name}: changed exact prior objective {previous_name}")
        value = sum((values[index] for index in selected), Fraction(0))
        value_by_name[name] = value
        native_value = float(-result.fun if maximize else result.fun)
        selected_objective = float(objective @ binary)
        if abs(native_value - selected_objective) > error:
            raise JointNumericalContractError(f"{name}: objective disagrees with its integer selection")
        grid_units = None
        if scale is not None:
            exact_units = value * scale
            if exact_units.denominator != 1 or selected_objective != int(exact_units):
                raise JointNumericalContractError(f"{name}: objective selection left the exact lattice")
            grid_units = int(exact_units)
            minimum = -grid_units if maximize else grid_units
            bound = getattr(result, "mip_dual_bound", None)
            if bound is None or not math.isfinite(bound) or math.ceil(float(bound) - error) < minimum:
                raise JointNumericalContractError(f"{name}: optimal status does not prove its integer lattice value")
            indices = np.flatnonzero(objective)
            rows.add([(int(index), objective[index]) for index in indices], grid_units, grid_units)
            fixed.append((name, values, value))
        diagnostics.update(native_objective_value=native_value, certified_objective_value=selected_objective,
                           objective_numeric_error_bound=error)
        stages.append(dict(stage=name, value=float(value), value_exact=str(value),
                           direction="MAX" if maximize else "MIN", opt_status="OPTIMAL",
                           numerical_diagnostics=diagnostics, status=int(result.status),
                           runtime_s=diagnostics["runtime_s"], mip_gap=float(result.mip_gap or 0),
                           mip_node_count=int(result.mip_node_count or 0),
                           objective_scale_exact=str(scale) if scale is not None else None,
                           grid_units=grid_units, exact_objective_lock=scale is not None,
                           distance_lattice_optimality_proved=scale is not None if final_float else None))
        budget.remaining(f"{name} certificate")

    for name in order:
        # This LP conditions on the integer-optimal tiers PRECEDING expected
        # service. It does not bind the expected-service optimum itself.
        if name == "expected_served":
            bound_rows = deepcopy(rows)
        stage(name, coefficients[name], True)
    distance_values = coefficients["expected_empty_distance_m"]
    distance_grid = _lattice(distance_values, rows)
    stage("expected_empty_distance_m", distance_values, False, final_float=True)
    if distance_grid is not None:
        tie_values, tie_encoding = _action_tie(model, action_columns, rows.columns)
        stage("stable_first_actions:mixed_radix", tie_values, False)
    else:
        # High-precision distance is the FINAL floating objective. No extra
        # solve is constrained to a fabricated +/-epsilon distance face.
        tie_encoding = dict(kind="FINAL_DISTANCE_OPTIMAL_SOLVER_SELECTION_NO_ADDITIONAL_TIE_SOLVE",
                            tie_only=True, floating_distance_lock=False,
                            exact_distance_tie_order_claimed=False)
    if selected is None:
        stage("feasibility", [Fraction(0)] * rows.columns, False, force=True)
    return selected, stages, bound_rows, value_by_name, tie_encoding


def _exact_lp_dual_upper(result, matrix, rows, objective, scale, budget):
    """Certify a service upper bound using an exact rational feasible LP dual.

    For min c'x, 0<=x<=1, each y_ub<=0 and unrestricted y_eq yields
    b'y + sum_i min(0, c_i - A_i'y) <= optimal objective. Binary IEEE dual
    values are converted to one exact power-of-two lattice. The constraint
    matrix and the service objective are integers; Python integer arithmetic
    certifies the bound without trusting an approximately feasible dual.
    """
    lower, upper = np.asarray(rows.lower), np.asarray(rows.upper)
    eq_ids = np.flatnonzero(lower == upper)
    upper_ids = np.flatnonzero(np.isfinite(upper) & (lower != upper))
    lower_ids = np.flatnonzero(np.isfinite(lower) & (lower != upper))
    eq_y = np.asarray(result.eqlin.marginals)
    ub_y = np.minimum(np.asarray(result.ineqlin.marginals), 0.0)
    if not np.all(np.isfinite(eq_y)) or not np.all(np.isfinite(ub_y)):
        raise JointNumericalContractError("LP has nonfinite dual multipliers")
    # linprog rows are [finite upper rows, negated finite lower rows].
    multipliers = [float(value).as_integer_ratio() for value in np.concatenate((eq_y, ub_y))]
    denominator = max((den for _, den in multipliers), default=1)
    weights = [num * (denominator // den) for num, den in multipliers]
    signed_row_weights = [0] * matrix.shape[0]
    dual_constant = 0
    position = 0
    for index in eq_ids:
        signed_row_weights[int(index)] += weights[position]
        dual_constant += int(upper[index]) * weights[position]
        position += 1
    for index in upper_ids:
        signed_row_weights[int(index)] += weights[position]
        dual_constant += int(upper[index]) * weights[position]
        position += 1
    for index in lower_ids:
        signed_row_weights[int(index)] -= weights[position]
        dual_constant -= int(lower[index]) * weights[position]
        position += 1
    reduced_bound = 0
    for column in range(matrix.shape[1]):
        if column % 256 == 0:
            budget.remaining("exact sparse LP dual certificate")
        coefficient = objective[column]
        if coefficient != int(coefficient):
            raise JointNumericalContractError("LP service coefficient is not an exact integer")
        reduced = -int(coefficient) * denominator
        for ptr in range(matrix.indptr[column], matrix.indptr[column + 1]):
            datum = matrix.data[ptr]
            if datum != int(datum):
                raise JointNumericalContractError("LP constraint coefficient is not an exact integer")
            reduced -= int(datum) * signed_row_weights[int(matrix.indices[ptr])]
        reduced_bound += min(0, reduced)
    return -Fraction(dual_constant + reduced_bound, denominator) / scale


def _service_bound(model, paths, rows, optimal, policy, budget):
    values = domain._objectives(model, paths)["expected_served"]
    grid = _lattice(values, rows)
    if grid is None:
        raise JointNumericalContractError("LP bound service grid is not safely representable")
    objective, scale, _ = grid
    matrix = rows.matrix()
    lower, upper = np.asarray(rows.lower), np.asarray(rows.upper)
    eq_ids = np.flatnonzero(lower == upper)
    upper_ids = np.flatnonzero(np.isfinite(upper) & (lower != upper))
    lower_ids = np.flatnonzero(np.isfinite(lower) & (lower != upper))
    from scipy.sparse import vstack
    inequality = vstack((matrix[upper_ids], -matrix[lower_ids]), format="csc")
    rhs = np.concatenate((upper[upper_ids], -lower[lower_ids]))
    started = perf_counter()
    result = linprog(-objective, A_ub=inequality, b_ub=rhs,
                     A_eq=matrix[eq_ids], b_eq=upper[eq_ids], bounds=(0., 1.), method="highs",
                     options=dict(time_limit=budget.remaining("finite-domain LP service upper bound"),
                                  primal_feasibility_tolerance=NUMERICAL_TOLERANCE,
                                  dual_feasibility_tolerance=NUMERICAL_TOLERANCE,
                                  ipm_optimality_tolerance=NUMERICAL_TOLERANCE))
    if result.status == 1:
        raise domain.DecisionTimeout("LP service bound was not solved within the common decision budget")
    if result.status != 0 or not result.success or result.x is None:
        raise RuntimeError(f"finite-domain LP service bound failed: {result.message}")
    budget.remaining("finite-domain LP service certificate")
    raw = np.asarray(result.x)
    violation = _violation(matrix, rows, raw)
    if violation > NUMERICAL_TOLERANCE or raw.min() < -NUMERICAL_TOLERANCE or raw.max() > 1 + NUMERICAL_TOLERANCE:
        raise JointNumericalContractError("LP returned a solution outside its explicit row/bound tolerances")
    exact_upper = _exact_lp_dual_upper(result, matrix, rows, objective, scale, budget)
    service = optimal["expected_served"]
    if exact_upper < service:
        raise JointNumericalContractError("exact LP dual upper bound is below the feasible integer solution")
    # Outward rounding is part of the *bound display*, never a tier-lock slack.
    upper_float = float(exact_upper)
    if Fraction(upper_float) < exact_upper:
        upper_float = float(np.nextafter(upper_float, np.inf))
    conditioned = ["critical_now"]
    if policy == "SERVICE_PRESERVING":
        conditioned.extend(("current_served", "carry_over_current_served"))
    return dict(status="OPTIMAL_LP_WITH_EXACT_RATIONAL_DUAL_UPPER_CERTIFICATE",
                LP_upper_bound=upper_float, LP_upper_bound_exact=str(exact_upper),
                lp_primal_expected_service=float(-result.fun / float(scale)),
                integer_optimal_expected_service=float(service), integer_optimal_expected_service_exact=str(service),
                integer_to_bound_gap=float(exact_upper - service),
                bound_scope="COMPLETE_SUPPLIED_FINITE_CONSTANT_TIME_CHAIN_DOMAIN_CONDITIONED_ON_PRECEDING_INTEGER_TIERS",
                conditioned_tiers={name: float(optimal[name]) for name in conditioned},
                expected_service_optimal_face_not_locked=True,
                distance_and_action_tie_faces_not_locked=True,
                exact_dual_arithmetic=True, sparse=True,
                lp_rows=matrix.shape[0], lp_columns=matrix.shape[1], lp_nonzeros=matrix.nnz,
                max_lp_primal_row_violation=violation, runtime_s=perf_counter() - started,
                online_or_citywide_optimality_claimed=False)


def solve_joint_epoch(problem, policy="CHAIN_DEFER", *, include_bound=True):
    """Jointly choose a common action and complete finite scenario chains.

    Preparation, exhaustive/exact-compressed enumeration, every MILP tier, the
    optional LP bound, and independent reconstruction share one total 10-s
    budget. Limits are inherited unchanged: <=3 resources, <=2 scenarios,
    <=64 tasks/scenario, 20k variables and 150k sparse nonzeros.
    """
    budget = domain._Budget()
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES!r}")
    if type(include_bound) is not bool:
        raise ValueError("include_bound must be a boolean")
    model = domain._prepare(problem)
    receipt = _solver_options_receipt()
    preparation_s = perf_counter() - budget.started
    budget.remaining("finite model preparation")
    started = perf_counter()
    paths, chain_counts, compression = domain._enumerate(model, budget)
    enumeration_s = perf_counter() - started
    started = perf_counter()
    rows, action_columns = domain._master(model, paths)
    base_rows, base_nonzeros = len(rows.lower), len(rows.values)
    master_s = perf_counter() - started
    budget.remaining("finite sparse master")
    started = perf_counter()
    chosen, stages, bound_rows, optimal, tie_encoding = _selection(
        model, paths, rows, action_columns, policy, budget, receipt)
    optimization_s = perf_counter() - started
    action_count = len(model.actions)
    selected_actions = [model.actions[index] for index in chosen if index < action_count]
    selected_paths = [paths[index - action_count] for index in chosen if index >= action_count]
    scenarios = {scenario.scenario_id: scenario for scenario in model.scenarios}
    expected = sum((scenarios[path.scenario_id].weight * path.served for path in selected_paths), Fraction(0))
    distance = sum((scenarios[path.scenario_id].weight * path.distance for path in selected_paths), Fraction(0))
    quality = (_service_bound(model, paths, bound_rows, optimal, policy, budget) if include_bound else
               dict(status="NOT_REQUESTED", LP_upper_bound=None,
                    integer_optimal_expected_service=float(expected), integer_optimal_expected_service_exact=str(expected),
                    bound_scope="COMPLETE_SUPPLIED_FINITE_CONSTANT_TIME_CHAIN_DOMAIN",
                    online_or_citywide_optimality_claimed=False))
    quality.update(complete_supplied_domain=True,
                   integer_service_tiers_exactly_certified=True,
                   total_decision_budget_s=domain.DECISION_TIME_LIMIT_S,
                   all_preparation_mip_lp_certification_share_budget=True)
    solution = dict(
        kind=KIND, policy=policy, opt_status="OPTIMAL", selected_actions=[deepcopy(action.raw) for action in selected_actions],
        expected_served=float(expected), critical_now=sum(action.kind == "SERVE" and action.critical for action in selected_actions),
        current_served=sum(action.kind == "SERVE" for action in selected_actions),
        carry_over_current_served=sum(action.kind == "SERVE" and action.carry_over for action in selected_actions),
        expected_empty_distance_m=float(distance),
        actual_first_empty_distance_m=float(sum((action.distance for action in selected_actions), Fraction(0))),
        recourse_paths=[domain._path_output(path, scenarios[path.scenario_id]) for path in selected_paths],
        quality_certificate=quality, numerical_contract=receipt,
        model=dict(rows=len(rows.lower), columns=rows.columns, cols=rows.columns,
                   nonzeros=len(rows.values), nnz=len(rows.values), sparse=True,
                   base_rows=base_rows, base_nonzeros=base_nonzeros,
                   first_action_variables=action_count, scenario_chain_variables=len(paths),
                   stable_first_actions_encoding=tie_encoding, chains_per_resource_scenario=chain_counts,
                   complete_enumeration=True, complete_finite_domain=True,
                   task_limit_per_scenario=model.task_limit, hard_task_limit_per_scenario=domain.HARD_MAX_TASKS_PER_SCENARIO,
                   default_task_limit_per_scenario=domain.MAX_TASKS_PER_SCENARIO,
                   exact_chain_compression=model.exact_chain_compression,
                   all_task_sequences_enumerated=not model.exact_chain_compression,
                   complete_enumeration_scope=("EXACT_EQUIVALENT_PREFIX_AND_MASK_COLUMN_REPRESENTATION"
                                               if model.exact_chain_compression else "ALL_FEASIBLE_TASK_SEQUENCES"),
                   compression_statistics=compression,
                   max_variables=domain.MAX_MASTER_VARIABLES, max_nonzeros=domain.MAX_MASTER_NONZEROS,
                   max_chains_per_resource_scenario=domain.MAX_CHAINS_PER_RESOURCE_SCENARIO),
        stages=stages, timing_s=dict(preparation=preparation_s, enumeration=enumeration_s,
                                    sparse_master=master_s, lexicographic_milp=optimization_s,
                                    optional_lp_and_certificate=quality.get("runtime_s", 0.0)),
        model_domain=dict(
            scope="COMPLETE_SUPPLIED_FINITE_SUPPORTED_CONSTANT_TIME_SCENARIOS_NOT_ONLINE_OPTIMUM",
            complete_finite_domain=True, historical_scenarios_are_forecast_proxies=True,
            revealed_future_recourse_is_a_two_stage_value_approximation=True,
            only_common_first_actions_are_executed=True, planned_recourse_is_not_realized_service=True,
            layout_and_relocation_are_common_first_actions=True,
            independent_relocation_between_future_customers_represented=False,
            deadline_is_original_and_never_reset=True, future_commitment_requires_release=True,
            resource_profile_is_fixed=True, committed_service_may_finish_after_admission_end=True,
            missing_or_unsupported_connections_do_not_prove_unroutability=True,
            exact_chain_compression=model.exact_chain_compression,
            compression_is_equivalent_column_and_prefix_dominance_not_top_k=True,
            decision_time_limit_s=domain.DECISION_TIME_LIMIT_S))
    started = perf_counter()
    solution["validation"] = domain.validate_epoch_solution(problem, solution)
    solution["timing_s"]["independent_validation"] = perf_counter() - started
    budget.remaining("independent selected-chain reconstruction")
    solution["runtime_s"] = solution["timing_s"]["total"] = perf_counter() - budget.started
    return solution
