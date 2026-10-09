"""Sparse Scheme-A master over supplied typed multi-service chain columns.

The caller builds chronological/profile-certified chain candidates. This
master preserves their customer identities and common first action, but does
not claim complete chain enumeration, a citywide optimum, or a global bound.
Empty continuation is implicit. Recourse moves consume shared 900-s movement
slots; they never contribute to service rewards.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
import math
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix

from . import scheme_a_joint_solver as contract


POLICIES = ("CHAIN_DEFER", "SERVICE_PRESERVING")
KINDS = frozenset(("SERVE", "WAIT", "BUSY", "RELOCATE", "LAYOUT"))
MAX_VARIABLES = 20_000
MAX_NONZEROS = 150_000
DECISION_TIME_LIMIT_S = 10.0


@dataclass(frozen=True)
class CurrentAction:
    action_id: str
    vehicle_id: int
    kind: str
    request_id: int | None
    critical: bool = False
    carry_over: bool = False
    empty_distance_m: float = 0.0
    payload: object = None


@dataclass(frozen=True)
class ServiceChain:
    chain_id: str
    scenario_id: str
    vehicle_id: int
    action_id: str
    request_ids: tuple[int, ...]
    empty_distance_m: float = 0.0
    payload: object = None
    relocation_slots: tuple[int, ...] = ()


class CityMasterLimitError(ValueError):
    pass


class CityMasterTimeout(TimeoutError):
    pass


class _Budget:
    def __init__(self, seconds):
        if isinstance(seconds, bool) or not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("time_limit_s must be positive and finite")
        self.seconds = min(float(seconds), DECISION_TIME_LIMIT_S)
        self.started = perf_counter()

    def remaining(self, stage):
        remaining = self.seconds - (perf_counter() - self.started)
        if remaining <= 0:
            raise CityMasterTimeout(f"{stage}: total {self.seconds:g}s city master budget expired")
        return remaining


class _NonzeroLedger:
    def __init__(self, limit):
        self.limit, self.total, self.lock_rows = limit, 0, 0

    def add(self, count, *, lock=False):
        if self.total + count > self.limit:
            raise CityMasterLimitError("sparse nonzero budget exceeded; no truncation or dense fallback")
        self.total += count
        self.lock_rows += int(lock)


class _Rows:
    def __init__(self, columns, ledger):
        self.columns, self.ledger = columns, ledger
        self.row_indices, self.column_indices, self.values = [], [], []
        self.lower, self.upper = [], []

    def add(self, entries, lower, upper, *, lock=False):
        entries = [(int(index), float(value)) for index, value in entries if value]
        self.ledger.add(len(entries), lock=lock)
        row = len(self.lower)
        for index, value in entries:
            self.row_indices.append(row)
            self.column_indices.append(index)
            self.values.append(value)
        self.lower.append(float(lower))
        self.upper.append(float(upper))

    def matrix(self):
        return coo_matrix((self.values, (self.row_indices, self.column_indices)),
                          shape=(len(self.lower), self.columns)).tocsc()


def _distance(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return Fraction(str(value))


def _identifier(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")


def _slot(value):
    if type(value) is not int or value < 0 or value % 900:
        raise ValueError("relocation slots must be integer seconds on the nonnegative 900-s grid")
    return value


def _current_slot(action):
    if action.kind != "RELOCATE":
        return None
    if not isinstance(action.payload, dict) or "relocation_slot_s" not in action.payload:
        raise ValueError("RELOCATE payload must identify its shared relocation_slot_s")
    return _slot(action.payload["relocation_slot_s"])


def _prepare(actions, chains, weights, variable_limit, nonzero_limit, budget):
    action_rows, chain_rows = [], []
    action_map, chain_ids = {}, set()
    for index, action in enumerate(actions):
        if index % 256 == 0:
            budget.remaining("current-action input preparation")
        if not isinstance(action, CurrentAction):
            raise ValueError("actions must contain CurrentAction instances")
        _identifier(action.action_id, "action_id")
        if action.action_id in action_map or type(action.vehicle_id) is not int or action.kind not in KINDS:
            raise ValueError("duplicate action_id, invalid vehicle_id, or unsupported action kind")
        if type(action.critical) is not bool or type(action.carry_over) is not bool:
            raise ValueError("critical and carry_over must be booleans")
        if action.kind == "SERVE":
            if type(action.request_id) is not int:
                raise ValueError("SERVE must identify an integer current request_id")
        elif action.request_id is not None or action.critical or action.carry_over:
            raise ValueError("non-SERVE actions cannot contain customer identity or service-priority flags")
        distance = _distance(action.empty_distance_m, "current empty_distance_m")
        if action.kind == "LAYOUT" and distance:
            raise ValueError("LAYOUT is a zero-cost initial-position choice")
        _current_slot(action)
        action_map[action.action_id] = action
        action_rows.append(action)
        if len(action_rows) > variable_limit:
            raise CityMasterLimitError("current-action variable limit exceeded")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("scenario_weights must be a nonempty dictionary")
    fractions = {}
    for sid, weight in weights.items():
        _identifier(sid, "scenario_id")
        fraction = _distance(weight, "scenario weight")
        if not fraction:
            raise ValueError("scenario weights must be strictly positive")
        fractions[sid] = fraction
    total = sum(fractions.values(), Fraction(0))
    if not math.isclose(float(total), 1., rel_tol=0., abs_tol=1e-12):
        raise ValueError("scenario weights must sum to one")
    fractions = {sid: fraction / total for sid, fraction in sorted(fractions.items())}
    current_move_count = sum(action.kind == "RELOCATE" for action in action_rows)
    # A cheap lower bound on the necessary base entries prevents constructing
    # a huge request-incidence dictionary before discovering the sparse cap.
    nonzero_floor = (len(action_rows) + sum(action.kind == "SERVE" for action in action_rows)
                     + current_move_count * (1 + len(fractions)))
    if nonzero_floor > nonzero_limit:
        raise CityMasterLimitError("sparse nonzero budget exceeded during input preparation")
    removed_empty = 0
    for index, chain in enumerate(chains):
        if index % 256 == 0:
            budget.remaining("typed-chain input preparation")
        if not isinstance(chain, ServiceChain):
            raise ValueError("chains must contain ServiceChain instances")
        _identifier(chain.chain_id, "chain_id")
        action = action_map.get(chain.action_id)
        if (chain.chain_id in chain_ids or chain.scenario_id not in fractions or action is None
                or type(chain.vehicle_id) is not int or chain.vehicle_id != action.vehicle_id):
            raise ValueError("duplicate chain_id, unknown scenario/action, or chain vehicle mismatch")
        chain_ids.add(chain.chain_id)
        if len(action_rows) + len(chain_ids) > variable_limit:
            raise CityMasterLimitError("supplied input variable limit exceeded")
        if not isinstance(chain.request_ids, tuple) or any(type(rid) is not int for rid in chain.request_ids):
            raise ValueError("chain request_ids must be a tuple of integer global identities")
        if len(chain.request_ids) > MAX_NONZEROS or len(set(chain.request_ids)) != len(chain.request_ids):
            raise ValueError("a chain repeats customer identity or exceeds the sparse row capacity")
        if action.kind == "SERVE" and action.request_id in chain.request_ids:
            raise ValueError("recourse chain must not repeat its first SERVE customer")
        _distance(chain.empty_distance_m, "future incremental empty_distance_m")
        if not isinstance(chain.relocation_slots, tuple) or len(chain.relocation_slots) > MAX_NONZEROS:
            raise ValueError("chain relocation_slots must be a tuple")
        for slot in chain.relocation_slots:
            _slot(slot)
        # A zero-service terminal continuation cannot improve this model:
        # it has nonnegative distance/cap use and implicit stopping costs zero.
        if not chain.request_ids:
            removed_empty += 1
            continue
        nonzero_floor += 1 + len(chain.request_ids) + len(set(chain.relocation_slots))
        if nonzero_floor > nonzero_limit:
            raise CityMasterLimitError("sparse nonzero budget exceeded during input preparation")
        chain_rows.append(chain)
        if len(action_rows) + len(chain_rows) > variable_limit:
            raise CityMasterLimitError("supplied restricted-chain variable limit exceeded")
    return (tuple(sorted(action_rows, key=lambda a: (a.vehicle_id, a.action_id))),
            tuple(sorted(chain_rows, key=lambda c: (c.scenario_id, c.vehicle_id, c.action_id, c.chain_id))),
            fractions, removed_empty)


def _master(actions, chains, weights, relocation_cap, ledger, budget):
    rows = _Rows(len(actions) + len(chains), ledger)
    action_columns = {action.action_id: index for index, action in enumerate(actions)}
    vehicles, first_requests = defaultdict(list), defaultdict(list)
    current_moves = defaultdict(list)
    for index, action in enumerate(actions):
        vehicles[action.vehicle_id].append(index)
        if action.kind == "SERVE":
            first_requests[action.request_id].append(index)
        if action.kind == "RELOCATE":
            current_moves[_current_slot(action)].append(index)
    for indices in vehicles.values():
        rows.add(((index, 1) for index in indices), 1, 1)
    for indices in first_requests.values():
        rows.add(((index, 1) for index in indices), -np.inf, 1)
    grouped, scene_requests, scene_moves = defaultdict(list), defaultdict(list), defaultdict(list)
    for index, chain in enumerate(chains, len(actions)):
        if index % 256 == 0:
            budget.remaining("global request and move-slot sparse rows")
        grouped[chain.scenario_id, chain.action_id].append(index)
        for rid in chain.request_ids:
            scene_requests[chain.scenario_id, rid].append(index)
        for slot, count in Counter(chain.relocation_slots).items():
            scene_moves[chain.scenario_id, slot].append((index, count))
    for (_, aid), indices in grouped.items():
        rows.add([*( (index, 1) for index in indices), (action_columns[aid], -1)], -np.inf, 0)
    for (_, rid), indices in scene_requests.items():
        rows.add(((index, 1) for index in [*indices, *first_requests.get(rid, ())]), -np.inf, 1)
    all_current_moves = [index for indices in current_moves.values() for index in indices]
    if all_current_moves:
        rows.add(((index, 1) for index in all_current_moves), -np.inf, relocation_cap)
    slot_rows = 0
    slots = sorted(set(slot for _, slot in scene_moves) | set(current_moves))
    for sid in weights:
        for slot in slots:
            entries = [*scene_moves.get((sid, slot), ()), *((index, 1) for index in current_moves.get(slot, ()))]
            if entries:
                rows.add(entries, -np.inf, relocation_cap)
                slot_rows += 1
    return rows, vehicles, slot_rows


def _components(rows, budget):
    """Exact factor decomposition including every global relocation row."""
    matrix = rows.matrix().tocsr()
    parents = list(range(rows.columns))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for row in range(matrix.shape[0]):
        if row % 256 == 0:
            budget.remaining("exact global-factor decomposition")
        ids = matrix.indices[matrix.indptr[row]:matrix.indptr[row + 1]]
        if not len(ids):
            if rows.lower[row] > 0 or rows.upper[row] < 0:
                raise RuntimeError("infeasible empty sparse constraint")
            continue
        first = find(int(ids[0]))
        for index in ids[1:]:
            other = find(int(index))
            if first != other:
                parents[other] = first
    groups = {}
    for index in range(rows.columns):
        groups.setdefault(find(index), dict(columns=[], rows=[]))["columns"].append(index)
    for row in range(matrix.shape[0]):
        ids = matrix.indices[matrix.indptr[row]:matrix.indptr[row + 1]]
        if len(ids):
            groups[find(int(ids[0]))]["rows"].append(row)
    return matrix, sorted(groups.values(), key=lambda group: group["columns"][0])


def _objectives(actions, chains, weights):
    size = len(actions) + len(chains)
    names = ("critical_now", "expected_served", "current_served", "carry_over_current_served", "expected_empty_distance_m")
    coefficients = {name: [Fraction(0)] * size for name in names}
    for index, action in enumerate(actions):
        if action.kind == "SERVE":
            coefficients["critical_now"][index] = Fraction(int(action.critical))
            coefficients["current_served"][index] = coefficients["expected_served"][index] = Fraction(1)
            coefficients["carry_over_current_served"][index] = Fraction(int(action.carry_over))
        coefficients["expected_empty_distance_m"][index] = _distance(action.empty_distance_m, "distance")
    for index, chain in enumerate(chains, len(actions)):
        coefficients["expected_served"][index] = weights[chain.scenario_id] * len(chain.request_ids)
        coefficients["expected_empty_distance_m"][index] = weights[chain.scenario_id] * _distance(chain.empty_distance_m, "distance")
    return coefficients


def _tier_order(policy):
    if policy == "CHAIN_DEFER":
        return ("critical_now", "expected_served", "current_served", "carry_over_current_served", "expected_empty_distance_m")
    return ("critical_now", "current_served", "carry_over_current_served", "expected_served", "expected_empty_distance_m")


def _single_vehicle(actions, chains, weights, relocation_cap, policy, budget):
    """Exact rational optimization for one independent vehicle component."""
    grouped = defaultdict(list)
    for chain in chains:
        grouped[chain.scenario_id, chain.action_id].append(chain)
    best = None
    best_selection = None
    for action in actions:
        budget.remaining("independent single-vehicle exact chain selection")
        if action.kind == "RELOCATE" and relocation_cap < 1:
            continue
        current_slot = _current_slot(action)
        chosen = []
        expected = Fraction(int(action.kind == "SERVE"))
        distance = _distance(action.empty_distance_m, "distance")
        for sid, weight in weights.items():
            scene_best = None
            for chain in grouped.get((sid, action.action_id), ()):
                if any(count + int(current_slot == slot) > relocation_cap
                       for slot, count in Counter(chain.relocation_slots).items()):
                    continue
                value = (len(chain.request_ids), -_distance(chain.empty_distance_m, "distance"))
                if scene_best is None or value > scene_best[0] or (value == scene_best[0] and chain.chain_id < scene_best[1].chain_id):
                    scene_best = value, chain
            if scene_best is not None:
                chosen.append(scene_best[1])
                expected += weight * scene_best[0][0]
                distance -= weight * scene_best[0][1]
        values = dict(critical_now=Fraction(int(action.kind == "SERVE" and action.critical)),
                      expected_served=expected, current_served=Fraction(int(action.kind == "SERVE")),
                      carry_over_current_served=Fraction(int(action.kind == "SERVE" and action.carry_over)),
                      expected_empty_distance_m=-distance)
        score = tuple(values[name] for name in _tier_order(policy))
        tie = (action.action_id, tuple(chain.chain_id for chain in chosen))
        if best is None or score > best[0] or (score == best[0] and tie < best[1]):
            best, best_selection = (score, tie), (action, chosen)
    if best_selection is None:
        raise RuntimeError("restricted city component has no feasible common first action")
    return best_selection


def _component_rows(matrix, rows, group):
    columns = group["columns"]
    submatrix = matrix[group["rows"]][:, columns].tocoo()
    subset = _Rows(len(columns), rows.ledger)
    # These base entries are already charged to the global nonzero ledger.
    subset.row_indices = submatrix.row.tolist()
    subset.column_indices = submatrix.col.tolist()
    subset.values = submatrix.data.tolist()
    subset.lower = [rows.lower[index] for index in group["rows"]]
    subset.upper = [rows.upper[index] for index in group["rows"]]
    return subset


def _coupled_selection(rows, objectives, local_actions, policy, budget, receipt):
    selected = None
    fixed, stages = [], []

    def stage(name, values, maximize, *, last_float=False, force=False):
        nonlocal selected
        budget.remaining(name)
        if not any(values) and not force:
            stages.append(dict(stage=name, value=0., opt_status="CONSTANT_EXACT", runtime_s=0.))
            return
        for _, prior_values, prior_value in fixed:
            if values == prior_values:
                # In a current-only baseline, expected total and current
                # service are identical. The exact earlier lock is already
                # the proof; do not invoke the same MILP a second time.
                stages.append(dict(stage=name, value=float(prior_value), value_exact=str(prior_value),
                                   opt_status="OPTIMAL", runtime_s=0., exact_objective_lock=True,
                                   proof="IDENTICAL_OBJECTIVE_ALREADY_EXACTLY_LOCKED"))
                return
        lattice = contract._lattice(values, rows)
        if lattice is None and not last_float:
            raise contract.JointNumericalContractError(f"{name}: no safely certified integer objective grid")
        if lattice is None:
            objective = np.asarray([float(value) for value in values])
            scale = None
            coefficient_sum = float(np.sum(np.abs(objective)))
            maximum = float(np.max(np.abs(objective), initial=0.))
            error = (contract.NUMERICAL_TOLERANCE * (coefficient_sum + len(rows.lower) * maximum)
                     + 64 * np.finfo(float).eps * max(1., coefficient_sum))
        else:
            objective, scale, error = lattice
        result, binary, diagnostics = contract._run_milp(
            -objective if maximize else objective, rows, budget, receipt, name)
        selected = tuple(int(index) for index in np.flatnonzero(binary))
        for old_name, old_values, old_value in fixed:
            if sum((old_values[index] for index in selected), Fraction(0)) != old_value:
                raise contract.JointNumericalContractError(f"{name}: changed exact prior tier {old_name}")
        value = sum((values[index] for index in selected), Fraction(0))
        native_value = float(-result.fun if maximize else result.fun)
        binary_value = float(objective @ binary)
        if abs(native_value - binary_value) > error:
            raise contract.JointNumericalContractError(f"{name}: numerical objective disagrees with binary certificate")
        units = None
        if scale is not None:
            exact_units = value * scale
            if exact_units.denominator != 1 or binary_value != int(exact_units):
                raise contract.JointNumericalContractError(f"{name}: objective left its exact lattice")
            units = int(exact_units)
            bound = getattr(result, "mip_dual_bound", None)
            optimum = -units if maximize else units
            if bound is None or not math.isfinite(bound) or math.ceil(float(bound) - error) < optimum:
                raise contract.JointNumericalContractError(f"{name}: solver status does not prove its integer grid value")
            rows.add(((int(index), objective[index]) for index in np.flatnonzero(objective)), units, units, lock=True)
            fixed.append((name, values, value))
        stages.append(dict(stage=name, value=float(value), value_exact=str(value),
                           opt_status="OPTIMAL", status=int(result.status), runtime_s=diagnostics["runtime_s"],
                           exact_objective_lock=scale is not None, objective_scale_exact=str(scale) if scale is not None else None,
                           grid_units=units, mip_gap=float(result.mip_gap or 0.), numerical_diagnostics=diagnostics,
                           distance_lattice_optimality_proved=(scale is not None) if last_float else None))

    for name in _tier_order(policy):
        stage(name, objectives[name], name != "expected_empty_distance_m", last_float=name == "expected_empty_distance_m")
    distance_lattice = contract._lattice(objectives["expected_empty_distance_m"], rows)
    # A deterministic mixed-radix tie is used only for genuinely small safe
    # components, after an exactly certified distance lock. It is not a risk
    # index and does not blend scientific objectives.
    tie_metadata = dict(applied=False, reason="FINAL_DISTANCE_OR_LARGE_COMPONENT_NO_EXTRA_TIE_SOLVE")
    groups = defaultdict(list)
    for local_index, action in local_actions:
        groups[action.vehicle_id].append((local_index, action))
    groups = [sorted(group, key=lambda pair: pair[1].action_id) for _, group in sorted(groups.items())]
    maximum_code = math.prod(len(group) for group in groups) - 1
    if distance_lattice is not None and maximum_code < 2**53:
        values = [Fraction(0)] * rows.columns
        multiplier = 1
        for group in reversed(groups):
            for rank, (index, _) in enumerate(group):
                values[index] = Fraction(rank * multiplier)
            multiplier *= len(group)
        if contract._lattice(values, rows) is not None:
            stage("stable_first_actions:mixed_radix", values, False)
            tie_metadata = dict(applied=True, kind="EXACT_MIXED_RADIX_AFTER_SCIENTIFIC_TIERS", maximum_code=maximum_code)
    if selected is None:
        stage("feasibility", [Fraction(0)] * rows.columns, False, force=True)
    budget.remaining("coupled restricted-column optimality certification")
    return selected, stages, tie_metadata


def _validate_selection(actions, chains, weights, action_ids, chain_ids, cap):
    chosen_actions = [action for action in actions if action.action_id in action_ids]
    chosen_chains = [chain for chain in chains if chain.chain_id in chain_ids]
    if len(chosen_actions) != len({action.vehicle_id for action in actions}) or len({a.vehicle_id for a in chosen_actions}) != len(chosen_actions):
        raise RuntimeError("integer solution must have one common first action per vehicle")
    first = [a.request_id for a in chosen_actions if a.kind == "SERVE"]
    if len(first) != len(set(first)):
        raise RuntimeError("first SERVE repeats a real current customer")
    actual_moves = Counter(_current_slot(a) for a in chosen_actions if a.kind == "RELOCATE")
    if sum(actual_moves.values()) > cap:
        raise RuntimeError("current relocation cap violated")
    scenario_jobs, state_chains = {sid: set(first) for sid in weights}, set()
    future_moves = {sid: Counter() for sid in weights}
    for chain in chosen_chains:
        key = chain.scenario_id, chain.action_id
        if chain.action_id not in action_ids or key in state_chains:
            raise RuntimeError("chain uses a different common action or repeats one action/scenario state")
        state_chains.add(key)
        if scenario_jobs[chain.scenario_id].intersection(chain.request_ids):
            raise RuntimeError("scenario repeats a first-SERVE or recourse customer identity")
        scenario_jobs[chain.scenario_id].update(chain.request_ids)
        future_moves[chain.scenario_id].update(chain.relocation_slots)
    combined = {}
    for sid, planned in future_moves.items():
        total = planned + actual_moves
        if any(count > cap for count in total.values()):
            raise RuntimeError("scenario/900-s relocation slot cap violated")
        combined[sid] = {str(slot): count for slot, count in sorted(total.items())}
    return chosen_actions, chosen_chains, dict(
        current_relocations_by_slot={str(slot): count for slot, count in sorted(actual_moves.items())},
        planned_future_relocations_by_scenario={sid: {str(slot): count for slot, count in sorted(counter.items())}
                                                for sid, counter in future_moves.items()},
        constrained_relocations_by_scenario_slot=combined)


def solve_city_master(actions, chains, scenario_weights, *, policy="CHAIN_DEFER", time_limit_s=10.,
                      max_variables=20_000, max_nonzeros=150_000, relocation_cap=50):
    """Solve the supplied restricted-column city master under one total budget.

    Independent components are exact factors of ALL sparse constraint rows,
    including global current and future-slot movement caps. Single-vehicle
    factors are optimized by exact rational comparison; coupled factors use
    strictly certified binary MILP tiers. No incumbent/time-limit fallback,
    future customer truth, dense matrix, or global optimality bound is used.
    """
    budget = _Budget(time_limit_s)
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES!r}")
    for value, label in ((max_variables, "max_variables"), (max_nonzeros, "max_nonzeros")):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    if type(relocation_cap) is not int or relocation_cap < 0:
        raise ValueError("relocation_cap must be a nonnegative integer")
    variable_limit, nonzero_limit = min(max_variables, MAX_VARIABLES), min(max_nonzeros, MAX_NONZEROS)
    actions, chains, weights, removed_empty = _prepare(
        actions, chains, scenario_weights, variable_limit, nonzero_limit, budget)
    receipt = contract._solver_options_receipt()
    preparation_s = perf_counter() - budget.started
    ledger = _NonzeroLedger(nonzero_limit)
    started = perf_counter()
    rows, vehicles, slot_rows = _master(actions, chains, weights, relocation_cap, ledger, budget)
    base_nonzeros = ledger.total
    matrix, components = _components(rows, budget)
    objectives = _objectives(actions, chains, weights)
    build_s = perf_counter() - started
    chosen_indices, component_reports = [], []
    exact_count = mip_count = 0
    optimize_s = 0.
    for component_id, group in enumerate(components):
        budget.remaining("exact restricted-column component selection")
        local_actions = [(local, actions[global_index]) for local, global_index in enumerate(group["columns"])
                         if global_index < len(actions)]
        component_chains = [chains[index - len(actions)] for index in group["columns"] if index >= len(actions)]
        started = perf_counter()
        if len({action.vehicle_id for _, action in local_actions}) == 1:
            selected_action, selected_chains = _single_vehicle(
                [action for _, action in local_actions], component_chains, weights, relocation_cap, policy, budget)
            ids = {selected_action.action_id}
            cids = {chain.chain_id for chain in selected_chains}
            local_chosen = [index for index in group["columns"]
                            if (index < len(actions) and actions[index].action_id in ids)
                            or (index >= len(actions) and chains[index - len(actions)].chain_id in cids)]
            chosen_indices.extend(local_chosen)
            exact_count += 1
            report = dict(component_id=component_id, backend="EXACT_RATIONAL_SINGLE_VEHICLE_SELECTION",
                          columns=len(group["columns"]), rows=len(group["rows"]),
                          distance_optimality_exact=True, deterministic_tie=True)
        else:
            subset = _component_rows(matrix, rows, group)
            local_objectives = {name: [values[index] for index in group["columns"]] for name, values in objectives.items()}
            selected, stages, tie = _coupled_selection(subset, local_objectives, local_actions, policy, budget, receipt)
            chosen_indices.extend(group["columns"][index] for index in selected)
            mip_count += 1
            report = dict(component_id=component_id, backend="STRICT_SPARSE_BINARY_SCIPY_HIGHS",
                          columns=len(group["columns"]), rows=len(subset.lower), nonzeros=len(subset.values),
                          stages=stages, stable_tie=tie)
        report["runtime_s"] = perf_counter() - started
        optimize_s += report["runtime_s"]
        component_reports.append(report)
    started = perf_counter()
    selected = tuple(sorted(chosen_indices))
    vector = np.zeros(rows.columns)
    vector[list(selected)] = 1.
    if contract._violation(matrix, rows, vector) != 0.:
        raise RuntimeError("component union violates a global sparse master constraint")
    action_ids = {actions[index].action_id for index in selected if index < len(actions)}
    chain_ids = {chains[index - len(actions)].chain_id for index in selected if index >= len(actions)}
    selected_actions, selected_chains, move_counts = _validate_selection(actions, chains, weights, action_ids, chain_ids, relocation_cap)
    values = {name: sum((coefficients[index] for index in selected), Fraction(0)) for name, coefficients in objectives.items()}
    validation_s = perf_counter() - started
    budget.remaining("global customer/common-action/relocation reconciliation")
    elapsed = perf_counter() - budget.started
    return dict(
        kind="SCHEME_A_CITY_RESTRICTED_SUPPLIED_MULTI_SERVICE_CHAIN_MASTER", policy=policy, opt_status="OPTIMAL",
        selected_action_ids=tuple(action.action_id for action in selected_actions),
        selected_chain_ids=tuple(chain.chain_id for chain in selected_chains),
        critical_now=int(values["critical_now"]), current_served=int(values["current_served"]),
        carry_over_current_served=int(values["carry_over_current_served"]),
        expected_served=float(values["expected_served"]), expected_served_exact=str(values["expected_served"]),
        expected_empty_distance_m=float(values["expected_empty_distance_m"]),
        expected_empty_distance_exact=str(values["expected_empty_distance_m"]),
        model=dict(columns=rows.columns, variables=rows.columns, rows=len(rows.lower) + ledger.lock_rows,
                   base_rows=len(rows.lower), nonzeros=ledger.total, base_nonzeros=base_nonzeros, sparse=True,
                   max_variables=variable_limit, max_nonzeros=nonzero_limit,
                   current_action_variables=len(actions), optional_chain_variables=len(chains),
                   empty_continuation_is_implicit=True, dominated_empty_columns_removed=removed_empty,
                   exact_factor_decomposition=True, component_count=len(components),
                   independent_single_vehicle_components=exact_count, coupled_milp_components=mip_count,
                   global_relocation_coupling_preserved=True, future_relocation_slot_rows=slot_rows,
                   relocation_cap=relocation_cap, relocation_slot_s=900,
                   **move_counts,
                   supplied_chain_domain="RESTRICTED_CALLER_CERTIFIED_COLUMNS_NOT_COMPLETE_CHAIN_ENUMERATION",
                   caller_certifies_timing_and_profile_support=True, chain_distance_is_incremental_future_only=True,
                   global_or_citywide_optimality_bound_claimed=False,
                   planned_recourse_is_not_realized_service=True),
        timing_s=dict(preparation=preparation_s, sparse_master_and_decomposition=build_s,
                      component_optimization=optimize_s, global_reconciliation=validation_s, total=elapsed),
        numerical_contract=dict(**receipt, tolerance=contract.NUMERICAL_TOLERANCE,
                                exact_service_priority_locks=True, floating_distance_face_lock=False,
                                one_total_decision_budget_s=budget.seconds, silent_fallback=False,
                                component_certificates=component_reports), runtime_s=elapsed)
