"""Independent finite-domain chain bounds and service-aware placement recovery.

The old city controller and coarse_layout solver are not modified. Exact
Fraction pricing certifies only the supplied MatchDeadlineSceneDAG. Full path
catalogues below are bounded benchmark/evaluation tools, not a city-scale
enumeration algorithm or future road certificates.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import isfinite, lcm
from time import perf_counter
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, OptimizeWarning, linprog, milp
from scipy.sparse import coo_matrix


class ChainBudgetExpired(TimeoutError):
    pass


def _check(check, deadline=None):
    check()
    if deadline is not None and perf_counter() >= deadline:
        raise ChainBudgetExpired("finite chain budget exhausted")


def _fraction(value):
    if isinstance(value, Fraction):
        return value
    if not isfinite(float(value)):
        raise ValueError("nonfinite exact price/weight")
    return Fraction.from_float(float(value))


def _weights(models, weights):
    if set(models) != set(weights):
        raise ValueError("scenario identities differ")
    result = {name: _fraction(weights[name]) for name in sorted(models)}
    if sum(result.values()) != 1 or any(w <= 0 for w in result.values()):
        raise ValueError("explicit exact scenario weights must be positive and sum to one")
    return result


class ExactPriceOracle:
    """Rational DP on the existing seven-readiness-bin, depth-three DAG."""

    def __init__(self, model, check=lambda: None, deadline=None):
        required = ("dispatch_bin", "root_ready_bin", "successor_ready_bin", "service_time")
        if any(not hasattr(model, field) for field in required):
            raise ValueError("only the declared match-deadline DAG is supported")
        self.model, self.check, self.deadline = model, check, deadline
        self.prices = tuple(_fraction(v) for v in model.alpha)
        if len(self.prices) != len(model.tasks) or any(v < 0 for v in self.prices):
            raise ValueError("invalid nonnegative customer prices")
        if np.any(model.service_time <= 0):
            raise ValueError("positive durations are required for time-forward pricing")
        for profile, successors in model.successors.items():
            for node, row in enumerate(successors):
                for following in row:
                    if following >= 0 and model.dispatch_bin[int(following)] <= model.dispatch_bin[node]:
                        raise ValueError("a repeated/same-time customer cycle is outside the declared DAG")
        self.calls = 0

    @lru_cache(maxsize=None)
    def tail(self, profile, end_bin, depth, node, ready_bin):
        self.calls += 1
        if self.calls % 64 == 0:
            _check(self.check, self.deadline)
        if depth == 0 or ready_bin >= min(6, end_bin):
            return Fraction(0)
        model, best = self.model, Fraction(0)
        for column, following in enumerate(model.successors[profile][node]):
            following = int(following)
            if (following < 0 or not model.allowed[profile][following]
                    or not ready_bin <= model.dispatch_bin[following] < min(6, end_bin)):
                continue
            candidate = (1 - self.prices[following] + self.tail(
                profile, end_bin, depth - 1, following,
                int(model.successor_ready_bin[profile][node, column])))
            best = max(best, candidate)
        return best

    def price(self, key):
        _check(self.check, self.deadline)
        profile, zone, _, end_bin = key
        entries = self.model.entries(key)
        ready = self.model.entry_ready_cache[key]
        value = Fraction(0)
        for node, ready_bin in zip(entries, ready):
            node = int(node)
            value = max(value, 1 - self.prices[node] + self.tail(
                profile, end_bin, 2, node, int(ready_bin)))
        if not 0 <= value <= 3:
            raise RuntimeError("exact quantity pricing left the depth-three range")
        return value


def certified_upper(models, weights, fixed_supply, fresh_groups, site_ids,
                    check=lambda: None, deadline=None):
    """Equations (6)-(7): common placement, including zero-supply roots."""
    weights = _weights(models, weights)
    if (not site_ids or len(set(site_ids)) != len(site_ids)
            or any(type(n) is not int or n < 0 for n in fresh_groups.values())
            or any(type(n) is not int or n < 0 for n in fixed_supply.values())):
        raise ValueError("invalid declared supply/site domain")
    values, terms, root_count = {}, {}, 0
    upper = Fraction(0)
    for name, model in models.items():
        oracle = ExactPriceOracle(model, check, deadline)
        base = sum(oracle.prices, Fraction(0))
        for key, count in sorted(fixed_supply.items()):
            base += count * oracle.price(key)
            root_count += 1
        upper += weights[name] * base
        for group in sorted(fresh_groups):
            for zone in range(len(site_ids)):
                values[name, group, zone] = oracle.price(("C", zone, *group))
                root_count += 1
        terms[name] = base
        oracle.tail.cache_clear()
    for group, count in fresh_groups.items():
        upper += count * max(sum(weights[name] * values[name, group, zone]
                                 for name in models) for zone in range(len(site_ids)))
    return dict(value=upper, upper_bound=float(upper),
                upper_bound_exact=[upper.numerator, upper.denominator],
                priced_root_count=root_count, scope="DECLARED_SPARSE_MATCH_DEADLINE_DAG",
                zero_supply_candidate_roots_included=True,
                common_placement_max_outside_scenario_mean=True,
                physical_road_or_online_capacity_bound=False)


def enumerate_paths(model, key, check=lambda: None, maximum=20000):
    """Complete paths only inside this fixed sparse DAG/depth domain."""
    profile, _, _, end_bin = key
    entries = model.entries(key)
    ready = model.entry_ready_cache[key]
    result = []

    def visit(path, ready_bin):
        if len(result) % 64 == 0:
            check()
        if len(result) >= maximum:
            raise ValueError("declared path catalogue exceeded its variable cap")
        result.append(path)
        if len(path) == 3 or ready_bin >= min(6, end_bin):
            return
        node = path[-1]
        for column, following in enumerate(model.successors[profile][node]):
            following = int(following)
            if (following < 0 or not model.allowed[profile][following]
                    or not ready_bin <= model.dispatch_bin[following] < min(6, end_bin)):
                continue
            if following in path:
                raise ValueError("one forecast customer cannot occur twice in a chain")
            visit((*path, following), int(model.successor_ready_bin[profile][node, column]))

    for node, ready_bin in zip(entries, ready):
        visit((int(node),), int(ready_bin))
    return tuple(sorted(set(result), key=lambda p: (-len(p), p)))


@dataclass(frozen=True)
class Root:
    scene: str
    kind: str
    key: tuple


class FiniteCatalogue:
    """Common bounded evaluation/reference model, never used by the city."""

    def __init__(self, models, weights, fixed_supply, fresh_groups, site_ids,
                 check=lambda: None, maximum_variables=20000, maximum_nonzeros=150000):
        self.models = models
        self.weights = _weights(models, weights)
        self.fixed, self.fresh = dict(fixed_supply), dict(fresh_groups)
        self.site_ids, self.check = tuple(site_ids), check
        self.groups = tuple(sorted(fresh_groups))
        self.scale = lcm(*(w.denominator for w in self.weights.values()))
        self.scene_units = {name: int(w * self.scale) for name, w in self.weights.items()}
        self.placement = {(group, zone): index for index, (group, zone) in enumerate(
            (group, zone) for group in self.groups for zone in range(len(site_ids)))}
        self.roots, self.columns, self.task_rows = [], [], {}
        lower, upper, rows, cols, data = [], [], [], [], []
        for name, model in sorted(models.items()):
            for key in sorted(fixed_supply):
                self.roots.append(Root(name, "FIXED", key))
            for group in self.groups:
                for zone in range(len(site_ids)):
                    self.roots.append(Root(name, "FRESH", ("C", zone, *group)))
        self.root_rows = {root: i for i, root in enumerate(self.roots)}
        for root in self.roots:
            lower.append(-np.inf)
            upper.append(fixed_supply[root.key] if root.kind == "FIXED" else 0)
            if root.kind == "FRESH":
                _, zone, ready, end = root.key
                rows.append(self.root_rows[root]); cols.append(self.placement[(ready, end), zone]); data.append(-1)
        for name, model in sorted(models.items()):
            for node in range(len(model.tasks)):
                self.task_rows[name, node] = len(lower)
                lower.append(-np.inf); upper.append(1)
        for group in self.groups:
            row = len(lower)
            lower.append(fresh_groups[group]); upper.append(fresh_groups[group])
            for zone in range(len(site_ids)):
                rows.append(row); cols.append(self.placement[group, zone]); data.append(1)
        for root in self.roots:
            for path in enumerate_paths(models[root.scene], root.key, check, maximum_variables):
                column = len(self.placement) + len(self.columns)
                if column >= maximum_variables:
                    raise ValueError("complete declared model exceeded 20000 variables")
                self.columns.append((root, path))
                for row in (self.root_rows[root], *(self.task_rows[root.scene, node] for node in path)):
                    rows.append(row); cols.append(column); data.append(1)
                if len(data) > maximum_nonzeros:
                    raise ValueError("complete declared model exceeded 150000 nonzeros")
        self.matrix = coo_matrix((data, (rows, cols)),
                                shape=(len(lower), len(self.placement) + len(self.columns))).tocsc()
        self.lower, self.upper = np.asarray(lower, float), np.asarray(upper, float)
        self.cost = np.r_[np.zeros(len(self.placement)),
                         [-self.scene_units[root.scene] * len(path) for root, path in self.columns]]
        self.bounds_lower = np.zeros(len(self.cost))
        self.bounds_upper = np.r_[
            [fresh_groups[group] for group, zone in self.placement], np.ones(len(self.columns))]
        self.info = dict(variables=len(self.cost), nonzeros=self.matrix.nnz,
                         path_columns=len(self.columns), complete_declared_domain=True,
                         domain="C_TOP3_HV_TOP6_MAX3_MATCH_300_PICKUP_600")

    def placement_vector(self, deployment):
        if set(deployment) != set(self.groups):
            raise ValueError("integer deployment groups differ")
        vector = np.zeros(len(self.placement), dtype=np.int64)
        for group in self.groups:
            values = np.asarray(deployment[group])
            if (values.shape != (len(self.site_ids),) or np.any(values < 0)
                    or np.any(values != np.round(values)) or int(values.sum()) != self.fresh[group]):
                raise ValueError("integer deployment does not conserve fixed supply")
            for zone, value in enumerate(values):
                vector[self.placement[group, zone]] = int(value)
        return vector

    def deployment(self, vector):
        return {group: np.asarray([int(vector[self.placement[group, z]])
                                  for z in range(len(self.site_ids))], dtype=np.int64)
                for group in self.groups}

    def validate(self, vector):
        values = np.asarray(vector, float)
        if values.shape != self.cost.shape or not np.all(np.isfinite(values)):
            raise ValueError("invalid integer chain vector")
        integer = np.rint(values).astype(np.int64)
        if np.max(np.abs(values - integer), initial=0.) > 1e-6:
            raise ValueError("fractional incumbent cannot be reported as integer service")
        if np.any(integer < self.bounds_lower) or np.any(integer > self.bounds_upper):
            raise ValueError("integer variable bounds violated")
        used = self.matrix @ integer
        if np.any(used < self.lower) or np.any(used > self.upper):
            raise ValueError("shared customer, supply or common-placement constraint violated")
        scaled = int(round(float(-self.cost @ integer)))
        scenes = Counter()
        for offset, (root, path) in enumerate(self.columns, len(self.placement)):
            if integer[offset]:
                if len(path) != len(set(path)):
                    raise ValueError("repeated physical task in witness")
                scenes[root.scene] += len(path)
        exact = sum((self.weights[name] * scenes[name] for name in self.models), Fraction(0))
        if exact != Fraction(scaled, self.scale):
            raise ValueError("reconstructed service objective differs")
        return integer, scaled, exact, dict(scenes)

    def idle(self, deployment):
        vector = np.zeros(len(self.cost), dtype=np.int64)
        vector[:len(self.placement)] = self.placement_vector(deployment)
        return vector

    def greedy(self, deployment):
        vector = self.idle(deployment)
        remaining = {root: (self.fixed[root.key] if root.kind == "FIXED"
            else int(deployment[root.key[2:]][root.key[1]])) for root in self.roots}
        used = {name: set() for name in self.models}
        for offset in sorted(range(len(self.columns)), key=lambda j: (
                -len(self.columns[j][1]), self.columns[j][0].scene,
                self.columns[j][0].kind, self.columns[j][0].key, self.columns[j][1])):
            root, path = self.columns[offset]
            if remaining[root] and not any(node in used[root.scene] for node in path):
                vector[len(self.placement) + offset] = 1
                remaining[root] -= 1
                used[root.scene].update(path)
        self.validate(vector)
        return vector

    def solve_integer(self, deployment=None, time_limit_s=10., incumbent=None):
        started = perf_counter()
        self.check()
        if time_limit_s <= 0:
            raise ChainBudgetExpired("no integer evaluation time remains")
        if deployment is None:
            deployment_seed = {g: np.r_[self.fresh[g], np.zeros(len(self.site_ids)-1, int)]
                               for g in self.groups}
        else:
            deployment_seed = deployment
        best = self.greedy(deployment_seed)
        best_scaled = self.validate(best)[1]
        if incumbent is not None:
            candidate, scaled, _, _ = self.validate(incumbent)
            if (deployment is not None and not np.array_equal(candidate[:len(self.placement)],
                                                             self.placement_vector(deployment))):
                raise ValueError("an incumbent belongs to another placement")
            if scaled > best_scaled:
                best, best_scaled = candidate.copy(), scaled
        low, high = self.bounds_lower.copy(), self.bounds_upper.copy()
        if deployment is not None:
            low[:len(self.placement)] = high[:len(self.placement)] = self.placement_vector(deployment)
        # Integer objective coefficients avoid interpreting service thirds as
        # fractional customer allocations. Full witness is rechecked exactly.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solved = milp(self.cost, integrality=np.ones(len(self.cost), np.int8),
                bounds=Bounds(low, high), constraints=LinearConstraint(self.matrix, self.lower, self.upper),
                options={"presolve": True, "time_limit": float(time_limit_s),
                         "mip_rel_gap": 0., "threads": 1})
        _warnings(caught)
        if solved.status not in (0, 1):
            raise RuntimeError(f"WAIT-feasible integer evaluation failed: {solved.message}")
        if solved.x is not None:
            candidate, scaled, _, _ = self.validate(solved.x)
            if scaled >= best_scaled:
                best, best_scaled = candidate, scaled
        best, scaled, exact, scenes = self.validate(best)
        if solved.status == 0 and abs(float(solved.fun) + scaled) > 1e-5:
            raise RuntimeError("solver optimum disagrees with integer witness")
        dual = getattr(solved, "mip_dual_bound", None)
        info = dict(service_value=float(exact), service_exact=[exact.numerator, exact.denominator],
                    service_scaled=scaled, scenario_services=scenes, solver_status=int(solved.status),
                    optimal_solver_termination=solved.status == 0,
                    solver_upper_bound=None if dual is None or not isfinite(float(dual)) else -float(dual)/self.scale,
                    solver_bound_is_not_rational_certificate=True, runtime_s=perf_counter()-started,
                    integer_witness_verified=True, common_placement_verified=True)
        return best, info

    def complete_lp(self, time_limit_s=10.):
        started = perf_counter()
        self.check()
        eq = np.flatnonzero(self.lower == self.upper)
        ub = np.flatnonzero(self.lower != self.upper)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", OptimizeWarning)
            solved = linprog(self.cost, A_ub=self.matrix[ub], b_ub=self.upper[ub],
                A_eq=self.matrix[eq], b_eq=self.upper[eq], bounds=(0, None), method="highs-ds",
                options={"presolve": True, "time_limit": float(time_limit_s), "threads": 1})
        _warnings(caught)
        if solved.status != 0:
            return None, dict(lp_status=int(solved.status), runtime_s=perf_counter()-started)
        inverse = {int(row): j for j, row in enumerate(ub)}
        for name, model in self.models.items():
            prices = np.asarray([max(0., -float(solved.ineqlin.marginals[
                inverse[self.task_rows[name, node]]])) for node in range(len(model.tasks))])
            model.set_prices(prices, {})
        bound = certified_upper(self.models, self.weights, self.fixed, self.fresh, self.site_ids, self.check)
        info = dict(lp_status=0, complete_lp_value=-float(solved.fun)/self.scale,
                    certificate_upper_bound=bound["upper_bound"],
                    certificate_exact=bound["upper_bound_exact"], runtime_s=perf_counter()-started)
        if info["complete_lp_value"] > bound["upper_bound"] + 1e-7:
            raise RuntimeError("exact DP dual bound is below the declared LP value")
        return bound, info


def _warnings(caught):
    for warning in caught:
        if "Unrecognized options detected" not in str(warning.message):
            raise RuntimeError(f"unexpected optimizer warning: {warning.message}")


def reset_prices(models):
    for model in models.values():
        model.set_prices(np.zeros(len(model.tasks)), {})


def independent_placement(models, weights, fresh_groups, site_ids, check=lambda: None):
    """Unpriced future-chain baseline: same graph, no shared-demand prices."""
    weights = _weights(models, weights)
    reset_prices(models)
    result = {}
    for group, count in sorted(fresh_groups.items()):
        values = []
        for zone in range(len(site_ids)):
            check()
            values.append(sum(float(weights[name]) * models[name].price(("C", zone, *group))[0]
                              for name in models))
        # A zero-value tie is retained, not replaced with selected good tasks.
        best = min(range(len(site_ids)), key=lambda z: (-values[z], str(site_ids[z])))
        result[group] = np.zeros(len(site_ids), dtype=np.int64)
        result[group][best] = count
    return result


def solve_bound_recovery(models, weights, fixed_supply, fresh_groups, site_ids, cfg,
                         catalogue, check=lambda: None, tie_targets=None, common_build_s=0.):
    """One independent method: unchanged six-round LP plus bounds/recovery."""
    from .coarse_layout import solve_placement
    started = perf_counter()
    deadline = started + max(.001, 30. - common_build_s)
    reset_prices(models)
    # Fixed before experiments: reserve 12 of the common 30 seconds for exact
    # bound construction and finite integer recovery, not extra gifted time.
    local = dict(cfg, layout_bin_wall_limit_s=max(.001, min(18., deadline-perf_counter())))
    seed, lp_info = solve_placement(models, {n: float(w) for n, w in weights.items()},
        fixed_supply, fresh_groups, site_ids, local, check, tie_targets=tie_targets)
    price_started = perf_counter()
    bound = certified_upper(models, weights, fixed_supply, fresh_groups, site_ids, check, deadline)
    bound_time = perf_counter()-price_started
    evaluate_started = perf_counter()
    best, evaluation = catalogue.solve_integer(seed, min(10., deadline-perf_counter()))
    best_scaled = evaluation["service_scaled"]
    seed_scaled = best_scaled
    deployment = {group: np.asarray(values, dtype=np.int64).copy() for group, values in seed.items()}
    attempts, accepted, events = 0, 0, []
    order = sorted(range(len(site_ids)), key=lambda z: str(site_ids[z]))
    for group in sorted(fresh_groups):
        for donor in order:
            for receiver in order:
                if attempts >= 8 or perf_counter() >= deadline:
                    break
                if donor == receiver or deployment[group][donor] == 0:
                    continue
                check()
                trial = {g: values.copy() for g, values in deployment.items()}
                trial[group][donor] -= 1
                trial[group][receiver] += 1
                vector, info = catalogue.solve_integer(trial, min(10., deadline-perf_counter()))
                attempts += 1
                improved = info["service_scaled"] > best_scaled
                events.append(dict(attempt=attempts, donor_zone=donor, receiver_zone=receiver,
                                   service_scaled=info["service_scaled"], accepted=improved,
                                   optimal_solver_termination=info["optimal_solver_termination"]))
                if improved:
                    best, best_scaled, deployment = vector, info["service_scaled"], trial
                    accepted += 1
            if attempts >= 8 or perf_counter() >= deadline:
                break
        if attempts >= 8 or perf_counter() >= deadline:
            break
    integer_info = catalogue.validate(best)
    if integer_info[1] < seed_scaled:
        raise RuntimeError("recovery lost its feasible seed")
    if integer_info[2] > bound["value"]:
        raise RuntimeError("integer feasible recovery exceeds its exact dual upper")
    return deployment, best, dict(
        lp_info=lp_info, upper_bound=bound["upper_bound"], upper_bound_exact=bound["upper_bound_exact"],
        priced_root_count=bound["priced_root_count"], bound_time_s=bound_time,
        recovery_time_s=perf_counter()-evaluate_started, seed_service_scaled=seed_scaled,
        recovered_service_scaled=integer_info[1],
        recovery_gain=float(Fraction(integer_info[1]-seed_scaled, catalogue.scale)),
        exchange_attempts=attempts, accepted_exchanges=accepted, exchange_events=events,
        runtime_s=perf_counter()-started, common_build_s=common_build_s,
        end_to_end_method_s=common_build_s+perf_counter()-started,
        method_deadline_exceeded=perf_counter() > deadline,
        recovery_evaluator_scope="COMPLETE_BOUNDED_DECLARED_CATALOGUE_NOT_CITY_ENUMERATION",
        integer_witness_verified=True, old_controller_modified=False)
