"""Sparse nonanticipative placement LP over the existing historical time DAG.

Placement counts are shared across scenarios; service-path variables are not.
Integer placement recovery is feasible, not an integer service-chain certificate.
No road query, target-day request or prediction refit enters this module.
"""
from __future__ import annotations

from collections import Counter
from math import isfinite
from time import perf_counter
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, linprog
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack


def integer_quotas(weights, count, site_ids):
    values = np.asarray(weights, float)
    if (values.ndim != 1 or len(values) != len(site_ids) or not len(values)
            or not np.all(np.isfinite(values)) or np.any(values < -1e-8)
            or type(count) is not int or count < 0):
        raise ValueError("invalid placement weights/count")
    values = np.maximum(values, 0.)
    if values.sum() <= 0:
        raise ValueError("zero placement weights need an explicitly chosen tie rule")
    exact = values / values.sum() * count
    result = np.floor(exact).astype(int)
    order = sorted(range(len(values)), key=lambda z: (-(exact[z] - result[z]), str(site_ids[z])))
    for z in order[:count - int(result.sum())]:
        result[z] += 1
    if int(result.sum()) != count or np.any(result < 0):
        raise RuntimeError("integer placement recovery lost a resource")
    return result


def compatible_head_weights(models, weights, group, zones):
    """Strong simple baseline: eligible first tasks, no continuation/competition."""
    ready, end = group
    return np.asarray([sum(weights[name] * len(model.entries(("C", z, ready, end)))
        for name, model in models.items()) for z in range(zones)], float)


def solve_placement(models, weights, fixed_supply, fresh_groups, site_ids, cfg, check, *, tie_targets=None):
    """Bounded column generation with one placement across all history scenarios.

    Each fixed resource has its own state capacity. Each fresh group has one
    shared count per site, constraining EVERY scenario. One physical forecast
    customer has one capacity row across HV/C within its scenario.
    """
    started = perf_counter()
    limit = float(cfg.get("layout_bin_wall_limit_s", 30.))
    if not 0 < limit <= 30 or not models or set(models) != set(weights):
        raise ValueError("invalid offline layout budget/scenarios")
    if not np.isclose(sum(weights.values()), 1.) or any(v <= 0 for v in weights.values()):
        raise ValueError("scenario weights must be positive and sum to one")
    groups = sorted(fresh_groups)
    if not groups:
        return {}, dict(status="NO_ACTIVE_COARSE_C_BIRTH_WINDOW", variables=0, nonzeros=0,
            pricing_closed=True, integer_placement_supply_preserved=True, wall_time_s=0.)
    if any(type(v) is not int or v <= 0 for v in fresh_groups.values()):
        raise ValueError("fresh placement groups require positive integer supply")
    if any(not isfinite(float(v)) or v <= 0 for v in fixed_supply.values()):
        raise ValueError("invalid fixed supply capacity")
    zones = len(site_ids)
    if not zones or len(set(site_ids)) != zones:
        raise ValueError("layout needs unique common candidate sites")
    deadline = started + limit

    def budget():
        check()
        if perf_counter() >= deadline:
            raise TimeoutError("bounded coarse layout LP budget exhausted")

    placement = {(group, zone): column for column, (group, zone) in
        enumerate((group, zone) for group in groups for zone in range(zones))}
    roots, rhs, root_rows, task_rows = [], [], {}, {}
    for name in sorted(models):
        model = models[name]
        model.set_prices(np.zeros(len(model.tasks)), {})
        for key in sorted(fixed_supply):
            root_rows[name, "FIXED", key] = len(rhs)
            rhs.append(float(fixed_supply[key]))
            roots.append((name, "FIXED", key))
        for group in groups:
            for zone in range(zones):
                key = ("C", zone, *group)
                root_rows[name, "FRESH", key] = len(rhs)
                rhs.append(0.)
                roots.append((name, "FRESH", key))
        for task in range(len(model.tasks)):
            task_rows[name, task] = len(rhs)
            rhs.append(1.)
    equality = coo_matrix((np.ones(len(placement)),
        ([groups.index(group) for group, zone in placement], list(placement.values()))),
        shape=(len(groups), len(placement))).tocsr()
    columns, seen = [], set()

    def add(root, path):
        signature = root, tuple(path)
        if path and signature not in seen:
            if len(path) != len(set(path)) or len(path) > 3:
                raise ValueError("layout path repeats a forecast customer")
            seen.add(signature)
            columns.append(signature)

    for ordinal, root in enumerate(roots):
        if ordinal % 64 == 0:
            budget()
        name, _, key = root
        add(root, models[name].price(key)[1])
    last = last_model = None
    info = dict(status="RESTRICTED_PLACEMENT_LP", pricing_closed=False, pricing_rounds=0,
        lp_time_s=0., pricing_time_s=0., placement_variables=len(placement), scenario_count=len(models),
        shared_scenario_placement=True, shared_customer_capacity=True, full_city_optimality_claimed=False,
        integer_service_optimality_claimed=False, dense_matrix=False)
    for iteration in range(int(cfg.get("coarse_value_max_pricing_rounds", 6))):
        try:
            budget()
            variable_count = len(placement) + len(columns)
            if variable_count > cfg.get("coarse_value_max_columns", 20000):
                info["status"] = "APPROXIMATE_PLACEMENT_COLUMN_LIMIT"
                break
            row, col, data = [], [], []
            for (name, kind, key), index in root_rows.items():
                if kind == "FRESH":
                    _, zone, ready, end = key
                    row.append(index); col.append(placement[(ready, end), zone]); data.append(-1.)
            costs = np.zeros(variable_count)
            for offset, (root, path) in enumerate(columns):
                if offset % 256 == 0:
                    budget()
                name, _, _ = root
                column = len(placement) + offset
                costs[column] = -weights[name] * len(path)
                for index in (root_rows[root], *(task_rows[name, node] for node in path)):
                    row.append(index); col.append(column); data.append(1.)
            if len(data) + len(placement) > cfg.get("coarse_value_max_nonzeros", 150000):
                info["status"] = "APPROXIMATE_PLACEMENT_NONZERO_LIMIT"
                break
            upper = coo_matrix((data, (row, col)), shape=(len(rhs), variable_count)).tocsr()
            equal = coo_matrix((equality.data, (equality.nonzero()[0], equality.nonzero()[1])),
                shape=(len(groups), variable_count)).tocsr()
            lp_started = perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", OptimizeWarning)
                solved = linprog(costs, A_ub=upper, b_ub=rhs, A_eq=equal,
                    b_eq=[fresh_groups[group] for group in groups], bounds=(0, None), method="highs-ds",
                    options={"presolve": True, "threads": 1, "time_limit": max(.001, deadline - perf_counter())})
            for warning in caught:
                if "Unrecognized options detected" not in str(warning.message):
                    raise RuntimeError(f"layout LP warning: {warning.message}")
            info["lp_time_s"] += perf_counter() - lp_started
            if solved.status == 1:
                info["status"] = "APPROXIMATE_PLACEMENT_TIME_LIMIT"
                break
            if not solved.success or solved.status != 0 or not np.all(np.isfinite(solved.x)):
                raise RuntimeError(f"idle-feasible placement LP failed: {solved.message}")
            if (np.max(np.abs(equal @ solved.x - [fresh_groups[g] for g in groups]), initial=0.) > 1e-6
                    or np.max(upper @ solved.x - rhs, initial=0.) > 1e-6):
                raise RuntimeError("layout LP violated supply or customer capacity")
            last = solved.x[:len(placement)].copy()
            last_model = upper, equal, costs, float(solved.fun), solved.x
            info.update(pricing_rounds=iteration + 1, variables=variable_count,
                nonzeros=len(data) + len(placement), lp_expected_services=float(-solved.fun),
                source_supply=sum(fixed_supply.values()), fresh_supply=sum(fresh_groups.values()))
            margins = np.asarray(solved.ineqlin.marginals)
            for name, model in models.items():
                alpha = np.maximum(0., -margins[[task_rows[name, i] for i in range(len(model.tasks))]] / weights[name])
                model.set_prices(alpha, {})
            priced = perf_counter()
            added = 0
            for ordinal, root in enumerate(roots):
                if ordinal % 64 == 0:
                    budget()
                name, _, key = root
                value, path = models[name].price(key)
                if weights[name] * value > -margins[root_rows[root]] + 1e-7 and (root, tuple(path)) not in seen:
                    add(root, path)
                    added += 1
            info["pricing_time_s"] += perf_counter() - priced
            if not added:
                info["pricing_closed"] = True
                break
        except TimeoutError:
            info["status"] = "APPROXIMATE_PLACEMENT_TIME_LIMIT"
            break
    if last is None:
        raise RuntimeError("no solved placement LP; no silent replacement by a hotspot layout")
    info["tie_status"] = "IDENTITY_STABLE_LP_RECOVERY"
    if tie_targets is not None:
        # Lexicographic tie projection, not a small ad-hoc weight in the service
        # objective. Prefer the strong simple baseline only on the solved face.
        target = np.asarray([tie_targets[group][zone] for group, zone in placement], float)
        if (not np.all(np.isfinite(target)) or np.any(target < 0)
                or any(abs(sum(tie_targets[group]) - fresh_groups[group]) > 1e-7 for group in groups)):
            raise ValueError("tie targets do not preserve fresh supply")
        upper, equal, costs, optimum, primary = last_model
        ns, nv = len(placement), len(costs)
        selector = coo_matrix((np.ones(ns), (np.arange(ns), np.arange(ns))), shape=(ns, nv)).tocsr()
        block = vstack((hstack((upper, csr_matrix((upper.shape[0], ns)))),
            hstack((selector, -eye(ns, format="csr"))), hstack((-selector, -eye(ns, format="csr"))),
            csr_matrix(np.r_[costs, np.zeros(ns)].reshape(1, -1))), format="csr")
        equality_block = hstack((equal, csr_matrix((len(groups), ns))), format="csr")
        if (nv + ns <= cfg.get("coarse_value_max_columns", 20000)
                and block.nnz + equality_block.nnz <= cfg.get("coarse_value_max_nonzeros", 150000)
                and deadline - perf_counter() > .01):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", OptimizeWarning)
                tied = linprog(np.r_[np.zeros(nv), np.ones(ns)], A_ub=block,
                    b_ub=np.r_[rhs, target, -target, optimum + 1e-7], A_eq=equality_block,
                    b_eq=[fresh_groups[group] for group in groups], bounds=(0, None), method="highs-ds",
                    options={"presolve": True, "threads": 1, "time_limit": max(.001, deadline - perf_counter())})
            for warning in caught:
                if "Unrecognized options detected" not in str(warning.message):
                    raise RuntimeError(f"layout tie LP warning: {warning.message}")
            if tied.success and tied.status == 0 and float(costs @ tied.x[:nv]) <= optimum + 2e-7:
                if np.max(block @ tied.x - np.r_[rhs, target, -target, optimum + 1e-7], initial=0.) > 1e-6:
                    raise RuntimeError("layout tie LP violated the primary face")
                last = tied.x[:ns].copy()
                info["tie_status"] = "COMPATIBLE_HOTSPOT_L1_ON_RESTRICTED_PRIMARY_FACE"
            else:
                info["tie_status"] = "TIE_TIME_LIMIT_PRIMARY_PLACEMENT_RETAINED"
        else:
            info["tie_status"] = "TIE_DOMAIN_OR_TIME_LIMIT_PRIMARY_PLACEMENT_RETAINED"
    fractional = {group: np.asarray([last[placement[group, zone]] for zone in range(zones)]) for group in groups}
    recovered = {group: integer_quotas(values, fresh_groups[group], site_ids) for group, values in fractional.items()}
    info.update(wall_time_s=perf_counter() - started,
        fractional_placement_cells=int(sum(np.count_nonzero(np.abs(v - np.round(v)) > 1e-7) for v in fractional.values())),
        placement_recovery="PER_GROUP_LARGEST_REMAINDER_IDENTITY_STABLE",
        lp_score_is_before_integer_recovery=True, post_recovery_service_score_available=False,
        integer_placement_supply_preserved=True)
    return recovered, info
