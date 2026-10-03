"""Exact sequential lexicographic sparse assignment using SciPy/HiGHS MILP."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, vstack


@dataclass(frozen=True)
class AssignmentArc:
    vehicle_id: int
    request_id: int
    pickup_eta_s: float
    critical: bool
    carry_over: bool
    payload: object = None
    vehicle_type: str = "HV"
    exposure_static: float = 0.0
    exposure_dynamic: float = 0.0
    exposure_speed: float = 0.0
    operating_cost: float = 0.0


@dataclass(frozen=True)
class LexicographicResult:
    selected_indices: tuple[int, ...]
    solve_time_s: float
    critical_matched: int
    total_matched: int
    carry_over_matched: int
    backend: str = "SCIPY_HIGHS_MILP_SEQUENTIAL_LEXICOGRAPHIC"
    enabled_gamma_constraint_count: int = 0
    cost_level_solved: bool = False
    pickup_eta_optimum_s: float = 0.0
    normalized_operating_cost: float = 0.0


def solve_lexicographic(
    arcs: list[AssignmentArc],
    *,
    exposure_state: object | None = None,
    gammas: dict[str, float | None] | None = None,
    cost_level_enabled: bool = False,
    pickup_cost_epsilon: float = 0.0,
    numerical_tolerance: float = 1e-7,
) -> LexicographicResult:
    started = time.perf_counter()
    if not arcs:
        return LexicographicResult((), 0.0, 0, 0, 0)
    vehicles = {
        value: index for index, value in enumerate(sorted({a.vehicle_id for a in arcs}))
    }
    requests = {
        value: index for index, value in enumerate(sorted({a.request_id for a in arcs}))
    }
    rows: list[int] = []
    cols: list[int] = []
    for column, arc in enumerate(arcs):
        rows.extend(
            (vehicles[arc.vehicle_id], len(vehicles) + requests[arc.request_id])
        )
        cols.extend((column, column))
    base = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(len(vehicles) + len(requests), len(arcs)),
    )
    matrix = base
    lower = np.full(base.shape[0], -np.inf)
    upper = np.ones(base.shape[0])
    enabled_gamma_count = 0
    gamma_values = gammas or {}
    if exposure_state is not None:
        for family in ("static", "dynamic", "speed"):
            gamma = gamma_values.get(family)
            if gamma is None:
                continue
            coefficients = np.asarray(
                [
                    (
                        float(getattr(arc, f"exposure_{family}")) - float(gamma)
                        if arc.vehicle_type == "AV"
                        else 0.0
                    )
                    for arc in arcs
                ]
            )
            rhs = float(gamma) * int(exposure_state.av_assignments) - float(
                getattr(exposure_state, family)
            )
            matrix = vstack(
                [matrix, csr_matrix(coefficients.reshape(1, -1))], format="csr"
            )
            lower = np.append(lower, -np.inf)
            upper = np.append(upper, rhs)
            enabled_gamma_count += 1
    integrality = np.ones(len(arcs), dtype=int)
    bounds = Bounds(np.zeros(len(arcs)), np.ones(len(arcs)))

    levels = [
        np.asarray([1.0 if arc.critical else 0.0 for arc in arcs]),
        np.ones(len(arcs)),
        np.asarray([1.0 if arc.carry_over else 0.0 for arc in arcs]),
    ]
    optima: list[int] = []
    for objective in levels:
        if not objective.any():
            optima.append(0)
            continue
        result = milp(
            -objective,
            integrality=integrality,
            bounds=bounds,
            constraints=LinearConstraint(matrix, lower, upper),
            options={"presolve": True},
        )
        if not result.success or result.x is None:
            raise RuntimeError(f"lexicographic MILP failed: {result.message}")
        optimum = int(round(float(objective @ result.x)))
        optima.append(optimum)
        matrix = vstack([matrix, csr_matrix(objective.reshape(1, -1))], format="csr")
        lower = np.append(lower, optimum)
        upper = np.append(upper, optimum)
    eta = np.asarray([float(arc.pickup_eta_s) for arc in arcs])
    result = milp(
        eta,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(matrix, lower, upper),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"pickup-ETA MILP failed: {result.message}")
    last_x = result.x
    pickup_optimum = float(eta @ last_x)
    cost_solved = False
    operating_cost = np.asarray([float(arc.operating_cost) for arc in arcs])
    if cost_level_enabled:
        if float(pickup_cost_epsilon) < 0.0:
            raise ValueError("pickup_cost_epsilon must be >= 0")
        pickup_upper = (1.0 + float(pickup_cost_epsilon)) * pickup_optimum + float(
            numerical_tolerance
        )
        cost_matrix = vstack([matrix, csr_matrix(eta.reshape(1, -1))], format="csr")
        cost_lower = np.append(lower, -np.inf)
        cost_upper = np.append(upper, pickup_upper)
        result = milp(
            operating_cost,
            integrality=integrality,
            bounds=bounds,
            constraints=LinearConstraint(cost_matrix, cost_lower, cost_upper),
            options={"presolve": True},
        )
        if not result.success or result.x is None:
            raise RuntimeError(f"operating-cost MILP failed: {result.message}")
        last_x = result.x
        cost_solved = True
    selected = tuple(int(index) for index in np.flatnonzero(last_x > 0.5))
    while len(optima) < 3:
        optima.append(0)
    return LexicographicResult(
        selected,
        time.perf_counter() - started,
        optima[0],
        optima[1],
        optima[2],
        enabled_gamma_constraint_count=enabled_gamma_count,
        cost_level_solved=cost_solved,
        pickup_eta_optimum_s=pickup_optimum,
        normalized_operating_cost=float(operating_cost @ last_x),
    )
