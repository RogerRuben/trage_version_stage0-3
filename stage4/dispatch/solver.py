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


@dataclass(frozen=True)
class LexicographicResult:
    selected_indices: tuple[int, ...]
    solve_time_s: float
    critical_matched: int
    total_matched: int
    carry_over_matched: int
    backend: str = "SCIPY_HIGHS_MILP_SEQUENTIAL_LEXICOGRAPHIC"


def solve_lexicographic(arcs: list[AssignmentArc]) -> LexicographicResult:
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
    integrality = np.ones(len(arcs), dtype=int)
    bounds = Bounds(np.zeros(len(arcs)), np.ones(len(arcs)))

    levels = [
        np.asarray([1.0 if arc.critical else 0.0 for arc in arcs]),
        np.ones(len(arcs)),
        np.asarray([1.0 if arc.carry_over else 0.0 for arc in arcs]),
    ]
    optima: list[int] = []
    last_x: np.ndarray | None = None
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
        last_x = result.x
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
    selected = tuple(int(index) for index in np.flatnonzero(last_x > 0.5))
    while len(optima) < 3:
        optima.append(0)
    return LexicographicResult(
        selected, time.perf_counter() - started, optima[0], optima[1], optima[2]
    )
