"""Opt-in persistent HiGHS model for the unchanged lexicographic objectives."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
import time

import numpy as np


def load_highspy(runtime_dir=None):
    try:
        return importlib.import_module("highspy")
    except ModuleNotFoundError as error:
        if error.name != "highspy" or runtime_dir is None:
            raise RuntimeError("HIGHS_PERSISTENT requires the optional highspy runtime") from error
        directory = Path(runtime_dir).resolve()
        if not (directory / "highspy").is_dir():
            raise RuntimeError("optional highspy runtime directory is missing") from error
        sys.path.insert(0, str(directory))
        try:
            return importlib.import_module("highspy")
        finally:
            sys.path.remove(str(directory))


def run_levels(rows, count, levels, limits, integrality=None):
    highspy = load_highspy(limits.highspy_runtime_dir)
    started = time.perf_counter()
    h = highspy.Highs()

    def checked(status):
        if status == highspy.HighsStatus.kError:
            raise RuntimeError("persistent HiGHS model/API error")

    for key, value in (("threads", 1), ("parallel", "off"), ("random_seed", 0),
                       ("mip_rel_gap", 0.0), ("mip_abs_gap", 0.0), ("output_flag", False)):
        checked(h.setOptionValue(key, value))
    matrix = rows.matrix(count).tocsc()
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = count, len(rows.lower)
    lp.col_cost_ = np.zeros(count)
    lp.col_lower_, lp.col_upper_ = np.zeros(count), np.ones(count)
    lp.row_lower_, lp.row_upper_ = np.asarray(rows.lower), np.asarray(rows.upper)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = np.asarray(matrix.indptr, dtype=np.int32)
    lp.a_matrix_.index_ = np.asarray(matrix.indices, dtype=np.int32)
    lp.a_matrix_.value_ = matrix.data
    mask = np.ones(count, dtype=np.int8) if integrality is None else integrality
    lp.integrality_ = [highspy.HighsVarType.kInteger if item else highspy.HighsVarType.kContinuous for item in mask]
    checked(h.passModel(lp))
    column_indices = np.arange(count, dtype=np.int32)
    active_levels = [(objective, maximize) for objective, maximize in levels if np.any(objective)]
    idle_only = not active_levels
    if not active_levels:
        active_levels = [(np.ones(count), False)]
    solution = None
    for objective, maximize in active_levels:
        remaining = limits.solver_time_limit_s - (time.perf_counter() - started)
        if remaining <= 0:
            raise RuntimeError("sparse model solver timeout")
        checked(h.setOptionValue("time_limit", remaining))
        checked(h.changeColsCost(count, column_indices, -objective if maximize else objective))
        if solution is not None:
            checked(h.setSolution(count, column_indices, solution))
        checked(h.run())
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"sparse model not proven optimal: {h.modelStatusToString(h.getModelStatus())}")
        solution = np.asarray(h.getSolution().col_value, dtype=float)
        optimum = float(objective @ solution)
        nonzero = np.flatnonzero(objective).astype(np.int32)
        if not idle_only:
            checked(h.addRow(optimum - 1e-7, optimum + 1e-7, len(nonzero), nonzero, objective[nonzero]))
            rows.add({int(j): float(objective[j]) for j in nonzero}, optimum - 1e-7, optimum + 1e-7)
    return solution, time.perf_counter() - started
