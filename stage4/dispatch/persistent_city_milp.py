"""Append-only city MILP reuse using SciPy's SAME bundled HiGHS engine.

Only objectives and exact equality-lock rows change between lexicographic
tiers. The prior certified binary solution is a warm start, never a fallback.
No epsilon objective faces, relaxed integrality or extra time are introduced.
"""
from __future__ import annotations

import numpy as np
import weakref
from scipy.optimize import OptimizeResult


class BundledIntegerSession:
    def __init__(self, receipt):
        from scipy.optimize._highspy import _core
        self.core = _core
        self.h = _core._Highs()
        if self.h.version() != receipt["highs_version"]:
            raise RuntimeError("persistent session must use the verified SciPy-bundled engine")
        for name, value in {**receipt["options"], "mip_rel_gap":0., "presolve":"on", "output_flag":False}.items():
            self._checked(self.h.setOptionValue(name, value))
            status, actual = self.h.getOptionValue(name)
            if status != _core.HighsStatus.kOk or actual != value:
                raise RuntimeError(f"persistent numerical option readback disagrees: {name}")
        self.owner = None
        self.previous = None
        self.lower = self.upper = None
        self.binary = None
        self.model_loads = self.incremental_rows = self.warm_starts = self.runs = 0

    def _checked(self, status):
        if status != self.core.HighsStatus.kOk:
            raise RuntimeError(f"bundled persistent HiGHS rejected an operation: {status}")

    def solve(self, objective, rows, matrix, remaining):
        core, h = self.core, self.h
        csr = matrix.tocsr()
        if self.owner is None:
            self.owner = weakref.ref(rows)
            csc = matrix.tocsc()
            lp = core.HighsLp()
            lp.num_col_, lp.num_row_ = rows.columns, len(rows.lower)
            lp.col_cost_ = np.asarray(objective, dtype=np.float64)
            lp.col_lower_, lp.col_upper_ = np.zeros(rows.columns), np.ones(rows.columns)
            lp.row_lower_, lp.row_upper_ = np.asarray(rows.lower), np.asarray(rows.upper)
            lp.a_matrix_.format_ = core.MatrixFormat.kColwise
            lp.a_matrix_.start_ = np.asarray(csc.indptr, dtype=np.int32)
            lp.a_matrix_.index_ = np.asarray(csc.indices, dtype=np.int32)
            lp.a_matrix_.value_ = csc.data
            lp.integrality_ = [core.HighsVarType.kInteger] * rows.columns
            self._checked(h.passModel(lp))
            self.model_loads += 1
        else:
            old_rows = self.previous.shape[0]
            if (rows is not self.owner() or matrix.shape[1] != self.previous.shape[1]
                    or matrix.shape[0] < old_rows or (csr[:old_rows] != self.previous).nnz
                    or not np.array_equal(np.asarray(rows.lower[:old_rows]), self.lower)
                    or not np.array_equal(np.asarray(rows.upper[:old_rows]), self.upper)):
                raise RuntimeError("persistent city model changed an existing column/row")
            added = csr[old_rows:]
            if added.shape[0]:
                self._checked(h.addRows(added.shape[0], np.asarray(rows.lower[old_rows:]),
                    np.asarray(rows.upper[old_rows:]), added.nnz,
                    np.asarray(added.indptr, dtype=np.int32), np.asarray(added.indices, dtype=np.int32), added.data))
                self.incremental_rows += added.shape[0]
        self.previous = csr
        self.lower, self.upper = np.asarray(rows.lower).copy(), np.asarray(rows.upper).copy()
        columns = np.arange(rows.columns, dtype=np.int32)
        self._checked(h.changeColsCost(rows.columns, columns, np.asarray(objective, dtype=np.float64)))
        if self.binary is not None:
            self._checked(h.setSolution(rows.columns, columns, self.binary))
            self.warm_starts += 1
        self._checked(h.setOptionValue("time_limit", float(remaining())))
        self._checked(h.run())
        self.runs += 1
        status = h.getModelStatus()
        mapping = {core.HighsModelStatus.kOptimal:0, core.HighsModelStatus.kTimeLimit:1,
            core.HighsModelStatus.kIterationLimit:1, core.HighsModelStatus.kInfeasible:2,
            core.HighsModelStatus.kUnbounded:3}
        code = mapping.get(status, 4)
        solution, info = h.getSolution(), h.getInfo()
        x = np.asarray(solution.col_value, dtype=float) if solution.value_valid else None
        return OptimizeResult(status=code, success=code == 0, x=x,
            fun=float(info.objective_function_value) if x is not None else None,
            message=h.modelStatusToString(status), mip_gap=float(info.mip_gap),
            mip_dual_bound=float(info.mip_dual_bound), mip_node_count=int(info.mip_node_count))

    def accept_certified_binary(self, binary):
        self.binary = np.asarray(binary, dtype=float).copy()

    def diagnostics(self):
        return dict(backend="SCIPY_BUNDLED_HIGHS_PERSISTENT", highs_version=self.h.version(),
            model_loads=self.model_loads, incremental_exact_rows=self.incremental_rows,
            certified_binary_warm_starts=self.warm_starts, solver_runs=self.runs,
            incumbent_fallback=False, objective_faces_use_epsilon=False)
