"""Private, bounded numerical observation of a caller-supplied legacy module.

No routes, native actors, data loaders, solver settings, or solver mathematics
are changed here. Saved vectors are observed during a frozen-input
reconstruction, never presented as recovery of the original missing vector.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
import hashlib
import inspect
import json
from math import isfinite
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import issparse, save_npz, vstack

from stage4.dispatch import flexibility_model as model


SCHEMA = "CITY_NUMERIC_PRIVATE_PROBLEM_V1"
SCOPE = "SAME_FROZEN_CODE_INPUT_RECONSTRUCTION_NOT_RECOVERED_ORIGINAL_VECTOR"
MAX_VARIABLES = 20_000
MAX_NONZEROS = 150_000
_REGISTRY = {name: getattr(model, name) for name in (
    "Problem", "Vehicle", "Request", "CurrentPickup", "FuturePickup", "Scenario", "ModelLimits")}
_LEVEL_NAMES = ("critical_now", "expected_total_service", "current_service", "current_carry_over", "current_pickup_eta_s")


def _encode(value):
    if isinstance(value, np.generic):
        value = value.item()
    if is_dataclass(value):
        name = type(value).__name__
        if name not in _REGISTRY:
            raise ValueError("unsupported problem dataclass")
        return {"type": "dataclass", "class": name,
                "fields": {field.name: _encode(getattr(value, field.name)) for field in fields(value)}}
    if isinstance(value, float):
        return {"type": "float64", "hex": value.hex()}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_encode(item) for item in value]}
    if isinstance(value, frozenset):
        items = [_encode(item) for item in value]
        return {"type": "frozenset", "items": sorted(items, key=lambda item: json.dumps(item, sort_keys=True))}
    if isinstance(value, list):
        return {"type": "list", "items": [_encode(item) for item in value]}
    if isinstance(value, dict):
        return {"type": "mapping", "items": [[_encode(key), _encode(item)] for key, item in value.items()]}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise ValueError("unsupported private problem value type")


def _decode(value):
    if not isinstance(value, dict):
        return value
    kind = value.get("type")
    if kind == "float64":
        return float.fromhex(value["hex"])
    if kind == "dataclass":
        return _REGISTRY[value["class"]](**{key: _decode(item) for key, item in value["fields"].items()})
    items = value.get("items", ())
    if kind == "tuple":
        return tuple(_decode(item) for item in items)
    if kind == "frozenset":
        return frozenset(_decode(item) for item in items)
    if kind == "list":
        return [_decode(item) for item in items]
    if kind == "mapping":
        return {_decode(key): _decode(item) for key, item in items}
    raise ValueError("unknown private problem encoding")


def _json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_problem(problem, path):
    _json(path, dict(schema_version=SCHEMA, scope=SCOPE, problem=_encode(problem)))
    return Path(path)


def load_problem(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("unsupported saved numeric problem schema")
    problem = _decode(payload["problem"])
    if not isinstance(problem, model.Problem):
        raise ValueError("saved artifact is not a complete Problem")
    return problem


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if isfinite(value) else None


def _anonymous_error(error):
    if "noninteger common first action" in str(error).casefold():
        kind = "LEGACY_NONINTEGER_COMMON_FIRST_ACTION_POSTCHECK"
    elif isinstance(error, TimeoutError) or "time limit" in str(error).casefold() or "timeout" in str(error).casefold():
        kind = "TIME_BUDGET_OR_OPTIMALITY_FAILURE"
    else:
        kind = "SOLVER_OR_NUMERICAL_CONTRACT_EXCEPTION"
    return dict(type=type(error).__name__, kind=kind)


class LegacyNumericProbe:
    """Temporarily wrap one legacy component compiler and its shared base MILP.

    observe_problem stores the entire epoch input without changing it. Only
    the most recent full numerical snapshot is retained in memory. Public
    summaries contain aggregate values; identity mappings are private files.
    close restores both functions even after the observed numerical failure.
    """
    def __init__(self, legacy_numeric, out_dir):
        self.legacy_numeric = legacy_numeric
        self.out_dir = Path(out_dir)
        self.last_problem = None
        self.last_failure_path = None
        self.stage_summaries = []
        self.component_summaries = []
        self._last_snapshot = None
        self._last_component = None
        self._context = None
        self._installed = False
        self._solve_ordinal = self._component_ordinal = 0

    def observe_problem(self, problem, now_s=None):
        if not isinstance(problem, model.Problem):
            raise ValueError("probe requires the complete frozen base Problem")
        self.last_problem = problem
        self.now_s = float(problem.now_s if now_s is None else now_s)
        self._solve_ordinal += 1
        self._component_ordinal = 0

    def install(self):
        if self._installed:
            return self
        legacy = self.legacy_numeric
        if getattr(legacy.base.milp, "_city_numeric_probe_owner", None) is not None:
            raise RuntimeError("close the existing numeric probe before installing another")
        self._original_component, self._original_milp = legacy._solve_component, legacy.base.milp
        signature = inspect.signature(self._original_component)

        def component_wrapper(*args, **kwargs):
            values = signature.bind(*args, **kwargs).arguments
            problem, waiting, options = values["problem"], values["waiting"], values["options"]
            by_vehicle, option_ids, records = values["by_vehicle"], values["option_ids"], values["records"]
            if self.last_problem is None:
                self.observe_problem(problem)
            self._component_ordinal += 1
            free_ids = [j for j in option_ids if len(by_vehicle[options[j].vehicle_id]) > 1]
            count = len(free_ids) + len(records)
            vectors = [np.zeros(count) for _ in range(5)]
            for index, j in enumerate(free_ids):
                option = options[j]
                if option.request_id is not None:
                    request = waiting[option.request_id]
                    vectors[0][index], vectors[1][index], vectors[2][index], vectors[3][index], vectors[4][index] = (
                        request.critical, 1., 1., request.carry_over, option.pickup_eta_s)
            for index, (scene, _, _) in enumerate(records, len(free_ids)):
                vectors[1][index] = scene.probability
            active = [(ordinal, name, -vector if ordinal < 5 else vector)
                      for ordinal, (name, vector) in enumerate(zip(_LEVEL_NAMES, vectors), 1) if np.any(vector)]
            context = dict(solve_ordinal=self._solve_ordinal, component_ordinal=self._component_ordinal,
                call_ordinal=0, active=active,
                skipped=[ordinal for ordinal, vector in enumerate(vectors, 1) if not np.any(vector)],
                private=dict(free_columns=[dict(column=index, option_index=j, vehicle_id=options[j].vehicle_id,
                                               request_id=options[j].request_id) for index, j in enumerate(free_ids)],
                             scenario_ids=sorted({scene.scenario_id for scene, _, _ in records})))
            previous = self._context
            self._context = context
            self._last_component = context
            started = perf_counter()
            try:
                result = self._original_component(*args, **kwargs)
                status = "SUCCESS"
                return result
            except Exception as error:
                status = "FAILED"
                context["error"] = _anonymous_error(error)
                if context["error"]["kind"] == "LEGACY_NONINTEGER_COMMON_FIRST_ACTION_POSTCHECK":
                    self._save_failure(context)
                raise
            finally:
                self.component_summaries.append(dict(solve_ordinal=context["solve_ordinal"],
                    component_ordinal=context["component_ordinal"], status=status,
                    active_legacy_level_ordinals=[item[0] for item in active],
                    skipped_constant_level_ordinals=context["skipped"],
                    runtime_s=perf_counter() - started, error=context.get("error")))
                self._context = previous

        def milp_wrapper(*args, **kwargs):
            result = self._original_milp(*args, **kwargs)
            if self._context is not None:
                self._record_milp(args, kwargs, result)
            return result

        milp_wrapper._city_numeric_probe_owner = self
        self._wrapped_component, self._wrapped_milp = component_wrapper, milp_wrapper
        legacy._solve_component, legacy.base.milp = component_wrapper, milp_wrapper
        self._installed = True
        return self

    def close(self):
        if self._installed:
            self.legacy_numeric._solve_component = self._original_component
            self.legacy_numeric.base.milp = self._original_milp
            self._installed = False
        if self.stage_summaries or self.component_summaries:
            _json(self.out_dir / "stage_summaries.json", dict(scope=SCOPE, stages=self.stage_summaries,
                                                            components=self.component_summaries))

    def _record_milp(self, args, kwargs, result):
        c = np.asarray(args[0] if args else kwargs["c"], dtype=float).copy()
        if len(c) > MAX_VARIABLES:
            raise ValueError("numeric probe variable cap exceeded")
        constraints = kwargs["constraints"]
        constraints = [constraints] if isinstance(constraints, LinearConstraint) else list(constraints)
        if any(not issparse(constraint.A) for constraint in constraints):
            raise ValueError("numeric probe refuses dense constraint snapshots")
        matrix = constraints[0].A.copy() if len(constraints) == 1 else vstack([constraint.A for constraint in constraints], format="csr")
        if matrix.nnz > MAX_NONZEROS:
            raise ValueError("numeric probe sparse nonzero cap exceeded")
        lower = np.concatenate([np.broadcast_to(constraint.lb, constraint.A.shape[0]) for constraint in constraints]).copy()
        upper = np.concatenate([np.broadcast_to(constraint.ub, constraint.A.shape[0]) for constraint in constraints]).copy()
        bounds = kwargs.get("bounds") or Bounds(0., np.inf)
        column_lower, column_upper = (np.broadcast_to(value, len(c)).copy() for value in (bounds.lb, bounds.ub))
        raw_integrality = kwargs.get("integrality")
        mask = np.broadcast_to(0 if raw_integrality is None else raw_integrality, len(c)).copy()
        x = None if result.x is None else np.asarray(result.x, dtype=float).copy()
        context = self._context
        context["call_ordinal"] += 1
        expected = context["active"][context["call_ordinal"] - 1] if context["call_ordinal"] <= len(context["active"]) else None
        matches = expected is not None and np.array_equal(c, expected[2])
        summary = dict(solve_ordinal=context["solve_ordinal"], component_ordinal=context["component_ordinal"],
            active_call_ordinal=context["call_ordinal"], legacy_stage_ordinal=expected[0] if matches else None,
            stage=expected[1] if matches else "UNMAPPED_LEGACY_MILP_CALL", legacy_objective_matched=matches,
            skipped_constant_level_ordinals=context["skipped"], status=int(result.status),
            message=str(result.message), variables=len(c), binary_variable_count=int(np.count_nonzero(mask == 1)),
            rows=matrix.shape[0], nonzeros=int(matrix.nnz), objective=_finite(getattr(result, "fun", None)),
            mip_dual_bound=_finite(getattr(result, "mip_dual_bound", None)), mip_gap=_finite(getattr(result, "mip_gap", None)),
            mip_node_count=_finite(getattr(result, "mip_node_count", None)), max_binary_deviation=None,
            max_raw_row_violation=None, max_raw_bound_violation=None)
        if x is not None:
            binary = x[mask == 1]
            lhs = matrix @ x
            summary.update(max_binary_deviation=_finite(np.max(np.abs(binary - np.rint(binary)), initial=0.)),
                max_raw_row_violation=_finite(max(0., float(np.max(lower - lhs, initial=0.)), float(np.max(lhs - upper, initial=0.)))),
                max_raw_bound_violation=_finite(max(0., float(np.max(column_lower - x, initial=0.)),
                                                     float(np.max(x - column_upper, initial=0.)))))
        self.stage_summaries.append(summary)
        self._last_snapshot = dict(matrix=matrix, arrays=dict(c=c, integrality=mask,
            x=np.empty(0) if x is None else x, row_lower=lower, row_upper=upper,
            column_lower=column_lower, column_upper=column_upper), x_available=x is not None,
            summary=summary, options=deepcopy(kwargs.get("options", {})))

    def _save_failure(self, context):
        if self._last_snapshot is None:
            raise RuntimeError("legacy numerical failure had no observed sparse solve")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        save_problem(self.last_problem, self.out_dir / "problem.json")
        snapshot = self._last_snapshot
        save_npz(self.out_dir / "last_milp_matrix.npz", snapshot["matrix"], compressed=True)
        np.savez_compressed(self.out_dir / "last_milp_arrays.npz", **snapshot["arrays"])
        _json(self.out_dir / "last_component_private.json", dict(context["private"],
              solver_options=_encode(snapshot["options"]), x_available=snapshot["x_available"]))
        self.last_failure_path = self.out_dir / "numerical_failure.json"
        _json(self.last_failure_path, dict(scope=SCOPE, original_missing_vector_recovered=False,
            now_s=self.now_s, observed_failure=context["error"], failed_component=context["component_ordinal"],
            last_stage=snapshot["summary"], snapshot_format=snapshot["matrix"].format,
            private_artifacts=["problem.json", "last_milp_matrix.npz", "last_milp_arrays.npz", "last_component_private.json"]))


def _decision_summary(problem, decision):
    requests = {request.request_id: request for request in problem.waiting_requests}
    etas = {(arc.vehicle_id, arc.request_id): arc.pickup_eta_s for arc in problem.current_pickups}
    pairs = decision.selected_pairs
    return dict(critical_now=sum(requests[rid].critical for _, rid in pairs),
        expected_total_service=float(decision.expected_total_service_count), current_service=len(pairs),
        current_carry_over=sum(requests[rid].carry_over for _, rid in pairs),
        current_pickup_eta_s=float(sum(etas[pair] for pair in pairs)))


def compare_saved_problem(problem_path, legacy_numeric, fixed_numeric):
    """Two ten-second solves of ONE saved input, not a native/window replay."""
    path = Path(problem_path)
    saved = load_problem(path)
    problem = replace(saved, limits=replace(saved.limits, solver_time_limit_s=10.0))
    directory = Path(tempfile.mkdtemp(prefix="comparison_legacy_", dir=path.parent))
    probe = LegacyNumericProbe(legacy_numeric, directory)
    probe.observe_problem(problem, problem.now_s)
    legacy = dict(status="NOT_RUN")
    probe.install()
    started = perf_counter()
    try:
        decision = legacy_numeric.solve_city_defer(problem)
        legacy.update(status="SUCCESS", objective=_decision_summary(problem, decision))
    except Exception as error:
        legacy.update(status="FAILED", error=_anonymous_error(error))
    finally:
        legacy["runtime_s"] = perf_counter() - started
        probe.close()  # Shared base.milp is restored BEFORE the fixed solve.
    legacy.update(same_integer_postcheck_failure=probe.last_failure_path is not None,
        max_binary_deviation=max((stage["max_binary_deviation"] for stage in probe.stage_summaries
                                  if stage["max_binary_deviation"] is not None), default=None),
        last_stage=deepcopy(probe.stage_summaries[-1]) if probe.stage_summaries else None,
        stage_count=len(probe.stage_summaries))
    fixed = dict(status="NOT_RUN")
    started = perf_counter()
    try:
        decision = fixed_numeric.solve_city_defer(problem)
        fixed.update(status="SUCCESS", objective=_decision_summary(problem, decision),
                     numerical_contract=deepcopy(getattr(decision, "numerical_contract", None)))
    except Exception as error:
        fixed.update(status="FAILED", error=_anonymous_error(error),
                     numerical_diagnostics=deepcopy(getattr(error, "diagnostics", None)))
    fixed["runtime_s"] = perf_counter() - started
    summary = dict(scope=SCOPE, original_missing_vector_recovered=False,
        saved_problem_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        comparison_budget_s_per_solver=10.0, saved_budget_s=saved.limits.solver_time_limit_s,
        source_parameters={field: getattr(saved.limits, field) for field in (
            "horizon_s", "rolling_step_s", "max_variables", "max_nonzeros", "recourse_mode", "solver_backend")},
        input_counts=dict(vehicles=len(saved.vehicles), waiting_requests=len(saved.waiting_requests),
                          current_arcs=len(saved.current_pickups), scenarios=len(saved.scenarios)),
        scenario_probabilities=[scene.probability for scene in saved.scenarios], legacy=legacy, fixed=fixed)
    _json(path.parent / "comparison.json", summary)
    return summary
