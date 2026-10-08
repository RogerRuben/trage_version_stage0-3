"""Sparse two-stage valuation of a common, executable current action.

Historical scenarios are forecast proxies. Their revealed-future recourse
chains approximate value and must never be counted as realized service. Only
the common first action is executed before the next decision epoch.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import math
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


MAX_TASKS_PER_SCENARIO = 20
MAX_RESOURCES = 3
MAX_SCENARIOS = 2
MAX_CHAINS_PER_RESOURCE_SCENARIO = 4_000
MAX_MASTER_VARIABLES = 20_000
MAX_MASTER_NONZEROS = 150_000
DECISION_TIME_LIMIT_S = 10.0
KIND = "TWO_STAGE_HISTORICAL_SCENARIO_RECEDING_HORIZON_NOT_MULTISTAGE_OPTIMUM"
POLICIES = ("SERVICE_PRESERVING", "CHAIN_DEFER")
_KINDS = frozenset(("SERVE", "WAIT", "BUSY", "RELOCATE", "LAYOUT"))
_TOLERANCE = 1e-7


class DecisionTimeout(TimeoutError):
    """The complete decision budget expired, or a stage was not optimal."""


class RollingModelLimitError(ValueError):
    """A declared enumeration or sparse master limit was exceeded."""


@dataclass(frozen=True)
class _Task:
    job_id: str
    release: Fraction
    deadline: Fraction
    service: Fraction
    profiles: frozenset[str]
    source_kind: str


@dataclass(frozen=True)
class _Resource:
    resource_id: str
    profile_id: str
    end: Fraction


@dataclass(frozen=True)
class _Action:
    action_id: str
    resource_id: str
    kind: str
    ready: Fraction
    location_id: str
    job_id: str | None
    distance: Fraction
    critical: bool
    carry_over: bool
    raw: dict


@dataclass(frozen=True)
class _Connection:
    travel: Fraction
    distance: Fraction
    supported: bool
    profiles: frozenset[str]


@dataclass(frozen=True)
class _Scenario:
    scenario_id: str
    weight: Fraction
    tasks: tuple[_Task, ...]
    connections: dict[tuple[str, str], _Connection]


@dataclass(frozen=True)
class _Problem:
    now: Fraction
    step: Fraction
    resources: tuple[_Resource, ...]
    actions: tuple[_Action, ...]
    scenarios: tuple[_Scenario, ...]


@dataclass(frozen=True)
class _Visit:
    job_id: str
    origin_id: str
    commit: Fraction
    pickup: Fraction
    finish: Fraction
    travel: Fraction
    distance: Fraction


@dataclass(frozen=True)
class _Path:
    resource_id: str
    scenario_id: str
    action_id: str
    first_job_id: str | None
    visits: tuple[_Visit, ...]
    mask: int
    distance: Fraction

    @property
    def served(self):
        return len(self.visits) + int(self.first_job_id is not None)


class _Budget:
    def __init__(self):
        self.started = perf_counter()

    def remaining(self, stage="decision"):
        remaining = DECISION_TIME_LIMIT_S - (perf_counter() - self.started)
        if remaining <= 0:
            raise DecisionTimeout(f"{stage}: total {DECISION_TIME_LIMIT_S:g}s decision budget expired")
        return remaining


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be a finite nonnegative number")
    return Fraction(str(value))


def _id(value, field):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _rows(problem, field, maximum=None):
    rows = problem.get(field)
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{field} must be a list of objects")
    if maximum is not None and len(rows) > maximum:
        raise RollingModelLimitError(f"{field} limit exceeded: maximum {maximum}")
    return rows


def _profiles(row, field):
    values = row.get("compatible_profiles")
    if not isinstance(values, list):
        raise ValueError(f"{field}.compatible_profiles must be a list")
    return frozenset(_id(value, f"{field}.compatible_profiles") for value in values)


def _same_number(row, field, expected, label):
    value = _number(row.get(field), f"{label}.{field}")
    if not math.isclose(float(value), float(expected), rel_tol=0.0, abs_tol=_TOLERANCE):
        raise ValueError(f"{label}: {field} disagrees with the reconstructed value")


def _prepare(problem):
    if not isinstance(problem, Mapping):
        raise ValueError("problem must be an object")
    now = _number(problem.get("now_s"), "now_s")
    step = _number(problem.get("step_s"), "step_s")
    if step != 30:
        raise ValueError("step_s must be 30")
    resources = []
    resource_ids = set()
    for row in _rows(problem, "resources", MAX_RESOURCES):
        resource_id = _id(row.get("resource_id"), "resource_id")
        if resource_id in resource_ids:
            raise ValueError("duplicate resource_id")
        resource_ids.add(resource_id)
        resources.append(_Resource(resource_id, _id(row.get("profile_id"), "profile_id"),
                                   _number(row.get("admission_end_s"), "admission_end_s")))
    if not resources:
        raise ValueError("expected at least one resource")
    scenarios = []
    scenario_ids = set()
    common_actual = None
    for row in _rows(problem, "scenarios", MAX_SCENARIOS):
        scenario_id = _id(row.get("scenario_id"), "scenario_id")
        if scenario_id in scenario_ids:
            raise ValueError("duplicate scenario_id")
        scenario_ids.add(scenario_id)
        weight = _number(row.get("weight"), "scenario.weight")
        if weight <= 0:
            raise ValueError("scenario weights must be positive")
        tasks = []
        job_ids = set()
        for task_row in _rows(row, "tasks", MAX_TASKS_PER_SCENARIO):
            job_id = _id(task_row.get("job_id"), "job_id")
            if job_id in job_ids:
                raise ValueError(f"{scenario_id}: duplicate job_id")
            job_ids.add(job_id)
            release = _number(task_row.get("release_s"), "release_s")
            deadline = _number(task_row.get("deadline_s"), "deadline_s")
            if release > deadline:
                raise ValueError("task release_s exceeds original deadline_s")
            source = task_row.get("source_kind")
            if source not in ("ACTUAL_PENDING", "FORECAST"):
                raise ValueError("task source_kind must be ACTUAL_PENDING or FORECAST")
            if source == "ACTUAL_PENDING" and release > now:
                raise ValueError("ACTUAL_PENDING task has not been released at now_s")
            tasks.append(_Task(job_id, release, deadline,
                               _number(task_row.get("service_time_s"), "service_time_s"),
                               _profiles(task_row, job_id), source))
        actual = {task.job_id: task for task in tasks if task.source_kind == "ACTUAL_PENDING"}
        if common_actual is not None and actual != common_actual:
            raise ValueError("all scenarios must contain identical ACTUAL_PENDING task attributes")
        common_actual = actual
        connections = {}
        for edge in _rows(row, "connections"):
            origin = _id(edge.get("origin_id"), "origin_id")
            target = _id(edge.get("target_job_id"), "target_job_id")
            if target not in job_ids or (origin, target) in connections:
                raise ValueError("unknown target or duplicate scenario connection")
            if type(edge.get("supported")) is not bool:
                raise ValueError("connection.supported must be a boolean")
            supported = edge["supported"]
            travel = (Fraction(0) if not supported and edge.get("travel_time_s") is None else
                      _number(edge.get("travel_time_s"), "connection.travel_time_s"))
            distance = (Fraction(0) if not supported and edge.get("empty_distance_m") is None else
                        _number(edge.get("empty_distance_m"), "connection.empty_distance_m"))
            connections[origin, target] = _Connection(travel, distance, supported,
                                                      _profiles(edge, "connection"))
        scenarios.append(_Scenario(scenario_id, weight, tuple(sorted(tasks, key=lambda task: task.job_id)), connections))
    if not scenarios or not math.isclose(float(sum((scenario.weight for scenario in scenarios), Fraction(0))),
                                          1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("expected positive scenario weights summing to one")
    # Normalize only the decimal representation's sum roundoff, preserving the
    # intended relative scenario probabilities and exact rational comparisons.
    weight_sum = sum((scenario.weight for scenario in scenarios), Fraction(0))
    scenarios = [_Scenario(scenario.scenario_id, scenario.weight / weight_sum,
                           scenario.tasks, scenario.connections) for scenario in scenarios]
    resource_map = {resource.resource_id: resource for resource in resources}
    actions = []
    action_ids = set()
    for row in _rows(problem, "actions"):
        action_id = _id(row.get("action_id"), "action_id")
        resource_id = _id(row.get("resource_id"), "action.resource_id")
        if action_id in action_ids or resource_id not in resource_map:
            raise ValueError("duplicate action_id or unknown action resource")
        action_ids.add(action_id)
        kind = row.get("kind")
        if kind not in _KINDS:
            raise ValueError("unknown first action kind")
        if type(row.get("critical")) is not bool or type(row.get("carry_over")) is not bool:
            raise ValueError("action critical/carry_over must be booleans")
        ready = _number(row.get("ready_s"), "action.ready_s")
        location = _id(row.get("location_id"), "action.location_id")
        distance = _number(row.get("empty_distance_m"), "action.empty_distance_m")
        job_id = row.get("job_id")
        if ready < now:
            raise ValueError("first action ready_s precedes now_s")
        if kind == "SERVE":
            job_id = _id(job_id, "SERVE.job_id")
            task = common_actual.get(job_id)
            resource = resource_map[resource_id]
            if task is None:
                raise ValueError("first SERVE must name a currently released ACTUAL_PENDING task, never FORECAST")
            if resource.profile_id not in task.profiles:
                raise ValueError("first SERVE has an incompatible resource profile")
            if now >= resource.end or location != job_id:
                raise ValueError("first SERVE violates resource admission end or completion location")
            pickup = ready - task.service
            if pickup < now or pickup > task.deadline:
                raise ValueError("first SERVE violates its original pickup deadline or service duration")
            if "commit_s" in row:
                _same_number(row, "commit_s", now, action_id)
            if "pickup_s" in row:
                _same_number(row, "pickup_s", pickup, action_id)
            if "travel_time_s" in row:
                _same_number(row, "travel_time_s", pickup - now, action_id)
            if "origin_id" in row:
                _id(row["origin_id"], "SERVE.origin_id")
        elif job_id is not None:
            raise ValueError("non-SERVE first actions cannot serve a customer")
        if kind == "WAIT" and ready < now + step:
            raise ValueError("WAIT must last at least one 30s decision step")
        if kind == "RELOCATE" and ready <= now:
            raise ValueError("RELOCATE must arrive strictly after now_s")
        if kind == "LAYOUT" and (now != 0 or ready != 0 or distance != 0):
            raise ValueError("LAYOUT is a zero-cost initial position choice only at time zero")
        actions.append(_Action(action_id, resource_id, kind, ready, location, job_id, distance,
                               row["critical"], row["carry_over"], dict(row)))
    if set(action.resource_id for action in actions) != resource_ids:
        raise ValueError("every resource must have at least one first action")
    if len(actions) > MAX_MASTER_VARIABLES:
        raise RollingModelLimitError("master variable limit exceeded by first actions")
    return _Problem(now, step, tuple(sorted(resources, key=lambda resource: resource.resource_id)),
                    tuple(sorted(actions, key=lambda action: (action.resource_id, action.action_id))),
                    tuple(sorted(scenarios, key=lambda scenario: scenario.scenario_id)))


def _ceil_step(value, step):
    quotient = value / step
    return ((quotient.numerator + quotient.denominator - 1) // quotient.denominator) * step


def _enumerate(model, budget):
    paths = []
    counts = []
    for resource in model.resources:
        actions = [action for action in model.actions if action.resource_id == resource.resource_id]
        for scenario in model.scenarios:
            bits = {task.job_id: 1 << index for index, task in enumerate(scenario.tasks)}
            count = 0
            per_action = {}

            def append(path):
                nonlocal count
                budget.remaining("chain enumeration")
                if count >= MAX_CHAINS_PER_RESOURCE_SCENARIO:
                    raise RollingModelLimitError(
                        f"chain limit exceeded for {resource.resource_id}/{scenario.scenario_id}; "
                        "enumeration was not truncated")
                if len(model.actions) + len(paths) >= MAX_MASTER_VARIABLES:
                    raise RollingModelLimitError("master variable limit exceeded; enumeration was not truncated")
                paths.append(path)
                count += 1

            for action in actions:
                initial_count = count
                first_job = action.job_id if action.kind == "SERVE" else None
                initial_mask = bits[first_job] if first_job else 0
                append(_Path(resource.resource_id, scenario.scenario_id, action.action_id,
                             first_job, (), initial_mask, action.distance))

                def extend(origin, ready, mask, visits, distance):
                    budget.remaining("chain enumeration")
                    for index, task in enumerate(scenario.tasks):
                        bit = 1 << index
                        if mask & bit or resource.profile_id not in task.profiles:
                            continue
                        connection = scenario.connections.get((origin, task.job_id))
                        if (connection is None or not connection.supported
                                or resource.profile_id not in connection.profiles):
                            continue
                        commit = _ceil_step(max(ready, task.release), model.step)
                        pickup = commit + connection.travel
                        if commit >= resource.end or pickup > task.deadline:
                            continue
                        finish = pickup + task.service
                        visit = _Visit(task.job_id, origin, commit, pickup, finish,
                                       connection.travel, connection.distance)
                        next_visits = visits + (visit,)
                        next_distance = distance + connection.distance
                        append(_Path(resource.resource_id, scenario.scenario_id, action.action_id,
                                     first_job, next_visits, mask | bit, next_distance))
                        extend(task.job_id, finish, mask | bit, next_visits, next_distance)

                extend(action.location_id, action.ready, initial_mask, (), action.distance)
                per_action[action.action_id] = count - initial_count
            counts.append(dict(resource_id=resource.resource_id, scenario_id=scenario.scenario_id,
                               chain_count=count, per_first_action=per_action))
    return tuple(paths), counts


class _SparseRows:
    def __init__(self, columns):
        self.columns = columns
        self.row_indices, self.column_indices, self.values = [], [], []
        self.lower, self.upper = [], []

    def add(self, entries, lower, upper):
        entries = [(index, float(value)) for index, value in entries if value != 0]
        if len(self.values) + len(entries) > MAX_MASTER_NONZEROS:
            raise RollingModelLimitError("master nonzero limit exceeded; no dense fallback or truncation")
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


def _master(model, paths):
    action_columns = {action.action_id: index for index, action in enumerate(model.actions)}
    offset = len(model.actions)
    rows = _SparseRows(offset + len(paths))
    grouped_paths = {}
    job_columns = {}
    for index, path in enumerate(paths, offset):
        grouped_paths.setdefault((path.resource_id, path.scenario_id, path.action_id), []).append(index)
        jobs = ((path.first_job_id,) if path.first_job_id else ()) + tuple(visit.job_id for visit in path.visits)
        for job_id in jobs:
            job_columns.setdefault((path.scenario_id, job_id), []).append(index)
    for resource in model.resources:
        actions = [action for action in model.actions if action.resource_id == resource.resource_id]
        rows.add([(action_columns[action.action_id], 1) for action in actions], 1, 1)
        for scenario in model.scenarios:
            for action in actions:
                columns = grouped_paths[resource.resource_id, scenario.scenario_id, action.action_id]
                rows.add([(index, 1) for index in columns] + [(action_columns[action.action_id], -1)], 0, 0)
    for scenario in model.scenarios:
        for task in scenario.tasks:
            rows.add([(index, 1) for index in job_columns.get((scenario.scenario_id, task.job_id), ())],
                     -np.inf, 1)
    current_job_actions = {}
    for action in model.actions:
        if action.kind == "SERVE":
            current_job_actions.setdefault(action.job_id, []).append(action_columns[action.action_id])
    for columns in current_job_actions.values():
        rows.add([(index, 1) for index in columns], -np.inf, 1)
    return rows, action_columns


def _objectives(model, paths):
    size = len(model.actions) + len(paths)
    names = ("critical_now", "current_served", "carry_over_current_served", "expected_served",
             "expected_empty_distance_m")
    coefficients = {name: [Fraction(0)] * size for name in names}
    for index, action in enumerate(model.actions):
        if action.kind == "SERVE":
            coefficients["critical_now"][index] = Fraction(int(action.critical))
            coefficients["current_served"][index] = Fraction(1)
            coefficients["carry_over_current_served"][index] = Fraction(int(action.carry_over))
    weights = {scenario.scenario_id: scenario.weight for scenario in model.scenarios}
    for index, path in enumerate(paths, len(model.actions)):
        coefficients["expected_served"][index] = weights[path.scenario_id] * path.served
        coefficients["expected_empty_distance_m"][index] = weights[path.scenario_id] * path.distance
    return coefficients


def _milp_selection(model, paths, rows, action_columns, policy, budget):
    coefficients = _objectives(model, paths)
    if policy == "SERVICE_PRESERVING":
        order = ["critical_now", "current_served", "carry_over_current_served", "expected_served",
                 "expected_empty_distance_m"]
    else:
        order = ["critical_now", "expected_served", "current_served", "carry_over_current_served",
                 "expected_empty_distance_m"]
    stages = []
    fixed = []
    selected_indices = None
    size = rows.columns
    binary_bounds = Bounds(np.zeros(size), np.ones(size))
    integrality = np.ones(size, dtype=np.int8)

    def stage(name, values, maximize, *, force=False):
        nonlocal selected_indices
        budget.remaining(name)
        if not any(values) and not force:
            stages.append(dict(stage=name, direction="MAX" if maximize else "MIN", value=0.0,
                               opt_status="CONSTANT_EXACT", status=0, runtime_s=0.0,
                               columns=size, rows=len(rows.lower), nonzeros=len(rows.values), mip_gap=0.0))
            return
        matrix = rows.matrix()
        objective = np.asarray([float(value) for value in values], dtype=float)
        # Integerize the weighted served objective when it can be represented
        # exactly by doubles. No epsilon or mixed-priority risk score is used.
        scale = 1
        if name == "expected_served":
            denominator = math.lcm(*(value.denominator for value in values if value))
            if max(abs(value * denominator) for value in values) <= 2**53:
                scale = denominator
                objective = np.asarray([float(value * scale) for value in values])
        started = perf_counter()
        result = milp(c=-objective if maximize else objective,
                      integrality=integrality, bounds=binary_bounds,
                      constraints=LinearConstraint(matrix, np.asarray(rows.lower), np.asarray(rows.upper)),
                      options=dict(time_limit=budget.remaining(name), mip_rel_gap=0.0, presolve=True))
        elapsed = perf_counter() - started
        if result.status == 1:
            raise DecisionTimeout(f"{name}: MILP was not proven optimal within the total decision budget: {result.message}")
        if result.status != 0 or not result.success or result.x is None:
            raise RuntimeError(f"{name}: sparse two-stage MILP did not reach optimality: {result.message}")
        budget.remaining(name)
        rounded = np.rint(result.x)
        if np.max(np.abs(result.x - rounded)) > _TOLERANCE:
            raise RuntimeError(f"{name}: returned variables are not binary")
        lhs = matrix @ rounded
        if (np.any(lhs < np.asarray(rows.lower) - _TOLERANCE)
                or np.any(lhs > np.asarray(rows.upper) + _TOLERANCE)):
            raise RuntimeError(f"{name}: returned integer selection violates sparse master constraints")
        selected_indices = tuple(int(index) for index in np.flatnonzero(rounded))
        for previous_name, previous_values, previous_value in fixed:
            actual = sum((previous_values[index] for index in selected_indices), Fraction(0))
            if actual != previous_value:
                raise RuntimeError(f"{name}: numerical solve changed fixed exact objective {previous_name}")
        value = sum((values[index] for index in selected_indices), Fraction(0))
        if not math.isclose(float(result.fun), float((-value if maximize else value) * scale),
                            rel_tol=1e-9, abs_tol=_TOLERANCE):
            raise RuntimeError(f"{name}: MILP objective disagrees with the binary selection")
        stages.append(dict(stage=name, direction="MAX" if maximize else "MIN", value=float(value),
                           opt_status="OPTIMAL", status=int(result.status), runtime_s=elapsed,
                           columns=size, rows=matrix.shape[0], nonzeros=int(matrix.nnz),
                           mip_gap=float(result.mip_gap or 0.0),
                           mip_node_count=int(result.mip_node_count or 0), objective_scale=scale))
        if any(values):
            rows.add([(index, coefficient) for index, coefficient in enumerate(values) if coefficient], value, value)
            fixed.append((name, values, value))

    for name in order:
        stage(name, coefficients[name], name != "expected_empty_distance_m")
    # One exact integer rank objective per resource fixes a common first
    # action. At most three extra solves are needed; no tiny epsilon is used.
    for resource in model.resources:
        actions = [action for action in model.actions if action.resource_id == resource.resource_id]
        ranks = [Fraction(0)] * size
        for rank, action in enumerate(actions):
            ranks[action_columns[action.action_id]] = Fraction(rank)
        stage(f"stable_first_action:{resource.resource_id}", ranks, False)
    if selected_indices is None:
        stage("feasibility", [Fraction(0)] * size, False, force=True)
    return selected_indices, stages


def _path_output(path, scenario):
    tasks = {task.job_id: task for task in scenario.tasks}
    jobs = [dict(job_id=visit.job_id, origin_id=visit.origin_id,
                 source_kind=tasks[visit.job_id].source_kind,
                 release_s=float(tasks[visit.job_id].release), deadline_s=float(tasks[visit.job_id].deadline),
                 commit_s=float(visit.commit), pickup_s=float(visit.pickup), finish_s=float(visit.finish),
                 empty_time_s=float(visit.travel), empty_distance_m=float(visit.distance))
            for visit in path.visits]
    served_ids = ([path.first_job_id] if path.first_job_id else []) + [visit.job_id for visit in path.visits]
    return dict(resource_id=path.resource_id, scenario_id=path.scenario_id, first_action_id=path.action_id,
                first_job_id=path.first_job_id, jobs=jobs, job_ids=[visit.job_id for visit in path.visits],
                served_job_ids=served_ids, total_served=path.served, empty_distance_m=float(path.distance))


def validate_epoch_solution(problem, solution):
    """Reconstruct selected first actions and every scenario path from input.

    This independently verifies feasibility and reported objective values, not
    optimality. In particular it never treats forecast recourse as realized
    service or permits a scenario-specific first action.
    """
    model = _prepare(problem)
    if not isinstance(solution, Mapping) or solution.get("policy") not in POLICIES:
        raise ValueError("solution must name a supported policy")
    selected_rows = solution.get("selected_actions")
    paths = solution.get("recourse_paths")
    if not isinstance(selected_rows, list) or not isinstance(paths, list):
        raise ValueError("selected_actions and recourse_paths must be lists")
    if len(selected_rows) != len(model.resources):
        raise ValueError("exactly one first action per resource is required")
    actions = {action.action_id: action for action in model.actions}
    resources = {resource.resource_id: resource for resource in model.resources}
    scenarios = {scenario.scenario_id: scenario for scenario in model.scenarios}
    selected = {}
    first_jobs = set()
    critical = current = carry = 0
    first_distance = Fraction(0)
    for row in selected_rows:
        if not isinstance(row, Mapping):
            raise ValueError("selected action must be an object")
        action = actions.get(row.get("action_id"))
        if action is None or dict(row) != action.raw or action.resource_id in selected:
            raise ValueError("unknown, altered, or repeated selected first action")
        selected[action.resource_id] = action
        first_distance += action.distance
        if action.kind == "SERVE":
            if action.job_id in first_jobs:
                raise ValueError("current first SERVE job is served by multiple resources")
            first_jobs.add(action.job_id)
            critical += int(action.critical)
            current += 1
            carry += int(action.carry_over)
    expected_served = expected_distance = Fraction(0)
    seen_paths = set()
    scenario_jobs = {scenario.scenario_id: set() for scenario in model.scenarios}
    for path in paths:
        if not isinstance(path, Mapping):
            raise ValueError("recourse path must be an object")
        key = (path.get("resource_id"), path.get("scenario_id"))
        if key[0] not in resources or key[1] not in scenarios or key in seen_paths:
            raise ValueError("unknown or repeated resource/scenario recourse path")
        seen_paths.add(key)
        resource, scenario, action = resources[key[0]], scenarios[key[1]], selected[key[0]]
        if path.get("first_action_id") != action.action_id:
            raise ValueError("scenario recourse must start from the same common first action")
        first_job = action.job_id if action.kind == "SERVE" else None
        if path.get("first_job_id") != first_job:
            raise ValueError("recourse first job disagrees with the executable first action")
        used_jobs = scenario_jobs[scenario.scenario_id]
        served_ids = []
        if first_job:
            if first_job in used_jobs:
                raise ValueError("scenario job is served more than once")
            used_jobs.add(first_job)
            served_ids.append(first_job)
        tasks = {task.job_id: task for task in scenario.tasks}
        visits = path.get("jobs")
        if not isinstance(visits, list):
            raise ValueError("recourse jobs must be a list")
        origin, ready, distance = action.location_id, action.ready, action.distance
        recourse_ids = []
        for visit in visits:
            if not isinstance(visit, Mapping):
                raise ValueError("recourse visit must be an object")
            job_id = visit.get("job_id")
            if job_id not in tasks or job_id in used_jobs:
                raise ValueError("unknown or repeated scenario job")
            task = tasks[job_id]
            connection = scenario.connections.get((origin, job_id))
            if (connection is None or not connection.supported
                    or resource.profile_id not in task.profiles
                    or resource.profile_id not in connection.profiles):
                raise ValueError("recourse task/connection is unsupported or profile-incompatible")
            if visit.get("origin_id") != origin or visit.get("source_kind") != task.source_kind:
                raise ValueError("recourse origin or task source disagrees with input")
            commit = _ceil_step(max(ready, task.release), model.step)
            pickup = commit + connection.travel
            finish = pickup + task.service
            if commit >= resource.end or pickup > task.deadline:
                raise ValueError("recourse violates its admission end or original deadline")
            for field, value in (("release_s", task.release), ("deadline_s", task.deadline),
                                 ("commit_s", commit), ("pickup_s", pickup), ("finish_s", finish),
                                 ("empty_time_s", connection.travel), ("empty_distance_m", connection.distance)):
                _same_number(visit, field, value, f"{resource.resource_id}/{scenario.scenario_id}/{job_id}")
            used_jobs.add(job_id)
            served_ids.append(job_id)
            recourse_ids.append(job_id)
            origin, ready = job_id, finish
            distance += connection.distance
        if (path.get("served_job_ids") != served_ids or path.get("job_ids") != recourse_ids
                or path.get("total_served") != len(served_ids)):
            raise ValueError("reported recourse identities/count do not match reconstructed jobs")
        _same_number(path, "empty_distance_m", distance, "recourse")
        expected_served += scenario.weight * len(served_ids)
        expected_distance += scenario.weight * distance
    if len(seen_paths) != len(model.resources) * len(model.scenarios):
        raise ValueError("one recourse path per resource/scenario is required")
    for field, value in (("critical_now", critical), ("current_served", current),
                         ("carry_over_current_served", carry), ("expected_served", expected_served),
                         ("expected_empty_distance_m", expected_distance),
                         ("actual_first_empty_distance_m", first_distance)):
        _same_number(solution, field, value, "solution")
    return dict(valid=True, checked_first_actions=len(selected), checked_recourse_paths=len(seen_paths),
                current_served=current, critical_now=critical, expected_served=float(expected_served),
                expected_empty_distance_m=float(expected_distance),
                independent_reconstruction=True, realized_service_is_first_actions_only=True)


def solve_epoch(problem, policy):
    """Solve lexicographic two-stage binary decisions within a total 10s budget."""
    budget = _Budget()
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES!r}")
    model = _prepare(problem)
    preparation_s = perf_counter() - budget.started
    enumeration_started = perf_counter()
    paths, chain_counts = _enumerate(model, budget)
    enumeration_s = perf_counter() - enumeration_started
    master_started = perf_counter()
    rows, action_columns = _master(model, paths)
    base_rows, base_nonzeros = len(rows.lower), len(rows.values)
    master_s = perf_counter() - master_started
    budget.remaining("sparse master construction")
    optimization_started = perf_counter()
    chosen, stages = _milp_selection(model, paths, rows, action_columns, policy, budget)
    optimization_s = perf_counter() - optimization_started
    action_count = len(model.actions)
    chosen_actions = [model.actions[index] for index in chosen if index < action_count]
    chosen_paths = [paths[index - action_count] for index in chosen if index >= action_count]
    scenarios = {scenario.scenario_id: scenario for scenario in model.scenarios}
    current = sum(action.kind == "SERVE" for action in chosen_actions)
    critical = sum(action.kind == "SERVE" and action.critical for action in chosen_actions)
    carry = sum(action.kind == "SERVE" and action.carry_over for action in chosen_actions)
    expected_served = sum((scenarios[path.scenario_id].weight * path.served for path in chosen_paths), Fraction(0))
    expected_distance = sum((scenarios[path.scenario_id].weight * path.distance for path in chosen_paths), Fraction(0))
    solution = dict(
        kind=KIND, policy=policy, opt_status="OPTIMAL", selected_actions=[deepcopy(action.raw) for action in chosen_actions],
        expected_served=float(expected_served), critical_now=int(critical), current_served=int(current),
        carry_over_current_served=int(carry), expected_empty_distance_m=float(expected_distance),
        actual_first_empty_distance_m=float(sum((action.distance for action in chosen_actions), Fraction(0))),
        recourse_paths=[_path_output(path, scenarios[path.scenario_id]) for path in chosen_paths],
        model=dict(rows=len(rows.lower), columns=rows.columns, cols=rows.columns,
                   nonzeros=len(rows.values), nnz=len(rows.values), base_rows=base_rows, base_nonzeros=base_nonzeros,
                   sparse=True, first_action_variables=len(model.actions), scenario_chain_variables=len(paths),
                   chains_per_resource_scenario=chain_counts, complete_enumeration=True,
                   max_chains_per_resource_scenario=MAX_CHAINS_PER_RESOURCE_SCENARIO,
                   max_variables=MAX_MASTER_VARIABLES, max_nonzeros=MAX_MASTER_NONZEROS),
        stages=stages,
        timing_s=dict(preparation=preparation_s, enumeration=enumeration_s,
                      sparse_master=master_s, lexicographic_milp=optimization_s),
        model_domain=dict(
            historical_scenarios_are_forecast_proxies=True,
            revealed_future_recourse_is_a_two_stage_value_approximation=True,
            only_common_first_actions_are_executed=True, planned_recourse_is_not_realized_service=True,
            scope="FINITE_SUPPORTED_CONSTANT_TIME_HISTORICAL_SCENARIOS_NOT_MULTISTAGE_OPTIMUM",
            missing_or_unsupported_connections_do_not_prove_unroutability=True,
            earliest_schedule_dominance="Constant connection times and waiting allow the earliest schedule to dominate later schedules.",
            deadline_is_original_and_never_reset=True, future_commitment_requires_release=True,
            resource_profile_is_fixed=True, committed_service_may_finish_after_admission_end=True,
            decision_time_limit_s=DECISION_TIME_LIMIT_S))
    validation_started = perf_counter()
    solution["validation"] = validate_epoch_solution(problem, solution)
    solution["timing_s"]["independent_validation"] = perf_counter() - validation_started
    budget.remaining("decision output validation")
    solution["timing_s"]["total"] = perf_counter() - budget.started
    solution["runtime_s"] = solution["timing_s"]["total"]
    return solution
