"""Bounded, fully known, constant-connection-time offline instance solver.

This module enumerates actual timed chains, rather than accepting a declared
family of toy chains.  Its LP bound covers only the complete finite input
snapshot: an absent or unsupported connection is excluded, not proven
unroutable.  It is neither an online dispatch policy nor a citywide bound.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
import math
from time import perf_counter

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


MAX_TASKS = 10
MAX_RESOURCES = 3
MAX_SITES = 4
MAX_CHAINS_PER_RESOURCE = 20_000
MAX_DP_STATES = 50_000
KIND = "OFFLINE_FULLY_KNOWN_SNAPSHOT_INSTANCE_NOT_ONLINE_POLICY"
SCHEMA_VERSION = "capability_chain_instance_v1"
_VALIDATION_TOLERANCE = 1e-8


class InstanceLimitError(ValueError):
    """The finite model exceeds its declared budget; no truncation is made."""


@dataclass(frozen=True)
class _Task:
    job_id: str
    release: Fraction
    deadline: Fraction
    service: Fraction
    profiles: frozenset[str]


@dataclass(frozen=True)
class _Resource:
    resource_id: str
    profile_id: str
    start: Fraction
    end: Fraction
    sites: tuple[str, ...]
    fixed_site: str


@dataclass(frozen=True)
class _Connection:
    travel: Fraction
    distance: Fraction
    supported: bool
    profiles: frozenset[str]


@dataclass(frozen=True)
class _Visit:
    job_id: str
    origin_id: str
    commit: Fraction
    pickup: Fraction
    finish: Fraction
    empty_time: Fraction
    empty_distance: Fraction


@dataclass(frozen=True)
class _Chain:
    resource_id: str
    site_id: str
    visits: tuple[_Visit, ...]
    mask: int
    distance: Fraction

    @property
    def stable_key(self):
        return self.site_id, tuple(visit.job_id for visit in self.visits)


@dataclass(frozen=True)
class _Model:
    tasks: tuple[_Task, ...]
    resources: tuple[_Resource, ...]
    sites: tuple[str, ...]
    connections: dict[tuple[str, str], _Connection]
    step: Fraction


def _number(value, name: str, *, nonnegative: bool = True) -> Fraction:
    requirement = "finite, nonnegative" if nonnegative else "finite"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a {requirement} number")
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{name} must be a {requirement} number")
    # Exact decimal comparisons avoid changing a grid boundary or a distance
    # tie through accumulated binary floating-point roundoff.
    return Fraction(str(value))


def _identifier(value, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _rows(instance, name: str, limit: int | None = None):
    rows = instance.get(name)
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{name} must be a list of objects")
    if limit is not None and len(rows) > limit:
        raise InstanceLimitError(f"{name} limit exceeded: maximum {limit}")
    return rows


def _profiles(row, name: str) -> frozenset[str]:
    values = row.get("compatible_profiles")
    if not isinstance(values, list):
        raise ValueError(f"{name}.compatible_profiles must be a list")
    return frozenset(_identifier(value, f"{name}.compatible_profiles") for value in values)


def _prepare(instance) -> _Model:
    if not isinstance(instance, Mapping) or instance.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected schema_version={SCHEMA_VERSION!r}")
    step = _number(instance.get("execution_step_s"), "execution_step_s")
    if step != 30:
        raise ValueError("execution_step_s must be 30 for this bounded model")
    task_rows = _rows(instance, "tasks", MAX_TASKS)
    site_rows = _rows(instance, "sites", MAX_SITES)
    resource_rows = _rows(instance, "resources", MAX_RESOURCES)
    used_ids = set()

    def distinct_id(row, field):
        value = _identifier(row.get(field), field)
        if value in used_ids:
            raise ValueError(f"job/site/resource IDs must be distinct: {value!r}")
        used_ids.add(value)
        return value

    tasks = []
    for row in task_rows:
        job_id = distinct_id(row, "job_id")
        release = _number(row.get("release_s"), f"{job_id}.release_s", nonnegative=False)
        deadline = _number(row.get("deadline_s"), f"{job_id}.deadline_s", nonnegative=False)
        if release > deadline:
            raise ValueError(f"{job_id}: release_s exceeds deadline_s")
        tasks.append(_Task(job_id, release, deadline,
                           _number(row.get("service_time_s"), f"{job_id}.service_time_s"),
                           _profiles(row, job_id)))
    sites = tuple(sorted(distinct_id(row, "site_id") for row in site_rows))
    site_set = set(sites)
    resources = []
    for row in resource_rows:
        resource_id = distinct_id(row, "resource_id")
        profile = _identifier(row.get("profile_id"), f"{resource_id}.profile_id")
        start = _number(row.get("start_s"), f"{resource_id}.start_s", nonnegative=False)
        end = _number(row.get("end_s"), f"{resource_id}.end_s", nonnegative=False)
        if start > end:
            raise ValueError(f"{resource_id}: start_s exceeds end_s")
        choices = row.get("site_ids")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{resource_id}.site_ids must be a nonempty list")
        choices = tuple(_identifier(site, f"{resource_id}.site_ids") for site in choices)
        if len(set(choices)) != len(choices) or not set(choices) <= site_set:
            raise ValueError(f"{resource_id}: duplicate or unknown site_ids")
        fixed_site = _identifier(row.get("fixed_site_id"), f"{resource_id}.fixed_site_id")
        if fixed_site not in choices:
            raise ValueError(f"{resource_id}: fixed_site_id is outside site_ids")
        resources.append(_Resource(resource_id, profile, start, end, tuple(sorted(choices)), fixed_site))
    task_ids = {task.job_id for task in tasks}
    origins = task_ids | site_set
    connections = {}
    for row in _rows(instance, "connections"):
        origin = _identifier(row.get("origin_id"), "connection.origin_id")
        target = _identifier(row.get("target_job_id"), "connection.target_job_id")
        if origin not in origins or target not in task_ids:
            raise ValueError(f"connection uses unknown origin/target: {(origin, target)!r}")
        if (origin, target) in connections:
            raise ValueError(f"duplicate connection: {(origin, target)!r}")
        if type(row.get("supported")) is not bool:
            raise ValueError("connection.supported must be a boolean")
        supported = row["supported"]
        # Unknown geometry remains nullable in the public input. A zero
        # internal placeholder is safe ONLY for an explicitly unsupported edge:
        # enumeration and validation reject it before inspecting these values.
        travel = (Fraction(0) if not supported and row.get("travel_time_s") is None else
                  _number(row.get("travel_time_s"), "connection.travel_time_s"))
        distance = (Fraction(0) if not supported and row.get("empty_distance_m") is None else
                    _number(row.get("empty_distance_m"), "connection.empty_distance_m"))
        connections[origin, target] = _Connection(
            travel, distance, supported, _profiles(row, f"connection[{origin},{target}]"))
    return _Model(tuple(sorted(tasks, key=lambda task: task.job_id)),
                  tuple(sorted(resources, key=lambda resource: resource.resource_id)),
                  sites, connections, step)


def _ceil_step(value: Fraction, step: Fraction) -> Fraction:
    quotient = value / step
    return ((quotient.numerator + quotient.denominator - 1) // quotient.denominator) * step


def _enumerate(model: _Model, optimize_layout: bool):
    families = {}
    site_summary = []
    for resource in model.resources:
        sites = resource.sites if optimize_layout else (resource.fixed_site,)
        # One idle column per resource.  Its site is the stable first choice.
        chains = [_Chain(resource.resource_id, sites[0], (), 0, Fraction(0))]
        if len(chains) > MAX_CHAINS_PER_RESOURCE:
            raise InstanceLimitError("chain limit exceeded for resource " + resource.resource_id)
        for site in sites:
            site_chains = 0
            maximum_served = 0

            def extend(origin, ready, mask, visits, distance):
                nonlocal site_chains, maximum_served
                for index, task in enumerate(model.tasks):
                    bit = 1 << index
                    if mask & bit or resource.profile_id not in task.profiles:
                        continue
                    connection = model.connections.get((origin, task.job_id))
                    if (connection is None or not connection.supported
                            or resource.profile_id not in connection.profiles):
                        continue
                    commit = _ceil_step(max(ready, task.release, resource.start), model.step)
                    pickup = commit + connection.travel
                    if commit >= resource.end or pickup > task.deadline:
                        continue
                    finish = pickup + task.service
                    visit = _Visit(task.job_id, origin, commit, pickup, finish,
                                   connection.travel, connection.distance)
                    next_visits = visits + (visit,)
                    next_distance = distance + connection.distance
                    if len(chains) >= MAX_CHAINS_PER_RESOURCE:
                        raise InstanceLimitError(
                            f"chain limit exceeded for resource {resource.resource_id}: "
                            f"maximum {MAX_CHAINS_PER_RESOURCE}; enumeration was not truncated")
                    chains.append(_Chain(resource.resource_id, site, next_visits, mask | bit, next_distance))
                    site_chains += 1
                    maximum_served = max(maximum_served, len(next_visits))
                    extend(task.job_id, finish, mask | bit, next_visits, next_distance)

            extend(site, resource.start, 0, (), Fraction(0))
            idle_here = site == sites[0]
            site_summary.append(dict(resource_id=resource.resource_id, site_id=site,
                                     chain_count=site_chains + int(idle_here),
                                     nonempty_chain_count=site_chains,
                                     includes_resource_idle_column=idle_here,
                                     max_served_in_single_chain=maximum_served))
        families[resource.resource_id] = tuple(sorted(chains, key=lambda chain: chain.stable_key))
    return families, site_summary


def _exact_selection(model: _Model, families):
    # Within one resource, columns with an identical job mask are
    # interchangeable for every other resource. Keep the cheapest stable one
    # for this integer DP, while retaining ALL columns in the LP below.
    states = {0: (Fraction(0), (), ())}
    visited_states = 1
    if visited_states > MAX_DP_STATES:
        raise InstanceLimitError(f"DP state limit exceeded: maximum {MAX_DP_STATES}")
    retained_counts = {}
    for resource in model.resources:
        by_mask = {}
        for chain in families[resource.resource_id]:
            previous = by_mask.get(chain.mask)
            if previous is None or (chain.distance, chain.stable_key) < (previous.distance, previous.stable_key):
                by_mask[chain.mask] = chain
        choices = tuple(sorted(by_mask.values(), key=lambda chain: chain.stable_key))
        retained_counts[resource.resource_id] = len(choices)
        next_states = {}
        for used, (distance, stable_key, selected) in states.items():
            for chain in choices:
                if used & chain.mask:
                    continue
                mask = used | chain.mask
                candidate = (distance + chain.distance, stable_key + (chain.stable_key,), selected + (chain,))
                previous = next_states.get(mask)
                if previous is None:
                    visited_states += 1
                    if visited_states > MAX_DP_STATES:
                        raise InstanceLimitError(f"DP state limit exceeded: maximum {MAX_DP_STATES}")
                if previous is None or candidate[:2] < previous[:2]:
                    next_states[mask] = candidate
        states = next_states
    mask, (distance, _, selected) = min(states.items(),
                                       key=lambda item: (-item[0].bit_count(), item[1][0], item[1][1]))
    return mask.bit_count(), distance, selected, visited_states, retained_counts


def _chain_output(chain: _Chain, resource: _Resource, tasks):
    visits = [dict(job_id=visit.job_id, origin_id=visit.origin_id,
                   release_s=float(tasks[visit.job_id].release),
                   deadline_s=float(tasks[visit.job_id].deadline),
                   commit_s=float(visit.commit), departure_s=float(visit.commit),
                   pickup_s=float(visit.pickup), finish_s=float(visit.finish),
                   empty_time_s=float(visit.empty_time), empty_distance_m=float(visit.empty_distance))
              for visit in chain.visits]
    return dict(resource_id=resource.resource_id, profile_id=resource.profile_id,
                site_id=chain.site_id, job_ids=[visit.job_id for visit in chain.visits],
                jobs=visits, served=len(visits), empty_distance_m=float(chain.distance),
                empty_time_s=float(sum((visit.empty_time for visit in chain.visits), Fraction(0))),
                finish_s=float(chain.visits[-1].finish if chain.visits else resource.start))


def _relaxed_bound(model: _Model, families, integer_served: int):
    columns = [chain for resource in model.resources for chain in families[resource.resource_id]]
    task_rows = {task.job_id: index for index, task in enumerate(model.tasks)}
    resource_rows = {resource.resource_id: index for index, resource in enumerate(model.resources)}
    if not columns:
        return dict(served_upper_bound=0.0, served_lp_value=0.0, gap_jobs=0.0,
                    rows=len(model.tasks), columns=0, nonzeros=0,
                    job_constraint_rows=len(model.tasks), resource_equality_rows=0,
                    full_column_domain=True, sparse_matrix=True, nonzero_solution=[],
                    job_prices={job: 0.0 for job in task_rows}, resource_prices={},
                    certificate=dict(checked_columns=0, minimum_dual_slack=0.0,
                                     primal_max_violation=0.0, dual_objective=0.0, primal_dual_gap=0.0))
    job_indices, job_columns = [], []
    resource_indices = []
    for index, chain in enumerate(columns):
        for visit in chain.visits:
            job_indices.append(task_rows[visit.job_id])
            job_columns.append(index)
        resource_indices.append(resource_rows[chain.resource_id])
    job_matrix = coo_matrix((np.ones(len(job_indices)), (job_indices, job_columns)),
                            shape=(len(model.tasks), len(columns))).tocsr()
    resource_matrix = coo_matrix((np.ones(len(columns)), (resource_indices, np.arange(len(columns)))),
                                 shape=(len(model.resources), len(columns))).tocsr()
    rewards = np.array([len(chain.visits) for chain in columns], dtype=float)
    result = linprog(-rewards, A_ub=job_matrix if model.tasks else None,
                     b_ub=np.ones(len(model.tasks)) if model.tasks else None,
                     A_eq=resource_matrix, b_eq=np.ones(len(model.resources)),
                     bounds=(0.0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"complete finite-input LP failed: {result.message}")
    alpha = np.maximum(0.0, -np.asarray(result.ineqlin.marginals)) if model.tasks else np.zeros(0)
    beta = -np.asarray(result.eqlin.marginals).copy()
    slacks = np.asarray(job_matrix.T @ alpha + resource_matrix.T @ beta).ravel() - rewards
    # Repair sub-tolerance floating-point dual slack, separately by resource.
    # This checks every enumerated column, including the one idle column.
    for resource_id, row in resource_rows.items():
        minimum = min(float(slacks[index]) for index, chain in enumerate(columns)
                      if chain.resource_id == resource_id)
        beta[row] += max(0.0, -minimum)
    slacks = np.asarray(job_matrix.T @ alpha + resource_matrix.T @ beta).ravel() - rewards
    dual_value = math.fsum(float(value) for value in alpha) + math.fsum(float(value) for value in beta)
    primal_value = float(-result.fun)
    job_violation = max(0.0, float(np.max(job_matrix @ result.x - 1.0))) if model.tasks else 0.0
    resource_violation = float(np.max(np.abs(resource_matrix @ result.x - 1.0)))
    primal_violation = max(job_violation, resource_violation, max(0.0, float(-result.x.min())))
    if (primal_violation > _VALIDATION_TOLERANCE or float(slacks.min()) < -_VALIDATION_TOLERANCE
            or abs(dual_value - primal_value) > _VALIDATION_TOLERANCE
            or primal_value < integer_served - _VALIDATION_TOLERANCE):
        raise RuntimeError("complete finite-input LP primal/dual certificate mismatch")
    upper_bound = min(float(len(model.tasks)), max(dual_value, primal_value, float(integer_served)))
    return dict(served_upper_bound=upper_bound, served_lp_value=primal_value,
                gap_jobs=max(0.0, upper_bound - integer_served),
                rows=len(model.tasks) + len(model.resources), columns=len(columns),
                nonzeros=int(job_matrix.nnz + resource_matrix.nnz),
                job_constraint_rows=len(model.tasks), resource_equality_rows=len(model.resources),
                full_column_domain=True, sparse_matrix=True,
                nonzero_solution=[dict(resource_id=chain.resource_id, site_id=chain.site_id,
                                       job_ids=[visit.job_id for visit in chain.visits], weight=float(weight))
                                  for chain, weight in zip(columns, result.x) if weight > 1e-9],
                job_prices={job: float(alpha[index]) for job, index in task_rows.items()},
                resource_prices={resource_id: float(beta[index]) for resource_id, index in resource_rows.items()},
                certificate=dict(checked_columns=len(columns), minimum_dual_slack=float(slacks.min()),
                                 primal_max_violation=primal_violation, dual_objective=dual_value,
                                 primal_dual_gap=abs(primal_value - dual_value)))


def _check_reported_number(row, field, expected, name):
    actual = _number(row.get(field), f"{name}.{field}", nonnegative=False)
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=_VALIDATION_TOLERANCE):
        raise ValueError(f"{name}: {field} disagrees with independently reconstructed trajectory")


def validate_solution(instance, solution, *, optimize_layout: bool | None = None):
    """Independently reconstruct feasibility; never use reported chain totals.

    Raises ValueError for an invalid selection or trajectory.  This checks the
    solver's earliest-schedule contract as well as resource, compatibility,
    release/deadline, grid, and global job uniqueness constraints.  It does not
    claim to certify optimality or any connection outside the input snapshot.
    """
    model = _prepare(instance)
    if not isinstance(solution, Mapping):
        raise ValueError("solution must be an object")
    if optimize_layout is None:
        optimize_layout = solution.get("optimize_layout")
    if type(optimize_layout) is not bool:
        raise ValueError("optimize_layout must be a boolean")
    selections = solution.get("selected")
    if not isinstance(selections, list) or any(not isinstance(row, Mapping) for row in selections):
        raise ValueError("selected must be a list of objects")
    resources = {resource.resource_id: resource for resource in model.resources}
    if len(selections) != len(resources):
        raise ValueError("selected must contain exactly one chain per resource")
    tasks = {task.job_id: task for task in model.tasks}
    seen_resources, seen_jobs = set(), set()
    distance_total = Fraction(0)
    for selected in selections:
        resource_id = selected.get("resource_id")
        if resource_id not in resources or resource_id in seen_resources:
            raise ValueError("unknown or repeated selected resource")
        seen_resources.add(resource_id)
        resource = resources[resource_id]
        if selected.get("profile_id") != resource.profile_id:
            raise ValueError(f"{resource_id}: resource profile cannot change along a chain")
        site = selected.get("site_id")
        allowed_sites = resource.sites if optimize_layout else (resource.fixed_site,)
        if site not in allowed_sites:
            raise ValueError(f"{resource_id}: selected site is outside the layout domain")
        visits = selected.get("jobs")
        if not isinstance(visits, list) or any(not isinstance(visit, Mapping) for visit in visits):
            raise ValueError(f"{resource_id}.jobs must be a list of objects")
        origin, ready = site, resource.start
        resource_distance, resource_empty_time = Fraction(0), Fraction(0)
        job_ids = []
        for visit in visits:
            job_id = visit.get("job_id")
            if job_id not in tasks or job_id in seen_jobs:
                raise ValueError(f"unknown or repeated served job: {job_id!r}")
            seen_jobs.add(job_id)
            job_ids.append(job_id)
            task = tasks[job_id]
            connection = model.connections.get((origin, job_id))
            if connection is None or not connection.supported:
                raise ValueError(f"{resource_id}/{job_id}: connection is outside the supported snapshot")
            if resource.profile_id not in task.profiles or resource.profile_id not in connection.profiles:
                raise ValueError(f"{resource_id}/{job_id}: task or connection profile is incompatible")
            if visit.get("origin_id") != origin:
                raise ValueError(f"{resource_id}/{job_id}: origin does not follow the prior task")
            # Reconstruct from INPUT values and previous reconstructed finish,
            # not from the reported ready, pickup, finish, or chain totals.
            commit = _ceil_step(max(ready, task.release, resource.start), model.step)
            pickup = commit + connection.travel
            finish = pickup + task.service
            if commit >= resource.end or pickup > task.deadline:
                raise ValueError(f"{resource_id}/{job_id}: reconstructed chain violates a time window")
            for field, expected in (("release_s", task.release), ("deadline_s", task.deadline),
                                    ("commit_s", commit), ("departure_s", commit),
                                    ("pickup_s", pickup), ("finish_s", finish),
                                    ("empty_time_s", connection.travel), ("empty_distance_m", connection.distance)):
                _check_reported_number(visit, field, expected, f"{resource_id}/{job_id}")
            resource_distance += connection.distance
            resource_empty_time += connection.travel
            origin, ready = job_id, finish
        if selected.get("job_ids") != job_ids or selected.get("served") != len(visits):
            raise ValueError(f"{resource_id}: reported job identities/count do not match its chain")
        _check_reported_number(selected, "empty_distance_m", resource_distance, resource_id)
        _check_reported_number(selected, "empty_time_s", resource_empty_time, resource_id)
        _check_reported_number(selected, "finish_s", ready, resource_id)
        distance_total += resource_distance
    objective = solution.get("objective")
    if not isinstance(objective, Mapping) or objective.get("served") != len(seen_jobs):
        raise ValueError("reported served objective disagrees with unique reconstructed jobs")
    _check_reported_number(objective, "empty_distance_m", distance_total, "objective")
    if solution.get("served") != len(seen_jobs):
        raise ValueError("reported served count disagrees with reconstructed jobs")
    _check_reported_number(solution, "empty_distance_m", distance_total, "solution")
    return dict(valid=True, checked_resources=len(seen_resources), checked_jobs=len(seen_jobs),
                served=len(seen_jobs), empty_distance_m=float(distance_total),
                independent_time_and_connection_reconstruction=True,
                scope="SELECTED_CHAINS_IN_FINITE_INPUT_SNAPSHOT_ONLY")


def solve_instance(instance, optimize_layout: bool = True):
    """Return exact (max served, min empty distance, stable tie) selection.

    Joint and fixed-site modes use the same fully known future tasks.  Constant
    connection durations and waiting at the current origin imply that earlier
    completion can reproduce every later feasible continuation. Thus one
    earliest schedule per task sequence is complete for this input model.
    Layout selection is an initial site choice, not subsequent repositioning.
    """
    started = perf_counter()
    if type(optimize_layout) is not bool:
        raise ValueError("optimize_layout must be a boolean")
    model = _prepare(instance)
    enumeration_started = perf_counter()
    families, site_summary = _enumerate(model, optimize_layout)
    enumeration_s = perf_counter() - enumeration_started
    integer_started = perf_counter()
    served, distance, selected, dp_states, retained_counts = _exact_selection(model, families)
    integer_s = perf_counter() - integer_started
    lp_started = perf_counter()
    lp = _relaxed_bound(model, families, served)
    lp_s = perf_counter() - lp_started
    resources = {resource.resource_id: resource for resource in model.resources}
    tasks = {task.job_id: task for task in model.tasks}
    solution = dict(
        schema_version="capability_chain_solution_v1", kind=KIND,
        optimize_layout=optimize_layout,
        layout_mode="JOINT_INITIAL_SITE_AND_CHAIN" if optimize_layout else "FIXED_INITIAL_SITE",
        objective=dict(served=served, empty_distance_m=float(distance),
                       order=["MAX_SERVED", "MIN_EMPTY_DISTANCE_M", "STABLE_RESOURCE_SITE_JOB_ORDER"]),
        served=served, empty_distance_m=float(distance),
        selected=[_chain_output(chain, resources[chain.resource_id], tasks) for chain in selected],
        lp=lp,
        enumeration=dict(chains_per_resource={name: len(chains) for name, chains in families.items()},
                         per_resource_site=site_summary, dp_states=dp_states,
                         integer_mask_choices_per_resource=retained_counts,
                         complete=True, idle_columns_per_resource=1,
                         max_chains_per_resource=MAX_CHAINS_PER_RESOURCE, max_dp_states=MAX_DP_STATES),
        model_domain=dict(
            scope="COMPLETE_FINITE_INPUT_SUPPORTED_CONNECTION_SNAPSHOT_ONLY",
            task_count=len(model.tasks), resource_count=len(model.resources), site_count=len(model.sites),
            constant_time_earliest_schedule_dominance=(
                "With constant input connection times and waiting at the current origin, "
                "the earliest feasible schedule dominates later schedules of the same task sequence."),
            unsupported_connections=(
                "Missing or supported=false connections are excluded from this input domain; "
                "their exclusion does not prove that a route is impossible."),
            future_information="ALL_INSTANCE_TASKS_ARE_KNOWN_IN_BOTH_LAYOUT_MODES",
            execution_step_s=30, commitment_requires_release=True,
            deadline_is_original_and_never_reset=True, committed_service_may_finish_after_resource_end=True,
            resource_profile_is_fixed=True, initial_site_only=True,
            lp_bound_scope="THIS_COMPLETE_FINITE_INPUT_DOMAIN_NOT_CITYWIDE"),
        timing_s=dict(enumeration=enumeration_s, exact_integer_selection=integer_s, sparse_lp=lp_s))
    validation_started = perf_counter()
    solution["validation"] = validate_solution(instance, solution, optimize_layout=optimize_layout)
    solution["timing_s"]["independent_validation"] = perf_counter() - validation_started
    solution["runtime_s"] = perf_counter() - started
    return solution
