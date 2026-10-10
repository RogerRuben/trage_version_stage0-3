"""TIME_TYPE_DEFER structural ablation of post-service recourse geometry.

The caller supplies a frozen, earlier-history typical pickup reference. This
module fits nothing and reads no orders, routes, files, or actual future data.
Only task-origin recourse connections are proxies. Executable first actions
and known current-state connections retain their physical checks unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math


REFERENCE_KIND = "EARLIER_HISTORY_TYPICAL_SERVICEABLE_PICKUP"
PROXY_PROVENANCE = "GENERIC_POST_SERVICE_TIME_TYPE_PROXY_NOT_PHYSICAL_CERTIFICATE"
MAX_REFERENCE_TRAVEL_TIME_S = 300


def _nonnegative_number(value, field):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"reference.{field} must be a finite nonnegative number")
    return value


def _identifier(value, field):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def project_post_service_geometry(problem, reference):
    """Deep-copy one epoch and replace only its post-service task-origin layer.

    For each distinct task pair i,j, create a constant-time, constant-distance
    valuation edge with j's unchanged type compatibility. This deliberately
    removes the location and direction of a future served customer's endpoint;
    it does not remove releases, original deadlines, occupied service durations,
    waiting/defer decisions, relocation actions, or fixed resource profiles.

    A supported proxy edge is NOT a physical routing certificate. The shared
    rolling kernel executes only the original common first action and must
    reapply genuine route checks when the next epoch is constructed. Completing
    the task-origin graph can increase chain enumeration substantially; existing
    chain, prefix, master-matrix and decision-time budgets are never relaxed.
    """
    if not isinstance(problem, Mapping) or not isinstance(reference, Mapping):
        raise ValueError("problem and reference must be objects")
    if reference.get("kind") != REFERENCE_KIND:
        raise ValueError(f"reference.kind must be {REFERENCE_KIND!r}")
    travel = _nonnegative_number(reference.get("travel_time_s"), "travel_time_s")
    distance = _nonnegative_number(reference.get("empty_distance_m"), "empty_distance_m")
    if travel > MAX_REFERENCE_TRAVEL_TIME_S:
        raise ValueError("reference.travel_time_s exceeds the original 300s serviceable-pickup limit")
    dates = reference.get("history_dates")
    if (not isinstance(dates, list) or not dates
            or any((type(date) not in (str, int) or isinstance(date, bool)
                    or (isinstance(date, str) and not date)) for date in dates)):
        raise ValueError("reference.history_dates must be a nonempty list of date identifiers")
    scenarios = problem.get("scenarios")
    if not isinstance(scenarios, list) or any(not isinstance(row, Mapping) for row in scenarios):
        raise ValueError("problem.scenarios must be a list of objects")
    projected = deepcopy(dict(problem))
    counts = []
    for scenario in projected["scenarios"]:
        scenario_id = _identifier(scenario.get("scenario_id"), "scenario_id")
        tasks = scenario.get("tasks")
        connections = scenario.get("connections")
        if (not isinstance(tasks, list) or any(not isinstance(task, Mapping) for task in tasks)
                or not isinstance(connections, list) or any(not isinstance(edge, Mapping) for edge in connections)):
            raise ValueError("each scenario must contain task and connection object lists")
        by_id = {}
        for task in tasks:
            job_id = _identifier(task.get("job_id"), "task.job_id")
            if job_id in by_id:
                raise ValueError("duplicate scenario task job_id")
            profiles = task.get("compatible_profiles")
            if not isinstance(profiles, list):
                raise ValueError("task.compatible_profiles must be a list")
            for profile in profiles:
                _identifier(profile, "task.compatible_profiles")
            by_id[job_id] = task
        retained, replaced_pairs, seen = [], set(), set()
        for edge in connections:
            origin = _identifier(edge.get("origin_id"), "connection.origin_id")
            target = _identifier(edge.get("target_job_id"), "connection.target_job_id")
            if (origin, target) in seen:
                raise ValueError("duplicate scenario connection")
            seen.add((origin, target))
            if origin in by_id and target != origin:
                if target not in by_id:
                    raise ValueError("post-service connection has an unknown target task")
                replaced_pairs.add((origin, target))
            else:
                retained.append(edge)
        proxies = []
        for origin in sorted(by_id):
            for target in sorted(by_id):
                if origin == target:
                    continue
                proxies.append(dict(
                    origin_id=origin, target_job_id=target, supported=True,
                    travel_time_s=travel, empty_distance_m=distance,
                    compatible_profiles=deepcopy(by_id[target]["compatible_profiles"]),
                    reason=PROXY_PROVENANCE, provenance=PROXY_PROVENANCE))
        scenario["connections"] = retained + proxies
        counts.append(dict(
            scenario_id=scenario_id, task_count=len(tasks),
            retained_known_state_connections=sum(edge["origin_id"] not in by_id for edge in retained),
            retained_self_connections=sum(edge["origin_id"] in by_id for edge in retained),
            replaced_post_service_connections=len(replaced_pairs),
            added_post_service_connections=len(proxies) - len(replaced_pairs),
            post_service_proxy_connections=len(proxies)))
    projected["valuation_model"] = dict(
        name="TIME_TYPE_DEFER", role="STRUCTURAL_ABLATION_OF_POST_SERVICE_POSITION_AND_DIRECTION",
        provenance=PROXY_PROVENANCE, reference=deepcopy(dict(reference)),
        source_description=(
            "Caller-supplied frozen typical serviceable pickup time/distance from earlier-history "
            "supported site-to-customer minima; no reference fitting or actual-day connection sampling occurs here."),
        post_service_connections_are_valuation_proxies=True,
        proxy_connections_are_not_physical_certificates=True,
        actual_first_actions_and_their_route_checks_preserved=True,
        known_wait_busy_relocate_state_connections_preserved=True,
        tasks_original_releases_deadlines_service_times_and_type_compatibility_preserved=True,
        resources_first_ready_times_first_travel_times_and_problem_flags_preserved=True,
        only_common_first_actions_are_executed=True,
        next_epoch_requires_real_routes_and_admission_checks=True,
        no_actual_future_information_or_parameter_fitting=True,
        kernel_budgets_unchanged=True,
        enumeration_risk=(
            "Completing the post-service task graph may increase feasible chains and master columns; "
            "the existing exact-compression, chain/prefix, sparse-master and 10s kernel limits still apply."),
        per_scenario_projection=counts)
    return projected
