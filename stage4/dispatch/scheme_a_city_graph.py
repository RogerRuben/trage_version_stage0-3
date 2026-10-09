"""Bounded, directed-evidence service chains for the city Scheme-A method.

The historical library refreshes every 300 s and provides stable prediction-
only scenarios between refreshes. The chain generator is a declared restricted
beam heuristic, not a complete city path domain or a global upper bound. Empty
connections are admitted only by the caller's typed, directed route evidence;
the spatial index is a candidate screen and never a routing certificate.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from math import ceil, cos, isfinite, pi
import sys
from typing import Callable, Mapping, Sequence
import weakref

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .acceptance import passenger_acceptance
from .scheme_a_city_master import ServiceChain


Position = tuple[float, float]


def _point(value) -> Position:
    result = tuple(map(float, value))
    if len(result) != 2 or not all(isfinite(v) for v in result):
        raise ValueError("a finite WGS84 lon/lat pair is required")
    if not -180 <= result[0] <= 180 or not -90 <= result[1] <= 90:
        raise ValueError("invalid WGS84 position")
    return result


def _xy(value: Position) -> tuple[float, float]:
    # Same local candidate geometry as the existing Xi'an native adapters.
    return value[0] * 111320 * cos(34.25 * pi / 180), value[1] * 110540


@dataclass(frozen=True)
class ChainTask:
    request_id: int
    release_s: float
    deadline_s: float
    service_time_s: float
    pickup: Position
    dropoff: Position
    compatible_profiles: frozenset[str]
    passenger_accepts_av: bool = True
    source_date: str = ""
    source_order_id: str = ""
    source_kind: str = "FORECAST"

    def __post_init__(self):
        if not all(isfinite(float(x)) for x in (
                self.release_s, self.deadline_s, self.service_time_s)):
            raise ValueError("nonfinite task timing")
        if self.deadline_s < self.release_s or self.service_time_s <= 0:
            raise ValueError("invalid task deadline or predicted service duration")
        object.__setattr__(self, "request_id", int(self.request_id))
        object.__setattr__(self, "pickup", _point(self.pickup))
        object.__setattr__(self, "dropoff", _point(self.dropoff))
        object.__setattr__(self, "compatible_profiles", frozenset(self.compatible_profiles))


@dataclass(frozen=True)
class RouteState:
    position: Position
    ready_s: float
    admission_end_s: float
    profile_id: str
    context: object = None
    last_move_s: float = -900.0
    moves_used: int = 0

    def __post_init__(self):
        object.__setattr__(self, "position", _point(self.position))
        if not all(isfinite(float(x)) for x in (
                self.ready_s, self.admission_end_s, self.last_move_s)):
            raise ValueError("nonfinite route-state timing")
        if self.moves_used < 0:
            raise ValueError("negative relocation count")


class CoarseHistoricalLibrary:
    """Earlier-history M3-P50 scenarios, with no raw target-day future orders.

    ``view(now)`` returns ``{TRAIN_DAY_date: list[ChainTask]}``; ``weights`` is
    an equal-probability dictionary. Source identities are private connector
    evidence, not public diagnostics. A fine view only removes tasks already
    due and clips the horizon: it does not redraw or retimestamp them.
    """

    _COLUMNS = (
        "date", "order_id", "release_second", "start_lon_wgs84", "start_lat_wgs84",
        "end_lon_wgs84", "end_lat_wgs84", "predicted_route_time_p50_s",
        "compatible_C", "compatible_M", "compatible_A",
    )

    def __init__(self, templates: pd.DataFrame, cfg: Mapping):
        self.cfg = dict(cfg)
        self.refresh_s = int(cfg.get("coarse_reference_update_s", 300))
        self.horizon_s = int(cfg.get("planning_horizon_s", 1800))
        if self.refresh_s != 300 or self.horizon_s != 1800:
            raise ValueError("Scheme-A historical scenarios use 300-s / 1800-s layers")
        self.end_s = float(cfg.get("measurement_end_s", 86400))
        if not isfinite(self.end_s) or self.end_s <= 0:
            raise ValueError("invalid request-admission horizon")
        dates = tuple(sorted(map(str, cfg["forecast_train_dates"])))
        if not dates or len(set(dates)) != len(dates) or any(
                len(d) != 8 or not d.isdigit() or d > "20161024" for d in dates):
            raise ValueError("historical library dates must precede Test31 and end by 20161024")
        missing = set(self._COLUMNS) - set(templates.columns)
        if missing:
            raise ValueError(f"historical prediction columns missing: {sorted(missing)}")
        prohibited = [c for c in templates.columns if str(c).startswith(
            ("realized_", "actual_", "observed_", "target_"))]
        if prohibited:
            raise ValueError("historical scenarios accept prediction-only templates")
        source_dates = templates.date.astype(str)
        if any(d > "20161024" for d in set(source_dates)):
            raise ValueError("target-day or future-day data in historical template input")
        if not set(dates) <= set(source_dates):
            raise ValueError("configured historical date has no templates")
        self._days = {}
        for date in dates:
            day = templates.loc[source_dates.eq(date), list(self._COLUMNS)].copy()
            if "research_data_ready" in templates:
                ready = templates.loc[source_dates.eq(date), "research_data_ready"]
                day = day.loc[ready.eq(True)]
            day = day.sort_values(["release_second", "order_id"], kind="stable")
            rows = []
            for row in day.itertuples(index=False):
                release = float(row.release_second)
                duration = float(row.predicted_route_time_p50_s)
                if not isfinite(release) or not isfinite(duration) or duration <= 0:
                    raise ValueError("invalid historical M3-P50 task")
                allowed = frozenset({"HV"} | {
                    p for p in ("C", "M", "A") if bool(getattr(row, f"compatible_{p}"))})
                rows.append((release, duration,
                    _point((row.start_lon_wgs84, row.start_lat_wgs84)),
                    _point((row.end_lon_wgs84, row.end_lat_wgs84)), allowed, str(row.order_id)))
            self._days[date] = (np.asarray([r[0] for r in rows], dtype=float), tuple(rows))
        self.weights = {f"TRAIN_DAY_{d}": 1.0 / len(dates) for d in dates}
        self.last_refresh_s = None
        self.refresh_count = 0
        self._snapshot = {}
        self.spatial_indexes = {}

    def _refresh(self, coarse_s: int):
        horizon_end = min(float(coarse_s + self.horizon_s), self.end_s - 1.0)
        multiplier = float(self.cfg.get("forecast_sampling_multiplier", 3.0))
        if multiplier != 3.0:
            raise ValueError("historical sampling multiplier is frozen at 3.0")
        if int(self.cfg.get("forecast_seed", 20261004)) != 20261004:
            raise ValueError("historical sampling seed is frozen")
        acceptance_seed = int(self.cfg.get("passenger_acceptance_seed", 20260827))
        if acceptance_seed != 20260827:
            raise ValueError("passenger common-random-number seed is frozen")
        rate = float(self.cfg.get("passenger_acceptance_rate", 0.7))
        snapshot = {}
        for date, (times, records) in self._days.items():
            left = int(np.searchsorted(times, coarse_s, side="right"))
            right = max(left, int(np.searchsorted(times, horizon_end, side="right")))
            size = right - left
            seed_key = f"20261004|{coarse_s}|{date}"
            seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            count = int(rng.poisson(size * multiplier)) if size else 0
            sampled = rng.integers(0, size, size=count) if count else ()
            tasks = []
            ids = set()
            for index, source_index in enumerate(sampled):
                release, duration, pickup, dropoff, allowed, order = records[left + int(source_index)]
                key = f"FORECAST|{coarse_s}|{date}|{index}"
                request_id = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (1 << 63)
                # Negative synthetic IDs cannot collide with native nonnegative IDs.
                request_id = -request_id - 1
                if request_id in ids:
                    raise RuntimeError("historical task identity collision")
                ids.add(request_id)
                release = float(np.clip(release + rng.uniform(-15, 15), coarse_s + 1, horizon_end))
                accepted = passenger_acceptance(key, rate, acceptance_seed).passenger_accepts_av
                tasks.append(ChainTask(request_id, release, release + float(self.cfg.get("patience_s", 300)),
                    duration, pickup, dropoff, allowed, accepted, date, order))
            tasks.sort(key=lambda task: (task.release_s, task.request_id))
            scene = f"TRAIN_DAY_{date}"
            snapshot[scene] = tasks
        self._snapshot = snapshot
        self.spatial_indexes = {scene: _task_index(tasks) for scene, tasks in snapshot.items()}
        self.last_refresh_s = coarse_s
        self.refresh_count += 1

    def view(self, now: float) -> dict[str, list[ChainTask]]:
        if not isfinite(float(now)) or now < 0:
            raise ValueError("invalid scenario decision time")
        coarse = int(now // self.refresh_s) * self.refresh_s
        if self.last_refresh_s != coarse:
            self._refresh(coarse)
        end = min(now + self.horizon_s, self.end_s - 1.0)
        return {scene: [task for task in tasks if now < task.release_s <= end]
            for scene, tasks in self._snapshot.items()}

    def diagnostics(self) -> dict:
        return dict(source_kind="EARLIER_HISTORY_M3_P50_ONLY", refresh_count=self.refresh_count,
            coarse_reference_update_s=self.refresh_s, planning_horizon_s=self.horizon_s,
            scenario_count=len(self.weights), sampled_task_count=sum(map(len, self._snapshot.values())),
            future_target_orders_used=False, realized_service_duration_used=False,
            source_order_identities_are_private=True)


class _TaskIndex:
    def __init__(self, tasks: Sequence[ChainTask]):
        unique = {}
        for task in tasks:
            if task.request_id in unique and unique[task.request_id] != task:
                raise ValueError("conflicting task records share a request identity")
            unique[task.request_id] = task
        self.tasks = tuple(unique.values())
        self.duplicates = len(tasks) - len(self.tasks)
        self.points = np.asarray([_xy(t.pickup) for t in self.tasks], dtype=float).reshape(-1, 2)
        self.tree = cKDTree(self.points) if self.tasks else None
        self.release_times = np.asarray([t.release_s for t in self.tasks], dtype=float)
        self.deadlines = np.asarray([t.deadline_s for t in self.tasks], dtype=float)
        ids = [t.request_id for t in self.tasks]
        self.request_ids = np.asarray(ids,
            dtype=np.int64 if all(-(1 << 63) <= rid < (1 << 63) for rid in ids) else object)
        self.accepts_av = np.asarray([bool(t.passenger_accepts_av) for t in self.tasks], dtype=bool)
        profiles = set().union(*(t.compatible_profiles for t in self.tasks)) if self.tasks else set()
        self.profile_masks = {profile: np.asarray(
            [profile in t.compatible_profiles for t in self.tasks], dtype=bool) for profile in profiles}


_TASK_INDEX_CACHE: OrderedDict[int, tuple[Sequence[ChainTask], _TaskIndex]] = OrderedDict()


def _task_index(tasks: Sequence[ChainTask]) -> _TaskIndex:
    # Retaining the sequence prevents Python object-ID reuse. The caller must
    # not mutate the sequence during a decision; ChainTask itself is frozen.
    key = id(tasks)
    cached = _TASK_INDEX_CACHE.get(key)
    if cached is not None and cached[0] is tasks:
        _TASK_INDEX_CACHE.move_to_end(key)
        return cached[1]
    result = _TaskIndex(tasks)
    _TASK_INDEX_CACHE[key] = (tasks, result)
    while len(_TASK_INDEX_CACHE) > 8:
        _TASK_INDEX_CACHE.popitem(last=False)
    return result


class _ExactSpatialOrderingCache:
    """Global byte-bounded LRU of complete radius neighborhoods, never top-N.

    The key uses exact origin coordinates and the task-index identity; timing,
    admission, profile and exclusions remain per-label filters. Weak ownership
    prevents this cache retaining historical task libraries after their index
    is evicted. The 1024-byte per-entry allowance conservatively accounts for
    keys, tuples, weak references and OrderedDict entries in addition to the
    ndarray's own allocation. No coordinate rounding or eligibility shortcut
    changes the old distance/deadline/release/identity ordering.
    """

    def __init__(self):
        self.limit_bytes = 32 * 1024 * 1024
        self.resident_bytes = 1024  # Cache instance / empty-container allowance.
        self.entries = OrderedDict()

    def _remove(self, key):
        _, _, size = self.entries.pop(key)
        self.resident_bytes -= size

    def ordered(self, index: _TaskIndex, position: Position, radius: float):
        key = (id(index), position, float(radius))
        cached = self.entries.get(key)
        if cached is not None:
            owner, ordered, _ = cached
            if owner() is index:
                self.entries.move_to_end(key)
                return ordered, True
            self._remove(key)
        here = np.asarray(_xy(position))
        neighbors = index.tree.query_ball_point(here, radius) if index.tree is not None else []
        # Preserve the original scalar norm evaluation as well as all four
        # sort keys. Vectorized norms could perturb a near-tie by one ulp.
        neighbors.sort(key=lambda neighbor: (
            float(np.linalg.norm(index.points[neighbor] - here)),
            index.tasks[neighbor].deadline_s, index.tasks[neighbor].release_s,
            index.tasks[neighbor].request_id))
        ordered = np.asarray(neighbors,
            dtype=np.int32 if len(index.tasks) < (1 << 31) else np.int64)
        ordered.flags.writeable = False
        size = sys.getsizeof(ordered) + 1024
        if size + 1024 <= self.limit_bytes:
            while self.entries and self.resident_bytes + size > self.limit_bytes:
                self._remove(next(iter(self.entries)))
            self.entries[key] = (weakref.ref(index), ordered, size)
            self.resident_bytes += size
        return ordered, False


_SPATIAL_ORDER_CACHE = _ExactSpatialOrderingCache()


def _candidate_tasks(index: _TaskIndex, state: RouteState, radius: float, top_k: int,
                     step: int, horizon: float, excluded: set[int], used: tuple[int, ...]):
    """Exact old eligible Top-K using a cached complete spatial ordering.

    All eligibility masks and both diagnostic cardinalities remain exact. The
    function allocates one-dimensional arrays only, not a vehicle-task matrix.
    Returned pairs are ``(departure_s, ChainTask)`` in the old deterministic
    order; unsupported directed routes are still checked later by the caller.
    """
    ordered, cache_hit = _SPATIAL_ORDER_CACHE.ordered(index, state.position, radius)
    spatial_count = len(ordered)
    if not spatial_count or state.profile_id not in index.profile_masks:
        return [], spatial_count, 0, cache_hit
    profile = index.profile_masks[state.profile_id]
    eligible = profile[ordered].copy()
    if state.profile_id != "HV":
        eligible &= index.accepts_av[ordered]
    departures = np.ceil(np.maximum(state.ready_s, index.release_times[ordered]) / step) * step
    eligible &= departures < min(state.admission_end_s, horizon)
    eligible &= departures <= index.deadlines[ordered]
    if excluded or used:
        request_ids = index.request_ids[ordered]
        for rid in excluded.union(used):
            eligible &= request_ids != rid
    eligible_count = int(np.count_nonzero(eligible))
    selected = np.flatnonzero(eligible)[:top_k]
    return [(float(departures[offset]), index.tasks[int(ordered[offset])]) for offset in selected], \
        spatial_count, eligible_count, cache_hit


@dataclass(frozen=True)
class _Label:
    state: RouteState
    request_ids: tuple[int, ...] = ()
    activities: tuple[dict, ...] = ()
    empty_distance_m: float = 0.0
    relocation_slots: tuple[int, ...] = ()


def _rank(label: _Label):
    return (-len(label.request_ids), label.empty_distance_m, label.state.ready_s,
            label.request_ids, label.relocation_slots)


def _certified(connection, profile: str) -> bool:
    if not isinstance(connection, Mapping):
        raise TypeError("typed connector must return a mapping")
    if not bool(connection.get("supported", False)):
        return False
    if profile not in connection.get("compatible_profiles", ()):
        return False
    eta = connection.get("travel_time_s")
    distance = connection.get("empty_distance_m")
    return (eta is not None and distance is not None and isfinite(float(eta))
            and isfinite(float(distance)) and float(eta) >= 0 and float(distance) >= 0)


def build_restricted_chains(
        action_id: str, vehicle_id: int, scenario_id: str, start: RouteState,
        tasks: Sequence[ChainTask], sites: Sequence[dict], connector: Callable,
        cfg: Mapping, budget_check: Callable[[], object]) -> tuple[list[ServiceChain], dict]:
    """Return at most two directed-supported nonempty multi-activity columns.

    ``connector(state, task_or_site, departure_s, profile)`` is the sole route
    certificate. Its optional ``arrival_context`` is used after MOVE; after
    SERVICE the context is the customer's declared path, unless it supplies an
    explicit ``customer_completion_context``. Route geometry stays outside the
    columns. ``cfg.exclude_request_ids`` avoids rebuilding scene trees for the
    common first SERVE. ``planning_horizon_end_s`` should be ``now+1800`` rather
    than the completion time of that first service plus 1800.
    """
    step = int(cfg.get("rolling_step_s", cfg.get("step_s", 30)))
    radius = float(cfg.get("future_search_radius_m", 2000))
    top_k = int(cfg.get("future_top_k_tasks", 3))
    beam = int(cfg.get("chain_beam_width", 2))
    service_cap = int(cfg.get("max_future_services", 3))
    chain_cap = int(cfg.get("max_future_chains", 2))
    move_cap = int(cfg.get("max_future_relocations", 2))
    interval = int(cfg.get("reposition_interval_s", 900))
    site_k = int(cfg.get("reposition_top_k", 3))
    move_eta_cap = float(cfg.get("reposition_max_eta_s", 300))
    move_radius = float(cfg.get("reposition_radius_m", 2000))
    total_move_cap = int(cfg.get("reposition_max_moves", 50))
    label_limit = int(cfg.get("max_chain_labels", 5000))
    if (step != 30 or radius != 2000 or top_k != 3 or beam != 2 or service_cap != 3
            or chain_cap != 2 or move_cap != 2 or interval != 900 or site_k != 3
            or move_eta_cap != 300 or move_radius != 2000 or label_limit <= 0):
        raise ValueError("chain search is outside the frozen bounded Scheme-A protocol")
    horizon = float(cfg.get("planning_horizon_end_s", start.ready_s + 1800))
    decision_time = float(cfg.get("decision_time_s", start.ready_s - 1))
    if not isfinite(horizon) or not isfinite(decision_time):
        raise ValueError("invalid chain planning horizon")
    excluded = set(map(int, cfg.get("exclude_request_ids", ())))
    index = _task_index(tasks)
    site_rows = tuple(dict(site, position=_point(site["position"])) for site in sites)
    site_points = np.asarray([_xy(s["position"]) for s in site_rows], dtype=float).reshape(-1, 2)
    site_tree = cKDTree(site_points) if len(site_rows) else None
    diagnostics = dict(scope="DECLARED_RESTRICTED_CHAIN_HEURISTIC",
        full_path_domain=False, global_upper_bound_claimed=False, dense_matrix_used=False,
        input_task_count=len(index.tasks), duplicate_input_task_count=index.duplicates,
        labels_generated=1, labels_expanded=0, task_spatial_candidates=0,
        task_temporal_profile_candidates=0, task_top_k_truncated=0,
        task_spatial_order_cache_hits=0, task_spatial_order_cache_misses=0,
        site_spatial_candidates=0, site_top_k_truncated=0,
        connector_queries=0, connector_rejected=0, deadline_rejected=0,
        beam_truncated=0, chain_prefix_count=0, retained_chain_truncated=0,
        chains_returned=0, max_service_chain_length=0, max_future_relocation_count=0,
        future_relocation_candidates=0, scalar_payload_only=True)
    frontier = [_Label(start)]
    prefixes = {}

    def check_budget():
        if budget_check() is False:
            raise RuntimeError("Scheme-A chain construction budget exceeded")
        if diagnostics["labels_generated"] > label_limit:
            raise RuntimeError("Scheme-A chain-label budget exceeded")

    def keep(label):
        diagnostics["labels_generated"] += 1
        check_budget()
        if label.request_ids:
            diagnostics["chain_prefix_count"] += 1
            # Equal customer sequences with a different MOVE path are separate
            # columns because they consume different global relocation slots.
            key = (label.request_ids, label.relocation_slots)
            old = prefixes.get(key)
            if old is None or _rank(label) < _rank(old):
                prefixes[key] = label

    for _ in range(service_cap + move_cap):
        expansions = []
        for label in frontier:
            check_budget()
            state = label.state
            diagnostics["labels_expanded"] += 1
            if state.ready_s >= min(horizon, state.admission_end_s):
                continue
            here = np.asarray(_xy(state.position))
            if len(label.request_ids) < service_cap and index.tree is not None:
                candidates, spatial_count, eligible_count, cache_hit = _candidate_tasks(
                    index, state, radius, top_k, step, horizon, excluded, label.request_ids)
                diagnostics["task_spatial_candidates"] += spatial_count
                diagnostics["task_temporal_profile_candidates"] += eligible_count
                diagnostics["task_top_k_truncated"] += max(0, eligible_count - top_k)
                diagnostics["task_spatial_order_cache_hits" if cache_hit else
                    "task_spatial_order_cache_misses"] += 1
                for depart, task in candidates:
                    check_budget()
                    connection = connector(state, task, depart, state.profile_id)
                    diagnostics["connector_queries"] += 1
                    if not _certified(connection, state.profile_id):
                        diagnostics["connector_rejected"] += 1
                        continue
                    eta = float(connection["travel_time_s"])
                    distance = float(connection["empty_distance_m"])
                    pickup = depart + eta
                    if pickup > task.deadline_s:
                        diagnostics["deadline_rejected"] += 1
                        continue
                    finish = pickup + task.service_time_s
                    context = connection.get("customer_completion_context",
                        dict(kind="CUSTOMER", task=task))
                    next_state = RouteState(task.dropoff, finish, state.admission_end_s,
                        state.profile_id, context, state.last_move_s, state.moves_used)
                    activity = dict(kind="SERVICE", request_id=task.request_id,
                        departure_s=depart, pickup_s=pickup, finish_s=finish,
                        origin=state.position, target=task.pickup, dropoff=task.dropoff,
                        profile_id=state.profile_id, certified_support=True,
                        empty_distance_m=distance)
                    child = _Label(next_state, label.request_ids + (task.request_id,),
                        label.activities + (activity,), label.empty_distance_m + distance,
                        label.relocation_slots)
                    keep(child)
                    expansions.append(child)
            if (len(label.relocation_slots) < move_cap and state.moves_used < total_move_cap
                    and site_tree is not None):
                depart = float(ceil(max(state.ready_s, state.last_move_s + interval) / interval) * interval)
                if depart <= decision_time:
                    depart = float((int(decision_time // interval) + 1) * interval)
                if depart >= min(horizon, state.admission_end_s, float(cfg.get("movement_day_end_s", float("inf")))):
                    continue
                neighbors = site_tree.query_ball_point(here, move_radius)
                diagnostics["site_spatial_candidates"] += len(neighbors)
                candidates = []
                for neighbor in neighbors:
                    distance_lower = float(np.linalg.norm(site_points[neighbor] - here))
                    if distance_lower < 1e-6:
                        continue
                    site = site_rows[neighbor]
                    candidates.append((distance_lower, str(site["site_id"]), site))
                candidates.sort(key=lambda row: row[:2])
                diagnostics["site_top_k_truncated"] += max(0, len(candidates) - site_k)
                for _, _, site in candidates[:site_k]:
                    check_budget()
                    connection = connector(state, site, depart, state.profile_id)
                    diagnostics["connector_queries"] += 1
                    if not _certified(connection, state.profile_id):
                        diagnostics["connector_rejected"] += 1
                        continue
                    eta = float(connection["travel_time_s"])
                    distance = float(connection["empty_distance_m"])
                    finish = depart + eta
                    if eta > move_eta_cap or finish >= min(horizon, state.admission_end_s):
                        diagnostics["deadline_rejected"] += 1
                        continue
                    context = connection.get("arrival_context", site.get("context"))
                    next_state = RouteState(site["position"], finish, state.admission_end_s,
                        state.profile_id, context, depart, state.moves_used + 1)
                    activity = dict(kind="MOVE", site_id=str(site["site_id"]),
                        departure_s=depart, finish_s=finish, origin=state.position,
                        target=site["position"], profile_id=state.profile_id,
                        certified_support=True, empty_distance_m=distance)
                    child = _Label(next_state, label.request_ids,
                        label.activities + (activity,), label.empty_distance_m + distance,
                        label.relocation_slots + (int(depart),))
                    diagnostics["future_relocation_candidates"] += 1
                    keep(child)
                    expansions.append(child)
        if not expansions:
            break
        expansions.sort(key=_rank)
        diagnostics["beam_truncated"] += max(0, len(expansions) - beam)
        frontier = expansions[:beam]
    selected = sorted(prefixes.values(), key=_rank)[:chain_cap]
    diagnostics["retained_chain_truncated"] = max(0, len(prefixes) - chain_cap)
    chains = []
    for ordinal, label in enumerate(selected):
        chains.append(ServiceChain(chain_id=f"{action_id}|{scenario_id}|CHAIN{ordinal}",
            scenario_id=str(scenario_id), vehicle_id=int(vehicle_id), action_id=str(action_id),
            request_ids=label.request_ids, empty_distance_m=label.empty_distance_m,
            relocation_slots=label.relocation_slots,
            payload=dict(activities=list(label.activities), finish_s=label.state.ready_s,
                service_count=len(label.request_ids), future_move_count=len(label.relocation_slots),
                scope="DECLARED_RESTRICTED_CHAIN_HEURISTIC")))
    diagnostics["chains_returned"] = len(chains)
    diagnostics["max_service_chain_length"] = max((len(c.request_ids) for c in chains), default=0)
    diagnostics["max_future_relocation_count"] = max((len(c.relocation_slots) for c in chains), default=0)
    diagnostics["task_spatial_order_cache_resident_bytes"] = _SPATIAL_ORDER_CACHE.resident_bytes
    diagnostics["task_spatial_order_cache_limit_bytes"] = _SPATIAL_ORDER_CACHE.limit_bytes
    return chains, diagnostics
