"""Opt-in native adapter: historical Train scenarios and sparse next services.

Current pickup estimates remain deterministic Valhalla. Prospective pickup
times use a Train-M3 median OD-chord-pace heuristic, NOT realized Test outcomes.
"""
from __future__ import annotations

import hashlib
from math import isfinite
import time
from collections import OrderedDict
from functools import lru_cache

import numpy as np
from scipy.spatial import cKDTree

from .acceptance import passenger_acceptance
from .exposure import exposure_excess
from .flexibility_model import (
    CurrentPickup, CurrentServiceFace, FuturePickup, ModelLimits, Problem, Request, Scenario, Vehicle, solve_dispatch,
)
from .solver import LexicographicResult, solve_lexicographic


def position_key(lon, lat):
    return f"{float(lon):.7f},{float(lat):.7f}"


def xy(position):
    lon, lat = map(float, position.split(","))
    return np.asarray((lon * 111320 * np.cos(np.deg2rad(34.25)), lat * 110540))


@lru_cache(maxsize=50_000)
def cached_xy(position):
    point = xy(position)
    point.flags.writeable = False
    return point


class ResearchRoutePolicy:
    def __init__(self, frame, profile, frozen_profiles):
        self.rows = frame.set_index("order_id").to_dict("index")
        self.profile = profile
        self.caps = next(p for p in frozen_profiles["profiles"] if p["profile_id"] == profile)

    def evaluate(self, request):
        row = self.rows[str(request.order_id)]
        static = self.caps["static_caps"]
        ratios = [row[k] / static[c] for k, c in {
            "route_max_A_c": "external_physical_connection_count",
            "route_max_M_c": "topological_movement_count", "route_max_D_c": "road_class_diversity",
            "route_max_L_c": "internal_length_m"}.items()]
        var_ratios = [row[f"{name}_{kind}"] / self.caps["dynamic_caps"][name][kind]
                      for name in ("speed_cv", "acceleration_rms") for kind in ("E", "Q", "C")]
        rho_static = max(ratios) if all(isfinite(x) for x in ratios) else np.nan
        rho_var = max(var_ratios) if all(isfinite(x) for x in var_ratios) else np.nan
        rho_speed = row["max_route_speed_domain_kmh"] / self.caps["speed_domain_max_kmh"]
        exposure = exposure_excess(rho_static, rho_var, rho_speed)
        return dict(research_route_compatible=bool(row[f"compatible_{self.profile}"]),
                    research_base_eligible=bool(row.get("common_eligible", True)),
                    research_data_ready=bool(row["research_data_ready"]), exposure=exposure,
                    research_exposure_available=exposure is not None,
                    research_control_assumption_count=int(row["control_assumption_count"]) if isfinite(row["control_assumption_count"]) else 0,
                    research_bearing_fallback_count=int(row["bearing_fallback_count"]) if isfinite(row["bearing_fallback_count"]) else 0)


class TrainDemandForecast:
    def __init__(self, templates, cfg, measurement_end_s):
        dates = set(templates.date.astype(str))
        if dates != set(cfg["forecast_train_dates"]) or any(d > "20161024" for d in dates):
            raise ValueError("future demand library is not Train-only")
        self.days = {date: templates.loc[templates.date.astype(str).eq(date)].sort_values(
            ["release_second", "order_id"]).reset_index(drop=True) for date in sorted(dates)}
        self.cfg = cfg
        self.end_s = int(measurement_end_s)
        self._compiled_days = {}
        if cfg.get("fast_forecast", False):
            profile_sets = {}
            for date, day in self.days.items():
                records = []
                for source in day.itertuples(index=False):
                    flags = tuple(bool(getattr(source, f"compatible_{k}")) for k in ("C", "M", "A"))
                    allowed = profile_sets.setdefault(flags, frozenset(k for k, flag in zip(("C", "M", "A"), flags) if flag))
                    records.append((float(source.release_second),
                        position_key(source.start_lon_wgs84, source.start_lat_wgs84),
                        position_key(source.end_lon_wgs84, source.end_lat_wgs84),
                        float(source.predicted_route_time_p50_s), allowed))
                self._compiled_days[date] = (day.release_second.to_numpy(float), tuple(records))
        # Prediction-only chord pace absorbs a crude detour/congestion factor.
        points_start = np.stack([xy(position_key(r.start_lon_wgs84, r.start_lat_wgs84))
                                 for r in templates.itertuples(index=False)])
        points_end = np.stack([xy(position_key(r.end_lon_wgs84, r.end_lat_wgs84))
                               for r in templates.itertuples(index=False)])
        distance = np.linalg.norm(points_end - points_start, axis=1)
        valid = (distance >= 500) & np.isfinite(templates.predicted_route_time_p50_s.to_numpy(float))
        pace = templates.predicted_route_time_p50_s.to_numpy(float)[valid] / distance[valid]
        slots = (templates.release_second.to_numpy(float)[valid] // 1800).astype(int)
        self.global_pace = float(np.median(pace))
        self.pace_by_slot = {int(s): float(np.median(pace[slots == s])) for s in np.unique(slots)}
        if not isfinite(self.global_pace) or self.global_pace <= 0:
            raise ValueError("Train M3 chord pace unavailable")

    def scenarios(self, now, profile, acceptance_rate, acceptance_seed):
        if self.cfg.get("fast_forecast", False):
            return self._indexed_scenarios(now, acceptance_rate, acceptance_seed)
        result = []
        horizon_end = min(now + self.cfg["forecast_horizon_s"], self.end_s - 1)
        for scenario_index, (date, day) in enumerate(self.days.items()):
            pool = day.loc[day.release_second.gt(now) & day.release_second.le(horizon_end)]
            seed = int.from_bytes(hashlib.sha256(f'{self.cfg["forecast_seed"]}|{now}|{date}'.encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            count = int(rng.poisson(len(pool) * self.cfg["forecast_sampling_multiplier"])) if len(pool) else 0
            requests, coordinates = [], {}
            for index, source_index in enumerate(rng.integers(0, len(pool), size=count) if count else []):
                source = pool.iloc[int(source_index)]
                rid = -(scenario_index + 1) * 1_000_000 - index - 1
                release = float(np.clip(source.release_second + rng.uniform(-15, 15), now + 1, horizon_end))
                pickup = position_key(source.start_lon_wgs84, source.start_lat_wgs84)
                dropoff = position_key(source.end_lon_wgs84, source.end_lat_wgs84)
                allowed = frozenset(k for k in ("C", "M", "A") if bool(source[f"compatible_{k}"]))
                accepts = passenger_acceptance(f"FORECAST|{now}|{date}|{index}", acceptance_rate, acceptance_seed).passenger_accepts_av
                request = Request(rid, release, release + self.cfg["patience_s"],
                                  float(source.predicted_route_time_p50_s), dropoff, allowed, accepts)
                requests.append(request)
                coordinates[rid] = pickup
            result.append((Scenario(f"TRAIN_DAY_{date}", 1 / len(self.days), -7 * 86400,
                                    tuple(requests), ()), coordinates))
        return result

    def _indexed_scenarios(self, now, acceptance_rate, acceptance_seed):
        """Same pool, RNG draws and request identities; no repeated iloc/strings."""
        result = []
        horizon_end = min(now + self.cfg["forecast_horizon_s"], self.end_s - 1)
        for scenario_index, (date, (release_times, records)) in enumerate(self._compiled_days.items()):
            left = int(np.searchsorted(release_times, now, side="right"))
            right = max(left, int(np.searchsorted(release_times, horizon_end, side="right")))
            size = right - left
            seed = int.from_bytes(hashlib.sha256(f'{self.cfg["forecast_seed"]}|{now}|{date}'.encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            count = int(rng.poisson(size * self.cfg["forecast_sampling_multiplier"])) if size else 0
            requests, coordinates = [], {}
            for index, source_index in enumerate(rng.integers(0, size, size=count) if count else []):
                release_s, pickup, dropoff, duration, allowed = records[left + int(source_index)]
                rid = -(scenario_index + 1) * 1_000_000 - index - 1
                release = float(np.clip(release_s + rng.uniform(-15, 15), now + 1, horizon_end))
                accepts = passenger_acceptance(f"FORECAST|{now}|{date}|{index}", acceptance_rate, acceptance_seed).passenger_accepts_av
                requests.append(Request(rid, release, release + self.cfg["patience_s"],
                                        duration, dropoff, allowed, accepts))
                coordinates[rid] = pickup
            result.append((Scenario(f"TRAIN_DAY_{date}", 1 / len(self.days), -7 * 86400,
                                    tuple(requests), ()), coordinates))
        return result


def predicted_vehicle_states(c, now, horizon, last_assignments, remaining_model=None, diagnostics=None):
    """Only observable availability and already-booked predicted tasks enter.

    No service_end_time, realized duration, future request enumeration, or
    position after an unbooked order is accessed here.
    """
    result = []
    calendar = getattr(c, "event_calendar", None)
    ids = calendar.horizon_ids(now, horizon) if calendar is not None else sorted(c.runtime_by_vid)
    for vid in ids:
        runtime = c.runtime_by_vid[vid]
        fixture = runtime.fixture
        if calendar is not None:
            start, end = calendar.windows[vid]
        else:
            start = c._fixture_seconds(fixture.availability_start_time)
            end = c._fixture_seconds(fixture.availability_end_time)
        if end <= now or start > now + horizon:
            continue
        native_free = runtime.native_vehicle.status == c.bindings.states.IDLE and not runtime.native_vehicle.assigned_route
        if native_free:
            lon, lat = c.routing_engine.return_position_coordinates(runtime.native_vehicle.pos)
            ready = max(float(now), start)
            position = position_key(lon, lat)
        else:
            assignment = last_assignments.get(int(vid))
            if assignment is None:
                continue
            predicted = float(assignment["predicted_service_time_s"])
            if not isfinite(predicted) or predicted <= 0:
                continue
            pickup = float(assignment["simulation_time_s"]) + float(assignment["pickup_eta_s"]) + c._pickup_overhead_s()
            ready = pickup + predicted
            legacy_ready = max(ready, now + c.dispatch_interval_s, start)
            if diagnostics is not None:
                diagnostics["busy_states"] += 1
                diagnostics["overdue_busy_states"] += int(ready <= now)
            if remaining_model is not None:
                estimate = remaining_model.estimate(predicted, max(0.0, now - pickup))
                if estimate.remaining_s is None:
                    if diagnostics is not None:
                        diagnostics["unsupported_busy_states_omitted"] += 1
                    continue
                ready = max(now, pickup) + estimate.remaining_s
                if diagnostics is not None:
                    diagnostics["conditional_busy_states"] += 1
                    diagnostics["busy_ready_shift_sum_s"] += max(ready, now + c.dispatch_interval_s, start) - legacy_ready
            # Still observed busy at t: a past predicted finish cannot make it idle NOW.
            ready = max(ready, now + c.dispatch_interval_s, start)
            booked = c.request_by_rid[int(assignment["native_request_id"])]
            position = position_key(booked.dropoff_lon_wgs84, booked.dropoff_lat_wgs84)
        if ready > end or ready > now + horizon + c.max_pickup_wait_s:
            continue
        result.append(Vehicle(int(vid), "HV" if fixture.vehicle_type == "HV" else c.config["profile_id"],
                              position, ready, end))
    return tuple(result)


class NativeFlexibilityAdapter:
    def __init__(self, policy, forecast, cfg, remaining_model=None):
        self.policy, self.forecast, self.cfg = policy, forecast, cfg
        self.remaining_model = remaining_model
        self.last_assignments = {}
        self.assignment_cursor = 0
        self.rows = []
        if cfg.get("solver_backend") == "HIGHS_PERSISTENT":
            from .persistent_highs import load_highspy
            load_highspy(cfg.get("highspy_runtime_dir"))

    def _refresh_booked(self, c):
        columns = ("simulation_time_s", "native_vehicle_id", "native_request_id", "pickup_eta_s", "predicted_service_time_s")
        for row in c.assignment_rows[self.assignment_cursor:]:
            self.last_assignments[int(row["native_vehicle_id"])] = {k: row[k] for k in columns}
        self.assignment_cursor = len(c.assignment_rows)

    def _problem(self, c, arcs, waiting_ids, now):
        self._refresh_booked(c)
        self.busy_diagnostics = dict(busy_states=0, overdue_busy_states=0, conditional_busy_states=0,
                                    unsupported_busy_states_omitted=0, busy_ready_shift_sum_s=0.0)
        vehicles = predicted_vehicle_states(c, now, self.cfg["forecast_horizon_s"], self.last_assignments,
                                            self.remaining_model, self.busy_diagnostics)
        requests = []
        coordinates = {}
        for rid in waiting_ids:
            source = c.request_by_rid[int(rid)]
            if not isfinite(source.predicted_service_time_s) or source.predicted_service_time_s <= 0:
                continue
            meta = c.request_meta[int(rid)]
            if not meta.get("research_base_eligible", True):
                continue
            allowed = frozenset((c.config["profile_id"],)) if meta["research_route_compatible"] else frozenset()
            remaining = meta["pickup_deadline_s"] - now
            requests.append(Request(int(rid), source.sim_time_s, meta["pickup_deadline_s"],
                source.predicted_service_time_s, position_key(source.dropoff_lon_wgs84, source.dropoff_lat_wgs84),
                allowed, meta["passenger_accepts_av"], c._pickup_overhead_s(),
                0 < remaining <= c.dispatch_interval_s, bool(meta["carry_over_flag"])))
            coordinates[int(rid)] = position_key(source.pickup_lon_wgs84, source.pickup_lat_wgs84)
        limits = ModelLimits(horizon_s=self.cfg["forecast_horizon_s"], rolling_step_s=c.dispatch_interval_s,
            max_variables=self.cfg["max_model_variables"], max_nonzeros=self.cfg["max_model_nonzeros"],
            solver_time_limit_s=self.cfg["solver_time_limit_s"],
            recourse_mode=self.cfg.get("recourse_mode", "BINARY"),
            solver_backend=self.cfg.get("solver_backend", "SCIPY"),
            highspy_runtime_dir=self.cfg.get("highspy_runtime_dir"),
            lock_current_face=self.cfg.get("lock_current_face", False),
            compress_fixed_states=self.cfg.get("compress_fixed_states", False),
            trim_flow_rows=self.cfg.get("trim_flow_rows", False))
        problem = Problem(now, vehicles, tuple(requests),
                          tuple(CurrentPickup(a.vehicle_id, a.request_id, a.pickup_eta_s) for a in arcs), (), limits)
        if self.policy not in ("LOOKAHEAD", "SERVICE_PRESERVING_LOOKAHEAD"):
            return problem
        future_started = time.perf_counter()
        by_rid = {r.request_id: r for r in requests}
        options = [(v, None, v.ready_time_s, v.ready_position) for v in vehicles]
        vehicle_map = {v.vehicle_id: v for v in vehicles}
        for arc in arcs:
            v, r = vehicle_map[arc.vehicle_id], by_rid[arc.request_id]
            options.append((v, r.request_id, now + arc.pickup_eta_s + r.pickup_overhead_s + r.predicted_service_time_s, r.dropoff_position))
        fast_graph = self.cfg.get("fast_future_graph", False)
        project = cached_xy if fast_graph else xy
        points = np.stack([project(state[3]) for state in options]) if options else np.empty((0, 2))
        tree = cKDTree(points) if options else None
        scenarios = []
        sampling_started = time.perf_counter()
        scenario_inputs = self.forecast.scenarios(now, c.config["profile_id"], c.acceptance_rate, c.acceptance_seed)
        self.stage_timings["forecast_sampling_time_s"] = time.perf_counter() - sampling_started
        if fast_graph:
            # Geometry is local to this epoch: live ready states may change.
            # Bound the cache, not a requests x vehicles distance matrix.
            geometry = OrderedDict()

            def pickups_for(request, pickup_position):
                nearby = geometry.get(pickup_position)
                if nearby is None:
                    point = project(pickup_position)
                    nearby = tuple((int(i), float(np.linalg.norm(points[i] - point)))
                        for i in tree.query_ball_point(point, self.cfg["future_search_radius_m"])) if tree else ()
                    geometry[pickup_position] = nearby
                    if len(geometry) > 128:
                        geometry.popitem(last=False)
                else:
                    geometry.move_to_end(pickup_position)
                candidates = []
                for index, distance in nearby:
                    v, after, ready, origin = options[index]
                    if v.profile_id != "HV" and (v.profile_id not in request.compatible_profiles or not request.passenger_accepts_av):
                        continue
                    departure = max(now + c.dispatch_interval_s, ready, request.release_time_s)
                    # Nonnegative ETA certificate, independent of routing.
                    if (departure > request.pickup_deadline_s or
                            departure + request.pickup_overhead_s + request.predicted_service_time_s > v.availability_end_s):
                        continue
                    pace = self.forecast.pace_by_slot.get(int(departure // 1800), self.forecast.global_pace)
                    eta = distance * pace
                    if (departure + eta > request.pickup_deadline_s or
                            departure + eta + request.pickup_overhead_s + request.predicted_service_time_s > v.availability_end_s):
                        continue
                    candidates.append((distance, v.vehicle_id, -1 if after is None else after, after, origin, eta))
                candidates.sort(key=lambda x: x[:3])
                chosen = set()
                for _, vid, *_ in candidates:
                    if len(chosen) < self.cfg["future_top_k_vehicles"]:
                        chosen.add(vid)
                return tuple(FuturePickup(vid, after, request.request_id, eta, origin)
                             for _, vid, _, after, origin, eta in candidates if vid in chosen)

            # Waiting requests are identical across all forecast scenarios.
            known_pickups = tuple(p for r in requests for p in pickups_for(r, coordinates[r.request_id]))
            from dataclasses import replace
            for scenario, pickups in scenario_inputs:
                future_arcs = list(known_pickups)
                for r in scenario.new_requests:
                    future_arcs.extend(pickups_for(r, pickups[r.request_id]))
                scenarios.append(replace(scenario, pickups=tuple(future_arcs)))
                if len(options) + sum(len(s.pickups) for s in scenarios) > limits.max_variables:
                    raise ValueError("variable resource cap exceeded during sparse future graph construction")
            self.stage_timings["future_graph_time_s"] = time.perf_counter() - future_started - self.stage_timings["forecast_sampling_time_s"]
            return replace(problem, scenarios=tuple(scenarios))
        for scenario, pickups in scenario_inputs:
            all_requests = (*requests, *scenario.new_requests)
            destination = {**coordinates, **pickups}
            future_arcs = []
            for r in all_requests:
                point = xy(destination[r.request_id])
                candidates = []
                for index in tree.query_ball_point(point, self.cfg["future_search_radius_m"]) if tree else []:
                    v, after, ready, origin = options[index]
                    if v.profile_id != "HV" and (v.profile_id not in r.compatible_profiles or not r.passenger_accepts_av):
                        continue
                    departure = max(now + c.dispatch_interval_s, ready, r.release_time_s)
                    distance = float(np.linalg.norm(points[index] - point))
                    pace = self.forecast.pace_by_slot.get(int(departure // 1800), self.forecast.global_pace)
                    eta = distance * pace
                    if departure + eta > r.pickup_deadline_s or departure + eta + r.pickup_overhead_s + r.predicted_service_time_s > v.availability_end_s:
                        continue
                    candidates.append((distance, v.vehicle_id, -1 if after is None else after, after, origin, eta))
                candidates.sort(key=lambda x: x[:3])
                chosen_vehicles = []
                for _, vid, *_ in candidates:
                    if vid not in chosen_vehicles and len(chosen_vehicles) < self.cfg["future_top_k_vehicles"]:
                        chosen_vehicles.append(vid)
                for _, vid, _, after, origin, eta in candidates:
                    if vid in chosen_vehicles:
                        future_arcs.append(FuturePickup(vid, after, r.request_id, eta, origin))
            from dataclasses import replace
            scenarios.append(replace(scenario, pickups=tuple(future_arcs)))
        from dataclasses import replace
        self.stage_timings["future_graph_time_s"] = time.perf_counter() - future_started - self.stage_timings["forecast_sampling_time_s"]
        return replace(problem, scenarios=tuple(scenarios))

    def solve(self, c, arcs, waiting_ids, now):
        started = time.perf_counter()
        fallback = None
        decision = None
        self.stage_timings = dict(forecast_sampling_time_s=0.0, future_graph_time_s=0.0,
                                 problem_setup_time_s=0.0, failed_model_attempt_time_s=0.0)
        protected = self.policy == "SERVICE_PRESERVING_LOOKAHEAD"
        current_oracle = solve_lexicographic(arcs) if protected else None
        self.busy_diagnostics = {}
        if self.policy == "MYOPIC" or not arcs:
            result = solve_lexicographic(arcs)
        else:
            try:
                preparation_started = time.perf_counter()
                problem = self._problem(c, arcs, waiting_ids, now)
                self.stage_timings["problem_setup_time_s"] = max(0.0, time.perf_counter() - preparation_started
                    - self.stage_timings["forecast_sampling_time_s"] - self.stage_timings["future_graph_time_s"])
                attempt_started = time.perf_counter()
                if self.cfg.get("lock_current_face", False) and current_oracle is not None:
                    face = CurrentServiceFace.from_arcs(arcs, current_oracle)
                    decision = solve_dispatch(problem, self.policy, current_face=face)
                else:
                    decision = solve_dispatch(problem, self.policy)
                selected = set(decision.selected_pairs)
                indices = tuple(i for i, a in enumerate(arcs) if (a.vehicle_id, a.request_id) in selected)
                if len(indices) != len(selected):
                    raise ValueError("new kernel returned an unknown current native arc")
                chosen = [arcs[i] for i in indices]
                result = LexicographicResult(indices, time.perf_counter() - started,
                    sum(a.critical for a in chosen), len(chosen), sum(a.carry_over for a in chosen),
                    backend="SPARSE_TWO_STAGE_RESEARCH", pickup_eta_optimum_s=sum(a.pickup_eta_s for a in chosen))
            except (ValueError, RuntimeError) as error:
                # Only the preregistered RESOURCE/TIME fallback, never hide data bugs.
                message = str(error).casefold()
                time_limited = ("solver timeout" in message or
                    ("sparse model not proven optimal:" in message and "time limit" in message))
                if "resource cap" not in message and not time_limited:
                    raise
                fallback = str(error)
                if "attempt_started" in locals():
                    self.stage_timings["failed_model_attempt_time_s"] = time.perf_counter() - attempt_started
                result = solve_lexicographic(arcs)
        self.rows.append(dict(simulation_time_s=now, policy=self.policy, current_arc_count=len(arcs),
            current_selected=result.total_matched, solver_time_s=time.perf_counter() - started,
            resource_fallback=fallback, variable_count=decision.variable_count if decision else None,
            nonzeros=decision.constraint_nonzeros if decision else None,
            expected_next_service=decision.expected_next_service_count if decision else None,
            integer_variable_count=decision.integer_variable_count if decision else None,
            eliminated_fixed_variables=decision.eliminated_fixed_variables if decision else 0,
            model_build_time_s=decision.model_build_time_s if decision else 0.0,
            optimization_time_s=decision.solve_time_s if decision else result.solve_time_s,
            recourse_recovery_time_s=decision.recourse_recovery_time_s if decision else 0.0,
            **self.stage_timings))
        if protected:
            actual = (result.critical_matched, result.total_matched, result.carry_over_matched)
            optimum = (current_oracle.critical_matched, current_oracle.total_matched,
                       current_oracle.carry_over_matched)
            if actual != optimum:
                raise RuntimeError("service-preserving current lexicographic face violated")
            self.rows[-1].update(current_critical_optimum=optimum[0], current_service_optimum=optimum[1],
                current_carry_optimum=optimum[2], current_face_preserved=True, **self.busy_diagnostics)
        return result
