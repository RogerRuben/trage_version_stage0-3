"""Opt-in native two-scale control: exact current execution, coarse future value.

No hypothetical connector or service-chain builder is called here. Current
customer pickup paths and current selected idle moves retain their original
physical/directional validators. BUSY and not-yet-online states are only a
five-minute aggregate supply input, never executable current actions.
"""
from __future__ import annotations

from collections import Counter
from math import floor, isfinite
from time import perf_counter

import numpy as np

from .flexibility_native import predicted_vehicle_states
from .scheme_a_city_adapter import NativeSchemeAAdapter, _xy, sites_for
from .scheme_a_city_graph import RouteState
from .scheme_a_city_master import CurrentAction
from .scheme_a_current_flow import solve_current_flow
from .solver import LexicographicResult


_TIMINGS = ("validator_time_s", "current_move_time_s", "coarse_time_s",
            "value_lookup_time_s", "current_flow_time_s", "solve_wall_time_s")
_BOOKED_FIELDS = ("simulation_time_s", "native_vehicle_id", "native_request_id",
                  "pickup_eta_s", "predicted_service_time_s")
_MOVE_COUNTS = ("current_move_proposed_count", "current_move_upper_screened_count",
                "current_move_routed_count", "current_move_arrival_screened_count",
                "current_move_budget_skipped_count", "current_move_value_unavailable_skipped_count")


class NativeTwoScaleAdapter(NativeSchemeAAdapter):
    """Reuse native current-action identities, not the old future search."""

    def __init__(self, policy, value_provider, reference, connectors, validator,
                 remaining_model, cfg):
        if policy not in ("SERVICE_PRESERVING", "CHAIN_DEFER"):
            raise ValueError("unknown two-scale current policy")
        fixed = {"coarse_reference_update_s": 300, "rolling_step_s": 30,
                 "reposition_interval_s": 900, "reposition_radius_m": 2000.,
                 "reposition_top_k": 3, "reposition_max_eta_s": 300.,
                 "reposition_max_moves": 50}
        if any(cfg.get(key, expected) != expected for key, expected in fixed.items()):
            raise ValueError("two-scale native execution retains the original current limits")
        if not 0 < float(cfg.get("solver_time_limit_s", 10.)) <= 10.:
            raise ValueError("current flow must retain at most a ten-second wall deadline")
        super().__init__(policy, value_provider, reference, connectors, validator,
                         remaining_model, cfg)
        self.value_provider = value_provider
        self.coarse_bucket = None
        self.coarse_info = dict(available=False, status="NOT_REFRESHED", refreshed=False)
        self.timing_totals = Counter()
        self.operation_totals = Counter()

    def _sync(self, c):
        # Store only booked, decision-time fields. Never read realized/service_end.
        for row in c.assignment_rows[self.assignment_cursor:]:
            vid = int(row["native_vehicle_id"])
            self.last_assignments[vid] = {key: row[key] for key in _BOOKED_FIELDS}
            self.move_contexts.pop(vid, None)
        self.assignment_cursor = len(c.assignment_rows)

    def _record(self, row):
        self.rows.append(row)
        for name in _TIMINGS:
            self.timing_totals[name] += float(row.get(name, 0.))
        self.operation_totals["coarse_refreshes"] += int(row.get("coarse_refreshed", False))
        self.operation_totals["current_move_connector_queries"] += int(row.get("current_move_connector_queries", 0))
        for name in _MOVE_COUNTS:
            self.operation_totals[name] += int(row.get(name, 0))
        self.operation_totals["optional_move_budget_cutoffs"] += int(row.get("optional_move_budget_cutoff", False))
        self.operation_totals["future_chain_variables"] += 0
        self.operation_totals["future_connector_queries"] += 0

    def progress_summary(self):
        return dict(decisions_recorded=len(self.rows), timings_s=dict(self.timing_totals),
                    operations=dict(self.operation_totals), timing_components_are_nested=True,
                    adapter_wall_time_field="solve_wall_time_s",
                    pre_adapter_candidate_routing_is_outside_this_timer=True, future_chain_variables=0,
                    future_connector_queries=0)

    @staticmethod
    def _validate_control(c):
        if (not isinstance(c.gammas, dict) or set(c.gammas) != {"static", "dynamic", "speed"}
                or any(value is not None for value in c.gammas.values())
                or c.cost_level_enabled):
            raise ValueError("two-scale current flow does not silently omit Gamma or platform-cost constraints")

    def _moving_wait(self, c, vid, now, profile):
        manager = c.repositioning_manager
        record = manager.rows[manager.active[vid]]
        point = tuple(map(float, c.routing_engine.return_position_coordinates(record["destination_position"])))
        ready = max(float(now + 30.), float(record["start_time_s"]) + float(record["planned_duration_s"]))
        context = self.move_contexts.get(vid, (dict(kind="EMPTY_ARRIVAL", point=point), 0.))[0]
        return RouteState(point, ready, float(c.fixture_windows_s[vid][1]), profile,
                          context, float(record["start_time_s"]), 1)

    def _current_states(self, c, valid, now, move_epoch):
        # At other epochs only vehicles with an actual SERVE choice need flow
        # columns; all other native idle vehicles keep their fixed WAIT.
        calendar = getattr(c, "event_calendar", None)
        ids = (calendar.active_ids(now) if calendar is not None else sorted(c.runtime_by_vid)) if move_epoch else (
            sorted({int(arc.vehicle_id) for arc in valid}))
        result = {}
        serving = {int(arc.vehicle_id) for arc in valid}
        for raw_vid in ids:
            vid = int(raw_vid)
            runtime = c.runtime_by_vid[vid]
            start, end = c.fixture_windows_s[vid]
            available = start <= now < end and c._available(runtime, now)
            if not available:
                if vid in serving:
                    raise ValueError("validated current SERVE uses an unavailable or committed native vehicle")
                continue
            profile = "HV" if runtime.fixture.vehicle_type == "HV" else c.config["profile_id"]
            if profile not in ("HV", "C"):
                raise ValueError("current two-scale native scope is the fixed HV/C comparison")
            if vid in c.repositioning_manager.active:
                result[vid] = self._moving_wait(c, vid, now, profile)
            else:
                point = tuple(map(float, c.routing_engine.return_position_coordinates(runtime.native_vehicle.pos)))
                result[vid] = RouteState(point, float(now + 30.), float(end), profile,
                    dict(kind="NATIVE_VEHICLE", native_vehicle_id=vid, timestamp_s=float(now)))
        if not serving <= result.keys():
            raise ValueError("validated current SERVE resource omitted from native current states")
        return result

    def _refresh_value(self, c, now):
        bucket = int(now // 300)
        if self.coarse_bucket == bucket:
            return dict(self.coarse_info, refreshed=False)
        busy = Counter()
        vehicles = predicted_vehicle_states(c, now, self.cfg.get("planning_horizon_s", 1800),
                                           self.last_assignments, self.remaining_model, busy)
        supply = {}
        for vehicle in vehicles:
            vid = int(vehicle.vehicle_id)
            if vid in c.repositioning_manager.active:
                state = self._moving_wait(c, vid, now, vehicle.profile_id)
            else:
                state = RouteState(tuple(map(float, vehicle.ready_position.split(","))),
                                   float(vehicle.ready_time_s), float(vehicle.availability_end_s),
                                   vehicle.profile_id)
            if state.ready_s < state.admission_end_s:
                supply[vid] = state
        # A live empty leg without any booked customer may not appear in the
        # generic busy predictor. Its observable planned destination still is
        # aggregate supply; do not erase it or count its current position twice.
        for vid in c.repositioning_manager.active:
            start, end = c.fixture_windows_s[int(vid)]
            if not start <= now < end:
                continue
            runtime = c.runtime_by_vid[int(vid)]
            profile = "HV" if runtime.fixture.vehicle_type == "HV" else c.config["profile_id"]
            state = self._moving_wait(c, int(vid), now, profile)
            if state.ready_s < state.admission_end_s:
                supply[int(vid)] = state
        try:
            info = self.value_provider.refresh(now, list(supply.values()))
        except TimeoutError as error:
            info = dict(available=False, status="COARSE_TIME_LIMIT", refreshed=True,
                        fallback_reason=str(error))
        if not isinstance(info, dict) or type(info.get("available")) is not bool or not info.get("status"):
            raise ValueError("coarse provider must explicitly report status and availability")
        self.coarse_bucket = bucket
        self.coarse_info = dict(info, supply_state_count=len(supply), **busy)
        return dict(self.coarse_info)

    def _value_units(self, state):
        value = float(self.value_provider.value(state))
        if not isfinite(value) or not 0. <= value <= 3.:
            raise ValueError("coarse value must be finite and bounded in [0,3]")
        return int(floor(3. * value + .5))

    def solve(self, c, original_arcs, waiting_ids, now):
        started = perf_counter()
        deadline = started + float(self.cfg.get("solver_time_limit_s", 10.))
        self._validate_control(c)
        self._sync(c)
        manager = c.repositioning_manager
        move_epoch = now % 900 == 0 and 0 <= now < self.cfg.get("movement_day_end_s", 86400)
        timings = {name: 0. for name in _TIMINGS}
        move_queries = 0
        move_counts = dict.fromkeys(_MOVE_COUNTS, 0)
        move_budget_cutoff = False
        move_skip_reason = None
        base = dict(simulation_time_s=now, policy=self.policy, future_chain_variables=0,
                    future_connector_queries=0, future_route_queries=0,
                    selected_multi_service_chains=0, selected_future_move_activities=0,
                    dense_matrix=False, gpu_used=False)

        def fixed_wait(coarse=None, unavailable=None):
            info = self.coarse_info if coarse is None else coarse
            manager.queue_actions([], now)
            timings["solve_wall_time_s"] = perf_counter() - started
            self._record(dict(base, **timings, **move_counts, opt_status="FIXED_CURRENT_WAIT", current_selected=0,
                current_move_count=0, variables=0, nonzeros=0,
                coarse_refreshed=bool(coarse is not None and info.get("refreshed", False)),
                coarse_available=bool(info["available"]), coarse_status=info["status"],
                effective_policy=self.policy if info["available"] and unavailable is None else "SERVICE_PRESERVING",
                resource_fallback=unavailable is not None, current_move_connector_queries=move_queries,
                continuation_value_unavailable_reason=unavailable, optional_move_budget_cutoff=move_budget_cutoff,
                optional_move_skip_reason=move_skip_reason, coarse_supply_state_count=info.get("supply_state_count", 0),
                adapter_wall_deadline_exceeded=timings["solve_wall_time_s"] > self.cfg.get("solver_time_limit_s", 10.)))
            return LexicographicResult((), timings["solve_wall_time_s"], 0, 0, 0,
                                       backend="TWO_SCALE_FIXED_CURRENT_WAIT")

        if not original_arcs and not move_epoch:
            return fixed_wait()
        stamp = perf_counter()
        valid = list(self.validator(c, original_arcs, now)) if original_arcs else []
        timings["validator_time_s"] = perf_counter() - stamp
        original_indices = {(int(a.vehicle_id), int(a.request_id)): i for i, a in enumerate(original_arcs)}
        if len(original_indices) != len(original_arcs):
            raise ValueError("duplicate current native arc identity")
        for arc in valid:
            key = int(arc.vehicle_id), int(arc.request_id)
            if key not in original_indices or int(arc.request_id) not in waiting_ids:
                raise ValueError("validator introduced an unknown/nonpending current native arc")
            original_arcs[original_indices[key]] = arc
        states = self._current_states(c, valid, now, move_epoch)
        if not states:
            return fixed_wait()
        actions, after = [], {}
        for vid, state in states.items():
            aid = f"V{vid}:WAIT"
            actions.append(CurrentAction(aid, vid, "WAIT", None))
            after[aid] = state
        for arc in valid:
            vid, rid = int(arc.vehicle_id), int(arc.request_id)
            task = self._task(c, rid)
            pickup = float(arc.pickup_eta_s)
            if not isfinite(pickup) or pickup < 0:
                raise ValueError("current pickup time is not a finite nonnegative prediction")
            aid = f"V{vid}:SERVE:{rid}"
            actions.append(CurrentAction(aid, vid, "SERVE", rid, bool(arc.critical), bool(arc.carry_over),
                float(arc.payload[2].route_distance_m), dict(arc_index=original_indices[vid, rid])))
            after[aid] = RouteState(task.dropoff, now + pickup + task.service_time_s,
                                   states[vid].admission_end_s, states[vid].profile_id,
                                   dict(kind="CUSTOMER", task=task))

        # Discover only the original current Top-3 site choices; no routing yet.
        proposals = []
        stamp = perf_counter()
        if move_epoch and any(vid not in manager.active for vid in states):
            sites = sites_for(self.reference, now, self.cfg)
            for vid, state in states.items():
                if vid in manager.active:
                    continue  # WAIT continues the live leg; no MOVE cancels it.
                near = sorted((float(np.linalg.norm(_xy(state.position) - _xy(site["position"]))),
                               str(site["site_id"]), site) for site in sites)
                for _, _, site in [item for item in near if 1. < item[0] <= 2000.][:3]:
                    proposals.append((vid, state, site))
        move_counts["current_move_proposed_count"] = len(proposals)
        timings["current_move_time_s"] += perf_counter() - stamp
        if not valid and not proposals:
            return fixed_wait()

        stamp = perf_counter()
        previous_bucket = self.coarse_bucket
        if perf_counter() >= deadline:
            coarse = dict(available=False, status="ADAPTER_WALL_BUDGET_EXHAUSTED_BEFORE_COARSE", refreshed=False)
            move_budget_cutoff = bool(proposals)
        else:
            coarse = self._refresh_value(c, now)
        timings["coarse_time_s"] = perf_counter() - stamp if previous_bucket != self.coarse_bucket else 0.
        units = {action.action_id: 0 for action in actions}
        effective_policy = self.policy
        unavailable = None
        stamp = perf_counter()
        if coarse["available"]:
            try:
                baseline = {}
                for vid, state in states.items():
                    if perf_counter() >= deadline:
                        raise TimeoutError("CURRENT_WALL_DEADLINE_DURING_VALUE_LOOKUP")
                    baseline[vid] = self._value_units(state)
                for action in actions:
                    if perf_counter() >= deadline:
                        raise TimeoutError("CURRENT_WALL_DEADLINE_DURING_VALUE_LOOKUP")
                    if action.kind != "WAIT":
                        units[action.action_id] = self._value_units(after[action.action_id]) - baseline[action.vehicle_id]
            except TimeoutError as error:
                units = dict.fromkeys(units, 0)
                unavailable = str(error)
                effective_policy = "SERVICE_PRESERVING"
        else:
            unavailable = coarse["status"]
            effective_policy = "SERVICE_PRESERVING"
        timings["value_lookup_time_s"] = perf_counter() - stamp
        if unavailable is not None:
            move_skip_reason = unavailable
            if "WALL" in unavailable or perf_counter() >= deadline:
                move_budget_cutoff = bool(proposals)
                move_counts["current_move_budget_skipped_count"] = len(proposals)
            else:
                move_counts["current_move_value_unavailable_skipped_count"] = len(proposals)
        else:
            # The provider's fixed-price root value is monotone nonincreasing
            # in ready time. Zero-time arrival at the target is an optimistic
            # upper bound, not a physical MOVE and never enters current flow.
            for index, (vid, state, site) in enumerate(proposals):
                if perf_counter() >= deadline:
                    move_budget_cutoff = True
                    move_counts["current_move_budget_skipped_count"] += len(proposals) - index
                    move_skip_reason = "CURRENT_WALL_DEADLINE_BEFORE_OPTIONAL_MOVE"
                    break
                optimistic = RouteState(tuple(site["position"]), float(now), state.admission_end_s,
                                        state.profile_id, site.get("context"), float(now), 1)
                stamp = perf_counter()
                try:
                    upper = self._value_units(optimistic)
                except TimeoutError as error:
                    unavailable = str(error)
                    move_skip_reason = "COARSE_VALUE_LOOKUP_TIME_LIMIT"
                    move_counts["current_move_value_unavailable_skipped_count"] += len(proposals) - index
                    break
                finally:
                    timings["value_lookup_time_s"] += perf_counter() - stamp
                if upper <= baseline[vid]:
                    move_counts["current_move_upper_screened_count"] += 1
                    continue
                if perf_counter() >= deadline:
                    move_budget_cutoff = True
                    move_counts["current_move_budget_skipped_count"] += len(proposals) - index
                    move_skip_reason = "CURRENT_WALL_DEADLINE_BEFORE_OPTIONAL_MOVE_ROUTE"
                    break
                origin = RouteState(state.position, float(now), state.admission_end_s,
                                    state.profile_id, state.context)
                stamp = perf_counter()
                link = self.connectors(origin, dict(site, require_geometry=True), now, state.profile_id)
                timings["current_move_time_s"] += perf_counter() - stamp
                move_queries += 1
                move_counts["current_move_routed_count"] += 1
                if perf_counter() >= deadline:
                    move_budget_cutoff = True
                    move_counts["current_move_budget_skipped_count"] += len(proposals) - index - 1
                    move_skip_reason = "CURRENT_OPTIONAL_MOVE_ROUTE_OVERRAN_WALL_DEADLINE"
                    break
                if (not link["supported"] or link["travel_time_s"] <= 0
                        or link["travel_time_s"] > 300. or now + link["travel_time_s"] > state.admission_end_s):
                    continue
                if link.get("routed") is None:
                    raise ValueError("admitted current MOVE has no physical executable route")
                arrival = RouteState(tuple(site["position"]), now + float(link["travel_time_s"]),
                    state.admission_end_s, state.profile_id, link["arrival_context"], float(now), 1)
                stamp = perf_counter()
                try:
                    actual = self._value_units(arrival)
                except TimeoutError as error:
                    unavailable = str(error)
                    move_skip_reason = "COARSE_VALUE_LOOKUP_TIME_LIMIT"
                    move_counts["current_move_value_unavailable_skipped_count"] += len(proposals) - index - 1
                    break
                finally:
                    timings["value_lookup_time_s"] += perf_counter() - stamp
                if actual > upper:
                    raise ValueError("coarse MOVE value violates its monotone-ready upper-bound contract")
                if actual <= baseline[vid]:
                    move_counts["current_move_arrival_screened_count"] += 1
                    continue
                aid = f"V{vid}:MOVE:{site['site_id']}"
                move = dict(native_vehicle_id=vid, position=site["position"], site_id=site["site_id"],
                            node_id=site["node_id"], routed=link["routed"])
                actions.append(CurrentAction(aid, vid, "RELOCATE", None,
                    empty_distance_m=float(link["empty_distance_m"]),
                    payload=dict(move=move, relocation_slot_s=int(now))))
                after[aid] = arrival
                units[aid] = actual - baseline[vid]
        if unavailable is not None or move_budget_cutoff:
            # Time/value degradation is explicit. No speculative or late MOVE
            # executes, but all physically validated current SERVEs remain.
            actions = [action for action in actions if action.kind != "RELOCATE"]
            units = {action.action_id: 0 for action in actions}
            effective_policy = "SERVICE_PRESERVING"
            unavailable = unavailable or move_skip_reason
        if not any(action.kind != "WAIT" for action in actions):
            return fixed_wait(coarse, unavailable)
        stamp = perf_counter()
        result = solve_current_flow(actions, units, policy=effective_policy,
            move_cap=50, time_limit_s=float(self.cfg.get("solver_time_limit_s", 10.)),
            deadline_s=deadline,
            maximum_actions=int(self.cfg.get("maximum_model_variables", 20000)))
        timings["current_flow_time_s"] = perf_counter() - stamp
        chosen = set(result["selected_action_ids"])
        serves = [action for action in actions if action.action_id in chosen and action.kind == "SERVE"]
        moves = [action for action in actions if action.action_id in chosen and action.kind == "RELOCATE"]
        for action in moves:
            self.move_contexts[action.vehicle_id] = after[action.action_id].context, float(now)
        manager.queue_actions([action.payload["move"] for action in moves], now)
        timings["solve_wall_time_s"] = perf_counter() - started
        self._record(dict(base, **timings, **move_counts, opt_status=result["opt_status"], effective_policy=effective_policy,
            current_selected=len(serves), current_move_count=len(moves), variables=len(actions),
            nonzeros=result["residual_edges"], coarse_refreshed=bool(coarse.get("refreshed", False)),
            coarse_available=bool(coarse["available"]), coarse_status=coarse["status"],
            coarse_supply_state_count=coarse.get("supply_state_count", 0),
            continuation_delta_units_selected=sum(units[action.action_id] for action in [*serves, *moves]),
            continuation_value_unavailable_reason=unavailable,
            resource_fallback=result["opt_status"].startswith("FEASIBLE") or unavailable is not None,
            current_flow_fallback_reason=result["fallback_reason"],
            current_move_connector_queries=move_queries,
            optional_move_budget_cutoff=move_budget_cutoff, optional_move_skip_reason=move_skip_reason,
            adapter_wall_deadline_exceeded=timings["solve_wall_time_s"] > self.cfg.get("solver_time_limit_s", 10.),
            current_flow_primary_optimality_proved=result["primary_optimality_proved"],
            backend=result["backend"], whole_day_optimality_claimed=False))
        return LexicographicResult(tuple(sorted(action.payload["arc_index"] for action in serves)),
            timings["solve_wall_time_s"], result["critical_now"], len(serves), result["carry_over_current_served"],
            backend="NATIVE_TWO_SCALE_" + result["backend"])
