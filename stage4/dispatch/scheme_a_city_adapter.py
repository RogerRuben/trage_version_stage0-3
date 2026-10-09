"""Native Scheme-A adapter over restricted multi-service chain candidates.

Placement and dispatch share the restricted-chain master. Physical connector
work is separately timed from the ten-second OR/model budget. No future truth
or scenario action is sent to the native executor.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
from math import ceil, cos, pi
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from .flexibility_native import predicted_vehicle_states
from .scheme_a_city_graph import ChainTask, RouteState, build_restricted_chains
from .scheme_a_city_master import CurrentAction, solve_city_master
from .solver import LexicographicResult


def _xy(point):
    return np.asarray((point[0] * 111320 * cos(34.25*pi/180), point[1] * 110540))


def _rank(text, seed=20261009):
    return hashlib.sha256(f"{seed}|{text}".encode()).digest()


def sites_for(reference, now, cfg):
    slot = int(now // 900) % 96
    rows = reference.loc[reference.time_bin_index.eq(slot)].sort_values(
        ["demand_share", "node_id"], ascending=[False, True], kind="stable")
    rows = rows.head(cfg["layout_candidate_site_count"])
    if rows.empty:
        raise ValueError("no earlier-history common site pool for this time slot")
    return [dict(site_id=f"NODE_{int(r.node_id)}", node_id=int(r.node_id),
        position=(float(r.lon_wgs84), float(r.lat_wgs84)), context=None,
        demand_share=float(r.demand_share)) for r in rows.itertuples(index=False)]


def _column_allocation(actions, scenes, states, limit, now, horizon):
    """Finite round-robin/hash chain budget, without dropping current actions.

    This is explicitly another restricted-column approximation. Allocation
    depends on identities/time and declared caps, not the outcome of a run.
    """
    available = int(limit) - len(actions)
    if available < 0:
        raise ValueError("current first-action domain alone exceeds fixed model cap")
    groups = defaultdict(list)
    for action in actions:
        state = states[action.action_id]
        if state.ready_s >= min(state.admission_end_s, horizon):
            continue
        for scene in scenes:
            groups[action.vehicle_id].append((action.action_id, scene))
    for vid, pairs in groups.items():
        pairs.sort(key=lambda pair: _rank(f"{now}|{vid}|{pair[0]}|{pair[1]}"))
    vids = sorted(groups, key=lambda vid: _rank(f"{now}|{vid}"))
    order = [groups[vid][index] for index in range(max(map(len, groups.values()), default=0))
        for vid in vids if index < len(groups[vid])]
    quotas = Counter()
    for _ in range(2):
        for key in order:
            if not available:
                break
            quotas[key] += 1
            available -= 1
    return dict(quotas), dict(eligible_option_scene_groups=len(order),
        zero_column_budget_groups=sum(key not in quotas for key in order),
        maximum_requested_chain_columns=2*len(order),
        allocated_chain_column_budget=sum(quotas.values()),
        allocation_rule="IDENTITY_HASH_VEHICLE_ROUND_ROBIN_MAX_TWO_NO_FIRST_ACTION_TRUNCATION",
        allocation_is_declared_restricted_domain=True)


def build_and_solve(actions, states, scenes, weights, connectors, sites, cfg, now, policy,
        *, pending=(), offline_layout=False, diagnostics_sink=None):
    started = perf_counter()
    evidence_before = float(connectors.timings["total_connection_time_s"])
    graph_limit = cfg["layout_graph_cpu_limit_s"] if offline_layout else cfg["solver_time_limit_s"]
    diagnostic = dict(stage="GRAPH_CONSTRUCTION", decision_time_s=now,
        offline_layout=offline_layout, graph_cpu_limit_s=graph_limit,
        realtime_or_limit_s=cfg["solver_time_limit_s"], current_action_count=len(actions),
        scenario_count=len(scenes), completed_option_scene_groups=0,
        generated_chain_columns=0)

    def or_elapsed():
        evidence = float(connectors.timings["total_connection_time_s"]) - evidence_before
        return max(0., perf_counter() - started - evidence)

    def check():
        if or_elapsed() >= graph_limit:
            diagnostic.update(graph_cpu_s=or_elapsed(),
                connector_evidence_time_s=float(connectors.timings["total_connection_time_s"])-evidence_before,
                wall_time_s=perf_counter()-started)
            if diagnostics_sink is not None:
                diagnostics_sink(dict(diagnostic))
            raise TimeoutError("Scheme-A graph/model OR budget exceeded; no fallback")

    horizon = min(float(now + cfg["planning_horizon_s"]), float(cfg["admission_end_s"]))
    quota, allocation = _column_allocation(actions, scenes, states,
        cfg["maximum_model_variables"], now, horizon)
    scene_tasks = {scene: [*pending, *tasks] for scene, tasks in scenes.items()}
    chains, statistics = [], Counter()
    for action in actions:
        for scene, tasks in scene_tasks.items():
            capacity = quota.get((action.action_id, scene), 0)
            if not capacity:
                continue
            diagnostic.update(current_action_id=action.action_id, current_scene=scene,
                current_scene_tasks=len(tasks))
            options = dict(cfg, decision_time_s=now, planning_horizon_end_s=horizon,
                exclude_request_ids=(() if action.request_id is None else (action.request_id,)))
            built, info = build_restricted_chains(action.action_id, action.vehicle_id, scene,
                states[action.action_id], tasks, sites, connectors, options, check)
            statistics["column_budget_discarded_chains"] += max(0, len(built)-capacity)
            chains.extend(built[:capacity])
            diagnostic["completed_option_scene_groups"] += 1
            diagnostic["generated_chain_columns"] = len(chains)
            for field in ("labels_generated", "labels_expanded", "connector_queries",
                "connector_rejected", "deadline_rejected", "beam_truncated", "retained_chain_truncated"):
                statistics[field] += int(info.get(field, 0))
            statistics["maximum_generated_service_chain_length"] = max(
                statistics["maximum_generated_service_chain_length"], info["max_service_chain_length"])
            statistics["maximum_generated_future_relocations"] = max(
                statistics["maximum_generated_future_relocations"], info["max_future_relocation_count"])
    check()
    graph_or_s = or_elapsed()
    graph_wall_s = perf_counter() - started
    diagnostic.update(stage="INTEGER_MASTER", graph_cpu_s=graph_or_s,
        connector_evidence_time_s=graph_wall_s-graph_or_s)
    master_budget = (cfg.get("layout_solver_time_limit_s", cfg["solver_time_limit_s"])
        if offline_layout else cfg["solver_time_limit_s"]-graph_or_s)
    diagnostic["integer_master_limit_s"] = master_budget
    try:
        solved = solve_city_master(actions, chains, weights, policy=policy,
            time_limit_s=master_budget,
            max_variables=cfg["maximum_model_variables"], max_nonzeros=cfg["maximum_model_nonzeros"],
            relocation_cap=cfg["reposition_max_moves"])
    except Exception as error:
        diagnostic.update(error=repr(error), generated_chain_columns=len(chains))
        if diagnostics_sink is not None:
            diagnostics_sink(dict(diagnostic))
        raise
    selected = set(solved["selected_chain_ids"])
    chosen = [chain for chain in chains if chain.chain_id in selected]
    solved["restricted_graph"] = dict(statistics, **allocation,
        generated_columns=len(chains), max_selected_chain_length=max((len(c.request_ids) for c in chosen), default=0),
        selected_multi_service_chains=sum(len(c.request_ids) >= 2 for c in chosen),
        selected_future_move_activities=sum(len(c.relocation_slots) for c in chosen),
        horizon_s=cfg["planning_horizon_s"], coarse_history_update_s=cfg["coarse_reference_update_s"],
        typed_connector_evidence=True, complete_city_chain_domain=False,
        global_optimality_bound_claimed=False, dense_matrix=False)
    solved["graph_or_time_s"] = graph_or_s
    solved["graph_wall_time_s"] = graph_wall_s
    solved["graph_connector_evidence_time_s"] = graph_wall_s-graph_or_s
    solved["total_or_time_s"] = graph_or_s + solved["runtime_s"]
    solved["offline_layout_graph_budget_separate"] = offline_layout
    solved["graph_cpu_limit_s"] = graph_limit
    solved["integer_master_limit_s"] = master_budget
    if not offline_layout and solved["total_or_time_s"] > cfg["solver_time_limit_s"]:
        raise TimeoutError("total Scheme-A OR/model budget exceeded")
    return solved


def compact_master(result):
    model = result["model"]
    return {k: model[k] for k in ("columns", "rows", "nonzeros", "current_action_variables",
        "optional_chain_variables", "component_count", "independent_single_vehicle_components",
        "coupled_milp_components", "future_relocation_slot_rows")}


class NativeSchemeAAdapter:
    def __init__(self, policy, library, reference, connectors, validator, remaining_model, cfg):
        self.policy, self.library, self.reference = policy, library, reference
        self.connectors, self.validator, self.remaining_model, self.cfg = connectors, validator, remaining_model, cfg
        self.rows = []
        self.last_assignments = {}
        self.assignment_cursor = 0
        self.move_contexts = {}

    def _task(self, c, rid):
        request = c.request_by_rid[int(rid)]
        allowed = frozenset({"HV"} | set(request.compatible_profiles))
        return ChainTask(int(rid), float(request.sim_time_s), float(c.request_meta[rid]["pickup_deadline_s"]),
            float(request.predicted_service_time_s),
            (float(request.pickup_lon_wgs84), float(request.pickup_lat_wgs84)),
            (float(request.dropoff_lon_wgs84), float(request.dropoff_lat_wgs84)),
            allowed, bool(request.passenger_accepts_av), self.cfg["test_date"], str(request.order_id), "ACTUAL_PENDING")

    def _sync(self, c):
        for row in c.assignment_rows[self.assignment_cursor:]:
            vid = int(row["native_vehicle_id"])
            self.last_assignments[vid] = row
            self.move_contexts.pop(vid, None)
        self.assignment_cursor = len(c.assignment_rows)

    def solve(self, c, original_arcs, waiting_ids, now):
        self._sync(c)
        started = perf_counter()
        valid = self.validator(c, original_arcs, now)
        physical_s = perf_counter()-started
        original_indices = {(int(a.vehicle_id), int(a.request_id)): i for i, a in enumerate(original_arcs)}
        for arc in valid:
            original_arcs[original_indices[int(arc.vehicle_id), int(arc.request_id)]] = arc
        pending = [self._task(c, rid) for rid in waiting_ids]
        by_request = {task.request_id: task for task in pending}
        scenes = self.library.view(now)
        sites = sites_for(self.reference, now, self.cfg)
        busy = Counter()
        vehicles = predicted_vehicle_states(c, now, self.cfg["planning_horizon_s"],
            self.last_assignments, self.remaining_model, busy)
        actions, starts = [], {}
        manager = c.repositioning_manager
        for vehicle in vehicles:
            vid = int(vehicle.vehicle_id)
            runtime = c.runtime_by_vid[vid]
            activation, end = c.fixture_windows_s[vid]
            prior = self.last_assignments.get(vid)
            context = (dict(kind="CUSTOMER", task=self._task(c, int(prior["native_request_id"])))
                if prior is not None else None)
            last_move, moves = -900., 0
            if vid in self.move_contexts:
                context, last_move = self.move_contexts[vid]
            if vid in manager.active:
                record = manager.rows[manager.active[vid]]
                target = tuple(c.routing_engine.return_position_coordinates(record["destination_position"]))
                ready = max(now+30., record["start_time_s"]+record["planned_duration_s"])
                context = self.move_contexts[vid][0]
                position, last_move = target, float(record["start_time_s"])
                kind = "BUSY"
            else:
                position = tuple(map(float, vehicle.ready_position.split(",")))
                ready = float(vehicle.ready_time_s)
                free = activation <= now and c._available(runtime, now)
                kind = "WAIT" if free else "BUSY"
                if free:
                    ready = now+30.
                    position = tuple(map(float, c.routing_engine.return_position_coordinates(runtime.native_vehicle.pos)))
                    # Incoming native prefixes must be reconstructed from the
                    # observed physical state, not from an unbooked future job.
                    context = dict(kind="NATIVE_VEHICLE", native_vehicle_id=vid, timestamp_s=now)
                elif activation > now:
                    context = None
            state = RouteState(position, ready, float(end), vehicle.profile_id, context, last_move, moves)
            aid = f"V{vid}:{kind}"
            actions.append(CurrentAction(aid, vid, kind, None))
            starts[aid] = state
            if (now % 900 == 0 and now < self.cfg["movement_day_end_s"] and kind == "WAIT"
                and vid not in manager.active):
                near = sorted((float(np.linalg.norm(_xy(state.position)-_xy(s["position"]))), s["site_id"], s)
                    for s in sites)
                for distance, _, site in [n for n in near if 1 < n[0] <= 2000][:3]:
                    target = dict(site, require_geometry=True)
                    origin_state = replace(state, ready_s=float(now))
                    link = self.connectors(origin_state, target, now, vehicle.profile_id)
                    if (not link["supported"] or link["travel_time_s"] <= 0
                        or link["travel_time_s"] > 300 or now+link["travel_time_s"] > end):
                        continue
                    mid = f"V{vid}:MOVE:{site['site_id']}"
                    move = dict(native_vehicle_id=vid, position=site["position"],
                        site_id=site["site_id"], node_id=site["node_id"], routed=link["routed"])
                    actions.append(CurrentAction(mid, vid, "RELOCATE", None,
                        empty_distance_m=link["empty_distance_m"],
                        payload=dict(move=move, relocation_slot_s=int(now))))
                    starts[mid] = RouteState(site["position"], now+link["travel_time_s"], end,
                        vehicle.profile_id, link["arrival_context"], float(now), moves+1)
        known_vids = {a.vehicle_id for a in actions}
        for arc in valid:
            vid, rid = int(arc.vehicle_id), int(arc.request_id)
            if vid not in known_vids:
                raise RuntimeError("actual available vehicle omitted from Scheme-A resource states")
            task = by_request[rid]
            end = c.fixture_windows_s[vid][1]
            aid = f"V{vid}:SERVE:{rid}"
            distance = float(arc.payload[2].route_distance_m)
            actions.append(CurrentAction(aid, vid, "SERVE", rid, bool(arc.critical),
                bool(arc.carry_over), distance, dict(arc_index=original_indices[vid,rid])))
            starts[aid] = RouteState(task.dropoff, now+float(arc.pickup_eta_s)+task.service_time_s,
                end, "HV" if arc.vehicle_type == "HV" else "C", dict(kind="CUSTOMER", task=task))
        if not any(a.kind in ("SERVE", "RELOCATE") for a in actions):
            manager.queue_actions([], now)
            self.rows.append(dict(simulation_time_s=now, policy=self.policy,
                opt_status="FIXED_CURRENT_ACTIONS", current_selected=0,
                physical_validation_time_s=physical_s, graph_or_time_s=0., master_time_s=0.,
                total_or_time_s=0., variables=len(actions), nonzeros=0,
                selected_multi_service_chains=0, selected_future_move_activities=0,
                expected_served=None, resource_fallback=False))
            return LexicographicResult((), 0., 0, 0, 0, backend="SCHEME_A_FIXED_CURRENT_ACTIONS")
        result = build_and_solve(actions, starts, scenes, self.library.weights,
            self.connectors, sites, self.cfg, now, self.policy, pending=pending)
        selected = set(result["selected_action_ids"])
        serves = [a for a in actions if a.action_id in selected and a.kind == "SERVE"]
        moves = [a for a in actions if a.action_id in selected and a.kind == "RELOCATE"]
        for action in moves:
            self.move_contexts[action.vehicle_id] = (starts[action.action_id].context, float(now))
        manager.queue_actions([a.payload["move"] for a in moves], now)
        self.rows.append(dict(simulation_time_s=now, policy=self.policy, opt_status=result["opt_status"],
            current_selected=len(serves), physical_validation_time_s=physical_s,
            graph_or_time_s=result["graph_or_time_s"], graph_wall_time_s=result["graph_wall_time_s"],
            graph_connector_evidence_time_s=result["graph_connector_evidence_time_s"],
            master_time_s=result["runtime_s"], total_or_time_s=result["total_or_time_s"],
            variables=result["model"]["columns"], nonzeros=result["model"]["nonzeros"],
            expected_served=result["expected_served"], current_move_count=len(moves),
            selected_multi_service_chains=result["restricted_graph"]["selected_multi_service_chains"],
            selected_future_move_activities=result["restricted_graph"]["selected_future_move_activities"],
            max_selected_chain_length=result["restricted_graph"]["max_selected_chain_length"],
            resource_fallback=False, model=compact_master(result), graph=result["restricted_graph"], **busy))
        return LexicographicResult(tuple(sorted(a.payload["arc_index"] for a in serves)),
            result["total_or_time_s"], result["critical_now"], len(serves), result["carry_over_current_served"],
            backend="SCHEME_A_RESTRICTED_MULTI_SERVICE_CHAIN_MASTER")


def plan_av_layout(episode, library, reference, connectors, cfg, guard, progress,
        *, checkpoint=None, diagnostics_sink=None, restored=None):
    """Earlier-history mixed-resource placement proxy; not target-day rollout.

    Batches contain the AV slots activating in one 30-min bin. Other planned
    slots are held at their preparation positions; they are NOT described as
    observed mid-day HV locations. All positions are frozen before native runs.
    """
    plans = episode.supply.plan()
    bins = sorted({int(p.activation_s // cfg["layout_planning_bin_s"])
        for p in plans if p.vehicle_type == "AV"})
    hotspot, joint, records = {}, {}, []
    completed = 0
    if restored is not None:
        completed = int(restored["completed_bins"])
        if (restored["total_bins"] != len(bins) or completed != len(restored["records"])
            or [r["bin_s"] for r in restored["records"]] != [b*cfg["layout_planning_bin_s"] for b in bins[:completed]]):
            raise ValueError("layout checkpoint is not an exact completed prefix")
        hotspot, joint, records = dict(restored["hotspot"]), dict(restored["joint"]), list(restored["records"])
    for ordinal, bin_id in enumerate(bins):
        if ordinal < completed:
            continue
        guard()
        now = bin_id*cfg["layout_planning_bin_s"]
        sites = sites_for(reference, now, cfg)
        av = [p for p in plans if p.vehicle_type == "AV" and int(p.activation_s // cfg["layout_planning_bin_s"]) == bin_id]
        av.sort(key=lambda p: _rank(p.vehicle_id))
        shares = np.asarray([s["demand_share"] for s in sites], float)
        quota = shares / shares.sum() * len(av)
        counts = np.floor(quota).astype(int)
        order = sorted(range(len(sites)), key=lambda i: (-(quota[i]-counts[i]), sites[i]["site_id"]))
        for i in order[:len(av)-int(counts.sum())]:
            counts[i] += 1
        assignments = [site for site, count in zip(sites, counts) for _ in range(int(count))]
        for p, site in zip(av, assignments):
            hotspot[str(p.vehicle_id)] = site
        chosen_ids = {p.native_id for p in av}
        resources = [p for p in plans if p.admission_end_s > now
            and p.activation_s < now+cfg["planning_horizon_s"]]
        actions, states = [], {}
        for p in resources:
            if p.native_id in chosen_ids:
                choices = sites
            else:
                point = joint.get(p.vehicle_id, {}).get("position", (p.initial_lon_wgs84, p.initial_lat_wgs84))
                choices = [dict(site_id="FIXED", position=point, context=None)]
            for site in choices:
                aid = f"V{p.native_id}:LAYOUT:{site['site_id']}"
                actions.append(CurrentAction(aid, p.native_id, "LAYOUT", None, payload=site))
                states[aid] = RouteState(site["position"], max(float(now), p.activation_s), p.admission_end_s,
                    p.profile_id, None)
        result = build_and_solve(actions, states, library.view(now), library.weights,
            connectors, sites, cfg, now, "CHAIN_DEFER", offline_layout=True,
            diagnostics_sink=diagnostics_sink)
        picked = set(result["selected_action_ids"])
        for action in actions:
            if action.action_id in picked and action.vehicle_id in chosen_ids:
                p = next(p for p in av if p.native_id == action.vehicle_id)
                joint[str(p.vehicle_id)] = action.payload
        if not all(p.vehicle_id in joint for p in av):
            raise RuntimeError("an AV initial location was omitted")
        records.append(dict(bin_s=now, newly_placed_AV_slots=len(av), expected_services=result["expected_served"],
            model=compact_master(result), graph=result["restricted_graph"],
            graph_cpu_s=result["graph_or_time_s"], master_or_s=result["runtime_s"],
            total_or_time_s=result["total_or_time_s"]))
        if checkpoint is not None:
            checkpoint(dict(hotspot=hotspot, joint=joint, records=records,
                completed_bins=ordinal+1, total_bins=len(bins),
                last_completed_bin_s=now))
        progress(ordinal+1, len(bins), len(joint))
    return dict(hotspot=hotspot, joint=joint, records=records,
        layout_reference_supply_states="PLANNED_TEMPLATE_PREPARATION_POSITIONS_NOT_OBSERVED_MIDDAY_LOCATIONS",
        future_target_requests_or_outcomes_used=False)


def apply_av_layout(episode, placements):
    before = episode.supply.plan()
    after = tuple(replace(p, initial_lon_wgs84=float(placements[p.vehicle_id]["position"][0]),
        initial_lat_wgs84=float(placements[p.vehicle_id]["position"][1])) if p.vehicle_type == "AV" else p for p in before)
    identity = lambda p: (p.vehicle_id, p.native_id, p.slot_id, p.vehicle_type, p.profile_id,
        p.activation_s, p.admission_end_s, p.availability_policy)
    if [identity(p) for p in before] != [identity(p) for p in after]:
        raise RuntimeError("layout changed supply labels or admission windows")
    episode.supply._plan = after
    episode.supply._by_vehicle = {p.vehicle_id: p for p in after}
    episode.supply._accounting["av_initial_position_rule"] = "SCHEME_A_FROZEN_EARLIER_HISTORY_COMMON_SITE_LAYOUT"
    return len([p for p in after if p.vehicle_type == "AV"])
