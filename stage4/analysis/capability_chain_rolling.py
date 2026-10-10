"""Independent finite rolling prototype, not a native FleetPy or daily run.

Only revealed replay requests enter decisions. Two earlier historical cohorts
provide forecast scenarios; only their common first action is executed. The
same historical M3/auto transfer proxy, initial layout, and movement rules are
shared by both policies. No actual empty-trip timing labels are claimed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa

from stage4.analysis.capability_chain_instances import (
    IDENTITY_COLUMNS, S2A, S3, TEMPLATES, InstanceRouter, JoinEvidence,
    HistoricalSnapshotETA, _gap_m, _read_columns, _select,
    atomic_json, atomic_parquet, choose_tasks, grid_keys, sha,
)

CONFIG = Path("stage4/config/capability_chain_rolling_v1.json")
OUT = Path("stage4/output/capability_chain_planning/rolling_v1")
DOC = Path("stage4/docs/capability_chain_planning/rolling_v1")


def priority(value, seed):
    return hashlib.sha256(f"{value}|{seed}".encode()).hexdigest()


def history_sites(history, cfg):
    """Common, all-demand candidates from earlier dates only; no C prefilter."""
    frame = history.copy()
    frame["cell"] = grid_keys(frame, "start_lon_wgs84", "start_lat_wgs84", cfg)
    cells = frame.groupby("cell").size().reset_index(name="count").sort_values(
        ["count", "cell"], ascending=[False, True], kind="stable")
    sites = []
    for i, row in enumerate(cells.head(cfg["candidate_site_count"]).itertuples(index=False)):
        group = frame.loc[frame.cell.eq(row.cell)].copy()
        x = group.start_lon_wgs84.to_numpy(float) * math.cos(math.radians(34.25))
        y = group.start_lat_wgs84.to_numpy(float)
        group["center_gap"] = (x - x.mean())**2 + (y - y.mean())**2
        anchor = group.sort_values(["center_gap", "order_id"], kind="stable").iloc[0]
        sites.append(dict(site_id=f"S{i+1:02d}", cell=str(row.cell), historical_pickups=int(row.count),
            lon_wgs84=float(anchor.start_lon_wgs84), lat_wgs84=float(anchor.start_lat_wgs84)))
    return sites


class HistoricalPredictionTimeProxy(HistoricalSnapshotETA):
    """Prediction-to-routing transfer, NOT observed pickup calibration."""
    def __init__(self, root):
        super().__init__(root, 1.0)
        self.factors = {}

    def beta_for(self, timestamp):
        local = pd.Timestamp(timestamp)
        second = local.hour * 3600 + local.minute * 60 + local.second
        cut = second // 1800 * 1800
        if cut not in self.factors:
            raise ValueError("no earlier-history time proxy for this window")
        return int((local.hour * 60 + local.minute) // 15), self.factors[cut]


def fit_time_proxy(root, history, cfg, adapter):
    rows, summaries = [], []
    for cut in cfg["window_starts_s"]:
        for date in cfg["history_dates"]:
            group = history.loc[history.date.astype(str).eq(date) & history.release_second.ge(cut)
                & history.release_second.lt(cut + cfg["horizon_s"])].copy()
            group["priority"] = group.order_id.map(lambda o: priority(f"ETA|{date}|{o}", cfg["selection_seed"]))
            group = group.sort_values(["priority", "order_id"], kind="stable").head(cfg["proxy_samples_per_history_date_window"])
            for row in group.itertuples(index=False):
                clock = pd.Timestamp(date, tz="Asia/Shanghai") + pd.Timedelta(seconds=row.release_second)
                request = dict(locations=[dict(lon=row.start_lon_wgs84, lat=row.start_lat_wgs84, type="break"),
                    dict(lon=row.end_lon_wgs84, lat=row.end_lat_wgs84, type="break")],
                    costing="auto", units="kilometers", directions_type="none",
                    date_time=dict(type=1, value=clock.strftime("%Y-%m-%dT%H:%M")))
                trip = adapter.actor.route(request)["trip"]
                raw = float(trip["summary"]["time"])
                predicted = float(row.predicted_route_time_p50_s)
                if raw <= 0 or predicted <= 0 or not math.isfinite(raw + predicted):
                    raise ValueError("invalid earlier-history prediction/auto proxy sample")
                rows.append(dict(source_date=date, source_order_id=row.order_id, window_start_s=cut,
                    raw_auto_time_s=raw, frozen_M3_customer_p50_s=predicted, ratio=predicted/raw))
        ratios = [r["ratio"] for r in rows if r["window_start_s"] == cut]
        factor = float(np.median(ratios))
        adapter.factors[cut] = factor
        summaries.append(dict(window_start_s=cut, sample_count=len(ratios), factor=factor,
            ratio_p10=float(np.quantile(ratios, .1)), ratio_p90=float(np.quantile(ratios, .9))))
    atomic_parquet(root / OUT / "historical_prediction_auto_proxy_samples.parquet", pd.DataFrame(rows))
    result = dict(kind="HISTORICAL_PREDICTION_TO_AUTO_TIME_TRANSFER_PROXY_NOT_OBSERVED_EMPTY_TIME",
        history_dates=list(cfg["history_dates"]), replay_date_used=False, Test31_used=False,
        factors=summaries, source_route_may_differ_from_auto_route=True,
        transfer_to_empty_route_is_model_assumption=True, new_M3_inference=False, retrained=False)
    atomic_json(root / DOC / "time_proxy.json", result)
    return result


class MixedDateEvidence(JoinEvidence):
    """Earlier forecast routes and revealed replay routes keep distinct identities."""
    def __init__(self, root, router, universe, cfg):
        self.router = router
        self.tokens = {}
        for date, group in universe.groupby("date", sort=False):
            path = root / S3 / f"cache/train/date={date}/identity.parquet"
            table = _select(_read_columns(path, IDENTITY_COLUMNS), "order_id", group.order_id)
            by_order = {str(o): g for o, g in table.to_pandas().groupby("order_id", sort=False)}
            for row in group.itertuples(index=False):
                if str(row.order_id) not in by_order:
                    raise ValueError("historical/replay customer identity missing")
                tokens = by_order[str(row.order_id)].sort_values("route_sequence").copy()
                # Parser grouping time is the hypothetical replay date. Original
                # source date is retained in the private universe, never changed.
                tokens["date"] = cfg["replay_date"]
                tokens["order_id"] = row.task_id
                self.tokens[row.task_id] = tokens
        reverse = {str(row.canonical_edge_uid) for frame in self.tokens.values()
            for row in frame.loc[frame.route_token_type.eq("HISTORICAL_REVERSE_OVERLAY")].itertuples(index=False)}
        overlay = _read_columns(root / S2A / "stage3_historical_direction_overlay.parquet",
            ["canonical_edge_uid", "physical_forward_stage3_edge_uid"])
        self.overlay = _select(overlay, "canonical_edge_uid", reverse).to_pandas()
        self.overlay_forward = dict(zip(self.overlay.canonical_edge_uid, self.overlay.physical_forward_stage3_edge_uid))
        self.selected = universe.set_index("task_id")
        self.simulation_date = cfg["replay_date"]

    def add_context(self, context_id, edges, point):
        self.tokens[context_id] = pd.DataFrame([dict(date=self.simulation_date, order_id=context_id,
            route_sequence=i, canonical_edge_uid=None, route_token_type="FULL_NETWORK_EDGE",
            resolved_stage3_edge_uid=e["stage3_edge_uid"]) for i, e in enumerate(edges)], columns=IDENTITY_COLUMNS)
        self.selected.loc[context_id, ["start_lon_wgs84", "start_lat_wgs84", "end_lon_wgs84", "end_lat_wgs84"]] = [*point, *point]

    def remove_context(self, context_id):
        del self.tokens[context_id]
        self.selected.drop(index=context_id, inplace=True)


class ConnectionProvider:
    """Sparse lazy physical cache; no list of unrevealed actual jobs is given to policy."""
    def __init__(self, router, evidence, universe, sites, cfg, cut):
        self.router, self.evidence, self.cfg = router, evidence, cfg
        self.clock = pd.Timestamp(cfg["replay_date"], tz="Asia/Shanghai") + pd.Timedelta(seconds=cut)
        self.rows = universe.set_index("task_id").to_dict("index")
        self.sites = {s["site_id"]: s for s in sites}
        self.locations = {task: (float(row["end_lon_wgs84"]), float(row["end_lat_wgs84"])) for task, row in self.rows.items()}
        self.locations.update({s["site_id"]: (s["lon_wgs84"], s["lat_wgs84"]) for s in sites})
        self.cache, self.moves = {}, {}
        self.hits = 0

    def _evaluate(self, origin, destination, target_key=None):
        routed = self.router.route(*self.locations[origin], *destination, self.clock)
        base = dict(supported=False, travel_time_s=routed["duration_s"], empty_distance_m=routed["distance_m"],
            compatible_profiles=[], reason_codes=list(routed["reason_codes"]), evidence_kind="MODELED_DIRECTED_CONNECTION")
        if base["empty_distance_m"] is not None:
            # The engine reports kilometre length; remove binary ×1000 dust
            # before exact lexicographic comparisons (one-micrometre precision).
            base["empty_distance_m"] = round(float(base["empty_distance_m"]), 6)
        if not routed["common_supported"]:
            return base, routed
        if max(routed["snap_gap_origin_m"], routed["snap_gap_target_m"]) > self.cfg["interface_snap_tolerance_m"]:
            base["reason_codes"] = ["SNAP_GAP_EXCEEDS_COMMON_INSTANCE_TOLERANCE"]
            return base, routed
        # Independent relocation ends at a stop, not a fabricated next customer
        # or duplicate outgoing edge. Its arrival direction is kept as context
        # and checked against the next real connector after this stop.
        join = self.evidence.evaluate(origin if origin in self.evidence.tokens else None,
            target_key, routed, self.cfg["interface_snap_tolerance_m"])
        if not join["supported"]:
            base["reason_codes"] = join["reason_codes"]
            return base, routed
        allowed = sorted(set(routed["compatible_profiles"]) & set(join["compatible_profiles"]) & {"HV", "C"})
        base.update(supported=True, compatible_profiles=allowed, reason_codes=[],
            C_reason_codes=sorted(set(routed["profile_reason_codes"].get("C", ())) | set(join["C_reason_codes"])),
            join_encounter_count=join["join_encounter_count"])
        return base, routed

    def connection(self, origin, target, allowed_targets):
        if target not in allowed_targets:
            raise ValueError("policy queried an unrevealed replay request")
        key = origin, target
        if key not in self.cache:
            row = self.rows[target]
            value, _ = self._evaluate(origin, (row["start_lon_wgs84"], row["start_lat_wgs84"]), target)
            self.cache[key] = dict(origin_id=origin, target_job_id=target, **value)
        else:
            self.hits += 1
        return self.cache[key]

    def relocation(self, origin, site_id):
        key = origin, site_id
        if key not in self.moves:
            point = self.locations[site_id]
            value, routed = self._evaluate(origin, point)
            context = "R:" + priority(origin + "|" + site_id, 0)[:16]
            if value["supported"] and routed["edges"]:
                self.evidence.add_context(context, routed["edges"], point)
                self.locations[context] = point
            else:
                context = site_id
            move = dict(value)
            move.update(arrival_location_id=context, site_id=site_id,
                origin_id=origin, evidence_kind="INDEPENDENT_IDLE_RELOCATION_NOT_PICKUP")
            self.moves[key] = move
        return self.moves[key]


def task_record(row, cut, cfg, actual):
    release = float(row.release_second - cut)
    return dict(job_id=row.task_id, release_s=release, deadline_s=release + cfg["patience_s"],
        service_time_s=float(row.predicted_route_time_p50_s),
        compatible_profiles=[k for k in ("HV", "C") if bool(getattr(row, f"compatible_{k}"))],
        source_kind="ACTUAL_PENDING" if actual else "FORECAST")


def visible_tasks(tasks, now, committed):
    """Never return not-yet-released or already-promised replay orders."""
    return [t for t in tasks if t["release_s"] <= now <= t["deadline_s"] and t["job_id"] not in committed]


def forecast_view(cohorts, now, cfg):
    # Common, fixed earlier cohorts. Removing already-due hypothetical requests
    # is not making them real pending demand. No replay-future fallback exists.
    return [dict(scenario_id=s["scenario_id"], weight=s["weight"],
        tasks=[t for t in s["tasks"] if t["release_s"] > now
            and t["release_s"] < cfg["horizon_s"]]) for s in cohorts]


class HistoricalForecastReference:
    """Five-minute reference refresh; fine filtering never turns forecasts real."""
    def __init__(self, cohorts, cfg):
        self.cohorts, self.cfg = cohorts, cfg
        self.last_update_s, self.snapshot, self.update_count = None, None, 0

    def view(self, now):
        coarse = now // self.cfg["coarse_reference_update_s"] * self.cfg["coarse_reference_update_s"]
        if coarse != self.last_update_s:
            self.snapshot = forecast_view(self.cohorts, coarse, self.cfg)
            self.last_update_s = coarse
            self.update_count += 1
        return forecast_view(self.snapshot, now, self.cfg)


def first_actions(resources, pending, provider, now, cfg):
    actions = []
    allowed = {t["job_id"] for t in pending}
    for resource in resources:
        rid, origin, profile = resource["resource_id"], resource["location_id"], resource["profile_id"]
        common = dict(resource_id=rid, job_id=None, empty_distance_m=0.0, critical=False, carry_over=False)
        if resource["ready_s"] > now:
            actions.append(dict(**common, action_id=rid+":BUSY", kind="BUSY", ready_s=resource["ready_s"], location_id=origin))
            continue
        actions.append(dict(**common, action_id=rid+":WAIT", kind="WAIT", ready_s=now+cfg["step_s"], location_id=origin))
        if now >= resource["admission_end_s"]:
            continue
        for task in pending:
            if profile not in task["compatible_profiles"]:
                continue
            link = provider.connection(origin, task["job_id"], allowed)
            if not link["supported"] or profile not in link["compatible_profiles"]:
                continue
            pickup = now + link["travel_time_s"]
            if pickup > task["deadline_s"]:
                continue
            actions.append(dict(resource_id=rid, action_id=rid+":SERVE:"+task["job_id"], kind="SERVE",
                job_id=task["job_id"], location_id=task["job_id"], origin_id=origin,
                commit_s=now, pickup_s=pickup, ready_s=pickup+task["service_time_s"],
                travel_time_s=link["travel_time_s"], empty_distance_m=link["empty_distance_m"],
                critical=0 < task["deadline_s"]-now <= cfg["critical_slack_s"],
                carry_over=task["release_s"] < now-cfg["step_s"]))
        if now-resource["last_relocation_s"] < cfg["reposition_interval_s"] or resource["relocation_count"] >= cfg["reposition_max_moves"]:
            continue
        nearby = [(s, _gap_m(provider.locations[origin], provider.locations[s])) for s in provider.sites]
        nearby = [(s, d) for s, d in nearby if 1 < d <= cfg["reposition_radius_m"]]
        nearby.sort(key=lambda pair: (pair[1], pair[0]))
        for site, _ in nearby[:cfg["reposition_top_k"]]:
            link = provider.relocation(origin, site)
            if (not link["supported"] or profile not in link["compatible_profiles"]
                or link["travel_time_s"] > cfg["reposition_max_eta_s"]
                or now+link["travel_time_s"] >= resource["admission_end_s"]):
                continue
            actions.append(dict(resource_id=rid, action_id=rid+":RELOCATE:"+site, kind="RELOCATE",
                job_id=None, location_id=link["arrival_location_id"], origin_id=origin, target_site_id=site,
                ready_s=now+link["travel_time_s"], empty_distance_m=link["empty_distance_m"],
                travel_time_s=link["travel_time_s"], critical=False, carry_over=False))
    return actions


def epoch_problem(resources, pending, forecasts, actions, provider, now, cfg):
    scenarios = []
    resource_profiles = {r["resource_id"]:r["profile_id"] for r in resources}
    for forecast in forecasts:
        tasks = [*pending, *forecast["tasks"]]
        allowed = {t["job_id"] for t in tasks}
        readiness = {}
        origin_profiles = {}
        for action in actions:
            loc = action["location_id"]
            readiness[loc] = min(action["ready_s"], readiness.get(loc, float("inf")))
            origin_profiles.setdefault(loc, set()).add(resource_profiles[action["resource_id"]])
        for task in tasks:
            # Optimistic zero-pickup lower bound is safe for routing pruning.
            value = max(now, task["release_s"]) + task["service_time_s"]
            readiness[task["job_id"]] = min(value, readiness.get(task["job_id"], float("inf")))
            origin_profiles.setdefault(task["job_id"], set()).update(task["compatible_profiles"])
        connections = []
        for origin, ready in readiness.items():
            if cfg.get("skip_planned_post_service_geometry_queries", False) and origin in allowed:
                continue  # A valuation projection supplies this layer, never actual first actions.
            for task in tasks:
                if origin == task["job_id"] or ready > task["deadline_s"]:
                    continue
                if cfg.get("prune_connection_profiles", False) and not origin_profiles[origin].intersection(task["compatible_profiles"]):
                    continue
                link = provider.connection(origin, task["job_id"], allowed)
                # The constant routing snapshot is unchanged across model builds.
                if link["supported"] and max(ready, task["release_s"]) + link["travel_time_s"] <= task["deadline_s"]:
                    connections.append(link)
        scenarios.append(dict(scenario_id=forecast["scenario_id"], weight=forecast["weight"],
            tasks=tasks, connections=connections))
    problem = dict(now_s=now, step_s=cfg["step_s"],
        resources=[{k:r[k] for k in ("resource_id", "profile_id", "admission_end_s")} for r in resources],
        actions=actions, scenarios=scenarios, solver_time_limit_s=cfg["solver_decision_limit_s"])
    if "task_limit_per_scenario" in cfg:
        problem["task_limit_per_scenario"] = cfg["task_limit_per_scenario"]
    if "exact_chain_compression" in cfg:
        problem["exact_chain_compression"] = cfg["exact_chain_compression"]
    return problem


def shared_initial_layout(provider, cohorts, sites, ranking, cfg):
    from stage4.dispatch.capability_chain_rolling_solver import solve_epoch
    resources = [dict(resource_id="H1", profile_id="HV", admission_end_s=cfg["horizon_s"]),
        dict(resource_id="H2", profile_id="HV", admission_end_s=cfg["horizon_s"]),
        dict(resource_id="C1", profile_id="C", admission_end_s=cfg["horizon_s"])]
    actions = []
    for resource, fixed in zip(resources, [ranking[0], ranking[1], None]):
        choices = [fixed] if fixed else [s["site_id"] for s in sites]
        for site in choices:
            actions.append(dict(resource_id=resource["resource_id"], action_id=resource["resource_id"]+":LAYOUT:"+site,
                kind="LAYOUT", ready_s=0.0, location_id=site, job_id=None,
                empty_distance_m=0.0, critical=False, carry_over=False))
    problem = epoch_problem(resources, [], forecast_view(cohorts, 0, cfg), actions, provider, 0, cfg)
    solution = solve_epoch(problem, "CHAIN_DEFER")
    initial = {a["resource_id"]: a["location_id"] for a in solution["selected_actions"]}
    atomic = [dict(**r, location_id=initial[r["resource_id"]], ready_s=0.0,
        last_relocation_s=-cfg["reposition_interval_s"], relocation_count=0) for r in resources]
    return atomic, solution


def replay_policy(actual_tasks, cohorts, initial, provider, policy, cfg, budget_check,
                  *, problem_transform=None, report_policy=None):
    from stage4.dispatch.capability_chain_rolling_solver import solve_epoch, validate_epoch_solution
    states = [dict(r) for r in initial]
    forecast_reference = HistoricalForecastReference(cohorts, cfg)
    committed, events, epochs = {}, [], []
    report_policy = report_policy or policy
    start = perf_counter()
    for now in range(0, cfg["horizon_s"], cfg["step_s"]):
        budget_check()
        pending = visible_tasks(actual_tasks, now, committed)
        action_started = perf_counter()
        actions = first_actions(states, pending, provider, now, cfg)
        problem = epoch_problem(states, pending, forecast_reference.view(now), actions, provider, now, cfg)
        if problem_transform is not None:
            problem = problem_transform(problem)
        build_s = perf_counter()-action_started
        solution = solve_epoch(problem, policy)
        validate_epoch_solution(problem, solution)
        selected = solution["selected_actions"]
        for action in selected:
            resource = next(r for r in states if r["resource_id"] == action["resource_id"])
            if action["kind"] in ("WAIT", "BUSY"):
                continue
            origin = resource["location_id"]
            if resource["ready_s"] > now or now >= resource["admission_end_s"]:
                raise RuntimeError("executed action violates physical availability")
            resource["location_id"] = action["location_id"]
            resource["ready_s"] = action["ready_s"]
            if action["kind"] == "SERVE":
                job = action["job_id"]
                task = next(t for t in pending if t["job_id"] == job)
                if job in committed or task["release_s"] > now or action["pickup_s"] > task["deadline_s"]:
                    raise RuntimeError("executed customer action violates release/uniqueness/deadline")
                committed[job] = dict(**action, release_s=task["release_s"], deadline_s=task["deadline_s"])
            elif action["kind"] == "RELOCATE":
                resource["last_relocation_s"] = now
                resource["relocation_count"] += 1
            else:
                raise RuntimeError("layout or hypothetical future action leaked into execution")
            events.append(dict(policy=report_policy, epoch_s=now, resource_id=resource["resource_id"],
                profile_id=resource["profile_id"], kind=action["kind"], job_id=action["job_id"],
                origin_id=origin, destination_id=resource["location_id"], empty_distance_m=action["empty_distance_m"],
                empty_time_s=action["travel_time_s"], pickup_s=action.get("pickup_s"), finish_s=resource["ready_s"],
                service_time_s=task["service_time_s"] if action["kind"] == "SERVE" else 0.0,
                release_s=committed[action["job_id"]]["release_s"] if action["kind"] == "SERVE" else None))
        metrics = {k:v for k,v in solution.items() if k not in ("recourse_paths", "selected_actions")}
        if problem_transform is not None:
            metrics["valuation_model"] = problem.get("valuation_model")
        epochs.append(dict(epoch_s=now, pending_count=len(pending), build_and_routing_s=build_s, **metrics))
        if now % 300 == 0:
            print(json.dumps(dict(policy=report_policy, epoch_s=now, committed=len(committed),
                actual_waiting=len(pending), rss_mib=round(psutil.Process().memory_info().rss/2**20, 2))), flush=True)
    waits = [a["pickup_s"]-a["release_s"] for a in committed.values()]
    customer = [e for e in events if e["kind"] == "SERVE"]
    moves = [e for e in events if e["kind"] == "RELOCATE"]
    duplicate_count = len(customer) - len({e["job_id"] for e in customer})
    errors = [abs(e["finish_s"]-e["epoch_s"]-e["empty_time_s"]-e["service_time_s"]) for e in events]
    overlaps = 0
    for rid in {e["resource_id"] for e in events}:
        ordered = sorted([e for e in events if e["resource_id"] == rid], key=lambda e:e["epoch_s"])
        overlaps += sum(b["epoch_s"] < a["finish_s"]-1e-8 for a,b in zip(ordered, ordered[1:]))
    if duplicate_count or overlaps or max(errors, default=0) > 1e-8:
        raise RuntimeError("executed finite resource ledger failed conservation/uniqueness")
    summary = dict(policy=report_policy, actual_cohort_count=len(actual_tasks), realized_committed=len(committed),
        realized_completed_after_drain=len(committed), unserved_count=len(actual_tasks)-len(committed),
        served_by_profile=dict(Counter(e["profile_id"] for e in customer)),
        customer_pickup_empty_distance_m=sum(e["empty_distance_m"] for e in customer),
        independent_idle_moves=len(moves), idle_move_distance_m=sum(e["empty_distance_m"] for e in moves),
        mean_pickup_wait_s=float(np.mean(waits)) if waits else None,
        waiting_beyond_first_epoch_count=sum(a["commit_s"] > math.ceil(a["release_s"] / cfg["step_s"])*cfg["step_s"] for a in committed.values()),
        cross_window_commitment_completion_count=sum(a["ready_s"] > cfg["horizon_s"] for a in committed.values()),
        physical_last_completion_s=max((r["ready_s"] for r in states), default=0),
        no_predicted_service_counted_as_realized=all(e["job_id"] in {t["job_id"] for t in actual_tasks} for e in customer),
        duplicate_realized_customer_count=duplicate_count, overlapping_resource_action_count=overlaps,
        maximum_time_conservation_error_s=max(errors, default=0), forecast_reference_updates=forecast_reference.update_count,
        runtime_s=perf_counter()-start, committed_jobs=committed, executed_events=events, epochs=epochs)
    return summary


def compact_policy(result):
    summary = {k:v for k,v in result.items() if k not in ("executed_events", "epochs", "committed_jobs")}
    summary["served_anonymous_jobs"] = sorted(result["committed_jobs"])
    summary["maximum_model_variables"] = max((e.get("model", {}).get("columns", 0) for e in result["epochs"]), default=0)
    summary["maximum_model_nonzeros"] = max((e.get("model", {}).get("nonzeros", 0) for e in result["epochs"]), default=0)
    summary["model_solver_total_s"] = sum(e.get("timing_s", {}).get("total", 0) for e in result["epochs"])
    summary["build_and_routing_total_s"] = sum(e["build_and_routing_s"] for e in result["epochs"])
    decisions = [e["timing_s"]["total"] for e in result["epochs"]]
    summary["decision_time_p50_s"] = float(np.quantile(decisions, .5))
    summary["decision_time_p95_s"] = float(np.quantile(decisions, .95))
    summary["maximum_decision_time_s"] = max(decisions, default=0)
    return summary


def summarize_existing(root, result):
    """Post-run diagnostics from saved files only; no new routing or simulation."""
    for case in result["cases"]:
        destination = root / case["private_output"]
        inputs = json.loads((destination / "input.json").read_text(encoding="utf-8"))
        layout = json.loads((destination / "historical_layout_solution.json").read_text(encoding="utf-8"))
        sites = inputs["sites"]
        pairs = [dict(left_site=a["site_id"], right_site=b["site_id"],
            distance_m=_gap_m((a["lon_wgs84"], a["lat_wgs84"]), (b["lon_wgs84"], b["lat_wgs84"])))
            for i,a in enumerate(sites) for b in sites[i+1:]]
        case["shared_candidate_geometry"] = dict(site_pair_count=len(pairs),
            minimum_site_separation_m=min((p["distance_m"] for p in pairs), default=None),
            pairs_within_existing_2km_move_radius=sum(p["distance_m"] <= result["config"]["reposition_radius_m"] for p in pairs),
            comparison_cannot_identify_idle_relocation_effect=not any(p["distance_m"] <= result["config"]["reposition_radius_m"] for p in pairs))
        case["historical_initial_layout"] = dict(expected_scenario_services=layout["expected_served"],
            model_columns=layout["model"]["columns"], model_nonzeros=layout["model"]["nonzeros"],
            runtime_s=layout["runtime_s"], no_actual_replay_request_input=True)
        full = {}
        case["policies"] = []
        for policy in result["config"]["policies"]:
            full[policy] = json.loads((destination / f"{policy}_full_result.json").read_text(encoding="utf-8"))
            case["policies"].append(compact_policy(full[policy]))
        before = [{k:v for k,v in e.items() if k != "policy"} for e in full["SERVICE_PRESERVING"]["executed_events"]]
        after = [{k:v for k,v in e.items() if k != "policy"} for e in full["CHAIN_DEFER"]["executed_events"]]
        case["comparison"]["executed_sequences_identical"] = before == after
        case["scientific_classification"] = "CONTROL_CONTRAST_DEGENERATE_NO_EFFECTIVENESS_INFERENCE"
    result["calibration_scalar_route_queries"] = sum(f["sample_count"] for f in result["time_proxy"]["factors"])
    result["total_scalar_route_queries_including_time_proxy"] = result["routing"]["scalar_route_calls"] + result["calibration_scalar_route_queries"]
    result["maximum_model_variables_including_layout"] = max(
        max([case["historical_initial_layout"]["model_columns"], *[p["maximum_model_variables"] for p in case["policies"]]])
        for case in result["cases"])
    result["maximum_model_nonzeros_including_layout"] = max(
        max([case["historical_initial_layout"]["model_nonzeros"], *[p["maximum_model_nonzeros"] for p in case["policies"]]])
        for case in result["cases"])
    atomic_json(root / OUT / "run_summary.json", result)
    atomic_json(root / DOC / "summary.json", result)
    return result


def run(root, config_path=CONFIG):
    started = perf_counter()
    cfg = json.loads((root / config_path).read_text(encoding="utf-8"))
    if any(d >= cfg["replay_date"] for d in cfg["history_dates"]) or cfg["replay_date"] > "20161024":
        raise ValueError("prototype must use strictly earlier historical dates and no Test31")
    if cfg["full_day_or_native"] or cfg["parameter_search"] or cfg["model_retraining"]:
        raise ValueError("this entry point only authorizes the finite independent prototype")
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    templates = pd.read_parquet(root / TEMPLATES)
    templates["date"] = templates.date.astype(str)
    templates["order_id"] = templates.order_id.astype(str)
    history = templates.loc[templates.date.isin(cfg["history_dates"])].copy()
    sites = history_sites(history, cfg)
    adapter = HistoricalPredictionTimeProxy(root)
    proxy = fit_time_proxy(root, history, cfg, adapter)
    router = InstanceRouter(root, adapter)
    process = psutil.Process()

    def budget_check():
        if perf_counter()-started > cfg["maximum_runtime_s"]:
            raise RuntimeError("finite rolling prototype exceeded declared 900-second limit")
        if process.memory_info().rss / 2**20 > cfg["maximum_rss_mib"]:
            raise RuntimeError("finite rolling prototype exceeded declared RSS bound")
        proxy_queries = sum(f["sample_count"] for f in proxy["factors"])
        if router.route_request_count + proxy_queries > cfg["maximum_route_queries"]:
            raise RuntimeError("finite rolling prototype exceeded route query bound")

    result = dict(status="RUNNING", kind="INDEPENDENT_LIMITED_ROLLING_PROTOTYPE_NOT_NATIVE_DAILY_EVALUATION",
        config=cfg, config_sha256=sha(root / config_path), templates_sha256=sha(root / TEMPLATES),
        time_proxy=proxy, history_dates=cfg["history_dates"], replay_date=cfg["replay_date"],
        Test31_read=False, frozen_baseline_modified=False, cases=[])
    atomic_json(root / OUT / "run_summary.json", result)
    for cut in cfg["window_starts_s"]:
        case_id = f"REPLAY_{cfg['replay_date']}_{cut//3600:02d}{cut%3600//60:02d}"
        cohorts, parts, samples = [], [], {}
        for date in [cfg["replay_date"], *cfg["history_dates"]]:
            selection_cfg = dict(cfg, historical_date=date)
            chosen, sample = choose_tasks(templates, sites, selection_cfg, cut)
            chosen["task_id"] = [("A" if date == cfg["replay_date"] else f"F{date[-2:]}_") + f"{i+1:02d}" for i in range(len(chosen))]
            records = [task_record(row, cut, cfg, date == cfg["replay_date"]) for row in chosen.itertuples(index=False)]
            samples[date] = sample
            if date == cfg["replay_date"]:
                actual_tasks = records
            else:
                cohorts.append(dict(scenario_id=date, weight=1/len(cfg["history_dates"]), tasks=records))
            parts.append(chosen)
        universe = pd.concat(parts, ignore_index=True)
        evidence = MixedDateEvidence(root, router, universe, cfg)
        provider = ConnectionProvider(router, evidence, universe, sites, cfg, cut)
        history_cells = history.assign(cell=grid_keys(history, "start_lon_wgs84", "start_lat_wgs84", cfg))
        ranking = sorted([s["site_id"] for s in sites], key=lambda sid: (-len(history_cells.loc[
            history_cells.cell.eq(next(s["cell"] for s in sites if s["site_id"] == sid))
            & history_cells.release_second.ge(cut) & history_cells.release_second.lt(cut+cfg["horizon_s"])]), sid))
        initial, layout = shared_initial_layout(provider, cohorts, sites, ranking, cfg)
        destination = root / OUT / case_id
        atomic_json(destination / "input.json", dict(actual_tasks=actual_tasks, forecast_cohorts=cohorts,
            sites=sites, shared_initial_resources=initial, config=cfg))
        atomic_json(destination / "historical_layout_solution.json", layout)
        atomic_parquet(destination / "private_source_mapping.parquet", universe)
        policies = []
        private_results = {}
        for policy in cfg["policies"]:
            replay = replay_policy(actual_tasks, cohorts, initial, provider, policy, cfg, budget_check)
            private_results[policy] = replay
            atomic_json(destination / f"{policy}_full_result.json", replay)
            atomic_parquet(destination / f"{policy}_events.parquet", pd.DataFrame(replay["executed_events"]))
            policies.append(compact_policy(replay))
        before = private_results["SERVICE_PRESERVING"]["committed_jobs"]
        after = private_results["CHAIN_DEFER"]["committed_jobs"]
        shared = sorted(set(before) & set(after))
        comparison = dict(realized_service_gain=len(after)-len(before),
            gained_jobs=sorted(set(after)-set(before)), lost_jobs=sorted(set(before)-set(after)),
            common_served_count=len(shared), paired_mean_pickup_wait_change_s=float(np.mean([
                after[j]["pickup_s"]-before[j]["pickup_s"] for j in shared])) if shared else None)
        atomic_parquet(destination / "sparse_connections.parquet", pd.DataFrame(provider.cache.values()))
        atomic_parquet(destination / "idle_move_connections.parquet", pd.DataFrame(provider.moves.values()))
        result["cases"].append(dict(case_id=case_id, sample_counts=samples, policies=policies,
            comparison=comparison, shared_initial_layout={r["resource_id"]:r["location_id"] for r in initial},
            cached_sparse_connections=len(provider.cache), cached_idle_move_connections=len(provider.moves),
            shared_cache_hits=provider.hits, common_input_failure_reasons=dict(Counter(reason
                for link in provider.cache.values() if not link["supported"] for reason in link["reason_codes"])),
            private_output=str(destination.relative_to(root))))
        atomic_json(root / OUT / "run_summary.json", result)
        del provider, evidence, universe, private_results
        gc.collect()
        budget_check()
    result.update(status="FINITE_ROLLING_PROTOTYPE_COMPLETE", runtime_s=perf_counter()-started,
        peak_rss_mib=process.memory_info().peak_wset/2**20, routing=router.diagnostics(),
        gpu_used=False, native_runs=0, full_day_runs=0, training_runs=0,
        runtime_comparison_caveat="SHARED_LAZY_CACHE_ORDER_NOT_A_PAIRED_ROUTING_SPEED_BENCHMARK")
    summarize_existing(root, result)
    print(json.dumps(dict(status=result["status"], runtime_s=result["runtime_s"], peak_rss_mib=result["peak_rss_mib"])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--summarize-existing", action="store_true", help="aggregate saved outputs without routing or replay")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.summarize_existing:
        saved = json.loads((root / OUT / "run_summary.json").read_text(encoding="utf-8"))
        summarize_existing(root, saved)
        print(json.dumps(dict(status="EXISTING_ROLLING_RESULTS_AGGREGATED_NO_RERUN")), flush=True)
    else:
        run(root, args.config)


if __name__ == "__main__":
    main()
