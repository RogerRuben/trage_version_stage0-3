"""Exactly two serial Test31 native full-day comparisons with symmetric support."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

import pandas as pd
import psutil
import pyarrow as pa

from stage3.scripts.traffic_state_batch1 import sha, write_json, write_parquet
from stage4.analysis.symmetric_research_prepare import CONFIG, OUT as INPUT, DOC
from stage4.dispatch.deterministic_routing import ArcDeterministicValhallaAdapter
from stage4.dispatch.flexibility_native import NativeFlexibilityAdapter, ResearchRoutePolicy, TrainDemandForecast
from stage4.dispatch.remaining_time import TrainRemainingTime
from stage4.dispatch.fleet_normalization import build_fleet_scenario
from stage4.dispatch.rolling_or_control import create_rolling_or_fleet_control
from stage4.fleetpy_adapter.test31_demand_adapter import attach_fleetpy_requests, load_all_test31_requests
from stage4.fleetpy_adapter.upstream import load_fleetpy_bindings, CoordinateRegistry
from stage4.fleetpy_adapter.native_network import create_native_network
from stage4.fleetpy_adapter.native_demand import create_native_demand
from stage4.fleetpy_adapter.mixed_fleet_adapter import create_native_vehicles
from stage4.fleetpy_adapter.native_simulation import create_native_simulation

OUTPUT = Path("stage4/output/symmetric_flexibility_v1/full_day")


def administrative_timeout(cfg, policy, override):
    if override is None:
        return int(cfg["scenario_timeout_s"])
    if policy != "SERVICE_PRESERVING_LOOKAHEAD" or override != 21600:
        raise ValueError("only the user-authorized second-policy 6h administrative retry is allowed")
    return int(override)


def acceleration_settings(root, path):
    if path is None:
        return {}, None, None
    path = (root / path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("acceleration configuration must stay in the workspace")
    spec = json.loads(path.read_text())
    allowed = {"recourse_mode", "solver_backend", "highspy_runtime_dir", "fast_forecast",
               "persistent_cache_size", "route_workers", "lock_current_face"}
    version = spec.get("version")
    if version in ("symmetric_flexibility_acceleration_v2", "symmetric_flexibility_acceleration_v3", "symmetric_flexibility_acceleration_v4"):
        allowed |= {"cached_geometry", "event_calendar", "fast_future_graph", "epoch_routing_queue",
                    "pre_route_session_certificate", "compress_fixed_states", "trim_flow_rows", "offline_assets",
                    "disk_route_cache", "disk_cache_limit_mib", "disk_cache_max_entries", "route_queue_chunk_size", "lazy_worker_actor"}
    if version in ("symmetric_flexibility_acceleration_v3", "symmetric_flexibility_acceleration_v4"):
        allowed |= {"decompose_model", "demand_routing", "grouped_sources"}
    if version == "symmetric_flexibility_acceleration_v4":
        allowed |= {"static_raw_od", "raw_od_cache_size", "certified_eta_pruning"}
    if (version not in ("symmetric_flexibility_acceleration_v1", "symmetric_flexibility_acceleration_v2", "symmetric_flexibility_acceleration_v3", "symmetric_flexibility_acceleration_v4")
            or Path(spec.get("base_config", "")) != CONFIG
            or set(spec.get("settings", {})) != allowed
            or set(spec) != {"version", "base_config", "settings", "process_group_memory_limit_mib"}):
        raise ValueError("only the declared technical acceleration settings are allowed")
    settings = dict(spec["settings"])
    runtime = (root / settings["highspy_runtime_dir"]).resolve()
    if not runtime.is_relative_to(root.resolve()):
        raise ValueError("optional solver runtime must stay in the workspace")
    settings["highspy_runtime_dir"] = str(runtime)
    if (settings["recourse_mode"] != "FLOW_RELAXED" or settings["solver_backend"] not in ("SCIPY", "HIGHS_PERSISTENT")
            or type(settings["fast_forecast"]) is not bool
            or type(settings["lock_current_face"]) is not bool
            or not 0 <= settings["persistent_cache_size"] <= 50000
            or not 1 <= settings["route_workers"] <= 4
            or spec["process_group_memory_limit_mib"] != 2048):
        raise ValueError("acceleration resource or solver settings are invalid")
    if version in ("symmetric_flexibility_acceleration_v2", "symmetric_flexibility_acceleration_v3", "symmetric_flexibility_acceleration_v4"):
        for key in ("cached_geometry", "event_calendar", "fast_future_graph", "epoch_routing_queue",
                    "pre_route_session_certificate", "compress_fixed_states", "trim_flow_rows", "lazy_worker_actor"):
            if type(settings[key]) is not bool:
                raise ValueError("v2 acceleration switches must be explicit booleans")
        for key in ("offline_assets", "disk_route_cache"):
            target = (root / settings[key]).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError("offline acceleration products must stay in the workspace")
            settings[key] = str(target)
        if (settings["route_workers"] > 2 or not 16 <= settings["disk_cache_limit_mib"] <= 512
                or not 1 <= settings["disk_cache_max_entries"] <= 2_000_000
                or not 1 <= settings["route_queue_chunk_size"] <= 256):
            raise ValueError("v2 acceleration resource bound exceeded")
    if version in ("symmetric_flexibility_acceleration_v3", "symmetric_flexibility_acceleration_v4"):
        if any(type(settings[key]) is not bool for key in ("decompose_model", "demand_routing", "grouped_sources")):
            raise ValueError("v3 switches must be explicit booleans")
    if version == "symmetric_flexibility_acceleration_v4":
        if (any(type(settings[key]) is not bool for key in ("static_raw_od", "certified_eta_pruning"))
                or type(settings["raw_od_cache_size"]) is not int or not 0 <= settings["raw_od_cache_size"] <= 50000
                or not settings["demand_routing"] or not settings["epoch_routing_queue"]):
            raise ValueError("v4 raw OD/pruning settings are invalid")
    return settings, path, spec["process_group_memory_limit_mib"]


def analyze(root, output_root=OUTPUT, doc_root=DOC):
    cfg = json.loads((root / CONFIG).read_text())
    rows = []
    outcomes = []
    provenance = []
    for policy in cfg["policies"]:
        directory = root / output_root / policy
        reused = False
        if policy == "MYOPIC" and output_root != OUTPUT and not (directory / "summary.json").exists():
            directory = root / OUTPUT / policy
            reused = True
        if not (directory / "summary.json").exists(): return
        row = json.loads((directory / "summary.json").read_text())
        if row["status"] != "COMPLETE": return
        rows.append(row)
        provenance.append(dict(policy=policy,path=str(directory),reused_completed_myopic=reused))
        outcomes.append(pd.read_parquet(directory / "cohort_outcomes.parquet"))
    first = {str(p).replace('\\','/'):h for p,h in rows[0]["inputs_sha256"].items()}
    second = {str(p).replace('\\','/'):h for p,h in rows[1]["inputs_sha256"].items()}
    if any(second.get(p) != h for p,h in first.items()):
        raise ValueError("paired frozen execution inputs differ")
    pair = outcomes[0].merge(outcomes[1], on="order_id", validate="one_to_one", suffixes=("_base", "_new"))
    if len(pair) != len(outcomes[0]) or len(pair) != len(outcomes[1]):
        raise ValueError("full-day paired cohort mismatch")
    if not pair.common_eligible_base.equals(pair.common_eligible_new):
        raise ValueError("comparison populations differ")
    common = pair.matched_base & pair.matched_new
    result = dict(status="COMPLETE", rows=rows, original_orders=len(pair),
        execution_provenance=provenance,
        common_eligible=int(pair.common_eligible_base.sum()),
        gained=int((~pair.matched_base & pair.matched_new).sum()), lost=int((pair.matched_base & ~pair.matched_new).sum()),
        net_matched_change=int(pair.matched_new.sum()-pair.matched_base.sum()),
        common_served=int(common.sum()),
        common_served_mean_wait_delta_s=float((pair.loc[common,"wait_s_new"]-pair.loc[common,"wait_s_base"]).mean()),
        classification="INTERNAL_FULL_DAY_PAIRED_COMPARISON_NOT_UNSEEN_TEST",
        combined_policy_not_module_causal_ablation=True, statistical_significance_claim=False)
    write_json(root / doc_root / "analysis.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def run(root, fleetpy, policy, resume, administrative_timeout_s=None, acceleration_config=None,
        calibration_config=None):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    cfg = json.loads((root / CONFIG).read_text())
    technical, acceleration_path, group_limit = acceleration_settings(root, acceleration_config)
    calibration = None
    calibration_path = None
    if calibration_config is not None:
        from stage4.analysis.replay_calibration_contract import load_calibration
        calibration, calibration_path = load_calibration(root, calibration_config)
        if (policy != calibration["reference_policy"] or acceleration_path is None
                or acceleration_path.relative_to(root.resolve()).as_posix() != calibration["acceleration_config"]
                or administrative_timeout_s is not None):
            raise ValueError("all-HV reference requires the fixed v4 policy and timeout")
    cfg.update(technical)
    is_v2 = "offline_assets" in technical
    is_v3 = "demand_routing" in technical
    is_v4 = "static_raw_od" in technical
    output_root = OUTPUT if acceleration_path is None else OUTPUT.parent / ("accelerated_v4_full_day" if is_v4 else "accelerated_v3_full_day" if is_v3 else "accelerated_v2_full_day" if is_v2 else "accelerated_full_day")
    doc_root = DOC if acceleration_path is None else Path("stage4/docs/flexibility_dispatch") / ("acceleration_v4" if is_v4 else "acceleration_v3" if is_v3 else "acceleration_v2" if is_v2 else "acceleration_v1")
    if calibration is not None:
        output_root = Path(calibration["all_hv_output"])
        doc_root = Path(calibration["doc_output"])
    if policy not in cfg["policies"] or cfg["profile"] != "M" or cfg["parameter_search"] or cfg["refit_m3"]:
        raise ValueError("outside authorized two-condition frozen protocol")
    timeout_s = administrative_timeout(cfg, policy, administrative_timeout_s)
    prep = json.loads((root / INPUT / "preparation_summary.json").read_text())
    if prep["status"] != "COMPLETE" or prep["config_sha256"] != sha(root / CONFIG):
        raise ValueError("prepared configuration mismatch")
    directory = root / output_root / policy
    if (directory / "summary.json").exists():
        done = json.loads((directory / "summary.json").read_text())
        if resume and done["status"] == "COMPLETE":
            print(json.dumps(dict(skipped_completed=policy)), flush=True)
            if acceleration_path is not None and done.get("acceleration_config_sha256") != sha(acceleration_path):
                raise ValueError("completed acceleration settings differ")
            if calibration is not None:
                if done.get("calibration_config_sha256") != sha(calibration_path):
                    raise ValueError("completed calibration settings differ")
            else:
                analyze(root, output_root, doc_root)
            return
        raise ValueError("existing partial run is not automatically retried")
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    summary = dict(status="RUNNING", policy=policy, profile="M", source_config=str(CONFIG),
        administrative_timeout_s=timeout_s,
        administrative_extension=administrative_timeout_s is not None,
        scientific_and_per_epoch_solver_parameters_unchanged=True)
    summary.update(execution_pid=os.getpid(),
        execution_started_at=datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Singapore")).isoformat(),
        execution_code_sha=subprocess.check_output(["git", "-c", f"safe.directory={root.resolve().as_posix()}",
            "rev-parse", "HEAD"], cwd=root, text=True).strip())
    if acceleration_path is not None:
        summary.update(acceleration_config_sha256=sha(acceleration_path), execution_settings=technical,
            optimum_equivalence_not_bitwise_trajectory_identity=True)
    if calibration is not None:
        summary.update(calibration_config_sha256=sha(calibration_path),
            experiment_role="CURRENT_VERSION_ALL_HV_ABSOLUTE_REPLAY_REFERENCE",
            controlled_changes={"requested_q_a": 0.0, "passenger_acceptance_rate": 1.0},
            historical_driver_chain_replayed=False, request_times_changed=False,
            pickup_eta_calibration_changed=False, repositioning_enabled=False)
    write_json(directory / "summary.json", summary)
    paths = [CONFIG, INPUT/"test31_research_routes.parquet", INPUT/"train_request_templates.parquet",
        Path(cfg["remaining_time_model"]), Path("stage3/config/stage3_av_capability_profiles.json"),
        Path("stage2/output_v5_2/development/M3/epoch_004.pt")]
    protected = {str(p):sha(root/p) for p in paths}
    if acceleration_path is not None:
        protected[str(acceleration_path.relative_to(root.resolve()))] = sha(acceleration_path)
    if calibration_path is not None:
        protected[str(calibration_path.relative_to(root.resolve()))] = sha(calibration_path)
    routing = None
    peak_group_rss = 0.0
    peak_group_private = 0.0
    try:
        bindings = load_fleetpy_bindings(fleetpy)
        source = root / "stage4/output/final_experiments" / cfg["source_scenario"]
        base = json.loads((source / "scenario_config.json").read_text())["runtime_configuration"]
        if calibration is not None:
            base = {**base, "av_vehicle_hour_share": 0.0, "passenger_acceptance_rate": 1.0}
        start = pd.Timestamp("2016-10-31T00:00:00+08:00")
        assets = None
        if is_v2:
            from stage4.analysis.acceleration_assets import asset_input_hashes
            from stage4.dispatch.runtime_assets import RuntimeAssets, routing_context
            assets = RuntimeAssets(cfg["offline_assets"], expected_inputs=asset_input_hashes(root, cfg))
            if assets.manifest["routing_context"] != routing_context(root):
                raise ValueError("frozen routing engine context changed")
            raw_requests = assets.requests()
            protected[str((assets.directory / "manifest.json").relative_to(root))] = sha(assets.directory / "manifest.json")
        else:
            raw_requests = load_all_test31_requests(root, start=start, end=start+pd.Timedelta(seconds=cfg["measurement_end_s"]), profile_id="M")
        if len(raw_requests) != 30000: raise ValueError("source Test31 cohort is not 30000")
        if assets is not None:
            routes = None
            route_rows = {str(r["order_id"]): dict(common_eligible=bool(r["common"]), compatible_M=bool(r["mask"] & 2),
                predicted_route_time_p50_s=float(r["predicted"])) for r in assets.orders}
        else:
            routes = pd.read_parquet(root / INPUT / "test31_research_routes.parquet")
            route_rows = routes.set_index("order_id").to_dict("index")
        requests = [r for r in raw_requests if route_rows[r.order_id]["common_eligible"]]
        for r in requests: r.predicted_service_time_s = float(route_rows[r.order_id]["predicted_route_time_p50_s"])
        # Administrative bound frozen before outcomes; never derive a vehicle's
        # modeled availability from unobserved future Test31 trip durations.
        drain_s = int(cfg["physical_drain_limit_s"])
        end = start+pd.Timedelta(seconds=drain_s)
        fleet = assets.fleet() if assets is not None and calibration is None else build_fleet_scenario(root, benchmark_start=start, simulation_end=end,
            requested_q_a=base["av_vehicle_hour_share"], seed=base["fleet_sampling_seed"], max_hv_hour_error_pct=base["max_hv_vehicle_hour_error_pct"])
        registry = CoordinateRegistry()
        attach_fleetpy_requests(requests, bindings, registry)
        network = create_native_network(bindings, registry)
        demand = create_native_demand(bindings, requests, registry, network, directory)
        vehicles, native_output = create_native_vehicles(fleet.native_fixtures, bindings, registry, demand.rq_db,
            directory/"runtime", native_movement=True, routing_engine=network)
        native_cfg = {**base, "profile_id":"M", "gamma_static":None, "gamma_dynamic":None, "gamma_speed":None,
            "cost_level_enabled":False, "prospective_gate_logging":False, "additional_pickup_overhead_s":0.,
            "matching_end_s":cfg["last_dispatch_s"], "benchmark_runtime_guard_s":timeout_s,
            **{k: cfg[k] for k in ("cached_geometry", "epoch_routing_queue", "pre_route_session_certificate", "certified_eta_pruning") if k in cfg}}
        route_options = {}
        if assets is not None:
            route_options = dict(disk_cache_path=cfg["disk_route_cache"], routing_context=assets.manifest["routing_context"],
                disk_cache_limit_mib=cfg["disk_cache_limit_mib"], disk_cache_max_entries=cfg["disk_cache_max_entries"],
                route_queue_chunk_size=cfg["route_queue_chunk_size"], lazy_worker_actor=cfg["lazy_worker_actor"])
        router_class = ArcDeterministicValhallaAdapter
        if cfg.get("demand_routing"):
            from stage4.dispatch.routing_v3 import DemandRoutingAdapter
            router_class = DemandRoutingAdapter
            route_options["grouped_sources"] = cfg["grouped_sources"]
        if is_v4:
            from stage4.dispatch.routing_v4 import StaticRawRoutingAdapter
            router_class = StaticRawRoutingAdapter
            route_options.update({k: cfg[k] for k in ("static_raw_od", "raw_od_cache_size", "certified_eta_pruning")})
        routing = router_class(root, routing_mode=cfg["routing_mode"],
            persistent_cache_size=cfg.get("persistent_cache_size", 0), route_workers=cfg.get("route_workers", 1), **route_options)
        c = create_rolling_or_fleet_control(bindings, vehicles, requests, demand, network, routing, start, end, native_cfg)
        profiles = json.loads((root / "stage3/config/stage3_av_capability_profiles.json").read_text())
        if assets is not None:
            from stage4.dispatch.runtime_assets import CompactResearchRoutePolicy, DrawTapeForecast
            c.research_route_policy = CompactResearchRoutePolicy(assets, "M", profiles)
            forecast = DrawTapeForecast(assets.directory / "forecast")
            if cfg.get("event_calendar"):
                # q=0 has 8,435 sessions, not the mixed offline asset's 4,444
                # fixtures. Reuse requests/geometry, but use the actual fleet.
                c.enable_event_calendar(assets.windows() if calibration is None else None)
        else:
            c.research_route_policy = ResearchRoutePolicy(routes, "M", profiles)
            templates = pd.read_parquet(root / INPUT / "train_request_templates.parquet")
            forecast = TrainDemandForecast(templates, cfg, cfg["measurement_end_s"])
        model = TrainRemainingTime(json.loads((root / cfg["remaining_time_model"]).read_text()))
        c.flexibility_adapter = NativeFlexibilityAdapter(policy, forecast, cfg, model)
        sim = create_native_simulation(bindings, simulation_end_s=drain_s, time_step_s=30, demand=demand,
            vehicles=[v.native_vehicle for v in vehicles], fleet_control=c, network=network, native_output=native_output)
        write_json(directory / "fleet_accounting.json", fleet.accounting)
        if calibration is not None:
            fixtures = []
            for fixture in fleet.native_fixtures:
                row = asdict(fixture)
                for key in ("availability_start_time", "availability_end_time"):
                    row[key] = row[key].isoformat()
                fixtures.append(row)
            write_json(directory / "fleet_fixtures.json", {"fixtures": fixtures})
        for tick in range(0, drain_s+30, 30):
            if time.monotonic()-started > timeout_s:
                raise TimeoutError("full-day hard timeout")
            sim.step(tick)
            routing.cache.clear()
            if acceleration_path is not None:
                resources = routing.process_group_resources()
                group_rss = resources["rss_mib"]
                peak_group_rss = max(peak_group_rss, group_rss)
                peak_group_private = max(peak_group_private, resources["private_committed_mib"])
                if group_rss > group_limit:
                    raise MemoryError("acceleration process group memory limit exceeded")
                if is_v2 and psutil.disk_usage(root.anchor).free < 512 * 2**20:
                    raise OSError("v2 local disk reserve below 512 MiB; output/cache preserved, no automatic retry")
            if tick % 1800 == 0:
                progress = dict(policy=policy, tick=tick, matched=len(c.assignment_rows),
                    runtime_s=time.monotonic()-started, rss_mib=psutil.Process().memory_info().rss/2**20)
                if is_v2:
                    progress.update(process_group_rss_mib=group_rss,
                        process_group_private_committed_mib=resources["private_committed_mib"],
                        system_available_ram_mib=psutil.virtual_memory().available / 2**20,
                        local_disk_free_gib=psutil.disk_usage(root.anchor).free / 2**30)
                if is_v4:
                    progress.update(routing_arc_evaluations=routing.routing_arc_evaluations,
                        arc_lookups=routing.arc_lookup_count, arc_cache_hits=routing.cache_hit_count,
                        raw_od_memory_queries_avoided=routing.raw_od_memory_queries_avoided,
                        raw_od_cache_evictions=routing.raw_od_cache_evictions,
                        optional_cache_write_error=routing._disk_cache.write_error if routing._disk_cache else None)
                write_json(directory / "progress.json", progress)
                print(json.dumps(progress), flush=True)
            if tick > cfg["last_dispatch_s"] and not (set(c.rid_to_assigned_vid)-c.completed_rids): break
        c.reconcile()
        if any((c.position_reconciliation_failures, c.request_state_reconciliation_failures,
                c.vehicle_state_reconciliation_failures, c.av_availability_violations)):
            raise RuntimeError("native physical reconciliation failure")
        if set(c.rid_to_assigned_vid)-c.completed_rids: raise RuntimeError("physical drain incomplete")
        assignments = pd.DataFrame(c.assignment_rows)
        assigned = assignments.set_index("native_request_id")
        outcomes = []
        for r in raw_requests:
            eligible = bool(route_rows[r.order_id]["common_eligible"])
            matched = r.native_id in assigned.index
            if eligible and not matched and r.native_id not in c.expired_rids: raise RuntimeError("unaccounted modeled request")
            row = dict(order_id=r.order_id, common_eligible=eligible, matched=matched,
                expired=eligible and not matched, excluded_input=not eligible, vehicle_type=None, wait_s=None)
            if matched:
                a = assigned.loc[r.native_id]
                wait = (pd.Timestamp(a.pickup_time)-r.request_time).total_seconds()
                if wait > cfg["patience_s"]+1e-6: raise RuntimeError("pickup patience violation")
                if a.vehicle_type == "AV" and not route_rows[r.order_id]["compatible_M"]: raise RuntimeError("AV capability violation")
                row.update(vehicle_type=a.vehicle_type, wait_s=wait)
            outcomes.append(row)
        outcome = pd.DataFrame(outcomes)
        for _, g in assignments.groupby("native_vehicle_id"):
            g=g.sort_values("assignment_time")
            if not (pd.to_datetime(g.assignment_time).iloc[1:].to_numpy() >= pd.to_datetime(g.service_end_time).iloc[:-1].to_numpy()).all():
                raise RuntimeError("vehicle task overlap")
        pu=(pd.to_datetime(assignments.pickup_time)-pd.to_datetime(assignments.assignment_time)).dt.total_seconds()-assignments.pickup_eta_s
        service=(pd.to_datetime(assignments.service_end_time)-pd.to_datetime(assignments.pickup_time)).dt.total_seconds()-assignments.realized_service_time_s
        if max(pu.abs().max(), service.abs().max()) > 1e-6: raise RuntimeError("physical time-chain violation")
        missing=~assignments.research_exposure_available.fillna(False)
        assignments.loc[missing,["exposure_static","exposure_dynamic","exposure_speed"]]=float("nan")
        traces=pd.DataFrame(c.flexibility_adapter.rows)
        epochs=pd.DataFrame(c.epoch_rows)
        for name,frame in (("assignments",assignments),("cohort_outcomes",outcome),("solver_trace",traces),("epochs",epochs)):
            write_parquet(directory/(name+".parquet"),frame)
        if protected != {p:sha(root/p) for p in protected}: raise RuntimeError("frozen execution input changed")
        summary.update(status="COMPLETE", original_orders=len(raw_requests), common_eligible=len(requests),
            matched=int(outcome.matched.sum()), av=int(outcome.vehicle_type.eq("AV").sum()), hv=int(outcome.vehicle_type.eq("HV").sum()),
            expired=int(outcome.expired.sum()), excluded_input=int(outcome.excluded_input.sum()),
            service_rate_common_population=float(outcome.matched.sum()/len(requests)),
            service_rate_original_30000=float(outcome.matched.mean()), mean_wait_s=float(outcome.wait_s.mean()),
            wait_quantiles_s={str(q):float(outcome.wait_s.quantile(q)) for q in (.5,.9,.95)},
            maximum_model_variables=int(traces.variable_count.max()) if traces.variable_count.notna().any() else 0,
            maximum_model_nonzeros=int(traces.nonzeros.max()) if traces.nonzeros.notna().any() else 0,
            resource_fallback_epochs=int(traces.resource_fallback.notna().sum()),
            current_face_violation_epochs=int((~traces.current_face_preserved).sum()) if "current_face_preserved" in traces else 0,
            dispatch_epochs=len(traces), maximum_topk_pairs=int(epochs.candidate_topk_pairs.max()),
            maximum_valid_arcs=int(epochs.valid_or_arcs.max()), routing_failures=routing.routing_failures,
            routing_arc_evaluations=routing.routing_arc_evaluations, solver_time_s=float(traces.solver_time_s.sum()),
            runtime_s=time.monotonic()-started, peak_rss_mib=psutil.Process().memory_info().peak_wset/2**20,
            gpu_used=False, dense_matrix=False, physical_reconciliation=True, inputs_sha256=protected,
            train_remaining_model_refitted=False, native_start_s=0, drain_end_s=tick)
        summary.update(timing_totals_s={
            "candidate_generation":float(epochs.candidate_generation_time_s.sum()),
            "routing_wall":float(epochs.routing_time_s.sum()),
            "solver_pipeline":float(traces.solver_time_s.sum())},
            solver_pipeline_components_s={k:float(traces[k].fillna(0).sum()) for k in
                ("forecast_sampling_time_s", "future_graph_time_s", "problem_setup_time_s",
                 "model_build_time_s", "optimization_time_s", "recourse_recovery_time_s",
                 "failed_model_attempt_time_s") if k in traces},
            arc_lookups=routing.arc_lookup_count, arc_cache_hits=routing.cache_hit_count,
            cross_epoch_exact_cache_hits=routing.persistent_cache_hits,
            disk_route_cache_hits=routing.disk_cache_hits, raw_lru_hits=routing.raw_lru_hits,
            raw_query_deduplications=routing.raw_query_deduplications, epoch_route_batches=routing.epoch_route_batches,
            prefetched_unused_raw_arcs=routing.prefetched_unused_raw_arcs,
            routing_queue_pipeline_time_s=routing.routing_queue_pipeline_time_s,
            disk_cache_lookup_time_s=routing.disk_cache_lookup_time_s,
            zero_eta_session_certified_prunes=int(epochs.zero_eta_session_certified_prunes.sum()),
            eliminated_fixed_variables=int(traces.eliminated_fixed_variables.sum()),
            peak_process_group_rss_mib=peak_group_rss if acceleration_path is not None else summary["peak_rss_mib"])
        if is_v2:
            summary["peak_process_group_private_committed_mib"] = peak_group_private
        if is_v3:
            summary.update(raw_prefetch_avoided=routing.raw_prefetch_avoided,
                source_group_queries=routing.source_group_queries,
                group_cell_fallback_queries=routing.group_cell_fallback_queries,
                decomposition_components=int(traces.decomposed_component_count.sum()),
                pure_future_components=int(traces.pure_future_component_count.sum()))
        if is_v4:
            summary["routing_v4"] = routing.diagnostics()
    except Exception as error:
        summary.update(status="STOPPED", error=repr(error), runtime_s=time.monotonic()-started)
        write_json(directory / "summary.json", summary)
        raise
    finally:
        if routing is not None:
            routing.close()
            if is_v4:
                summary["routing_v4"] = routing.diagnostics()
        summary.update(runtime_s=time.monotonic()-started,
            execution_finished_at=datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Singapore")).isoformat())
        if summary.get("status") == "STOPPED":
            write_json(directory / "summary.json", summary)
    write_json(directory / "summary.json", summary)
    (root / doc_root).mkdir(parents=True,exist_ok=True)
    summary_name = "all_hv_summary.json" if calibration is not None else policy.lower()+"_summary.json"
    write_json(root / doc_root / summary_name, summary)
    print(json.dumps(summary),flush=True)
    gc.collect()
    if calibration is None:
        analyze(root, output_root, doc_root)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleetpy-root",type=Path,required=True)
    parser.add_argument("--policy",choices=("MYOPIC","SERVICE_PRESERVING_LOOKAHEAD"),required=True)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--administrative-timeout-s",type=int,choices=(21600,),
        help="User-authorized 6h retry of the second policy after the original 3h administrative timeout")
    parser.add_argument("--acceleration-config",type=Path,
        help="Opt-in technical acceleration; writes to a separate output directory")
    parser.add_argument("--calibration-config", type=Path,
        help="One opt-in all-HV absolute replay reference; never changes frozen mixed runs")
    args=parser.parse_args()
    run(Path.cwd(),args.fleetpy_root,args.policy,args.resume,args.administrative_timeout_s,
        args.acceleration_config,args.calibration_config)
