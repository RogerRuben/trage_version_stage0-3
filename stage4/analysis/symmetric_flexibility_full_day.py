"""Exactly two serial Test31 native full-day comparisons with symmetric support."""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time

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


def analyze(root):
    cfg = json.loads((root / CONFIG).read_text())
    rows = []
    outcomes = []
    for policy in cfg["policies"]:
        directory = root / OUTPUT / policy
        if not (directory / "summary.json").exists(): return
        row = json.loads((directory / "summary.json").read_text())
        if row["status"] != "COMPLETE": return
        rows.append(row)
        outcomes.append(pd.read_parquet(directory / "cohort_outcomes.parquet"))
    pair = outcomes[0].merge(outcomes[1], on="order_id", validate="one_to_one", suffixes=("_base", "_new"))
    if len(pair) != len(outcomes[0]) or len(pair) != len(outcomes[1]):
        raise ValueError("full-day paired cohort mismatch")
    if not pair.common_eligible_base.equals(pair.common_eligible_new):
        raise ValueError("comparison populations differ")
    common = pair.matched_base & pair.matched_new
    result = dict(status="COMPLETE", rows=rows, original_orders=len(pair),
        common_eligible=int(pair.common_eligible_base.sum()),
        gained=int((~pair.matched_base & pair.matched_new).sum()), lost=int((pair.matched_base & ~pair.matched_new).sum()),
        net_matched_change=int(pair.matched_new.sum()-pair.matched_base.sum()),
        common_served=int(common.sum()),
        common_served_mean_wait_delta_s=float((pair.loc[common,"wait_s_new"]-pair.loc[common,"wait_s_base"]).mean()),
        classification="INTERNAL_FULL_DAY_PAIRED_COMPARISON_NOT_UNSEEN_TEST",
        combined_policy_not_module_causal_ablation=True, statistical_significance_claim=False)
    write_json(root / DOC / "analysis.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def run(root, fleetpy, policy, resume):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    cfg = json.loads((root / CONFIG).read_text())
    if policy not in cfg["policies"] or cfg["profile"] != "M" or cfg["parameter_search"] or cfg["refit_m3"]:
        raise ValueError("outside authorized two-condition frozen protocol")
    prep = json.loads((root / INPUT / "preparation_summary.json").read_text())
    if prep["status"] != "COMPLETE" or prep["config_sha256"] != sha(root / CONFIG):
        raise ValueError("prepared configuration mismatch")
    directory = root / OUTPUT / policy
    if (directory / "summary.json").exists():
        done = json.loads((directory / "summary.json").read_text())
        if resume and done["status"] == "COMPLETE":
            print(json.dumps(dict(skipped_completed=policy)), flush=True)
            analyze(root)
            return
        raise ValueError("existing partial run is not automatically retried")
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    summary = dict(status="RUNNING", policy=policy, profile="M", source_config=str(CONFIG))
    write_json(directory / "summary.json", summary)
    paths = [CONFIG, INPUT/"test31_research_routes.parquet", INPUT/"train_request_templates.parquet",
        Path(cfg["remaining_time_model"]), Path("stage3/config/stage3_av_capability_profiles.json"),
        Path("stage2/output_v5_2/development/M3/epoch_004.pt")]
    protected = {str(p):sha(root/p) for p in paths}
    try:
        bindings = load_fleetpy_bindings(fleetpy)
        source = root / "stage4/output/final_experiments" / cfg["source_scenario"]
        base = json.loads((source / "scenario_config.json").read_text())["runtime_configuration"]
        start = pd.Timestamp("2016-10-31T00:00:00+08:00")
        raw_requests = load_all_test31_requests(root, start=start, end=start+pd.Timedelta(seconds=cfg["measurement_end_s"]), profile_id="M")
        if len(raw_requests) != 30000: raise ValueError("source Test31 cohort is not 30000")
        routes = pd.read_parquet(root / INPUT / "test31_research_routes.parquet")
        route_rows = routes.set_index("order_id").to_dict("index")
        requests = [r for r in raw_requests if route_rows[r.order_id]["common_eligible"]]
        for r in requests: r.predicted_service_time_s = float(route_rows[r.order_id]["predicted_route_time_p50_s"])
        drain_s = math.ceil((cfg["last_dispatch_s"]+cfg["patience_s"]+max(r.realized_service_time_s for r in raw_requests))/30)*30+30
        end = start+pd.Timedelta(seconds=drain_s)
        fleet = build_fleet_scenario(root, benchmark_start=start, simulation_end=end,
            requested_q_a=base["av_vehicle_hour_share"], seed=base["fleet_sampling_seed"],
            max_hv_hour_error_pct=base["max_hv_vehicle_hour_error_pct"])
        registry = CoordinateRegistry()
        attach_fleetpy_requests(requests, bindings, registry)
        network = create_native_network(bindings, registry)
        demand = create_native_demand(bindings, requests, registry, network, directory)
        vehicles, native_output = create_native_vehicles(fleet.native_fixtures, bindings, registry, demand.rq_db,
            directory/"runtime", native_movement=True, routing_engine=network)
        native_cfg = {**base, "profile_id":"M", "gamma_static":None, "gamma_dynamic":None, "gamma_speed":None,
            "cost_level_enabled":False, "prospective_gate_logging":False, "additional_pickup_overhead_s":0.,
            "matching_end_s":cfg["last_dispatch_s"], "benchmark_runtime_guard_s":cfg["scenario_timeout_s"]}
        routing = ArcDeterministicValhallaAdapter(root, routing_mode=cfg["routing_mode"])
        c = create_rolling_or_fleet_control(bindings, vehicles, requests, demand, network, routing, start, end, native_cfg)
        profiles = json.loads((root / "stage3/config/stage3_av_capability_profiles.json").read_text())
        c.research_route_policy = ResearchRoutePolicy(routes, "M", profiles)
        templates = pd.read_parquet(root / INPUT / "train_request_templates.parquet")
        model = TrainRemainingTime(json.loads((root / cfg["remaining_time_model"]).read_text()))
        c.flexibility_adapter = NativeFlexibilityAdapter(policy, TrainDemandForecast(templates, cfg, cfg["measurement_end_s"]), cfg, model)
        sim = create_native_simulation(bindings, simulation_end_s=drain_s, time_step_s=30, demand=demand,
            vehicles=[v.native_vehicle for v in vehicles], fleet_control=c, network=network, native_output=native_output)
        write_json(directory / "fleet_accounting.json", fleet.accounting)
        for tick in range(0, drain_s+30, 30):
            if time.monotonic()-started > cfg["scenario_timeout_s"]:
                raise TimeoutError("full-day hard timeout")
            sim.step(tick)
            routing.cache.clear()
            if tick % 1800 == 0:
                progress = dict(policy=policy, tick=tick, matched=len(c.assignment_rows),
                    runtime_s=time.monotonic()-started, rss_mib=psutil.Process().memory_info().rss/2**20)
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
    except Exception as error:
        summary.update(status="STOPPED", error=repr(error), runtime_s=time.monotonic()-started)
        write_json(directory / "summary.json", summary)
        raise
    write_json(directory / "summary.json", summary)
    write_json(root / DOC / (policy.lower()+"_summary.json"),summary)
    print(json.dumps(summary),flush=True)
    gc.collect()
    analyze(root)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleetpy-root",type=Path,required=True)
    parser.add_argument("--policy",choices=("MYOPIC","SERVICE_PRESERVING_LOOKAHEAD"),required=True)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    run(Path.cwd(),args.fleetpy_root,args.policy,args.resume)
