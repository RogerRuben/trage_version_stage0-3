"""Eighteen serial native continuations from the two prespecified checkpoints."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import pandas as pd
import psutil
from joblib.externals import cloudpickle

from stage3.scripts.traffic_state_batch1 import sha, write_json, write_parquet
from stage4.analysis.flexibility_prepare import CONFIG, OUT as INPUT, DOC
from stage4.dispatch.deterministic_routing import ArcDeterministicValhallaAdapter
from stage4.dispatch.flexibility_native import NativeFlexibilityAdapter, ResearchRoutePolicy, TrainDemandForecast
from stage4.fleetpy_adapter.upstream import load_fleetpy_bindings

OUTPUT = Path("stage4/output/flexibility_dispatch_v1/windows")


def condition(root, checkpoint, cfg, cut, profile, policy, destination, routes, templates):
    started = time.monotonic()
    with (root / checkpoint["path"]).open("rb") as stream:
        sim = cloudpickle.load(stream)
    c = sim.operators[0]
    if c.dispatch_interval_s != cfg["rolling_step_s"] or c.max_pickup_wait_s != cfg["patience_s"]:
        raise ValueError("checkpoint timing does not match prespecified experiment")
    if c.config["candidate_top_k"] != cfg["current_top_k"]:
        raise ValueError("checkpoint current Top-K does not match prespecified experiment")
    if c.config.get("repositioning_enabled", False):
        raise ValueError("repositioning is outside this experiment")
    end = cut + cfg["window_duration_s"]
    last_dispatch = end + cfg["patience_s"]
    c.config = {**c.config, "profile_id": profile, "gamma_static": None,
                "gamma_dynamic": None, "gamma_speed": None, "additional_pickup_overhead_s": 0.0}
    c.gammas = {k: None for k in ("static", "dynamic", "speed")}
    c.cost_level_enabled = False
    c.prospective_gate_logging = False
    c.traffic_policy = None
    frozen_profiles = json.loads((root / "stage3/config/stage3_av_capability_profiles.json").read_text())
    c.research_route_policy = ResearchRoutePolicy(routes, profile, frozen_profiles)
    # Restore ORIGINAL frozen M3 P50 for all requests, including old NONE/FALLBACK.
    # This is a decision attribute; physical realized progression stays untouched.
    p50 = routes.set_index("order_id").predicted_route_time_p50_s.to_dict()
    for request in c.request_by_rid.values():
        request.predicted_service_time_s = float(p50[str(request.order_id)])
    for rid, meta in c.request_meta.items():
        meta.update(c.research_route_policy.evaluate(c.request_by_rid[rid]))
    c.flexibility_adapter = NativeFlexibilityAdapter(policy, TrainDemandForecast(templates, cfg, end), cfg)
    c.eta_adapter = ArcDeterministicValhallaAdapter(root, routing_mode=cfg["routing_mode"])
    c.run_started_perf = time.perf_counter()
    c.runtime_guard_s = cfg["scenario_timeout_s"]
    c.matching_end_s = last_dispatch
    sim.demand.future_requests = {t: r for t, r in sim.demand.future_requests.items() if t < end}
    cohort = {rid: r for rid, r in c.request_by_rid.items() if cut <= r.sim_time_s < end}
    initial_assignments, initial_epochs = len(c.assignment_rows), len(c.epoch_rows)
    # Realized durations belong ONLY to physical draining/evaluation, not forecasts.
    max_duration = max(r.realized_service_time_s for r in c.request_by_rid.values())
    drain_limit = int(last_dispatch + cfg["patience_s"] + max_duration + 90)
    for tick in range(cut, drain_limit + 30, 30):
        if time.monotonic() - started > cfg["scenario_timeout_s"]:
            raise TimeoutError("native continuation exceeded prespecified hard timeout")
        if tick == cut:
            c.time_trigger(tick)
        else:
            sim.step(tick)
        c.eta_adapter.cache.clear()
        if tick % 300 == 0 and tick <= last_dispatch:
            rss = psutil.Process().memory_info().rss / 2**20
            print(json.dumps(dict(profile=profile, policy=policy, cut=cut, tick=tick,
                                  assignments=len(c.assignment_rows) - initial_assignments,
                                  rss_mib=rss, rss_warning=rss > cfg["rss_warning_mib"])), flush=True)
        if tick > last_dispatch and not (set(c.rid_to_assigned_vid) - c.completed_rids):
            break
    c.reconcile()
    if any((c.position_reconciliation_failures, c.request_state_reconciliation_failures,
            c.vehicle_state_reconciliation_failures, c.av_availability_violations)):
        raise RuntimeError("native physical reconciliation failure")
    if set(c.rid_to_assigned_vid) - c.completed_rids:
        raise RuntimeError("physical drain did not finish all accepted services")
    assignments = pd.DataFrame(c.assignment_rows[initial_assignments:])
    assigned = assignments.set_index("native_request_id") if len(assignments) else pd.DataFrame()
    outcome = []
    for rid, request in cohort.items():
        matched = len(assignments) > 0 and rid in assigned.index
        if not matched and rid not in c.expired_rids:
            raise RuntimeError("unaccounted cohort request")
        row = dict(order_id=request.order_id, matched=matched, expired=not matched,
                   vehicle_type=None, wait_s=None, research_data_ready=c.research_route_policy.rows[request.order_id]["research_data_ready"],
                   profile_compatible=c.research_route_policy.rows[request.order_id][f"compatible_{profile}"])
        if matched:
            assignment = assigned.loc[rid]
            wait = (pd.Timestamp(assignment.pickup_time) - request.request_time).total_seconds()
            if wait > cfg["patience_s"] + 1e-6:
                raise RuntimeError("pickup patience violation")
            row.update(vehicle_type=assignment.vehicle_type, wait_s=wait)
        outcome.append(row)
    outcome = pd.DataFrame(outcome)
    for _, group in pd.DataFrame(c.assignment_rows).groupby("native_vehicle_id"):
        group = group.sort_values("assignment_time")
        if not (pd.to_datetime(group.assignment_time).iloc[1:].to_numpy() >= pd.to_datetime(group.service_end_time).iloc[:-1].to_numpy()).all():
            raise RuntimeError("vehicle service overlap")
    if len(assignments):
        pickup_error = (pd.to_datetime(assignments.pickup_time) - pd.to_datetime(assignments.assignment_time)).dt.total_seconds() - assignments.pickup_eta_s
        service_error = (pd.to_datetime(assignments.service_end_time) - pd.to_datetime(assignments.pickup_time)).dt.total_seconds() - assignments.realized_service_time_s
        if max(pickup_error.abs().max(), service_error.abs().max()) > 1e-6:
            raise RuntimeError("physical time-chain violation")
        av = assignments.loc[assignments.vehicle_type.eq("AV")]
        if any(not c.research_route_policy.rows[str(order)][f"compatible_{profile}"] for order in av.order_id):
            raise RuntimeError("assigned AV outside the declared route compatibility")
        # No synthetic zeros are advertised as measured continuous exposure.
        missing = ~assignments.research_exposure_available.fillna(False)
        assignments.loc[missing, ["exposure_static", "exposure_dynamic", "exposure_speed"]] = float("nan")
    traces = pd.DataFrame(c.flexibility_adapter.rows)
    write_parquet(destination / "assignments.parquet", assignments)
    write_parquet(destination / "cohort_outcomes.parquet", outcome)
    write_parquet(destination / "epochs.parquet", pd.DataFrame(c.epoch_rows[initial_epochs:]))
    write_parquet(destination / "solver_trace.parquet", traces)
    mem = psutil.Process().memory_info()
    result = dict(status="COMPLETE", profile=profile, policy=policy, cut=cut, cohort=len(outcome),
        matched=int(outcome.matched.sum()), av=int(outcome.vehicle_type.eq("AV").sum()),
        hv=int(outcome.vehicle_type.eq("HV").sum()), expired=int(outcome.expired.sum()),
        mean_wait_s=float(outcome.wait_s.mean()), data_ready=int(outcome.research_data_ready.sum()),
        compatible=int(outcome.profile_compatible.sum()),
        routing_failures=c.eta_adapter.routing_failures, routing_arc_evaluations=c.eta_adapter.routing_arc_evaluations,
        resource_fallback_epochs=int(traces.resource_fallback.notna().sum()) if len(traces) else 0,
        maximum_model_variables=int(traces.variable_count.max()) if traces.variable_count.notna().any() else 0,
        maximum_model_nonzeros=int(traces.nonzeros.max()) if traces.nonzeros.notna().any() else 0,
        solver_time_s=float(traces.solver_time_s.sum()), runtime_s=time.monotonic() - started,
        peak_rss_mib=getattr(mem, "peak_wset", mem.rss) / 2**20, physical_reconciliation=True,
        drain_end_s=tick, source_checkpoint_sha256=checkpoint["sha256"],
        forecast_source_dates=cfg["forecast_train_dates"], forecast_source_is_test31=False,
        current_routing="DETERMINISTIC_VALHALLA", future_routing=cfg["future_pickup_eta_model"],
        gamma_constraints_enabled=False, operating_cost_enabled=False)
    write_json(destination / "summary.json", result)
    return result


def compare(root, cfg):
    rows, pairs = [], []
    for cut in cfg["cuts_s"]:
        for profile in cfg["profiles"]:
            baseline = pd.read_parquet(root / OUTPUT / f"{profile}_{cut}_MYOPIC/cohort_outcomes.parquet")
            for policy in cfg["policies"]:
                directory = root / OUTPUT / f"{profile}_{cut}_{policy}"
                result = json.loads((directory / "summary.json").read_text())
                rows.append(result)
                outcomes = pd.read_parquet(directory / "cohort_outcomes.parquet")
                paired = baseline.merge(outcomes, on="order_id", validate="one_to_one", suffixes=("_base", "_policy"))
                if len(paired) != len(baseline) or len(outcomes) != len(baseline):
                    raise ValueError("paired cohort mismatch")
                pairs.append(dict(cut=cut, profile=profile, policy=policy, cohort=len(paired),
                    net_matched_change=int(paired.matched_policy.sum() - paired.matched_base.sum()),
                    gained=int((~paired.matched_base & paired.matched_policy).sum()),
                    lost=int((paired.matched_base & ~paired.matched_policy).sum())))
    result = dict(status="COMPLETE", rows=rows, pairs=pairs, classification="BOUNDED_EXPLORATORY_NATIVE_COMPARISON")
    write_json(root / DOC / "native_window_comparison.json", result)
    return result


def run(root, fleetpy, resume):
    cfg = json.loads((root / CONFIG).read_text())
    prepared = json.loads((root / INPUT / "preparation_summary.json").read_text())
    if prepared["status"] != "COMPLETE" or prepared["config_sha256"] != sha(root / CONFIG):
        raise ValueError("prepared input/config mismatch")
    load_fleetpy_bindings(fleetpy)
    checkpoints = json.loads((root / "stage4/output/traffic_mechanism_v1/states/checkpoints.json").read_text())["rows"]
    checkpoints = {r["cut"]: r for r in checkpoints if r["profile"] == cfg["source_profile"]}
    protected = {str(p): sha(root / p) for p in (CONFIG, Path("stage3/config/stage3_av_capability_profiles.json"),
                 Path("stage2/output_v5_2/development/M3/epoch_004.pt"))}
    routes = pd.read_parquet(root / INPUT / "test31_research_routes.parquet")
    templates = pd.read_parquet(root / INPUT / "train_request_templates.parquet")
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=resume)
    summary = dict(status="RUNNING", rows=[], protected_sha256=protected, active=None)
    started = time.monotonic()
    try:
        for cut in cfg["cuts_s"]:
            checkpoint = checkpoints[cut]
            if sha(root / checkpoint["path"]) != checkpoint["sha256"]:
                raise ValueError("common physical checkpoint changed")
            for profile in cfg["profiles"]:
                for policy in cfg["policies"]:
                    name = f"{profile}_{cut}_{policy}"
                    destination = output / name
                    done = destination / "summary.json"
                    if resume and done.exists() and json.loads(done.read_text()).get("status") == "COMPLETE":
                        summary["rows"].append(json.loads(done.read_text()))
                        continue
                    destination.mkdir(exist_ok=resume)
                    summary["active"] = name
                    write_json(output / "summary.json", summary)
                    result = condition(root, checkpoint, cfg, cut, profile, policy, destination, routes, templates)
                    summary["rows"].append(result)
                    write_json(output / "summary.json", summary)
                    print(json.dumps(dict(completed=name, matched=result["matched"], av=result["av"],
                        runtime_s=result["runtime_s"], fallbacks=result["resource_fallback_epochs"])), flush=True)
                    gc.collect()
        compare(root, cfg)
        if protected != {p: sha(root / p) for p in protected}:
            raise RuntimeError("a frozen input changed during execution")
        summary.update(status="COMPLETE", active=None, runtime_s=time.monotonic() - started,
                       peak_rss_mib=psutil.Process().memory_info().peak_wset / 2**20)
    except Exception as error:
        summary.update(status="STOPPED", error=repr(error))
        write_json(output / "summary.json", summary)
        raise
    write_json(output / "summary.json", summary)
    write_json(root / DOC / "native_window_summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleetpy-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(Path.cwd(), args.fleetpy_root, args.resume)
