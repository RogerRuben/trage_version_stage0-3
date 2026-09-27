"""Two bounded native continuations; no full-day rerun or policy selection."""
import argparse
import gc
import json
import math
from pathlib import Path
import time
import pandas as pd
import numpy as np
import psutil
from joblib.externals import cloudpickle

from stage3.scripts.traffic_state_batch1 import sha, write_json
from stage4.dispatch.traffic_research_policy import MODE, TrafficResearchPolicy
from stage4.dispatch.deterministic_routing import ArcDeterministicValhallaAdapter
from stage4.fleetpy_adapter.upstream import load_fleetpy_bindings

CONFIG = Path("stage4/config/traffic_research_window.json")
OUT = Path("stage4/output/traffic_research/window")
DOC = Path("stage4/docs/traffic_research")


def condition(root, cfg, mode, dest):
    start = time.monotonic()
    checkpoint = root/cfg["checkpoint"]
    checkpoint_sha = sha(checkpoint)
    with checkpoint.open("rb") as f:
        sim = cloudpickle.load(f)
    c = sim.operators[0]
    assert c.dispatch_interval_s == 30 and c.max_pickup_wait_s == 300
    assert c.acceptance_rate == .7 and c.config["profile_id"] == cfg["profile_id"]
    table_path = root/"stage4/output/traffic_research/policy_date=20161031.parquet"
    table = pd.read_parquet(table_path)
    policy = TrafficResearchPolicy(table, cfg["main_budget"])
    # Every originally eligible request must have exact selected-original identity.
    ready = 0
    for r in c.request_by_rid.values():
        if not r.av_smoke_eligible:
            continue
        result = policy.evaluate(r)
        key = (r.request_time.strftime("%Y%m%d"), str(r.order_id), str(r.profile_id), "ORIGINAL:"+r.order_id)
        assert np.isclose(policy.rows[key]["rho_dynamic_frozen"], r.rho_dynamic, atol=1e-6, rtol=1e-6)
        assert np.isclose(policy.rows[key]["predicted_time_s"], r.predicted_service_time_s, atol=1e-3, rtol=1e-5)
        assert result["exposure"].dynamic <= max(0, r.rho_dynamic-1)+1e-6
        ready += 1
    c.config = {**c.config, "traffic_research_policy":mode, "traffic_research_budget": cfg["main_budget"],
                "traffic_research_table":str(table_path), "additional_pickup_overhead_s":0}
    c.traffic_policy = policy if mode == MODE else None
    old_dynamic = c.exposure_state.dynamic
    if mode == MODE:
        past_av = [r for r in c.assignment_rows if r["vehicle_type"] == "AV"]
        assert len(past_av) == c.exposure_state.av_assignments
        c.exposure_state.dynamic = sum(policy.evaluate(c.request_by_rid[int(a["native_request_id"])])["exposure"].dynamic for a in past_av)
        assert c.exposure_state.dynamic <= old_dynamic+1e-6
        for rid, meta in c.request_meta.items():
            result = policy.evaluate(c.request_by_rid[rid])
            if result is not None:
                meta.update(result)
    rebased_dynamic = c.exposure_state.dynamic
    c.exposure_state.validate(c.gammas, c.solver_tolerance)
    c.eta_adapter = ArcDeterministicValhallaAdapter(root, routing_mode=cfg["routing_mode"])
    c.run_started_perf = time.perf_counter()
    c.runtime_guard_s = cfg["scenario_timeout_s"]
    c.matching_end_s = cfg["last_dispatch_s"]
    sim.demand.future_requests = {t:v for t,v in sim.demand.future_requests.items() if t<cfg["measurement_end_s"]}
    cohort = {rid:r for rid,r in c.request_by_rid.items() if cfg["measurement_start_s"] <= r.sim_time_s < cfg["measurement_end_s"]}
    drain = math.ceil((cfg["last_dispatch_s"]+300+max(r.realized_service_time_s for r in c.request_by_rid.values()))/30)*30+30
    initial_rows, initial_epochs, initial_exposure = len(c.assignment_rows), len(c.epoch_rows), len(c.exposure_rows)
    for tick in range(cfg["checkpoint_s"], drain+30, 30):
        if time.monotonic()-start > cfg["scenario_timeout_s"]:
            raise TimeoutError("traffic scenario exceeded prespecified 1800 seconds")
        if tick == cfg["checkpoint_s"]:
            c.time_trigger(tick)
        else:
            sim.step(tick)
        c.eta_adapter.cache.clear()
        if tick % 300 == 0:
            rss = psutil.Process().memory_info().rss/2**20
            print(json.dumps(dict(mode=mode,tick=tick,new_assignments=len(c.assignment_rows)-initial_rows,rss_mib=rss,
                                  rss_warning=rss>cfg["rss_warning_mib"])),flush=True)
        if tick>cfg["last_dispatch_s"] and not(set(c.rid_to_assigned_vid)-c.completed_rids):
            break
    c.reconcile()
    assert not(set(c.rid_to_assigned_vid)-c.completed_rids)
    assert not any((c.position_reconciliation_failures,c.request_state_reconciliation_failures,c.vehicle_state_reconciliation_failures,c.av_availability_violations))
    assignments = pd.DataFrame(c.assignment_rows[initial_rows:])
    assigned = assignments.set_index("native_request_id")
    outcomes = []
    for rid,r in cohort.items():
        matched = rid in assigned.index
        assert matched or rid in c.expired_rids
        update = policy.evaluate(r)
        row = dict(order_id=r.order_id, matched=matched, expired=not matched, vehicle_type=None, wait_s=None,
            research_outside_share=update["traffic_outside_share"] if update else None,
            research_unknown_share=update["traffic_unknown_share"] if update else None)
        if matched:
            a = assigned.loc[rid]
            wait = (pd.Timestamp(a.pickup_time)-r.request_time).total_seconds()
            assert wait<=300+1e-6
            row.update(vehicle_type=a.vehicle_type, wait_s=wait)
        outcomes.append(row)
    outcome = pd.DataFrame(outcomes)
    av = assignments.loc[assignments.vehicle_type.eq("AV")]
    if mode == MODE:
        assert av.traffic_allowed.all() and av.traffic_outside_share.le(cfg["main_budget"]+1e-6).all()
        assert av.exposure_dynamic.notna().all()
    pickup_error = (pd.to_datetime(assignments.pickup_time)-pd.to_datetime(assignments.assignment_time)).dt.total_seconds()-assignments.pickup_eta_s
    service_error = (pd.to_datetime(assignments.service_end_time)-pd.to_datetime(assignments.pickup_time)).dt.total_seconds()-assignments.realized_service_time_s
    assert pickup_error.abs().max()<1e-6 and service_error.abs().max()<1e-6
    # All physical tasks, including inherited history, remain non-overlapping.
    for _, g in pd.DataFrame(c.assignment_rows).groupby("native_vehicle_id"):
        g = g.sort_values("assignment_time")
        assert (pd.to_datetime(g.assignment_time).iloc[1:].to_numpy() >= pd.to_datetime(g.service_end_time).iloc[:-1].to_numpy()).all()
    baseline_exact = None
    if mode == "FROZEN":
        prior = pd.read_parquet(root/"stage4/output/paper_enhancement/dwell_deterministic_window/q50_dwell0/assignments.parquet")
        keys = ["simulation_time_s","native_request_id","native_vehicle_id","pickup_eta_s"]
        a = assignments.loc[assignments.simulation_time_s.lt(cfg["measurement_end_s"]),keys].sort_values(keys[:3]).reset_index(drop=True)
        b = prior.loc[prior.simulation_time_s.lt(cfg["measurement_end_s"]),keys].sort_values(keys[:3]).reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=True)
        baseline_exact = True
    assignments.to_parquet(dest/"assignments.parquet",index=False)
    outcome.to_parquet(dest/"cohort_outcomes.parquet",index=False)
    pd.DataFrame(c.epoch_rows[initial_epochs:]).to_parquet(dest/"epochs.parquet",index=False)
    pd.DataFrame(c.exposure_rows[initial_exposure:]).to_parquet(dest/"exposure.parquet",index=False)
    assert sha(checkpoint) == checkpoint_sha
    mem = psutil.Process().memory_info()
    result = dict(mode=mode,status="COMPLETE",cohort_orders=len(outcome),matched=int(outcome.matched.sum()),
        expired=int(outcome.expired.sum()),matched_AV=int(outcome.vehicle_type.eq("AV").sum()),matched_HV=int(outcome.vehicle_type.eq("HV").sum()),
        mean_wait_s=float(outcome.wait_s.mean()),post_checkpoint_assignments=len(assignments),eligible_requests_checked=ready,
        inherited_dynamic_before=old_dynamic,inherited_dynamic_after=rebased_dynamic,
        baseline_exact_until_demand_cutoff=baseline_exact,pickup_error_s=float(pickup_error.abs().max()),service_error_s=float(service_error.abs().max()),
        routing_failures=c.eta_adapter.routing_failures,routing_arcs=c.eta_adapter.routing_arc_evaluations,
        runtime_s=time.monotonic()-start,peak_rss_mib=getattr(mem,"peak_wset",mem.rss)/2**20,
        checkpoint_sha256=checkpoint_sha,policy_table_sha256=sha(table_path),drain_end_s=tick,
        cohort_eligible_unknown_share_mean=float(outcome.research_unknown_share.mean()))
    write_json(dest/"summary.json",result)
    return result


def run(root, fleetpy_root):
    cfg = json.loads((root/CONFIG).read_text())
    load_fleetpy_bindings(fleetpy_root)
    assert json.loads((root/DOC/"offline_summary.json").read_text())["status"] == "PASS"
    output = root/OUT; output.mkdir(parents=True,exist_ok=False)
    protected = [root/CONFIG,root/"stage3/config/stage3_av_capability_profiles.json",root/"stage2/output_v5_2/development/M3/epoch_004.pt"]
    before = {str(p):sha(p) for p in protected}
    summary = dict(status="RUNNING",protocol_commit="e13325d",protected_sha256=before,rows=[])
    for mode in cfg["dynamic_variants"]:
        dest=output/mode; dest.mkdir()
        summary["active"]=mode; write_json(output/"summary.json",summary)
        try:
            result=condition(root,cfg,mode,dest)
        except Exception as error:
            summary.update(status="STOPPED",error=repr(error))
            write_json(output/"summary.json",summary)
            raise
        summary["rows"].append(result); write_json(output/"summary.json",summary)
        print(json.dumps(result),flush=True)
        gc.collect()
    a=pd.read_parquet(output/"FROZEN/cohort_outcomes.parquet")
    b=pd.read_parquet(output/f"{MODE}/cohort_outcomes.parquet")
    assert set(a.order_id)==set(b.order_id)
    assert before=={str(p):sha(p) for p in protected}
    summary.update(status="COMPLETE",active=None,matched_difference=int(b.matched.sum()-a.matched.sum()))
    write_json(output/"summary.json",summary); write_json(root/DOC/"window_summary.json",summary)


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--fleetpy-root",type=Path,required=True)
    run(Path.cwd(),p.parse_args().fleetpy_root)
