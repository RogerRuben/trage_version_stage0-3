"""Bounded algorithm/forecast/query QA, never a new native scientific scenario."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time


def sparse_case(nvehicles, ncurrent, nfuture):
    import numpy as np
    from stage4.dispatch.flexibility_model import Vehicle, Request, CurrentPickup, FuturePickup, Scenario, Problem, ModelLimits
    rng = np.random.default_rng(20261005)
    vehicles = tuple(Vehicle(v, "HV" if v % 2 else "M", f"V{v}", 0, 1800) for v in range(nvehicles))
    requests = tuple(Request(1000+r, -280 if r % 7 == 0 else -60 if r % 3 == 0 else 0,
        20 if r % 7 == 0 else 240 if r % 3 == 0 else 300,
        120 if r % 2 else 360, f"D{r}", passenger_accepts_av=r % 5 != 0,
        critical=r % 7 == 0, carry_over=r % 3 == 0 or r % 7 == 0) for r in range(ncurrent))
    current = tuple(CurrentPickup(int(v), r.request_id, float(rng.integers(0, 40)))
                    for r in requests for v in rng.choice(nvehicles, min(4,nvehicles), replace=False))
    states = {v.vehicle_id:[(None,v.ready_position)] for v in vehicles}
    for arc in current:
        states[arc.vehicle_id].append((arc.request_id, requests[arc.request_id-1000].dropoff_position))
    scenarios = []
    for s in range(3):
        future = tuple(Request(10000+s*1000+r, 90+r%4*30, 390+r%4*30, 60, f"F{s}_{r}",
                               passenger_accepts_av=r%4!=0) for r in range(nfuture))
        arcs = tuple(FuturePickup(int(v), after, r.request_id, float(rng.integers(0,90)), origin)
                     for r in future for v in rng.choice(nvehicles,min(5,nvehicles),replace=False)
                     for after,origin in states[int(v)])
        scenarios.append(Scenario(f"S{s}",1/3,-86400,future,arcs))
    return Problem(0,vehicles,requests,current,tuple(scenarios),ModelLimits(solver_time_limit_s=10))


def vector(problem, decision):
    requests = {r.request_id:r for r in problem.waiting_requests}
    selected = set(decision.selected_pairs)
    return [sum(requests[r].critical for _,r in selected), len(selected),
            sum(requests[r].carry_over for _,r in selected), decision.expected_next_service_count,
            -sum(a.pickup_eta_s for a in problem.current_pickups if (a.vehicle_id,a.request_id) in selected)]


def benchmark_math(root):
    import numpy as np
    from stage4.dispatch.flexibility_model import solve_dispatch
    from stage4.dispatch.persistent_highs import load_highspy
    runtime = root / "stage4/output/runtime_dependencies/highspy_1_12_0"
    highspy = load_highspy(runtime)
    cases = []
    for dimensions in ((12,8,12),(36,24,30)):
        problem = sparse_case(*dimensions)
        rows = []
        for mode,backend,lock in (("BINARY","SCIPY",False),("FLOW_RELAXED","SCIPY",False),
                                  ("FLOW_RELAXED","SCIPY",True),("FLOW_RELAXED","HIGHS_PERSISTENT",True)):
            changed = replace(problem,limits=replace(problem.limits,recourse_mode=mode,
                solver_backend=backend,highspy_runtime_dir=str(runtime),lock_current_face=lock))
            started = time.perf_counter()
            try:
                d = solve_dispatch(changed,"SERVICE_PRESERVING_LOOKAHEAD")
                row = dict(status="OPTIMAL",mode=mode,backend=backend,lock_current_face=lock,wall_s=time.perf_counter()-started,
                    objective_vector=vector(changed,d),variables=d.variable_count,integer_variables=d.integer_variable_count,
                    nonzeros=d.constraint_nonzeros,matrix_bytes=d.sparse_matrix_bytes,
                    build_s=d.model_build_time_s,optimization_s=d.solve_time_s,recovery_s=d.recourse_recovery_time_s)
            except RuntimeError as error:
                if "solver timeout" not in str(error) and "not proven optimal" not in str(error): raise
                row = dict(status="NOT_PROVEN_WITHIN_BUDGET",mode=mode,backend=backend,lock_current_face=lock,
                           wall_s=time.perf_counter()-started,error=str(error))
            rows.append(row)
        completed = [r for r in rows if r["status"]=="OPTIMAL"]
        equivalent = len(completed)==len(rows) and all(np.allclose(completed[0]["objective_vector"],r["objective_vector"],rtol=0,atol=1e-6) for r in completed)
        if len(completed)>1 and not all(np.allclose(completed[0]["objective_vector"],r["objective_vector"],rtol=0,atol=1e-6) for r in completed):
            raise RuntimeError("completed model objective vectors disagree")
        cases.append(dict(dimensions=dimensions,all_variants_optimal_and_equivalent=equivalent,rows=rows))
    return dict(highs_version=highspy.Highs().version(),synthetic_not_native=True,cases=cases)


def benchmark_forecast(root,cfg):
    import pandas as pd
    from stage4.dispatch.flexibility_native import TrainDemandForecast
    frame = pd.read_parquet(root / "stage4/output/symmetric_flexibility_v1/input/train_request_templates.parquet")
    started = time.perf_counter()
    slow = TrainDemandForecast(frame,cfg,cfg["measurement_end_s"])
    slow_init = time.perf_counter()-started
    started = time.perf_counter()
    fast = TrainDemandForecast(frame,{**cfg,"fast_forecast":True},cfg["measurement_end_s"])
    fast_init = time.perf_counter()-started
    slow_s=fast_s=0.
    count=0
    for now in (0,18000,25200,32400,43200,54000,64800,75600):
        started=time.perf_counter(); a=slow.scenarios(now,"M",.7,20260827); slow_s+=time.perf_counter()-started
        started=time.perf_counter(); b=fast.scenarios(now,"M",.7,20260827); fast_s+=time.perf_counter()-started
        if a != b: raise RuntimeError("indexed forecast changed an RNG draw or request")
        count+=sum(len(s.new_requests) for s,_ in a)
    return dict(train_rows=len(frame),windows=8,generated_requests=count,exact_equal=True,
                legacy_initialization_s=slow_init,indexed_initialization_s=fast_init,
                legacy_sampling_s=slow_s,indexed_sampling_s=fast_s)


def benchmark_routing(root):
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree
    from stage4.dispatch.deterministic_routing import ArcDeterministicValhallaAdapter,SINGLE_SOURCE_MATRIX
    from stage4.dispatch.candidate_graph import SpatialVehicle
    frame=pd.read_parquet(root / "stage4/output/symmetric_flexibility_v1/input/train_request_templates.parquet").head(600)
    origins=frame[["end_lon_wgs84","end_lat_wgs84"]].to_numpy(float)
    targets=frame[["start_lon_wgs84","start_lat_wgs84"]].to_numpy(float)
    scale=np.array([111320*np.cos(np.deg2rad(34.25)),110540])
    tree=cKDTree(origins*scale)
    queries=[]
    for lon,lat in targets[:100]:
        ids=sorted(tree.query_ball_point(np.array([lon,lat])*scale,2000))[:20]
        if ids: queries.append(([SpatialVehicle(f"Q{i}",i,"HV",*origins[i]) for i in ids],lon,lat))
    timestamp=pd.Timestamp("2016-10-31T08:00:00+08:00")
    base=ArcDeterministicValhallaAdapter(root,routing_mode=SINGLE_SOURCE_MATRIX)
    fast=ArcDeterministicValhallaAdapter(root,routing_mode=SINGLE_SOURCE_MATRIX,route_workers=2,persistent_cache_size=50000)
    def run(adapter,t):
        started=time.perf_counter()
        values=[adapter.estimate_many(v,lon,lat,t) for v,lon,lat in queries]
        adapter.cache.clear()
        return values,time.perf_counter()-started
    try:
        first,base_cold=run(base,timestamp)
        second,fast_cold=run(fast,timestamp)
        repeated,fast_repeated=run(fast,timestamp+pd.Timedelta(seconds=30))
        for a,b,c in zip(first,second,repeated):
            if set(a)!=set(b) or set(a)!=set(c): raise RuntimeError("routing reachability changed")
            for key in a:
                fields=("valhalla_time_s","corrected_pickup_eta_s","route_distance_m","beta","time_bin_index")
                if any(getattr(a[key],f)!=getattr(b[key],f) or getattr(a[key],f)!=getattr(c[key],f) for f in fields):
                    raise RuntimeError("independent routing result changed")
        return dict(query_batches=len(queries),arc_lookups=sum(len(q[0]) for q in queries),exact_equal=True,
            legacy_cold_s=base_cold,parallel_cold_s=fast_cold,repeated_identical_within_minute_s=fast_repeated,
            persistent_cache_hits=fast.persistent_cache_hits,process_group_rss_mib=fast.process_group_rss_mib(),
            caveat="Cold parallel time includes worker startup. Repeated queries measure cache reuse, not real full-day hit rate.")
    finally:
        base.close();fast.close()


def main():
    import psutil
    from stage3.scripts.traffic_state_batch1 import write_json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-routing",action="store_true")
    args=parser.parse_args()
    root=Path.cwd()
    native=root / "stage4/output/symmetric_flexibility_v1/full_day/SERVICE_PRESERVING_LOOKAHEAD/summary.json"
    running_summary=native.exists() and json.loads(native.read_text())["status"]=="RUNNING"
    active=False
    for process in psutil.process_iter(["cmdline","cwd"]):
        try:
            command=" ".join(process.info["cmdline"] or [])
            if "stage4.analysis.symmetric_flexibility_full_day" in command and process.info["cwd"] and Path(process.info["cwd"]).resolve()==root.resolve():
                active=True
                break
        except (psutil.AccessDenied,psutil.NoSuchProcess):
            pass
    if args.with_routing and active:
        raise RuntimeError("defer real routing benchmark until the frozen native run ends")
    cfg=json.loads((root / "stage4/config/symmetric_flexibility_full_day_v1.json").read_text())
    started=time.perf_counter()
    result=dict(status="COMPLETE",native_scientific_conditions_added=0,dense_matrix=False,gpu_used=False,
                concurrent_native_run=active,stale_native_running_summary=running_summary and not active,
                math=benchmark_math(root),forecast=benchmark_forecast(root,cfg))
    if args.with_routing: result["routing"]=benchmark_routing(root)
    result.update(runtime_s=time.perf_counter()-started,peak_parent_rss_mib=getattr(psutil.Process().memory_info(),"peak_wset",psutil.Process().memory_info().rss)/2**20)
    destination=root / "stage4/docs/flexibility_dispatch/acceleration_v1/benchmark_summary.json"
    destination.parent.mkdir(parents=True,exist_ok=True)
    write_json(destination,result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__": main()
