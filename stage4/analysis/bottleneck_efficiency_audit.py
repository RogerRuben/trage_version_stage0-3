"""Bounded synthetic profiling of existing joint code, NOT a city experiment.

No route calls, replay, historical data, predictions, parameter tuning or policy
changes. The only output is diagnostic aggregates. Timings are NOT city-speed
estimates; the profiler adds overhead. Run explicitly as a local code check.
"""
from collections import Counter
from contextlib import ExitStack
from fractions import Fraction
from functools import wraps
import inspect
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import psutil

from stage4.dispatch.bottleneck_incremental import BlockWorkspace,incremental_search
from stage4.dispatch.bottleneck_joint import JointBudgetStop,JointPlan,JointProblem,seed_plan
from stage4.dispatch.bottleneck_scene import PredictionSceneOracle
from stage4.dispatch.bottleneck_topology import ForecastTopology
from stage4.dispatch.scheme_a_city_graph import ChainTask,RouteState
from stage4.dispatch.scheme_a_city_master import CurrentAction,ServiceChain


POINT=(108.9,34.2)


def guard(deadline,counts,process=None):
    """Same deadline/RSS work as the native guard, with diagnostic call counts."""
    peak=0.
    def check():
        nonlocal peak
        counts["calls"]+=1
        if perf_counter()>=deadline:
            raise JointBudgetStop("bounded diagnostic deadline")
        counts["rss_reads"]+=1
        active=psutil.Process() if process is None else process
        peak=max(peak,active.memory_info().rss/2**20)
        if peak>512:
            raise JointBudgetStop("512-MiB diagnostic RSS limit")
    return check


def cache_hit_cost():
    result={}
    for mode in ("deadline_only","native_style_rss_guard","same_rss_check_reused_process"):
        counts=Counter(); deadline=perf_counter()+10
        def clock_check():
            counts["calls"]+=1
            if perf_counter()>=deadline:
                raise TimeoutError("bounded cache diagnostic")
        check=(clock_check if mode=="deadline_only" else
            guard(deadline,counts,psutil.Process() if mode.endswith("reused_process") else None))
        action=CurrentAction("W",1,"WAIT",None)
        task=ChainTask(-1,300.,600.,30.,POINT,POINT,{"C","HV"},source_date="20161010")
        q=PredictionSceneOracle([action],{"W":RouteState(POINT,30.,1800.,"HV")},
            {"S":(task,)},(),SimpleNamespace(global_pace=.01,pace_by_slot={}),0.,check=check)
        q.query("S","W",check=check)
        counts.clear(); samples=[]; labels=q.labels
        for _ in range(3):
            started=perf_counter()
            for _ in range(5000):
                q.query("S","W",check=check)
            samples.append(perf_counter()-started)
        result[mode]=dict(queries=15000,median_5000_queries_s=median(samples),
            sum_s=sum(samples),new_deep_labels=q.labels-labels,guard_counts=dict(counts))
    return result


def evaluate_cost():
    result={}
    for size in (6,60,170):
        actions=[CurrentAction(f"W{v}",v,"WAIT",None) for v in range(size)]
        p=JointProblem(actions,SimpleNamespace(accepts=lambda c:True),
            {s:Fraction(1,3) for s in ("S1","S2","S3")})
        chains=[]
        for si,sid in enumerate(p.weights):
            for v,a in enumerate(actions):
                tasks=tuple(-(1+si*size*3+v*3+j) for j in range(3))
                chains.append(p.register(ServiceChain(f"{sid}|{a.action_id}",sid,v,a.action_id,tasks)))
        plan=JointPlan(tuple(a.action_id for a in actions),tuple(chains))
        p.evaluate(plan); samples=[]
        for _ in range(3):
            started=perf_counter()
            for _ in range(300):
                p.evaluate(plan)
            samples.append(perf_counter()-started)
        result[str(size)]=dict(evaluations=900,median_300_evaluations_s=median(samples),
            median_single_evaluation_us=median(samples)/300*1e6,scenarios=3,
            selected_chains=len(chains),selected_request_visits=len(chains)*3)
    return result


def synthetic_problem(check):
    actions=[]; roots={}; selected=[]
    for v in range(1,25):
        profile="C" if v%2==0 else "HV"
        aid=f"V{v}:WAIT"; actions.append(CurrentAction(aid,v,"WAIT",None)); selected.append(aid)
        point=(POINT[0]+(v%6)*.0002,POINT[1])
        roots[aid]=RouteState(point,30.,2400.,profile)
        if v<=6:
            aid=f"V{v}:MOVE"
            actions.append(CurrentAction(aid,v,"RELOCATE",None,empty_distance_m=20.,
                payload={"relocation_slot_s":0}))
            roots[aid]=RouteState(POINT,60.,2400.,profile)
    scenes={sid:tuple(ChainTask(-1000*si-i-1,float(300+(i//6)*300),
        float(600+(i//6)*300),45.,POINT,POINT,
        {"HV"} if i%4==0 else {"HV","C"},source_date="20161010")
        for i in range(24)) for si,sid in enumerate(("S1","S2","S3"),1)}
    q=PredictionSceneOracle(actions,roots,scenes,(),
        SimpleNamespace(global_pace=.01,pace_by_slot={}),0.,check=check,
        topology=ForecastTopology(),forecast_snapshot=scenes)
    p=JointProblem(actions,q,{s:Fraction(1,3) for s in scenes})
    initial,_=seed_plan(p,selected,check)
    return p,initial


def profile_search(strategy):
    setup_counts=Counter(); setup_check=guard(perf_counter()+8,setup_counts)
    p,initial=synthetic_problem(setup_check)
    seeds=tuple(p.oracle.seeds(p.evaluate(initial)["owners"],setup_check))
    counts=Counter(); deadline=perf_counter()+3; check=guard(deadline,counts)
    p.oracle.check=check
    # Explicit wrappers keep counts across native solver calls. cProfile on
    # this installed Python/HiGHS combination omitted outer solve frames;
    # do not treat those incomplete profiler records as quantitative evidence.
    from stage4.dispatch import bottleneck_incremental as incremental
    from stage4.dispatch import bottleneck_joint as kernel
    from stage4.dispatch import bottleneck_scene as scene
    operations={}
    def wrapped(fn,label):
        entry=operations.setdefault(label,dict(calls=0,inclusive_s=0.))
        @wraps(fn)
        def call(*args,**kwargs):
            entry["calls"]+=1; entered=perf_counter()
            try:
                return fn(*args,**kwargs)
            finally:
                entry["inclusive_s"]+=perf_counter()-entered
        return call
    started=perf_counter()
    with ExitStack() as patches:
        for owner,name,label in ((JointProblem,"evaluate","full_plan_evaluate"),
                (BlockWorkspace,"populate","populate"),
                (incremental,"solve_block","solve_block"),
                (incremental,"close_coalition","close_coalition"),
                (kernel,"milp","scipy_milp"),
                (PredictionSceneOracle,"query","oracle_query"),
                (PredictionSceneOracle,"_leg","leg_validation"),
                (PredictionSceneOracle,"_rank","path_ranking"),
                (scene,"_xy","coordinate_projection")):
            original=getattr(owner,name); method=wrapped(original,label)
            if isinstance(inspect.getattr_static(owner,name),staticmethod):
                method=staticmethod(method)
            patches.enter_context(patch.object(owner,name,method))
        check=wrapped(check,"resource_guard"); p.oracle.check=check
        plan,info=incremental_search(p,initial,seeds,strategy=strategy,
            deadline=deadline,check=check)
    elapsed=perf_counter()-started
    checked=p.evaluate(plan)
    return dict(profiled_s=elapsed,resources=len(p.by_vehicle),scenarios=len(p.weights),
        seeds=len(seeds),score_before=list(p.evaluate(initial)["score"]),
        score_after=list(checked["score"]),partial_states=info["partial_states"],
        neighborhoods=len(info["neighborhoods"]),search_work=info["search_work"],
        accepted_coalitions=info["accepted_coalitions"],stop_reason=info["stop_reason"],
        labels=p.oracle.labels,oracle_query_calls=p.oracle.query_calls,
        guard_counts=dict(counts),setup_guard_counts=dict(setup_counts),
        core_operations=operations,timers_are_nested_do_not_sum=True,
        explicit_solver_counter_agrees=(operations.get("scipy_milp",{}).get("calls",0)
            ==info["search_work"]["solver_calls"]))


def main():
    started=perf_counter()
    result=dict(kind="BOUNDED_SYNTHETIC_CODE_DIAGNOSTICS_NOT_CITY_BENCHMARK",
        policy_implementation_changed=False,city_simulation_invoked=False,
        routing_invoked=False,inference_invoked=False,dense_matrix=False,gpu_used=False,
        cache_hit_cost=cache_hit_cost(),full_plan_evaluation_scaling=evaluate_cost(),
        profiles={s:profile_search(s) for s in ("SPATIAL_JOINT","BOTTLENECK_CLOSURE")})
    result.update(runtime_s=perf_counter()-started,
        peak_rss_mib=psutil.Process().memory_info().peak_wset/2**20,
        timings_include_instrumentation_overhead=True,city_speedup_inference_permitted=False)
    from stage4.analysis.scheme_a_two_scale_check import _atomic_json
    path=Path("stage4/docs/capability_chain_planning/bottleneck_efficiency_review_20261015/local_diagnostics.json")
    _atomic_json(path,result)
    print(json.dumps({k:v for k,v in result.items() if k!="profiles"}))
    for s,profile in result["profiles"].items():
        print(json.dumps({s:profile}))


if __name__=="__main__":
    main()
