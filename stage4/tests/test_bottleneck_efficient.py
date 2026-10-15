"""Focused V3 correctness/work-count contracts; no city replay or dataset."""
from fractions import Fraction
from time import perf_counter
from types import SimpleNamespace as NS

import pytest

from stage4.dispatch.bottleneck_joint import JointPlan,JointProblem,solve_block
from stage4.dispatch.bottleneck_plan_index import PlanIndex
from stage4.dispatch.bottleneck_efficient import EfficientWorkspace,efficient_search
from stage4.dispatch.bottleneck_incremental import RollingJointMemory
from stage4.dispatch.bottleneck_scene import PredictionSceneOracle
from stage4.dispatch.bottleneck_topology import ForecastTopology
from stage4.dispatch.scheme_a_city_graph import ChainTask,RouteState
from stage4.dispatch.scheme_a_city_master import CurrentAction
from stage4.tests.test_bottleneck_joint import complementary,populate,SuppliedOracle


def test_indexed_patch_preserves_shared_capacity_and_persistent_exact_tiers():
    p,initial=complementary(); populate(p); index=PlanIndex(p,initial)
    forbidden=next(c for c in p.oracle.columns.values() if c.action_id=="C1M")
    with pytest.raises(ValueError,match="outside"):
        index.propose({2},("C1M",),tuple(next(c for c in p.oracle.columns.values()
            if c.scenario_id==s and c.action_id=="C1M") for s in p.weights))
    patch,info=solve_block(p,None,{1,2,3},perf_counter()+5,plan_index=index,persistent=True,
        skip_constant_stages=True)
    assert patch.score[0]==12
    index.apply(patch)
    checked=p.evaluate(index.materialize())
    assert checked["score"]==index.score and checked["owners"]==index.owners
    assert info["persistent_session"]["model_loads"]==1
    assert info["persistent_session"]["certified_binary_warm_starts"]>=1
    assert info["persistent_session"]["objective_faces_use_epsilon"] is False
    assert info["persistent_session"]["parallel_option"]=="off"


@pytest.mark.parametrize("now",(0.,60.,330.))
def test_batched_catalog_equals_single_target_queries_and_reuses_irrelevant_forbidden(now):
    point=(108.9,34.2); actions=[CurrentAction("W",1,"WAIT",None)]
    tasks=tuple(ChainTask(-i,float(t),float(t+300),30.,point,point,{"C","HV"},
        source_date="20161010") for i,t in enumerate((300,600,900,1200,1500),1))
    snapshot={"S":tuple(t for t in tasks if t.release_s>int(now//300)*300)}
    scenes={"S":tuple(t for t in snapshot["S"] if t.release_s>now)}
    root={"W":RouteState(point,now+30,2400.,"HV")}; pace=NS(global_pace=.01,pace_by_slot={})
    old=PredictionSceneOracle(actions,root,scenes,(),pace,now,topology=ForecastTopology(),forecast_snapshot=snapshot)
    new=PredictionSceneOracle(actions,root,scenes,(),pace,now,topology=ForecastTopology(),forecast_snapshot=snapshot,
        efficient=True)
    for blocked in (frozenset(),frozenset({-2})):
        answer=new.query_many("S","W",{-1,-2,-3,-4,-5},blocked)
        for required in (None,-1,-2,-3,-4,-5):
            expected=old.query("S","W",required=required,forbidden=blocked)
            assert [(c.request_ids,c.empty_distance_m,c.payload) for c in answer[required]]==[
                (c.request_ids,c.empty_distance_m,c.payload) for c in expected]
    labels=new.labels
    new.query_many("S","W",{-3},frozenset({-2,999999}))
    assert new.labels==labels and new.batch_reuses>0 and new.catalog_refs<=20000


def test_dependency_cache_ignores_unrelated_outside_change_but_not_shared_capacity():
    actions=[CurrentAction("A",1,"WAIT",None),CurrentAction("M",1,"RELOCATE",None,
        empty_distance_m=10.,payload={"relocation_slot_s":0}),CurrentAction("B",2,"WAIT",None)]
    q=SuppliedOracle(actions,{("S","A"):[(-2,),(-1,)],("S","M"):[(-3,)],
        ("S","B"):[(-99,),(-100,),(-1,)]})
    p=JointProblem(actions,q,{"S":1}); populate(p); p.protected_critical=0
    def chain(a,r): return next(c for c in q.columns.values() if c.action_id==a and c.request_ids==(r,))
    initial=JointPlan(("A","B"),(chain("A",-2),chain("B",-99)))
    index=PlanIndex(p,initial); w=EfficientWorkspace(p,index,lambda:None)
    w.evaluate({1},{"S":{-1}},perf_counter()+5)
    calls=w.metrics["solver_calls"]
    index.apply(index.propose({2},("B",),(chain("B",-100),)))
    assert w.evaluate({1},{"S":{-1}},perf_counter()+5) is None
    assert w.metrics["solver_calls"]==calls and w.counts["early_blocks_skipped"]==1
    index.apply(index.propose({2},("B",),(chain("B",-1),)))
    assert w.evaluate({1},{"S":{-1}},perf_counter()+5) is not None
    assert w.metrics["solver_calls"]>calls


def test_batched_closure_retains_neutral_combinations_and_avoids_full_scans_per_block(monkeypatch):
    p,initial=complementary(); populate(p); calls=[]; original=p.evaluate
    def evaluate(plan): calls.append(1); return original(plan)
    monkeypatch.setattr(p,"evaluate",evaluate)
    seeds=tuple(p.oracle.seeds(original(initial)["owners"]))
    best,info=efficient_search(p,initial,seeds,strategy="BOTTLENECK_CLOSURE",
        deadline=perf_counter()+5,check=lambda:None)
    assert original(best)["score"][0]==12 and set(best.action_ids)=={"H","C1M","C2M"}
    assert len(calls)==2  # Initial index and final independent verification only.
    assert info["search_work"]["groups_merged"]>0
    assert info["search_work"]["persistent_model_loads"]<info["search_work"]["solver_calls"]


def test_efficient_frontier_does_not_reactivate_own_future_rearrangement_or_backlog():
    point=(108.9,34.2); memory=RollingJointMemory(efficient=True)
    actions=[CurrentAction("W",1,"WAIT",None),CurrentAction("M",1,"RELOCATE",None,
        empty_distance_m=1.,payload={"relocation_slot_s":0})]
    task=ChainTask(-1,300.,600.,30.,point,point,{"C","HV"},source_date="20161010")
    roots={a.action_id:RouteState(point,30.,1800.,"HV") for a in actions}; snapshot={"S":(task,)}
    q=PredictionSceneOracle(actions,roots,snapshot,(),NS(global_pace=.01,pace_by_slot={}),0.,
        topology=memory.topology,forecast_snapshot=snapshot,efficient=True)
    p=JointProblem(actions,q,{"S":1})
    initial,_,_=memory.initial(p,("W",),lambda:None,perf_counter()+5)
    memory.save(p,initial,initial,(("S",-1),),0.)
    assert memory.touched==set()
    assert memory.frontier(p,initial,set(),lambda:None)==()


def test_performance_prefix_preserves_full_day_prediction_inputs_and_isolates_outputs(monkeypatch,tmp_path):
    from stage4.analysis import bottleneck_v3_preflight as prefix
    from stage4.analysis import bottleneck_whole_method as protocol
    cfg=dict(version=protocol.V3,full_day=True,measurement_end_s=86400,
        admission_end_s=86400,physical_drain_limit_s=97200,joint_execution_mode="INCREMENTAL_V3")
    monkeypatch.setattr(protocol,"load",lambda *args:(cfg,{},{}))
    monkeypatch.setattr(protocol,"group_config",lambda c,g:dict(c,group=g))
    monkeypatch.setattr(protocol,"binding_for",lambda *args:{"code_sha":"fixed"})
    def run(root,fleetpy,config,chosen,**kwargs):
        assert chosen["full_day"] and chosen["measurement_end_s"]==86400
        assert chosen["admission_end_s"]==86400 and chosen["diagnostic_stop_tick_s"]==2400
        assert "/preflight/" in chosen["disk_route_cache"] and "/native/" not in chosen["disk_route_cache"]
        assert kwargs["directory"].parent.name=="preflight"
        return {"status":"MOCK_PREFIX_ONLY"}
    monkeypatch.setattr(prefix,"_run_native",run)
    assert prefix.worker(tmp_path,protocol.ARMS[1])["status"]=="MOCK_PREFIX_ONLY"
    with pytest.raises(ValueError,match="two V3"):
        prefix.worker(tmp_path,protocol.ARMS[0])
