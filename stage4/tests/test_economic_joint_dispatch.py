"""Economic mechanism + native seams on tiny inputs; no replay/actor/inference."""
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from time import perf_counter
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from stage4.dispatch.operating_economics import (OperatingEconomics,OperatingLedger,MODE,
    ORDER_LEDGER_COLUMNS,RELOCATION_LEDGER_COLUMNS)
from stage4.dispatch.economic_inputs import EconomicDistanceStore,aggregate_distances
from stage4.dispatch.bottleneck_joint import JointProblem,JointPlan,seed_plan,solve_block,JointBudgetStop
from stage4.dispatch.bottleneck_scene import PredictionSceneOracle
from stage4.dispatch.bottleneck_plan_index import PlanIndex
from stage4.dispatch.bottleneck_efficient import EfficientWorkspace
from stage4.dispatch.bottleneck_incremental import RollingJointMemory
from stage4.dispatch.scheme_a_city_graph import ChainTask,RouteState,CoarseHistoricalLibrary
from stage4.dispatch.scheme_a_city_master import CurrentAction
from stage4.dispatch.scheme_a_current_flow import solve_current_flow
from stage4.dispatch.candidate_graph import SparseCandidateIndex,SpatialVehicle
from stage4.dispatch.c_connection_support import SharedCConnectionSupport


POINT=(108.9,34.2)
MODEL=OperatingEconomics(8.,2.,.25,.67,0.)
PACE=NS(global_pace=.01,pace_by_slot={})


def task(rid,release=300.,duration=30.,distance=1000.,allowed=("C","HV"),pickup=POINT,dropoff=POINT):
    return ChainTask(rid,release,release+300.,duration,pickup,dropoff,frozenset(allowed),
        source_date="20161010" if rid<0 else "20161031",source_order_id=str(rid),
        source_kind="FORECAST" if rid<0 else "ACTUAL_PENDING",service_distance_m=distance)


def money_problem():
    current=task(11,0.,400.)
    actions=[CurrentAction("HW",1,"WAIT",None),CurrentAction("CW",2,"WAIT",None),
        CurrentAction("HS",1,"SERVE",11,False,True,100.),CurrentAction("CS",2,"SERVE",11,False,True,200.)]
    roots={a.action_id:RouteState(POINT,400. if a.kind=="SERVE" else 30.,1800.,
        "HV" if a.vehicle_id==1 else "C",dict(kind="CUSTOMER",task=current) if a.kind=="SERVE" else None)
        for a in actions}
    scenes={sid:(task(-1,100.,30.,2000.,("HV",)),) for sid in ("A","B","C")}
    oracle=PredictionSceneOracle(actions,roots,scenes,(current,),PACE,0.,efficient=True,economics=MODEL)
    p=JointProblem(actions,oracle,{sid:Fraction(1,3) for sid in scenes},(11,),economics=MODEL)
    initial,_=seed_plan(p,("HS","CW"))
    for sid in p.weights:
        for a in actions:
            p.register(p.empty(sid,a))
            p.offer(sid,a.action_id)
    return p,initial


def test_cashflow_full_distance_break_even_and_single_settlement():
    t=task(11,0.,distance=3000.)
    c=MODEL.quote(t,"C",1000.); h=MODEL.quote(t,"HV",1000.)
    assert (c["fare_cents"],h["fare_cents"])==(1400,1400)
    assert h["driver_payout_cents"]==1050 and h["variable_cost_cents"]==0
    assert c["contribution_cents"]==1132>h["contribution_cents"]==350
    assert MODEL.quote(t,"C",20000.)["contribution_cents"]<h["contribution_cents"]
    ledger=OperatingLedger(); ledger.commit(11,1,"order",c); ledger.commit(11,1,"order",c)
    assert ledger.summary({11})["completed_orders"]==1
    assert ledger.summary(set())["operating_contribution_cny"]==0
    with pytest.raises(ValueError,match="two financial"):
        ledger.commit(11,2,"order",h)
    ledger.move(1,900,"C",1000.,MODEL)
    ledger.settle_moves([dict(native_vehicle_id=1,start_time_s=900,vehicle_type="AV",
        actual_distance_m=200.,status="INTERRUPTED")],MODEL)
    assert ledger.summary({11})["relocation_cost_cny"]==.13
    assert set(pd.DataFrame(ledger.orders.values(),columns=ORDER_LEDGER_COLUMNS))==set(ledger.orders[11])
    assert tuple(pd.DataFrame([],columns=RELOCATION_LEDGER_COLUMNS))==RELOCATION_LEDGER_COLUMNS
    with pytest.raises(ValueError): MODEL.quote(replace(t,service_distance_m=None),"C",0.)


def test_immediate_money_is_after_service_guarantees_not_an_AV_priority():
    t=task(11,0.)
    actions=[CurrentAction("HW",1,"WAIT",None),CurrentAction("CW",2,"WAIT",None),
        CurrentAction("HS",1,"SERVE",11,False,True,0.),CurrentAction("CS",2,"SERVE",11,False,True,300.)]
    roots={a.action_id:RouteState(POINT,30.,1800.,"HV" if a.vehicle_id==1 else "C",dict(task=t)) for a in actions}
    def solve(cost):
        return solve_current_flow(actions,{a.action_id:0 for a in actions},backend="INTEGER_REFERENCE",
            operating_contribution_units={a.action_id:cost.action_cents(a,roots[a.action_id]) for a in actions})
    assert "CS" in solve(MODEL)["selected_action_ids"]
    # A sufficiently costly AV can lose. No type bonus or idle penalty exists.
    assert "HS" in solve(OperatingEconomics(8.,2.,.25,10.,0.))["selected_action_ids"]
    assert solve(MODEL)["current_served"]==solve(MODEL)["carry_over_current_served"]==1


def test_joint_money_shared_customers_service_floors_and_incremental_delta():
    p,initial=money_problem()
    plan,info=solve_block(p,initial,{1,2},perf_counter()+5)
    checked=p.evaluate(plan)
    assert set(plan.action_ids)=={"HW","CS"}
    assert checked["current_served"]==checked["current_carry"]==1
    assert checked["expected_future_served"]==1  # NOT three actual customers.
    assert checked["score"][0]>p.evaluate(initial)["score"][0]
    index=PlanIndex(p,initial)
    patch=index.propose({1,2},plan.action_ids,plan.chains); index.apply(patch)
    assert index.score==checked["score"] and index.view()["expected_future_served"]==1
    with pytest.raises(ValueError,match="protected current"):
        p.evaluate(JointPlan(("HW","CW"),tuple(p.empty(s,p.actions[a]) for s in p.weights for a in ("HW","CW"))))
    duplicate=p.oracle.query("A","HW",required=11)[0]; p.register(duplicate)
    broken=replace(plan,chains=tuple(duplicate if c.scenario_id=="A" and c.vehicle_id==1 else c for c in plan.chains))
    with pytest.raises(ValueError,match="allocated twice"):
        p.evaluate(broken)
    from stage4.dispatch.economic_price_guide import restricted_prices
    before=p.evaluate(plan)["score"]
    prices,receipt=restricted_prices(p,plan,perf_counter()+5,lambda:None)
    assert receipt["global_bound_claimed"] is False and all(v>=0 for v in prices.values())
    assert p.evaluate(plan)["score"]==before


def test_chain_ranking_is_profit_not_count_and_keeps_negative_prefix_opportunities():
    a=CurrentAction("W",1,"WAIT",None)
    roots={"W":RouteState(POINT,30.,1800.,"HV")}
    tasks=(task(-1,300.),task(-2,600.),task(-3,900.),task(-4,300.,1200.,20000.))
    oracle=PredictionSceneOracle([a],roots,{"S":tasks},(),PACE,0.,efficient=True,economics=MODEL)
    best=oracle.query("S","W")[0]
    assert best.request_ids==(-4,)  # One high-contribution trip beats three low ones.
    answers=oracle.query_many("S","W",{-1,-4})
    assert answers[None][0].request_ids==(-4,)
    far=(108.94,34.2)
    cmodel=OperatingEconomics(8.,0.,.25,3.,0.)
    tasks=(task(-1,300.,30.,5000.,dropoff=far),task(-2,600.,30.,100.,pickup=far))
    oracle=PredictionSceneOracle([a],{"W":replace(roots["W"],profile_id="C")},
        {"S":tasks},(),PACE,0.,efficient=True,economics=cmodel)
    best=oracle.query("S","W")[0]
    assert best.request_ids==(-1,-2) and best.payload["operating_contribution_cents"]==70


def test_both_future_types_are_cache_only_and_block_budget_is_shared(tmp_path,monkeypatch):
    class Connector:
        def __init__(self): self.counts=Counter(); self.calls=[]
        def __call__(self,state,t,departure,profile,**kwargs):
            assert kwargs["cache_only"] is True
            self.calls.append(profile)
            return dict(supported=True,travel_time_s=10. if profile=="C" else 20.,
                empty_distance_m=100. if profile=="C" else 200.,reason_codes=[])
    connector=Connector(); support=SharedCConnectionSupport(connector,None,dict(
        joint_c_connection_mode="BOUNDED_CERTIFIED_C_V1",economic_future_connections="BOTH_CACHE_CERTIFIED_V1"))
    t=task(-1)
    for profile in ("C","HV"):
        assert support.lookup(RouteState(POINT,30.,1800.,profile),t,300.)["supported"]
    assert connector.calls==["C","HV"]
    # Exercise the real connector boundary too: HV prefetch may not bypass
    # the same two-extra-query allowance used by C recovery/future support.
    from stage4.tests.test_c_connection_support import connector_case
    from stage4.tests.test_scheme_a_city_connections import task as connector_task,ORIGIN
    con,router=connector_case(tmp_path,monkeypatch)
    con.bounded_all_profile_future=True
    con.train_pairs.update(("20161010",str(i)) for i in range(3))
    source=RouteState(ORIGIN,61200.,90000.,"HV")
    targets=[connector_task(str(i)) for i in range(3)]
    for i,target in enumerate(targets): target.pickup=(108.901+.0001*i,34.2)
    results=[con(source,target,61200.,"HV",query_scope="FUTURE",
        enforce_pickup_deadline=False) for target in targets]
    assert all(r["supported"] for r in results[:2])
    assert results[2]["reason_codes"]==["HV_CONNECTION_NOT_EVALUATED_BUDGET"]
    assert len(router.calls)==2 and con.c_connection_budget.epoch_calls==2
    p,initial=money_problem(); p.economic_budget["blocks"]=3
    w=EfficientWorkspace(p,PlanIndex(p,initial),lambda:None)
    with pytest.raises(JointBudgetStop,match="THREE_BLOCK"):
        w.evaluate({1,2},{"A":{11}},perf_counter()+5)
    memory=RollingJointMemory(efficient=True)
    assert memory.fork().economic_budget is memory.economic_budget


def test_type_retention_preserves_old_topk_and_never_adds_ineligible_AV():
    hv=[SpatialVehicle(f"H{i}",i,"HV",108.9+.00001*i,34.2) for i in range(1,25)]
    av=[SpatialVehicle(f"C{i}",100+i,"AV",108.901+.00001*i,34.2) for i in range(1,5)]
    index=SparseCandidateIndex([*hv,*av],cached_geometry=True)
    old,_=index.query(*POINT,2000.,20,True)
    new,_=index.query(*POINT,2000.,20,True,extra_av_top_k=3)
    assert len(new)<=23 and {v.native_vehicle_id for v,_ in old}<={v.native_vehicle_id for v,_ in new}
    assert sum(v.vehicle_type=="AV" for v,_ in new)==3
    refused,_=index.query(*POINT,2000.,20,False,extra_av_top_k=3)
    assert all(v.vehicle_type=="HV" for v,_ in refused)


def test_streamed_distances_are_full_route_parts_and_history_rejects_target_day(tmp_path):
    frame=pd.DataFrame(dict(order_id=["x","x","y"],route_part_length_m=[100.,250.,200.]))
    source=tmp_path/"parts.parquet"; frame.to_parquet(source,index=False,row_group_size=1)
    rows,count=aggregate_distances(source,"20161010",{"x"})
    assert count==2 and rows==[dict(date="20161010",order_id="x",service_distance_m=350.)]
    store=EconomicDistanceStore(pd.DataFrame(rows))
    attached=store.attach_history(pd.DataFrame(dict(date=["20161010"],order_id=["x"])))
    assert attached.service_distance_m.iloc[0]==350.
    with pytest.raises(ValueError,match="target-day"):
        store.attach_history(pd.DataFrame(dict(date=["20161031"],order_id=["x"])))
    with pytest.raises(ValueError,match="incomplete"):
        aggregate_distances(source,"20161010",{"missing"})


@pytest.mark.parametrize("controller,sparse",(("MYOPIC",False),("SPATIAL",False),("SUBSTITUTION",False),("SUBSTITUTION",True)))
def test_native_uses_same_economics_and_books_only_actual_commit(monkeypatch,controller,sparse):
    from stage4.dispatch import scheme_a_two_scale_native as native
    from stage4.tests.test_scheme_a_two_scale_native import fixture,Value
    from stage4.dispatch.solver import AssignmentArc
    c,cfg=fixture(); c.assignment_rows=[]; c.completed_rids=set(); c.sim_time=930.
    c.repositioning_manager.active.clear(); c.repositioning_manager.rows=[]
    cfg.update(current_decision_mode="BOTTLENECK_JOINT",joint_execution_mode="INCREMENTAL_V3",
        joint_c_connection_mode="BOUNDED_CERTIFIED_C_V1",economic_future_connections="BOTH_CACHE_CERTIFIED_V1",
        economic_controller=controller,joint_neighborhood_strategy="SPATIAL_JOINT" if controller=="SPATIAL" else "BOTTLENECK_CLOSURE",
        dispatch_economics=dict(mode=MODE,
            **{k:getattr(MODEL,k) for k in MODEL.__dataclass_fields__}))
    if sparse:
        cfg.update(economic_sparse_connections="SHARED_SPARSE_ROAD_SUPPORT_V1",sparse_connection_namespace="test-native")
    store=EconomicDistanceStore(pd.DataFrame([dict(date="20161031",order_id="current",service_distance_m=3000.)]))
    value=Value(); value.library=NS(view=lambda now:{"S":()}); value.pace=PACE
    value.library.weights={"S":1}
    class NoBackend:
        def __init__(self):
            self.counts=Counter(); self.router=NS(route_call_count=0)
            self.control=NS(_timestamp=lambda t:t,eta_adapter=NS(beta_for=lambda t:(0,1.)))
        def __call__(self,*args,**kwargs):
            if controller=="MYOPIC": raise AssertionError("no future routing in economic myopic control")
            return dict(supported=False,reason_codes=["NOT_EVALUATED_TEST_NO_BACKEND"])
    no_route=NoBackend()
    def validator(control,arcs,now): return arcs
    monkeypatch.setattr(native,"solve_current_flow",lambda actions,units,**kw:
        solve_current_flow(actions,units,backend="INTEGER_REFERENCE",**kw))
    from stage4.dispatch import bottleneck_native
    monkeypatch.setattr(bottleneck_native,"predicted_vehicle_states",lambda *args:())
    adapter=native.NativeTwoScaleAdapter("CHAIN_DEFER",value,None,no_route,validator,None,cfg,economic_inputs=store)
    arcs=[AssignmentArc(vid,11,eta,False,True,(c.runtime_by_vid[vid],c.request_by_rid[11],NS(route_distance_m=dist)),kind)
        for vid,eta,dist,kind in ((1,60.,400.,"AV"),(2,10.,100.,"HV"))]
    result=adapter.solve(c,arcs,[11],930.)
    assert result.selected_indices==(0,) and adapter.economic_ledger.orders=={}
    row=adapter.rows[-1]
    assert row["current_carry_served"]==1 and row["current_critical_served"]==0
    assert row["economic_config_sha256"]==MODEL.fingerprint
    assert row["current_operating_contribution_cny"]==MODEL.quote(NS(service_distance_m=3000.),"C",400.)["contribution_cents"]/100
    c.assignment_rows.append(dict(simulation_time_s=930.,native_vehicle_id=1,native_request_id=11,
        pickup_eta_s=60.,predicted_service_time_s=120.))
    c.completed_rids.add(11)
    report=adapter.economic_report(c)
    assert report["completed_orders"]==1 and report["by_profile"]["C"]["completed_orders"]==1
    assert report["rolling_planning_values_added_to_realized_ledger"] is False
    assert adapter.economic_report(c)==report
