"""Bounded V4 execution contracts, not a parameter study or city experiment."""
from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from stage4.dispatch.bottleneck_lazy_moves import solve_lazy_moves
from stage4.dispatch.bottleneck_incremental import RollingJointMemory
from stage4.dispatch.bottleneck_joint import JointProblem
from stage4.dispatch.bottleneck_scene import PredictionSceneOracle
from stage4.dispatch.scheme_a_city_graph import ChainTask,RouteState
from stage4.dispatch.scheme_a_city_master import CurrentAction
from stage4.tests.test_scheme_a_two_scale_native import fixture,Value,ORIGIN


def test_lazy_routes_selected_coalition_only_and_replans_exact_arrivals(monkeypatch):
    from stage4.dispatch import bottleneck_lazy_moves as lazy
    clock=[1000.]; monkeypatch.setattr(lazy,"perf_counter",lambda:clock[0])
    waits=[CurrentAction(f"W{v}",v,"WAIT",None) for v in range(1,6)]
    moves=[CurrentAction(f"M{v}",v,"RELOCATE",None,
        payload=dict(optimistic_move=True,relocation_slot_s=0)) for v in range(1,6)]
    roots={a.action_id:RouteState(ORIGIN,0.,1800.,"C") for a in (*waits,*moves)}
    baseline=dict(selected_action_ids=tuple(a.action_id for a in waits))
    memory=RollingJointMemory(efficient=True); calls=[]
    def plan(actions,after,seed,deadline,state,cap):
        calls.append((deadline,cap)); state.last_now=123.
        if len(calls)==1:
            assert len(actions)==10
            return dict(selected_action_ids=("M1","M2","W3","W4","W5"),
                joint_control=dict(resource_fallback=False,planning_labels=7,phase_timings_s={"search_total_time_s":1.}))
        assert {a.action_id for a in actions}=={"W1","W2","W3","W4","W5","M1","M2"}
        assert after["M1"].ready_s==120. and after["M2"].ready_s==160.
        return dict(selected_action_ids=("M1","M2","W3","W4","W5"),
            joint_control=dict(resource_fallback=False,planning_labels=3,phase_timings_s={"search_total_time_s":2.}))
    def certify(action):
        assert memory.last_now is None  # Probe paths have not leaked into execution.
        return replace(action,empty_distance_m=400.,payload=dict(
            move={"routed":{"physical":True}},relocation_slot_s=0)),replace(
                roots[action.action_id],ready_s=120. if action.vehicle_id==1 else 160.)
    result,actions,after,counts=solve_lazy_moves([*waits,*moves],roots,baseline,
        planner=plan,certify=certify,deadline=1010.,memory=memory,maximum_labels=10)
    assert counts["route_calls"]==counts["certified_moves"]==2
    assert calls==[(1004.,10),(1010.,3)]
    assert result["joint_control"]["planning_labels"]==10
    assert result["joint_control"]["phase_timings_s"]["search_total_time_s"]==3.
    assert memory.last_now==123. and all(not a.payload.get("optimistic_move")
        for a in actions if a.kind=="RELOCATE")


def test_late_failed_cell_keeps_earlier_certified_move_and_refinement_timeout_keeps_baseline(monkeypatch):
    from stage4.dispatch import bottleneck_lazy_moves as lazy
    clock=[0.]; monkeypatch.setattr(lazy,"perf_counter",lambda:clock[0])
    waits=[CurrentAction(f"W{v}",v,"WAIT",None) for v in (1,2,3)]
    moves=[CurrentAction(f"M{v}",v,"RELOCATE",None,payload=dict(
        optimistic_move=True,relocation_slot_s=0)) for v in (1,2,3)]
    roots={a.action_id:RouteState(ORIGIN,0.,1800.,"C") for a in (*waits,*moves)}
    baseline=dict(selected_action_ids=("W1","W2","W3")); calls=[]; memory=RollingJointMemory()
    def plan(actions,after,seed,deadline,state,cap):
        calls.append(1); state.last_now=99.
        if len(calls)==1:
            clock[0]=2.
            return dict(selected_action_ids=("M1","M2","M3"),joint_control={"resource_fallback":False})
        assert {a.action_id for a in actions}=={"W1","W2","W3","M1"}
        return dict(baseline,joint_control={"resource_fallback":True})
    def certify(action):
        if action.action_id=="M2":
            clock[0]=7.5; return None
        return replace(action,payload={"move":{"routed":{}},"relocation_slot_s":0}),roots[action.action_id]
    result,actions,_,counts=solve_lazy_moves([*waits,*moves],roots,baseline,planner=plan,
        certify=certify,deadline=10.,memory=memory)
    assert counts["route_calls"]==2 and counts["budget_skipped_selected"]==1
    assert any(a.action_id=="M1" for a in actions)  # Not an all-or-nothing wipe.
    assert result["selected_action_ids"]==baseline["selected_action_ids"] and memory.last_now is None


def test_forecast_warm_reuse_is_exact_and_invalidates_changed_ready_bin_or_pending(monkeypatch):
    point=ORIGIN; action=CurrentAction("W",1,"WAIT",None)
    future=ChainTask(-1,300.,600.,30.,point,point,{"HV","C"},source_date="20161010")
    memory=RollingJointMemory(efficient=True,exact_warm_reuse=True)
    pace=NS(global_pace=.01,pace_by_slot={})
    def problem(now,ready,pending=()):
        q=PredictionSceneOracle([action],{"W":RouteState(point,ready,1800.,"C")},
            {"S":(future,)},pending,pace,now,topology=memory.topology,
            forecast_snapshot={"S":(future,)},efficient=True)
        return JointProblem([action],q,{"S":1},[t.request_id for t in pending])
    first=problem(0.,30.); initial,_,_=memory.initial(first,("W",),lambda:None,perf_counter()+2)
    memory.save(first,initial,initial,(),0.)
    p=problem(30.,60.); calls=[]; original=p.oracle.repair
    monkeypatch.setattr(p.oracle,"repair",lambda *a:(calls.append(a),original(*a))[1])
    kept,info,_=memory.initial(p,("W",),lambda:None,perf_counter()+2)
    assert info["warm_exact_reused_continuations"]==1 and calls==[]
    assert kept.chains[0].payload==initial.chains[0].payload
    p=problem(60.,301.); changed,info,_=memory.initial(p,("W",),lambda:None,perf_counter()+2)
    assert not info.get("warm_exact_reused_continuations") and info["warm_retime_calls"]==1
    assert p.evaluate(changed)["expected_future_served"]==0
    pending=ChainTask(11,0.,120.,30.,point,point,{"HV","C"},source_kind="ACTUAL_PENDING")
    p=problem(0.,30.,(pending,)); initial,_,_=memory.initial(p,("W",),lambda:None,perf_counter()+2)
    # Build a pending-only selected continuation and ensure it is NOT reused.
    chain=p.oracle.repair("S","W",(11,),frozenset(),lambda:None); p.register(chain)
    initial=replace(initial,chains=(chain,)); memory.save(p,initial,initial,(),0.)
    p=problem(60.,90.,(pending,)); kept,info,_=memory.initial(p,("W",),lambda:None,perf_counter()+2)
    assert not info.get("warm_exact_reused_continuations")
    assert kept.chains[0].payload["legs"][0][1]==90.


def test_adaptive_joint_native_executes_only_certified_selected_moves(monkeypatch,tmp_path):
    from stage4.dispatch import scheme_a_two_scale_native as native,bottleneck_native
    c,cfg=fixture(); c.repositioning_manager.active.clear()
    cfg.update(current_decision_mode="BOTTLENECK_JOINT",joint_execution_mode="INCREMENTAL_V3",
        joint_neighborhood_strategy="ADAPTIVE_SHARED",joint_move_validation_mode="LAZY_SELECTED_V4",
        joint_exact_warm_reuse=True,idle_movement_mode="VALUE_SELECTED")
    value=Value(); point=(108.932,34.2)
    tasks=tuple(ChainTask(-i,float(t),float(t+300),30.,point,point,{"C","HV"},
        source_date="20161010") for i,t in enumerate((1200,1500,1800),1))
    value.library=NS(view=lambda now:{"S":tasks},weights={"S":1})
    value.pace=NS(global_pace=.01,pace_by_slot={}); routed=[]
    def connector(state,target,now,profile):
        routed.append(profile)
        return dict(supported=True,travel_time_s=100.,empty_distance_m=100.,
            routed={"physical":True},arrival_context={"kind":"EMPTY_ARRIVAL"})
    monkeypatch.setattr(native,"sites_for",lambda *args:[dict(site_id="site",node_id=1,position=(108.915,34.2))])
    monkeypatch.setattr(bottleneck_native,"predicted_vehicle_states",lambda *args:())
    adapter=native.NativeTwoScaleAdapter("SERVICE_PRESERVING",value,None,connector,
        lambda *args:[],None,cfg)
    result=adapter.solve(c,[],[],900.)
    row=adapter.rows[-1]; moves=c.repositioning_manager.queued[-1][0]
    assert result.selected_indices==() and moves and len(moves)==len(routed)==1
    assert row["current_move_proposed_count"]==2 and row["current_move_routed_count"]==1
    assert all(move["routed"]["physical"] for move in moves)
    assert row["joint_control"]["lazy_moves"]["planning_rounds"]==2
    assert row["joint_control"]["future_directed_road_certificate"] is False
    pd.DataFrame(adapter.rows).to_parquet(tmp_path/"trace.parquet",index=False)


def test_C_uses_typed_historical_sites_and_HV_keeps_common_sites(monkeypatch):
    from stage4.dispatch import scheme_a_two_scale_native as native,bottleneck_native
    c,cfg=fixture(); c.repositioning_manager.active.clear()
    cfg.update(current_decision_mode="BOTTLENECK_JOINT",joint_execution_mode="INCREMENTAL_V3",
        joint_neighborhood_strategy="ADAPTIVE_SHARED",joint_move_validation_mode="LAZY_SELECTED_V4",
        joint_move_target_mode="PROFILE_HISTORY",idle_movement_mode="VALUE_SELECTED")
    value=Value(); value.library=NS(view=lambda now:{"S":()},weights={"S":1})
    value.pace=NS(global_pace=.01,pace_by_slot={})
    common=pd.DataFrame(dict(time_bin_index=[1],node_id=[1],demand_share=[1.],
        lon_wgs84=[108.901],lat_wgs84=[34.2]))
    typed=common.assign(profile_id="C",node_id=2,lon_wgs84=108.902)
    seen=[]; original=native.sites_for
    def sites(reference,now,cfg):
        seen.extend(reference.node_id.tolist()); return original(reference,now,cfg)
    monkeypatch.setattr(native,"sites_for",sites)
    monkeypatch.setattr(bottleneck_native,"predicted_vehicle_states",lambda *args:())
    adapter=native.NativeTwoScaleAdapter("SERVICE_PRESERVING",value,common,
        lambda *args:pytest.fail("no forecast gain: optional routes must not be queried"),
        lambda *args:[],None,cfg,typed_reference=typed)
    adapter.solve(c,[],[],900.)
    assert sorted(seen)==[1,2] and adapter.rows[-1]["current_move_routed_count"]==0
