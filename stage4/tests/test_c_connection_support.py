"""Small connection/search contracts. No native day, inference or new scenarios."""
from collections import Counter
from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from stage4.dispatch.c_connection_support import ConnectionBudget, SharedCConnectionSupport, MODE
from stage4.dispatch.bottleneck_scene import PredictionSceneOracle
from stage4.dispatch.bottleneck_joint import JointProblem
from stage4.dispatch.bottleneck_incremental import RollingJointMemory
from stage4.dispatch.scheme_a_city_connections import SchemeACityConnectors
from stage4.dispatch.scheme_a_city_graph import ChainTask, RouteState
from stage4.dispatch.scheme_a_city_master import CurrentAction
from stage4.dispatch.city_pickup_support import CityPickupSupportValidator
from stage4.tests.test_scheme_a_city_connections import control_and_evidence, task, ORIGIN, PICKUP, MIDNIGHT
from stage4.tests.test_city_pickup_support import case, Routes, AllowedJoin, ETA


def connector_case(tmp_path, monkeypatch):
    c, router, evidence = control_and_evidence(tmp_path)
    con = SchemeACityConnectors(c, router, NS(seam_evidence=evidence),
        pd.DataFrame(columns=["date", "order_id", "compatible_C"]), geometry_timestamp=MIDNIGHT)
    con.train_pairs.add(("20161010", "target"))
    con.c_connection_budget = ConnectionBudget()
    con.c_connection_budget.begin(61200., perf_counter() + 10)
    monkeypatch.setattr(con, "_recovery_headings", lambda *args: (90., 90.))
    monkeypatch.setattr(con, "_join", lambda *args: dict(supported=True, compatible_profiles=("C",)))
    original = router.route
    def blocked(*args):
        r = original(*args)
        r.update(compatible_profiles=("HV",), profile_reason_codes={"C": ["UTURN_OUTSIDE_RESEARCH_PROFILE"]})
        return r
    monkeypatch.setattr(router, "route", blocked)
    def alternatives(origin, target, stamp, **kwargs):
        router.calls.append(("ALTERNATIVE", kwargs))
        r = original(*origin, *target, stamp)
        r.update(raw_time_s=150., distance_m=400., shape_polyline6="legal-detour", route_variant="BOUNDED_STATIC_ALTERNATIVE")
        yield r
    monkeypatch.setattr(router, "alternative_routes", alternatives, raising=False)
    return con, router


def test_current_recovery_and_future_share_full_direction_certificate_and_retime(tmp_path, monkeypatch):
    con, router = connector_case(tmp_path, monkeypatch)
    source = RouteState(ORIGIN, 61200., 90000., "C")
    target = task("target"); target.deadline_s = 62060.
    first = con(source, target, 62040., "C", require_geometry=True, enforce_pickup_deadline=False)
    assert first["supported"] and first["travel_time_s"] == 150.
    assert first["routed"]["shape_polyline6"] == "legal-detour"
    n = len(router.calls)
    second = con(source, target, 62100., "C", cache_only=True, enforce_pickup_deadline=False)
    assert second["supported"] and second["travel_time_s"] == 300. and len(router.calls) == n
    assert con(source, target, 62040., "C", cache_only=True)["reason_codes"] == ["CONNECTOR_EXCEEDS_ORIGINAL_PICKUP_DEADLINE"]
    assert con.route_cache[(*ORIGIN, *PICKUP)]["compatible_profiles"] == ("HV",)
    first["routed"]["points"].append((0., 0.))
    again = con(source, target, 62040., "C", cache_only=True, require_geometry=True, enforce_pickup_deadline=False)
    assert (0., 0.) not in again["routed"]["points"]
    changed = replace(source, context=dict(kind="EMPTY_ARRIVAL", point=ORIGIN, edge_uids=("different-direction",)))
    assert con(changed, target, 62040., "C", cache_only=True)["reason_codes"] == ["C_CONNECTION_NOT_EVALUATED_CACHE_MISS"]
    assert len(router.calls) == n


def test_optional_budget_is_shared_and_deferred_failure_is_not_cached(tmp_path, monkeypatch):
    con, router = connector_case(tmp_path, monkeypatch)
    con.c_connection_budget = ConnectionBudget(maximum_per_epoch=0)
    con.c_connection_budget.begin(61200., perf_counter()+10)
    source = RouteState(ORIGIN, 61200., 90000., "C")
    first = con(source, task("target"), 61200., "C", enforce_pickup_deadline=False)
    assert not first["supported"] and len(router.calls) == 1
    assert not con.certificate_cache
    future = con(source, task("target"), 61200., "C", cache_only=True)
    assert future["reason_codes"] == ["C_CONNECTION_NOT_EVALUATED_CACHE_MISS"] and len(router.calls) == 1
    budget = ConnectionBudget(maximum_per_snapshot=2)
    budget.begin(0., perf_counter()+10)
    assert budget.claim("ALTERNATIVE") and budget.claim("FUTURE") and not budget.claim("FUTURE")
    budget.begin(0., perf_counter()+100)  # A lazy probe cannot reset the same epoch.
    assert not budget.claim("FUTURE")
    budget.begin(30., perf_counter()+10)
    assert not budget.claim("FUTURE")
    budget.begin(300., perf_counter()+10)
    assert budget.claim("FUTURE")
    budget.record(.2)
    assert not budget.claim("ALTERNATIVE")


def oracle_case(support=None, *, now=0., memory=None):
    actions=[CurrentAction("C", 1, "WAIT", None), CurrentAction("H", 2, "WAIT", None)]
    tasks=tuple(ChainTask(-i, float(t), float(t+300), 30., PICKUP, ORIGIN, {"C", "HV"},
        source_date="20161010", source_order_id=f"t{i}") for i,t in ((1,300),(2,600)))
    roots={a.action_id:RouteState(ORIGIN, now+30., 1800., "C" if a.vehicle_id==1 else "HV") for a in actions}
    options=dict(topology=memory.topology,forecast_snapshot={"S":tasks}) if memory is not None else {}
    q=PredictionSceneOracle(actions,roots,{"S":tasks},(),NS(global_pace=.01,pace_by_slot={}),now,
        efficient=True,c_support=support,**options)
    return q, actions, tasks


def test_future_search_uses_cached_road_times_and_never_turns_a_miss_into_support():
    calls=[]
    def lookup(state, target, departure):
        calls.append((state.context,target.request_id,departure))
        return dict(supported=target.request_id==-1,travel_time_s=250.,empty_distance_m=700.)
    q,_,_=oracle_case(NS(lookup=lookup))
    c=q.query("S","C")[0]
    assert c.request_ids == (-1,) and c.payload["legs"][0][2:] == (250.,580.,700.)
    assert c.payload["support"] == "C_CACHED_DIRECTED_CONNECTIONS_M3_CUSTOMER_TIME"
    assert any(context and context["kind"]=="CUSTOMER" for context,_,_ in calls)
    before=len(calls)
    q.query_many("S","C",{-1,-2})
    assert len(calls) == before  # Per-oracle compact lookup reuse; no Actor exists here.
    h=q.query("S","H")[0]
    assert h.request_ids == (-1,-2) and h.payload["legs"][0][2] < 10.
    assert len(calls) == before
    blocked,_,_=oracle_case(NS(lookup=lambda *args:dict(supported=False)))
    assert blocked.query("S","C") == () and blocked.query("S","H")
    # Identical synthetic IDs/coordinates in different scenarios are not the
    # same historical route identity and must not share a support answer.
    original=q.tasks["S"][-1]
    other=replace(original,source_order_id="different-body")
    check=NS(lookup=lambda state,t,depart:dict(supported=t.source_order_id!="different-body",
        travel_time_s=20.,empty_distance_m=100.))
    same_ids=PredictionSceneOracle(list(q.actions.values()),q.roots,{"S":(original,),"T":(other,)},(),
        NS(global_pace=.01,pace_by_slot={}),0.,efficient=True,c_support=check)
    assert same_ids.query("S","C") and not same_ids.query("T","C")


def test_C_warm_paths_recheck_support_instead_of_reusing_proxy_certificate():
    allowed=[True]
    support=NS(lookup=lambda *args:dict(supported=allowed[0],travel_time_s=10.,empty_distance_m=100.))
    memory=RollingJointMemory(efficient=True,exact_warm_reuse=True)
    q,actions,_=oracle_case(support,memory=memory)
    p=JointProblem(actions,q,{"S":1})
    initial,_,_=memory.initial(p,("C","H"),lambda:None,perf_counter()+2)
    first=q.repair("S","C",(-1,),frozenset(),lambda:None)
    second=q.repair("S","H",(-2,),frozenset(),lambda:None)
    p.register(first); p.register(second)
    initial=replace(initial,chains=(first,second))
    memory.save(p,initial,initial,(),0.)
    allowed[0]=False
    q,actions,_=oracle_case(support,now=30.,memory=memory)
    p=JointProblem(actions,q,{"S":1})
    after,info,_=memory.initial(p,("C","H"),lambda:None,perf_counter()+2)
    assert next(c for c in after.chains if c.vehicle_id==1).request_ids == ()
    assert info.get("warm_exact_reused_continuations",0)==1  # HV still uses the optimization.


def test_shared_current_validator_preserves_HV_and_independent_pickup_cap():
    eta=ETA(); c,av=case(eta); _,hv=case(eta,vehicle_type="HV",vid=1,rid=101)
    c.request_meta[101]=dict(pickup_deadline_s=62400.,research_route_compatible=False)
    c.config=dict(patience_mode="MATCH_DEADLINE_AND_PICKUP_CAP",max_match_wait_s=300,
                  max_pickup_wait_s=300,max_pickup_eta_s=600)
    route=Routes(eta).route(*ORIGIN,*PICKUP,MIDNIGHT)
    route.update(beta=1.,time_bin_index=68,duration_s=150.)
    shared=NS(current=lambda *args:dict(supported=True,travel_time_s=150.,empty_distance_m=300.,
        routed=route,connection_evidence_state="C_DIRECTED_SEAM_CERTIFIED"))
    validator=CityPickupSupportValidator(Routes(eta),geometry_timestamp=MIDNIGHT,
        routing_budget_s=10,seam_evidence=AllowedJoin())
    validator.shared_C_support=shared
    first=validator(c,[av,hv],62370.)  # Assignment allowed with only 30s left; pickup takes 150s.
    assert len(first)==2 and first[0].pickup_eta_s==150. and first[1] is hv
    shared.current=lambda *args:dict(supported=True,travel_time_s=601.,empty_distance_m=300.,
        routed=route,connection_evidence_state="C_DIRECTED_SEAM_CERTIFIED")
    assert validator(c,[av,hv],62370.)==[hv]


def test_static_alternatives_bound_paths_deduplicate_and_leave_default_cache(tmp_path,monkeypatch):
    from stage4.tests.test_controlled_routes import router as make_router
    router,actor=make_router(tmp_path,signalized=True)
    default=router.route(*ORIGIN,*PICKUP,MIDNIGHT)
    before=dict(router.cache); seen=[]
    def backend(request):
        seen.append(request)
        def trip(shape): return dict(trip=dict(legs=[dict(shape=shape)]))
        return dict(**trip("A"),alternates=[trip("B"),trip("C"),trip("MUST_NOT_READ")])
    monkeypatch.setattr(actor,"route",backend)
    def parse(trip,*args):
        r=router._base_result(); r.update(common_supported=True,shape_polyline6=trip["legs"][0]["shape"])
        return r
    monkeypatch.setattr(router,"_trip_result",parse)
    paths=list(router.alternative_routes(ORIGIN,PICKUP,MIDNIGHT,origin_heading=90.,target_heading=90.,exclude_shapes=("A",)))
    assert [p["shape_polyline6"] for p in paths]==["B","C"]
    assert seen[0]["date_time"]==dict(type=3,value="2016-10-31T00:00") and seen[0]["alternates"]==2
    assert seen[0]["locations"][0]["heading"]==90. and router.cache==before
    assert default["shape_polyline6"] not in ("B","C")


def test_future_prefetch_is_small_and_unrevealed_Test31_identity_never_routes(tmp_path,monkeypatch):
    con,router=connector_case(tmp_path,monkeypatch)
    before=len(router.calls)
    unrevealed=task("not-revealed",date="20161031",kind="ACTUAL_PENDING")
    result=con(RouteState(ORIGIN,61200.,90000.,"C"),unrevealed,61200.,"C",cache_only=True)
    assert result["reason_codes"]==["TEST31_CUSTOMER_NOT_YET_REVEALED"] and len(router.calls)==before
    class Fake:
        def __init__(self): self.counts=Counter(); self.cache=set(); self.calls=0
        def __call__(self,state,target,departure,profile,*,cache_only=False,**kwargs):
            key=state.position,target.source_order_id
            if key not in self.cache:
                if cache_only or not self.c_connection_budget.claim("FUTURE"):
                    return dict(supported=False,reason_codes=["C_CONNECTION_NOT_EVALUATED_BUDGET"])
                self.calls+=1; self.counts["certificate_cache_misses"]+=1; self.cache.add(key)
            return dict(supported=True,travel_time_s=20.,empty_distance_m=100.,reason_codes=[])
    fake=Fake(); support=SharedCConnectionSupport(fake,None,dict(joint_c_connection_mode=MODE))
    support.begin(0.,perf_counter()+5)
    q,actions,tasks=oracle_case()
    support.prefill(actions,q.roots,{"S":tasks},(),("C","H"),None,0.,lambda:None)
    assert 0<fake.calls<=2 and support.counts["prefetch_connections_examined"]<=64
    before=fake.calls
    for _ in range(1000): support.lookup(q.roots["C"],tasks[0],300.)
    assert fake.calls==before


def test_native_joint_bridge_keeps_partial_C_scope_explicit(tmp_path,monkeypatch):
    from stage4.dispatch import bottleneck_native as native
    from stage4.dispatch.scheme_a_current_flow import solve_current_flow
    monkeypatch.setattr(native,"predicted_vehicle_states",lambda *args:())
    support=NS(prefill=lambda *args:(),lookup=lambda *args:dict(supported=True,travel_time_s=20.,empty_distance_m=100.),
               counts=Counter(),connectors=NS(router=NS(route_call_count=0)),
               diagnostics=lambda:dict(search_road_queries=0,counts={"lookups":1}))
    q,actions,tasks=oracle_case(support)
    c=NS(shared_C_connection_support=support,repositioning_manager=NS(active={}),request_by_rid={},request_meta={})
    library=NS(view=lambda now:{"S":tasks},weights={"S":1})
    provider=NS(library=library,pace=NS(global_pace=.01,pace_by_slot={}))
    cfg=dict(joint_c_connection_mode=MODE,joint_execution_mode="INCREMENTAL_V3",
             joint_neighborhood_strategy="ADAPTIVE_SHARED",joint_maximum_labels=50000)
    baseline=solve_current_flow(actions,{a.action_id:0 for a in actions},policy="SERVICE_PRESERVING",time_limit_s=1.)
    result=native.solve_joint_current(c,actions,q.roots,baseline,0.,cfg,provider,{},None,None,None,(),perf_counter()+5,
        execution_state=RollingJointMemory(efficient=True,exact_warm_reuse=True))
    assert not result["joint_control"]["resource_fallback"]
    assert result["joint_control"]["C_future_directed_road_certificate"]
    assert not result["joint_control"]["C_future_domain_complete"]
    assert not result["joint_control"]["HV_future_directed_road_certificate"]
    pd.DataFrame([result["joint_control"]]).to_parquet(tmp_path/"trace.parquet",index=False)
