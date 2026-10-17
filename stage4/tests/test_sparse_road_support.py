"""Small shared-evidence/frontier contracts. No Actor, replay, or inference."""
from collections import Counter
from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace as NS

import pytest

from stage4.dispatch.c_connection_support import SharedCConnectionSupport,context_key
from stage4.dispatch.sparse_road_support import SparseRoadIndex,MODE,task_key
from stage4.dispatch.bottleneck_scene import PredictionSceneOracle
from stage4.dispatch.bottleneck_topology import ForecastTopology
from stage4.dispatch.bottleneck_joint import JointProblem
from stage4.dispatch.bottleneck_incremental import RollingJointMemory
from stage4.dispatch.scheme_a_city_graph import RouteState
from stage4.dispatch.scheme_a_city_master import CurrentAction
from stage4.tests.test_economic_joint_dispatch import MODEL,PACE,POINT,task


def test_raw_evidence_retimes_and_only_positive_history_persists(tmp_path):
    path=tmp_path/"support.sqlite3"
    before=task(-1); target=task(-2,600.)
    state=RouteState(before.dropoff,330.,1800.,"C",dict(kind="CUSTOMER",task=before))
    index=SparseRoadIndex(context_key,"frozen-physical-inputs",path=path)
    index.begin(0.)
    index.remember(state,target,dict(supported=True,raw_time_s=12.,empty_distance_m=80.,
        connection_evidence_state="C_DIRECTED_SEAM_CERTIFIED"))
    alias=replace(target,request_id=-200,release_s=650.,service_time_s=75.)
    assert index.lookup(state,alias,2.)["travel_time_s"]==24.
    index.remember(state,task(-3),dict(supported=False,search_deferred=True,reason_codes=["DEFERRED"]))
    assert not index.assessed(state,task(-3))
    current=task(10,0.)
    index.remember(state,current,dict(supported=True,raw_time_s=10.,empty_distance_m=100.))
    native=replace(state,context=dict(kind="NATIVE_VEHICLE",native_vehicle_id=1,timestamp_s=0.))
    index.remember(native,target,dict(supported=True,raw_time_s=10.,empty_distance_m=100.))
    index.begin(0.)  # Same-tick re-entry must not retain a stale incoming native prefix.
    assert not index.assessed(native,target) and index.assessed(state,target)
    assert index.db.execute("SELECT count(*) FROM road_support").fetchone()[0]==1
    index.close()
    again=SparseRoadIndex(context_key,"frozen-physical-inputs",path=path)
    again.hydrate(state)
    assert again.lookup(state,alias,3.)["travel_time_s"]==36.
    assert not again.lookup(replace(state,profile_id="HV"),target,1.)["supported"]
    assert not again.lookup(state,current,1.)["supported"]
    assert again.bytes<=again.maximum_bytes
    again.close()
    other=SparseRoadIndex(context_key,"changed-physical-inputs",path=path)
    other.hydrate(state); assert not other.lookup(state,target,1.)["supported"]
    other.close()


class Connector:
    def __init__(self):
        self.counts=Counter(); self.records={}; self.requests=[]
        self.control=NS(_timestamp=lambda x:x,eta_adapter=NS(beta_for=lambda t:(0,1.)))
    def __call__(self,state,t,departure,profile,*,cache_only=False,**kwargs):
        self.counts["calls"]+=1
        key=profile,state.position,context_key(state.context),task_key(t)
        if key not in self.records:
            if cache_only or not self.c_connection_budget.claim("FUTURE"):
                return dict(supported=False,reason_codes=[profile+"_CONNECTION_NOT_EVALUATED_BUDGET"])
            self.requests.append(t.request_id)
            self.records[key]=dict(supported=True,raw_time_s=10.,travel_time_s=10.,
                empty_distance_m=100.,connection_evidence_state="C_TYPED" if profile=="C" else "HV_AUTO",reason_codes=[])
        return self.records[key]


def prepared_case(extra_roots=0):
    current=task(11,0.,400.)
    actions=[CurrentAction("HW",1,"WAIT",None),CurrentAction("HS",1,"SERVE",11,False,True,100.),
        CurrentAction("CW",2,"WAIT",None)]
    roots={a.action_id:RouteState(POINT,400. if a.kind=="SERVE" else 30.,1800.,
        "C" if a.vehicle_id==2 else "HV",dict(kind="CUSTOMER",task=current) if a.kind=="SERVE" else None) for a in actions}
    for i in range(extra_roots):
        a=CurrentAction(f"EXTRA{i}",10+i,"WAIT",None); actions.append(a)
        roots[a.action_id]=RouteState((POINT[0]+.00001*(i+1),POINT[1]),30.,1800.,"HV")
    tasks=(task(-1,300.),task(-2,600.))
    connector=Connector()
    support=SharedCConnectionSupport(connector,None,dict(joint_c_connection_mode="BOUNDED_CERTIFIED_C_V1",
        economic_future_connections="BOTH_CACHE_CERTIFIED_V1",economic_sparse_connections=MODE,
        sparse_connection_namespace="test-snapshot"))
    support.begin(0.,perf_counter()+5)
    q=PredictionSceneOracle(actions,roots,{"S":tasks},(current,),PACE,0.,efficient=True,
        economics=MODEL,c_support=support)
    selected=("HS","CW",*(f"EXTRA{i}" for i in range(extra_roots)))
    memory=RollingJointMemory(efficient=True)
    q.prepare_road_frontier(selected,memory,lambda:None)
    return q,support,connector,selected,memory


def test_shared_prefetch_skips_booked_requests_and_seeds_a_real_supported_chain():
    q,support,connector,selected,memory=prepared_case()
    assert 11 not in connector.requests and len(connector.requests)==2
    before=connector.counts["calls"]
    p=JointProblem(list(q.actions.values()),q,{"S":1},(11,),economics=MODEL)
    plan,info,_=memory.initial(p,selected,lambda:None,perf_counter()+2)
    assert p.evaluate(plan)["expected_future_served"]==2
    assert next(c for c in plan.chains if c.vehicle_id==2).request_ids==(-1,-2)
    assert info["warm_unprepared_roots_skipped"]>=1
    assert connector.counts["calls"]==before  # Search/seed reads indexed evidence only.
    assert support.budget.epoch_calls==2
    assert q.prepared_frontier.counts["shared_candidate_assessments"]<=64


def test_unprepared_roots_do_not_repeat_connector_work_and_match_reference_paths():
    q,support,connector,selected,memory=prepared_case(80)
    # Reference uses the SAME evidence, via the old expensive cache-only path.
    reference_support=NS(all_profiles=True,lookup=lambda state,t,dep:
        connector(state,t,dep,state.profile_id,cache_only=True))
    old=PredictionSceneOracle(list(q.actions.values()),q.roots,{"S":tuple(t for t in q.tasks["S"].values()
        if t.source_kind=="FORECAST")},(q.tasks["S"][11],),PACE,0.,efficient=True,
        economics=MODEL,c_support=reference_support)
    before=connector.counts["calls"]
    reference={a:old.query_many("S",a,{-1,-2}) for a in old.actions}
    reference_calls=connector.counts["calls"]-before
    before=connector.counts["calls"]
    actual={a:q.query_many("S",a,{-1,-2}) for a in q.actions}
    assert connector.counts["calls"]==before and reference_calls>100
    for aid in actual:
        for rid in (-1,-2,None):
            assert [c.request_ids for c in actual[aid][rid]]==[c.request_ids for c in reference[aid][rid]]
    assert q.prepared_frontier.root_references<=20000


def test_prepared_historical_catalog_matches_live_topk_with_pending_and_fine_time(tmp_path):
    from stage4.analysis.economic_sparse_connections import write_catalog
    forecasts=tuple(task(-i,90.+i*100.,10.,1000.+i*200.,allowed=("C","HV") if i%3 else ("HV",))
        for i in range(1,12))+(task(-90,1750.,10.,50000.),)
    snapshot={"S":forecasts}; action=CurrentAction("W",1,"WAIT",None)
    roots={"W":RouteState(POINT,30.,1800.,"HV")}
    top=ForecastTopology()
    producer=PredictionSceneOracle([action],roots,snapshot,(),PACE,0.,efficient=True,economics=MODEL,
        topology=top,forecast_snapshot=snapshot)
    write_catalog(tmp_path/"slot=0.parquet",producer,MODEL.fingerprint)
    pending=task(12,60.,10.,50000.)
    def oracle(directory=None):
        t=ForecastTopology(); t.prepared_directory=directory; t.prepared_fingerprint=MODEL.fingerprint
        return PredictionSceneOracle([action],roots,snapshot,(pending,),PACE,60.,efficient=True,economics=MODEL,
            topology=t,forecast_snapshot=snapshot)
    raw,prepared=oracle(),oracle(tmp_path)
    assert prepared.topology.counts["prepared_catalog_loads"]==1
    for node in forecasts:
        for profile in ("C","HV"):
            assert [t.request_id for t in raw._following("S",node,profile)]==[
                t.request_id for t in prepared._following("S",node,profile)]
