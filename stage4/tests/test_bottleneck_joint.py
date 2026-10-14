"""Four mathematical/code contracts, no data replay, actor or inference."""
from dataclasses import replace
from fractions import Fraction
from time import perf_counter
from types import SimpleNamespace as NS

import pytest

from stage4.dispatch.bottleneck_joint import (
    Coalition, JointPlan, JointProblem, close_coalition, improve, seed_plan, solve_block,
)
from stage4.dispatch.bottleneck_scene import PredictionSceneOracle
from stage4.dispatch.scheme_a_city_graph import ChainTask, RouteState
from stage4.dispatch.scheme_a_city_master import CurrentAction, ServiceChain


class SuppliedOracle:
    """Tiny explicitly supplied legal families; not a city path catalogue."""
    def __init__(self, actions, families):
        self.actions={a.action_id:a for a in actions}
        self.columns={}
        for (sid,aid),paths in families.items():
            for i,path in enumerate(paths):
                c=ServiceChain(f"{sid}|{aid}|{i}",sid,self.actions[aid].vehicle_id,aid,tuple(path))
                self.columns[c.chain_id]=c
        self.registered=self.columns

    def accepts(self,c):
        return self.columns.get(c.chain_id)==c

    def query(self,sid,aid,*,required=None,forbidden=frozenset(),check=lambda:None):
        check()
        found=[c for c in self.columns.values() if c.scenario_id==sid and c.action_id==aid
               and (required is None or required in c.request_ids)
               and not set(c.request_ids)&forbidden]
        return tuple(sorted(found,key=lambda c:(-len(c.request_ids),c.request_ids))[:1])

    def supports(self,sid,rid,check=lambda:None):
        check()
        return tuple(sorted({c.action_id for c in self.columns.values()
                             if c.scenario_id==sid and rid in c.request_ids}))

    def seeds(self,owners,check=lambda:None):
        for key in sorted({(c.scenario_id,r) for c in self.columns.values() for r in c.request_ids}):
            check()
            if key not in owners:
                yield key


def complementary():
    actions=[CurrentAction("H",1,"WAIT",None),
             CurrentAction("C1W",2,"WAIT",None),CurrentAction("C2W",3,"WAIT",None),
             CurrentAction("C1M",2,"RELOCATE",None,empty_distance_m=10.,
                           payload={"relocation_slot_s":0}),
             CurrentAction("C2M",3,"RELOCATE",None,empty_distance_m=10.,
                           payload={"relocation_slot_s":0})]
    families={}
    for s in ("S1","S2","S3"):
        families[s,"H"]=[(-1,-2,-3),(-4,-5),(-2,-3),(-1,-3)]
        families[s,"C1M"]=[(-1,)]
        families[s,"C2M"]=[(-2,)]
    oracle=SuppliedOracle(actions,families)
    p=JointProblem(actions,oracle,{s:Fraction(1,3) for s in ("S1","S2","S3")},move_cap=2)
    plan,_=seed_plan(p,("H","C1W","C2W"))
    return p,plan


def populate(p):
    for s in p.weights:
        for aid,a in p.actions.items():
            p.register(p.empty(s,a))
            # Contract fixtures register the complete tiny declared family.
            for c in p.oracle.columns.values():
                if c.scenario_id==s and c.action_id==aid:
                    p.register(c)


def test_joint_search_crosses_nonimproving_single_resource_steps():
    p,initial=complementary()
    populate(p)
    start=p.evaluate(initial)["score"]
    for v in (1,2,3):
        alone,_=solve_block(p,initial,{v},perf_counter()+5)
        assert p.evaluate(alone)["score"]<=start
    best,info=improve(p,initial,deadline=perf_counter()+10)
    assert p.evaluate(best)["score"][0]==12  # 4 services × exact weight denominator 3.
    assert set(best.action_ids)=={"H","C1M","C2M"}
    assert info["accepted_coalitions"]>0
    assert any(r["coalition_size"]==3 and r["improved"] for r in info["neighborhoods"])
    assert any(not r["improved"] for r in info["neighborhoods"])


def test_common_first_action_and_current_future_single_use():
    actions=[CurrentAction("L",1,"LAYOUT",None),CurrentAction("R",1,"LAYOUT",None)]
    families={("A","L"):[(-1,-2)],("B","R"):[(-3,-4)]}
    oracle=SuppliedOracle(actions,families)
    p=JointProblem(actions,oracle,{"A":Fraction(1,2),"B":Fraction(1,2)})
    initial,_=seed_plan(p,("L",)); populate(p)
    solved,_=solve_block(p,initial,{1},perf_counter()+5)
    assert p.evaluate(solved)["score"][0]==2  # Expected 1, not clairvoyant 2.
    bad=JointPlan(("L",),(next(iter(oracle.columns.values())),
        next(c for c in oracle.columns.values() if c.action_id=="R")))
    with pytest.raises(ValueError,match="scenario first"):
        p.evaluate(bad)
    actions=[CurrentAction("S",1,"SERVE",11,True,True),
             CurrentAction("W",2,"WAIT",None)]
    oracle=SuppliedOracle(actions,{("A","W"):[(11,)]})
    p=JointProblem(actions,oracle,{"A":1},(11,))
    initial,_=seed_plan(p,("S","W")); populate(p)
    solved,_=solve_block(p,initial,{2},perf_counter()+5)
    assert p.evaluate(solved)["score"][0]==1
    repeated=next(c for c in oracle.columns.values() if c.request_ids)
    bad=JointPlan(initial.action_ids,(p.empty("A",p.actions["S"]),repeated))
    with pytest.raises(ValueError,match="allocated twice"):
        p.evaluate(bad)


def test_closure_outside_ownership_and_residual_move_capacity():
    p,plan=complementary(); populate(p)
    c=next(c for c in p.oracle.columns.values() if c.action_id=="C1M")
    closed=close_coalition(p,plan,Coalition(frozenset({2}),(c,)))
    assert closed[0].vehicles==frozenset({1,2})  # A is not silently taken from H.
    assert set(closed[1])=={"S1","S2","S3"}  # Changing C1 root affects every scene.
    opposite=replace(c,chain_id="opposite",action_id="C1W")
    p.oracle.columns[opposite.chain_id]=opposite
    p.register(opposite)
    assert close_coalition(p,plan,Coalition(frozenset({2}),(c,opposite))) is None
    # An outside current MOVE consumes one slot, so the two inside moves cannot execute together.
    extra=CurrentAction("OUT",4,"RELOCATE",None,empty_distance_m=1.,
                        payload={"relocation_slot_s":0})
    actions=[*p.actions.values(),extra]
    oracle=SuppliedOracle(actions,{key:[c.request_ids for c in p.oracle.columns.values()
        if (c.scenario_id,c.action_id)==key] for key in p.pool})
    q=JointProblem(actions,oracle,p.weights,move_cap=2)
    initial,_=seed_plan(q,("H","C1W","C2W","OUT")); populate(q)
    solved,_=solve_block(q,initial,{1,2,3},perf_counter()+5)
    assert q.evaluate(solved)["current_moves"]<=2
    assert not {"C1M","C2M"}<=set(solved.action_ids)


def test_prediction_columns_keep_original_pending_clock_and_reject_future_targets():
    action=CurrentAction("W",1,"WAIT",None)
    state=RouteState((108.9,34.2),30.,500.,"HV")
    pending=ChainTask(11,0.,120.,30.,state.position,state.position,{"HV","C"},
                      source_date="20161031",source_kind="ACTUAL_PENDING")
    pace=NS(global_pace=.01,pace_by_slot={})
    oracle=PredictionSceneOracle([action],{"W":state},{"S":()},(pending,),pace,0.)
    chain=oracle.query("S","W",required=11)[0]
    assert chain.payload["legs"][0][1]==30.
    assert chain.payload["legs"][0][1]<pending.deadline_s
    late=PredictionSceneOracle([action],{"W":replace(state,ready_s=120.)},
                               {"S":()},(pending,),pace,0.)
    assert late.query("S","W",required=11)==()
    with pytest.raises(ValueError,match="revealed"):
        PredictionSceneOracle([action],{"W":state},{"S":()},(replace(pending,release_s=10.),),pace,0.)
    wrong=replace(pending,request_id=-1,source_kind="FORECAST",release_s=1.)
    with pytest.raises(ValueError,match="earlier-history"):
        PredictionSceneOracle([action],{"W":state},{"S":(wrong,)},(),pace,0.)
