"""Focused synthetic contracts; no city data, actor, inference or benchmark."""
from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace as NS

import pytest

from stage4.dispatch.bottleneck_joint import JointProblem,JointBudgetStop,JointLabelStop,solve_block
from stage4.dispatch.bottleneck_scene import PredictionSceneOracle
from stage4.dispatch.bottleneck_topology import ForecastTopology
from stage4.dispatch.bottleneck_incremental import RollingJointMemory,BlockWorkspace,incremental_search
from stage4.dispatch.scheme_a_city_graph import ChainTask,RouteState
from stage4.dispatch.scheme_a_city_master import CurrentAction,ServiceChain
from stage4.tests.test_bottleneck_joint import complementary,populate


PACE=NS(global_pace=.01,pace_by_slot={})
WEST=(108.9,34.2)
EAST=(108.96,34.2)


def forecast(rid,release,point=WEST):
    return ChainTask(rid,release,release+300.,30.,point,point,{"C","HV"},source_date="20161010")


def oracle(actions,states,snapshot,now=0.,pending=(),cache=None):
    scenes={s:tuple(t for t in tasks if now<t.release_s<=now+1800) for s,tasks in snapshot.items()}
    return PredictionSceneOracle(actions,states,scenes,pending,PACE,now,
        topology=cache,forecast_snapshot=snapshot if cache is not None else None)


def test_topology_reuse_and_reverse_pruning_preserve_declared_chain_domain():
    actions=[CurrentAction("W",1,"WAIT",None),CurrentAction("E",2,"WAIT",None)]
    tasks=tuple(forecast(-i,release) for i,release in enumerate((300.,600.,900.,1200.),1))
    tasks+=tuple(forecast(-i,release,EAST) for i,release in enumerate((300.,600.,900.,1790.),11))
    tasks=tuple(replace(t,passenger_accepts_av=False) if t.request_id==-12 else t for t in tasks)
    snapshot={"S":tasks}; cache=ForecastTopology()
    for now in (0.,30.,60.):
        roots={"W":RouteState(WEST,now+30,2400.,"HV"),"E":RouteState(EAST,now+30,2400.,"C")}
        pending=() if now<60 else (ChainTask(11,60.,360.,10.,WEST,WEST,{"HV","C"},
            source_date="20161031",source_kind="ACTUAL_PENDING"),)
        old=oracle(actions,roots,snapshot,now,pending)
        new=oracle(actions,roots,snapshot,now,pending,cache)
        for aid in ("W","E"):
            for required in (None,-2,-12,-14):
                for forbidden in (frozenset(),frozenset({-1})):
                    a=old.query("S",aid,required=required,forbidden=forbidden)
                    b=new.query("S",aid,required=required,forbidden=forbidden)
                    assert [(c.request_ids,c.empty_distance_m,c.payload) for c in a]==[
                        (c.request_ids,c.empty_distance_m,c.payload) for c in b]
        assert cache.references<=20000
    assert cache.counts["snapshot_rebuilds"]==1 and cache.counts["snapshot_reuses"]==2
    roots={"W":RouteState(WEST,30.,2400.,"HV"),"E":RouteState(EAST,30.,2400.,"C")}
    old=oracle(actions,roots,snapshot); new=oracle(actions,roots,snapshot,cache=cache)
    assert old.query("S","W",required=-12)==new.query("S","W",required=-12)==()
    assert new.labels<old.labels and new.target_pruned_nodes>0
    future={"S":tuple(t for t in tasks if t.release_s>300)}
    oracle(actions,{k:replace(v,ready_s=330.) for k,v in roots.items()},future,300.,cache=cache)
    assert cache.counts["snapshot_rebuilds"]==2
    with pytest.raises(ValueError,match="snapshot"):
        oracle(actions,roots,{"S":(replace(tasks[0],source_date="20161031"),)},cache=cache)


def test_warm_plan_quiet_epoch_reuses_owners_without_new_search_or_greedy_seed():
    memory=RollingJointMemory()
    actions=[CurrentAction("A",1,"WAIT",None),CurrentAction("B",2,"WAIT",None)]
    snapshot={"S":(forecast(-1,300.),forecast(-2,600.))}
    states={a.action_id:RouteState(WEST,30.,1800.,"HV") for a in actions}
    q=oracle(actions,states,snapshot,cache=memory.topology)
    p=JointProblem(actions,q,{"S":1})
    initial,_,_=memory.initial(p,("A","B"),lambda:None,perf_counter()+5)
    memory.save(p,initial,initial,(),0.)
    q=oracle(actions,{k:replace(v,ready_s=60.) for k,v in states.items()},snapshot,30.,cache=memory.topology)
    p=JointProblem(actions,q,{"S":1})
    kept,info,dirty=memory.initial(p,("A","B"),lambda:None,perf_counter()+5)
    assert p.evaluate(kept)["expected_future_served"]==2
    assert info.get("warm_greedy_seed_queries",0)==0 and q.labels==0
    assert memory.frontier(p,kept,dirty,lambda:None)==()
    solved,search=incremental_search(p,kept,(),strategy="BOTTLENECK_CLOSURE",
        deadline=perf_counter()+5,check=lambda:None)
    assert solved==kept and search["search_work"]["solver_calls"]==0
    # A new coarse snapshot has new task IDs: stale chains are repaired, not trusted.
    fresh={"S":(forecast(-3,600.),forecast(-4,900.))}
    q=oracle(actions,{k:replace(v,ready_s=330.) for k,v in states.items()},fresh,300.,cache=memory.topology)
    p=JointProblem(actions,q,{"S":1})
    kept,_,_=memory.initial(p,("A","B"),lambda:None,perf_counter()+5)
    assert all(set(c.request_ids)<={-3,-4} for c in kept.chains)
    assert p.evaluate(kept)["expected_future_served"]==2


def test_repair_retimes_pending_original_deadline_and_never_repeats_current_service():
    memory=RollingJointMemory(); action=CurrentAction("W",1,"WAIT",None)
    task=ChainTask(11,0.,120.,30.,WEST,WEST,{"HV","C"},
        source_date="20161031",source_kind="ACTUAL_PENDING")
    state=RouteState(WEST,30.,900.,"HV"); snapshot={"S":()}
    q=oracle([action],{"W":state},snapshot,pending=(task,),cache=memory.topology)
    p=JointProblem([action],q,{"S":1},(11,))
    plan,_,_=memory.initial(p,("W",),lambda:None,perf_counter()+5)
    assert plan.chains[0].payload["legs"][0][1]==30.
    memory.save(p,plan,plan,(),0.)
    q=oracle([action],{"W":replace(state,ready_s=90.)},snapshot,60.,(task,),memory.topology)
    p=JointProblem([action],q,{"S":1},(11,))
    plan,_,_=memory.initial(p,("W",),lambda:None,perf_counter()+5)
    assert plan.chains[0].payload["legs"][0][1]==90.
    q=oracle([action],{"W":replace(state,ready_s=120.)},snapshot,90.,(task,),memory.topology)
    p=JointProblem([action],q,{"S":1},(11,))
    expired,_,_=memory.initial(p,("W",),lambda:None,perf_counter()+5)
    assert p.evaluate(expired)["expected_future_served"]==0
    serve=CurrentAction("S",1,"SERVE",11,critical=True)
    q=oracle([serve],{"S":replace(state,ready_s=180.)},snapshot,90.,(task,),memory.topology)
    p=JointProblem([serve],q,{"S":1},(11,))
    plan,_,_=memory.initial(p,("S",),lambda:None,perf_counter()+5)
    assert p.evaluate(plan)["current_served"]==1 and not plan.chains[0].request_ids


def test_constant_phase_and_exact_duplicate_block_do_not_restart_integer_solver():
    p,initial=complementary(); populate(p)
    old,legacy=solve_block(p,initial,{1,2,3},perf_counter()+5)
    w=BlockWorkspace(); new,info=w.solve(p,initial,{1,2,3},perf_counter()+5,lambda:None)
    assert p.evaluate(old)["score"]==p.evaluate(new)["score"]
    assert info["solver_calls"]<legacy["solver_calls"]
    assert info["constant_objective_stages_skipped"]==1
    again,skipped=w.solve(p,new,{1,2,3},perf_counter()+5,lambda:None)
    assert again==new and skipped["solver_calls"]==0 and skipped["duplicate_block_skipped"]
    # Changed legal columns invalidate the visited state, even if the incumbent stays fixed.
    c=ServiceChain("new","S1",1,"H",(-4,))
    p.oracle.columns[c.chain_id]=c; p.register(c)
    _,changed=w.solve(p,new,{1,2,3},perf_counter()+5,lambda:None)
    assert changed["solver_calls"]>0
    # A different owner/first-action plan conservatively invalidates all cached states.
    _,changed=w.solve(p,replace(new),{1,2,3},perf_counter()+5,lambda:None)
    assert changed["solver_calls"]>0


def test_label_budget_flush_still_evaluates_already_legal_columns(monkeypatch):
    p,initial=complementary(); populate(p); w=BlockWorkspace()
    def exhausted(*args,**kwargs):
        raise JointLabelStop("synthetic exhausted label allowance")
    monkeypatch.setattr(p.oracle,"query",exhausted)
    w.populate(p,initial,{1,2,3},{"S1":{-4}},lambda:None)
    assert w.columns_closed and w.counts["label_budget_flushes"]==1
    plan,info=w.solve(p,initial,{1,2,3},perf_counter()+5,lambda:None)
    assert p.evaluate(plan)["score"][0]==12 and info["solver_calls"]>0
    # Attempt counters and elapsed time also survive an interrupted solver call.
    from stage4.dispatch import bottleneck_joint as kernel
    fresh=BlockWorkspace()
    monkeypatch.setattr(kernel,"milp",exhausted)
    with pytest.raises(JointBudgetStop):
        fresh.solve(p,initial,{1,2,3},perf_counter()+5,lambda:None)
    assert fresh.metrics["solver_calls"]==1 and fresh.metrics["solver_wall_time_s"]>=0


def test_lazy_closure_keeps_neutral_partial_coalitions_and_can_change_current_roots():
    p,initial=complementary(); populate(p)
    seeds=tuple(p.oracle.seeds(p.evaluate(initial)["owners"]))
    best,info=incremental_search(p,initial,seeds,strategy="BOTTLENECK_CLOSURE",
        deadline=perf_counter()+10,check=lambda:None)
    assert p.evaluate(best)["score"][0]==12
    assert set(best.action_ids)=={"H","C1M","C2M"}
    assert info["search_work"]["fixed_only_blocks_skipped"]>0
    assert any(not r["improved"] for r in info["neighborhoods"])
    assert any(r["coalition_size"]==3 and r["improved"] for r in info["neighborhoods"])
