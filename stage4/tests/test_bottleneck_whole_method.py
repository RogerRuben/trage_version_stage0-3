"""Three focused contracts, no city data, routing, inference or native replay."""
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from stage4.analysis import bottleneck_whole_method as protocol
from stage4.dispatch.bottleneck_joint import JointProblem,seed_plan
from stage4.dispatch.bottleneck_layout import solve_birth_batch
from stage4.dispatch.bottleneck_scene import PredictionSceneOracle
from stage4.dispatch.scheme_a_city_graph import ChainTask,RouteState
from stage4.dispatch.scheme_a_city_master import CurrentAction


def test_cheap_initial_witness_does_not_exhaust_search_labels():
    actions=[CurrentAction("W1",1,"WAIT",None),CurrentAction("W2",2,"WAIT",None)]
    state=RouteState((108.9,34.2),30.,1800.,"HV")
    task=ChainTask(11,0.,120.,30.,state.position,state.position,{"HV","C"},
                   source_date="20161031",source_kind="ACTUAL_PENDING")
    oracle=PredictionSceneOracle(actions,{a.action_id:state for a in actions},
        {"HISTORY":()},(task,),NS(global_pace=.01,pace_by_slot={}),0.,maximum_labels=1)
    problem=JointProblem(actions,oracle,{"HISTORY":1},(11,))
    plan,limited=seed_plan(problem,("W1","W2"))
    assert not limited and oracle.labels==0
    assert problem.evaluate(plan)["expected_future_served"]==1
    assert sum(11 in c.request_ids for c in plan.chains)==1
    # Ending only the initialization slice leaves a valid terminal plan,
    # rather than spending the rest of the decision budget on deep seeds.
    blank=JointProblem(actions,oracle,{"HISTORY":1},(11,))
    plan,limited=seed_plan(blank,("W1","W2"),seed_deadline=-1.)
    assert limited and blank.evaluate(plan)["expected_future_served"]==0
    assert oracle.labels==0


def test_birth_bridge_uses_same_root_across_scenarios_and_keeps_fixed_supply():
    left=dict(site_id="left",node_id=1,position=(108.9,34.2))
    right=dict(site_id="right",node_id=2,position=(108.94,34.2))
    def task(rid,point):
        return ChainTask(rid,300.,600.,60.,point,point,{"HV","C"},
                         source_date="20161010",source_kind="FORECAST")
    scenes={"A":(task(-1,right["position"]),),"B":(task(-2,right["position"]),)}
    library=NS(view=lambda now:scenes,weights={"A":Fraction(1,2),"B":Fraction(1,2)})
    slots=[dict(vehicle_id="AV",native_id=1,profile_id="C",activation_s=0.,
                admission_end_s=1800.,position=left["position"]),
           dict(vehicle_id="HV",native_id=2,profile_id="HV",activation_s=0.,
                admission_end_s=1800.,position=(108.8,34.2))]
    for strategy in ("BOTTLENECK_CLOSURE","SPATIAL_JOINT"):
        chosen,info=solve_birth_batch(slots,{1},(left,right),{"AV":left},library,
            NS(global_pace=.01,pace_by_slot={}),0.,strategy=strategy,wall_limit_s=5.)
        assert set(chosen)=={"AV"} and chosen["AV"]["site_id"]=="right"
        assert info["scenario_shared_birth_position"] and info["same_first_action_across_scenarios"]
        assert not info["target_day_requests_loaded"]
        assert info["expected_future_served"]==1
        assert info["max_joint_variables"]<=20000 and info["max_joint_nonzeros"]<=150000
    wrong=[dict(slots[0],activation_s=1800.),slots[1]]
    with pytest.raises(ValueError,match="births"):
        solve_birth_batch(wrong,{1},(left,right),{"AV":left},library,
            NS(global_pace=.01,pace_by_slot={}),0.)


def test_protocol_preparation_is_metadata_only_and_arms_share_conditions(tmp_path,monkeypatch):
    base=dict(test_date="20161031",profile_id="C",requested_q_a=.1,supply_seed=20260824,
        passenger_acceptance_rate=.7,solver_time_limit_s=10.,full_day=True,
        max_match_wait_s=300,max_pickup_eta_s=600,maximum_rss_mib=1536,
        minimum_initial_available_ram_mib=1536,scenario_timeout_s=10800)
    seed=dict(records=[{}]*47,placements={"opportunity_joint":{str(i):{} for i in range(851)}})
    declared=dict(version="bottleneck_whole_method_v1",seed_layout_sha256="seed")
    cfg=protocol.assemble(base,declared,seed,tmp_path)
    path=tmp_path/"config.json"; path.write_text("{}",encoding="utf-8")
    monkeypatch.setattr(protocol,"load",lambda *args:(cfg,declared,seed))
    monkeypatch.setattr(protocol,"code_sha",lambda *args:"unit-code")
    def forbidden(*args,**kwargs):
        pytest.fail("metadata preparation cannot start computation")
    monkeypatch.setattr(protocol,"layouts",forbidden)
    monkeypatch.setattr(protocol,"run",forbidden)
    monkeypatch.setattr(protocol,"worker",forbidden)
    result=protocol.prepare(tmp_path,Path("config.json"))
    assert result["status"]=="PREPARED_NOT_EXECUTED"
    assert not any(result[k] for k in ("simulation_started","layout_optimization_started",
                                      "routing_started","inference_started","new_layouts_produced"))
    fixed=("test_date","profile_id","requested_q_a","supply_seed","passenger_acceptance_rate",
           "solver_time_limit_s","max_match_wait_s","max_pickup_eta_s","maximum_rss_mib")
    for group in protocol.ARMS:
        c=json.loads((tmp_path/protocol.OUT/"setup"/f"{group}.json").read_text(encoding="utf-8"))
        assert all(c[k]==base[k] for k in fixed)
    a,b,c=(protocol.group_config(cfg,g) for g in protocol.ARMS)
    assert a["current_decision_mode"]=="INDEPENDENT_VALUE"
    assert a["joint_neighborhood_strategy"] is None
    assert b["current_decision_mode"]==c["current_decision_mode"]=="BOTTLENECK_JOINT"
    assert b["joint_neighborhood_strategy"]=="SPATIAL_JOINT"
    assert c["joint_neighborhood_strategy"]=="BOTTLENECK_CLOSURE"
    command=protocol.child_command(tmp_path,Path("config.json"),protocol.ARMS[0],tmp_path)
    assert command[3:7]==["-m","stage4.analysis.bottleneck_whole_method","worker","--execute"]
    with protocol.execution_lock(tmp_path):
        with pytest.raises(RuntimeError,match="lock"):
            with protocol.execution_lock(tmp_path):
                pytest.fail("duplicate execution acquired the lock")


def test_v2_retains_reference_without_relabeling_or_launching_it(tmp_path,monkeypatch):
    base=dict(test_date="20161031",profile_id="C",requested_q_a=.1,supply_seed=20260824,
        passenger_acceptance_rate=.7,solver_time_limit_s=10.,full_day=True,
        max_match_wait_s=300,max_pickup_eta_s=600,maximum_rss_mib=1536,
        minimum_initial_available_ram_mib=1536,scenario_timeout_s=10800)
    seed=dict(records=[{}]*47,placements={"opportunity_joint":{str(i):{} for i in range(851)}})
    declared=dict(version=protocol.V2,reuse_layout="retained/layout.json",
        retained_reference_summary="retained/summary.json",retained_reference_receipt="retained/receipt.json",
        seed_layout_sha256="seed")
    source=tmp_path/"retained"; source.mkdir(); (source/"layout.json").write_text("{}",encoding="utf-8")
    cfg=protocol.assemble(base,declared,seed,tmp_path)
    old=protocol.group_config(cfg,protocol.ARMS[0])
    for k in ("joint_execution_mode","retained_reference_summary","retained_reference_receipt"):
        old.pop(k,None)
    old.update(version="bottleneck_whole_method_v1",process_group_memory_limit_mib=1536)
    summary=dict(status=protocol.COMPLETE,group=protocol.ARMS[0],full_day=True,frozen_protocol=old,
                 execution_binding={"code_sha":"old-code"})
    (source/"summary.json").write_text(json.dumps(summary),encoding="utf-8")
    (source/"receipt.json").write_text(json.dumps(dict(status="COMPLETE",exit_code=0,parent_hard_timeout=False)),encoding="utf-8")
    assert protocol.retained_reference(tmp_path,cfg)["execution_binding"]["code_sha"]=="old-code"
    assert protocol.source_directory(tmp_path,cfg,protocol.ARMS[0])==source
    assert protocol.output_root(cfg).name==protocol.V2
    assert protocol.group_config(cfg,protocol.ARMS[1])["joint_execution_mode"]=="INCREMENTAL_V2"
    config=tmp_path/"config.json"; config.write_text("{}",encoding="utf-8")
    monkeypatch.setattr(protocol,"load",lambda *args:(cfg,declared,seed))
    monkeypatch.setattr(protocol,"code_sha",lambda *args:"new-code")
    prepared=protocol.prepare(tmp_path,Path("config.json"))
    assert prepared["reference_retained"] and prepared["native_group_count"]==2
    assert not prepared["simulation_started"] and not prepared["layouts_recomputed"]
    assert not (tmp_path/protocol.output_root(cfg)/"native").exists()
    with pytest.raises(ValueError,match="no reference worker"):
        protocol.worker(tmp_path,Path("config.json"),protocol.ARMS[0],tmp_path)
    old["max_match_wait_s"]=301
    (source/"summary.json").write_text(json.dumps(summary),encoding="utf-8")
    with pytest.raises(ValueError,match="different physical/scientific"):
        protocol.retained_reference(tmp_path,cfg)
