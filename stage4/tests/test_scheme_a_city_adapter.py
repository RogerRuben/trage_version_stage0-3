from collections import Counter
from types import SimpleNamespace

import pandas as pd
import pytest

from stage4.dispatch.flexibility_model import Vehicle
from stage4.dispatch.scheme_a_city_master import CurrentAction
from stage4.dispatch.scheme_a_city_graph import RouteState
from stage4.dispatch.scheme_a_city_adapter import NativeSchemeAAdapter, _column_allocation, apply_av_layout
from stage4.fleetpy_adapter.research_world import SupplyPlanRow
from stage4.dispatch.solver import AssignmentArc


def test_column_budget_keeps_current_actions_and_is_identity_deterministic():
    actions = [CurrentAction(f"V{v}:A{i}", v, "WAIT", None) for v in range(3) for i in range(2)]
    states = {a.action_id: RouteState((108.,34.),30.,1800.,"HV") for a in actions}
    q, info = _column_allocation(actions, ("h1","h2"), states, 14, 0,1800)
    assert sum(q.values()) == 8 and info["zero_column_budget_groups"] == 4
    assert q == _column_allocation(list(reversed(actions)), ("h2","h1"), states,14,0,1800)[0]
    assert len(actions) == 6 and max(q.values()) == 1


def test_layout_changes_only_av_origin_not_supply_or_hv_state():
    hv = SupplyPlanRow("H",1,"slotH","HV","HV",20.,900.,108.,34.,"STOP_ADMISSION_FINISH_COMMITTED")
    av = SupplyPlanRow("A",2,"slotA","AV","C",20.,900.,108.,34.,"STOP_ADMISSION_FINISH_COMMITTED")
    supply = SimpleNamespace(_plan=(hv,av),_by_vehicle={"H":hv,"A":av},_accounting={})
    supply.plan = lambda:supply._plan
    episode = SimpleNamespace(supply=supply)
    assert apply_av_layout(episode,{"A":{"position":(108.1,34.1)}}) == 1
    assert supply._plan[0] is hv
    assert supply._plan[1].activation_s == av.activation_s
    assert supply._plan[1].admission_end_s == av.admission_end_s
    assert supply._plan[1].slot_id == av.slot_id
    assert supply._plan[1].initial_lon_wgs84 == 108.1


def test_native_adapter_uses_real_vehicle_fields_and_original_arc_index(monkeypatch):
    from stage4.dispatch import scheme_a_city_adapter as adapter_module
    cfg = dict(test_date="20161031",layout_candidate_site_count=20,planning_horizon_s=1800,
        coarse_reference_update_s=300,rolling_step_s=30,patience_s=300,
        future_search_radius_m=2000,future_top_k_tasks=3,chain_beam_width=2,
        max_future_services=3,max_future_chains=2,max_future_relocations=2,
        max_chain_labels=5000,reposition_interval_s=900,reposition_top_k=3,
        reposition_max_eta_s=300,reposition_radius_m=2000,reposition_max_moves=50,
        movement_day_end_s=86400,admission_end_s=1800,
        solver_time_limit_s=10.,maximum_model_variables=20000,maximum_model_nonzeros=150000)
    point = (108.000000041,34.000000042)
    reference = pd.DataFrame([dict(time_bin_index=0,node_id=9,demand_share=1.,lon_wgs84=point[0],lat_wgs84=point[1])])
    library = SimpleNamespace(view=lambda now:{"h":[]},weights={"h":1.})
    class Connectors:
        timings = Counter()
        def __call__(self,*args):
            return dict(supported=False,compatible_profiles=[],travel_time_s=None,empty_distance_m=None)
    manager = SimpleNamespace(active={},rows=[],queue_actions=lambda moves,now:None)
    runtime = SimpleNamespace(native_vehicle=SimpleNamespace(pos=(9,None,None)))
    request = SimpleNamespace(compatible_profiles=frozenset({"HV"}),sim_time_s=0,
        predicted_service_time_s=60.,pickup_lon_wgs84=point[0],pickup_lat_wgs84=point[1],
        dropoff_lon_wgs84=108.001,dropoff_lat_wgs84=34.001,passenger_accepts_av=True,order_id="known")
    control = SimpleNamespace(assignment_rows=[],request_by_rid={5:request},
        request_meta={5:{"pickup_deadline_s":300.}},runtime_by_vid={7:runtime},
        fixture_windows_s={7:(0,1800)},repositioning_manager=manager,
        routing_engine=SimpleNamespace(return_position_coordinates=lambda pos:point),
        _available=lambda runtime,now:True)
    monkeypatch.setattr(adapter_module,"predicted_vehicle_states",lambda *args:
        (Vehicle(7,"HV","108.0000000,34.0000000",0.,1800.),))
    estimate = SimpleNamespace(route_distance_m=40.)
    arc = AssignmentArc(7,5,10.,True,False,(runtime,request,estimate),vehicle_type="HV")
    planner = NativeSchemeAAdapter("CHAIN_DEFER",library,reference,Connectors(),
        lambda c,arcs,now:arcs,None,cfg)
    result = planner.solve(control,[arc],[5],0)
    assert result.selected_indices == (0,) and result.total_matched == 1
    assert planner.rows[-1]["variables"] > 0


def test_offline_graph_and_master_budgets_are_separate_and_realtime_stays_10(monkeypatch):
    from stage4.dispatch import scheme_a_city_adapter as module
    cfg = dict(layout_graph_cpu_limit_s=60., solver_time_limit_s=10., planning_horizon_s=1800,
        admission_end_s=1800, maximum_model_variables=20000,maximum_model_nonzeros=150000,
        reposition_max_moves=50,coarse_reference_update_s=300)
    action = CurrentAction("L",1,"LAYOUT",None)
    starts = {"L":RouteState((108.,34.),0.,1800.,"C")}
    connector = SimpleNamespace(timings=Counter())
    info = dict(max_service_chain_length=0,max_future_relocation_count=0)
    monkeypatch.setattr(module,"build_restricted_chains",lambda *args:([],info))
    limits=[]
    scopes=[]
    def master(*args,**kwargs):
        limits.append(kwargs["time_limit_s"])
        scopes.append(kwargs["offline_layout"])
        return dict(selected_chain_ids=(),runtime_s=.1,model={})
    monkeypatch.setattr(module,"solve_city_master",master)
    ticks=iter([0.,12.,12.,12.])
    monkeypatch.setattr(module,"perf_counter",lambda:next(ticks,12.))
    result=module.build_and_solve([action],starts,{"h":[]},{"h":1.},connector,[],cfg,0,"CHAIN_DEFER",offline_layout=True)
    assert limits == [10.] and result["graph_or_time_s"] == 12.
    assert result["offline_layout_graph_budget_separate"]
    ticks=iter([0.,12.,12.,12.])
    with pytest.raises(TimeoutError,match="OR budget"):
        module.build_and_solve([action],starts,{"h":[]},{"h":1.},connector,[],cfg,0,"CHAIN_DEFER")
    assert limits == [10.]
    cfg["layout_solver_time_limit_s"] = 30.
    ticks=iter([0.,12.,12.,12.])
    result=module.build_and_solve([action],starts,{"h":[]},{"h":1.},connector,[],cfg,0,"CHAIN_DEFER",offline_layout=True)
    assert limits == [10.,30.] and result["integer_master_limit_s"] == 30.
    ticks=iter([0.,12.,12.,12.])
    with pytest.raises(TimeoutError,match="OR budget"):
        module.build_and_solve([action],starts,{"h":[]},{"h":1.},connector,[],cfg,0,"CHAIN_DEFER")
    ticks=iter([0.,.5,.5,.5])
    monkeypatch.setattr(module,"perf_counter",lambda:next(ticks,.5))
    result=module.build_and_solve([action],starts,{"h":[]},{"h":1.},connector,[],cfg,0,"CHAIN_DEFER")
    assert limits[-1] == 9.5 and result["integer_master_limit_s"] == 9.5
    assert scopes == [True,True,False]


def test_prefix_recovery_allows_only_hash_bound_offline_10_to_30_budget(tmp_path):
    import json
    from pathlib import Path
    from stage4.analysis.scheme_a_city_experiment import recover_layout_prefix
    from stage4.analysis.capability_chain_instances import sha
    old=dict(solver_time_limit_s=10.,layout_graph_cpu_limit_s=60.,max_future_services=3,seed=7)
    snapshot=tmp_path/"configuration_at_run.json"
    snapshot.write_text(json.dumps(old),encoding="utf-8")
    prior={"cfg.json":sha(snapshot),"history.parquet":"unchanged-history"}
    current={"cfg.json":"new-config","history.parquet":"unchanged-history"}
    partial=dict(completed_bins=38,total_bins=47)
    recovered=dict(inputs_sha256=prior,partial=partial,code_sha="old-code")
    cfg=dict(old,layout_solver_time_limit_s=30.)
    restored,info=recover_layout_prefix(cfg,current,tmp_path,recovered,Path("cfg.json"))
    assert restored is partial and info["old_value_s"] == 10. and info["new_value_s"] == 30.
    with pytest.raises(ValueError,match="only the authorized"):
        recover_layout_prefix(dict(cfg,max_future_services=4),current,tmp_path,recovered,Path("cfg.json"))
    with pytest.raises(ValueError,match="raw inputs changed"):
        recover_layout_prefix(cfg,dict(current,**{"history.parquet":"changed"}),tmp_path,recovered,Path("cfg.json"))


def test_full_day_history_source_rejects_window_manifest_and_truncated_rows(tmp_path):
    import json
    from stage4.analysis.scheme_a_city_experiment import load_history_templates
    dates = ["20161010", "20161017", "20161024"]
    rows = [dict(date=d, order_id=f"{d}_{h}", release_second=h*3600.+30.,
        start_lon_wgs84=108., start_lat_wgs84=34., end_lon_wgs84=108.01, end_lat_wgs84=34.01,
        predicted_route_time_p50_s=60., compatible_C=True, compatible_M=True, compatible_A=True,
        research_data_ready=True) for d in dates for h in range(24)]
    frame = pd.DataFrame(rows)
    frame.to_parquet(tmp_path/"history.parquet", index=False)
    manifest = dict(status="COMPLETE", full_day_Train_library_not_test31_future_demand=True,
        train_templates=[dict(date=d, templates=24) for d in dates])
    (tmp_path/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cfg = dict(source_history_templates="history.parquet", source_history_manifest="manifest.json",
        forecast_train_dates=dates)
    loaded, inventory = load_history_templates(tmp_path, cfg)
    assert len(loaded) == 72 and inventory["full_day_source_verified"]
    assert all(len(v["hourly_row_counts"]) == 24 for v in inventory["dates"].values())
    manifest["full_day_Train_library_not_test31_future_demand"] = False
    (tmp_path/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="full-day"):
        load_history_templates(tmp_path, cfg)
    manifest["full_day_Train_library_not_test31_future_demand"] = True
    (tmp_path/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    frame.loc[frame.release_second.between(36000, 43200)].to_parquet(tmp_path/"history.parquet", index=False)
    with pytest.raises(ValueError, match="complete-day source counts"):
        load_history_templates(tmp_path, cfg)
