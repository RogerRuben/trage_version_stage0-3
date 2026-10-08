"""Only complete-window construction and safe profile pruning checks."""
import pandas as pd

from stage4.analysis.capability_chain_local import complete_window, dense_sites
from stage4.analysis.capability_chain_rolling import epoch_problem


def test_local_candidates_ignore_C_labels_and_complete_window_keeps_all_routes():
    cfg = dict(grid_origin_wgs84=[108,34], grid_size_degrees=.02, site_subgrid_degrees=.005,
        replay_date="20161024", horizon_s=1800)
    history = pd.DataFrame([dict(order_id=f"H{i}", date="20161010", start_lon_wgs84=x,
        start_lat_wgs84=34.221, compatible_C=False) for i,x in enumerate([108.901,108.902,108.908,108.914])])
    other = pd.DataFrame([dict(order_id="other",date="20161017",start_lon_wgs84=108.943,
        start_lat_wgs84=34.221,compatible_C=True)])
    history = pd.concat([history,other],ignore_index=True)
    region, sites, _ = dense_sites(history,cfg)
    changed = history.copy(); changed["compatible_C"] = ~changed.compatible_C
    region2, sites2, _ = dense_sites(changed,cfg)
    assert region == region2 == "45_11" and sites == sites2
    observed = set(zip(history.start_lon_wgs84,history.start_lat_wgs84))
    assert all((s["lon_wgs84"],s["lat_wgs84"]) in observed for s in sites)
    actual = pd.DataFrame([dict(order_id=f"A{i}",date="20161024",start_lon_wgs84=108.901,
        start_lat_wgs84=34.221,end_lon_wgs84=109.1,release_second=27000+i*300,
        compatible_C=i==2,common_eligible=True) for i in range(3)])
    complete = complete_window(actual,region,"20161024",27000,cfg)
    assert len(complete) == 3 and list(complete.order_id) == ["A0","A1","A2"]
    assert complete.end_lon_wgs84.eq(109.1).all()  # No favorable destination censoring.


def test_profile_pruning_preserves_unions_and_explicit_model_bounds():
    class Spy:
        def __init__(self): self.calls=[]
        def connection(self,origin,target,allowed):
            assert target in allowed
            self.calls.append((origin,target))
            return dict(supported=False)
    resources=[dict(resource_id="C1",profile_id="C",admission_end_s=600),
        dict(resource_id="H1",profile_id="HV",admission_end_s=600)]
    actions=[dict(resource_id="C1",location_id="S_C",ready_s=30),
        dict(resource_id="H1",location_id="S_H",ready_s=30)]
    tasks=[dict(job_id="F_H",release_s=30,deadline_s=330,service_time_s=30,compatible_profiles=["HV"]),
        dict(job_id="F_C",release_s=30,deadline_s=330,service_time_s=30,compatible_profiles=["HV","C"])]
    forecasts=[dict(scenario_id="history",weight=1,tasks=tasks)]
    cfg=dict(step_s=30,solver_decision_limit_s=10,prune_connection_profiles=True,
        task_limit_per_scenario=64,exact_chain_compression=True)
    spy=Spy();problem=epoch_problem(resources,[],forecasts,actions,spy,0,cfg)
    assert ("S_C","F_H") not in spy.calls and ("S_C","F_C") in spy.calls
    assert ("F_C","F_H") in spy.calls  # A common C/HV job can end with either type.
    assert problem["task_limit_per_scenario"]==64 and problem["exact_chain_compression"] is True
    resources[1]["resource_id"]="H2"
    actions[1]=dict(resource_id="H2",location_id="S_C",ready_s=30)
    spy=Spy();epoch_problem(resources,[],forecasts,actions,spy,0,cfg)
    assert ("S_C","F_H") in spy.calls  # Sharing an origin cannot erase the HV alternative.
