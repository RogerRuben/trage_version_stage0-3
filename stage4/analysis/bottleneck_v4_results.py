"""Analyze one completed V4 against retained V3 products; never run experiments."""
import gc
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import pandas as pd

from .bottleneck_operational_v4 import OUT,DOC,load
from .bottleneck_whole_method import code_sha
from .capability_chain_instances import atomic_json,sha
from .c_operations_diagnostic import START
from .operations_v5_results import _collect,_peak,ORDER_COLUMNS
from .operations_v5_ablation_results import metric_differences,paired_counts
from stage4.dispatch.controlled_fleet import build_controlled_fleet


def analyze(root):
    root=Path(root); started=perf_counter(); cfg=load(root)
    group=cfg["group"]; directory=root/OUT/"native"/group
    current=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    if (current["status"]!="OPERATIONAL_V4_NATIVE_COMPLETE" or not current["full_day"]
            or current["group"]!=group or current["code_sha"]!=current["execution_binding"]["code_sha"]):
        raise ValueError("only the completed single V4 full-day result may be analyzed")
    olddoc=root/"stage4/docs/capability_chain_planning/bottleneck_whole_method_v3/comparison.json"
    retained=json.loads(olddoc.read_text(encoding="utf-8"))
    previous=dict(retained["groups"])
    paths={name:root/("stage4/output/capability_chain_planning/bottleneck_whole_method_v1/native/"+name
        if name=="INDEPENDENT_REFERENCE" else
        "stage4/output/capability_chain_planning/bottleneck_whole_method_v3/native/"+name) for name in previous}
    old=json.loads((paths["BOTTLENECK_SHARED_PLAN"]/"summary.json").read_text(encoding="utf-8"))
    controls=("test_date","profile_id","requested_q_a","passenger_acceptance_rate","passenger_acceptance_seed",
        "supply_seed","forecast_seed","source_layout_sha256","layout_mode","max_match_wait_s","max_pickup_eta_s",
        "post_assignment_cancellation","solver_time_limit_s","scenario_timeout_s","maximum_rss_mib",
        "rolling_step_s","current_top_k","measurement_end_s","admission_end_s","physical_drain_limit_s",
        "joint_maximum_coalition","joint_maximum_partial_states","joint_maximum_labels")
    if any(old["frozen_protocol"].get(k)!=current["frozen_protocol"].get(k) for k in controls):
        raise ValueError("V4 changed a fixed data, physical, layout, or resource control")
    for path,digest in current["inputs_sha256"].items():
        if sha(root/path)!=digest:
            raise ValueError("V4 frozen input bytes changed")
        if path in old["inputs_sha256"] and old["inputs_sha256"][path]!=digest:
            raise ValueError("shared V3/V4 input differs")
    fleet=build_controlled_fleet(root,benchmark_start=START,
        simulation_end=START+pd.Timedelta(seconds=cfg["physical_drain_limit_s"]),
        requested_q_a=cfg["requested_q_a"],seed=cfg["supply_seed"])
    labels=fleet.accounting["type_label_sha256"]
    if current["all_day_supply"]["accounting"]["type_label_sha256"]!=labels:
        raise ValueError("V4 effective supply labels differ")
    plans={int(r.native_id):(str(r.vehicle_type),float((r.availability_start_time-START).total_seconds()),
        float((r.availability_end_time-START).total_seconds())) for r in fleet.scenario_fleet.itertuples(index=False)}
    del fleet
    row,after=_collect(root,group,current,plans,directory=directory)
    row["joint_control_operations"]=current["adapter_performance"]["operations"]
    row["joint_phase_times_s"]=current["adapter_performance"]["joint_timings_s"]
    row["secondary_distance"]["scope"]="JOINT_PLAN_NOT_CURRENT_MAXIMUM_SERVICE_FACE"
    eligible=after.C_compatible & after.passenger_accepts_av
    hours=(after.release_s//3600).astype(int)
    for hour in row["hourly"]:
        hour["C_eligible_requests"]=int((eligible & hours.eq(hour["hour"])).sum())
    competition=pd.read_parquet(directory/"cohort_outcomes.parquet",columns=["order_id",
        "typed_last_min_AV_eta_s","typed_AV_alternatives_idle_after_choice"]).set_index("order_id")
    eta=pd.read_parquet(directory/"assignments.parquet",columns=["order_id","pickup_eta_s"]).set_index("order_id")
    mask=eligible & after.vehicle_type.eq("HV") & competition.typed_AV_alternatives_idle_after_choice.gt(0)
    difference=competition.loc[mask,"typed_last_min_AV_eta_s"]-eta.pickup_eta_s.reindex(competition.index[mask])
    if difference.isna().any(): raise ValueError("missing ETA in the final-C competition subset")
    row["C_pickup_competition"]=dict(request_count=int(mask.sum()),
        minimum_C_minus_selected_HV_eta_p50_s=float(difference.median()),
        minimum_C_no_slower_count=int(difference.le(1e-9).sum()),
        minimum_C_extra_30s_or_less_count=int(difference.le(30).sum()),
        scope="ADMITTED_GRAPH_MINIMUM_OVER_ALL_C_NOT_NECESSARILY_THE_IDLE_C_NOT_A_SWAP_COUNTERFACTUAL")
    del competition,eta,difference
    operations=current["adapter_performance"]["operations"]
    connector=current["connector"]["counts"]
    row["move_certification_partition"]=dict(examined=operations["current_move_routed_count"],
        connection_support_passed=connector.get("supported_connections",0),
        connection_support_rejected=connector.get("rejected_connections",0),
        certified_current_moves=operations.get("lazy_move_certified_moves",0),
        supported_but_move_time_or_admission_conditions_rejected=connector.get("supported_connections",0)
            -operations.get("lazy_move_certified_moves",0),executed=row["actual_idle_moves"],
        not_all_certification_rejections_are_road_or_profile_failures=True)
    statuses=pd.read_parquet(directory/"solver_trace.parquet",columns=["joint_control","resource_fallback"])
    row["fallback_classes"]=dict(Counter(r.joint_control.get("status")
        for r in statuses.loc[statuses.resource_fallback].itertuples(index=False) if r.joint_control is not None))
    del statuses
    pairs={}
    for name,public in previous.items():
        for product,digest in public["source_products_sha256"].items():
            if sha(paths[name]/product)!=digest:
                raise ValueError("retained result product changed")
        before=pd.read_parquet(paths[name]/"cohort_outcomes.parquet",columns=ORDER_COLUMNS).set_index("order_id")
        distance=pd.read_parquet(paths[name]/"assignments.parquet",columns=["order_id","pickup_route_distance_m"])
        if not before.index.is_unique or not distance.order_id.is_unique:
            raise ValueError("retained paired request identity is duplicated")
        before["pickup_distance_m"]=distance.set_index("order_id").pickup_route_distance_m
        pairs[name+"_to_V4"]=dict(difference_after_minus_before=metric_differences(public,row),
            paired=paired_counts(before,after))
        del before,distance
        gc.collect(); _peak()
    result=dict(status="OPERATIONAL_V4_FULL_DAY_ANALYZED",groups={**previous,group:row},comparisons=pairs,
        common_requests=28367,source_code_sha=current["code_sha"],analysis_code_sha=code_sha(root),retained_report_sha256=sha(olddoc),
        same_birth_layout_as="BOTTLENECK_SHARED_PLAN",same_type_and_admission_calendar=True,
        new_full_day_run_count=1,retained_results_rerun=False,
        analysis_invoked_simulation=False,analysis_invoked_routing=False,analysis_invoked_inference=False,
        one_day_one_seed=True,statistical_significance_claimed=False,global_optimality_claimed=False,
        individual_component_effects_identified=False,raw_identifiers_or_coordinates_exported=False,
        runtime_s=perf_counter()-started,peak_rss_mib=_peak())
    atomic_json(root/DOC/"comparison.json",result)
    return result


if __name__=="__main__":
    result=analyze(Path.cwd())
    print(json.dumps({"status":result["status"],"runtime_s":result["runtime_s"],
                      "peak_rss_mib":result["peak_rss_mib"]}),flush=True)
