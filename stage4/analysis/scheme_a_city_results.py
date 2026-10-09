"""Read completed Scheme-A products only; no native execution or refitting."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage4.analysis.capability_chain_instances import atomic_json
from stage4.analysis.scheme_a_city_experiment import OUT, DOC, GROUPS


def collect(root):
    public, per_order = {}, {}
    for group in GROUPS:
        directory = root/OUT/group
        summary = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
        if summary["status"] != "SCHEME_A_CITY_GROUP_COMPLETE":
            raise ValueError("only all-three completed groups can produce the comparison")
        outcome = pd.read_parquet(directory/"cohort_outcomes.parquet")
        assignments = pd.read_parquet(directory/"assignments.parquet")
        traces = pd.read_parquet(directory/"solver_trace.parquet")
        if len(outcome) != 28367 or not outcome.order_id.is_unique or not assignments.order_id.is_unique:
            raise ValueError("common cohort reconciliation failed")
        if not ((outcome.matched.astype(int)+outcome.expired.astype(int)) == 1).all():
            raise ValueError("request not fully assigned or expired")
        if not outcome.completed.equals(outcome.matched):
            raise ValueError("committed physical work incomplete")
        if not outcome.wait_s.dropna().between(0,300+1e-6).all():
            raise ValueError("original pickup deadline violated")
        av = assignments.loc[assignments.vehicle_type.eq("AV")]
        counts = av.groupby("native_vehicle_id").size()
        planned = summary["all_day_supply"]["accounting"]
        busy_s = (pd.to_datetime(av.service_end_time)-pd.to_datetime(av.assignment_time)).dt.total_seconds()
        loaded_s = (pd.to_datetime(av.service_end_time)-pd.to_datetime(av.pickup_time)).dt.total_seconds()
        row = dict(completed=summary["completed"], service_rate=summary["service_rate"],
            expired=summary["expired"], served_C=summary["served_C"], served_HV=summary["served_HV"],
            served_wait_mean_s=float(outcome.wait_s.mean()), served_wait_p90_s=float(outcome.wait_s.quantile(.9)),
            pickup_empty_km=summary["pickup_empty_distance_m"]/1000,
            idle_empty_km=summary["idle_movement"]["empty_distance_m"]/1000,
            total_empty_km=(summary["pickup_empty_distance_m"]+summary["idle_movement"]["empty_distance_m"])/1000,
            actual_idle_moves=summary["idle_movement"]["movement_rows"],
            C_serving_slots=int(len(counts)), C_slots_serving_at_least_two=int((counts>=2).sum()),
            C_max_services_per_slot=int(counts.max()) if len(counts) else 0,
            C_services_per_planned_vehicle_hour=float(len(av)/planned["achieved_av_vehicle_hours"]),
            C_customer_busy_hours=float(busy_s.sum()/3600), C_loaded_hours=float(loaded_s.sum()/3600),
            total_customer_busy_hours=float(((pd.to_datetime(assignments.service_end_time)
                -pd.to_datetime(assignments.assignment_time)).dt.total_seconds()).sum()/3600),
            customer_work_after_calendar_day_hours=float((assignments.service_end_time_s-86400).clip(lower=0).sum()/3600),
            runtime_s=summary["runtime_s"], peak_rss_mib=summary["peak_rss_mib"],
            max_total_OR_s=summary["maximum_total_or_s"], OR_total_s=summary["OR_total_s"],
            max_variables=summary["maximum_model_variables"], max_nonzeros=summary["maximum_model_nonzeros"],
            selected_future_multi_service_records=summary["selected_multi_service_chain_records"],
            selected_future_move_records=summary["selected_future_move_records"],
            fixed_current_action_rounds=int(traces.opt_status.eq("FIXED_CURRENT_ACTIONS").sum()),
            physical_accounting=summary["physical_accounting"],
            planned_AV_hour_ratio=planned["achieved_q_a"], full_day_slot_count=summary["all_day_supply"]["slot_count"],
            source_code_sha=summary["code_sha"], layout_mode=summary["layout_mode"], policy=summary["policy"])
        per_order[group] = dict(served=set(outcome.loc[outcome.completed,"order_id"]),
            wait=dict(zip(outcome.loc[outcome.completed,"order_id"],outcome.loc[outcome.completed,"wait_s"])))
        public[group] = row
        del outcome, assignments, traces, av
        gc.collect()
    pairs = {}
    for name, before, after in (("layout_effect",GROUPS[0],GROUPS[1]),
        ("control_effect",GROUPS[1],GROUPS[2]),("combined_effect",GROUPS[0],GROUPS[2])):
        b,a = per_order[before],per_order[after]
        shared = b["served"]&a["served"]
        delta = [a["wait"][oid]-b["wait"][oid] for oid in shared]
        pairs[name] = dict(before=before, after=after,
            net_service_difference=public[after]["completed"]-public[before]["completed"],
            service_rate_difference_percentage_points=100*(public[after]["service_rate"]-public[before]["service_rate"]),
            gained_service_count=len(a["served"]-b["served"]), lost_service_count=len(b["served"]-a["served"]),
            shared_served_count=len(shared), shared_served_wait_mean_change_s=float(np.mean(delta)) if delta else None,
            shared_served_wait_p90_change_s=float(np.quantile(delta,.9)) if delta else None,
            C_service_difference=public[after]["served_C"]-public[before]["served_C"],
            HV_service_difference=public[after]["served_HV"]-public[before]["served_HV"],
            total_empty_km_difference=public[after]["total_empty_km"]-public[before]["total_empty_km"])
    for group in GROUPS:
        if public[group]["full_day_slot_count"] != 8435:
            raise ValueError("supply slot count differs")
    if public[GROUPS[1]]["layout_mode"] != public[GROUPS[2]]["layout_mode"]:
        raise ValueError("control comparison did not share the joint layout")
    result = dict(status="SCHEME_A_THREE_GROUP_RESULTS_ANALYZED", common_cohort_count=28367,
        test_date="20161031", profile="C", q_a=.1, passenger_acceptance=.7,
        groups=public, paired=pairs, per_order_or_coordinate_information_exported=False,
        one_day_one_seed=True, statistical_significance_claimed=False,
        restricted_chain_domain=True, global_city_optimality_claimed=False,
        predictor_decision_superiority_claimed=False, no_new_experiments=True)
    atomic_json(root/DOC/"comparison.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    args=parser.parse_args()
    result=collect(args.root.resolve())
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
