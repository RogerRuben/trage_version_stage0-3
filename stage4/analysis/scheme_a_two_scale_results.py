"""Descriptive comparison of the three completed two-scale city groups only."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage4.analysis.capability_chain_instances import atomic_json


OUT = Path("stage4/output/capability_chain_planning/two_scale_city_v2")
DOC = Path("stage4/docs/capability_chain_planning/two_scale_city_v2")
GROUPS = ("HOTSPOT_BASELINE", "CHAIN_LAYOUT_BASELINE", "CHAIN_LAYOUT_JOINT")
POLICIES = ("SERVICE_PRESERVING", "SERVICE_PRESERVING", "CHAIN_DEFER")
LAYOUTS = ("hotspot", "joint", "joint")


def collect(root):
    root = Path(root).resolve()
    public, private, shared_inputs = {}, {}, None
    for group,policy,layout in zip(GROUPS,POLICIES,LAYOUTS):
        directory = root/OUT/group
        summary = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
        if (summary["status"] != "TWO_SCALE_CITY_GROUP_COMPLETE" or not summary["full_day"]
                or summary["policy"] != policy or summary["layout_mode"] != layout):
            raise ValueError("comparison requires all three completed fixed full-day groups")
        common = summary["inputs_sha256"],summary["code_sha"],summary["layout_sha256"]
        if shared_inputs is not None and common != shared_inputs:
            raise ValueError("code, input, configuration or completed layout differs across groups")
        shared_inputs = common
        outcome = pd.read_parquet(directory/"cohort_outcomes.parquet")
        assignment = pd.read_parquet(directory/"assignments.parquet")
        trace = pd.read_parquet(directory/"solver_trace.parquet")
        epochs = pd.read_parquet(directory/"epochs.parquet")
        moves = pd.read_parquet(directory/"empty_movements.parquet")
        steps = pd.read_parquet(directory/"step_timings.parquet")
        coarse_history = json.loads((directory/"coarse_refreshes.json").read_text(encoding="utf-8"))
        scenes = [scene for record in coarse_history for scene in record["scenario_info"]]
        cohort = set(outcome.order_id.astype(str))
        if len(cohort) != len(outcome) or len(outcome) != summary["revealed_orders"]:
            raise ValueError("duplicate or missing common request")
        if private and cohort != private[GROUPS[0]]["cohort"]:
            raise ValueError("the three policies do not share exactly the same request population")
        if (not assignment.order_id.is_unique
                or not ((outcome.matched.astype(int)+outcome.expired.astype(int))==1).all()
                or not outcome.completed.equals(outcome.matched)
                or not outcome.wait_s.dropna().between(0.,300.+1e-6).all()):
            raise ValueError("final service, physical drain, uniqueness or patience accounting failed")
        if (summary["physical_accounting"]["duplicate_orders"]
                or summary["physical_accounting"]["vehicle_task_overlap"]
                or summary["physical_accounting"]["maximum_time_error_s"] > 1e-6):
            raise ValueError("reported physical conservation failed")
        av = assignment.loc[assignment.vehicle_type.eq("AV")]
        av_counts = av.groupby("native_vehicle_id").size()
        planned = summary["all_day_supply"]["accounting"]
        busy = (pd.to_datetime(assignment.service_end_time)-pd.to_datetime(assignment.assignment_time)).dt.total_seconds()
        av_busy = (pd.to_datetime(av.service_end_time)-pd.to_datetime(av.assignment_time)).dt.total_seconds()
        av_loaded = (pd.to_datetime(av.service_end_time)-pd.to_datetime(av.pickup_time)).dt.total_seconds()
        pickup_km = float(assignment.pickup_route_distance_m.sum()/1000.)
        move_km = float(moves.actual_distance_m.sum()/1000.) if len(moves) else 0.
        active_steps = steps.loc[steps.completed_native_epoch_records.gt(0)]
        eligible = outcome.C_compatible & outcome.passenger_accepts_av
        refreshes = trace.loc[trace.coarse_refreshed]
        row = dict(policy=policy,layout_mode=layout,common_requests=len(outcome),
            completed=int(outcome.completed.sum()),expired=int(outcome.expired.sum()),
            service_rate=float(outcome.completed.mean()),served_C=len(av),served_HV=len(assignment)-len(av),
            C_body_compatible_orders=int(outcome.C_compatible.sum()),
            C_after_passenger_acceptance_orders=int(eligible.sum()),
            C_eligible_served_by_HV=int((eligible & outcome.vehicle_type.eq("HV")).sum()),
            served_wait_mean_s=float(outcome.wait_s.mean()),served_wait_p90_s=float(outcome.wait_s.quantile(.9)),
            pickup_empty_km=pickup_km,idle_empty_km=move_km,total_empty_km=pickup_km+move_km,
            actual_idle_moves=len(moves),movement_status= moves.status.value_counts().to_dict() if len(moves) else {},
            C_serving_slots=len(av_counts),C_slots_serving_at_least_two=int((av_counts>=2).sum()),
            C_max_services_per_slot=int(av_counts.max()) if len(av_counts) else 0,
            C_services_per_planned_vehicle_hour=len(av)/planned["achieved_av_vehicle_hours"],
            C_customer_busy_hours=float(av_busy.sum()/3600.),C_loaded_hours=float(av_loaded.sum()/3600.),
            total_customer_busy_hours=float(busy.sum()/3600.),
            planned_AV_vehicle_hours=planned["achieved_av_vehicle_hours"],
            planned_HV_vehicle_hours=planned["achieved_hv_vehicle_hours"],
            baseline_vehicle_hours=planned["h_base_exact"],planned_AV_hour_ratio=planned["achieved_q_a"],
            supply_slots=summary["all_day_supply"]["slot_count"],
            runtime_s=summary["total_runner_wall_s"],setup_time_s=summary["setup_time_s"],
            simulation_loop_wall_s=summary["simulation_loop_wall_s"],peak_rss_mib=summary["peak_rss_mib"],
            active_step_p50_s=float(active_steps.step_wall_s.median()),
            active_step_p90_s=float(active_steps.step_wall_s.quantile(.9)),
            maximum_complete_step_wall_s=float(steps.step_wall_s.max()),
            maximum_adapter_wall_s=float(trace.solve_wall_time_s.max()),
            candidate_generation_time_s=float(epochs.candidate_generation_time_s.sum()),
            candidate_routing_time_s=float(epochs.routing_time_s.sum()),
            adapter_timings_s=summary["adapter_performance"]["timings_s"],
            maximum_current_actions=int(trace.variables.max()),maximum_current_residual_edges=int(trace.nonzeros.max()),
            decision_count=len(trace),current_flow_status=trace.opt_status.value_counts().to_dict(),
            effective_policy_counts=trace.effective_policy.value_counts().to_dict(),
            coarse_refresh_count=len(refreshes),
            unavailable_coarse_refresh_count=int((~refreshes.coarse_available).sum()),
            pricing_closed_refresh_count=sum(record["all_pricing_closed"] for record in coarse_history),
            coarse_record_count=len(coarse_history),
            maximum_sparse_LP_columns=max((scene["columns"] for scene in scenes),default=0),
            maximum_sparse_LP_nonzeros=max((scene["nonzeros"] for scene in scenes),default=0),
            adapter_wall_overrun_count=int(trace.adapter_wall_deadline_exceeded.sum()),
            explicit_fallback_decisions=int(trace.resource_fallback.sum()),
            optional_move_budget_cutoffs=int(trace.optional_move_budget_cutoff.sum()),
            move_screen_counts=summary["adapter_performance"]["operations"],
            physical_accounting=summary["physical_accounting"],
            future_connector_queries=summary["adapter_performance"]["future_connector_queries"],
            coarse_prices_are_approximate=True,source_code_sha=summary["code_sha"])
        if row["completed"] != summary["completed_orders"] or row["future_connector_queries"]:
            raise ValueError("summary/product count mismatch or old future-routing path entered")
        outcome["release_hour"]=(outcome.release_s//3600).astype(int)
        row["hourly"] = [dict(hour=int(hour),requests=len(day),completed=int(day.completed.sum()),
            expired=int(day.expired.sum()),served_C=int(day.vehicle_type.eq("AV").sum()),
            served_HV=int(day.vehicle_type.eq("HV").sum()),service_rate=float(day.completed.mean()))
            for hour,day in outcome.groupby("release_hour",sort=True)]
        served=outcome.loc[outcome.completed]
        private[group]=dict(cohort=cohort,served=set(served.order_id.astype(str)),
            wait=dict(zip(served.order_id.astype(str),served.wait_s)))
        public[group]=row
        del outcome,assignment,trace,epochs,moves,steps,av,served
        gc.collect()
    pairs={}
    for label,before,after in (("layout_effect",GROUPS[0],GROUPS[1]),
            ("control_effect",GROUPS[1],GROUPS[2]),("combined_effect",GROUPS[0],GROUPS[2])):
        b,a=private[before],private[after]
        shared=b["served"] & a["served"]
        waits=[a["wait"][order]-b["wait"][order] for order in sorted(shared)]
        pairs[label]=dict(before=before,after=after,
            net_service_difference=public[after]["completed"]-public[before]["completed"],
            service_rate_difference_percentage_points=100*(public[after]["service_rate"]-public[before]["service_rate"]),
            gained_service_count=len(a["served"]-b["served"]),lost_service_count=len(b["served"]-a["served"]),
            shared_served_count=len(shared),
            shared_served_wait_mean_change_s=float(np.mean(waits)) if waits else None,
            C_service_difference=public[after]["served_C"]-public[before]["served_C"],
            HV_service_difference=public[after]["served_HV"]-public[before]["served_HV"],
            total_empty_km_difference=public[after]["total_empty_km"]-public[before]["total_empty_km"])
    result=dict(status="TWO_SCALE_THREE_GROUP_RESULTS_ANALYZED",groups=public,paired=pairs,
        common_cohort_count=len(private[GROUPS[0]]["cohort"]),inputs_sha256=shared_inputs[0],
        code_sha=shared_inputs[1],layout_sha256=shared_inputs[2],
        test_date="20161031",profile="C",q_a=.1,passenger_acceptance=.7,
        one_day_one_seed=True,statistical_significance_claimed=False,
        predictor_decision_superiority_claimed=False,global_city_optimality_claimed=False,
        no_new_experiments=True,per_order_or_coordinate_information_exported=False)
    atomic_json(root/DOC/"comparison.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    args=parser.parse_args()
    result=collect(args.root)
    print(json.dumps(dict(status=result["status"],paired=result["paired"]),ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
