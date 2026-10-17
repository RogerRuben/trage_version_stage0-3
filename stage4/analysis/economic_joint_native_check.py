"""One fixed 08:15--08:30 native integration check, never a full-day launcher.

Reuse the physical runner, q10 supply and completed birth layout. The check
does not choose parameters, compare cold-start service rates, or refit models.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace as NS

from . import economic_joint_dispatch as base
from .scheme_a_two_scale_check import _run_native,_sha,_atomic_json,FLEETPY


CONFIG=Path("stage4/config/economic_joint_native_check_v1.json")
OUT=Path("stage4/output/capability_chain_planning/economic_joint_v1/native_check")
DOC=Path("stage4/docs/capability_chain_planning/economic_joint_v1")
COMPLETE="ECONOMIC_JOINT_NATIVE_SHORT_CHECK_COMPLETE"
FIXED=dict(version="economic_joint_native_check_v1",economic_controller="SUBSTITUTION",
    group="ECONOMIC_SUBSTITUTION_NATIVE_CHECK",window_start_s=29700,measurement_end_s=30600,
    admission_end_s=30900,last_dispatch_s=30900,physical_drain_limit_s=37800,
    carry_in_lookback_s=300,include_carry_in=True,scenario_timeout_s=900,
    full_day=False,automatic_retry=False)


def load(root):
    root=Path(root)
    declared=json.loads((root/CONFIG).read_text(encoding="utf-8"))
    if (set(declared)!=set(FIXED)|{"base_config","base_config_sha256"}
            or any(declared[k]!=v for k,v in FIXED.items())
            or declared["base_config"]!=base.CONFIG.as_posix()
            or declared["base_config_sha256"]!=_sha(root/base.CONFIG)):
        raise ValueError("only the fixed single economic native short check is supported")
    cfg=base.load(root,controller=declared["economic_controller"])
    cfg.update(declared,groups=[declared["group"]],
        disk_route_cache=(OUT/"native/routing/cache.sqlite3").as_posix(),
        experiment_scope="ONE_COLD_START_NATIVE_INTERFACE_CHECK_NOT_POLICY_COMPARISON")
    return cfg


def run(root,fleetpy=FLEETPY):
    from .bottleneck_whole_method import execution_lock
    root=Path(root).resolve(); cfg=load(root)
    with execution_lock(root,OUT):
        return _run_native(root,fleetpy,CONFIG,cfg,directory=root/OUT/"native",
            doc_summary_path=root/OUT/"native_summary.json",completion_status=COMPLETE,
            group=cfg["group"])


def analyze(root):
    """Reconcile completed small products only; no simulation or road queries."""
    import pandas as pd
    from stage4.dispatch.economic_inputs import EconomicDistanceStore
    from stage4.dispatch.operating_economics import (OperatingEconomics,ORDER_LEDGER_COLUMNS,
        RELOCATION_LEDGER_COLUMNS)
    root=Path(root); cfg=load(root); directory=root/OUT/"native"
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    if (summary.get("status")!=COMPLETE or summary.get("full_day") is not False
            or not summary.get("native_execution_performed")):
        raise ValueError("requires a completed, physically drained economic native check")
    model=OperatingEconomics.from_config(cfg)
    store=EconomicDistanceStore.load(root/cfg["economic_distance_product"])
    data={name:pd.read_parquet(directory/(name+".parquet")) for name in (
        "assignments","cohort_outcomes","solver_trace","epochs","step_timings",
        "empty_movements","operating_order_ledger","operating_relocation_ledger")}
    assignments=data["assignments"]; orders=data["operating_order_ledger"]
    moves=data["empty_movements"]; move_ledger=data["operating_relocation_ledger"]
    if (set(orders)!=set(ORDER_LEDGER_COLUMNS) or set(move_ledger)!=set(RELOCATION_LEDGER_COLUMNS)
            or orders.native_request_id.duplicated().any()
            or assignments.native_request_id.duplicated().any()
            or set(orders.native_request_id)!=set(assignments.native_request_id)):
        raise AssertionError("financial order identity/schema reconciliation failed")
    assignment_by_id=assignments.set_index("native_request_id")
    totals=dict(fare_cents=0,driver_payout_cents=0,variable_cost_cents=0,contribution_cents=0)
    c_advantage=0
    for row in orders.itertuples(index=False):
        actual=assignment_by_id.loc[row.native_request_id]
        profile="HV" if actual.vehicle_type=="HV" else "C"
        if (not bool(actual.completed) or row.native_vehicle_id!=actual.native_vehicle_id
                or row.order_id!=actual.order_id or row.profile_id!=profile
                or row.economic_config_sha256!=model.fingerprint
                or abs(row.pickup_distance_m-float(actual.pickup_route_distance_m))>1e-6):
            raise AssertionError("committed financial quote differs from native assignment")
        distance=store.distance(cfg["test_date"],row.order_id)
        if abs(distance-row.service_distance_m)>1e-6:
            raise AssertionError("ledger does not use the complete declared service route")
        quote=model.quote(NS(service_distance_m=distance),profile,row.pickup_distance_m)
        for field in totals:
            if getattr(row,field)!=quote[field]: raise AssertionError("order contribution equation failed")
            totals[field]+=quote[field]
        if profile=="C":
            c_advantage+=row.contribution_cents-model.quote(NS(service_distance_m=distance),"HV",0)["contribution_cents"]
    if (len(moves)!=len(move_ledger) or move_ledger.duplicated(["native_vehicle_id","start_time_s"]).any()):
        raise AssertionError("one physical move must have one financial settlement")
    move_cost=0
    if len(moves):
        actual_moves=moves.set_index(["native_vehicle_id","start_time_s"],verify_integrity=True)
        for row in move_ledger.itertuples(index=False):
            actual=actual_moves.loc[row.native_vehicle_id,row.start_time_s]
            if (actual.status=="IN_PROGRESS" or abs(row.empty_distance_m-float(actual.actual_distance_m))>1e-6
                    or row.cost_cents!=model.movement_cost_cents(row.profile_id,row.empty_distance_m)):
                raise AssertionError("relocation settlement must use drained actual distance once")
            move_cost+=row.cost_cents
    expected=dict(fare_cny=totals["fare_cents"]/100,driver_payout_cny=totals["driver_payout_cents"]/100,
        variable_cost_cny=totals["variable_cost_cents"]/100,relocation_cost_cny=move_cost/100,
        operating_contribution_cny=(totals["contribution_cents"]-move_cost)/100)
    if any(abs(summary["operating_economics"][k]-v)>1e-7 for k,v in expected.items()):
        raise AssertionError("native financial summary does not equal unique executed ledgers")
    trace=data["solver_trace"]; active=trace.loc[trace.variables.gt(0)]
    joint=[]; floor_checks=0
    for row in trace.to_dict("records"):
        info=row.get("joint_control")
        if not isinstance(info,dict): continue
        joint.append(info)
        goal=info.get("protected_current_service")
        if goal is not None:
            if (row["current_selected"]<goal or row["current_carry_served"]<info["protected_current_carry"]):
                raise AssertionError("executed current counts violated the declared service floor")
            floor_checks+=1
    repairs=max(((i.get("search_work") or {}).get("economic_blocks_solved") or 0 for i in joint),default=0)
    if repairs>3: raise AssertionError("shared three-block epoch budget was exceeded")
    steps=data["step_timings"]; epochs=data["epochs"]
    performance=summary["adapter_performance"]
    parts=dict(candidate_generation_s=float(steps.candidate_generation_time_s.sum()),
        candidate_routing_s=float(steps.candidate_routing_time_s.sum()),
        current_validation_s=performance["timings_s"].get("validator_time_s",0.),
        current_move_work_s=performance["timings_s"].get("current_move_time_s",0.),
        immediate_and_joint_selection_s=performance["timings_s"].get("current_flow_time_s",0.))
    parts["remaining_step_work_s"]=float(steps.step_wall_s.sum())-sum(parts.values())
    # Joint phase timers are nested in immediate_and_joint_selection_s.
    outcomes=data["cohort_outcomes"]
    eligible=outcomes.loc[outcomes.C_compatible & outcomes.passenger_accepts_av]
    future_values=[float(i["expected_future_served"]) for i in joint if i.get("expected_future_served") is not None]
    by_profile={p:dict(completed_orders=int(orders.profile_id.eq(p).sum()),
        fare_cny=int(orders.loc[orders.profile_id.eq(p),"fare_cents"].sum())/100,
        driver_payout_cny=int(orders.loc[orders.profile_id.eq(p),"driver_payout_cents"].sum())/100,
        variable_cost_cny=int(orders.loc[orders.profile_id.eq(p),"variable_cost_cents"].sum())/100,
        order_contribution_cny=int(orders.loc[orders.profile_id.eq(p),"contribution_cents"].sum())/100)
        for p in ("C","HV")}
    result=dict(status="ECONOMIC_NATIVE_INTERFACE_CHECK_COMPLETE",code_sha=summary["code_sha"],
        check_config_sha256=_sha(root/CONFIG),economic_config_sha256=model.fingerprint,
        window="08:15-08:30",cold_start=True,full_day_executed=False,policy_comparison=False,
        revealed_orders=len(outcomes),completed_orders=len(orders),expired_orders=int(outcomes.expired.sum()),
        C_completed_orders=int(orders.profile_id.eq("C").sum()),HV_completed_orders=int(orders.profile_id.eq("HV").sum()),
        financial_equations_and_unique_settlement=True,physical_accounting=summary["physical_accounting"],
        current_floor_checked_decisions=floor_checks,maximum_shared_economic_blocks_per_epoch=int(repairs),
        finances=expected,finances_by_profile=by_profile,C_orders_same_order_contract_advantage_cny=c_advantage/100,
        contract_advantage_is_not_a_counterfactual_policy_effect=True,
        C_candidate_funnel=dict(body_compatible_orders=int(outcomes.C_compatible.sum()),
            body_and_passenger_eligible_orders=len(eligible),
            eligible_without_spatial_AV=int(eligible.spatial_av_opportunities.eq(0).sum()),
            eligible_with_any_final_AV=int(eligible.typed_final_AV_arc_attempts.gt(0).sum()),
            eligible_HV_without_final_AV=int((eligible.vehicle_type.eq("HV")
                &eligible.typed_last_final_AV_count.eq(0)).sum()),
            eligible_HV_with_spatial_but_no_final_AV=int((eligible.vehicle_type.eq("HV")
                &eligible.spatial_av_opportunities.gt(0)&eligible.typed_last_final_AV_count.eq(0)).sum()),
            HV_assigned_with_final_AV_same_epoch=int(outcomes.typed_HV_assigned_with_final_AV_same_epoch.eq(True).sum()),
            counts_are_declared_candidate_graph_not_complete_city_reachability=True),
        joint_decisions=len(joint),joint_current_changed_decisions=sum(bool(i.get("current_action_changed")) for i in joint),
        joint_fallback_decisions=sum(bool(i.get("resource_fallback")) for i in joint),
        same_state_current_contribution_delta_cny=round(float((active.current_operating_contribution_cny
            -active.immediate_operating_contribution_cny).sum()),2),
        current_delta_is_not_a_whole_policy_profit_comparison=True,
        final_plan_future_values=dict(recorded_decisions=len(future_values),
            positive_decisions=sum(v>0 for v in future_values),maximum=max(future_values,default=0)),
        executed_joint_action_change_observed=any(bool(i.get("current_action_changed")) for i in joint),
        cold_start_check_cannot_establish_policy_advantage=True,
        adapter_wall_overrun_decisions=int(active.adapter_wall_deadline_exceeded.sum()),
        active_decision_wall_s={k:float(v) for k,v in active.solve_wall_time_s.quantile([.5,.9,.95,1.]).rename(
            index={.5:"p50",.9:"p90",.95:"p95",1.:"max"}).items()} if len(active) else {},
        total_runner_wall_s=summary["total_runner_wall_s"],setup_time_s=summary["setup_time_s"],
        simulation_loop_wall_s=summary["simulation_loop_wall_s"],step_count=len(steps),
        peak_rss_mib=summary["peak_rss_mib"],nonoverlapping_step_timings_s=parts,
        nested_joint_phase_timings_s=performance.get("joint_timings_s",{}),
        operations=performance["operations"],epoch_records=len(epochs),
        actual_routing=summary["actual_routing"],connector=summary["connector"],
        routing_diagnostics_snapshot_pre_finally_close=True,
        no_full_day_speed_or_C_utilization_improvement_claim=True,
        new_inference_or_training=False,gpu_used=False,dense_matrix=False)
    _atomic_json(root/DOC/"native_check_results.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("describe","run","analyze"),nargs="?",default="describe")
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args(); root=Path.cwd()
    if args.action=="run" and not args.execute: parser.error("one fixed native check requires --execute")
    result=run(root) if args.action=="run" else analyze(root) if args.action=="analyze" else load(root)
    print(json.dumps(result if args.action=="describe" else {k:result[k] for k in (
        "status","total_runner_wall_s","completed_orders","peak_rss_mib") if k in result},ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__": main()
