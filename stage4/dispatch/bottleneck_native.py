"""Opt-in native bridge to joint fleet control; no simulation launcher."""
from __future__ import annotations

from time import perf_counter

import psutil

from .bottleneck_joint import JointBudgetStop, JointProblem, seed_plan, improve, spatial_improve
from .bottleneck_scene import PredictionSceneOracle
from .flexibility_native import predicted_vehicle_states
from .scheme_a_city_graph import RouteState
from .scheme_a_city_master import CurrentAction


def solve_joint_current(c, actions, after, baseline, now, cfg, provider,
                        last_assignments, remaining_model, task_factory,
                        moving_wait, waiting_ids, deadline):
    """Return the old physical action schema; future chains never execute."""
    if not baseline["primary_optimality_proved"]:
        return dict(baseline,joint_control=dict(status="BASELINE_URGENT_PROTECTION_NOT_PROVED",
                                              resource_fallback=True))
    initial_ids=list(baseline["selected_action_ids"])
    current_actions=list(actions); roots=dict(after)
    original_ids=set(roots); vids={a.vehicle_id for a in actions}
    started=perf_counter(); peak=0.
    def check():
        nonlocal peak
        if perf_counter()>=deadline:
            raise JointBudgetStop("one shared native joint decision deadline exhausted")
        peak=max(peak,psutil.Process().memory_info().rss/2**20)
        if peak>min(1536.,float(cfg.get("maximum_rss_mib",1536))):
            raise JointBudgetStop("joint scratch exceeds the original process RAM cap")
    try:
        check()
        # Fixed current prefixes, but their prediction-only completion states
        # can take part in shared future competition. No realized finish read.
        predicted=predicted_vehicle_states(c,now,1800,last_assignments,remaining_model)
        for vehicle in predicted:
            vid=int(vehicle.vehicle_id)
            if vid in vids:
                continue
            state=(moving_wait(c,vid,now,vehicle.profile_id)
                if vid in c.repositioning_manager.active else RouteState(
                    tuple(map(float,vehicle.ready_position.split(","))),
                    float(vehicle.ready_time_s),float(vehicle.availability_end_s),vehicle.profile_id))
            aid=f"V{vid}:BUSY"
            current_actions.append(CurrentAction(aid,vid,"BUSY",None))
            roots[aid]=state; initial_ids.append(aid); vids.add(vid)
        pending=tuple(task_factory(c,int(rid)) for rid in sorted(waiting_ids)
            if c.request_by_rid[int(rid)].sim_time_s<=now<c.request_meta[int(rid)]["pickup_deadline_s"])
        scenes=provider.library.view(now)
        oracle=PredictionSceneOracle(current_actions,roots,scenes,pending,provider.pace,now,
            check=check,maximum_labels=int(cfg.get("joint_maximum_labels",50000)),
            maximum_columns=int(cfg.get("maximum_model_variables",20000)))
        problem=JointProblem(current_actions,oracle,provider.library.weights,
            [t.request_id for t in pending],move_cap=50,
            maximum_columns=int(cfg.get("maximum_model_variables",20000)),
            maximum_nonzeros=int(cfg.get("maximum_model_nonzeros",150000)))
        initial,seed_limited=seed_plan(problem,initial_ids,check,
                                     seed_deadline=perf_counter()+max(0.,deadline-perf_counter())*.25)
        search=spatial_improve if cfg.get("joint_neighborhood_strategy")=="SPATIAL_JOINT" else improve
        plan,info=search(problem,initial,deadline=deadline,check=check,
            maximum_coalition=int(cfg.get("joint_maximum_coalition",6)),
            maximum_partial_states=int(cfg.get("joint_maximum_partial_states",128)))
        checked=problem.evaluate(plan)
        selected=tuple(sorted(set(plan.action_ids)&original_ids))
        chosen=[problem.actions[a] for a in selected]
        info.update(seed_limited=seed_limited,initial_score=list(problem.evaluate(initial)["score"]),
            planning_labels=oracle.labels,planning_resources=len(problem.by_vehicle),
            foreground_resources=len({a.vehicle_id for a in actions}),joint_wall_time_s=perf_counter()-started,
            peak_process_rss_mib=peak,current_actions_validated_by_original_interface=True,
            future_directed_road_certificate=False,realized_targets_used=False,
            expected_future_served_scope="INTEGRAL_HISTORY_PLAN_NOT_EXACT_Q_OR_REALIZED_SERVICE",
            future_connector_queries=0,scenario_clock="EXISTING_HISTORY_GRID_PLUS_REAL_PENDING_30S",
            current_strategy="PROTECT_URGENT_THEN_JOINT_TOTAL_SERVICE_CARRY_EMPTY")
        info["resource_fallback"]=False
        # Never accumulate per-neighborhood raw traces over an entire day.
        trace=info.pop("neighborhoods")
        info.update(neighborhoods_evaluated=len(trace),
            max_joint_variables=max((r["variables"] for r in trace),default=0),
            max_joint_future_variables=max((r["future_chain_variables"] for r in trace),default=0),
            generated_continuation_columns=info["pool_columns"],
            max_joint_nonzeros=max((r["nonzeros"] for r in trace),default=0),
            restricted_pool_proved_count=sum(r["restricted_pool_optimality_proved"] for r in trace))
        return dict(baseline,selected_action_ids=selected,critical_now=checked["critical"],
            current_served=checked["current_served"],
            carry_over_current_served=sum(a.carry_over for a in chosen),
            current_moves=checked["current_moves"],opt_status=info["status"],
            backend="JOINT_"+info["neighborhood_strategy"]+"_SPARSE_MILP",primary_optimality_proved=False,
            secondary_status="JOINT_PLANNING_NOT_OLD_CURRENT_PRIMARY_FACE",
            secondary_optimality_proved=False,secondary_time_s=0.,
            future_chain_variables=info["max_joint_future_variables"],joint_control=info,
            fallback_reason=info["stop_reason"],dense_matrix=False,
            current_empty_distance_m=sum(a.empty_distance_m for a in chosen),
            whole_day_optimality_claimed=False)
    except JointBudgetStop as error:
        return dict(baseline,fallback_reason=str(error),
            joint_control=dict(status="JOINT_BUDGET_KEEP_VALIDATED_BASELINE",resource_fallback=True,
            reason=str(error),joint_wall_time_s=perf_counter()-started,
            peak_process_rss_mib=peak,whole_day_optimality_claimed=False))
