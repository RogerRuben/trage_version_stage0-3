"""Birth-position bridge to the SAME joint planner; no dataset/actor loading."""
from __future__ import annotations

from time import perf_counter

from .bottleneck_joint import JointBudgetStop, JointProblem, seed_plan, improve, spatial_improve
from .bottleneck_scene import PredictionSceneOracle
from .scheme_a_city_graph import RouteState
from .scheme_a_city_master import CurrentAction


def solve_birth_batch(slots,fresh_ids,sites,seed_positions,library,pace,now,*,
                      strategy="BOTTLENECK_CLOSURE",wall_limit_s=30.,check=lambda:None):
    if strategy not in ("BOTTLENECK_CLOSURE","SPATIAL_JOINT") or not 0<wall_limit_s<=30:
        raise ValueError("unknown birth planner or extended offline solve budget")
    started=perf_counter(); deadline=started+wall_limit_s
    def budget():
        check()
        if perf_counter()>=deadline:
            raise JointBudgetStop("joint birth batch uses one shared 30-second budget")
    fresh_ids=frozenset(fresh_ids)
    sites={str(site["site_id"]):dict(site) for site in sites}
    if not sites or len(fresh_ids)!=sum(row["native_id"] in fresh_ids for row in slots):
        raise ValueError("birth candidate/supply identities differ")
    actions,roots,initial=[],{},[]
    by_native={row["native_id"]:row for row in slots}
    for row in slots:
        vid=row["native_id"]
        if vid not in fresh_ids and not (row["admission_end_s"]>now and row["activation_s"]<now+1800):
            continue
        ready=max(float(now),float(row["activation_s"]))
        if vid in fresh_ids:
            if row["profile_id"]!="C" or not now<=row["activation_s"]<now+1800:
                raise ValueError("only this bin's original C births may change position")
            seed=seed_positions[str(row["vehicle_id"])]
            if str(seed["site_id"]) not in sites:
                raise ValueError("frozen seed position is outside the common site universe")
            for site_id,site in sorted(sites.items()):
                aid=f"V{vid}:LAYOUT:{site_id}"
                actions.append(CurrentAction(aid,vid,"LAYOUT",None,payload={"site":site}))
                roots[aid]=RouteState(tuple(site["position"]),ready,row["admission_end_s"],"C")
                if site_id==str(seed["site_id"]):
                    initial.append(aid)
        else:
            point=seed_positions.get(str(row["vehicle_id"]),{}).get("position",row["position"])
            aid=f"V{vid}:FIXED"
            actions.append(CurrentAction(aid,vid,"WAIT",None))
            roots[aid]=RouteState(tuple(point),ready,row["admission_end_s"],row["profile_id"])
            initial.append(aid)
    result={str(by_native[v]["vehicle_id"]):dict(seed_positions[str(by_native[v]["vehicle_id"])])
            for v in fresh_ids}
    try:
        budget()
        scenes=library.view(now)
        oracle=PredictionSceneOracle(actions,roots,scenes,(),pace,now,check=budget)
        problem=JointProblem(actions,oracle,library.weights)
        plan,seed_limited=seed_plan(problem,initial,budget,
                                   seed_deadline=perf_counter()+max(0.,deadline-perf_counter())*.25)
        search=improve if strategy=="BOTTLENECK_CLOSURE" else spatial_improve
        best,info=search(problem,plan,deadline=deadline,check=budget)
        for aid in best.action_ids:
            action=problem.actions[aid]
            if action.vehicle_id in fresh_ids:
                result[str(by_native[action.vehicle_id]["vehicle_id"])]=dict(action.payload["site"])
        trace=info.pop("neighborhoods")
        info.update(initial_score=list(problem.evaluate(plan)["score"]),seed_limited=seed_limited,
            neighborhoods_evaluated=len(trace),max_joint_variables=max((x["variables"] for x in trace),default=0),
            max_joint_nonzeros=max((x["nonzeros"] for x in trace),default=0),
            changed_births=sum(result[key]["site_id"]!=seed_positions[key]["site_id"] for key in result),
            planned_fixed_background_not_observed_midday_supply=True,
            target_day_requests_loaded=False,scenario_shared_birth_position=True)
    except JointBudgetStop as error:
        info=dict(status="BIRTH_BUDGET_KEEP_FROZEN_SEED",reason=str(error),
                  neighborhood_strategy=strategy,changed_births=0)
    info["runtime_s"]=perf_counter()-started
    info["wall_overrun"]=perf_counter()>deadline
    return result,info
