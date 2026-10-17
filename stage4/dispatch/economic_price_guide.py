"""A small restricted-pool LP guides pricing; it is NOT a citywide bound.

The incumbent, common first actions, and per-scenario customer capacity remain
in the model. Dual prices only order heuristic column exploration. Feasibility
and acceptance always use the integral plan and its unmodified monetary sum.
"""
from collections import defaultdict
from time import perf_counter
import warnings

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .bottleneck_joint import JointBudgetStop


def restricted_prices(p, plan, deadline, check):
    if p.economics is None:
        return {}, {"status":"NOT_ECONOMIC"}
    started=perf_counter(); check()
    actions=list(p.actions.values()); chains=list(p.columns.values())
    n=len(actions)+len(chains)
    if n>p.column_cap:
        return {}, {"status":"PRICE_GUIDE_SIZE_SKIPPED"}
    x={a.action_id:i for i,a in enumerate(actions)}
    eq=[]; eq_rhs=[]; ub=[]; ub_rhs=[]; labels=[]
    groups=defaultdict(list); requests=defaultdict(list)
    for j,c in enumerate(chains,len(actions)):
        groups[c.scenario_id,c.action_id].append((j,1))
        for rid in c.request_ids: requests[c.scenario_id,rid].append((j,1))
    for v,ids in p.by_vehicle.items():
        eq.append([(x[a],1) for a in ids]); eq_rhs.append(1)
    for sid in p.weights:
        for a in actions:
            eq.append([*groups[sid,a.action_id],(x[a.action_id],-1)]); eq_rhs.append(0)
            if a.request_id is not None: requests[sid,a.request_id].append((x[a.action_id],1))
    for key,entries in requests.items():
        ub.append(entries); ub_rhs.append(1); labels.append(key)
    ub.append([(x[a.action_id],1) for a in actions if a.kind=="RELOCATE"])
    ub_rhs.append(p.move_cap); labels.append(None)
    for field,goal in (("critical",p.protected_critical),
                       ("current",p.protected_current),("carry",p.protected_carry)):
        if goal is None: continue
        entries=[(x[a.action_id],1) for a in actions if (
            a.critical if field=="critical" else a.kind=="SERVE" if field=="current" else a.carry_over)]
        if field=="critical": eq.append(entries); eq_rhs.append(goal)
        else: ub.append([(j,-v) for j,v in entries]); ub_rhs.append(-goal); labels.append(None)
    total=sum(map(len,eq))+sum(map(len,ub))
    if total>p.nonzero_cap:
        return {}, {"status":"PRICE_GUIDE_SPARSE_CAP_SKIPPED"}
    def matrix(rows):
        rr=[]; cc=[]; vv=[]
        for i,entries in enumerate(rows):
            for j,v in entries: rr.append(i); cc.append(j); vv.append(v)
        return coo_matrix((vv,(rr,cc)),shape=(len(rows),n)).tocsc()
    objective=np.asarray([p.action_score(a)[0] for a in actions]
        +[p.chain_score(c)[0] for c in chains],dtype=float)
    seconds=min(.2,deadline-perf_counter())
    if seconds<=0: raise JointBudgetStop("shared deadline before optional monetary price guide")
    check()
    try:
        with warnings.catch_warnings(record=True):
            result=linprog(-objective,A_ub=matrix(ub),b_ub=ub_rhs,A_eq=matrix(eq),b_eq=eq_rhs,
                bounds=(0,1),method="highs-ds",options={"time_limit":seconds,"presolve":True,"threads":1})
    except RuntimeError as error:
        return {},dict(status="PRICE_GUIDE_SOLVER_UNAVAILABLE_USE_TRUE_CONTRIBUTION",
            error_type=type(error).__name__,runtime_s=perf_counter()-started,global_bound_claimed=False)
    prices={}
    if result.success and result.status==0:
        for key,value in zip(labels,result.ineqlin.marginals):
            if key is not None and np.isfinite(value):
                prices[key]=max(0.,-float(value)/p.units[key[0]])
    return prices,dict(status="RESTRICTED_POOL_GUIDANCE" if prices else "PRICE_GUIDE_UNAVAILABLE_USE_TRUE_CONTRIBUTION",
        variables=n,nonzeros=total,runtime_s=perf_counter()-started,
        monetary_unit="CENTS_PER_UNWEIGHTED_SCENARIO_TASK",global_bound_claimed=False,
        whole_city_optimality_claimed=False,wall_overrun=perf_counter()>deadline)
