"""Joint first actions and shared integral recourse, with coalition closure.

This is the algorithm kernel, not a benchmark. A caller supplies legal current
actions and a bounded continuation oracle. No actor, data loading, prediction,
or dense vehicle/order/time matrix exists here. Certificates are restricted
column-pool certificates, never full-city or realized-service guarantees.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
from math import gcd, isfinite, lcm
from time import perf_counter
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .scheme_a_city_master import CurrentAction, ServiceChain
from . import scheme_a_joint_solver as numerical


class JointBudgetStop(TimeoutError):
    """Keep the already validated plan when the shared budget ends."""


class JointSizeStop(JointBudgetStop):
    pass


class JointLabelStop(JointBudgetStop):
    """Column generation ended; already legal columns may still be solved."""


@dataclass(frozen=True)
class JointPlan:
    action_ids: tuple[str, ...]
    chains: tuple[ServiceChain, ...]


@dataclass(frozen=True)
class Coalition:
    vehicles: frozenset[int]
    hints: tuple[ServiceChain, ...] = ()


class JointProblem:
    def __init__(self, actions, oracle, weights, current_ids=(), *, move_cap=50,
                 maximum_columns=20000, maximum_nonzeros=150000):
        self.actions = {a.action_id:a for a in actions}
        if len(self.actions) != len(actions) or not actions:
            raise ValueError("joint actions must be nonempty and uniquely identified")
        self.by_vehicle = defaultdict(list)
        self.current_ids = frozenset(current_ids)
        move_slots=set()
        for a in actions:
            if (not isinstance(a, CurrentAction) or type(a.vehicle_id) is not int
                    or a.kind not in ("SERVE","WAIT","BUSY","RELOCATE","LAYOUT")
                    or type(a.critical) is not bool or type(a.carry_over) is not bool
                    or not isfinite(a.empty_distance_m) or a.empty_distance_m < 0):
                raise ValueError("invalid legal current action")
            if a.kind == "SERVE":
                if type(a.request_id) is not int or a.request_id < 0 or a.request_id not in self.current_ids:
                    raise ValueError("future/unrevealed task cannot be a current SERVE")
            elif a.request_id is not None or a.critical or a.carry_over:
                raise ValueError("nonservice action contains a customer priority")
            if a.kind=="RELOCATE":
                slot=a.payload.get("relocation_slot_s") if isinstance(a.payload,dict) else None
                if type(slot) is not int or slot<0 or slot%900:
                    raise ValueError("current moves retain the original 900-second slot")
                move_slots.add(slot)
            elif a.kind in ("WAIT","BUSY","LAYOUT") and a.empty_distance_m:
                raise ValueError("nontravel current action must have zero incremental distance")
            self.by_vehicle[a.vehicle_id].append(a.action_id)
        if len(move_slots)>1:
            raise ValueError("one joint decision cannot combine different current move slots")
        for ids in self.by_vehicle.values():
            ids.sort()
            if any(self.actions[i].kind == "BUSY" for i in ids) and len(ids) != 1:
                raise ValueError("committed resources cannot have alternative current actions")
        self.weights = {}
        for sid,w in weights.items():
            if not isinstance(sid,str) or not sid or not isfinite(float(w)) or w <= 0:
                raise ValueError("invalid scenario weight")
            exact = w if isinstance(w,Fraction) else Fraction(str(w)).limit_denominator(1000)
            if abs(float(exact)-float(w)) > 1e-12:
                raise ValueError("scenario weight cannot be represented on the declared lattice")
            self.weights[sid] = exact
        if not self.weights or sum(self.weights.values(),Fraction()) != 1:
            raise ValueError("scenario weights must sum exactly to one")
        self.scale = lcm(*(w.denominator for w in self.weights.values()))
        self.units = {s:int(w*self.scale) for s,w in self.weights.items()}
        if self.scale > 100000:
            raise ValueError("scenario lattice exceeds the bounded integer objective")
        if type(move_cap) is not int or not 0 <= move_cap <= 50:
            raise ValueError("the original shared current move cap must be retained")
        self.move_cap, self.oracle = move_cap, oracle
        self.column_cap, self.nonzero_cap = min(20000,maximum_columns), min(150000,maximum_nonzeros)
        self.pool, self.columns, self.entry_count = defaultdict(dict), {}, len(actions)
        self.pool_versions = defaultdict(int)
        self.vehicle_pool_versions = defaultdict(int)
        self.protected_critical = None

    @staticmethod
    def empty(sid, a):
        return ServiceChain(f"{sid}|{a.action_id}|EMPTY",sid,a.vehicle_id,a.action_id,())

    def register(self, chain):
        a = self.actions.get(chain.action_id)
        if (a is None or chain.scenario_id not in self.weights or chain.vehicle_id != a.vehicle_id
                or not isinstance(chain.request_ids,tuple)
                or len(chain.request_ids) > 3 or len(set(chain.request_ids)) != len(chain.request_ids)
                or any(type(r) is not int for r in chain.request_ids)
                or not isfinite(chain.empty_distance_m) or chain.empty_distance_m < 0
                or chain.relocation_slots or a.request_id in chain.request_ids):
            raise ValueError("invalid root-linked, single-use continuation")
        if chain.request_ids and not self.oracle.accepts(chain):
            raise ValueError("continuation was not produced by the legal scene oracle")
        if not chain.request_ids and chain.empty_distance_m:
            raise ValueError("terminal empty continuation must cost zero")
        if chain.chain_id in self.columns:
            if self.columns[chain.chain_id] != chain:
                raise ValueError("chain identity was reused for another path")
            return chain
        if (len(self.actions)+len(self.columns)+1 > self.column_cap
                or self.entry_count+len(chain.request_ids)+1 > self.nonzero_cap):
            raise JointSizeStop("bounded sparse joint column pool exhausted")
        self.columns[chain.chain_id] = chain
        self.pool[chain.scenario_id,chain.action_id][chain.chain_id] = chain
        self.pool_versions[chain.scenario_id,chain.action_id] += 1
        self.vehicle_pool_versions[chain.vehicle_id] += 1
        self.entry_count += len(chain.request_ids)+1
        return chain

    def offer(self, sid, aid, *, required=None, forbidden=frozenset(), check=lambda:None):
        offered = self.oracle.query(sid,aid,required=required,forbidden=forbidden,check=check)
        for c in offered:
            self.register(c)
        return offered

    def evaluate(self, plan):
        chosen = {}
        current, critical, carry, moves, distance = set(),0,0,0,0
        for aid in plan.action_ids:
            a = self.actions.get(aid)
            if a is None or a.vehicle_id in chosen:
                raise ValueError("joint plan duplicates/loses a resource")
            chosen[a.vehicle_id] = aid
            if a.kind == "SERVE":
                if a.request_id in current:
                    raise ValueError("current request was assigned twice")
                current.add(a.request_id)
                critical += a.critical; carry += a.carry_over
            moves += a.kind == "RELOCATE"
            distance += self.scale*int(round(a.empty_distance_m*1000))
        if set(chosen) != set(self.by_vehicle) or moves > self.move_cap:
            raise ValueError("joint resource/current movement conservation failed")
        if self.protected_critical is not None and critical != self.protected_critical:
            raise ValueError("joint plan changed protected urgent service")
        seen, groups, future, owners = {s:set(current) for s in self.weights},set(),0,{}
        for c in plan.chains:
            key = c.scenario_id,c.vehicle_id
            if (key in groups or c.scenario_id not in self.weights
                    or chosen.get(c.vehicle_id) != c.action_id
                    or self.columns.get(c.chain_id) != c):
                raise ValueError("different scenario first actions or unregistered chain")
            groups.add(key)
            for rid in c.request_ids:
                if rid in seen[c.scenario_id]:
                    raise ValueError("current/future/shared request was allocated twice")
                seen[c.scenario_id].add(rid)
                owners[c.scenario_id,rid] = c.vehicle_id
            future += self.units[c.scenario_id]*len(c.request_ids)
            distance += self.units[c.scenario_id]*int(round(c.empty_distance_m*1000))
        if len(groups) != len(self.by_vehicle)*len(self.weights):
            raise ValueError("a selected root lacks its scenario continuation")
        for sid in self.weights:
            owners.update({(sid,rid):v for v,aid in chosen.items()
                           for rid in (self.actions[aid].request_id,) if rid is not None})
        return dict(score=(self.scale*len(current)+future,carry,-distance),
            critical=critical,current_served=len(current),current_moves=moves,
            expected_future_served=float(Fraction(future,self.scale)),
            owners=owners,chosen=chosen,chains={(c.scenario_id,c.vehicle_id):c for c in plan.chains})


def seed_plan(problem, selected, check=lambda:None, *, seed_deadline=None):
    """Integral common-customer incumbent, never a sum of private vehicle values."""
    selected = tuple(selected)
    selected_map = {problem.actions[a].vehicle_id:a for a in selected}
    first = {problem.actions[a].request_id for a in selected if problem.actions[a].kind == "SERVE"}
    chains, limited = [], False
    # Reserve all terminal witnesses before any optional column generation.
    for sid in problem.weights:
        for aid in selected:
            problem.register(problem.empty(sid,problem.actions[aid]))
    for sid in sorted(problem.weights):
        occupied = set(first)
        for vid in sorted(problem.by_vehicle):
            aid = selected_map[vid]
            terminal = problem.register(problem.empty(sid,problem.actions[aid]))
            alternatives = ()
            if not limited:
                try:
                    check()
                    if seed_deadline is not None and perf_counter()>=seed_deadline:
                        raise JointBudgetStop("initial witness budget reserved for coalition search")
                    seeder=getattr(problem.oracle,"seed",None)
                    if seeder is None:
                        alternatives = problem.offer(sid,aid,forbidden=frozenset(occupied),check=check)
                    else:
                        alternatives=seeder(sid,aid,frozenset(occupied),check)
                        for column in alternatives:
                            problem.register(column)
                except JointBudgetStop:
                    limited = True
            chain = min((terminal,*alternatives),key=lambda c:(-len(c.request_ids),
                c.empty_distance_m,c.request_ids,c.chain_id))
            chains.append(chain); occupied.update(chain.request_ids)
    plan = JointPlan(selected,tuple(chains))
    checked = problem.evaluate(plan)
    problem.protected_critical = checked["critical"]
    return plan,limited


def close_coalition(problem, plan, coalition, maximum=6):
    """Close proposed columns over ALL-scenario roots and outside task owners.

    Conflicting root hints are refused, not silently made scenario-specific.
    Outside-owned requested work adds its owner; it never disappears.
    """
    base = problem.evaluate(plan)
    vehicles, roots, displaced = set(coalition.vehicles),{},defaultdict(set)
    for c in coalition.hints:
        if problem.columns.get(c.chain_id) != c:
            raise ValueError("coalition hints must be registered legal columns")
        old = roots.setdefault(c.vehicle_id,c.action_id)
        if old != c.action_id:
            return None  # Two scenario-specific current roots cannot both execute.
        vehicles.add(c.vehicle_id)
        action = problem.actions[c.action_id]
        for sid in problem.weights if base["chosen"][c.vehicle_id] != c.action_id else (c.scenario_id,):
            previous = base["chains"][sid,c.vehicle_id]
            displaced[sid].update(set(previous.request_ids)-set(c.request_ids) if sid==c.scenario_id
                                  else previous.request_ids)
        for rid in (action.request_id,*c.request_ids):
            if rid is None:
                continue
            # Current SERVE consumes this identity in every scenario.
            sids = problem.weights if rid == action.request_id else (c.scenario_id,)
            for sid in sids:
                owner = base["owners"].get((sid,rid))
                if owner is not None:
                    vehicles.add(owner)
    if not vehicles <= problem.by_vehicle.keys() or len(vehicles) > maximum:
        return None
    return Coalition(frozenset(vehicles),coalition.hints),displaced


def solve_block(problem, plan, vehicles, deadline, check=lambda:None, *,
                skip_constant_stages=False,metrics=None,plan_index=None,persistent=False,fixed_objectives=None):
    """Sparse integral block; outside plans/capacity remain in every row."""
    check()
    built_started=perf_counter()
    base = problem.evaluate(plan) if plan_index is None else plan_index.view()
    vehicles = frozenset(vehicles)
    action_ids = [a for v in sorted(vehicles) for a in problem.by_vehicle[v]]
    paths = [c for sid in sorted(problem.weights) for aid in action_ids
             for c in problem.pool[sid,aid].values()]
    n = len(action_ids)+len(paths)
    if n > problem.column_cap:
        raise JointSizeStop("joint neighborhood variable cap exceeded")
    xcol = {a:i for i,a in enumerate(action_ids)}
    zcols = {(s,a):[] for s in problem.weights for a in action_ids}
    ridxs,cidxs,data,lower,upper = [],[],[],[],[]
    def row(entries,lo,hi):
        entries = [(i,float(v)) for i,v in entries if v]
        if len(data)+len(entries) > problem.nonzero_cap:
            raise JointSizeStop("joint neighborhood nonzero cap exceeded")
        j = len(lower)
        for i,v in entries:
            ridxs.append(j); cidxs.append(i); data.append(v)
        lower.append(lo); upper.append(hi)
    if plan_index is None:
        outside = [problem.actions[a] for v,a in base["chosen"].items() if v not in vehicles]
        outside_current = {a.request_id for a in outside if a.kind == "SERVE"}
        out_moves = sum(a.kind=="RELOCATE" for a in outside)
        out_critical = sum(a.critical for a in outside)
    else:
        outside=None
        outside_current={a.request_id for a in (problem.actions[aid] for aid in action_ids)
            if a.request_id is not None and any(base["owners"].get((s,a.request_id)) not in (None,*vehicles)
                for s in problem.weights)}
        out_moves=plan_index.moves-sum(plan_index.contributions[v][4] for v in vehicles)
        out_critical=plan_index.critical-sum(plan_index.contributions[v][3] for v in vehicles)
    current_rows, future_rows = defaultdict(list),defaultdict(list)
    objectives = [np.zeros(n) for _ in range(3)]
    for aid,i in xcol.items():
        a = problem.actions[aid]
        objectives[0][i] = problem.scale*(a.kind=="SERVE")
        objectives[1][i] = a.carry_over
        objectives[2][i] = -problem.scale*int(round(a.empty_distance_m*1000))
        if a.kind=="SERVE":
            current_rows[a.request_id].append((i,1))
            for sid in problem.weights:
                future_rows[sid,a.request_id].append((i,1))
    for j,c in enumerate(paths,len(action_ids)):
        zcols[c.scenario_id,c.action_id].append((j,1))
        objectives[0][j] = problem.units[c.scenario_id]*len(c.request_ids)
        objectives[2][j] = -problem.units[c.scenario_id]*int(round(c.empty_distance_m*1000))
        for rid in c.request_ids:
            future_rows[c.scenario_id,rid].append((j,1))
    for v in sorted(vehicles):
        row([(xcol[a],1) for a in problem.by_vehicle[v]],1,1)
    for (sid,aid),entries in zcols.items():
        row([*entries,(xcol[aid],-1)],0,0)
    for rid,entries in current_rows.items():
        row(entries,0,0 if rid in outside_current else 1)
    for (sid,rid),entries in future_rows.items():
        used = base["owners"].get((sid,rid))
        row(entries,0,0 if used is not None and used not in vehicles else 1)
    row([(xcol[a],1) for a in action_ids if problem.actions[a].kind=="RELOCATE"],
        0,problem.move_cap-out_moves)
    row([(xcol[a],1) for a in action_ids if problem.actions[a].critical],
        problem.protected_critical-out_critical,problem.protected_critical-out_critical)
    best,proved,stages = (plan if plan_index is None else None),True,0
    best_score=base["score"]
    receipt=numerical._solver_options_receipt()
    session=None; matrix=None; matrix_rows=matrix_entries=0
    if persistent:
        from .persistent_city_milp import BundledIntegerSession
        session=BundledIntegerSession(dict(receipt,options=dict(receipt["options"],parallel="off")))
        class Rows:
            pass
        session_rows=Rows(); session_rows.columns=n
        session_rows.lower=lower; session_rows.upper=upper
    maximum_lattice_error=0.
    build_wall=perf_counter()-built_started
    if metrics is not None:
        metrics["model_build_time_s"]+=build_wall
    solver_wall=0.; solver_calls=0; constant_stages=0
    recorded_loads=recorded_warms=recorded_runs=0
    def remaining():
        check(); value=deadline-perf_counter()
        if value<=0: raise JointBudgetStop("persistent block deadline exhausted")
        return value
    for objective_index,objective in enumerate(objectives):
        check()
        seconds = deadline-perf_counter()
        if seconds <= 0:
            raise JointBudgetStop("shared joint solve deadline exhausted")
        if fixed_objectives is not None and objective_index in fixed_objectives:
            optimum=fixed_objectives[objective_index]
            row([(i,int(v)) for i,v in enumerate(objective) if v],optimum,optimum)
            stages+=1
            if metrics is not None: metrics["capacity_bound_stages_skipped"]+=1
            continue
        if skip_constant_stages and not np.any(objective):
            # All feasible integer witnesses have exactly the same zero value.
            stages+=1; constant_stages+=1
            if metrics is not None:
                metrics["constant_objective_stages_skipped"]+=1
            continue
        if np.sum(np.abs(objective)) >= 2**50:
            raise ValueError("joint integer objective exceeds safe numerical range")
        stamp=perf_counter()
        if not persistent or matrix is None:
            matrix = coo_matrix((data,(ridxs,cidxs)),shape=(len(lower),n)).tocsc()
        elif len(lower)>matrix_rows:
            from scipy.sparse import vstack
            extra=coo_matrix((data[matrix_entries:],
                ([j-matrix_rows for j in ridxs[matrix_entries:]],cidxs[matrix_entries:])),
                shape=(len(lower)-matrix_rows,n)).tocsc()
            matrix=vstack((matrix,extra),format="csc")
        matrix_rows=len(lower); matrix_entries=len(data)
        divisor=gcd(*(abs(int(v)) for v in objective if v)) or 1
        primitive=objective/divisor
        total=float(np.sum(np.abs(primitive)))
        lattice_error=(numerical.NUMERICAL_TOLERANCE*total
            +numerical.NUMERICAL_TOLERANCE*len(lower)*float(np.max(np.abs(primitive),initial=0))
            +64*np.finfo(float).eps*max(1.,total))
        maximum_lattice_error=max(maximum_lattice_error,lattice_error)
        elapsed=perf_counter()-stamp; build_wall+=elapsed
        if metrics is not None:
            metrics["model_build_time_s"]+=elapsed
        stamp=perf_counter()
        if metrics is not None:
            metrics["solver_calls"]+=1
        try:
            caught=[]
            if session is not None:
                result=session.solve(-primitive,session_rows,matrix,remaining)
            else:
                with warnings.catch_warnings(record=True) as caught:
                    result = milp(-primitive,integrality=np.ones(n,np.int8),
                        bounds=Bounds(np.zeros(n),np.ones(n)),
                        constraints=LinearConstraint(matrix,np.asarray(lower),np.asarray(upper)),
                        options=dict(receipt["options"],time_limit=seconds,mip_rel_gap=0.,presolve=True))
        finally:
            elapsed=perf_counter()-stamp; solver_wall+=elapsed
            if metrics is not None:
                metrics["solver_wall_time_s"]+=elapsed
                if session is not None:
                    metrics["persistent_model_loads"]+=session.model_loads-recorded_loads
                    metrics["persistent_warm_starts"]+=session.warm_starts-recorded_warms
                    metrics["persistent_solver_runs"]+=session.runs-recorded_runs
                    recorded_loads,recorded_warms,recorded_runs=session.model_loads,session.warm_starts,session.runs
        solver_calls+=1
        if any("Unrecognized options" not in str(w.message) for w in caught):
            raise RuntimeError("unexpected warning in sparse joint solver")
        stages += 1
        if result.x is None:
            if result.status == 1:
                proved = False; break
            raise RuntimeError(f"feasible joint block rejected by solver: {result.message}")
        raw = np.asarray(result.x)
        if (not np.all(np.isfinite(raw))
                or np.max(np.abs(raw-np.rint(raw)),initial=0) > numerical.NUMERICAL_TOLERANCE):
            raise RuntimeError("joint solver returned a nonintegral witness")
        binary = np.rint(raw)
        lhs = matrix@binary
        if (np.any((binary!=0)&(binary!=1)) or np.any(lhs<lower) or np.any(lhs>upper)):
            raise RuntimeError("joint integer witness violates an exact sparse row")
        selected = tuple(aid for aid,i in xcol.items() if binary[i])
        continued = tuple(c for j,c in enumerate(paths,len(action_ids)) if binary[j])
        if plan_index is None:
            candidate = JointPlan(tuple(sorted((*selected,*(a.action_id for a in outside)))),
                tuple(c for c in plan.chains if c.vehicle_id not in vehicles)+continued)
            candidate_score=problem.evaluate(candidate)["score"]
        else:
            candidate=plan_index.propose(vehicles,selected,continued)
            candidate_score=candidate.score
        if candidate_score > best_score:
            best,best_score = candidate,candidate_score
        if session is not None:
            session.accept_certified_binary(binary)
        if (result.status != 0 or getattr(result,"mip_gap",0.) not in (0.,None)
                or lattice_error>=.25):
            proved = False; break
        optimum = int(round(float(objective@binary)))
        row([(i,int(v)) for i,v in enumerate(objective) if v],optimum,optimum)
    return best,dict(variables=n,nonzeros=len(data),stages=stages,
        solver_calls=solver_calls,solver_wall_time_s=solver_wall,model_build_time_s=build_wall,
        constant_objective_stages_skipped=constant_stages,
        future_chain_variables=len(paths),current_action_variables=len(action_ids),
        integer_objective_lattice_error_bound=maximum_lattice_error,
        restricted_pool_optimality_proved=proved and stages==3,
        full_domain_optimality_proved=False,
        persistent_session=session.diagnostics() if session is not None else None)


def populate_block(problem,plan,vehicles,targets,check):
    base=problem.evaluate(plan)
    outside={s:frozenset(r for (scene,r),owner in base["owners"].items()
        if scene==s and owner not in vehicles) for s in problem.weights}
    for v in sorted(vehicles):
        for aid in problem.by_vehicle[v]:
            for s in sorted(problem.weights):
                problem.register(problem.empty(s,problem.actions[aid]))
                problem.offer(s,aid,forbidden=outside[s],check=check)
                for wanted in sorted(targets.get(s,())):
                    problem.offer(s,aid,required=wanted,forbidden=outside[s],check=check)


def spatial_improve(problem,initial,*,deadline,check,maximum_coalition=6,maximum_partial_states=128):
    """Same model/oracle/budget, ordinary spatial groups instead of displacement closure."""
    if not 1<=maximum_coalition<=6 or not 1<=maximum_partial_states<=512:
        raise ValueError("coalition/search memory cap exceeds the opt-in design")
    best,trace=initial,[]
    accepted=0; stopped=None
    try:
        for sid,rid in problem.oracle.seeds(problem.evaluate(best)["owners"],check):
            check()
            owners=problem.evaluate(best)["owners"]
            nearby=problem.oracle.nearby_resources(sid,rid,maximum_coalition,check)
            group=set(nearby)
            owner=owners.get((sid,rid))
            if owner is not None:
                if len(group)>=maximum_coalition and owner not in group:
                    group.remove(nearby[-1])
                group.add(owner)
            if not group:
                continue
            targets=defaultdict(set)
            targets[sid].add(rid)
            for c in best.chains:
                if c.vehicle_id in group:
                    targets[c.scenario_id].update(c.request_ids)
            populate_block(problem,best,group,targets,check)
            candidate,info=solve_block(problem,best,group,deadline,check)
            old,new=problem.evaluate(best)["score"],problem.evaluate(candidate)["score"]
            trace.append(dict(coalition_size=len(group),before=list(old),after=list(new),
                              improved=new>old,**info))
            check()
            if new>old:
                best=candidate; accepted+=1
            if len(trace)>=maximum_partial_states:
                stopped="PARTIAL_STATE_CAP"; break
    except JointBudgetStop as error:
        stopped=type(error).__name__
    result=problem.evaluate(best)
    return best,dict(status="JOINT_FEASIBLE_IMPROVED" if accepted else "JOINT_FEASIBLE_INCUMBENT",
        neighborhood_strategy="SPATIAL_JOINT",score=list(result["score"]),
        expected_future_served=result["expected_future_served"],
        accepted_coalitions=accepted,partial_states=len(trace),neighborhoods=trace,
        stop_reason=stopped,pool_columns=len(problem.columns),
        same_first_action_across_scenarios=True,shared_customer_capacity=True,
        whole_day_optimality_claimed=False,full_domain_optimality_proved=False,
        dense_matrix=False,gpu_used=False,wall_overrun=perf_counter()>deadline)


def improve(problem, initial, *, deadline, check=lambda:None, maximum_coalition=6,
            maximum_partial_states=128):
    """Bottleneck-guided multi-resource search; neutral branches stay alive."""
    if not 1 <= maximum_coalition <= 6 or not 1 <= maximum_partial_states <= 512:
        raise ValueError("coalition/search memory cap exceeds the opt-in design")
    best,records,partial,accepted = initial,[],0,0
    problem.evaluate(best)
    stopped = None
    try:
        seeds = problem.oracle.seeds(problem.evaluate(best)["owners"],check)
        for sid,rid in seeds:
            check()
            base = problem.evaluate(best)
            queue,visited = deque(),set()
            for aid in problem.oracle.supports(sid,rid,check):
                for c in problem.offer(sid,aid,required=rid,check=check):
                    queue.append(Coalition(frozenset((c.vehicle_id,)),(c,)))
                    if len(queue) >= maximum_partial_states:
                        break
                if len(queue) >= maximum_partial_states:
                    break
            while queue and partial < maximum_partial_states:
                check()
                state = queue.popleft()
                closed = close_coalition(problem,best,state,maximum_coalition)
                if closed is None:
                    continue
                state,displaced = closed
                key = (state.vehicles,tuple(sorted(c.chain_id for c in state.hints)))
                if key in visited:
                    continue
                visited.add(key); partial += 1
                # Every changed/current root is offered in ALL scenarios.
                target = defaultdict(set)
                for s,orders in displaced.items():
                    target[s].update(orders)
                for c in state.hints:
                    target[c.scenario_id].update(c.request_ids)
                populate_block(problem,best,state.vehicles,target,check)
                candidate,diagnostic = solve_block(problem,best,state.vehicles,deadline,check)
                old_score,new_score = problem.evaluate(best)["score"],problem.evaluate(candidate)["score"]
                records.append(dict(coalition_size=len(state.vehicles),before=list(old_score),
                    after=list(new_score),improved=new_score>old_score,**diagnostic))
                check()
                if new_score>old_score:
                    best=candidate; accepted+=1
                    # Owner maps change. Old queued hints are closed again, never trusted.
                if len(state.vehicles) < maximum_coalition:
                    for s,orders in sorted(displaced.items()):
                        for wanted in sorted(orders):
                            for aid in problem.oracle.supports(s,wanted,check):
                                v=problem.actions[aid].vehicle_id
                                if v in state.vehicles:
                                    continue
                                for c in problem.offer(s,aid,required=wanted,check=check):
                                    if len(queue) >= maximum_partial_states:
                                        break
                                    queue.append(Coalition(state.vehicles|{v},(*state.hints,c)))
                # NO rejection based on singleton/partial gain.
            if partial >= maximum_partial_states:
                stopped="PARTIAL_STATE_CAP"; break
    except JointBudgetStop as error:
        stopped=type(error).__name__
    checked=problem.evaluate(best)
    return best,dict(status="JOINT_FEASIBLE_IMPROVED" if accepted else "JOINT_FEASIBLE_INCUMBENT",
        neighborhood_strategy="BOTTLENECK_CLOSURE",
        score=list(checked["score"]),expected_future_served=checked["expected_future_served"],
        accepted_coalitions=accepted,partial_states=partial,neighborhoods=records,
        stop_reason=stopped,pool_columns=len(problem.columns),
        same_first_action_across_scenarios=True,shared_customer_capacity=True,
        whole_day_optimality_claimed=False,full_domain_optimality_proved=False,
        dense_matrix=False,gpu_used=False,wall_overrun=perf_counter()>deadline)
