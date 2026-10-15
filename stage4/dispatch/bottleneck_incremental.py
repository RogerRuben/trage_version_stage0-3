"""Opt-in incremental joint execution. No replay, routing or data loading.

Retained identities are repaired against NEW decision-time states. No previous
travel time, physical current action or optimality certificate crosses epochs.
"""
from collections import Counter,OrderedDict,defaultdict,deque
from contextlib import contextmanager
from math import ceil
from time import perf_counter

from .bottleneck_joint import (Coalition,JointBudgetStop,JointLabelStop,JointPlan,
    JointProblem,close_coalition,solve_block)
from .bottleneck_topology import ForecastTopology


def domains(problem):
    oracle=problem.oracle
    return {v:tuple((aid,problem.actions[aid].critical,problem.actions[aid].carry_over,
        problem.actions[aid].empty_distance_m,oracle.roots[aid].position,
        ceil(oracle.roots[aid].ready_s/300),oracle.roots[aid].admission_end_s,
        oracle.roots[aid].profile_id) for aid in ids) for v,ids in problem.by_vehicle.items()}


class RollingJointMemory:
    """One selected plan and one forecast snapshot, not all previous column pools."""
    def __init__(self,efficient=False):
        self.efficient=efficient
        self.topology=ForecastTopology()
        self.paths={}; self.domains={}; self.task_keys=frozenset(); self.owners={}
        self.deferred=(); self.touched=set(); self.last_now=None

    def initial(self,problem,selected,check,seed_deadline):
        oracle=problem.oracle
        chosen={problem.actions[aid].vehicle_id:aid for aid in selected}
        first={problem.actions[aid].request_id for aid in selected
               if problem.actions[aid].kind=="SERVE"}
        chains={}; fill=[]; dirty=set(); counts=Counter(); limited=False
        # Reserve a complete feasible terminal witness before optional work.
        for sid in problem.weights:
            occupied=set(first)
            for v,aid in sorted(chosen.items()):
                chains[sid,v]=problem.register(problem.empty(sid,problem.actions[aid]))
            for v,aid in sorted(chosen.items()):
                check()
                old=self.paths.get((sid,v),())
                c=oracle.repair(sid,aid,old,frozenset(occupied),check) if old else None
                if c is not None:
                    c=problem.register(c); chains[sid,v]=c; occupied.update(c.request_ids)
                    counts["warm_restored_continuations"]+=1
                if old and (c is None or c.request_ids!=old):
                    dirty.add(v); counts["warm_repaired_continuations"]+=1
                if v not in self.domains or (old and (c is None or len(c.request_ids)<len(old))):
                    fill.append((sid,v))
            # All retained owners have priority over newly greedy assignments.
        occupied={sid:set(first) for sid in problem.weights}
        for (sid,v),c in chains.items():
            occupied[sid].update(c.request_ids)
        for sid,v in fill:
            if perf_counter()>=seed_deadline:
                limited=True; break
            check(); aid=chosen[v]; old=chains[sid,v]
            forbidden=frozenset(occupied[sid]-set(old.request_ids))
            candidates=oracle.seed(sid,aid,forbidden,check)
            counts["warm_greedy_seed_queries"]+=1
            for c in candidates:
                problem.register(c)
            best=min((old,*candidates),key=lambda c:(-len(c.request_ids),c.empty_distance_m,c.request_ids))
            chains[sid,v]=best
            occupied[sid].difference_update(old.request_ids); occupied[sid].update(best.request_ids)
        plan=JointPlan(tuple(selected),tuple(chains.values()))
        checked=problem.evaluate(plan); problem.protected_critical=checked["critical"]
        return plan,dict(counts,seed_limited=limited),dirty

    def frontier(self,problem,plan,dirty,check):
        oracle=problem.oracle; base=problem.evaluate(plan); current=domains(problem)
        live=frozenset((sid,rid) for sid,tasks in oracle.tasks.items() for rid in tasks)
        candidates=set(live-self.task_keys)
        if not self.efficient: candidates|=set(self.deferred)&set(live)
        candidates.update(k for k in self.owners if k in live and k not in base["owners"])
        changed={v for v,d in current.items() if self.domains.get(v)!=d}|dirty
        if not self.efficient: changed|=self.touched
        for sid,tasks in oracle.tasks.items():
            candidates.update((sid,r) for r,t in tasks.items() if t.source_kind=="ACTUAL_PENDING")
            for v in sorted(changed & problem.by_vehicle.keys()):
                for aid in problem.by_vehicle[v]:
                    check()
                    candidates.update((sid,t.request_id) for t in oracle._near(sid,oracle.roots[aid].position))
        if not any(len(ids)>1 for ids in problem.by_vehicle.values()):
            return tuple(sorted(candidates,key=lambda k:(
                oracle.tasks[k[0]][k[1]].source_kind!="ACTUAL_PENDING",
                oracle.tasks[k[0]][k[1]].deadline_s,k)))
        relevant=list(oracle.seeds(base["owners"],check,candidates=candidates))
        valid=set(relevant)
        pending=[k for k in relevant if oracle.tasks[k[0]][k[1]].source_kind=="ACTUAL_PENDING"]
        tail=[] if self.efficient else [k for k in self.deferred if k in valid and k not in pending]
        seen=set(pending)|set(tail)
        tail.extend(k for k in relevant if k not in seen)
        return tuple((*pending,*tail))

    def save(self,problem,initial,plan,deferred,now):
        checked=problem.evaluate(plan)
        before={(c.scenario_id,c.vehicle_id):c.request_ids for c in initial.chains}
        after={(c.scenario_id,c.vehicle_id):c.request_ids for c in plan.chains}
        self.touched=set() if self.efficient else {v for (s,v),path in after.items() if before.get((s,v))!=path}
        self.paths=after; self.domains=domains(problem)
        self.task_keys=frozenset((s,r) for s,tasks in problem.oracle.tasks.items() for r in tasks)
        self.owners={k:v for k,v in checked["owners"].items() if k in self.task_keys}
        self.deferred=tuple(dict.fromkeys(k for k in deferred if k in self.task_keys))
        self.last_now=float(now)


class BlockWorkspace:
    """Exact per-epoch query reuse plus bounded duplicate-neighborhood suppression.

    A repeated unproved witness is only a visited heuristic state, NOT an
    optimality certificate. New columns or changed outside capacity invalidate it.
    """
    def __init__(self):
        self.offers=OrderedDict(); self.blocks=OrderedDict(); self.counts=Counter()
        self.metrics=Counter(); self.block_wall_s=0.
        self.columns_closed=False
        self._problem=None; self._plan=None; self._generation=0

    def offer(self,p,sid,aid,required,forbidden,check):
        if self.columns_closed:
            return ()
        key=sid,aid,required,forbidden
        if key in self.offers:
            self.offers.move_to_end(key); self.counts["query_reuses"]+=1
            return self.offers[key]
        try:
            answer=p.offer(sid,aid,required=required,forbidden=forbidden,check=check)
        except JointLabelStop:
            self.columns_closed=True; self.counts["label_budget_flushes"]+=1
            return ()
        self.offers[key]=answer
        while len(self.offers)>512:
            self.offers.popitem(last=False)
        return answer

    def populate(self,p,plan,vehicles,targets,check):
        base=p.evaluate(plan)
        outside={s:frozenset(r for (scene,r),v in base["owners"].items()
            if scene==s and v not in vehicles) for s in p.weights}
        for v in sorted(vehicles):
            for aid in p.by_vehicle[v]:
                for sid in sorted(p.weights):
                    p.register(p.empty(sid,p.actions[aid]))
                    for r in sorted(targets.get(sid,())):
                        self.offer(p,sid,aid,r,outside[sid],check)
                    self.offer(p,sid,aid,None,outside[sid],check)

    def key(self,p,plan,vehicles):
        if self._problem is None:
            self._problem=p
        if self._problem is not p:
            raise ValueError("block workspace cannot retain certificates across joint epochs")
        if self._plan is not plan:
            # Conservative invalidation: ANY owner/current-root change clears
            # duplicate states. Keys therefore stay O(coalition), not O(all tasks).
            self._plan=plan; self._generation+=1; self.blocks.clear()
        return self._generation,tuple((v,p.vehicle_pool_versions[v]) for v in sorted(vehicles))

    def solve(self,p,plan,vehicles,deadline,check):
        check(); started=perf_counter(); key=self.key(p,plan,vehicles)
        if key in self.blocks:
            self.blocks.move_to_end(key); self.counts["duplicate_blocks_skipped"]+=1
            self.block_wall_s+=perf_counter()-started
            return plan,dict(variables=0,nonzeros=0,stages=0,solver_calls=0,
                future_chain_variables=0,current_action_variables=0,
                restricted_pool_optimality_proved=False,full_domain_optimality_proved=False,
                duplicate_block_skipped=True)
        try:
            best,info=solve_block(p,plan,vehicles,deadline,check,
                skip_constant_stages=True,metrics=self.metrics)
        finally:
            self.block_wall_s+=perf_counter()-started
        # Mark both the incoming and returned exact states; never cache across epochs.
        self.blocks[key]=None; self.blocks[self.key(p,best,vehicles)]=None
        while len(self.blocks)>128:
            self.blocks.popitem(last=False)
        return best,info


def incremental_search(p,initial,seeds,*,strategy,deadline,check,
                       maximum_coalition=6,maximum_partial_states=128):
    """Active-frontier search; lazy closure and ordinary spatial reference share work."""
    if strategy not in ("BOTTLENECK_CLOSURE","SPATIAL_JOINT"):
        raise ValueError("unknown incremental neighborhood strategy")
    if not 1<=maximum_coalition<=6 or not 1<=maximum_partial_states<=128:
        raise ValueError("incremental execution retains the declared resource/search caps")
    w=BlockWorkspace(); best=initial; trace=[]; partial=0; accepted=0; future_only=0
    mutable={v for v,ids in p.by_vehicle.items() if len(ids)>1}
    stopped=None; deferred=(); current_seed=None; index=-1
    if not mutable:
        stopped="NO_CURRENT_ACTION_CHOICE"; deferred=tuple(seeds); seeds=()

    def supported(sid,rid,state=None):
        for aid in p.oracle.supports(sid,rid,check):
            check()
            if w.columns_closed:
                return
            v=p.actions[aid].vehicle_id
            if state is not None and v in state.vehicles:
                continue
            for c in w.offer(p,sid,aid,rid,frozenset(),check):
                yield (Coalition(frozenset((v,)),(c,)) if state is None else
                       Coalition(state.vehicles|{v},(*state.hints,c)))

    def expansions(state,displaced):
        for sid,requests in sorted(displaced.items()):
            for rid in sorted(requests):
                yield from supported(sid,rid,state)

    def evaluate_group(vehicles,targets):
        nonlocal best,accepted,future_only
        if not vehicles & mutable:
            w.counts["fixed_only_blocks_skipped"]+=1
            return
        w.populate(p,best,vehicles,targets,check)
        before=p.evaluate(best)
        candidate,info=w.solve(p,best,vehicles,deadline,check)
        after=p.evaluate(candidate); changed=after["score"]>before["score"]
        trace.append(dict(coalition_size=len(vehicles),before=list(before["score"]),
            after=list(after["score"]),improved=changed,**info))
        if changed:
            # A completed, validated improvement survives a subsequent deadline.
            first_changed=any(before["chosen"][v]!=after["chosen"][v] for v in vehicles)
            future_only+=not first_changed
            best=candidate; accepted+=1

    try:
        for index,current_seed in enumerate(seeds):
            check(); sid,rid=current_seed
            if strategy=="SPATIAL_JOINT":
                nearby=p.oracle.nearby_resources(sid,rid,maximum_coalition,check)
                group=set(nearby); owner=p.evaluate(best)["owners"].get(current_seed)
                if owner is not None and owner not in group:
                    if len(group)>=maximum_coalition:
                        group.remove(nearby[-1])
                    group.add(owner)
                if group and not group & mutable:
                    active=p.oracle.nearby_resources(sid,rid,1,check,eligible_vehicles=mutable)
                    if active:
                        if len(group)>=maximum_coalition:
                            # Adding an outside owner may already have removed
                            # the original farthest neighbor. Evict only a
                            # member still present, retaining the owner when
                            # the declared coalition size can hold both.
                            evict=next((v for v in reversed(nearby)
                                if v in group and v!=owner),min(group))
                            group.remove(evict)
                        group.add(active[0])
                if group:
                    targets=defaultdict(set); targets[sid].add(rid)
                    for c in best.chains:
                        if c.vehicle_id in group:
                            targets[c.scenario_id].update(c.request_ids)
                    partial+=1; evaluate_group(group,targets)
            else:
                # Generators alternate: do not price 128 roots before the first solve.
                queue=deque((iter(supported(sid,rid)),)); visited=set()
                while queue and partial<maximum_partial_states:
                    check(); generator=queue.popleft()
                    try:
                        state=next(generator)
                    except StopIteration:
                        continue
                    queue.append(generator)
                    closed=close_coalition(p,best,state,maximum_coalition)
                    if closed is None:
                        continue
                    state,displaced=closed
                    key=state.vehicles,tuple(sorted(c.chain_id for c in state.hints))
                    if key in visited:
                        continue
                    visited.add(key); partial+=1
                    targets=defaultdict(set)
                    for s,requests in displaced.items():
                        targets[s].update(requests)
                    for c in state.hints:
                        targets[c.scenario_id].update(c.request_ids)
                    evaluate_group(state.vehicles,targets)
                    if len(state.vehicles)<maximum_coalition and not w.columns_closed:
                        queue.append(iter(expansions(state,displaced)))
                    if w.columns_closed:
                        break
            if w.columns_closed or partial>=maximum_partial_states:
                stopped="LABEL_CAP_FLUSHED" if w.columns_closed else "PARTIAL_STATE_CAP"
                deferred=tuple((*seeds[index+1:],current_seed)); break
    except JointBudgetStop as error:
        stopped=type(error).__name__
        deferred=tuple((*seeds[index+1:],*((current_seed,) if current_seed is not None else ())))
    base=p.evaluate(best)
    return best,dict(status="JOINT_FEASIBLE_IMPROVED" if accepted else "JOINT_FEASIBLE_INCUMBENT",
        execution_mode="INCREMENTAL_V2",neighborhood_strategy=strategy,score=list(base["score"]),
        expected_future_served=base["expected_future_served"],accepted_coalitions=accepted,
        future_only_accepted_coalitions=future_only,partial_states=partial,neighborhoods=trace,
        stop_reason=stopped,pool_columns=len(p.columns),same_first_action_across_scenarios=True,
        shared_customer_capacity=True,whole_day_optimality_claimed=False,
        full_domain_optimality_proved=False,dense_matrix=False,gpu_used=False,
        wall_overrun=perf_counter()>deadline,search_work=dict(w.counts,
            solver_calls=w.metrics["solver_calls"],
            constant_objective_stages_skipped=w.metrics["constant_objective_stages_skipped"]),
        integer_solve_time_s=w.metrics["solver_wall_time_s"],
        integer_model_build_time_s=w.metrics["model_build_time_s"],
        block_processing_time_s=w.block_wall_s,_deferred_seeds=deferred)


def execute_incremental(actions,roots,selected,scenes,pending,pace,library,now,cfg,
                        memory,deadline,check):
    from .bottleneck_scene import PredictionSceneOracle
    if memory.last_now is not None and now<memory.last_now:
        raise ValueError("incremental plan cannot be reused backwards in simulation time")
    timings={}; memory.last_attempt_timings=timings
    @contextmanager
    def timed(key):
        started=perf_counter()
        try:
            yield
        finally:
            timings[key]=perf_counter()-started
    graph_before=Counter(memory.topology.counts)
    with timed("forecast_setup_time_s"):
        snapshot=library.coarse_snapshot(now) if hasattr(library,"coarse_snapshot") else scenes
        oracle=PredictionSceneOracle(actions,roots,scenes,pending,pace,now,check=check,
            maximum_labels=int(cfg.get("joint_maximum_labels",50000)),
            maximum_columns=int(cfg.get("maximum_model_variables",20000)),
            topology=memory.topology,forecast_snapshot=snapshot,efficient=memory.efficient)
        p=JointProblem(actions,oracle,library.weights,[t.request_id for t in pending],move_cap=50,
            maximum_columns=int(cfg.get("maximum_model_variables",20000)),
            maximum_nonzeros=int(cfg.get("maximum_model_nonzeros",150000)))
    with timed("warm_plan_repair_time_s"):
        initial,warm,dirty=memory.initial(p,selected,check,
            perf_counter()+max(0.,deadline-perf_counter())*.25)
    with timed("frontier_build_time_s"):
        seeds=memory.frontier(p,initial,dirty,check)
    with timed("search_total_time_s"):
        search=incremental_search
        if memory.efficient:
            from .bottleneck_efficient import efficient_search
            search=efficient_search
        plan,info=search(p,initial,seeds,
            strategy=cfg.get("joint_neighborhood_strategy","BOTTLENECK_CLOSURE"),
            deadline=deadline,check=check,
            maximum_coalition=int(cfg.get("joint_maximum_coalition",6)),
            maximum_partial_states=int(cfg.get("joint_maximum_partial_states",128)))
    deferred=info.pop("_deferred_seeds")
    with timed("plan_retention_time_s"):
        memory.save(p,initial,plan,deferred,now)
    timings.update(column_query_time_s=oracle.query_wall_time_s,
        integer_solve_time_s=info["integer_solve_time_s"],
        integer_model_build_time_s=info["integer_model_build_time_s"],
        block_processing_time_s=info["block_processing_time_s"])
    timings["search_other_time_s"]=(timings["search_total_time_s"]-oracle.query_wall_time_s
        -info["integer_solve_time_s"]-info["integer_model_build_time_s"])
    before=p.evaluate(initial)["chosen"]; after=p.evaluate(plan)["chosen"]
    changed=sum(before[v]!=after[v] for v in p.by_vehicle)
    info.update(warm,phase_timings_s=timings,frontier_seed_count=len(seeds),
        deferred_seed_count=len(deferred),changed_current_action_count=changed,
        current_action_changed=bool(changed),target_pruned_nodes=oracle.target_pruned_nodes,
        actual_query_calls=oracle.query_calls,
        topology_work={k:v-graph_before[k] for k,v in memory.topology.counts.items()},
        topology_cached_references=memory.topology.references,
        warm_state_persisted=True,wall_overrun=perf_counter()>deadline,
        batch_root_queries=oracle.batch_queries,batch_query_reuses=oracle.batch_reuses,
        phase_timers_are_nested=True)
    return p,oracle,initial,plan,info
