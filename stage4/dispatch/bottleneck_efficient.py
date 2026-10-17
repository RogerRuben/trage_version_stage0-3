"""V3: bounded dependency-local batching and indexed exact joint blocks."""
from collections import Counter,OrderedDict,defaultdict,deque
from time import perf_counter

from .bottleneck_joint import Coalition,JointBudgetStop,JointLabelStop,solve_block
from .bottleneck_plan_index import PlanIndex


class EfficientWorkspace:
    def __init__(self,p,index,check):
        self.p=p; self.index=index; self.check=check
        self.counts=Counter(); self.metrics=Counter(); self.visits=OrderedDict()
        self.visit_refs=0; self.columns_closed=False; self.block_wall_s=0.

    def footprint(self,vehicles):
        p=self.p; found=set()
        for v in sorted(vehicles):
            for aid in p.by_vehicle[v]:
                a=p.actions[aid]
                for s in p.weights:
                    if a.request_id is not None: found.add((s,a.request_id))
                    if hasattr(p.oracle,"possible"):
                        found.update((s,r) for r in p.oracle.possible(s,aid,self.check))
                    else:
                        found.update((s,r) for c in p.oracle.columns.values()
                            if c.scenario_id==s and c.action_id==aid for r in c.request_ids)
        return found

    def signature(self,vehicles,footprint):
        i=self.index; p=self.p
        outside=frozenset(k for k in footprint if i.owners.get(k) is not None
            and i.owners[k] not in vehicles)
        local=tuple((v,i.chosen[v],tuple(i.chains[s,v].chain_id for s in p.weights),
            p.vehicle_pool_versions[v]) for v in sorted(vehicles))
        residual=(p.move_cap-i.moves+sum(i.contributions[v][4] for v in vehicles),
            p.protected_critical-i.critical+sum(i.contributions[v][3] for v in vehicles))
        if p.economics is not None:
            residual=(*residual,p.protected_current-i.current+sum(i.contributions[v][5] for v in vehicles),
                p.protected_carry-i.carry+sum(i.contributions[v][6] for v in vehicles))
        return local,outside,residual

    def offer_many(self,sid,aid,targets,forbidden):
        if self.columns_closed: return {}
        p=self.p
        try:
            if hasattr(p.oracle,"query_many"):
                answers=p.oracle.query_many(sid,aid,targets,forbidden,self.check)
                if p.oracle.many_label_exhausted:
                    self.columns_closed=True; self.counts["label_budget_flushes"]+=1
                for paths in answers.values():
                    for c in paths: p.register(c)
            else:
                answers={r:p.offer(sid,aid,required=r,forbidden=forbidden,check=self.check)
                    for r in (*sorted(targets),None)}
        except JointLabelStop:
            self.columns_closed=True; self.counts["label_budget_flushes"]+=1
            return {}
        self.counts["batch_root_queries"]+=1
        return answers

    def hint(self,sid,aid,rid):
        return self.offer_many(sid,aid,{rid},frozenset()).get(rid,())

    def remember(self,vehicles,signature,covered):
        if vehicles in self.visits:
            old=self.visits.pop(vehicles); self.visit_refs-=len(old[0][1])+len(old[1])
        self.visits[vehicles]=(signature,covered); self.visit_refs+=len(signature[1])+len(covered)
        while len(self.visits)>128 or self.visit_refs>20000:
            _,old=self.visits.popitem(last=False); self.visit_refs-=len(old[0][1])+len(old[1])

    def evaluate(self,vehicles,targets,deadline):
        if self.p.economics is not None and self.p.economic_budget["blocks"]>=3:
            raise JointBudgetStop("ECONOMIC_THREE_BLOCK_REPAIR_CAP")
        self.check(); started=perf_counter(); p=self.p; i=self.index
        vehicles=frozenset(vehicles); footprint=self.footprint(vehicles)
        targets={s:set(ids) for s,ids in targets.items()}
        actual={r for ids in targets.values() for r in ids if r in p.current_ids}
        for s in p.weights: targets.setdefault(s,set()).update(actual)
        wanted=frozenset((s,r) for s,ids in targets.items() for r in ids)
        signature=self.signature(vehicles,footprint)
        previous=self.visits.get(vehicles)
        if previous is not None and previous[0]==signature and wanted<=previous[1]:
            self.counts["early_blocks_skipped"]+=1
            return None
        for v in sorted(vehicles):
            for aid in p.by_vehicle[v]:
                for s in p.weights:
                    p.register(p.empty(s,p.actions[aid]))
                    forbidden=frozenset(r for scene,r in footprint if scene==s
                        and i.owners.get((s,r)) is not None and i.owners[s,r] not in vehicles)
                    required=set(targets.get(s,()))
                    if hasattr(p.oracle,"possible"): required&=p.oracle.possible(s,aid,self.check)
                    self.offer_many(s,aid,required,forbidden)
        after_pricing=self.signature(vehicles,footprint)
        if previous is not None and previous[0]==after_pricing:
            self.remember(vehicles,after_pricing,wanted|previous[1])
            self.counts["unchanged_column_blocks_skipped"]+=1
            return None
        before=i.score; old_chosen={v:i.chosen[v] for v in vehicles}
        fixed={}
        if p.economics is None and hasattr(p.oracle,"tasks"):
            upper=sum(p.units[s]*len(set(p.oracle.tasks[s])|p.current_ids) for s in p.weights)
            if i.score[0]==upper:
                fixed[0]=sum(i.contributions[v][0] for v in vehicles)
            elif hasattr(p.oracle,"possible"):
                # Relax root consistency across scenes and chain adjacency,
                # but retain unique tasks, outside occupation and <=3 futures.
                # This is a valid upper bound over ALL legal roots, not only
                # the restricted columns generated so far.
                local_upper=0
                for s in p.weights:
                    union=set(); capacity=0
                    for v in vehicles:
                        vehicle_cap=0
                        for aid in p.by_vehicle[v]:
                            a=p.actions[aid]
                            if a.request_id is not None and i.owners.get((s,a.request_id)) not in (None,*vehicles):
                                continue
                            available={r for r in p.oracle.possible(s,aid,self.check)
                                if r!=a.request_id and i.owners.get((s,r)) in (None,*vehicles)}
                            union.update(available)
                            if a.request_id is not None: union.add(a.request_id)
                            vehicle_cap=max(vehicle_cap,int(a.kind=="SERVE")+min(3,len(available)))
                        capacity+=vehicle_cap
                    local_upper+=p.units[s]*min(capacity,len(union))
                if local_upper==sum(i.contributions[v][0] for v in vehicles):
                    fixed[0]=local_upper
        carry_upper=sum(max(int(p.actions[a].carry_over) for a in ids) for ids in p.by_vehicle.values())
        # Attaining a carry upper bound in the old incumbent does NOT prove it
        # remains feasible after a better service tier. Only fix the lower tier
        # when the preceding tier is already fixed at this same witness.
        if 0 in fixed and i.score[1]==carry_upper:
            fixed[1]=sum(i.contributions[v][1] for v in vehicles)
        try:
            if p.economics is not None:
                self.counts["economic_blocks_solved"]+=1; p.economic_budget["blocks"]+=1
            patch,info=solve_block(p,None,vehicles,deadline,self.check,skip_constant_stages=True,
                metrics=self.metrics,plan_index=i,persistent=True,fixed_objectives=fixed)
        finally:
            self.block_wall_s+=perf_counter()-started
        changed=patch is not None
        first_changed=changed and any(old_chosen[p.actions[a].vehicle_id]!=a for a in patch.actions)
        if changed: i.apply(patch)
        covered=wanted|(previous[1] if previous is not None and previous[0]==signature else frozenset())
        signature=self.signature(vehicles,footprint)
        self.remember(vehicles,signature,covered)
        return dict(coalition_size=len(vehicles),before=list(before),after=list(i.score),
            improved=changed,current_root_changed=bool(first_changed),**info)


def coalesce(groups,maximum,counts):
    """Combine overlapping supports only while their union remains <= six."""
    merged=[]
    for vehicles,targets in groups:
        vehicles=set(vehicles); targets={s:set(ids) for s,ids in targets.items()}
        for j,(old,old_targets) in enumerate(merged):
            if vehicles&old and len(vehicles|old)<=maximum:
                merged[j]=(vehicles|old,{s:targets.get(s,set())|old_targets.get(s,set())
                    for s in targets.keys()|old_targets.keys()})
                counts["groups_merged"]+=1
                break
        else: merged.append((vehicles,targets))
    return merged


def efficient_search(p,initial,seeds,*,strategy,deadline,check,
                     maximum_coalition=6,maximum_partial_states=128,first_current_improvement=True):
    if strategy not in ("SPATIAL_JOINT","BOTTLENECK_CLOSURE","ADAPTIVE_SHARED"):
        raise ValueError("unknown efficient strategy")
    if not 1<=maximum_coalition<=6 or not 1<=maximum_partial_states<=128:
        raise ValueError("V3 cannot enlarge search/resource limits")
    index=PlanIndex(p,initial); w=EfficientWorkspace(p,index,check)
    mutable={v for v,ids in p.by_vehicle.items() if len(ids)>1}
    trace=[]; partial=0; accepted=0; future_only=0; stopped=None; unprocessed=()
    current_support=defaultdict(list)
    for a in p.actions.values():
        if a.kind=="SERVE": current_support[a.request_id].append(a.action_id)
    if p.economics is not None:
        def priority(key):
            sid,rid=key; task=p.oracle.tasks[sid][rid]
            alternatives=[p.economics._quote_values(task.service_distance_m,profile,0.)[3]
                for profile in ("C","HV") if p.oracle._allowed(task,profile)]
            estimate=max(alternatives,default=0)-p.oracle.price_units.get(key,0.)
            return task.source_kind!="ACTUAL_PENDING",-estimate,task.deadline_s,key
        seeds=tuple(sorted(set(seeds),key=priority))
    def target_map(): return defaultdict(set)
    def supported(sid,rid,state=None):
        # A last-chance current request need not fit any FUTURE continuation.
        # Its already validated current SERVE roots must still seed a swap.
        for aid in sorted(current_support.get(rid,())):
            v=p.actions[aid].vehicle_id
            if state is not None and v in state.vehicles: continue
            c=p.register(p.empty(sid,p.actions[aid]))
            yield Coalition(frozenset((v,)),(c,)) if state is None else Coalition(
                state.vehicles|{v},(*state.hints,c))
        for aid in p.oracle.supports(sid,rid,check):
            check()
            if w.columns_closed: return
            v=p.actions[aid].vehicle_id
            if state is not None and v in state.vehicles: continue
            for c in w.hint(sid,aid,rid):
                yield Coalition(frozenset((v,)),(c,)) if state is None else Coalition(
                    state.vehicles|{v},(*state.hints,c))
    def expand(state,displaced):
        for sid,requests in sorted(displaced.items()):
            for rid in sorted(requests): yield from supported(sid,rid,state)
    def flush(groups):
        nonlocal accepted,future_only
        for vehicles,targets in coalesce(groups,maximum_coalition,w.counts):
            if not vehicles&mutable:
                w.counts["fixed_only_blocks_skipped"]+=1; continue
            result=w.evaluate(vehicles,targets,deadline)
            if result is None: continue
            trace.append(result)
            if result["improved"]:
                accepted+=1; future_only+=not result["current_root_changed"]
                if first_current_improvement and result["current_root_changed"]:
                    return True
        return False
    groups=[]
    try:
        if not mutable: stopped="NO_CURRENT_ACTION_CHOICE"
        for n,(sid,rid) in enumerate(seeds if mutable else ()):
            check(); unprocessed=tuple(seeds[n:])
            mode=strategy
            if mode=="ADAPTIVE_SHARED":
                options=current_support.get(rid,())
                profiles={p.oracle.roots[a].profile_id for a in options}
                contested=profiles=={"HV","C"}
                uncovered=bool(options) and (sid,rid) not in index.owners
                mode="BOTTLENECK_CLOSURE" if contested or uncovered else "SPATIAL_JOINT"
                w.counts["adaptive_bottleneck_seeds" if mode=="BOTTLENECK_CLOSURE"
                         else "adaptive_spatial_seeds"]+=1
                if mode=="BOTTLENECK_CLOSURE" and groups:
                    improved=flush(groups); groups=[]
                    if improved:
                        stopped="FIRST_CURRENT_ACTION_IMPROVEMENT"; break
            if mode=="SPATIAL_JOINT":
                near=p.oracle.nearby_resources(sid,rid,maximum_coalition,check)
                group=set(near); owner=index.owners.get((sid,rid))
                if owner is not None and owner not in group:
                    if len(group)>=maximum_coalition: group.remove(near[-1])
                    group.add(owner)
                if group and not group&mutable:
                    active=p.oracle.nearby_resources(sid,rid,1,check,eligible_vehicles=mutable)
                    if active:
                        if len(group)>=maximum_coalition:
                            group.remove(next((v for v in reversed(near) if v in group and v!=owner),min(group)))
                        group.add(active[0])
                if group:
                    targets=target_map(); targets[sid].add(rid)
                    groups.append((group,targets)); partial+=1
                # Collect the entire bounded spatial frontier before pricing.
            else:
                if rid in current_support:
                    # Cheap direct-current closure before deep future pricing.
                    # Last-chance requests need no fictitious future pickup.
                    direct=[]
                    ranked=sorted(current_support[rid],key=lambda a:(
                        -p.action_score(p.actions[a])[0] if p.economics is not None else 0,
                        p.actions[a].empty_distance_m,p.actions[a].vehicle_id,a))
                    for aid in ranked:
                        if partial>=maximum_partial_states: break
                        c=p.register(p.empty(sid,p.actions[aid])); a=p.actions[aid]
                        closed=index.close(Coalition(frozenset((a.vehicle_id,)),(c,)),maximum_coalition)
                        if closed is None: continue
                        state,displaced=closed; targets=target_map(); targets[sid].add(rid)
                        for s,ids in displaced.items(): targets[s].update(ids)
                        direct.append((state.vehicles,targets)); partial+=1
                    if flush(direct):
                        stopped="FIRST_CURRENT_ACTION_IMPROVEMENT"; break
                queue=deque((iter(supported(sid,rid)),)); visited=set()
                while queue and partial<maximum_partial_states:
                    check(); generator=queue.popleft()
                    try: state=next(generator)
                    except StopIteration: continue
                    queue.append(generator)
                    closed=index.close(state,maximum_coalition)
                    if closed is None: continue
                    state,displaced=closed; key=state.vehicles,tuple(sorted(c.chain_id for c in state.hints))
                    if key in visited: continue
                    visited.add(key); partial+=1; targets=target_map()
                    for s,ids in displaced.items(): targets[s].update(ids)
                    for c in state.hints: targets[c.scenario_id].update(c.request_ids)
                    groups.append((state.vehicles,targets))
                    if len(state.vehicles)<maximum_coalition and not w.columns_closed:
                        queue.append(iter(expand(state,displaced)))
                    if w.columns_closed: break
                improved=flush(groups); groups=[]
                if improved:
                    stopped="FIRST_CURRENT_ACTION_IMPROVEMENT"; break
            if partial>=maximum_partial_states or w.columns_closed:
                stopped="LABEL_CAP_FLUSHED" if w.columns_closed else "PARTIAL_STATE_CAP"; break
            unprocessed=tuple(seeds[n+1:])
        if flush(groups): stopped="FIRST_CURRENT_ACTION_IMPROVEMENT"
        groups=[]
    except JointBudgetStop as error:
        stopped=type(error).__name__
    plan=index.materialize(); checked=p.evaluate(plan)
    if checked["score"]!=index.score: raise ValueError("indexed final score conservation failed")
    return plan,dict(status="JOINT_FEASIBLE_IMPROVED" if accepted else "JOINT_FEASIBLE_INCUMBENT",
        execution_mode="INCREMENTAL_V3",neighborhood_strategy=strategy,score=list(index.score),
        expected_future_served=checked["expected_future_served"],accepted_coalitions=accepted,
        future_only_accepted_coalitions=future_only,partial_states=partial,neighborhoods=trace,
        stop_reason=stopped,pool_columns=len(p.columns),same_first_action_across_scenarios=True,
        shared_customer_capacity=True,whole_day_optimality_claimed=False,full_domain_optimality_proved=False,
        dense_matrix=False,gpu_used=False,wall_overrun=perf_counter()>deadline,
        search_work={**w.counts,**{k:v for k,v in w.metrics.items() if not k.endswith("_s")},
            "solver_calls":w.metrics["solver_calls"],
            "first_current_improvement_stops":int(stopped=="FIRST_CURRENT_ACTION_IMPROVEMENT")},integer_solve_time_s=w.metrics["solver_wall_time_s"],
        integer_model_build_time_s=w.metrics["model_build_time_s"],block_processing_time_s=w.block_wall_s,
        _deferred_seeds=unprocessed)
