"""One epoch's validated plan index; exact small-block deltas, no dense arrays."""
from dataclasses import dataclass

from .bottleneck_joint import JointPlan


@dataclass(frozen=True)
class PlanPatch:
    vehicles: frozenset
    actions: tuple
    chains: tuple
    score: tuple
    critical: int
    moves: int
    current: int


class PlanIndex:
    def __init__(self,problem,plan):
        self.p=problem
        checked=problem.evaluate(plan)
        self.chosen=dict(checked["chosen"]); self.chains=dict(checked["chains"])
        self.owners=dict(checked["owners"]); self.score=checked["score"]
        self.critical=checked["critical"]; self.moves=checked["current_moves"]
        self.current=checked["current_served"]; self.revision=0
        self.vehicle_revisions={v:0 for v in self.chosen}
        self.request_revisions={}; self.contributions={v:self._contribution(v) for v in self.chosen}

    def _contribution(self,v,aid=None,chains=None):
        p=self.p; a=p.actions[self.chosen[v] if aid is None else aid]
        chains=(tuple(self.chains[s,v] for s in p.weights) if chains is None else chains)
        service=p.scale*int(a.kind=="SERVE")
        distance=p.scale*int(round(a.empty_distance_m*1000))
        for c in chains:
            service+=p.units[c.scenario_id]*len(c.request_ids)
            distance+=p.units[c.scenario_id]*int(round(c.empty_distance_m*1000))
        return service,int(a.carry_over),-distance,int(a.critical),int(a.kind=="RELOCATE"),int(a.kind=="SERVE")

    def view(self):
        return dict(chosen=self.chosen,chains=self.chains,owners=self.owners,score=self.score,
            critical=self.critical,current_moves=self.moves,current_served=self.current,
            expected_future_served=(self.score[0]-self.p.scale*self.current)/self.p.scale)

    def local_keys(self,vehicles):
        keys=set()
        for v in vehicles:
            a=self.p.actions[self.chosen[v]]
            if a.request_id is not None:
                keys.update((s,a.request_id) for s in self.p.weights)
            for s in self.p.weights:
                keys.update((s,r) for r in self.chains[s,v].request_ids)
        return keys

    def propose(self,vehicles,actions,chains):
        p=self.p; vehicles=frozenset(vehicles)
        chosen={}; groups={}; used={}; contributions={}
        for aid in actions:
            a=p.actions.get(aid)
            if a is None or a.vehicle_id not in vehicles or a.vehicle_id in chosen:
                raise ValueError("indexed patch duplicates/changes an outside resource")
            chosen[a.vehicle_id]=aid
            if a.request_id is not None:
                for s in p.weights:
                    key=s,a.request_id
                    if key in used:
                        raise ValueError("indexed patch duplicates current request")
                    used[key]=a.vehicle_id
        if chosen.keys()!=vehicles:
            raise ValueError("indexed patch loses a resource")
        for c in chains:
            key=c.scenario_id,c.vehicle_id
            if (c.vehicle_id not in vehicles or key in groups or c.scenario_id not in p.weights
                    or chosen[c.vehicle_id]!=c.action_id or p.columns.get(c.chain_id)!=c):
                raise ValueError("indexed patch has an invalid scenario/root chain")
            groups[key]=c
            for r in c.request_ids:
                request=c.scenario_id,r
                if request in used:
                    raise ValueError("indexed patch allocates shared request twice")
                used[request]=c.vehicle_id
        if len(groups)!=len(vehicles)*len(p.weights):
            raise ValueError("indexed patch loses scenario continuation")
        for request in used:
            owner=self.owners.get(request)
            if owner is not None and owner not in vehicles:
                raise ValueError("indexed patch steals outside request capacity")
        totals=[*self.score,self.critical,self.moves,self.current]
        for v in vehicles:
            contributions[v]=self._contribution(v,chosen[v],tuple(groups[s,v] for s in p.weights))
            for i,(old,new) in enumerate(zip(self.contributions[v],contributions[v])):
                totals[i]+=new-old
        if totals[4]>p.move_cap or (p.protected_critical is not None and totals[3]!=p.protected_critical):
            raise ValueError("indexed patch violates shared move/critical capacity")
        return PlanPatch(vehicles,tuple(actions),tuple(chains),tuple(totals[:3]),*totals[3:])

    def apply(self,patch):
        old=self.local_keys(patch.vehicles)
        for key in old:
            del self.owners[key]
        for aid in patch.actions:
            a=self.p.actions[aid]; self.chosen[a.vehicle_id]=aid
            if a.request_id is not None:
                for s in self.p.weights:
                    self.owners[s,a.request_id]=a.vehicle_id
        for c in patch.chains:
            self.chains[c.scenario_id,c.vehicle_id]=c
            for r in c.request_ids:
                self.owners[c.scenario_id,r]=c.vehicle_id
        self.score=patch.score; self.critical=patch.critical
        self.moves=patch.moves; self.current=patch.current; self.revision+=1
        for v in patch.vehicles:
            self.vehicle_revisions[v]+=1; self.contributions[v]=self._contribution(v)
        for key in old|self.local_keys(patch.vehicles):
            self.request_revisions[key]=self.request_revisions.get(key,0)+1

    def materialize(self):
        return JointPlan(tuple(sorted(self.chosen.values())),tuple(self.chains.values()))

    def close(self,coalition,maximum):
        from collections import defaultdict
        from .bottleneck_joint import Coalition
        p=self.p; vehicles=set(coalition.vehicles); roots={}; displaced=defaultdict(set)
        for c in coalition.hints:
            if p.columns.get(c.chain_id)!=c:
                raise ValueError("unregistered indexed closure hint")
            if roots.setdefault(c.vehicle_id,c.action_id)!=c.action_id:
                return None
            vehicles.add(c.vehicle_id); a=p.actions[c.action_id]
            for s in p.weights if self.chosen[c.vehicle_id]!=c.action_id else (c.scenario_id,):
                previous=self.chains[s,c.vehicle_id].request_ids
                displaced[s].update(set(previous)-set(c.request_ids) if s==c.scenario_id else previous)
            for r in (a.request_id,*c.request_ids):
                if r is None: continue
                for s in p.weights if r==a.request_id else (c.scenario_id,):
                    owner=self.owners.get((s,r))
                    if owner is not None: vehicles.add(owner)
        if len(vehicles)>maximum or not vehicles<=p.by_vehicle.keys(): return None
        return Coalition(frozenset(vehicles),coalition.hints),displaced
