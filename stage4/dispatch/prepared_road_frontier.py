"""One shared, sparse candidate/evidence frontier for prefetch and search.

Candidate geometry is not a road certificate. Only indexed positive evidence
is traversed by physical future chains; optimistic current-move probes keep
their old explicitly non-executable discovery semantics.
"""
from collections import Counter,defaultdict,deque
from itertools import groupby
from math import ceil

from .scheme_a_city_graph import RouteState
from .sparse_road_support import task_key


class PreparedRoadFrontier:
    def __init__(self,oracle,support):
        self.q=oracle; self.support=support; self.index=support.sparse_index
        self.aliases={s:defaultdict(list) for s in oracle.tasks}
        for sid,tasks in oracle.tasks.items():
            for task in tasks.values(): self.aliases[sid][task_key(task)].append(task)
        self.root_views={}; self.following_views={}; self.reverse={}
        self.occupied=frozenset(); self.useful_seeds=set(); self.counts=Counter()
        self.root_references=0; self.following_references=0

    @staticmethod
    def relaxed(state):
        return (state.context or {}).get("kind")=="RELAXED_CURRENT_MOVE_PROBE"

    def _leg(self,state,task):
        ready=ceil(state.ready_s/300)*300 if task.source_kind=="FORECAST" else state.ready_s
        return self.q._leg(task,ready,state.position,state.admission_end_s)

    def _choices(self,sid,aid,state,previous=None):
        q=self.q
        rows=q._root_candidates(sid,state) if previous is None else q._following(sid,previous,state.profile_id)
        return [t for t in rows if t.request_id not in self.occupied
            and t.request_id!=q.actions[aid].request_id and q._allowed(t,state.profile_id)
            and self._leg(state,t) is not None]

    def prepare(self,selected,memory,check):
        q=self.q; selected=set(selected)
        self.occupied=frozenset(q.actions[a].request_id for a in selected if q.actions[a].request_id is not None)
        by_type={p:[] for p in ("C","HV")}
        for aid,a in q.actions.items():
            state=q.roots[aid]
            if self.relaxed(state) or state.ready_s>=min(state.admission_end_s,q.horizon): continue
            warm=any(getattr(memory,"paths",{}).get((s,a.vehicle_id)) for s in q.tasks)
            # Exact refinement must prepare the certified move roots that its
            # preceding discovery round actually selected, not unrelated WAITs.
            by_type[state.profile_id].append((not warm,a.kind!="RELOCATE",aid not in selected,
                state.ready_s,a.vehicle_id,aid))
        for profile,rows in by_type.items():
            rows.sort(); rotated=[]
            for _,group in groupby(rows,key=lambda r:r[:4]):
                group=list(group); offset=int(q.now//30)%len(group)
                rotated.extend(group[offset:]+group[:offset])
            by_type[profile]=rotated
        order=("C","HV") if int(q.now//30)%2==0 else ("HV","C")
        scenes=sorted(q.tasks); offset=int(q.now//30)%max(1,len(scenes)); scenes=scenes[offset:]+scenes[:offset]
        queue=deque(); roots_examined=0
        # Keep the old four viable roots/type breadth and 64 assessment cap.
        # Unlike lexicographic IDs, viability and retained supported work matter.
        viable={p:0 for p in order}
        iterators={p:iter(by_type[p]) for p in order}
        while roots_examined<64 and any(viable[p]<4 for p in order):
            advanced=False
            for profile in order:
                if roots_examined>=64: break
                if viable[profile]>=4: continue
                row=next(iterators[profile],None)
                if row is None: viable[profile]=4; continue
                advanced=True; roots_examined+=1; check(); aid=row[-1]; state=q.roots[aid]
                self.index.hydrate(state)
                offers=[]
                for sid in scenes:
                    choices=self._choices(sid,aid,state)
                    old=getattr(memory,"paths",{}).get((sid,q.actions[aid].vehicle_id),())
                    choices.sort(key=lambda t:t.request_id not in old)
                    # Stable sort retains exactly the oracle's contribution order.
                    offers.extend((sid,aid,state,t,1,frozenset()) for t in choices[:3])
                if offers:
                    queue.extend(offers); viable[profile]+=1
            if not advanced: break
        seen=set()
        while queue and len(seen)<64:
            sid,aid,state,task,depth,used=queue.popleft()
            key=self.index.source_key(state),task_key(task)
            if key in seen or task.request_id in used: continue
            seen.add(key); check(); upper=self._leg(state,task)
            if upper is None: continue
            result=self.support.prepare_edge(state,task,upper[1])
            if not result["supported"] or not 0<=float(result["travel_time_s"])<=600.: continue
            self.counts["prepared_positive_assessments"]+=1
            if depth>=3: continue
            after=RouteState(task.dropoff,upper[1]+float(result["travel_time_s"])+task.service_time_s,
                state.admission_end_s,state.profile_id,dict(kind="CUSTOMER",task=task))
            self.index.hydrate(after)
            for nxt in reversed(self._choices(sid,aid,after,task)):
                queue.appendleft((sid,aid,after,nxt,depth+1,used|{task.request_id}))
        self.counts.update(roots_examined=roots_examined,shared_candidate_assessments=len(seen),
            globally_occupied_current_requests_skipped=len(self.occupied))
        self._build_reverse(check)
        return tuple(sorted(self.useful_seeds))

    def root_rows(self,sid,aid):
        key=sid,aid
        if key in self.root_views: return self.root_views[key]
        q=self.q; state=q.roots[aid]
        if self.relaxed(state): return q._root_candidates(sid,state)
        rows=[t for physical in self.index.targets(state) for t in self.aliases[sid].get(physical,())
            if t.request_id!=q.actions[aid].request_id and q._allowed(t,state.profile_id) and self._leg(state,t) is not None]
        q._sort_root_rows(rows,sid,state)
        if self.root_references+len(rows)>20000:
            from .bottleneck_joint import JointSizeStop
            raise JointSizeStop("prepared sparse root-reference cap")
        self.root_views[key]=tuple(rows)
        self.root_references+=len(rows)
        if not rows: self.counts["root_without_indexed_support_skipped"]+=1
        return self.root_views[key]

    def following(self,sid,node,profile):
        key=sid,node.request_id,profile
        if key not in self.following_views:
            state=RouteState(node.dropoff,self.q.now,self.q.horizon,profile,dict(kind="CUSTOMER",task=node))
            targets=set(self.index.targets(state))
            # Filter AFTER the original C Top3 / nested HV Top6; never backfill
            # with an out-of-domain road just because it is already cached.
            rows=tuple(t for t in self.q._following(sid,node,profile) if task_key(t) in targets)
            if self.following_references+len(rows)>20000:
                from .bottleneck_joint import JointSizeStop
                raise JointSizeStop("prepared sparse successor-reference cap")
            self.following_views[key]=rows; self.following_references+=len(rows)
        return self.following_views[key]

    def _build_reverse(self,check):
        q=self.q; reverse=defaultdict(list); served_by_vehicle=defaultdict(set)
        for action in q.actions.values():
            if action.request_id is not None: served_by_vehicle[action.vehicle_id].add(action.request_id)
        for sid in q.tasks:
            for aid,a in q.actions.items():
                if self.relaxed(q.roots[aid]): continue  # Discovery-only roots use the old geometric superset.
                check(); rows=self.root_rows(sid,aid)
                useful=[t for t in rows if t.request_id not in self.occupied]
                for t in rows: reverse[sid,t.request_id].append(aid)
                if useful:
                    self.useful_seeds.update((sid,t.request_id) for t in useful)
                    self.useful_seeds.update((sid,r) for r in served_by_vehicle[a.vehicle_id])
        self.reverse={k:tuple(sorted(v,key=lambda a:(q.actions[a].vehicle_id,a))) for k,v in reverse.items()}

    def supports(self,sid,rid):
        return self.reverse.get((sid,rid),())

    def reprice(self):
        self.root_views.clear(); self.root_references=0

    def diagnostics(self):
        return dict(counts=dict(self.counts),indexed_root_task_references=sum(map(len,self.root_views.values())),
            useful_seed_count=len(self.useful_seeds),search_candidate_pool="SAME_PREPARED_ROAD_SUPPORT",
            candidate_geometry_is_not_a_certificate=True)
