"""Bounded prediction-only columns for joint control; no future road queries.

Earlier-history tasks retain the existing 300-s dispatch grid, 2-km neighbor
domain and C Top3 / nested HV Top6. Real pending customers may be booked at
later 30-s execution epochs, strictly before their ORIGINAL deadline.
Connections are the declared Train-M3 chord-pace planning proxy, not directed
Valhalla certificates. Physical current actions are validated by the caller.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
from math import ceil, isfinite

import numpy as np
from scipy.spatial import cKDTree

from .bottleneck_joint import JointBudgetStop,JointLabelStop
from time import perf_counter
from .scheme_a_city_graph import ChainTask, RouteState, _xy
from .scheme_a_city_master import ServiceChain


class PredictionSceneOracle:
    def __init__(self, actions, roots, scenes, pending, pace, now, *, check=lambda:None,
                 maximum_labels=50000,maximum_columns=20000,topology=None,forecast_snapshot=None,
                 efficient=False):
        self.actions={a.action_id:a for a in actions}
        self.roots=dict(roots)
        if self.actions.keys()!=self.roots.keys():
            raise ValueError("every joint current root needs its predicted after-state")
        self.now=float(now); self.horizon=self.now+1800
        self.pace=pace; self.check=check
        self.maximum_labels=maximum_labels; self.maximum_columns=maximum_columns
        self.labels=0; self.registered={}; self.cache=OrderedDict()
        self.tasks={}; self.trees={}; self.successors={}
        self.topology=topology; self.predecessors=OrderedDict(); self.reachable=OrderedDict()
        self.predecessor_refs=0; self.reachable_refs=0
        self.query_wall_time_s=0.; self.query_calls=0; self.target_pruned_nodes=0
        self.efficient=efficient; self.xy={}; self.distances=OrderedDict()
        self.support_cache=OrderedDict(); self.neighbor_cache=OrderedDict()
        self.catalogs=OrderedDict(); self.catalog_refs=0; self.many_label_exhausted=False
        self.batch_queries=0; self.batch_reuses=0
        self.possible_cache=OrderedDict(); self.possible_refs=0
        if topology is not None:
            topology.bind(scenes if forecast_snapshot is None else forecast_snapshot,pace,now,check)
        pending=tuple(pending)
        for task in pending:
            if (not isinstance(task,ChainTask) or task.source_kind!="ACTUAL_PENDING"
                    or task.request_id<0 or not task.release_s<=now<task.deadline_s):
                raise ValueError("pending continuation input must already be revealed and unexpired")
        for sid,forecast in scenes.items():
            check()
            future=tuple(forecast)
            for task in future:
                if (not isinstance(task,ChainTask) or task.source_kind!="FORECAST"
                        or task.request_id>=0 or not task.source_date
                        or task.source_date>"20161024" or task.release_s<=now):
                    raise ValueError("future columns require earlier-history predictions, not Test31 futures")
                if topology is not None and topology.task_map.get(sid,{}).get(task.request_id)!=task:
                    raise ValueError("fine prediction view differs from the cached coarse snapshot")
            tasks=tuple(sorted((*pending,*future),key=lambda t:(t.release_s,t.request_id)))
            if len({t.request_id for t in tasks})!=len(tasks):
                raise ValueError("observed/history request identity collision")
            if any("C" in t.compatible_profiles and "HV" not in t.compatible_profiles for t in tasks):
                raise ValueError("the declared planning profile domain must be nested")
            self.tasks[sid]={t.request_id:t for t in tasks}
            indexed=tasks if topology is None else pending
            points=np.asarray([_xy(t.pickup) for t in indexed],float).reshape(-1,2)
            self.trees[sid]=(cKDTree(points) if len(points) else None,indexed,points)
        points=np.asarray([_xy(t.dropoff) for t in pending],float).reshape(-1,2)
        self.pending_dropoffs=(cKDTree(points) if len(points) else None,pending)
        self.aids=tuple(sorted(self.actions))
        points=np.asarray([_xy(self.roots[a].position) for a in self.aids],float).reshape(-1,2)
        self.root_tree=cKDTree(points) if len(points) else None
        self.root_points=points
        for aid,state in self.roots.items():
            if not isinstance(state,RouteState) or state.profile_id not in ("C","HV"):
                raise ValueError("joint roots retain typed C/HV time-position states")

    def accepts(self,c):
        return self.registered.get(c.chain_id)==c

    def _remember_path(self,sid,aid,path):
        digest=hashlib.sha256(repr((sid,aid,path)).encode()).hexdigest()[:24]
        c=ServiceChain(f"{sid}|{aid}|{digest}",sid,self.actions[aid].vehicle_id,aid,
            tuple(l[0] for l in path),sum(l[4] for l in path),dict(legs=path,
                support="EARLIER_HISTORY_M3_CHORD_PACE_NOT_DIRECTED_ROUTE_CERTIFICATE"))
        if len(self.registered)>=self.maximum_columns and c.chain_id not in self.registered:
            raise JointBudgetStop("joint oracle column memory cap exhausted")
        self.registered[c.chain_id]=c
        return c

    def seed(self,sid,aid,forbidden,check):
        """Cheap disjoint witness; deliberately consumes NO deep-search labels."""
        state=self.roots[aid]
        forbidden=set(forbidden)
        if self.actions[aid].request_id is not None:
            forbidden.add(self.actions[aid].request_id)
        position,ready=state.position,state.ready_s
        path=[]
        for _ in range(3):
            check()
            choices=[]
            for t in self._near(sid,position):
                if t.request_id in forbidden or not self._allowed(t,state.profile_id):
                    continue
                adjusted=ceil(ready/300)*300 if t.source_kind=="FORECAST" else ready
                leg=self._leg(t,adjusted,position,state.admission_end_s)
                if leg is not None:
                    choices.append((leg[1],leg[4],t.request_id,t,leg))
            if not choices:
                break
            _,_,_,task,leg=min(choices,key=lambda x:x[:3])
            # After the first task preserve the SAME sparse successor domain.
            if path and task not in self._following(sid,self.tasks[sid][path[-1][0]],state.profile_id):
                eligible=set(t.request_id for t in self._following(
                    sid,self.tasks[sid][path[-1][0]],state.profile_id))
                choices=[c for c in choices if c[2] in eligible]
                if not choices:
                    break
                _,_,_,task,leg=min(choices,key=lambda x:x[:3])
            path.append(leg); forbidden.add(task.request_id)
            position,ready=task.dropoff,leg[3]
        return (self._remember_path(sid,aid,tuple(path)),) if path else ()

    def nearby_resources(self,sid,rid,maximum,check,*,eligible_vehicles=None):
        """Ordinary proximity grouping, including vehicles unable to serve the seed."""
        check()
        cache_key=sid,rid,maximum,None if eligible_vehicles is None else tuple(sorted(eligible_vehicles))
        if self.efficient and cache_key in self.neighbor_cache:
            return self.neighbor_cache[cache_key]
        point=_xy(self.tasks[sid][rid].pickup)
        ids=self.root_tree.query_ball_point(point,2000) if self.root_tree else ()
        ranked=sorted((float(np.linalg.norm(self.root_points[i]-point)),
            self.actions[self.aids[i]].vehicle_id,self.aids[i]) for i in ids)
        result=[]
        for _,vid,_ in ranked:
            if eligible_vehicles is not None and vid not in eligible_vehicles:
                continue
            if vid not in result:
                result.append(vid)
            if len(result)>=maximum:
                break
        result=tuple(result)
        if self.efficient:
            self.neighbor_cache[cache_key]=result
            while len(self.neighbor_cache)>128: self.neighbor_cache.popitem(last=False)
        return result

    def _checks(self,check):
        check()
        if not self.efficient or check is not self.check:
            self.check()

    def possible(self,sid,aid,check=lambda:None):
        """Forbidden-independent SUPerset of this root's depth-three task domain."""
        key=sid,aid
        if key in self.possible_cache:
            self.possible_cache.move_to_end(key); return self.possible_cache[key]
        state=self.roots[aid]
        frontier=set()
        for t in self._near(sid,state.position):
            if not self._allowed(t,state.profile_id): continue
            ready=ceil(state.ready_s/300)*300 if t.source_kind=="FORECAST" else state.ready_s
            if self._leg(t,ready,state.position,state.admission_end_s) is not None:
                frontier.add(t.request_id)
        found=set(frontier)
        for _ in range(2):
            added=set()
            for i,rid in enumerate(sorted(frontier)):
                if i%32==0: check()
                added.update(t.request_id for t in self._following(sid,self.tasks[sid][rid],state.profile_id))
            frontier=added-found; found.update(added)
        result=frozenset(found)
        if len(result)<=20000:
            self.possible_cache[key]=result; self.possible_refs+=len(result)
            while len(self.possible_cache)>128 or self.possible_refs>20000:
                self.possible_refs-=len(self.possible_cache.popitem(last=False)[1])
        return result

    def local_forbidden(self,sid,aid,forbidden,check):
        return frozenset(forbidden)&self.possible(sid,aid,check) if self.efficient else frozenset(forbidden)

    def _distance(self,position,pickup):
        key=position,pickup
        if key in self.distances:
            self.distances.move_to_end(key); return self.distances[key]
        for point in key:
            if point not in self.xy: self.xy[point]=np.asarray(_xy(point))
        value=float(np.linalg.norm(self.xy[position]-self.xy[pickup]))
        self.distances[key]=value
        while len(self.distances)>20000: self.distances.popitem(last=False)
        return value

    def _allowed(self,task,profile):
        return profile in task.compatible_profiles and (profile=="HV" or task.passenger_accepts_av)

    def _leg(self,task,ready,position,end,*,horizon=None):
        dispatch=(ceil(max(ready,self.now+30,task.release_s)/30)*30
                  if task.source_kind=="ACTUAL_PENDING" else ceil(task.release_s/300)*300)
        horizon=self.horizon if horizon is None else horizon
        if not max(ready,task.release_s)<=dispatch<min(task.deadline_s,end,horizon):
            return None
        distance=(self._distance(position,task.pickup) if self.efficient else
            float(np.linalg.norm(np.asarray(_xy(position))-np.asarray(_xy(task.pickup)))))
        if distance>2000:
            return None
        pace=float(self.pace.pace_by_slot.get(int(dispatch//1800),self.pace.global_pace))
        if not isfinite(pace) or pace<=0:
            raise ValueError("frozen historical planning pace must be positive")
        eta=distance*pace
        if not 0<=eta<=600:
            return None
        return (task.request_id,float(dispatch),eta,
                float(dispatch+eta+task.service_time_s),distance)

    def _near(self,sid,point):
        tree,tasks,points=self.trees[sid]
        ids=tree.query_ball_point(_xy(point),2000) if tree is not None else ()
        found=[tasks[i] for i in ids]
        if self.topology is not None:
            found.extend(t for t in self.topology.near(sid,point) if t.request_id in self.tasks[sid])
        return found

    def _following(self,sid,node,profile):
        key=sid,node.request_id,profile
        if key in self.successors:
            return self.successors[key]
        # Candidate topology uses a zero-pickup lower bound; actual timing is
        # checked again in every label. It does not promise all city successors.
        earliest=(self.now+30 if node.source_kind=="ACTUAL_PENDING" else ceil(node.release_s/300)*300)
        ready=earliest+node.service_time_s
        candidates=[]
        if self.topology is not None and node.source_kind=="FORECAST":
            def build():
                rows=[]
                for i,t in enumerate(self.topology.near(sid,node.dropoff)):
                    if i%32==0:
                        self.check()
                    if t.request_id==node.request_id:
                        continue
                    leg=self._leg(t,ready,node.dropoff,float("inf"),horizon=float("inf"))
                    if leg is not None:
                        rows.append((leg[4],leg[1],t.request_id,t))
                return sorted(rows,key=lambda x:x[:3])
            candidates.extend(r for r in self.topology.candidates(sid,node,build)
                if r[2] in self.tasks[sid] and r[1]<self.horizon)
            tree,pending,points=self.trees[sid]
            ids=tree.query_ball_point(_xy(node.dropoff),2000) if tree is not None else ()
            nearby=(pending[i] for i in ids)
        else:
            nearby=self._near(sid,node.dropoff)
        for t in nearby:
            if t.request_id==node.request_id:
                continue
            leg=self._leg(t,ready,node.dropoff,float("inf"))
            if leg is not None:
                candidates.append((leg[4],leg[1],t.request_id,t))
        candidates.sort(key=lambda x:x[:3])
        c=[r[3] for r in candidates if self._allowed(r[3],"C")][:3]
        h=[r[3] for r in candidates if self._allowed(r[3],"HV")][:3]
        combined={t.request_id:t for t in (*h,*c)}
        self.successors[key]=tuple(c if profile=="C" else combined.values())
        return self.successors[key]

    def _target_reach(self,sid,rid,profile,check):
        """Exact sparse-domain reverse reachability; forbidden/time only relax it."""
        key=sid,rid,profile
        if key in self.reachable:
            self.reachable.move_to_end(key); return self.reachable[key]
        reached={rid}; layers=[frozenset(reached)]
        frontier={rid}
        for _ in range(2):
            added=set()
            for target in sorted(frontier):
                check()
                pkey=sid,target,profile
                if pkey in self.predecessors:
                    previous=self.predecessors[pkey]; self.predecessors.move_to_end(pkey)
                else:
                    pickup=self.tasks[sid][target].pickup
                    candidates=list(self.topology.near(sid,pickup,"dropoff"))
                    tree,pending=self.pending_dropoffs
                    if tree is not None:
                        candidates.extend(pending[i] for i in tree.query_ball_point(_xy(pickup),2000))
                    previous=[]
                    for i,t in enumerate(candidates):
                        if i%32==0:
                            self._checks(check)
                        if (t.request_id in self.tasks[sid] and t.request_id!=target
                                and self._allowed(t,profile)
                                and any(n.request_id==target for n in self._following(sid,t,profile))):
                            previous.append(t.request_id)
                    previous=tuple(previous)
                    if len(previous)<=20000:
                        self.predecessors[pkey]=previous; self.predecessor_refs+=len(previous)
                        while len(self.predecessors)>128 or self.predecessor_refs>20000:
                            self.predecessor_refs-=len(self.predecessors.popitem(last=False)[1])
                added.update(previous)
            frontier=added-reached; reached.update(added); layers.append(frozenset(reached))
        result=tuple(layers); size=sum(map(len,result))
        if size<=20000:
            self.reachable[key]=result; self.reachable_refs+=size
            while len(self.reachable)>128 or self.reachable_refs>20000:
                self.reachable_refs-=sum(map(len,self.reachable.popitem(last=False)[1]))
        return result

    def repair(self,sid,aid,request_ids,forbidden,check):
        """Re-anchor retained identities, NEVER retain old travel times or ownership."""
        state=self.roots[aid]; position,ready=state.position,state.ready_s
        used=set(forbidden); used.add(self.actions[aid].request_id)
        path=[]; previous=None
        for rid in request_ids:
            check()
            t=self.tasks[sid].get(rid)
            if t is None or rid in used or not self._allowed(t,state.profile_id):
                continue
            if previous is not None and not any(n.request_id==rid for n in self._following(sid,previous,state.profile_id)):
                continue
            adjusted=ceil(ready/300)*300 if t.source_kind=="FORECAST" else ready
            leg=self._leg(t,adjusted,position,state.admission_end_s)
            if leg is None:
                continue
            path.append(leg); used.add(rid); previous=t; position,ready=t.dropoff,leg[3]
            if len(path)==3:
                break
        return self._remember_path(sid,aid,tuple(path)) if path else None

    def supports(self,sid,rid,check=lambda:None):
        check()
        key=sid,rid
        if self.efficient and key in self.support_cache:
            self.support_cache.move_to_end(key); return self.support_cache[key]
        task=self.tasks[sid][rid]
        candidates=self.root_tree.query_ball_point(_xy(task.pickup),2000) if self.root_tree else ()
        result=[]
        for i in candidates:
            aid=self.aids[i]; state=self.roots[aid]
            if (self.actions[aid].request_id!=rid and self._allowed(task,state.profile_id)
                    and self._leg(task,state.ready_s,state.position,state.admission_end_s) is not None):
                result.append(aid)
        result=tuple(sorted(result,key=lambda a:(self.actions[a].vehicle_id,a)))
        if self.efficient:
            self.support_cache[key]=result
            while len(self.support_cache)>128: self.support_cache.popitem(last=False)
        return result

    def seeds(self,owners,check=lambda:None,*,candidates=None):
        # Known waiting requests first, then history; no AV utilization bonus.
        profile={a.vehicle_id:self.roots[aid].profile_id for aid,a in self.actions.items()}
        supplied=None if candidates is None else set(candidates)
        for sid in sorted(self.tasks):
            for task in sorted((t for t in self.tasks[sid].values()
                    if supplied is None or (sid,t.request_id) in supplied),key=lambda t:
                    (t.source_kind!="ACTUAL_PENDING",t.deadline_s,t.request_id)):
                check()
                owner=owners.get((sid,task.request_id))
                if owner is None:
                    yield sid,task.request_id
                elif (profile[owner]=="HV" and self._allowed(task,"C")
                      and any(profile[self.actions[aid].vehicle_id]=="C"
                              and self.actions[aid].vehicle_id!=owner
                              for aid in self.supports(sid,task.request_id,check))):
                    yield sid,task.request_id

    def query(self,sid,aid,*,required=None,forbidden=frozenset(),check=lambda:None):
        started=perf_counter(); self.query_calls+=1
        try:
            return self._query(sid,aid,required=required,forbidden=forbidden,check=check)
        finally:
            self.query_wall_time_s+=perf_counter()-started

    def _query(self,sid,aid,*,required=None,forbidden=frozenset(),check=lambda:None):
        check()
        action,state=self.actions[aid],self.roots[aid]
        forbidden=self.local_forbidden(sid,aid,forbidden,check)|({action.request_id} if action.request_id is not None else set())
        if required in forbidden:
            return ()
        if self.topology is not None and required is not None and required not in self.tasks[sid]:
            return ()
        if self.topology is not None and required is not None and not self._allowed(self.tasks[sid][required],state.profile_id):
            return ()
        key=sid,aid,required,forbidden
        catalog_key=sid,aid,forbidden
        if self.efficient and catalog_key in self.catalogs:
            self.catalogs.move_to_end(catalog_key); self.batch_reuses+=1
            path=self.catalogs[catalog_key].get(required)
            return (self._remember_path(sid,aid,path),) if path else ()
        if key in self.cache:
            self.cache.move_to_end(key); return self.cache[key]
        memo={}
        reach=(self._target_reach(sid,required,state.profile_id,check)
               if self.topology is not None and required is not None and required in self.tasks[sid] else None)
        def visit(node,leg,depth,used,found):
            if reach is not None and not found and node.request_id not in reach[3-depth]:
                self.target_pruned_nodes+=1
                return None
            self.labels+=1
            if self.labels>self.maximum_labels:
                cls=JointLabelStop if self.topology is not None else JointBudgetStop
                raise cls("joint planning label budget exhausted")
            if self.labels%64==0:
                self._checks(check)
            memo_key=leg,depth,used,found
            if memo_key in memo:
                return memo[memo_key]
            best=(leg,) if found else None
            if depth<3 and leg[3]<min(state.admission_end_s,self.horizon):
                for target in self._following(sid,node,state.profile_id):
                    if target.request_id in used or target.request_id in forbidden:
                        continue
                    ready=leg[3]
                    if target.source_kind=="FORECAST":
                        ready=ceil((ready-(self.now//300)*300)/300)*300+(self.now//300)*300
                    next_leg=self._leg(target,ready,node.dropoff,state.admission_end_s)
                    if next_leg is None:
                        continue
                    tail=visit(target,next_leg,depth+1,used|{target.request_id},
                               found or target.request_id==required)
                    candidate=None if tail is None else (leg,*tail)
                    if candidate is not None and (best is None or self._rank(candidate)<self._rank(best)):
                        best=candidate
            memo[memo_key]=best
            return best
        best=None
        for node in self._near(sid,state.position):
            if (node.request_id in forbidden or not self._allowed(node,state.profile_id)):
                continue
            if reach is not None and node.request_id not in reach[2]:
                self.target_pruned_nodes+=1
                continue
            ready=state.ready_s
            if node.source_kind=="FORECAST":
                ready=ceil(ready/300)*300
            leg=self._leg(node,ready,state.position,state.admission_end_s)
            if leg is None:
                continue
            found=required is None or node.request_id==required
            candidate=visit(node,leg,1,frozenset((node.request_id,)),found)
            if candidate is not None and (best is None or self._rank(candidate)<self._rank(best)):
                best=candidate
        result=()
        if best is not None:
            result=(self._remember_path(sid,aid,best),)
        self.cache[key]=result
        while len(self.cache)>128:
            self.cache.popitem(last=False)
        return result

    def query_many(self,sid,aid,targets,forbidden=frozenset(),check=lambda:None):
        """One exact depth-three enumeration for all targets, bounded best-path map.

        The map stores only the best path per encountered request, NOT all paths.
        A label-limited map supplies legal columns but is never cached as complete.
        """
        if not self.efficient:
            return {r:self.query(sid,aid,required=r,forbidden=forbidden,check=check)
                    for r in (*sorted(targets),None)}
        started=perf_counter(); self.batch_queries+=1; self.many_label_exhausted=False
        action,state=self.actions[aid],self.roots[aid]
        try:
            check()
            forbidden=self.local_forbidden(sid,aid,forbidden,check)|({action.request_id} if action.request_id is not None else set())
            key=sid,aid,forbidden
            if key in self.catalogs:
                self.catalogs.move_to_end(key); self.batch_reuses+=1
                best=self.catalogs[key]
            else:
                best={}; ranks={}
                def visit(node,leg,path,used,distance):
                    self.labels+=1
                    if self.labels>self.maximum_labels:
                        raise JointLabelStop("joint batch label budget exhausted")
                    if self.labels%64==0: self._checks(check)
                    path=(*path,leg); ids=(*used,node.request_id); distance+=leg[4]
                    rank=-len(path),distance,ids
                    for target in (None,*ids):
                        if target not in ranks or rank<ranks[target]:
                            best[target]=path; ranks[target]=rank
                    if len(path)>=3 or leg[3]>=min(state.admission_end_s,self.horizon): return
                    for target in self._following(sid,node,state.profile_id):
                        if target.request_id in ids or target.request_id in forbidden: continue
                        ready=leg[3]
                        if target.source_kind=="FORECAST":
                            ready=ceil((ready-(self.now//300)*300)/300)*300+(self.now//300)*300
                        following=self._leg(target,ready,node.dropoff,state.admission_end_s)
                        if following is not None: visit(target,following,path,ids,distance)
                try:
                    for node in self._near(sid,state.position):
                        if node.request_id in forbidden or not self._allowed(node,state.profile_id): continue
                        ready=ceil(state.ready_s/300)*300 if node.source_kind=="FORECAST" else state.ready_s
                        leg=self._leg(node,ready,state.position,state.admission_end_s)
                        if leg is not None: visit(node,leg,(),(),0.)
                except JointLabelStop:
                    self.many_label_exhausted=True
                if not self.many_label_exhausted and len(best)<=20000:
                    self.catalogs[key]=best; self.catalog_refs+=len(best)
                    while len(self.catalogs)>128 or self.catalog_refs>20000:
                        self.catalog_refs-=len(self.catalogs.popitem(last=False)[1])
            answer={}
            for target in (*sorted(set(targets)),None):
                path=best.get(target)
                answer[target]=(self._remember_path(sid,aid,path),) if path else ()
            return answer
        finally:
            self.query_wall_time_s+=perf_counter()-started

    @staticmethod
    def _rank(path):
        return (-len(path),sum(l[4] for l in path),tuple(l[0] for l in path))
