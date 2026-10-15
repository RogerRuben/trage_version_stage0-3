"""One prediction snapshot's lazy geometry/topology, not a dense OD matrix."""
from collections import Counter,OrderedDict

import numpy as np
from scipy.spatial import cKDTree

from .scheme_a_city_graph import ChainTask,_xy


class ForecastTopology:
    """Only one coarse snapshot; at most 20,000 cached candidate references."""
    def __init__(self,maximum_references=20000):
        self.maximum_references=min(20000,maximum_references)
        self.signature=None; self.index={}; self.task_map={}; self.links=OrderedDict(); self.references=0
        self.counts=Counter()

    def bind(self,snapshot,pace,now,check):
        epoch=int(now//300)*300
        frozen=tuple((sid,tuple(tasks)) for sid,tasks in sorted(snapshot.items()))
        signature=(epoch,float(pace.global_pace),tuple(sorted(pace.pace_by_slot.items())),frozen)
        if signature==self.signature:
            self.counts["snapshot_reuses"]+=1
            return
        index={}
        for sid,tasks in frozen:
            check()
            if (len({t.request_id for t in tasks})!=len(tasks)
                    or any(not isinstance(t,ChainTask) or t.source_kind!="FORECAST"
                        or t.request_id>=0 or not t.source_date or t.source_date>"20161024"
                        or t.release_s<=epoch for t in tasks)):
                raise ValueError("topology cache accepts only the immutable earlier-history snapshot")
            points={kind:np.asarray([_xy(getattr(t,kind)) for t in tasks],float).reshape(-1,2)
                    for kind in ("pickup","dropoff")}
            index[sid]=(tasks,{k:cKDTree(p) if len(p) else None for k,p in points.items()})
        self.signature=signature; self.index=index
        self.task_map={s:{t.request_id:t for t in tasks} for s,tasks in frozen}
        self.links.clear(); self.references=0
        self.counts["snapshot_rebuilds"]+=1

    def near(self,sid,point,kind="pickup"):
        tasks,trees=self.index[sid]; tree=trees[kind]
        return tuple(tasks[i] for i in tree.query_ball_point(_xy(point),2000)) if tree is not None else ()

    def candidates(self,sid,node,build):
        key=sid,node
        if key in self.links:
            self.links.move_to_end(key); self.counts["successor_reuses"]+=1
            return self.links[key]
        rows=tuple(build())
        self.counts["successor_builds"]+=1
        if len(rows)<=self.maximum_references:
            while self.links and (self.references+len(rows)>self.maximum_references or len(self.links)>=2048):
                self.references-=len(self.links.popitem(last=False)[1])
            self.links[key]=rows; self.references+=len(rows)
        return rows
