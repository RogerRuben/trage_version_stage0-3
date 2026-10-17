"""One prediction snapshot's lazy geometry/topology, not a dense OD matrix."""
from collections import Counter,OrderedDict
from dataclasses import asdict,is_dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .scheme_a_city_graph import ChainTask,_xy


class ForecastTopology:
    """Only one coarse snapshot; at most 20,000 cached candidate references."""
    def __init__(self,maximum_references=20000):
        self.maximum_references=min(20000,maximum_references)
        self.signature=None; self.index={}; self.task_map={}; self.links=OrderedDict(); self.references=0
        self.counts=Counter()
        self.prepared_directory=None; self.prepared_fingerprint=None
        self.prepared=None

    def snapshot_digest(self):
        def canonical(value):
            if is_dataclass(value): return asdict(value)
            if isinstance(value,(set,frozenset)): return sorted(value)
            raise TypeError(f"unsupported snapshot signature value: {type(value).__name__}")
        payload=json.dumps((self.signature,self.prepared_fingerprint),sort_keys=True,
            separators=(",",":"),default=canonical,allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _load_prepared(self,epoch):
        self.prepared=None
        if self.prepared_directory is None: return
        path=Path(self.prepared_directory)/f"slot={epoch}.parquet"
        if not path.is_file(): self.counts["prepared_catalog_absent"]+=1; return
        import pyarrow.parquet as pq
        table=pq.read_table(path)
        meta=table.schema.metadata or {}
        if meta.get(b"sparse_snapshot_signature",b"").decode()!=self.snapshot_digest():
            self.counts["prepared_catalog_scope_mismatch"]+=1; return
        rows=tuple(zip(table["scenario_id"].to_pylist(),table["source_request_id"].to_pylist(),
            table["target_request_id"].to_pylist()))
        digest=hashlib.sha256(repr(rows).encode()).hexdigest()
        if digest!=meta.get(b"candidate_rows_sha256",b"").decode() or len(rows)>self.maximum_references:
            self.counts["prepared_catalog_invalid"]+=1; return
        result={}
        for sid,source,target in rows:
            if sid not in self.task_map or source not in self.task_map[sid] or target not in self.task_map[sid]:
                self.counts["prepared_catalog_invalid"]+=1; return
            result.setdefault((sid,source),[]).append(target)
        self.prepared={k:tuple(v) for k,v in result.items()}
        self.counts["prepared_catalog_loads"]+=1

    def prepared_successors(self,sid,node):
        if self.prepared is None: return None
        self.counts["prepared_successor_reuses"]+=1
        return tuple(self.task_map[sid][rid] for rid in self.prepared.get((sid,node.request_id),()))

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
        self._load_prepared(epoch)

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
