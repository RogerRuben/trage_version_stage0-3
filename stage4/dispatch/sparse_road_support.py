"""Small reusable road evidence, NOT a complete road-connectivity matrix.

Only assessed positive historical-to-historical scalar proofs persist. Missing
and deferred answers never become physical impossibility certificates. Search
reads in-memory adjacency only; SQLite is touched during preparation, not DFS.
"""
from collections import Counter,OrderedDict,defaultdict
import json
from math import isfinite
from pathlib import Path
import sqlite3


MODE="SHARED_SPARSE_ROAD_SUPPORT_V1"


def task_key(task):
    # A sampled scenario ID/release/P50 is not a different physical road body.
    return (str(task.source_date),str(task.source_order_id),tuple(task.pickup),tuple(task.dropoff))


def _tuple(value):
    return tuple(_tuple(v) for v in value) if isinstance(value,list) else value


class SparseRoadIndex:
    def __init__(self,context_key,namespace,*,path=None):
        if not namespace: raise ValueError("sparse road evidence needs a physical-input namespace")
        self.context_key=context_key; self.namespace=str(namespace)
        self.records=OrderedDict(); self.outgoing=defaultdict(set)
        self.epoch=None; self.bytes=0
        self.maximum_entries=8192; self.maximum_bytes=8*2**20
        self.counts=Counter(); self.db=None
        if path is not None:
            path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
            self.db=sqlite3.connect(path,timeout=1.)
            self.db.execute("PRAGMA cache_size=-512")
            self.db.execute("PRAGMA journal_mode=DELETE")
            self.db.execute("PRAGMA max_page_count=8192")  # default 4-KiB pages: <=32 MiB
            self.db.execute("CREATE TABLE IF NOT EXISTS road_support (namespace TEXT, source TEXT, target TEXT, value TEXT, PRIMARY KEY(namespace,source,target))")
            self.db.commit()

    def source_key(self,state):
        return state.profile_id,tuple(state.position),self.context_key(state.context)

    def begin(self,now):
        self.epoch=now
        # Native direction may change on a repeated decision at the same tick.
        # Its compact view is decision-local; the connector's exact-prefix
        # certificate remains the authority for any subsequent reuse.
        for (source,target),(value,size) in tuple(self.records.items()):
            if source[2][0]=="NATIVE_VEHICLE":
                del self.records[source,target]; self.bytes-=size
                self.outgoing[source].discard(target)
                if not self.outgoing[source]: del self.outgoing[source]

    def _put(self,source,target,value):
        key=source,target
        size=2*len(repr((key,value)))+256
        if size>self.maximum_bytes: return
        old=self.records.pop(key,None)
        if old is not None:
            self.bytes-=old[1]; self.outgoing[source].discard(target)
        while self.records and (len(self.records)>=self.maximum_entries or self.bytes+size>self.maximum_bytes):
            (s,t),(_,n)=self.records.popitem(last=False)
            self.bytes-=n; self.outgoing[s].discard(t)
            if not self.outgoing[s]: del self.outgoing[s]
            self.counts["memory_evictions"]+=1
        self.records[key]=value,size; self.bytes+=size
        if value["status"]=="SUPPORTED": self.outgoing[source].add(target)

    def remember(self,state,task,result):
        source,target=self.source_key(state),task_key(task)
        reasons=tuple(result.get("reason_codes") or ())
        if result.get("supported"):
            raw=result.get("raw_time_s"); distance=result.get("empty_distance_m")
            if raw is None or distance is None or not all(isfinite(float(v)) and float(v)>=0 for v in (raw,distance)):
                raise ValueError("sparse support requires assessed static raw road time/distance")
            value=dict(status="SUPPORTED",raw_time_s=float(raw),empty_distance_m=float(distance),
                connection_evidence_state=result.get("connection_evidence_state"),reason_codes=())
        elif result.get("search_deferred") or any(any(word in r for word in (
                "NOT_EVALUATED","DEFERRED","EXCEPTION","UNAVAILABLE","MISSING","INVALID")) for r in reasons):
            self.counts["unassessed_results"]+=1
            return
        else:
            value=dict(status="REJECTED",reason_codes=reasons)
        self._put(source,target,value)
        # Native/current Test31 evidence stays process-local. No forecast
        # synthetic identity or real-time beta is written to the road store.
        context=source[2]
        if (self.db is not None and value["status"]=="SUPPORTED" and context[0]=="CUSTOMER"
                and str(context[1])<="20161024" and target[0]<="20161024"):
            try:
                self.db.execute("DELETE FROM road_support WHERE rowid IN (SELECT rowid FROM road_support ORDER BY rowid LIMIT max(0,(SELECT count(*) FROM road_support)-8191))")
                self.db.execute("INSERT OR REPLACE INTO road_support VALUES (?,?,?,?)",
                    (self.namespace,json.dumps(source),json.dumps(target),json.dumps(value)))
                self.db.commit(); self.counts["persisted_positive_records"]+=1
            except sqlite3.Error:
                self.db.rollback(); self.counts["optional_store_write_errors"]+=1

    def hydrate(self,state):
        """Preparation-only, bounded historical-source read; never called by search."""
        source=self.source_key(state); context=source[2]
        if self.db is None or context[0]!="CUSTOMER" or str(context[1])>"20161024": return
        for target,value in self.db.execute("SELECT target,value FROM road_support WHERE namespace=? AND source=? LIMIT 128",
                (self.namespace,json.dumps(source))):
            self._put(source,_tuple(json.loads(target)),json.loads(value))
            self.counts["persistent_positive_reuses"]+=1

    def targets(self,state):
        return tuple(self.outgoing.get(self.source_key(state),()))

    def lookup(self,state,task,beta):
        key=self.source_key(state),task_key(task)
        stored=self.records.get(key)
        if stored is None:
            self.counts["fast_unassessed_lookups"]+=1
            return dict(supported=False,reason_codes=[state.profile_id+"_CONNECTION_NOT_EVALUATED_SPARSE_INDEX"])
        self.records.move_to_end(key); value=stored[0]
        self.counts["indexed_lookups"]+=1
        if value["status"]!="SUPPORTED":
            return dict(supported=False,reason_codes=value["reason_codes"])
        if not isfinite(float(beta)) or float(beta)<=0: raise ValueError("road proof needs current departure beta")
        return dict(supported=True,raw_time_s=value["raw_time_s"],
            travel_time_s=value["raw_time_s"]*float(beta),empty_distance_m=value["empty_distance_m"],
            connection_evidence_state=value["connection_evidence_state"],reason_codes=[])

    def assessed(self,state,task):
        return (self.source_key(state),task_key(task)) in self.records

    def diagnostics(self):
        return dict(mode=MODE,records=len(self.records),positive_adjacency_entries=sum(map(len,self.outgoing.values())),
            memory_mib=self.bytes/2**20,memory_limit_mib=8,maximum_entries=self.maximum_entries,
            unassessed_is_not_incapable=True,persisted_scope="POSITIVE_HISTORICAL_TO_HISTORICAL_STATIC_SCALARS_ONLY",
            search_disk_reads=0,counts=dict(self.counts))

    def close(self):
        if self.db is not None: self.db.close(); self.db=None
