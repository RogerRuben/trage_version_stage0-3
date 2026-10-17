"""Prepare reusable sparse candidates/support storage, with NO replay launcher."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from . import economic_joint_dispatch as base
from .scheme_a_two_scale_check import _sha,_atomic_json


CONFIG=Path("stage4/config/economic_sparse_connections_v1.json")
OUT=Path("stage4/output/capability_chain_planning/economic_sparse_connections_v1")
DOC=Path("stage4/docs/capability_chain_planning/economic_joint_v1")


def physical_namespace(root,cfg):
    from stage4.dispatch.runtime_assets import routing_context
    from stage4.dispatch.controlled_routes import S2A,S2B,CONTROL_REL
    from stage4.dispatch.city_pickup_support import OVERLAY_REL
    from stage4.analysis.flexibility_prepare import S3
    paths=[Path(cfg["frozen_profile_config"]),CONTROL_REL,OVERLAY_REL,
        S2A/"stage3_full_network_edges.parquet",S2B/"stage3_edge_complex_boundary_index.parquet",
        S2B/"stage3_route_movement_lookup.parquet",Path(cfg["source_history_manifest"]),
        Path("stage4/dispatch/scheme_a_city_connections.py"),Path("stage4/dispatch/controlled_routes.py"),
        Path("stage3/odd_tod/research_compatibility.py")]
    paths.extend(S3/f"cache/train/date={d}/identity.parquet" for d in cfg["forecast_train_dates"])
    inputs={p.as_posix():_sha(root/p) for p in paths}
    scope=dict(schema="SPARSE_STATIC_SCALAR_SUPPORT_V1",routing=routing_context(root),physical_inputs=inputs)
    return hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest()


def load(root,*,controller=None):
    root=Path(root).resolve(); declared=json.loads((root/CONFIG).read_text(encoding="utf-8"))
    if (declared["base_config"]!=base.CONFIG.as_posix() or declared["base_config_sha256"]!=_sha(root/base.CONFIG)
            or declared["economic_sparse_connections"]!="SHARED_SPARSE_ROAD_SUPPORT_V1"
            or any(declared.get(k) is not False for k in ("full_day_authorized","automatic_experiment_launch",
                "economic_parameters_changed","C_reward_added"))):
        raise ValueError("sparse preparation cannot change the frozen economic problem or launch experiments")
    cfg=base.load(root,controller=controller); cfg.update(declared)
    cfg["sparse_candidate_catalog"]=str(root/declared["sparse_candidate_catalog"])
    cfg["sparse_connection_store"]=str(root/declared["sparse_connection_store"])
    cfg["sparse_connection_namespace"]=physical_namespace(root,cfg)
    return cfg


def write_catalog(path,oracle,fingerprint,check=lambda:None):
    """Historical Top3(C) union Top3(HV); real pending tasks are added live."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows=[]; original_horizon=oracle.horizon
    # At a later 30-s epoch in the same 300-s snapshot, dispatches at the old
    # 1800-s boundary become admissible. The union of both grid cases preserves
    # TopK for the entire snapshot, not just its first decision.
    try:
        for horizon in (original_horizon,original_horizon+300):
            oracle.horizon=horizon; oracle.successors.clear()
            for sid,tasks in sorted(oracle.tasks.items()):
                for task in tasks.values():
                    check()
                    rows.extend((sid,task.request_id,target.request_id) for target in oracle._following(sid,task,"HV"))
    finally:
        oracle.horizon=original_horizon; oracle.successors.clear()
    rows=tuple(sorted(set(rows)))
    if len(rows)>20000: raise MemoryError("one prepared candidate catalog exceeds the existing 20,000-reference cap")
    oracle.topology.prepared_fingerprint=fingerprint
    metadata={b"sparse_snapshot_signature":oracle.topology.snapshot_digest().encode(),
        b"candidate_rows_sha256":hashlib.sha256(repr(rows).encode()).hexdigest().encode(),
        b"meaning":b"SPARSE_CANDIDATES_NOT_CERTIFIED_ROAD_CONNECTIONS"}
    table=pa.table({"scenario_id":[r[0] for r in rows],"source_request_id":[r[1] for r in rows],
        "target_request_id":[r[2] for r in rows]},schema=pa.schema([
            ("scenario_id",pa.string()),("source_request_id",pa.int64()),("target_request_id",pa.int64())],metadata=metadata))
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".parquet.tmp"); pq.write_table(table,temporary); temporary.replace(path)
    return len(rows)


def prepare(root):
    import psutil
    from stage4.analysis.scheme_a_city_experiment import load_history_templates
    from stage4.dispatch.scheme_a_city_graph import CoarseHistoricalLibrary,RouteState
    from stage4.dispatch.scheme_a_city_master import CurrentAction
    from stage4.dispatch.bottleneck_topology import ForecastTopology
    from stage4.dispatch.bottleneck_scene import PredictionSceneOracle
    from stage4.dispatch.flexibility_native import TrainDemandForecast
    from stage4.dispatch.operating_economics import OperatingEconomics
    from stage4.dispatch.c_connection_support import context_key
    from stage4.dispatch.sparse_road_support import SparseRoadIndex
    root=Path(root).resolve(); started=perf_counter(); cfg=load(root); process=psutil.Process()
    def check():
        if process.memory_info().rss>1536*2**20: raise MemoryError("sparse preparation retains the 1536-MiB cap")
        if perf_counter()-started>300: raise TimeoutError("bounded sparse static preparation exceeded 300 seconds")
    if psutil.virtual_memory().available<1536*2**20: raise MemoryError("release RAM before sparse preparation")
    templates,_=load_history_templates(root,cfg)
    library=CoarseHistoricalLibrary(templates,dict(cfg,measurement_end_s=cfg["coarse_reference_day_end_s"]))
    pace=TrainDemandForecast(templates,dict(cfg,forecast_horizon_s=1800,fast_forecast=True),cfg["coarse_reference_day_end_s"])
    money=OperatingEconomics.from_config(cfg); catalogs=[]; comparisons=0
    for slot in cfg["prepared_coarse_slots_s"]:
        check(); topology=ForecastTopology(); scenes=library.view(slot)
        action=CurrentAction("CATALOG_ONLY",0,"WAIT",None)
        q=PredictionSceneOracle([action],{action.action_id:RouteState((108.9,34.2),slot+30.,86434.,"HV")},
            scenes,(),pace,slot,topology=topology,forecast_snapshot=library.coarse_snapshot(slot),
            efficient=True,economics=money,check=check)
        path=Path(cfg["sparse_candidate_catalog"])/f"slot={slot}.parquet"
        count=write_catalog(path,q,money.fingerprint,check)
        for fine in (slot,slot+30):
            fine_scenes=library.view(fine)
            prepared_topology=ForecastTopology()
            prepared_topology.prepared_directory=cfg["sparse_candidate_catalog"]
            prepared_topology.prepared_fingerprint=money.fingerprint
            options=dict(efficient=True,economics=money,check=check,forecast_snapshot=library.coarse_snapshot(fine))
            verified=PredictionSceneOracle([action],q.roots,fine_scenes,(),pace,fine,topology=prepared_topology,**options)
            reference=PredictionSceneOracle([action],q.roots,fine_scenes,(),pace,fine,topology=ForecastTopology(),**options)
            if prepared_topology.prepared is None: raise AssertionError("prepared catalog did not bind the exact live snapshot")
            for sid,tasks in verified.tasks.items():
                for task in tasks.values():
                    check()
                    for profile in ("C","HV"):
                        if (tuple(t.request_id for t in verified._following(sid,task,profile))
                                !=tuple(t.request_id for t in reference._following(sid,task,profile))):
                            raise AssertionError("prepared candidate catalog changed the declared live TopK domain")
                        comparisons+=1
            del verified,reference,prepared_topology
        catalogs.append(dict(slot_s=slot,rows=count,source_tasks=sum(map(len,q.tasks.values())),
            path=path.relative_to(root).as_posix(),sha256=_sha(path)))
        del q,topology
    index=SparseRoadIndex(context_key,cfg["sparse_connection_namespace"],path=cfg["sparse_connection_store"])
    persisted=index.db.execute("SELECT count(*) FROM road_support WHERE namespace=?",(index.namespace,)).fetchone()[0]
    index.close(); check()
    result=dict(status="SPARSE_CANDIDATES_AND_ROAD_SUPPORT_STORE_PREPARED",catalogs=catalogs,
        physical_namespace=cfg["sparse_connection_namespace"],economic_fingerprint=money.fingerprint,
        assessed_persisted_historical_road_connections=persisted,
        live_candidate_equivalence_comparisons=comparisons,
        candidate_catalogs_are_not_road_certificates=True,new_road_queries=0,simulation_invoked=False,
        new_inference_or_training=False,economic_parameters_changed=False,C_reward_added=False,
        per_epoch_backend_query_cap=cfg["c_extra_backend_requests_per_epoch"],
        per_300s_backend_query_cap=cfg["c_extra_backend_requests_per_snapshot"],
        runtime_s=perf_counter()-started,peak_rss_mib=process.memory_info().peak_wset/2**20)
    _atomic_json(root/DOC/"sparse_connection_preparation.json",result)
    return result


class OfflineRoadBudget:
    """A single history-preparation batch, NOT an online epoch budget reset."""
    def __init__(self,deadline,maximum=96):
        if not 0<maximum<=96: raise ValueError("one offline batch has at most 96 query slots")
        self.deadline=deadline; self.maximum=maximum; self.calls=0; self.spent_epoch_s=0.
        self.counts={}
    def claim(self,purpose):
        if self.calls>=self.maximum or perf_counter()>=self.deadline: return False
        self.calls+=1; self.counts[purpose]=self.counts.get(purpose,0)+1; return True
    def record(self,seconds): self.spent_epoch_s+=seconds


def prepare_roads(root):
    """One bounded static historical batch; never a vehicle/demand simulation."""
    from types import SimpleNamespace as NS
    import pandas as pd
    import pyarrow.parquet as pq
    import psutil
    from math import ceil
    from stage4.analysis.scheme_a_city_experiment import load_history_templates,make_routing
    from stage4.dispatch.scheme_a_city_graph import CoarseHistoricalLibrary,RouteState
    from stage4.dispatch.controlled_routes import ControlledEmptyRouter
    from stage4.dispatch.city_pickup_support import CityPickupSupportValidator
    from stage4.dispatch.scheme_a_city_connections import SchemeACityConnectors
    from stage4.dispatch.c_connection_support import context_key
    from stage4.dispatch.sparse_road_support import SparseRoadIndex,task_key
    from .scheme_a_two_scale_check import _HardStop
    root=Path(root).resolve(); started=perf_counter(); cfg=load(root)
    if (cfg["offline_preparation_candidate_pairs"],cfg["offline_preparation_query_slots"],cfg["offline_preparation_wall_limit_s"])!=(32,96,180):
        raise ValueError("only one fixed, bounded history road-preparation batch is supported")
    directory=root/OUT/"road_preparation"
    receipt=directory/"summary.json"
    if receipt.is_file(): raise ValueError("existing road-preparation output retained; no implicit rerun")
    if psutil.virtual_memory().available<1536*2**20: raise MemoryError("release RAM before road preparation")
    if psutil.disk_usage(root.anchor).free<512*2**20: raise OSError("road preparation retains the 512-MiB reserve")
    process=psutil.Process(); routing=index=None
    summary=dict(status="RUNNING_STATIC_HISTORY_ROAD_PREPARATION",physical_namespace=cfg["sparse_connection_namespace"],
        simulation_invoked=False,online_query_budget_changed=False,maximum_query_slots=96,hard_limit_s=180)
    stop=_HardStop(directory,started,180,summary); stop.publish(summary); stop.start()
    def check():
        if perf_counter()-started>=180: raise TimeoutError("bounded historical road preparation expired")
        if process.memory_info().rss>1536*2**20: raise MemoryError("history road preparation exceeded 1536 MiB")
    try:
        templates,_=load_history_templates(root,cfg)
        library=CoarseHistoricalLibrary(templates,dict(cfg,measurement_end_s=cfg["coarse_reference_day_end_s"]))
        choices={}
        for slot in cfg["prepared_coarse_slots_s"]:
            check(); scenes=library.view(slot); maps={s:{t.request_id:t for t in tasks} for s,tasks in scenes.items()}
            table=pq.read_table(Path(cfg["sparse_candidate_catalog"])/f"slot={slot}.parquet")
            for sid,source,target in zip(*(table[k].to_pylist() for k in ("scenario_id","source_request_id","target_request_id"))):
                before,after=maps[sid][source],maps[sid][target]
                if not all("C" in t.compatible_profiles and t.passenger_accepts_av for t in (before,after)): continue
                physical=task_key(before),task_key(after)
                priority=hashlib.sha256(repr(("HISTORY_ROAD_BATCH_V1",physical)).encode()).hexdigest()
                choices.setdefault(physical,(priority,before,after))
        selected=sorted(choices.values(),key=lambda r:r[0])[:32]
        if not selected: raise ValueError("no historical C-capable candidate pair in the prepared catalog")
        identities={(t.source_date,t.source_order_id) for _,a,b in selected for t in (a,b)}
        templates=templates.loc[[(str(r.date),str(r.order_id)) in identities for r in templates.itertuples()]].copy()
        del choices,library,table,maps,scenes
        runtime_cfg=dict(cfg,disk_route_cache=(OUT/"road_preparation/routing/cache.sqlite3").as_posix())
        routing=make_routing(root,runtime_cfg); router=ControlledEmptyRouter(root,routing)
        stamp=pd.Timestamp("2016-10-31T00:00:00+08:00")
        control=NS(eta_adapter=routing,sim_time=29700.,request_by_rid={},assignment_rows=[],completed_rids=set(),
            runtime_by_vid={},config=dict(scheme_a_fast_forecast_hv=True),
            _timestamp=lambda t:stamp+pd.Timedelta(seconds=t))
        validator=CityPickupSupportValidator(router,geometry_timestamp=stamp,routing_budget_s=180.)
        connector=SchemeACityConnectors(control,router,validator,templates,geometry_timestamp=stamp)
        connector.bounded_all_profile_future=True
        budget=OfflineRoadBudget(started+180,96); connector.c_connection_budget=budget
        index=SparseRoadIndex(context_key,cfg["sparse_connection_namespace"],path=cfg["sparse_connection_store"])
        rows=[]
        for priority,before,after in selected:
            for profile in ("HV","C"):
                check(); departure=float(ceil(after.release_s/300)*300)
                state=RouteState(before.dropoff,departure,90000.,profile,dict(kind="CUSTOMER",task=before))
                index.hydrate(state)
                if index.assessed(state,after): result=index.lookup(state,after,1.)
                else:
                    result=connector(state,after,departure,profile,query_scope="FUTURE",enforce_pickup_deadline=False)
                    index.remember(state,after,result)
                rows.append(dict(pair_hash=priority,profile_id=profile,supported=bool(result["supported"]),
                    reason_codes=list(result.get("reason_codes") or ())))
                summary.update(completed_assessments=len(rows),query_slots_consumed=budget.calls,
                    runtime_s=perf_counter()-started,peak_rss_mib=process.memory_info().peak_wset/2**20)
                stop.publish(summary)
        count=index.db.execute("SELECT count(*) FROM road_support WHERE namespace=?",(index.namespace,)).fetchone()[0]
        summary.update(status="STATIC_HISTORICAL_ROAD_PREPARATION_COMPLETE",selected_pairs=len(selected),rows=rows,
            persisted_positive_records=count,supported_HV=sum(r["supported"] and r["profile_id"]=="HV" for r in rows),
            supported_C=sum(r["supported"] and r["profile_id"]=="C" for r in rows),query_slots_by_purpose=budget.counts,
            scalar_route_calls=router.route_call_count,edge_walk_calls=router.trace_call_count,
            HV_matrix_evaluations=connector.counts["HV_future_auto_OD_evaluations"],
            offline_batch_is_not_a_change_to_online_two_requests_per_epoch=True,
            Test31_future_demand_read=False,customer_routes_replanned=False,economic_parameters_changed=False,C_reward_added=False)
        stop.publish(summary)
        _atomic_json(root/DOC/"historical_road_preparation.json",summary)
        return summary
    except Exception as error:
        summary.update(status="STOPPED_STATIC_ROAD_PREPARATION",error=repr(error),runtime_s=perf_counter()-started)
        stop.publish(summary); raise
    finally:
        stop.close()
        if index is not None: index.close()
        if routing is not None: routing.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("describe","prepare","prepare-roads"),nargs="?",default="describe")
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if args.action=="prepare-roads" and not args.execute: parser.error("bounded road preparation requires --execute")
    result=prepare_roads(Path.cwd()) if args.action=="prepare-roads" else prepare(Path.cwd()) if args.action=="prepare" else load(Path.cwd())
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
