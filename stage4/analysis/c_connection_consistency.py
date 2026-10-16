"""Prepare opt-in code and check eight historical links; no simulation launcher."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import psutil
from scipy.spatial import cKDTree

from . import bottleneck_operational_v4 as previous
from .capability_chain_instances import atomic_json, sha


CONFIG=Path("stage4/config/c_connection_consistency_v1.json")
DOC=Path("stage4/docs/capability_chain_planning/c_connection_consistency_v1")


def load(root):
    root=Path(root)
    declared=json.loads((root/CONFIG).read_text(encoding="utf-8"))
    if (declared["base_config"]!=previous.CONFIG.as_posix()
            or sha(root/previous.CONFIG)!=declared["base_config_sha256"]):
        raise ValueError("C connection work requires the unchanged completed V4 config")
    if any(declared.get(k) is not False for k in ("full_day_authorized","forced_C_priority",
            "idle_utilization_reward","objective_changed","supply_calendar_changed",
            "profile_thresholds_changed","dense_matrix","gpu_used")):
        raise ValueError("outside the scoped connection-only implementation")
    cfg=previous.load(root)
    cfg.update({k:declared[k] for k in ("version","group","joint_c_connection_mode",
        "c_extra_backend_requests_per_epoch","c_extra_backend_requests_per_snapshot","c_optional_connection_wall_s")})
    cfg["disk_route_cache"]="stage4/output/capability_chain_planning/c_connection_consistency_v1/routing/cache.sqlite3"
    return cfg


def prepare(root):
    cfg=load(root)
    result=dict(status="C_CONNECTION_IMPLEMENTATION_PREPARED_NOT_SIMULATED",
        config_sha256=sha(Path(root)/CONFIG),base_config_sha256=sha(Path(root)/previous.CONFIG),
        mode=cfg["joint_c_connection_mode"],group=cfg["group"],
        extra_backend_requests_per_epoch=cfg["c_extra_backend_requests_per_epoch"],
        extra_backend_requests_per_300s=cfg["c_extra_backend_requests_per_snapshot"],
        optional_wall_budget_s=cfg["c_optional_connection_wall_s"],
        existing_certificate_cache_mib=32,per_search_C_lookup_entries=4096,
        old_default_and_results_unchanged=True,full_day_executed=False,
        supply_calendar_changed=False,objective_changed=False,profile_thresholds_changed=False,
        complete_C_routing_claimed=False,C_idle_improvement_established=False)
    atomic_json(Path(root)/DOC/"preparation.json",result)
    return result


def _pairs(frame):
    """Stable, outcome-independent historical pairs in the existing 2-km domain."""
    from stage4.dispatch.scheme_a_city_graph import _xy
    frame=frame.loc[frame.common_eligible & frame.compatible_C].copy()
    frame["date"]=frame.date.astype(str)
    frame["priority"]=[hashlib.sha256(f"C_LINK_CHECK_V1|{d}|{o}".encode()).hexdigest()
                       for d,o in frame[["date","order_id"]].itertuples(index=False,name=None)]
    selected=[]
    for date, group in frame.groupby("date",sort=True):
        ordered=group.sort_values("priority",kind="stable").reset_index(drop=True)
        points=np.asarray([_xy((r.start_lon_wgs84,r.start_lat_wgs84)) for r in ordered.itertuples()])
        tree=cKDTree(points)
        for source in ordered.itertuples():
            ready=float(source.release_second)+float(source.predicted_route_time_p50_s)
            candidates=tree.query_ball_point(_xy((source.end_lon_wgs84,source.end_lat_wgs84)),2000.)
            choices=[]
            for i in candidates:
                target=ordered.iloc[i]
                if (str(target.order_id)==str(source.order_id)
                        or not ready<=float(target.release_second)<min(ready+1800,86100)):
                    continue
                distance=float(np.linalg.norm(points[i]-_xy((source.end_lon_wgs84,source.end_lat_wgs84))))
                if distance>1.:
                    choices.append((distance,str(target.priority),int(i)))
            if choices:
                selected.append((source._asdict(),ordered.iloc[min(choices)[2]].to_dict()))
                if len(selected)==8:
                    return selected
    raise ValueError("fewer than eight supported-domain historical pairs")


def probe(root):
    """Eight isolated route/interface checks, not an experiment or day extrapolation."""
    from stage4.fleetpy_adapter.valhalla_time_adapter import ValhallaPickupTimeAdapter
    from stage4.dispatch.controlled_routes import ControlledEmptyRouter
    from stage4.dispatch.city_pickup_support import CityPickupSupportValidator
    from stage4.dispatch.scheme_a_city_connections import SchemeACityConnectors
    from stage4.dispatch.c_connection_support import SharedCConnectionSupport,ConnectionBudget
    from stage4.dispatch.scheme_a_city_graph import ChainTask,RouteState
    from stage4.dispatch.scheme_a_city_master import CurrentAction
    from stage4.dispatch.bottleneck_scene import PredictionSceneOracle
    root=Path(root); cfg=load(root); started=perf_counter(); process=psutil.Process()
    if psutil.virtual_memory().available<1536*2**20:
        raise MemoryError("release RAM first; the original 1536-MiB startup floor is retained")
    def guard():
        if perf_counter()-started>180 or process.memory_info().rss>1536*2**20:
            raise RuntimeError("bounded connection check exceeded time/RAM allowance")
    columns=["order_id","date","release_second","start_lon_wgs84","start_lat_wgs84",
             "end_lon_wgs84","end_lat_wgs84","predicted_route_time_p50_s","compatible_C","common_eligible"]
    frame=pd.read_parquet(root/cfg["source_history_templates"],columns=columns)
    pairs=_pairs(frame)
    identities={(str(r["date"]),str(r["order_id"])) for pair in pairs for r in pair}
    templates=frame.loc[[ (str(r.date),str(r.order_id)) in identities for r in frame.itertuples()]].copy()
    del frame
    stamp=pd.Timestamp("2016-10-31T00:00:00+08:00")
    eta=ValhallaPickupTimeAdapter(root)
    router=ControlledEmptyRouter(root,eta)
    control=NS(eta_adapter=eta,sim_time=0.,request_by_rid={},assignment_rows=[],completed_rids=set(),
        runtime_by_vid={},config={},_timestamp=lambda t:stamp+pd.Timedelta(seconds=t))
    validator=CityPickupSupportValidator(router,geometry_timestamp=stamp,routing_budget_s=10.)
    connector=SchemeACityConnectors(control,router,validator,templates,geometry_timestamp=stamp)
    support=SharedCConnectionSupport(connector,None,cfg)
    results=[]
    def task(row,index):
        release=float(row["release_second"])
        return ChainTask(-index,release,release+300,float(row["predicted_route_time_p50_s"]),
            (float(row["start_lon_wgs84"]),float(row["start_lat_wgs84"])),
            (float(row["end_lon_wgs84"]),float(row["end_lat_wgs84"])),{"C","HV"},
            source_date=str(row["date"]),source_order_id=str(row["order_id"]))
    for i,(before,after) in enumerate(pairs,1):
        guard(); source,target=task(before,2*i),task(after,2*i+1)
        departure=max(source.release_s+source.service_time_s,target.release_s)
        state=RouteState(source.dropoff,departure,86400.,"C",dict(kind="CUSTOMER",task=source))
        # Legacy certificate and new certificate have separate keys. Same
        # source/target/time/beta/rules; only bounded recovery differs.
        connector.c_connection_budget=None
        entered=perf_counter()
        old=connector(state,target,departure,"C",enforce_pickup_deadline=False)
        old_time=perf_counter()-entered
        support.budget=ConnectionBudget(); connector.c_connection_budget=support.budget
        support.begin(departure,perf_counter()+10)
        entered=perf_counter()
        new=connector(state,target,departure,"C",query_scope="FUTURE",enforce_pickup_deadline=False)
        new_time=perf_counter()-entered
        backend_before=router.route_call_count
        entered=perf_counter()
        for _ in range(1000):
            cached=support.lookup(state,target,departure)
        lookup_time=perf_counter()-entered
        if router.route_call_count!=backend_before:
            raise AssertionError("search-style cache lookups unexpectedly routed")
        action=CurrentAction("C_PROBE",1,"WAIT",None)
        oracle=PredictionSceneOracle([action],{action.action_id:state},{"CHECK":(target,)},(),
            NS(global_pace=.01,pace_by_slot={}),target.release_s-30.,efficient=True,c_support=support)
        before_lookups=support.counts["planning_cache_lookups"]
        entered=perf_counter()
        for _ in range(10000):
            oracle._supported_leg(target,departure,state.position,state.admission_end_s,"C",state.context)
        memo_time=perf_counter()-entered
        memo_lookups=support.counts["planning_cache_lookups"]-before_lookups
        if memo_lookups>1 or router.route_call_count!=backend_before:
            raise AssertionError("per-search memo failed to suppress repeated certificate work")
        if old["supported"] and not new["supported"]:
            raise AssertionError("valid default C link was lost in the isolated check")
        if cached["supported"]!=new["supported"] and not any("NOT_EVALUATED" in r for r in cached["reason_codes"]):
            raise AssertionError("cached static support disagrees")
        results.append(dict(pair_number=i,source_date=str(before["date"]),
            default_supported=old["supported"],new_supported=new["supported"],
            default_pickup_eta_s=old["travel_time_s"],new_pickup_eta_s=new["travel_time_s"],
            new_within_pickup_cap=bool(new["supported"] and new["travel_time_s"]<=600.),
            default_reasons=old["reason_codes"],new_reasons=new["reason_codes"],
            recovered=not old["supported"] and new["supported"],
            default_wall_s=old_time,new_wall_s=new_time,cache_1000_lookups_wall_s=lookup_time,
            repeated_leg_10000_wall_s=memo_time,repeated_leg_certificate_lookups=memo_lookups,
            cache_backend_queries=router.route_call_count-backend_before,
            optional_query_budget_slots=support.budget.counts["optional_query_slots_consumed"]))
    guard()
    result=dict(status="C_CONNECTION_EIGHT_LINK_CHECK_COMPLETE",pair_count=len(results),rows=results,
        source="STABLE_HASH_EARLIER_HISTORY_C_BODY_PAIRS_WITHIN_2KM",
        simulator_invoked=False,inference_invoked=False,future_Test31_orders_read=False,
        full_day_executed=False,C_idle_improvement_established=False,
        runtime_s=perf_counter()-started,peak_rss_mib=process.memory_info().peak_wset/2**20,
        config_sha256=sha(root/CONFIG),connector=connector.diagnostics(),router=router.diagnostics())
    atomic_json(root/DOC/"eight_link_check.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","probe"),nargs="?",default="prepare")
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if args.action=="probe" and not args.execute:
        parser.error("the bounded router check requires --execute; no simulation command exists here")
    result=prepare(Path.cwd()) if args.action=="prepare" else probe(Path.cwd())
    print(json.dumps({k:result[k] for k in ("status","runtime_s","peak_rss_mib") if k in result}),flush=True)
