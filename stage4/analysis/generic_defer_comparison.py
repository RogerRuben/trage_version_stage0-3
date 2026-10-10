"""One new, wait-enabled time/type control on the two fixed local windows.

No target-day truth fits the generic reference. Existing map-state controls are
reused, not rerun. Physical first actions and execution remain identical; only
future post-customer geometry is replaced in the valuation problem.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa

from stage4.analysis.capability_chain_instances import atomic_json, atomic_parquet, sha
from stage4.analysis.capability_chain_rolling import (
    ConnectionProvider, HistoricalPredictionTimeProxy, InstanceRouter, MixedDateEvidence,
    compact_policy, replay_policy,
)
from stage4.dispatch.generic_defer import project_post_service_geometry

CONFIG=Path("stage4/config/generic_defer_comparison_v1.json")
OUT=Path("stage4/output/capability_chain_planning/generic_defer_v1")
DOC=Path("stage4/docs/capability_chain_planning/generic_defer_v1")


def fit_reference(source, inputs, cfg):
    forecast_ids={t["job_id"] for scene in inputs["forecast_cohorts"] for t in scene["tasks"]}
    site_ids={s["site_id"] for s in inputs["sites"]}
    links=pd.read_parquet(source/"sparse_connections.parquet")
    eligible=links.loc[links.origin_id.isin(site_ids)&links.target_job_id.isin(forecast_ids)&links.supported].copy()
    eligible=eligible.loc[eligible.compatible_profiles.map(lambda profiles:"HV" in profiles)
        &eligible.travel_time_s.le(cfg["reference_max_pickup_s"])]
    best=eligible.sort_values(["travel_time_s","empty_distance_m","origin_id"],kind="stable").drop_duplicates("target_job_id")
    if best.empty:
        raise ValueError("no earlier-history generic pickup reference; no target-day fallback")
    return dict(kind="EARLIER_HISTORY_TYPICAL_SERVICEABLE_PICKUP", history_dates=cfg["history_dates"],
        travel_time_s=float(best.travel_time_s.median()), empty_distance_m=float(best.empty_distance_m.median()),
        historical_customer_sample_count=len(best), historical_customer_population=len(forecast_ids),
        target_day_actual_orders_used=False, source_definition=cfg["reference_rule"],
        estimation_scope="AVAILABLE_FROZEN_SITE_TO_HISTORY_FORECAST_CONNECTIONS_NOT_OBSERVED_EMPTY_TRIPS",
        source_cached_connections_sha256=sha(source/"sparse_connections.parquet"))


def prepare(root, cfg):
    previous=root/cfg["fixed_local_output"]
    fixed=json.loads((root/cfg["fixed_local_config"]).read_text(encoding="utf-8"))
    cases=[]
    for cut in fixed["window_starts_s"]:
        case_id=f"LOCAL_{fixed['replay_date']}_{cut//3600:02d}{cut%3600//60:02d}"
        source=previous/case_id
        inputs=json.loads((source/"input.json").read_text(encoding="utf-8"))
        if inputs["config"] != fixed:
            raise ValueError("fixed local input/config mismatch")
        reference=fit_reference(source, inputs, cfg)
        cases.append(dict(case_id=case_id,cut=cut,source=source,inputs=inputs,reference=reference))
    info=dict(status="GENERIC_REFERENCE_FIXED_BEFORE_NEW_CONTROL", base_output=cfg["fixed_local_output"],
        config=cfg, fixed_local_config_sha256=sha(root/cfg["fixed_local_config"]),
        cases=[dict(case_id=c["case_id"],reference=c["reference"],fixed_input_sha256=sha(c["source"]/"input.json")) for c in cases],
        no_existing_control_reruns=True)
    atomic_json(root/DOC/"reference.json",info)
    return fixed,cases,info


def summarize(root,result):
    for case in result["cases"]:
        source=root/case["fixed_source"]
        destination=root/case["private_output"]
        controls={policy:json.loads((source/f"{policy}_full_result.json").read_text(encoding="utf-8"))
            for policy in ("SERVICE_PRESERVING","CHAIN_DEFER")}
        controls["TIME_TYPE_DEFER"]=json.loads((destination/"TIME_TYPE_DEFER_full_result.json").read_text(encoding="utf-8"))
        case["policies"]=[]
        for policy,full in controls.items():
            compact=compact_policy(full)
            events=full["executed_events"]
            compact.update(total_executed_empty_distance_m=sum(e["empty_distance_m"] for e in events),
                modeled_passenger_service_time_s=sum(e["service_time_s"] for e in events),
                post_window_committed_work_s=sum(max(0,e["finish_s"]-max(1800,e["epoch_s"])) for e in events))
            case["policies"].append(compact)
        main=controls["CHAIN_DEFER"]["committed_jobs"];generic=controls["TIME_TYPE_DEFER"]["committed_jobs"]
        common=set(main)&set(generic)
        case["main_vs_wait_enabled_generic"]=dict(net_service_difference=len(main)-len(generic),
            main_only_jobs=sorted(set(main)-set(generic)), generic_only_jobs=sorted(set(generic)-set(main)),
            common_served_count=len(common),paired_mean_pickup_wait_difference_s=float(np.mean([
                main[j]["pickup_s"]-generic[j]["pickup_s"] for j in common])) if common else None)
    atomic_json(root/OUT/"run_summary.json",result)
    atomic_json(root/DOC/"summary.json",result)
    return result


def run(root,cfg_path=CONFIG,prepare_only=False,resume=False):
    started=perf_counter()
    cfg=json.loads((root/cfg_path).read_text(encoding="utf-8"))
    if cfg["full_day_or_native"] or cfg["refit_M3_or_time_proxy"] or cfg["parameter_search"]:
        raise ValueError("only the fixed two-window generic control is authorized")
    fixed,cases,info=prepare(root,cfg)
    if prepare_only:
        print(json.dumps(info),flush=True)
        return info
    previous=None
    summary_path=root/OUT/"run_summary.json"
    if resume and summary_path.is_file():
        previous=json.loads(summary_path.read_text(encoding="utf-8"))
        if previous["reference_preparation"] != info:
            raise ValueError("resume requires exactly the same reference, data and protocol")
        if previous["status"]=="FAILED":
            failure_path=root/OUT/"failed_attempt_001.json"
            if not failure_path.exists():
                atomic_json(failure_path,previous)
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    proxy=json.loads((root/fixed["time_proxy_source"]).read_text(encoding="utf-8"))
    adapter=HistoricalPredictionTimeProxy(root)
    adapter.factors={int(r["window_start_s"]):float(r["factor"]) for r in proxy["factors"]}
    router=InstanceRouter(root,adapter);process=psutil.Process()
    result=dict(status="RUNNING",kind="WAIT_ENABLED_GENERIC_STRUCTURE_COMPARISON_NOT_CITY_EVALUATION",
        reference_preparation=info,cases=[],resume_requested=resume,reused_completed_cases=[])
    atomic_json(root/OUT/"run_summary.json",result)

    def budget_check():
        if perf_counter()-started > fixed["maximum_runtime_s"]:
            raise RuntimeError("new generic control exceeded fixed total time budget")
        if process.memory_info().rss/2**20 > fixed["maximum_rss_mib"]:
            raise RuntimeError("new generic control exceeded fixed RSS budget")
        if router.route_request_count > fixed["maximum_route_queries"]:
            raise RuntimeError("new generic control exceeded fixed route budget")

    for case in cases:
        inputs=case["inputs"];source=case["source"]
        destination=root/OUT/case["case_id"]
        case_meta=dict(case_id=case["case_id"],fixed_source=str(source.relative_to(root)),
            private_output=str(destination.relative_to(root)),reference=case["reference"],
            shared_initial_layout={r["resource_id"]:r["location_id"] for r in inputs["shared_initial_resources"]},
            reference_is_not_a_physical_recourse_certificate=True)
        completed_path=destination/"TIME_TYPE_DEFER_full_result.json"
        if resume and completed_path.is_file():
            completed=json.loads(completed_path.read_text(encoding="utf-8"))
            if (completed["policy"]!="TIME_TYPE_DEFER" or completed["actual_cohort_count"]!=len(inputs["actual_tasks"])
                or completed["epochs"][0]["valuation_model"]["reference"]!=case["reference"]):
                raise ValueError("completed case does not match the fixed control reference")
            result["cases"].append(case_meta)
            result["reused_completed_cases"].append(case["case_id"])
            print(json.dumps(dict(reused_completed_case=case["case_id"],new_execution=False)),flush=True)
            continue
        universe=pd.read_parquet(source/"private_source_mapping.parquet")
        evidence=MixedDateEvidence(root,router,universe,fixed)
        provider=ConnectionProvider(router,evidence,universe,inputs["sites"],fixed,case["cut"])
        # Warm only frozen physical query values. Contexts for actual relocations
        # are still reconstructed by relocation(), not fabricated from scalars.
        for row in pd.read_parquet(source/"sparse_connections.parquet").to_dict("records"):
            row={key:(value.tolist() if isinstance(value,np.ndarray) else value.item()
                if isinstance(value,np.generic) else value) for key,value in row.items()}
            provider.cache[row["origin_id"],row["target_job_id"]]=row
        run_cfg=dict(fixed,skip_planned_post_service_geometry_queries=True)
        reference=case["reference"]
        replay=replay_policy(inputs["actual_tasks"],inputs["forecast_cohorts"],inputs["shared_initial_resources"],
            provider,"CHAIN_DEFER",run_cfg,budget_check,
            problem_transform=lambda problem:project_post_service_geometry(problem,reference),
            report_policy="TIME_TYPE_DEFER")
        atomic_json(destination/"TIME_TYPE_DEFER_full_result.json",replay)
        atomic_parquet(destination/"TIME_TYPE_DEFER_events.parquet",pd.DataFrame(replay["executed_events"]))
        result["cases"].append(case_meta)
        atomic_json(root/OUT/"run_summary.json",result)
        budget_check()
    result.update(status="WAIT_ENABLED_GENERIC_COMPARISON_COMPLETE",runtime_s=perf_counter()-started,
        peak_rss_mib=process.memory_info().peak_wset/2**20,routing=router.diagnostics(),
        existing_control_reruns=0,gpu_used=False,full_day_runs=0,native_runs=0)
    summarize(root,result)
    print(json.dumps(dict(status=result["status"],runtime_s=result["runtime_s"],peak_rss_mib=result["peak_rss_mib"])),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--config",type=Path,default=CONFIG)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--summarize-existing",action="store_true")
    parser.add_argument("--resume",action="store_true",help="reuse exact completed windows without rerunning them")
    args=parser.parse_args();root=args.root.resolve()
    if args.summarize_existing:
        summarize(root,json.loads((root/OUT/"run_summary.json").read_text(encoding="utf-8")))
    else:
        try:
            run(root,args.config,args.prepare_only,args.resume)
        except Exception as exc:
            path=root/OUT/"run_summary.json"
            if path.is_file() and not args.prepare_only:
                failed=json.loads(path.read_text(encoding="utf-8"))
                if failed.get("status")=="RUNNING":
                    failed.update(status="FAILED",failure_type=type(exc).__name__,failure_detail=str(exc))
                    atomic_json(path,failed)
            raise


if __name__=="__main__":main()
