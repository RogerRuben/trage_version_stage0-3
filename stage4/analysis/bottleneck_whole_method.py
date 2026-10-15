"""Prepare a bounded whole-method comparison; execution always needs --execute.

No arguments means metadata/configuration preparation, NOT layout optimization
or simulation. The three fixed arms share inputs and resource budgets. Reuse
the existing physical runner/accounting without widening its old public CLI.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from time import perf_counter,sleep

from .scheme_a_two_scale_check import _atomic_json,_sha,FLEETPY


CONFIG=Path("stage4/config/bottleneck_whole_method_v1.json")
BASE=Path("stage4/config/scheme_a_two_scale_city_v5_common.json")
OUT=Path("stage4/output/capability_chain_planning/bottleneck_whole_method_v1")
DOC=Path("stage4/docs/capability_chain_planning/bottleneck_whole_method_v1")
ARMS=("INDEPENDENT_REFERENCE","SPATIAL_SHARED_PLAN","BOTTLENECK_SHARED_PLAN")
MODES=("reference","spatial_joint","bottleneck_joint")
COMPLETE="WHOLE_METHOD_NATIVE_GROUP_COMPLETE"
V2="bottleneck_whole_method_v2"
V3="bottleneck_whole_method_v3"


def output_root(cfg):
    return OUT.with_name(cfg["version"])


def document_root(cfg):
    return DOC.with_name(cfg["version"])


def source_directory(root,cfg,group):
    if group==ARMS[0] and cfg.get("retained_reference_summary"):
        return (root/cfg["retained_reference_summary"]).parent
    return root/output_root(cfg)/"native"/group


def receipt_path(root,cfg,group):
    if group==ARMS[0] and cfg.get("retained_reference_receipt"):
        return root/cfg["retained_reference_receipt"]
    return root/output_root(cfg)/"native/receipts"/f"{group}.json"


@contextmanager
def execution_lock(root,relative=OUT):
    """One layout/coordinator at a time; a crash leaves an explicit stop marker."""
    path=root/relative/"execution.lock"
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("another execution or retained crash lock exists; inspect it before restarting") from error
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as handle:
            json.dump(dict(pid=os.getpid(),automatic_stale_lock_removal=False),handle)
        yield
    finally:
        path.unlink()


def code_sha(root):
    return subprocess.check_output(["git","-c",f"safe.directory={root.as_posix()}",
                                   "rev-parse","HEAD"],cwd=root,text=True).strip()


def assemble(base,declared,seed,root):
    """Pure protocol assembly, also used by non-running contract checks."""
    cfg=dict(base,version=declared["version"],base_config=BASE.as_posix(),
        layout_product="PREPARED_300_600",layout_generators=list(MODES),
        source_layout=(OUT/"layout/placements.json").as_posix(),
        expected_layout_bins=len(seed["records"]),
        expected_av_slots=len(seed["placements"]["opportunity_joint"]),
        groups=list(ARMS),idle_movement_mode="VALUE_SELECTED",
        method_scope="SHARED_INTEGER_HISTORY_PLAN_AND_BOUNDED_RESOURCE_SEARCH_WITH_INDEPENDENT_REFERENCE",
        experiment_scope="FIXED_THREE_ARM_WHOLE_METHOD_COMPARISON_NOT_PARAMETER_SEARCH",
        joint_maximum_coalition=6,joint_maximum_partial_states=128,joint_maximum_labels=50000)
    fixed=dict(test_date="20161031",profile_id="C",requested_q_a=.1,supply_seed=20260824,
        passenger_acceptance_rate=.7,solver_time_limit_s=10.,full_day=True,
        max_match_wait_s=300,max_pickup_eta_s=600,maximum_rss_mib=1536,
        minimum_initial_available_ram_mib=1536,scenario_timeout_s=10800)
    if any(cfg.get(k)!=v for k,v in fixed.items()):
        raise ValueError("whole-method comparison changed the inherited scientific/resource conditions")
    if cfg["expected_av_slots"]!=851 or cfg["expected_layout_bins"]!=47:
        raise ValueError("the original q10 seed layout is not the complete 851-slot/47-bin product")
    if declared["version"] in (V2,V3):
        cfg.update(source_layout=declared["reuse_layout"],joint_execution_mode=declared.get("joint_execution_mode","INCREMENTAL_V2"),
            retained_reference_summary=declared["retained_reference_summary"],
            retained_reference_receipt=declared["retained_reference_receipt"],
            method_scope="INCREMENTAL_SHARED_PLAN_WITH_RETAINED_INDEPENDENT_REFERENCE")
        if declared["version"]==V3:
            cfg["joint_acceptance_rule"]=declared.get("joint_acceptance_rule","FIRST_CURRENT_IMPROVEMENT")
    cfg["source_layout_sha256"]=_sha(root/cfg["source_layout"]) if (root/cfg["source_layout"]).is_file() else None
    return cfg


def load(root,config=CONFIG):
    from .scheme_a_two_scale_experiment import load_protocol
    declared=json.loads((root/config).read_text(encoding="utf-8"))
    if (declared["version"] not in ("bottleneck_whole_method_v1",V2,V3)
            or declared["base_config"]!=BASE.as_posix() or declared["groups"]!=list(ARMS)
            or declared["default_action"]!="prepare"
            or any(declared[k] is not False for k in ("automatic_retry","parameter_search","model_refit","dense_matrix","gpu_used"))
            or _sha(root/BASE)!=declared["base_config_sha256"]
            or _sha(root/declared["seed_layout"])!=declared["seed_layout_sha256"]):
        raise ValueError("outside the fixed three-arm whole-method protocol")
    seed=json.loads((root/declared["seed_layout"]).read_text(encoding="utf-8"))
    if seed["status"]!="LAYOUTS_PREPARED_300_600":
        raise ValueError("only the completed original layout may initialize all arms")
    if declared["version"] in (V2,V3):
        if declared.get("joint_execution_mode")!=("INCREMENTAL_V3" if declared["version"]==V3 else "INCREMENTAL_V2"):
            raise ValueError("V2 must explicitly enable incremental execution")
        if declared["version"]==V3 and declared.get("joint_acceptance_rule")!="FIRST_CURRENT_IMPROVEMENT":
            raise ValueError("V3 freezes first-current-improvement execution, not parameter search")
        for key in ("reuse_layout","retained_reference_summary","retained_reference_receipt"):
            if _sha(root/declared[key])!=declared[key+"_sha256"]:
                raise ValueError("a retained V1 product differs; no implicit reconstruction")
    cfg=assemble(load_protocol(root,BASE),declared,seed,root)
    if cfg["version"] in (V2,V3):
        retained_reference(root,cfg)
    return cfg,declared,seed


def group_config(cfg,group):
    if group not in ARMS:
        raise ValueError("unknown whole-method arm")
    i=ARMS.index(group)
    chosen=dict(cfg,group=group,layout_mode=MODES[i],policy="SERVICE_PRESERVING",
        current_decision_mode="INDEPENDENT_VALUE" if i==0 else "BOTTLENECK_JOINT",
        joint_neighborhood_strategy=None if i==0 else "SPATIAL_JOINT" if i==1 else "BOTTLENECK_CLOSURE",
        disk_route_cache=(output_root(cfg)/"native"/group/"routing/cache.sqlite3").as_posix())
    if cfg["version"] in (V2,V3):
        chosen["joint_execution_mode"]="REBUILD_V1" if i==0 else cfg["joint_execution_mode"]
    return chosen


def retained_reference(root,cfg):
    """Validate old-result provenance; never copy it into a new-run directory."""
    summary=json.loads((root/cfg["retained_reference_summary"]).read_text(encoding="utf-8"))
    receipt=json.loads((root/cfg["retained_reference_receipt"]).read_text(encoding="utf-8"))
    if (summary.get("status")!=COMPLETE or not summary.get("full_day")
            or summary.get("group")!=ARMS[0] or receipt.get("status")!="COMPLETE"
            or receipt.get("exit_code")!=0 or receipt.get("parent_hard_timeout")):
        raise ValueError("retained reference must be a complete clean original run")
    expected=dict(group_config(cfg,ARMS[0]),process_group_memory_limit_mib=cfg["maximum_rss_mib"])
    old=summary["frozen_protocol"]
    administrative={"version","method_scope","experiment_scope","base_config","disk_route_cache"}
    if any(expected.get(k)!=v for k,v in old.items() if k not in administrative):
        raise ValueError("retained reference has different physical/scientific settings")
    if old.get("current_decision_mode","INDEPENDENT_VALUE")!="INDEPENDENT_VALUE":
        raise ValueError("retained result is not the independent reference")
    return summary


def prepare(root,config=CONFIG):
    cfg,declared,seed=load(root,config)
    configs={g:group_config(cfg,g) for g in ARMS}
    for group,chosen in configs.items():
        _atomic_json(root/output_root(cfg)/"setup"/f"{group}.json",chosen)
    result=dict(status="PREPARED_NOT_EXECUTED",code_sha=code_sha(root),
        config_sha256=_sha(root/config),seed_layout_sha256=declared["seed_layout_sha256"],
        groups=list(ARMS),native_group_count=2 if cfg.get("retained_reference_summary") else 3,layout_optimization_started=False,
        simulation_started=False,routing_started=False,inference_started=False,
        methods={g:{"layout_mode":c["layout_mode"],"current_decision_mode":c["current_decision_mode"],
                    "joint_neighborhood_strategy":c["joint_neighborhood_strategy"],
                    "joint_execution_mode":c.get("joint_execution_mode","REBUILD_V1")} for g,c in configs.items()},
        new_layouts_produced=False,expected_av_slots=cfg["expected_av_slots"],
        expected_layout_bins=cfg["expected_layout_bins"],
        reference_results_are_new_runs_not_mislabeled_old_common_control=not bool(cfg.get("retained_reference_summary")),
        reference_retained=bool(cfg.get("retained_reference_summary")),
        retained_reference_source=cfg.get("retained_reference_summary"),
        reference_execution="RETAINED_COMPLETED_V1" if cfg.get("retained_reference_summary") else "NEW_RUN",
        new_native_jobs=list(ARMS[1:] if cfg.get("retained_reference_summary") else ARMS),
        layouts_recomputed=False)
    _atomic_json(root/output_root(cfg)/"setup/prepared.json",result)
    return result


def _layouts(root,config=CONFIG,*,resume=False):
    """Explicit compute command; never called by prepare()."""
    import pandas as pd
    import psutil
    from .c_operations_diagnostic import START
    from .scheme_a_city_experiment import load_history_templates
    from stage4.dispatch.controlled_fleet import FLEET_REL,build_controlled_fleet
    from stage4.dispatch.repositioning_policy import load_train_demand_reference,REFERENCE_REL,REFERENCE_MANIFEST_REL
    from stage4.dispatch.flexibility_native import TrainDemandForecast
    from stage4.dispatch.scheme_a_city_graph import CoarseHistoricalLibrary
    from stage4.dispatch.scheme_a_city_adapter import sites_for
    from stage4.dispatch.bottleneck_layout import solve_birth_batch
    cfg,declared,seed=load(root,config)
    if cfg["version"] in (V2,V3):
        raise ValueError("V2 retains the complete V1 layouts; it does not run layout production")
    directory=root/output_root(cfg)/"layout"; checkpoint=directory/"checkpoint.json"
    if (directory/"placements.json").exists():
        raise ValueError("completed new layout retained; no overwrite/recomputation")
    if checkpoint.exists() and not resume:
        raise ValueError("partial new layout retained; use explicit identical --resume")
    if psutil.virtual_memory().available/2**20<1536:
        raise MemoryError("layout requires 1536 MiB free RAM; other applications are not closed")
    started=perf_counter()
    def guard():
        rss=psutil.Process().memory_info()
        if max(rss.rss,getattr(rss,"peak_wset",rss.rss))/2**20>512:
            raise MemoryError("offline layout exceeds the original 512 MiB cap")
        if perf_counter()-started>=1800:
            raise TimeoutError("new layout preparation reached its fixed administrative limit")
        if psutil.disk_usage(root.anchor).free/2**20<512:
            raise OSError("disk reserve below 512 MiB")
    paths=(config,BASE,Path(declared["seed_layout"]),FLEET_REL,REFERENCE_REL,REFERENCE_MANIFEST_REL,
           Path(cfg["source_history_templates"]),Path(cfg["source_history_manifest"]),Path(cfg["frozen_profile_config"]))
    binding=dict(code_sha=code_sha(root),inputs_sha256={p.as_posix():_sha(root/p) for p in paths},
                 waiting_contract=dict(max_match_wait_s=300,max_pickup_eta_s=600))
    maps={m:dict(seed["placements"]["opportunity_joint"]) for m in MODES}; records=[]
    if checkpoint.exists():
        previous=json.loads(checkpoint.read_text(encoding="utf-8"))
        if previous["binding"]!=binding:
            raise ValueError("new layout checkpoint code/inputs differ")
        maps,records=previous["placements"],previous["records"]
    reference,_=load_train_demand_reference(root)
    templates,inventory=load_history_templates(root,cfg)
    library=CoarseHistoricalLibrary(templates,cfg)
    pace=TrainDemandForecast(templates,dict(cfg,forecast_horizon_s=1800,fast_forecast=True),86434)
    del templates
    fleet=build_controlled_fleet(root,benchmark_start=START,
        simulation_end=START+pd.Timedelta(seconds=cfg["physical_drain_limit_s"]),
        requested_q_a=.1,seed=20260824)
    labels=fleet.accounting["type_label_sha256"]
    if labels!=seed["type_label_sha256"]:
        raise ValueError("new layout changed original fleet labels")
    slots=[dict(vehicle_id=str(r.vehicle_id),native_id=int(r.native_id),
        profile_id="C" if r.vehicle_type=="AV" else "HV",
        activation_s=float((r.availability_start_time-START).total_seconds()),
        admission_end_s=float((r.availability_end_time-START).total_seconds()),
        position=(float(r.initial_lon_wgs84),float(r.initial_lat_wgs84)))
        for r in fleet.scenario_fleet.itertuples(index=False)]
    del fleet
    births=[r for r in slots if r["profile_id"]=="C"]
    bins=sorted({int(r["activation_s"]//1800)*1800 for r in births})
    if [r["bin_s"] for r in records]!=bins[:len(records)]:
        raise ValueError("new layout checkpoint is not a completed bin prefix")
    for now in bins[len(records):]:
        guard(); sites=sites_for(reference,now,cfg)
        fresh={r["native_id"] for r in births if int(r["activation_s"]//1800)*1800==now}
        row=dict(bin_s=now,newly_placed_AV_slots=len(fresh),methods={})
        for mode,strategy in (("spatial_joint","SPATIAL_JOINT"),("bottleneck_joint","BOTTLENECK_CLOSURE")):
            chosen,info=solve_birth_batch(slots,fresh,sites,maps[mode],library,pace,now,
                                         strategy=strategy,check=guard)
            maps[mode].update(chosen); row["methods"][mode]=info
        records.append(row)
        _atomic_json(checkpoint,dict(binding=binding,placements=maps,records=records))
        print(json.dumps(dict(layout_bins=len(records),total=len(bins),runtime_s=perf_counter()-started)),flush=True)
        gc.collect()
    expected={r["vehicle_id"] for r in births}
    if len(records)!=47 or any(set(m)!=expected for m in maps.values()):
        raise RuntimeError("new layouts lost original slots")
    if any(_sha(root/p)!=h for p,h in binding["inputs_sha256"].items()):
        raise RuntimeError("a layout source changed")
    guard()
    result=dict(status="LAYOUTS_PREPARED_300_600",binding=binding,type_label_sha256=labels,
        supply_contract=dict(requested_q_a=.1,supply_seed=20260824),placements=maps,records=records,
        native_execution=False,target_day_requests_loaded=False,new_inference=False,
        preparation_supply_positions_not_observed_midday_locations=True,
        no_future_road_certificates=True,historical_inventory=inventory,
        runtime_s=perf_counter()-started)
    _atomic_json(directory/"placements.json",result)
    return {k:v for k,v in result.items() if k!="placements"}


def layouts(root,config=CONFIG,*,resume=False):
    cfg,_,_=load(root,config)
    with execution_lock(root,output_root(cfg)):
        return _layouts(root,config,resume=resume)


def binding_for(root,config,cfg):
    from .scheme_a_two_scale_check import load_frozen_layout
    if cfg["source_layout_sha256"] is None:
        raise ValueError("explicit layout computation must finish before native execution")
    _,protected=load_frozen_layout(root,cfg)
    protected.update({config.as_posix():_sha(root/config),cfg["base_config"]:_sha(root/cfg["base_config"]),
                      cfg["source_layout"]:_sha(root/cfg["source_layout"])})
    if cfg["group"]==ARMS[0] and cfg.get("retained_reference_summary"):
        old=retained_reference(root,cfg)["execution_binding"]
        if any(_sha(root/p)!=h for p,h in old["inputs_sha256"].items()):
            raise ValueError("a retained reference input changed")
        return old
    bound=dict(code_sha=code_sha(root),inputs_sha256=protected,group=cfg["group"],
        layout_mode=cfg["layout_mode"],policy=cfg["policy"],version=cfg["version"],
        layout_sha256=cfg["source_layout_sha256"],
        current_decision_mode=cfg["current_decision_mode"],
        joint_neighborhood_strategy=cfg["joint_neighborhood_strategy"])
    if cfg["version"] in (V2,V3):
        bound["joint_execution_mode"]=cfg["joint_execution_mode"]
    return bound


def completed(root,group,binding,*,cfg=None):
    from .scheme_a_two_scale_experiment import PRODUCTS
    cfg={"version":"bottleneck_whole_method_v1"} if cfg is None else cfg
    path=source_directory(root,cfg,group)
    summary=json.loads((path/"summary.json").read_text(encoding="utf-8"))
    if (summary.get("status")!=COMPLETE or summary.get("execution_binding")!=binding
            or not summary.get("full_day")
            or any(not (path/p).is_file() for p in (*PRODUCTS,"request_decisions.parquet"))):
        raise ValueError("not a complete identical new-method group; no automatic retry")
    return summary


def worker(root,config,group,fleetpy):
    from .scheme_a_two_scale_check import _run_native
    cfg,_,_=load(root,config); cfg=group_config(cfg,group)
    if group==ARMS[0] and cfg.get("retained_reference_summary"):
        raise ValueError("V2 reuses the completed reference; no reference worker is launched")
    bound=binding_for(root,config,cfg)
    return _run_native(root,fleetpy,config,cfg,directory=root/output_root(cfg)/"native"/group,
        doc_summary_path=root/document_root(cfg)/f"{group}_summary.json",completion_status=COMPLETE,
        group=group,execution_binding=bound)


def child_command(root,config,group,fleetpy):
    return [sys.executable,"-B","-u","-m","stage4.analysis.bottleneck_whole_method",
        "worker","--execute","--root",str(root),"--config",str(config),
        "--group",group,"--fleetpy-root",str(fleetpy)]


def _run(root,config=CONFIG,*,resume=False,fleetpy=FLEETPY):
    from .scheme_a_two_scale_experiment import _BoundedLog
    cfg,_,_=load(root,config)
    common_binding={g:binding_for(root,config,group_config(cfg,g)) for g in ARMS}
    directory=root/output_root(cfg)/"native"
    stamp=directory/"comparison_binding.json"
    if stamp.exists() and (not resume or json.loads(stamp.read_text(encoding="utf-8"))!=common_binding):
        raise ValueError("existing comparison retained; only explicit identical --resume is allowed")
    # Validate all existing states BEFORE launching any new child.
    done=[]
    for group in ARMS:
        if group==ARMS[0] and cfg.get("retained_reference_summary"):
            completed(root,group,common_binding[group],cfg=cfg); done.append(group)
            continue
        path=directory/group
        receipt=directory/"receipts"/f"{group}.json"
        log_path=directory/"logs"/f"{group}.log"
        if (path.exists() and any(path.iterdir())) or receipt.exists() or log_path.exists():
            if not resume:
                raise ValueError("existing native group retained")
            completed(root,group,common_binding[group],cfg=cfg)
            recorded=json.loads(receipt.read_text(encoding="utf-8"))
            if recorded.get("status")!="COMPLETE" or recorded["exit_code"]!=0 or recorded["parent_hard_timeout"]:
                raise ValueError("unfinished/failed child retained; no automatic retry")
            done.append(group)
    _atomic_json(stamp,common_binding)
    for group in ARMS:
        if group in done:
            continue
        command=child_command(root,config,group,fleetpy)
        env=dict(os.environ)
        n=int(env.get("GIT_CONFIG_COUNT","0"))
        for path in (fleetpy,root):
            env[f"GIT_CONFIG_KEY_{n}"]="safe.directory"; env[f"GIT_CONFIG_VALUE_{n}"]=str(path); n+=1
        env["GIT_CONFIG_COUNT"]=str(n)
        log=_BoundedLog(directory/"logs"/f"{group}.log",group,cfg)
        child=subprocess.Popen(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        started=perf_counter()
        receipt=directory/"receipts"/f"{group}.json"
        timeout=False
        try:
            reader=threading.Thread(target=log.drain,args=(child.stdout,),daemon=True); reader.start()
            _atomic_json(receipt,dict(status="RUNNING",child_pid=child.pid,command=command,automatic_retry=False))
            while child.poll() is None:
                if perf_counter()-started>=cfg["scenario_timeout_s"]:
                    timeout=True
                    print(json.dumps(dict(group=group,action="STOP_OWN_CHILD_AT_HARD_LIMIT",pid=child.pid)),flush=True)
                    child.terminate(); break
                if log.error is not None:
                    child.terminate(); break
                sleep(.2)
            code=child.wait(); reader.join(10.)
        finally:
            if child.poll() is None:
                child.terminate(); child.wait()
        clean=code==0 and not timeout and not reader.is_alive() and log.error is None
        _atomic_json(receipt,
            dict(status="COMPLETE" if clean else "STOPPED",child_pid=child.pid,exit_code=code,
                 wall_time_s=perf_counter()-started,parent_hard_timeout=timeout,
                 log_statistics=log.stats,log_capture_error=log.error,automatic_retry=False))
        if not clean:
            raise RuntimeError("native child/log failed; remaining groups not started, no retry")
        completed(root,group,common_binding[group],cfg=cfg); done.append(group)
    _atomic_json(directory/"status.json",dict(status="WHOLE_METHOD_COMPARISON_COMPLETE",groups=done))
    return dict(status="WHOLE_METHOD_COMPARISON_COMPLETE",groups=done)


def run(root,config=CONFIG,*,resume=False,fleetpy=FLEETPY):
    cfg,_,_=load(root,config)
    with execution_lock(root,output_root(cfg)):
        return _run(root,config,resume=resume,fleetpy=fleetpy)


def analyze(root,config=CONFIG):
    """Read the three completed arms sequentially; never rerun missing work."""
    import pandas as pd
    from .c_operations_diagnostic import START
    from .operations_v5_results import _collect,_peak
    from .operations_v5_ablation_results import metric_differences,paired_counts
    from stage4.dispatch.controlled_fleet import build_controlled_fleet
    started=perf_counter()
    cfg,_,_=load(root,config)
    directory=root/output_root(cfg)/"native"
    state=json.loads((directory/"status.json").read_text(encoding="utf-8"))
    binding=json.loads((directory/"comparison_binding.json").read_text(encoding="utf-8"))
    expected={g:dict(binding_for(root,config,group_config(cfg,g)),code_sha=binding[g]["code_sha"]) for g in ARMS}
    if state!={"status":"WHOLE_METHOD_COMPARISON_COMPLETE","groups":list(ARMS)} or binding!=expected:
        raise ValueError("requires the complete unchanged three-arm comparison")
    fresh=ARMS[1:] if cfg.get("retained_reference_summary") else ARMS
    if len({binding[g]["code_sha"] for g in fresh})!=1:
        raise ValueError("the three native arms used different code")
    layout=json.loads((root/cfg["source_layout"]).read_text(encoding="utf-8"))
    fleet=build_controlled_fleet(root,benchmark_start=START,
        simulation_end=START+pd.Timedelta(seconds=cfg["physical_drain_limit_s"]),
        requested_q_a=cfg["requested_q_a"],seed=cfg["supply_seed"])
    label=fleet.accounting["type_label_sha256"]
    plans={int(r.native_id):(str(r.vehicle_type),float((r.availability_start_time-START).total_seconds()),
        float((r.availability_end_time-START).total_seconds())) for r in fleet.scenario_fleet.itertuples(index=False)}
    del fleet
    public,private,receipts={},{},{}
    for group in ARMS:
        summary=completed(root,group,binding[group],cfg=cfg)
        receipt=json.loads(receipt_path(root,cfg,group).read_text(encoding="utf-8"))
        if receipt["status"]!="COMPLETE" or receipt["exit_code"]!=0 or receipt["parent_hard_timeout"]:
            raise ValueError("an unfinished/failed arm is not a scientific result")
        if summary["all_day_supply"]["accounting"]["type_label_sha256"]!=label:
            raise ValueError("comparison supply labels differ")
        row,private[group]=_collect(root,group,summary,plans,directory=source_directory(root,cfg,group))
        row["method"]=group_config(cfg,group)["current_decision_mode"]
        row["joint_control_operations"]=summary["adapter_performance"]["operations"]
        mode=group_config(cfg,group)["layout_mode"]
        row["retained_layout_solver_time_s"]=sum(
            r["methods"].get(mode,{}).get("runtime_s",0.) for r in layout["records"])
        row["incremental_layout_solver_time_s"]=0. if cfg["version"] in (V2,V3) else row["retained_layout_solver_time_s"]
        row["result_retained_not_rerun"]=bool(group==ARMS[0] and cfg.get("retained_reference_summary"))
        if group!=ARMS[0] and "secondary_distance" in row:
            row["secondary_distance"]["scope"]="JOINT_PLAN_NOT_CURRENT_MAXIMUM_SERVICE_PRIMARY_FACE"
        public[group]=row
        receipts[group]=receipt
        gc.collect()
    comparisons={f"{before}_to_{after}":dict(before=before,after=after,
        difference_after_minus_before=metric_differences(public[before],public[after]),
        paired=paired_counts(private[before],private[after])) for before,after in combinations(ARMS,2)}
    result=dict(status="WHOLE_METHOD_COMPARISON_ANALYZED",groups=public,comparisons=comparisons,
        test_date="20161031",profile_id="C",requested_AV_baseline_hour_ratio=.1,
        passenger_acceptance_rate=.7,request_to_assignment_limit_s=300,independent_pickup_eta_cap_s=600,
        common_requests=28367,execution_binding=binding,child_receipts=receipts,
        analysis_invoked_simulation=False,analysis_invoked_routing=False,analysis_invoked_inference=False,
        one_day_one_seed=True,statistical_significance_claimed=False,full_city_optimality_claimed=False,
        deployment_vs_online_effect_separately_identified=False,
        retained_reference=bool(cfg.get("retained_reference_summary")),
        new_native_run_count=2 if cfg.get("retained_reference_summary") else 3,
        layouts_reused_from_v1=cfg["version"] in (V2,V3),
        runtime_differences_scope="NATIVE_ONLY_OFFLINE_INCREMENTAL_LAYOUT_TIME_REPORTED_SEPARATELY",
        reference_offline_layout_is_retained_not_newly_recomputed=True,analysis_code_sha=code_sha(root),
        raw_identifiers_or_coordinates_exported=False,runtime_s=perf_counter()-started,peak_rss_mib=_peak())
    _atomic_json(root/document_root(cfg)/"comparison.json",result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",nargs="?",default="prepare",choices=("prepare","layout","run","worker","analyze"))
    p.add_argument("--root",type=Path,default=Path.cwd()); p.add_argument("--config",type=Path,default=CONFIG)
    p.add_argument("--execute",action="store_true"); p.add_argument("--resume",action="store_true")
    p.add_argument("--group",choices=ARMS); p.add_argument("--fleetpy-root",type=Path,default=FLEETPY)
    args=p.parse_args(); root=args.root.resolve()
    if args.action in ("layout","run","worker") and not args.execute:
        p.error("compute commands require explicit --execute; default preparation never starts experiments")
    if args.action=="prepare":
        result=prepare(root,args.config)
    elif args.action=="layout":
        result=layouts(root,args.config,resume=args.resume)
    elif args.action=="run":
        result=run(root,args.config,resume=args.resume,fleetpy=args.fleetpy_root)
    elif args.action=="analyze":
        result=analyze(root,args.config)
    else:
        if args.group is None:
            p.error("worker requires one fixed group")
        result=worker(root,args.config,args.group,args.fleetpy_root)
    print(json.dumps({k:result[k] for k in ("status","groups","native_group_count") if k in result}),flush=True)


if __name__=="__main__":
    main()
