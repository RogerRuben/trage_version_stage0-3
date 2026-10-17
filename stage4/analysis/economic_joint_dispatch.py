"""Prepare static economic inputs and runtime config; NEVER launch a replay.

The monetary model is independently switchable. Current route/capability,
waiting, physical drain, original supply and birth layout contracts are reused.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

from . import c_connection_consistency as base
from stage4.dispatch.operating_economics import OperatingEconomics


CONFIG=Path("stage4/config/economic_joint_dispatch_v1.json")


def load(root, *, controller=None, require_rates=True):
    root=Path(root)
    declared=json.loads((root/CONFIG).read_text(encoding="utf-8"))
    if declared["base_config"]!=base.CONFIG.as_posix():
        raise ValueError("economic dispatch reuses the connection-aligned frozen-supply foundation")
    if declared.get("maximum_economic_block_repairs_per_epoch")!=3:
        raise ValueError("economic dispatch uses the shared three-block repair budget")
    cfg=base.load(root)
    cfg.update(declared)
    mode=controller or cfg["economic_controller"]
    if mode not in ("MYOPIC","SPATIAL","SUBSTITUTION"):
        raise ValueError("only the three comparable economic controls are implemented")
    cfg.update(economic_controller=mode,group="ECONOMIC_"+mode,
        joint_neighborhood_strategy="SPATIAL_JOINT" if mode=="SPATIAL" else "BOTTLENECK_CLOSURE",
        disk_route_cache="stage4/output/capability_chain_planning/economic_joint_v1/routing/cache.sqlite3")
    if require_rates:
        OperatingEconomics.from_config(cfg)
    if any(cfg.get(k) is not False for k in ("full_day_authorized","forced_C_priority",
            "idle_utilization_reward","fixed_cost_in_dispatch","automatic_experiment_launch",
            "recompute_birth_layout","parameter_search","gpu_used","dense_matrix")):
        raise ValueError("this implementation/configuration task does not authorize experiments or fixed-cost changes")
    return cfg


def prepare(root):
    """Stream frozen route lengths for existing history and the common target input.

    Test31 supplies STATIC ROUTE GEOMETRY only. The policy exposes its length
    solely through the already-released request interface, never as forecast.
    """
    import pandas as pd
    import psutil
    from stage4.dispatch.economic_inputs import aggregate_distances
    from .capability_chain_instances import atomic_json,atomic_parquet,sha
    root=Path(root); cfg=load(root,require_rates=False); started=perf_counter()
    history=pd.read_parquet(root/cfg["source_history_templates"],columns=["date","order_id"])
    dates=set(map(str,cfg["forecast_train_dates"]))
    history=history.loc[history.date.astype(str).isin(dates)]
    target_path=root/cfg["source_routes"]
    target=pd.read_parquet(target_path,columns=["order_id","common_eligible"])
    target=target.loc[target.common_eligible.eq(True)]
    populations={str(d):set(g.order_id.astype(str)) for d,g in history.groupby(history.date.astype(str))}
    populations[str(cfg["test_date"])]=set(target.order_id.astype(str))
    rows=[]; inputs=[]; peak=0.
    process=psutil.Process()
    for date,ids in sorted(populations.items()):
        path=root/f"stage2/output_v4/route_conditioned_dataset/revealed_route_proxy/day={date}.parquet"
        values,count=aggregate_distances(path,date,ids)
        rows.extend(values)
        inputs.append(dict(date=date,path=path.relative_to(root).as_posix(),sha256=sha(path),
            orders=len(values),selected_route_parts=count))
        memory=process.memory_info()
        peak=max(peak,getattr(memory,"peak_wset",memory.rss)/2**20)
        if peak>1536: raise MemoryError("economic preparation exceeded the existing 1536-MiB process cap")
    product=root/cfg["economic_distance_product"]
    atomic_parquet(product,pd.DataFrame(rows))
    memory=process.memory_info()
    peak=max(peak,getattr(memory,"peak_wset",memory.rss)/2**20)
    if peak>1536: raise MemoryError("economic preparation exceeded the existing process RAM cap")
    result=dict(status="ECONOMIC_STATIC_INPUTS_PREPARED_NO_SIMULATION",inputs=inputs,
        output=cfg["economic_distance_product"],output_sha256=sha(product),orders=len(rows),
        schema=["date","order_id","service_distance_m"],
        static_target_geometry_is_not_future_demand_or_dynamic_truth=True,
        historical_forecast_dates=sorted(dates),test_date=str(cfg["test_date"]),
        dispatch_rates_ready=all(cfg["dispatch_economics"][k] is not None for k in ("fare_base_cny","fare_per_km_cny")),
        fare_parameter_status=cfg["fare_parameter_status"],runtime_s=perf_counter()-started,peak_rss_mib=peak,
        simulator_invoked=False,routing_invoked=False,inference_invoked=False,full_day_executed=False)
    atomic_json(product.with_suffix(".json"),result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","describe"),nargs="?",default="describe")
    args=parser.parse_args()
    result=prepare(Path.cwd()) if args.action=="prepare" else load(Path.cwd(),require_rates=False)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
