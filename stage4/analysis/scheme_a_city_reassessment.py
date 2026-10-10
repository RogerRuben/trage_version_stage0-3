"""Read stopped traces and time the CURRENT-only kernel; never run replay.

Synthetic kernel timings exclude native routing and the unimplemented coarse
value provider. They are not a full-day speed prediction or policy experiment.
No private order identifiers, coordinates or trace rows are printed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import pandas as pd
import psutil

from stage4.dispatch.scheme_a_city_master import CurrentAction
from stage4.dispatch.scheme_a_current_flow import solve_current_flow


def summarize_trace(root):
    path = root/"stage4/output/capability_chain_planning/scheme_a_city_v1/HOTSPOT_BASELINE/partial_solver_trace.parquet"
    frame = pd.read_parquet(path, columns=["simulation_time_s", "opt_status", "solve_wall_time_s",
        "physical_validation_time_s", "graph_wall_time_s", "graph_or_time_s",
        "graph_connector_evidence_time_s", "master_time_s", "variables", "nonzeros", "graph"])
    optimized = frame.loc[frame.master_time_s.gt(0)]
    graph = [value for value in frame.graph if isinstance(value,dict)]
    count = lambda name:sum(int(value.get(name) or 0) for value in graph)
    executed, reused = count("chain_builds_executed"), count("chain_builds_reused")
    total = float(frame.solve_wall_time_s.sum())
    times = {name:float(frame[name].sum()) for name in ("physical_validation_time_s",
        "graph_wall_time_s", "graph_or_time_s", "graph_connector_evidence_time_s", "master_time_s")}
    return dict(source=str(path.relative_to(root)), rows=len(frame), optimized_rows=len(optimized),
        last_completed_tick_s=int(frame.simulation_time_s.max()), chain_builds_executed=executed,
        chain_builds_reused=reused, chain_reuse_share=reused/(executed+reused),
        connector_queries=count("connector_queries"), labels_generated=count("labels_generated"),
        deadline_rejections=count("deadline_rejected"), discarded_chain_columns=count("column_budget_discarded_chains"),
        retained_prefixes_discarded=count("retained_chain_truncated"),
        adapter_solve_wall_s=total, component_timings_s=times,
        future_graph_share=times["graph_wall_time_s"]/total,
        connection_evidence_share=times["graph_connector_evidence_time_s"]/total,
        master_share=times["master_time_s"]/total,
        optimization_epoch_solve_wall_p50_s=float(optimized.solve_wall_time_s.median()),
        optimization_epoch_solve_wall_max_s=float(optimized.solve_wall_time_s.max()),
        successful_master_p50_s=float(optimized.master_time_s.median()),
        successful_master_max_s=float(optimized.master_time_s.max()),
        max_variables=int(frame.variables.max()), max_nonzeros=int(frame.nonzeros.max()),
        failing_epoch_model_not_saved=True, private_rows_exported=False)


def benchmark_kernel():
    results=[]
    if psutil.virtual_memory().available < 1536*2**20:
        return dict(status="SKIPPED_LOW_AVAILABLE_RAM", cases=[])
    for count in (100,500,1000):
        actions,values=[],{}
        for vid in range(count):
            wait=CurrentAction(f"{vid}:WAIT",vid,"WAIT",None)
            actions.append(wait)
            values[wait.action_id]=0
            for offset in range(10):
                rid=(vid*7+offset*11)%count
                action=CurrentAction(f"{vid}:SERVE:{rid}",vid,"SERVE",rid,
                    critical=rid%17==0,carry_over=rid%3==0,empty_distance_m=50.+offset*10)
                actions.append(action)
                values[action.action_id]=(vid+rid)%19-9
            move=CurrentAction(f"{vid}:MOVE",vid,"RELOCATE",None,empty_distance_m=500.,
                payload={"relocation_slot_s":900})
            actions.append(move)
            values[move.action_id]=(vid%10)-1
        for policy in ("SERVICE_PRESERVING","CHAIN_DEFER"):
            started=perf_counter()
            result=solve_current_flow(actions,values,policy=policy)
            results.append(dict(vehicles=count, admitted_actions=len(actions), policy=policy,
                measured_wall_s=perf_counter()-started, opt_status=result["opt_status"],
                backend=result["backend"], native_backend_fallback_reason=result["native_backend_fallback_reason"],
                served=result["current_served"], moves=result["current_moves"], nodes=result["nodes"],
                residual_edges=result["residual_edges"], augmentations=result["augmentations"],
                future_chain_variables=result["future_chain_variables"], routing_queries=result["routing_queries"]))
    return dict(status="SYNTHETIC_CURRENT_KERNEL_ONLY", cases=results,
        native_routing_included=False, coarse_value_preparation_included=False,
        full_day_speedup_estimated=False, peak_process_rss_mib=psutil.Process().memory_info().peak_wset/2**20)


def main():
    start=datetime.now(timezone.utc).isoformat()
    trace=summarize_trace(Path.cwd())
    kernel=benchmark_kernel()
    print(json.dumps(dict(start_time_utc=start,end_time_utc=datetime.now(timezone.utc).isoformat(),
        stopped_trace=trace,current_kernel=kernel),indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
