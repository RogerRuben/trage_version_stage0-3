"""Read-only city episode snapshots; never dispatch, move, or load execution truth."""
from __future__ import annotations

from dataclasses import fields
import math
from pathlib import Path
from time import perf_counter

import psutil

from stage4.analysis.capability_chain_instances import atomic_json
from stage4.fleetpy_adapter.research_world import DecisionRequest, load_research_episode


def inventory(root):
    started=perf_counter()
    episode=load_research_episode(root,profile_id="C",requested_q_a=.1)
    record=episode.inventory()
    snapshots=[]
    for now in (32400,46800,63000):
        snapshot=episode.snapshot(now)
        outside=sum((math.floor((r.pickup_lon_wgs84-108)/.02),
            math.floor((r.pickup_lat_wgs84-34)/.02))!=(45,11) for r in snapshot.requests)
        snapshots.append(dict(clock_s=now,counts=episode.counts(),
            pending_outside_previous_local_region=outside,
            pending_C_incompatible_kept=sum(not r.av_compatible for r in snapshot.requests),
            decision_truth_fields_present=[f.name for f in fields(DecisionRequest)
                if f.name.startswith(("realized_","observed_"))]))
    record.update(status="CITY_DATA_INTERFACE_DRY_READ_COMPLETE_NOT_SIMULATION",snapshots=snapshots,
        execution_truth_values_loaded=episode.execution_truth._values is not None,
        dynamic_movement_performed=False,dispatch_performed=False,
        expired_counts_are_uncommitted_feed_jumps_not_service_failures=True,
        runtime_s=perf_counter()-started,peak_rss_mib=psutil.Process().memory_info().peak_wset/2**20)
    atomic_json(root/"stage4/docs/capability_chain_planning/generic_defer_v1/city_interface_inventory.json",record)
    return record


if __name__=="__main__":
    result=inventory(Path.cwd())
    print({"status":result["status"],"common_orders":result["requests"]["common_population_requests"],
        "native_solver":result["native_multi_vehicle_solver_status"],"runtime_s":result["runtime_s"]})
