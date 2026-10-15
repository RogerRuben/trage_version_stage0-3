"""Bounded same-state timing of existing route/validation work; no dispatch run."""
from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import types
from time import perf_counter
from types import SimpleNamespace as NS

import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from . import lazy_current_validation_benchmark as fixed
from stage4.dispatch.city_pickup_support import CityPickupSupportValidator
from stage4.dispatch.controlled_routes import ControlledEmptyRouter
from stage4.dispatch.deterministic_routing import SINGLE_SOURCE_MATRIX
from stage4.dispatch.routing_v4 import StaticRawRoutingAdapter
from stage4.dispatch.scheme_a_city_connections import SchemeACityConnectors


OUT = Path("stage4/output/capability_chain_planning/base_cost_optimization")


def digest(value):
    def normalize(item):
        if isinstance(item,dict):
            return {key:normalize(value) for key,value in item.items() if key != "cache_hit"}
        if isinstance(item,(set,frozenset)):
            return [normalize(value) for value in sorted(item,key=repr)]
        if isinstance(item,(tuple,list)):
            return [normalize(value) for value in item]
        return item
    return hashlib.sha256(json.dumps(normalize(value), sort_keys=True, default=str,
        separators=(",", ":")).encode()).hexdigest()


def _reference_classes(root):
    """Read-only baseline source, never checkout/reset the working tree."""
    def module(path,name):
        source = subprocess.check_output(["git","-c",f"safe.directory={root.as_posix()}",
            "show",f"3362391:{path}"],cwd=root,text=True,encoding="utf-8")
        result = types.ModuleType(name)
        result.__package__ = name.rsplit(".",1)[0]
        exec(compile(source,f"3362391:{path}","exec"),result.__dict__)
        return result
    routes = module("stage4/dispatch/controlled_routes.py","stage4.dispatch._reference_routes")
    connectors = module("stage4/dispatch/scheme_a_city_connections.py","stage4.dispatch._reference_connections")
    instances = module("stage4/analysis/capability_chain_instances.py","stage4.analysis._reference_instances")
    symmetric = module("stage4/analysis/symmetric_research_prepare.py","stage4.analysis._reference_symmetric")
    instances.HistoricalResearchParser = symmetric.HistoricalResearchParser
    # The validator and connector import the SAME class; keep its reference
    # implementation for this isolated baseline process only.
    from stage4.analysis.capability_chain_instances import JoinEvidence
    JoinEvidence.evaluate = instances.JoinEvidence.evaluate
    return routes.ControlledEmptyRouter,connectors.SchemeACityConnectors


def run(root, label, reference=None, case_set="initial", legacy=False):
    started = perf_counter()
    process = psutil.Process()
    def check():
        rss = max(process.memory_info().rss, getattr(process.memory_info(), "peak_wset", 0))/2**20
        if rss > 1536 or psutil.virtual_memory().available/2**20 < 512:
            raise MemoryError("bounded base-cost check memory reserve exceeded")
        if perf_counter()-started > 600:
            raise TimeoutError("base-cost check exceeds ten minutes; no automatic retry")
        return rss
    if psutil.virtual_memory().available/2**20 < 1536:
        raise MemoryError("at least 1536 MiB available RAM required before timing")
    case_root = root/OUT/("av_loaded" if case_set == "av_loaded" else "")
    output = case_root/label
    output.mkdir(parents=True, exist_ok=False)
    router_class,connector_class = _reference_classes(root) if legacy else (
        ControlledEmptyRouter,SchemeACityConnectors)
    cfg = json.loads((root/fixed.CONFIG).read_text(encoding="utf-8"))
    protocol, binding, jobs, moves, requests, plans = fixed.load_source(root, cfg)
    rows = []
    profiler = cProfile.Profile()
    with threadpool_limits(limits=1):
        eta = StaticRawRoutingAdapter(root, routing_mode=SINGLE_SOURCE_MATRIX,
            grouped_sources=True, route_workers=1, persistent_cache_size=20000,
            raw_od_cache_size=20000)
        try:
            for now in ((43170, 56700) if case_set == "av_loaded" else (30600, 64800)):
                check()
                parked, pending, past_jobs, past_moves, inventory = fixed.reconstruct(
                    now, jobs, moves, requests, plans, cfg)
                c, states, spatial = fixed.make_context(now, parked, pending,
                    past_jobs, past_moves, requests, protocol, eta)
                c.sim_time = now
                before = perf_counter()
                # Only constructs the unchanged sparse current arcs. No flow,
                # future policy, assignment, or simulated movement is executed.
                original, *_ = fixed.graph_inputs(now, c, states, spatial, pending,
                    eta, NS(value=lambda state: 0.), cfg)
                routing_s = perf_counter()-before
                empty = router_class(root, eta)
                validator = CityPickupSupportValidator(empty,
                    geometry_timestamp=fixed.DAY_START, routing_budget_s=60.)
                selected_vids = [v.native_vehicle_id for v in spatial if v.vehicle_type == "AV"][:2]
                selected_vids += [v.native_vehicle_id for v in spatial if v.vehicle_type == "HV"][:2]
                wanted = {str(a.payload[1].order_id) for a in original.values() if a.vehicle_type == "AV"}
                previous_vids = {a.vehicle_id for a in original.values() if a.vehicle_type == "AV"} | set(selected_vids)
                for row in reversed(c.assignment_rows):
                    vid = row["native_vehicle_id"]
                    if vid in previous_vids and row["completed"]:
                        record = c.request_by_rid.get(row["native_request_id"])
                        if record is not None:
                            wanted.add(str(record.order_id))
                        previous_vids.remove(vid)
                validator.prepare_private_identity_store(wanted)
                con = connector_class(c, empty, validator,
                    pd.DataFrame(columns=["date", "order_id", "compatible_C"]),
                    geometry_timestamp=fixed.DAY_START)
                case = dict(tick=now, inventory=inventory, arc_count=len(original),
                    av_arc_count=sum(a.vehicle_type == "AV" for a in original.values()),
                    routing_s=routing_s, graph_hash=digest([
                        (a.vehicle_id,a.request_id,a.pickup_eta_s,a.payload[2].route_distance_m)
                        for a in original.values()]), passes=[])
                for tick in (now, now+30):
                    check()
                    c.sim_time = tick
                    before = perf_counter()
                    profiler.enable()
                    valid = validator(c, list(original.values()), tick)
                    profiler.disable()
                    validation_s = perf_counter()-before
                    exact = [(a.vehicle_id,a.request_id,a.pickup_eta_s,
                        a.payload[2].route_distance_m,a.payload[2].beta,a.payload[2].time_bin_index,
                        validator.pickup_paths.get((a.vehicle_id,a.request_id))) for a in valid]
                    case["passes"].append(dict(tick=tick, validation_s=validation_s,
                        valid_count=len(valid), output_hash=digest(exact),
                        reasons=validator.last_diagnostics["rejection_reason_counts"]))
                if len(pending):
                    point = tuple(map(float, pending.iloc[0][["pickup_lon_wgs84", "pickup_lat_wgs84"]]))
                    case["move_passes"] = []
                    for tick in (now+30, now+30):
                        c.sim_time = tick
                        before = perf_counter()
                        answers = []
                        for vid in selected_vids:
                            state = states[vid]
                            origin = NS(position=state.position, context=dict(kind="NATIVE_VEHICLE",
                                native_vehicle_id=vid, timestamp_s=tick))
                            answer = con(origin, dict(site_id="FIXED_KNOWN_PICKUP", position=point,
                                require_geometry=True), tick, state.profile_id)
                            if answer.get("routed") is not None:
                                answer["routed"].pop("cache_hit", None)
                            answers.append(answer)
                        case["move_passes"].append(dict(time_s=perf_counter()-before,
                            output_hash=digest(answers), count=len(answers)))
                case.update(pickup_diagnostics=validator.diagnostics(),
                    connector_diagnostics=con.diagnostics(), peak_rss_mib=check())
                rows.append(case)
                print(json.dumps(dict(label=label,tick=now,arcs=len(original),
                    validation_s=[p["validation_s"] for p in case["passes"]],
                    move_s=[p["time_s"] for p in case.get("move_passes",[])])),flush=True)
                del con, validator, empty, c, states, spatial, original
                gc.collect()
        finally:
            eta.close()
    profiler.dump_stats(str(output/"validation.prof"))
    result = dict(status="FIXED_STATE_BASE_COST_TIMING_COMPLETE",label=label,
        source_bindings=binding,cases=rows,runtime_s=perf_counter()-started,
        reference_code="3362391" if legacy else None,
        peak_rss_mib=check(),native_replay=False,dispatch_experiment=False,gpu_used=False)
    if reference:
        previous = json.loads((case_root/reference/"summary.json").read_text(encoding="utf-8"))
        same = previous["source_bindings"] == binding and all(
            a["graph_hash"] == b["graph_hash"] and
            [p["output_hash"] for p in a["passes"]] == [p["output_hash"] for p in b["passes"]] and
            [p["reasons"] for p in a["passes"]] == [p["reasons"] for p in b["passes"]] and
            [p["output_hash"] for p in a.get("move_passes",[])] ==
                [p["output_hash"] for p in b.get("move_passes",[])]
            for a,b in zip(previous["cases"],rows)) and len(previous["cases"]) == len(rows)
        result["exact_fixed_state_outputs_equal"] = same
    (output/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    if reference and not result["exact_fixed_state_outputs_equal"]:
        raise RuntimeError("fixed-state route/validation outputs differ; inspect before proceeding")
    return result


def native_prefix(root):
    """One existing V3 arm, same full-day inputs, stop after 40 simulated minutes."""
    from . import bottleneck_whole_method as protocol
    from .scheme_a_two_scale_check import _run_native,FLEETPY
    config = Path("stage4/config/bottleneck_whole_method_v3.json")
    cfg,_,_ = protocol.load(root,config)
    group = "BOTTLENECK_SHARED_PLAN"
    cfg = protocol.group_config(cfg,group)
    directory = root/OUT/"native_prefix"/group
    if directory.exists():
        raise FileExistsError("preserve prior native prefix; no automatic overwrite/retry")
    cfg["diagnostic_stop_tick_s"] = 2400
    cfg["disk_route_cache"] = (directory.relative_to(root)/"routing/cache.sqlite3").as_posix()
    with protocol.execution_lock(root,protocol.output_root(cfg)):
        return _run_native(root,FLEETPY,config,cfg,directory=directory,
            doc_summary_path=directory/"prefix_receipt.json",completion_status="NOT_FULL_DAY",
            group=group,execution_binding=protocol.binding_for(root,config,cfg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=["before", "after","before_exact","after_exact","after_final","after_complete"])
    parser.add_argument("--reference", choices=["before","before_exact"])
    parser.add_argument("--case-set", choices=["initial", "av_loaded"], default="initial")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--native-prefix", action="store_true")
    args = parser.parse_args()
    if args.native_prefix:
        if args.label or args.reference or args.legacy:
            parser.error("native prefix is separate from fixed-state timing")
        native_prefix(Path.cwd())
    elif args.label:
        run(Path.cwd(),args.label,args.reference,args.case_set,args.legacy)
    else:
        parser.error("choose --label or the single --native-prefix check")
