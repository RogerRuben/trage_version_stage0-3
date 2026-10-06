"""Bounded same-state raw-OD reuse/ETA checks, never a native replay."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa

from stage3.scripts.traffic_state_batch1 import sha, write_json
from stage4.analysis.acceleration_benchmark import vector
from stage4.analysis.acceleration_v3_prerun import STATES, snapshot
from stage4.analysis.symmetric_flexibility_full_day import acceleration_settings
from stage4.analysis.symmetric_research_prepare import CONFIG
from stage4.dispatch.deterministic_routing import SINGLE_SOURCE_MATRIX
from stage4.dispatch.flexibility_model import CurrentServiceFace
from stage4.dispatch.flexibility_v3 import solve_dispatch_v3
from stage4.dispatch.routing_v3 import DemandRoutingAdapter
from stage4.dispatch.routing_v4 import PickupEtaBudget, StaticRawRoutingAdapter
from stage4.dispatch.runtime_assets import RuntimeAssets, routing_context
from stage4.dispatch.solver import solve_lexicographic
from stage4.fleetpy_adapter.valhalla_time_adapter import CALIBRATION_REL

DOC = Path("stage4/docs/flexibility_dispatch/acceleration_v4")
OUT = Path("stage4/output/runtime_acceleration_v4/prerun")
TECHNICAL = Path("stage4/config/symmetric_flexibility_acceleration_v4.json")
VARIANTS = ("V3_DEFAULT", "V4_RAW_OD", "V4_RAW_OD_GROUPED")
OFFSETS = (0, 60, 900)  # Same fixed coordinates, not physical time advancement.


def all_values(router, estimates):
    result = []
    pruned = getattr(router, "last_pruned_estimates", [{} for _ in estimates])
    for row, removed in zip(estimates, pruned):
        combined = {**row, **removed}
        result.append({vid: (e.valhalla_time_s, e.corrected_pickup_eta_s,
            e.route_distance_m, e.beta, e.time_bin_index) for vid, e in combined.items()})
    return result


def eligible_values(estimates, budgets):
    return [{vid: e.corrected_pickup_eta_s for vid, e in row.items()
             if budgets[i][vid].rejection(e.corrected_pickup_eta_s) is None}
            for i, row in enumerate(estimates)]


def rss_mib():
    process = psutil.Process()
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total / 2**20


def known_eta_check():
    """One already-used physical state; cover the cached admission branch."""
    root = Path.cwd()
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    cfg = json.loads((root / CONFIG).read_text())
    assets = RuntimeAssets(root / "stage4/output/runtime_acceleration_v2/assets")
    if assets.manifest["routing_context"] != routing_context(root):
        raise RuntimeError("routing context changed before the cached-ETA check")
    runtime = json.loads((root / "stage4/output/final_experiments" / cfg["source_scenario"] / "scenario_config.json").read_text())["runtime_configuration"]
    source = root / "stage4/output/symmetric_flexibility_v1/accelerated_full_day/SERVICE_PRESERVING_LOOKAHEAD/assignments.parquet"
    started = time.monotonic()
    p, _, batches, info = snapshot(root, assets, pd.read_parquet(source), cfg, runtime, STATES[0])
    batches = batches[:20]
    requests = list(p.waiting_requests)[:len(batches)]
    windows = assets.windows()
    policies = {f.native_id: f.availability_policy for f in assets.fleet().native_fixtures}
    budgets = [{v.native_vehicle_id: PickupEtaBudget(r.pickup_deadline_s - p.now_s, p.now_s,
        r.predicted_service_time_s + r.pickup_overhead_s, windows[v.native_vehicle_id][1]
        if policies[v.native_vehicle_id] == "EMPIRICAL_SESSION" else None) for v in b[0]}
        for r, b in zip(requests, batches)]
    router = StaticRawRoutingAdapter(root, routing_mode=SINGLE_SOURCE_MATRIX, route_workers=2,
        persistent_cache_size=50000, lazy_worker_actor=True, grouped_sources=True)
    try:
        cold = router.estimate_epoch(batches)
        reference = eligible_values(cold, budgets)
        # Prove that the separate raw cache, rather than either old ETA cache,
        # supplies the known values. No physical state/time advance occurs.
        router.cache.clear(); router._persistent_cache.clear()
        q0, a0 = router.routing_queries, router.routing_arc_evaluations
        warm = router.estimate_epoch(batches, eta_budgets=budgets)
        same = eligible_values(warm, budgets) == reference
        if not same or router.certified_eta_prunes == 0:
            raise RuntimeError("cached admission branch did not preserve the known sparse graph")
        result = dict(status="COMPLETE", scope="ONE_EXISTING_ASOF_STATE_CACHED_ETA_API_CHECK",
            tick=info["tick"], route_batches=len(batches), logical_arcs=sum(len(b[0]) for b in batches),
            valid_arcs_before=sum(len(x) for x in reference), valid_arcs_after=sum(len(x) for x in eligible_values(warm, budgets)),
            valid_arc_values_equal=same, warm_backend_queries=router.routing_queries - q0,
            warm_raw_arcs=router.routing_arc_evaluations - a0, diagnostics=router.diagnostics(),
            runtime_s=time.monotonic() - started, process_group_rss_snapshot_mib=rss_mib(),
            native_scientific_conditions_added=0, physical_state_advanced=False,
            unknown_od_queries_were_not_pruned=True, independent_pruning_speedup_not_claimed=True)
    finally:
        router.close(); router.actor = None
    (root / DOC).mkdir(parents=True, exist_ok=True)
    write_json(root / DOC / "known_eta_check.json", result)
    print(json.dumps(result), flush=True)


def main():
    root = Path.cwd()
    (root / DOC).mkdir(parents=True, exist_ok=True)
    (root / OUT).mkdir(parents=True, exist_ok=True)
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    for process in psutil.process_iter(["cmdline"]):
        try:
            if process.pid != psutil.Process().pid and "stage4.analysis.symmetric_flexibility_full_day" in " ".join(process.info["cmdline"] or []):
                raise RuntimeError("bounded pre-run cannot overlap a native full-day run")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    technical, _, _ = acceleration_settings(root, TECHNICAL)
    cfg = json.loads((root / CONFIG).read_text())
    assets = RuntimeAssets(root / "stage4/output/runtime_acceleration_v2/assets")
    if assets.manifest["routing_context"] != routing_context(root):
        raise RuntimeError("frozen routing context differs from the offline assets")
    runtime = json.loads((root / "stage4/output/final_experiments" / cfg["source_scenario"] / "scenario_config.json").read_text())["runtime_configuration"]
    source = root / "stage4/output/symmetric_flexibility_v1/accelerated_full_day/SERVICE_PRESERVING_LOOKAHEAD/assignments.parquet"
    assignments = pd.read_parquet(source)
    windows = assets.windows()
    policies = {f.native_id: f.availability_policy for f in assets.fleet().native_fixtures}
    order_rows = {int(r["native_id"]): r for r in assets.orders}
    protected_paths = (CONFIG, TECHNICAL, CALIBRATION_REL,
        Path("stage3/config/stage3_av_capability_profiles.json"),
        Path("stage2/output_v5_2/development/M3/epoch_004.pt"),
        Path("stage4/output/runtime_acceleration_v2/assets/manifest.json"))
    protected = {str(p): sha(root / p) for p in protected_paths}
    source_sha = sha(source)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Singapore")).isoformat()
    peak_group = rss_mib()
    rows = []

    def guard():
        nonlocal peak_group
        peak_group = max(peak_group, rss_mib())
        if peak_group > 2048:
            raise RuntimeError("bounded pre-run process group exceeded 2048 MiB")
        if time.monotonic() - started > 300:
            raise RuntimeError("bounded pre-run exceeded its 300-second administrative budget")

    for number, now in enumerate(STATES):
        guard()
        p, arcs, batches, info = snapshot(root, assets, assignments, cfg, runtime, now)
        batches = batches[:20]
        # Snapshot order is deterministic and matches its waiting-request order.
        waiting = list(p.waiting_requests)
        subset_ids = [r.request_id for r in waiting[:len(batches)]]
        if len(subset_ids) != len(batches):
            raise RuntimeError("waiting route batch alignment changed")
        budgets = []
        for rid, batch in zip(subset_ids, batches):
            r = order_rows[rid]
            budgets.append({v.native_vehicle_id: PickupEtaBudget(float(r["release"]) + cfg["patience_s"] - now,
                now, float(r["predicted"]), windows[v.native_vehicle_id][1]
                if policies[v.native_vehicle_id] == "EMPIRICAL_SESSION" else None) for v in batch[0]})
        results, reference, expected_valid = {}, None, None
        # Alternate benchmark order; each adapter uses its own cold bounded LRU.
        order = VARIANTS if number % 2 == 0 else tuple(reversed(VARIANTS))
        for name in order:
            guard()
            router = (DemandRoutingAdapter(root, routing_mode=SINGLE_SOURCE_MATRIX, route_workers=2,
                        persistent_cache_size=50000, lazy_worker_actor=True, grouped_sources=False)
                      if name == "V3_DEFAULT" else StaticRawRoutingAdapter(root,
                        routing_mode=SINGLE_SOURCE_MATRIX, route_workers=2, persistent_cache_size=50000,
                        lazy_worker_actor=True, grouped_sources=name == "V4_RAW_OD_GROUPED",
                        static_raw_od=technical["static_raw_od"], raw_od_cache_size=technical["raw_od_cache_size"],
                        certified_eta_pruning=technical["certified_eta_pruning"]))
            try:
                if batches:
                    router.estimate_epoch(batches[:1])
                    router.cache.clear(); router._persistent_cache.clear()
                    if hasattr(router, "_raw_od_cache"):
                        router._raw_od_cache.clear()
                base_arcs, base_queries = router.routing_arc_evaluations, router.routing_queries
                base_lookups = router.arc_lookup_count
                passes, cold_valid = [], None
                for offset in OFFSETS:
                    router.cache.clear()
                    shifted = [(vs, lon, lat, stamp + pd.Timedelta(seconds=offset)) for vs, lon, lat, stamp in batches]
                    arc0, query0 = router.routing_arc_evaluations, router.routing_queries
                    t = time.perf_counter()
                    estimates = (router.estimate_epoch(shifted, eta_budgets=budgets)
                                 if offset == 0 and name != "V3_DEFAULT" else router.estimate_epoch(shifted))
                    elapsed = time.perf_counter() - t
                    full = all_values(router, estimates)
                    if reference is None:
                        # First variant may be v4: all variants must still agree.
                        reference = {}
                    if offset not in reference:
                        reference[offset] = full
                    elif full != reference[offset]:
                        raise RuntimeError(f"routing answers changed: {name}, state={now}, offset={offset}")
                    if offset == 0:
                        cold_valid = eligible_values(estimates, budgets)
                        if expected_valid is None:
                            expected_valid = cold_valid
                        elif cold_valid != expected_valid:
                            raise RuntimeError("ETA certificates changed the valid sparse candidate graph")
                    passes.append(dict(offset_s=offset, wall_s=elapsed, raw_arcs=router.routing_arc_evaluations - arc0,
                        backend_queries=router.routing_queries - query0, exact_answers_equal=True,
                        cold_actual_state=offset == 0, physical_state_advanced=False))
                    guard()
                results[name] = dict(passes=passes, wall_s=sum(x["wall_s"] for x in passes),
                    raw_arcs=router.routing_arc_evaluations - base_arcs,
                    backend_queries=router.routing_queries - base_queries,
                    arc_lookups=router.arc_lookup_count - base_lookups,
                    valid_cold_arcs=sum(len(r) for r in cold_valid),
                    process_group_rss_snapshot_mib=rss_mib(),
                    diagnostics=router.diagnostics() if hasattr(router, "diagnostics") else None)
            finally:
                router.close(); router.actor = None
                gc.collect()
        # Replace only the checked subset, preserving all other original arcs.
        checked = {(vid, rid): eta for rid, found in zip(subset_ids, expected_valid) for vid, eta in found.items()}
        original = {(cp.vehicle_id, cp.request_id): cp.pickup_eta_s for cp in p.current_pickups if cp.request_id in subset_ids}
        if checked != original:
            raise RuntimeError("bounded fresh routing did not reproduce the snapshot's original valid arcs")
        current = tuple(replace(cp, pickup_eta_s=checked[cp.vehicle_id, cp.request_id])
                        if cp.request_id in subset_ids else cp for cp in p.current_pickups)
        oracle = solve_lexicographic(arcs)
        face = CurrentServiceFace.from_arcs(arcs, oracle)
        selection = tuple((arcs[i].vehicle_id, arcs[i].request_id) for i in oracle.selected_indices)
        objective_rows = []
        for case in (p, replace(p, current_pickups=current)):
            t = time.perf_counter()
            d = solve_dispatch_v3(case, "SERVICE_PRESERVING_LOOKAHEAD", current_face=face, current_selection=selection)
            objective_rows.append(dict(wall_s=time.perf_counter() - t, objective=vector(case, d),
                variables=d.variable_count, nonzeros=d.constraint_nonzeros))
        if not np.allclose(objective_rows[0]["objective"], objective_rows[1]["objective"], rtol=0, atol=1e-6):
            raise RuntimeError("v4 routing changed a lexicographic model objective")
        row = dict(**info, limited_route_batches=len(batches), variants=results,
            checked_cold_valid_arcs=len(checked), valid_arcs_equal=True, all_objectives_equal=True,
            solver_check=objective_rows)
        rows.append(row)
        write_json(root / OUT / "progress.json", dict(completed_states=len(rows), total_states=len(STATES), latest=row,
            runtime_s=time.monotonic() - started, peak_process_group_rss_snapshot_mib=peak_group))
        print(json.dumps(row), flush=True)
        gc.collect()
    unchanged = all(sha(root / p) == digest for p, digest in protected.items()) and sha(source) == source_sha
    if not unchanged:
        raise RuntimeError("protected inputs changed during the bounded pre-run")
    totals = {name: {key: sum(row["variants"][name][key] for row in rows)
        for key in ("wall_s", "raw_arcs", "backend_queries", "arc_lookups")} for name in VARIANTS}
    result = dict(status="COMPLETE", scope="SIX_ASOF_V1_STATES_PLUS_FIXED_COORDINATE_CLOCK_API_DIAGNOSTIC",
        started_at=started_at, finished_at=datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Singapore")).isoformat(),
        runtime_s=time.monotonic() - started, states=rows, totals=totals,
        original_clock_cold_queries_are_real_state=True, shifted_clock_passes_are_not_native_replays=True,
        offsets_s=list(OFFSETS), native_scientific_conditions_added=0,
        future_completion_times_used_for_prediction=False, gpu_used=False, dense_matrix=False,
        protected_inputs_unchanged=unchanged, inputs_sha256=protected, source_assignments_sha256=source_sha,
        peak_parent_rss_mib=psutil.Process().memory_info().peak_wset / 2**20,
        peak_process_group_rss_snapshot_mib=peak_group,
        query_reuse_speedup_is_not_full_day_speedup=True)
    write_json(root / DOC / "prerun.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--known-eta-check", action="store_true")
    args = parser.parse_args()
    known_eta_check() if args.known_eta_check else main()
