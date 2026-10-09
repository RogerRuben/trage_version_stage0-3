"""Fixed three-group continuous-day Scheme-A mixed-fleet comparison.

No midnight-to-17:00 reset, scenario replay of future predictions, dense matrix,
or hidden one-next-service fallback. Completed groups are reused on resume;
failed experiments are never retried automatically.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa

from stage4.analysis.capability_chain_instances import atomic_json, atomic_parquet, sha
from stage4.analysis.city_joint_validation import code_sha, native_configuration, outcome_table, time_chain_check
from stage4.dispatch.controlled_routes import ControlledEmptyRouter
from stage4.dispatch.remaining_time import TrainRemainingTime
from stage4.dispatch.repositioning_policy import load_train_demand_reference
from stage4.dispatch.rolling_or_control import create_rolling_or_fleet_control
from stage4.dispatch.routing_v4 import StaticRawRoutingAdapter
from stage4.dispatch.runtime_assets import routing_context
from stage4.dispatch.city_pickup_support import CityPickupSupportValidator
from stage4.dispatch.scheme_a_city_adapter import NativeSchemeAAdapter, apply_av_layout, plan_av_layout
from stage4.dispatch.scheme_a_city_connections import SchemeACityConnectors, SelectedMovementManager
from stage4.dispatch.scheme_a_city_graph import CoarseHistoricalLibrary
from stage4.fleetpy_adapter.mixed_fleet_adapter import create_native_vehicles
from stage4.fleetpy_adapter.native_network import create_native_network
from stage4.fleetpy_adapter.research_native_bridge import ResearchNativeBridge
from stage4.fleetpy_adapter.research_world import load_research_episode, ORDER_BASE_REL, FLEET_REL
from stage4.fleetpy_adapter.upstream import CoordinateRegistry, load_fleetpy_bindings


CONFIG = Path("stage4/config/scheme_a_city_v1.json")
OUT = Path("stage4/output/capability_chain_planning/scheme_a_city_v1")
DOC = Path("stage4/docs/capability_chain_planning/scheme_a_city_v1")
FLEETPY = Path("D:/pycodes/didi_xian_raw/.external/FleetPy")
GROUPS = ("HOTSPOT_BASELINE", "CHAIN_LAYOUT_BASELINE", "CHAIN_LAYOUT_JOINT")


def protocol(root, config_path=CONFIG):
    cfg = json.loads((root / config_path).read_text(encoding="utf-8"))
    fixed = dict(test_date="20161031", profile_id="C", requested_q_a=.1,
        passenger_acceptance_rate=.7, planning_horizon_s=1800,
        coarse_reference_update_s=300, rolling_step_s=30, patience_s=300,
        solver_time_limit_s=10., maximum_model_variables=20000, maximum_model_nonzeros=150000,
        solver_fallback="NONE", dense_matrix=False, gpu_used=False, parameter_search=False, model_refit=False)
    if any(cfg.get(k) != v for k, v in fixed.items()) or tuple(cfg["groups"]) != GROUPS:
        raise ValueError("outside the fixed three-group Scheme-A protocol")
    protected = [config_path, ORDER_BASE_REL, FLEET_REL, Path(cfg["source_routes"]),
        Path(cfg["source_history_templates"]), Path(cfg["source_history_manifest"]), Path(cfg["remaining_time_model"]),
        Path(cfg["controlled_source_config"]), Path(cfg["frozen_profile_config"])]
    return cfg, {p.as_posix(): sha(root/p) for p in protected}


def load_history_templates(root, cfg):
    """Use the existing full-day history, never the old two-window library.

    The source producer records complete-day row counts. Matching those counts
    catches a truncated/window-only input without imposing a new demand or
    quality threshold. Only prediction/identity columns enter the planner.
    """
    source = Path(cfg["source_history_templates"])
    manifest_path = Path(cfg["source_history_manifest"])
    manifest = json.loads((root/manifest_path).read_text(encoding="utf-8"))
    if (manifest.get("status") != "COMPLETE"
        or manifest.get("full_day_Train_library_not_test31_future_demand") is not True):
        raise ValueError("Scheme-A requires a verified full-day historical template source, not local windows")
    dates = tuple(sorted(map(str, cfg["forecast_train_dates"])))
    declared = {str(row["date"]): int(row["templates"]) for row in manifest["train_templates"]}
    if not set(dates) <= set(declared):
        raise ValueError("full-day history manifest does not cover the configured historical dates")
    columns = list(CoarseHistoricalLibrary._COLUMNS) + ["research_data_ready"]
    templates = pd.read_parquet(root/source, columns=columns)
    templates = templates.loc[templates.date.astype(str).isin(dates)].copy()
    counts = {str(d): int(n) for d, n in templates.groupby(templates.date.astype(str)).size().items()}
    if counts != {d: declared[d] for d in dates}:
        raise ValueError("historical templates do not match complete-day source counts; window/truncated input rejected")
    if templates.duplicated(["date", "order_id"]).any():
        raise ValueError("duplicate historical source identity")
    inventory = dict(source=source.as_posix(), source_manifest=manifest_path.as_posix(),
        full_day_source_verified=True, total_rows=len(templates),
        prediction_only_projection=True, new_M3_inference=False, dates={})
    for date in dates:
        day = templates.loc[templates.date.astype(str).eq(date)]
        hours = (day.release_second//3600).astype(int).value_counts().sort_index()
        inventory["dates"][date] = dict(row_count=len(day), ready_row_count=int(day.research_data_ready.eq(True).sum()),
            release_min_s=float(day.release_second.min()), release_max_s=float(day.release_second.max()),
            hourly_row_counts={str(h): int(n) for h, n in hours.items()})
    return templates, inventory


def guard(root, cfg, started, *, phase="scenario", initial=False):
    process = psutil.Process()
    rss = process.memory_info().rss / 2**20
    available = psutil.virtual_memory().available / 2**20
    disk = psutil.disk_usage(root.anchor).free / 2**20
    maximum = cfg["layout_timeout_s"] if phase == "layout" else cfg["scenario_timeout_s"]
    if perf_counter()-started > maximum:
        raise TimeoutError(f"fixed {phase} wall-time limit exceeded")
    if rss > cfg["maximum_rss_mib"]:
        raise MemoryError(f"Scheme-A RSS {rss:.1f} MiB exceeds fixed1536MiB cap")
    if initial and available < cfg["minimum_initial_available_ram_mib"]:
        raise MemoryError("insufficient available RAM; no other applications will be closed")
    if disk < cfg["minimum_disk_reserve_mib"]:
        raise OSError("disk reserve below fixed minimum")
    return dict(rss_mib=rss, available_ram_mib=available, free_disk_mib=disk)


def make_routing(root, cfg):
    return StaticRawRoutingAdapter(root, routing_mode=cfg["routing_mode"], route_workers=1,
        persistent_cache_size=cfg["persistent_cache_size"], static_raw_od=True,
        raw_od_cache_size=cfg["raw_od_cache_size"], certified_eta_pruning=True,
        disk_cache_path=str(root/cfg["disk_route_cache"]), routing_context=routing_context(root),
        disk_cache_limit_mib=cfg["disk_cache_limit_mib"], disk_cache_max_entries=cfg["disk_cache_max_entries"],
        route_queue_chunk_size=cfg["route_queue_chunk_size"], grouped_sources=True, lazy_worker_actor=True)


def _clock_cfg(episode, cfg):
    last_release = max(r.release_time_s for r in episode.requests._requests)
    measurement = int(np.floor(last_release))+1
    admission = int(np.ceil((last_release+cfg["patience_s"])/30))*30
    if measurement < 86400 or admission > cfg["physical_drain_limit_s"]:
        raise ValueError("unexpected release span outside fixed daily protocol")
    return dict(cfg, window_start_s=0, measurement_end_s=measurement,
        admission_end_s=admission, last_dispatch_s=admission,
        process_group_memory_limit_mib=cfg["maximum_rss_mib"])


def archive_failed_layout(root):
    base = (root/OUT).resolve()
    source = (base/"layout").resolve()
    source.relative_to(base)
    status = json.loads((source/"status.json").read_text(encoding="utf-8"))
    if status["status"] != "STOPPED":
        raise ValueError("only STOPPED layout can be archived for explicit retry")
    index = 1
    while (base/"failed_attempts"/f"layout_{index:03d}").exists():
        index += 1
    destination = (base/"failed_attempts"/f"layout_{index:03d}").resolve()
    destination.relative_to(base)
    checkpoint = source/"partial_layout_checkpoint.json"
    restored = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source),str(destination))
    return destination, restored


def prepare_layout(root, cfg, protected, *, resume=False, retry_layout=False):
    destination = root/OUT/"layout"
    path = destination/"placements.json"
    archived, restored = None, None
    if retry_layout:
        archived, recovered = archive_failed_layout(root)
        if recovered is not None and recovered["inputs_sha256"] == protected:
            restored = recovered["partial"]
    if path.exists():
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if not resume or receipt["status"] != "LAYOUT_COMPLETE" or receipt["inputs_sha256"] != protected:
            raise ValueError("existing layout requires same-protocol completed resume; no overwrite")
        return receipt
    if (destination/"status.json").exists():
        raise ValueError("layout attempt exists but is not complete; no automatic failed-attempt retry")
    started = perf_counter()
    guard(root, cfg, started, phase="layout", initial=True)
    receipt = dict(status="RUNNING", phase="EARLIER_HISTORY_INITIAL_LAYOUT",
        code_sha=code_sha(root), inputs_sha256=protected,
        explicitly_authorized_failed_layout_retry=retry_layout,
        archived_failure=str(archived.relative_to(root)) if archived is not None else None,
        restored_completed_bins=int(restored["completed_bins"]) if restored is not None else 0)
    atomic_json(destination/"status.json", receipt)
    routing = None
    try:
        episode = load_research_episode(root, profile_id="C", requested_q_a=.1)
        clock_cfg = _clock_cfg(episode, cfg)
        start = pd.Timestamp("2016-10-31T00:00:00+08:00")
        templates, history_inventory = load_history_templates(root, cfg)
        reference, reference_manifest = load_train_demand_reference(root)
        routing = make_routing(root, cfg)
        pseudo = SimpleNamespace(eta_adapter=routing, request_by_rid={}, runtime_by_vid={},
            assignment_rows=[], completed_rids=set(), config=dict(scheme_a_fast_forecast_hv=True),
            _timestamp=lambda seconds: start+pd.Timedelta(seconds=float(seconds)))
        empty_router = ControlledEmptyRouter(root, routing)
        validator = CityPickupSupportValidator(empty_router, geometry_timestamp=start,
            routing_budget_s=cfg["layout_timeout_s"])
        connectors = SchemeACityConnectors(pseudo, empty_router, validator, templates, geometry_timestamp=start)
        library = CoarseHistoricalLibrary(templates, clock_cfg)
        del templates

        def check():
            guard(root, cfg, started, phase="layout")

        def progress(done, total, placed):
            info = dict(status="PLACING_AV", bins_done=done, bins_total=total,
                placed_AV_slots=placed, runtime_s=perf_counter()-started,
                **guard(root, cfg, started, phase="layout"))
            atomic_json(destination/"progress.json", info)
            print(json.dumps(info), flush=True)

        def checkpoint(partial):
            atomic_json(destination/"partial_layout_checkpoint.json", dict(
                status="PARTIAL_LAYOUT_NOT_AN_OPERATING_RESULT", code_sha=code_sha(root),
                inputs_sha256=protected, partial=partial))

        def diagnostics(info):
            atomic_json(destination/"failed_graph_diagnostics.json", info)

        positions = plan_av_layout(episode, library, reference, connectors, clock_cfg, check, progress,
            checkpoint=checkpoint, diagnostics_sink=diagnostics, restored=restored)
        if len(positions["hotspot"]) != 851 or set(positions["joint"]) != set(positions["hotspot"]):
            raise RuntimeError("initial layout does not cover the identical851AV slots")
        receipt.update(status="LAYOUT_COMPLETE", clock_cfg=clock_cfg, placements=positions,
            runtime_s=perf_counter()-started, connectors=connectors.diagnostics(),
            library=library.diagnostics(), reference_manifest=reference_manifest,
            history_template_inventory=history_inventory,
            peak_rss_mib=psutil.Process().memory_info().peak_wset/2**20,
            all_day_supply=episode.supply.inventory(), new_M3_inference=False,
            future_target_requests_or_outcomes_used=False)
        atomic_json(path, receipt)
        atomic_json(destination/"status.json", {k:v for k,v in receipt.items() if k != "placements"})
        return receipt
    except Exception as error:
        receipt.update(status="STOPPED", error=repr(error), runtime_s=perf_counter()-started)
        atomic_json(destination/"status.json", receipt)
        raise
    finally:
        if routing is not None:
            routing.close()


def run_group(root, cfg, protected, layout, group, *, resume=False):
    directory = root/OUT/group
    summary_path = directory/"summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if (resume and existing["status"] == "SCHEME_A_CITY_GROUP_COMPLETE"
            and existing["inputs_sha256"] == protected and existing["layout_sha256"] == sha(root/OUT/"layout/placements.json")):
            print(json.dumps(dict(reused_completed_group=group, new_run=False)), flush=True)
            return existing
        raise ValueError("existing group is not a same-protocol completed group; no implicit rerun")
    started = perf_counter()
    guard(root, cfg, started, initial=True)
    policy = "CHAIN_DEFER" if group == "CHAIN_LAYOUT_JOINT" else "SERVICE_PRESERVING"
    layout_mode = "hotspot" if group == "HOTSPOT_BASELINE" else "joint"
    summary = dict(status="RUNNING", group=group, policy=policy, layout_mode=layout_mode,
        code_sha=code_sha(root), inputs_sha256=protected, frozen_protocol=cfg,
        layout_sha256=sha(root/OUT/"layout/placements.json"), full_day=True,
        initial_position_semantics="ORIGINAL_HV_START_AND_FROZEN_EARLIER_HISTORY_AV_LAYOUT_FROM_MIDNIGHT",
        gpu_used=False, dense_matrix=False, city_global_optimality_claimed=False)
    atomic_json(summary_path, summary)
    routing = bridge = control = adapter = manager = None
    tick = None
    try:
        episode = load_research_episode(root, profile_id="C", requested_q_a=.1)
        clock_cfg = _clock_cfg(episode, cfg)
        apply_av_layout(episode, layout["placements"][layout_mode])
        start = pd.Timestamp("2016-10-31T00:00:00+08:00")
        bindings = load_fleetpy_bindings(FLEETPY)
        registry = CoordinateRegistry()
        bridge = ResearchNativeBridge(episode, bindings, registry, window_start_s=0,
            measurement_end_s=clock_cfg["measurement_end_s"], admission_end_s=clock_cfg["admission_end_s"], include_carry_in=False)
        if bridge.input_counts()["window_new_requests"] != len(episode.requests._requests):
            raise RuntimeError("full-day native bridge dropped original common requests")
        network = create_native_network(bindings, registry)
        demand = bridge.create_demand(network, directory)
        fixtures = bridge.native_fixtures()
        vehicles, native_output = create_native_vehicles(fixtures, bindings, registry,
            demand.rq_db, directory/"runtime", native_movement=True, routing_engine=network)
        routing = make_routing(root, cfg)
        native_cfg = native_configuration(root, clock_cfg)
        native_cfg["scheme_a_fast_forecast_hv"] = True
        control = create_rolling_or_fleet_control(bindings, vehicles, [], demand, network, routing,
            start, start+pd.Timedelta(seconds=cfg["physical_drain_limit_s"]), native_cfg)
        control.enable_event_calendar()
        reference, _ = load_train_demand_reference(root)
        empty_router = ControlledEmptyRouter(root, routing)
        manager = SelectedMovementManager(control, reference, empty_router,
            max_moves=50, radius_m=2000., max_eta_s=300., top_k=3, day_end_s=cfg["movement_day_end_s"])
        control.repositioning_manager = manager
        bridge.install(control)
        validator = CityPickupSupportValidator(empty_router, geometry_timestamp=start,
            routing_budget_s=cfg["scenario_timeout_s"])
        scope = [r.order_id for r in episode.requests._requests if "C" in r.compatible_profiles]
        summary["private_feature_store_setup"] = validator.prepare_private_identity_store(scope)
        del scope
        templates, history_inventory = load_history_templates(root, cfg)
        library = CoarseHistoricalLibrary(templates, clock_cfg)
        connectors = SchemeACityConnectors(control, empty_router, validator, templates, geometry_timestamp=start)
        del templates
        remaining = TrainRemainingTime(json.loads((root/cfg["remaining_time_model"]).read_text(encoding="utf-8")))
        adapter = NativeSchemeAAdapter(policy, library, reference, connectors, validator, remaining, clock_cfg)
        control.flexibility_adapter = adapter
        simulation = bridge.create_simulation(simulation_end_s=cfg["physical_drain_limit_s"],
            time_step_s=30, vehicles=vehicles, fleet_control=control, network=network, native_output=native_output)
        summary.update(native_input_counts=bridge.input_counts(), all_day_supply=episode.supply.inventory(),
            history_template_inventory=history_inventory)
        atomic_json(summary_path, summary)
        for tick in range(0, cfg["physical_drain_limit_s"]+30, 30):
            resources = guard(root, cfg, started)
            simulation.step(tick)
            routing.cache.clear()
            if tick % 900 == 0:
                progress = dict(group=group, tick=tick, matched=len(control.assignment_rows),
                    completed=len(control.completed_rids), expired=len(control.expired_rids),
                    runtime_s=perf_counter()-started, **resources)
                atomic_json(directory/"progress.json", progress)
                print(json.dumps(progress), flush=True)
            if tick > clock_cfg["admission_end_s"] and not (
                set(control.rid_to_assigned_vid)-control.completed_rids) and not manager.active:
                break
        control.reconcile()
        if any((control.position_reconciliation_failures, control.request_state_reconciliation_failures,
            control.vehicle_state_reconciliation_failures, control.av_availability_violations)):
            raise RuntimeError("native state reconciliation failed")
        if set(control.rid_to_assigned_vid)-control.completed_rids:
            raise RuntimeError("physical committed-work drain incomplete")
        assignments = bridge.environment_assignments()
        physics = time_chain_check(assignments)
        outcomes = outcome_table(control, assignments, clock_cfg)
        traces = pd.DataFrame(adapter.rows)
        epochs = pd.DataFrame(control.epoch_rows)
        movements = pd.DataFrame(manager.rows)
        if len(movements):
            movements["destination_position"] = movements.destination_position.astype(str)
        for name, frame in (("assignments", assignments), ("cohort_outcomes", outcomes),
            ("solver_trace", traces), ("epochs", epochs), ("empty_movements", movements)):
            atomic_parquet(directory/f"{name}.parquet", frame)
        if protected != {p: sha(root/p) for p in protected}:
            raise RuntimeError("frozen inputs changed during group")
        summary.update(status="SCHEME_A_CITY_GROUP_COMPLETE", runtime_s=perf_counter()-started,
            peak_rss_mib=psutil.Process().memory_info().peak_wset/2**20, drain_end_s=tick,
            common_requests=len(outcomes), matched=int(outcomes.matched.sum()), completed=int(outcomes.completed.sum()),
            expired=int(outcomes.expired.sum()), service_rate=float(outcomes.completed.mean()),
            served_C=int(outcomes.vehicle_type.eq("AV").sum()), served_HV=int(outcomes.vehicle_type.eq("HV").sum()),
            served_wait_mean_s=float(outcomes.wait_s.mean()), pickup_empty_distance_m=float(assignments.pickup_route_distance_m.sum()),
            idle_movement=manager.summary(), physical_accounting=physics,
            maximum_model_variables=int(traces.variables.max()), maximum_model_nonzeros=int(traces.nonzeros.max()),
            maximum_total_or_s=float(traces.total_or_time_s.max()), OR_total_s=float(traces.total_or_time_s.sum()),
            selected_multi_service_chain_records=int(traces.selected_multi_service_chains.sum()),
            selected_future_move_records=int(traces.selected_future_move_activities.sum()),
            connector=connectors.diagnostics(), library=library.diagnostics(), current_pickup=validator.diagnostics(),
            execution_bridge=dict(bridge.diagnostics(), initial_position_source=summary["initial_position_semantics"]),
            actual_routing=routing.diagnostics(), current_candidate_time_s=float(epochs.candidate_generation_time_s.sum()),
            current_routing_time_s=float(epochs.routing_time_s.sum()), realized_truth_only_after_commitment=True,
            prediction="FROZEN_M3_P50_AND_FROZEN_EARLIER_HISTORY_REMAINING_MODEL_NO_REFIT")
        atomic_json(summary_path, summary)
        atomic_json(root/DOC/f"{group.lower()}_summary.json", summary)
        print(json.dumps(dict(group=group, status=summary["status"], completed=summary["completed"], runtime_s=summary["runtime_s"])), flush=True)
        return summary
    except Exception as error:
        summary.update(status="STOPPED", error=repr(error), tick=tick, runtime_s=perf_counter()-started)
        if bridge is not None and control is not None:
            atomic_parquet(directory/"partial_assignments.parquet", bridge.environment_assignments())
        if adapter is not None:
            atomic_parquet(directory/"partial_solver_trace.parquet", pd.DataFrame(adapter.rows))
        atomic_json(summary_path, summary)
        raise
    finally:
        if routing is not None:
            routing.close()


def run(root, *, config_path=CONFIG, resume=False, prepare_only=False, retry_layout=False):
    root = Path(root).resolve()
    cfg, protected = protocol(root, config_path)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    templates, history_inventory = load_history_templates(root, cfg)
    del templates
    atomic_json(root/DOC/"history_template_inventory.json", history_inventory)
    layout = prepare_layout(root, cfg, protected, resume=resume, retry_layout=retry_layout)
    gc.collect()
    if prepare_only:
        return layout
    results = []
    for group in GROUPS:
        results.append(run_group(root, cfg, protected, layout, group, resume=resume))
        gc.collect()
    result = dict(status="SCHEME_A_THREE_GROUP_CITY_COMPARISON_COMPLETE", inputs_sha256=protected,
        frozen_protocol=cfg, history_template_inventory=history_inventory,
        groups=results, full_day_runs=3, no_parameter_search=True,
        no_extra_conditions=True, same_joint_layout_for_groups_2_3=True,
        restricted_chain_candidate_heuristic=True, city_global_optimality_claimed=False,
        gpu_used=False, dense_matrix=False)
    atomic_json(root/OUT/"summary.json", result)
    atomic_json(root/DOC/"summary.json", result)
    print(json.dumps(dict(status=result["status"], served={g["group"]:g["completed"] for g in results})), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--retry-layout", action="store_true",
        help="archive STOPPED layout only after explicit user approval; preserve completed same-input prefix")
    args = parser.parse_args()
    run(args.root, config_path=args.config, resume=args.resume, prepare_only=args.prepare_only, retry_layout=args.retry_layout)


if __name__ == "__main__":
    main()
