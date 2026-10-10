"""One fixed citywide native short-window comparison, never a full-day run.

Historical future templates are predictions. Only released requests are given
to native control. Actual loaded durations are queried by the environment after
commitment; they are not copied into decision records or forecast construction.
"""
from __future__ import annotations

import argparse
import gc
import json
from math import isfinite
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

import pandas as pd
import psutil
import pyarrow as pa

from stage4.analysis.capability_chain_instances import atomic_json, atomic_parquet, sha
from stage4.dispatch.controlled_movement import CommonIdleMovementManager
from stage4.dispatch.controlled_routes import ControlledEmptyRouter
from stage4.dispatch.flexibility_native import TrainDemandForecast
from stage4.dispatch.remaining_time import TrainRemainingTime
from stage4.dispatch.repositioning_policy import load_train_demand_reference
from stage4.dispatch.rolling_or_control import create_rolling_or_fleet_control
from stage4.dispatch.routing_v4 import StaticRawRoutingAdapter
from stage4.dispatch.runtime_assets import routing_context
from stage4.fleetpy_adapter.mixed_fleet_adapter import VehicleFixture, create_native_vehicles
from stage4.fleetpy_adapter.native_network import create_native_network
from stage4.fleetpy_adapter.research_world import load_research_episode, ORDER_BASE_REL, FLEET_REL
from stage4.fleetpy_adapter.upstream import CoordinateRegistry, load_fleetpy_bindings

CONFIG = Path("stage4/config/city_joint_validation_v1.json")
OUT = Path("stage4/output/city_joint_validation_v1")
DOC = Path("stage4/docs/capability_chain_planning/city_joint_v1")
FLEETPY = Path("D:/pycodes/didi_xian_raw/.external/FleetPy")


def code_sha(root):
    return subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}",
                                    "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def load_protocol(root, path=CONFIG):
    cfg = json.loads((root / path).read_text(encoding="utf-8"))
    fixed = {
        "version": "city_joint_validation_v1", "test_date": "20161031", "profile_id": "C",
        "requested_q_a": .1, "passenger_acceptance_rate": .7,
        "window_start_s": 61200, "measurement_end_s": 63000, "last_dispatch_s": 63300,
        "physical_drain_limit_s": 70200, "rolling_step_s": 30, "patience_s": 300,
        "solver_time_limit_s": 10., "max_model_variables": 20000, "max_model_nonzeros": 150000,
        "solver_fallback": "NONE", "full_day": False, "dense_matrix": False,
        "gpu_used": False, "parameter_search": False, "model_refit": False,
    }
    if any(cfg.get(k) != v for k, v in fixed.items()):
        raise ValueError("outside the fixed city short-window protocol")
    if cfg["policies"] != ["LOCATION_AWARE_DEFER", "TIME_TYPE_DEFER"]:
        raise ValueError("only the two predeclared wait-enabled controls are authorized")
    prepared_reference = json.loads((root / cfg["generic_reference_path"]).read_text(encoding="utf-8"))
    reference = next(c["reference"] for c in prepared_reference["cases"]
                     if c["case_id"] == cfg["generic_reference_case"])
    if reference["target_day_actual_orders_used"] or not isfinite(reference["travel_time_s"]):
        raise ValueError("generic reference must be the previously fixed historical proxy")
    protected_paths = [path, ORDER_BASE_REL, FLEET_REL, Path(cfg["source_routes"]),
        Path(cfg["source_history_templates"]), Path(cfg["remaining_time_model"]),
        Path(cfg["generic_reference_path"]), Path(cfg["controlled_source_config"]),
        Path(cfg["frozen_profile_config"])]
    protected = {str(p.as_posix()): sha(root / p) for p in protected_paths}
    return cfg, reference, protected


def resource_guard(root, cfg, *, initial=False):
    process = psutil.Process()
    ram = psutil.virtual_memory().available / 2**20
    disk = psutil.disk_usage(root.anchor).free / 2**20
    rss = process.memory_info().rss / 2**20
    if initial and ram < cfg["minimum_initial_available_ram_mib"]:
        raise MemoryError(f"available RAM {ram:.1f} MiB below fixed starting reserve")
    if rss > cfg["process_group_memory_limit_mib"]:
        raise MemoryError(f"single-process RSS {rss:.1f} MiB exceeds fixed memory limit")
    if disk < cfg["minimum_disk_reserve_mib"]:
        raise OSError(f"disk reserve {disk:.1f} MiB below fixed minimum")
    return dict(rss_mib=rss, available_ram_mib=ram, disk_free_mib=disk)


def native_fixtures(episode, start, cfg):
    """Instantiate only relevant slots, without relabeling or clipping windows."""
    rows = [p for p in episode.supply.plan()
            if p.activation_s < cfg["last_dispatch_s"] and p.admission_end_s > cfg["window_start_s"]]
    return [VehicleFixture(p.vehicle_id, p.native_id, p.vehicle_type,
        p.initial_lon_wgs84, p.initial_lat_wgs84,
        start + pd.Timedelta(seconds=p.activation_s),
        start + pd.Timedelta(seconds=p.admission_end_s), p.slot_id, False,
        availability_policy="STOP_ADMISSION_FINISH_COMMITTED") for p in rows]


class CommonBodyPolicy:
    """Reuse the existing common-input/body compatibility, no second soft cap."""
    def evaluate(self, request):
        return dict(research_base_eligible=True,
            research_route_compatible=request.profile_id in request.compatible_profiles,
            research_data_ready=True, research_exposure_available=False,
            research_control_assumption_count=0, research_bearing_fallback_count=0)


def native_configuration(root, cfg):
    source = root / "stage4/output/final_experiments" / cfg["source_scenario"] / "scenario_config.json"
    base = json.loads(source.read_text(encoding="utf-8"))["runtime_configuration"]
    return {**base, "profile_id": cfg["profile_id"], "av_vehicle_hour_share": cfg["requested_q_a"],
        "passenger_acceptance_rate": cfg["passenger_acceptance_rate"],
        "passenger_acceptance_seed": cfg["passenger_acceptance_seed"],
        "gamma_static": None, "gamma_dynamic": None, "gamma_speed": None,
        "cost_level_enabled": False, "prospective_gate_logging": False,
        "additional_pickup_overhead_s": 0., "controlled_replay_v2": True,
        "matching_end_s": cfg["last_dispatch_s"], "benchmark_runtime_guard_s": cfg["scenario_timeout_s"],
        "candidate_top_k": cfg["current_top_k"], "cached_geometry": cfg["cached_geometry"],
        "epoch_routing_queue": cfg["epoch_routing_queue"],
        "pre_route_session_certificate": cfg["pre_route_session_certificate"],
        "certified_eta_pruning": cfg["certified_eta_pruning"]}


def outcome_table(control, assignments, cfg):
    assigned = assignments.set_index("native_request_id") if len(assignments) else None
    rows = []
    for rid, request in sorted(control.request_by_rid.items()):
        matched = assigned is not None and rid in assigned.index
        row = dict(order_id=request.order_id, native_request_id=rid,
            cohort="NEW_WINDOW" if request.sim_time_s >= cfg["window_start_s"] else "CARRY_IN",
            release_s=request.sim_time_s, C_compatible=request.profile_id in request.compatible_profiles,
            passenger_accepts_av=bool(request.passenger_accepts_av), matched=matched,
            completed=False, expired=rid in control.expired_rids, vehicle_type=None, wait_s=None,
            completed_within_window=False)
        if matched:
            a = assigned.loc[rid]
            wait = (pd.Timestamp(a.pickup_time) - request.request_time).total_seconds()
            if wait > cfg["patience_s"] + 1e-6:
                raise RuntimeError("original pickup patience violated")
            if a.vehicle_type == "AV" and (not row["C_compatible"] or not row["passenger_accepts_av"]):
                raise RuntimeError("body capability or passenger acceptance violated")
            row.update(completed=bool(a.completed), vehicle_type=a.vehicle_type, wait_s=wait,
                completed_within_window=bool(a.completed) and pd.Timestamp(a.service_end_time)
                    <= control.start + pd.Timedelta(seconds=cfg["measurement_end_s"]))
        elif not row["expired"]:
            raise RuntimeError("revealed native customer not assigned or expired")
        rows.append(row)
    return pd.DataFrame(rows)


def time_chain_check(assignments):
    if not len(assignments):
        return dict(duplicate_orders=0, vehicle_task_overlap=0, maximum_time_error_s=0.)
    duplicate = int(assignments.order_id.duplicated().sum())
    overlap = 0
    for _, group in assignments.groupby("native_vehicle_id"):
        group = group.sort_values("assignment_time", kind="stable")
        overlap += int((pd.to_datetime(group.assignment_time).iloc[1:].to_numpy()
            < pd.to_datetime(group.service_end_time).iloc[:-1].to_numpy()).sum())
    pickup_error = (pd.to_datetime(assignments.pickup_time)
        - pd.to_datetime(assignments.assignment_time)).dt.total_seconds() - assignments.pickup_eta_s
    service_error = (pd.to_datetime(assignments.service_end_time)
        - pd.to_datetime(assignments.pickup_time)).dt.total_seconds() - assignments.realized_service_time_s
    error = float(max(pickup_error.abs().max(), service_error.abs().max()))
    if duplicate or overlap or error > 1e-6 or not isfinite(error):
        raise RuntimeError("native time-chain or unique-resource accounting failed")
    return dict(duplicate_orders=duplicate, vehicle_task_overlap=overlap, maximum_time_error_s=error)


def run_policy(root, fleetpy, cfg, reference, protected, policy):
    from stage4.dispatch.city_defer_native import CityDeferNativeAdapter
    from stage4.dispatch.city_pickup_support import CityPickupSupportValidator
    from stage4.fleetpy_adapter.research_native_bridge import ResearchNativeBridge

    started = perf_counter()
    directory = root / OUT / policy
    directory.mkdir(parents=True, exist_ok=True)
    resource_guard(root, cfg, initial=True)
    summary = dict(status="RUNNING", policy=policy, comparison_kind=cfg["comparison_kind"],
        code_sha=code_sha(root), inputs_sha256=protected, frozen_protocol=cfg,
        generic_reference=reference, gpu_used=False, dense_matrix=False, full_day=False)
    atomic_json(directory / "summary.json", summary)
    routing = None
    bridge = control = None
    tick = None
    try:
        start = pd.Timestamp("2016-10-31T00:00:00+08:00")
        episode = load_research_episode(root, profile_id=cfg["profile_id"], requested_q_a=cfg["requested_q_a"])
        summary["source_inventory_before_connection"] = episode.inventory()
        bindings = load_fleetpy_bindings(fleetpy)
        registry = CoordinateRegistry()
        bridge = ResearchNativeBridge(episode, bindings, registry,
            window_start_s=cfg["window_start_s"], measurement_end_s=cfg["measurement_end_s"],
            admission_end_s=cfg["last_dispatch_s"])
        network = create_native_network(bindings, registry)
        demand = bridge.create_demand(network, directory)
        fixtures = bridge.native_fixtures()
        vehicles, native_output = create_native_vehicles(fixtures, bindings, registry,
            demand.rq_db, directory / "runtime", native_movement=True, routing_engine=network)
        routing = StaticRawRoutingAdapter(root, routing_mode=cfg["routing_mode"],
            persistent_cache_size=cfg["persistent_cache_size"], route_workers=1,
            disk_cache_path=str(root / cfg["disk_route_cache"]),
            routing_context=routing_context(root),
            disk_cache_limit_mib=cfg["disk_cache_limit_mib"], disk_cache_max_entries=cfg["disk_cache_max_entries"],
            route_queue_chunk_size=cfg["route_queue_chunk_size"], lazy_worker_actor=True,
            grouped_sources=True, static_raw_od=True, raw_od_cache_size=cfg["raw_od_cache_size"],
            certified_eta_pruning=True)
        control = create_rolling_or_fleet_control(bindings, vehicles, [], demand, network, routing,
            start, start + pd.Timedelta(seconds=cfg["physical_drain_limit_s"]), native_configuration(root, cfg))
        control.enable_event_calendar()
        control.research_route_policy = CommonBodyPolicy()
        empty_router = ControlledEmptyRouter(root, routing)
        movement_reference, _ = load_train_demand_reference(root)
        manager = CommonIdleMovementManager(control, movement_reference, empty_router,
            max_moves=cfg["reposition_max_moves"], radius_m=cfg["reposition_radius_m"],
            max_eta_s=cfg["reposition_max_eta_s"], top_k=cfg["reposition_top_k"],
            day_end_s=cfg["measurement_end_s"])
        control.repositioning_manager = manager
        bridge.install(control)
        validator = CityPickupSupportValidator(empty_router,
            geometry_timestamp=start + pd.Timedelta(seconds=cfg["window_start_s"]),
            routing_budget_s=cfg["scenario_timeout_s"])
        # This store is an environment-only preprojection of frozen identity
        # and geometry. The callback still accepts only revealed/current or
        # already-completed identities. No scope list is handed to the policy.
        scope = [r.order_id for r in episode.requests._requests
            if r.release_time_s < cfg["measurement_end_s"] and r.expires_at_s > cfg["window_start_s"]
            and r.release_time_s >= cfg["window_start_s"] - cfg["carry_in_lookback_s"]]
        summary["private_feature_store_setup"] = validator.prepare_private_identity_store(scope)
        del scope
        summary["native_input_counts"] = bridge.input_counts()
        atomic_json(directory / "summary.json", summary)
        templates = pd.read_parquet(root / cfg["source_history_templates"])
        forecast = TrainDemandForecast(templates, cfg, cfg["measurement_end_s"])
        del templates
        remaining = TrainRemainingTime(json.loads((root / cfg["remaining_time_model"]).read_text(encoding="utf-8")))
        control.flexibility_adapter = CityDeferNativeAdapter(policy, forecast, cfg,
            generic_reference=reference, current_arc_filter=validator, remaining_model=remaining)
        simulation = bridge.create_simulation(simulation_end_s=cfg["physical_drain_limit_s"],
            time_step_s=30, vehicles=vehicles,
            fleet_control=control, network=network, native_output=native_output)
        peak_rss = 0.
        for tick in range(cfg["window_start_s"], cfg["physical_drain_limit_s"] + 30, 30):
            if perf_counter() - started > cfg["scenario_timeout_s"]:
                raise TimeoutError("fixed city joint execution hard timeout")
            resources = resource_guard(root, cfg)
            peak_rss = max(peak_rss, resources["rss_mib"])
            simulation.step(tick)
            routing.cache.clear()
            if tick % 300 == 0:
                progress = dict(policy=policy, tick=tick, matched=len(control.assignment_rows),
                    completed=len(control.completed_rids), expired=len(control.expired_rids),
                    runtime_s=perf_counter()-started, **resources)
                atomic_json(directory / "progress.json", progress)
                print(json.dumps(progress), flush=True)
            if (tick > cfg["last_dispatch_s"] and not (set(control.rid_to_assigned_vid) - control.completed_rids)
                    and not manager.active):
                break
        control.reconcile()
        if any((control.position_reconciliation_failures, control.request_state_reconciliation_failures,
                control.vehicle_state_reconciliation_failures, control.av_availability_violations)):
            raise RuntimeError("native physical reconciliation failure")
        if set(control.rid_to_assigned_vid) - control.completed_rids:
            raise RuntimeError("physical drain incomplete at the fixed horizon")
        manager.finalize(tick)
        assignments = bridge.environment_assignments()
        check = time_chain_check(assignments)
        outcomes = outcome_table(control, assignments, cfg)
        traces = pd.DataFrame(control.flexibility_adapter.rows)
        epochs = pd.DataFrame(control.epoch_rows)
        movements = pd.DataFrame(manager.rows)
        if len(movements):
            movements["destination_position"] = movements.destination_position.astype(str)
        for name, frame in (("assignments", assignments), ("cohort_outcomes", outcomes),
            ("solver_trace", traces), ("epochs", epochs), ("empty_movements", movements)):
            atomic_parquet(directory / f"{name}.parquet", frame)
        if protected != {p: sha(root / p) for p in protected}:
            raise RuntimeError("frozen inputs changed during execution")
        new = outcomes.loc[outcomes.cohort.eq("NEW_WINDOW")]
        carry = outcomes.loc[outcomes.cohort.eq("CARRY_IN")]
        model_rows = traces.loc[traces.variable_count.notna()] if "variable_count" in traces else traces.iloc[0:0]
        summary.update(status="CITY_NATIVE_SHORT_WINDOW_COMPLETE", native_execution_performed=True,
            native_start_s=cfg["window_start_s"], drain_end_s=tick, runtime_s=perf_counter()-started,
            peak_rss_mib=max(peak_rss, psutil.Process().memory_info().peak_wset/2**20),
            instantiated_native_slots=len(fixtures), all_day_supply=episode.supply.inventory(),
            new_window_orders=len(new), carry_in_orders=len(carry),
            matched_new_window=int(new.matched.sum()), matched_carry_in=int(carry.matched.sum()),
            completed_new_window=int(new.completed.sum()), completed_within_window=int(new.completed_within_window.sum()),
            served_C=int(new.vehicle_type.eq("AV").sum()), served_HV=int(new.vehicle_type.eq("HV").sum()),
            mean_wait_s=float(new.wait_s.mean()), expired_new_window=int(new.expired.sum()),
            total_executed_pickup_empty_m=float(assignments.pickup_route_distance_m.sum()) if len(assignments) else 0.,
            independent_idle_movements=manager.summary(), pickup_support=validator.diagnostics(),
            execution_bridge=bridge.diagnostics(), native_input_counts=bridge.input_counts(), physical_accounting=check,
            maximum_model_variables=int(model_rows.variable_count.max()) if len(model_rows) else 0,
            maximum_model_nonzeros=int(model_rows.nonzeros.max()) if len(model_rows) else 0,
            maximum_decision_time_s=float(traces.solver_time_s.max()) if len(traces) else 0.,
            solver_pipeline_time_s=float(traces.solver_time_s.sum()) if len(traces) else 0.,
            candidate_generation_time_s=float(epochs.candidate_generation_time_s.sum()) if len(epochs) else 0.,
            routing_time_s=float(epochs.routing_time_s.sum()) if len(epochs) else 0.,
            actual_routing=routing.diagnostics(), native_physical_reconciliation=True,
            truth_source="HISTORICAL_LOADED_DURATION_REUSED_ENVIRONMENT_ONLY_AFTER_COMMITMENT",
            prediction_source="FROZEN_M3_DECISION_ONLY_NO_REFIT",
            initialization="COLD_START_AT_ORIGINAL_SLOT_TEMPLATE_POSITIONS_NOT_OBSERVED_1700_POSITIONS")
        atomic_json(directory / "summary.json", summary)
        atomic_json(root / DOC / f"{policy.lower()}_summary.json", summary)
        print(json.dumps(dict(policy=policy,status=summary["status"],new_orders=len(new),
            served=summary["matched_new_window"],runtime_s=summary["runtime_s"])), flush=True)
        return summary
    except Exception as error:
        summary.update(status="STOPPED", error=repr(error), runtime_s=perf_counter()-started)
        if control is not None:
            summary["partial_execution_not_a_finished_result"] = dict(
                last_tick_s=tick, committed=len(control.assignment_rows),
                completed=len(control.completed_rids), expired=len(control.expired_rids))
            # Observed/committed environment records only; never label them as
            # a completed window or calculate a service rate from this prefix.
            if bridge is not None:
                atomic_parquet(directory / "partial_assignments.parquet", bridge.environment_assignments())
            if getattr(control, "flexibility_adapter", None) is not None:
                atomic_parquet(directory / "partial_solver_trace.parquet", pd.DataFrame(control.flexibility_adapter.rows))
        atomic_json(directory / "summary.json", summary)
        raise
    finally:
        if routing is not None:
            routing.close()


def archive_stopped_run(root, policy):
    """Recoverable exact-child archive, only for a user-authorized retry."""
    base = (root / OUT).resolve()
    source = (base / policy).resolve()
    source.relative_to(base)
    receipt = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if receipt.get("status") != "STOPPED":
        raise ValueError("only a stopped run may be archived for an explicit retry")
    parent = base / "failed_attempts"
    number = 1
    while (parent / f"attempt_{number:03d}" / policy).exists():
        number += 1
    destination = (parent / f"attempt_{number:03d}" / policy).resolve()
    destination.relative_to(base)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return str(destination.relative_to(root))


def run(root, fleetpy=FLEETPY, config=CONFIG, *, prepare_only=False, retry_failed=False):
    root = Path(root).resolve()
    cfg, reference, protected = load_protocol(root, config)
    preparation = dict(status="FIXED_CITY_JOINT_PROTOCOL", code_sha=code_sha(root),
        config=cfg, generic_reference=reference, inputs_sha256=protected,
        preparation_only=prepare_only, no_new_inference_or_training=True)
    atomic_json(root / DOC / "preparation.json", preparation)
    if prepare_only:
        print(json.dumps(preparation), flush=True)
        return preparation
    resource_guard(root, cfg, initial=True)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    summaries=[]
    archived=[]
    for policy in cfg["policies"]:
        directory = root / OUT / policy
        if (directory / "summary.json").exists():
            if not retry_failed:
                raise ValueError("existing real run found; no implicit rerun or overwrite")
            archived.append(archive_stopped_run(root, policy))
        summaries.append(run_policy(root, Path(fleetpy), cfg, reference, protected, policy))
        gc.collect()
    summary = dict(status="CITY_JOINT_VALIDATION_COMPLETE", kind=cfg["comparison_kind"],
        preparation=preparation, policies=summaries, full_day_runs=0, new_native_conditions=2,
        gpu_used=False, dense_matrix=False, parameter_search=False, model_refit=False,
        archived_failed_attempts=archived)
    atomic_json(root / DOC / "summary.json", summary)
    atomic_json(root / OUT / "summary.json", summary)
    print(json.dumps(dict(status=summary["status"],service_counts={s["policy"]:
        s["matched_new_window"] for s in summaries})),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--fleetpy-root",type=Path,default=FLEETPY)
    parser.add_argument("--config",type=Path,default=CONFIG)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--retry-failed",action="store_true",
        help="archive STOPPED output then rerun only after explicit user authorization")
    args=parser.parse_args()
    run(args.root,args.fleetpy_root,args.config,prepare_only=args.prepare_only,retry_failed=args.retry_failed)


if __name__ == "__main__":
    main()
