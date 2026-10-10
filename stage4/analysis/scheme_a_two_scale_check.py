"""One authorized high-load native check of the two-scale Scheme-A interface.

This is a cold-start 08:15--08:30 Test31 check, not a recovery of the old 08:30
simulation state, a policy comparison, or a full-day experiment. Only this CLI
process is stopped by its fixed 900-second watchdog. Failed output is retained;
there is no retry, archive, layout rebuild, or old-scenario loop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
from time import perf_counter


CONFIG = Path("stage4/config/scheme_a_two_scale_v2.json")
OUT = Path("stage4/output/capability_chain_planning/two_scale_v2/native_check")
DOC = Path("stage4/docs/capability_chain_planning/two_scale_v2")
FLEETPY = Path("D:/pycodes/didi_xian_raw/.external/FleetPy")

_BASE_FIXED = dict(test_date="20161031", profile_id="C", requested_q_a=.1,
    passenger_acceptance_rate=.7, passenger_acceptance_seed=20260827,
    supply_seed=20260824, forecast_seed=20261004, planning_horizon_s=1800,
    coarse_reference_update_s=300, rolling_step_s=30, patience_s=300,
    current_top_k=20, reposition_interval_s=900, reposition_radius_m=2000.,
    reposition_max_eta_s=300., reposition_top_k=3, reposition_max_moves=50,
    solver_time_limit_s=10., maximum_rss_mib=1536,
    minimum_initial_available_ram_mib=1536, minimum_disk_reserve_mib=512,
    route_workers=1, dense_matrix=False, gpu_used=False,
    parameter_search=False, model_refit=False)
_CHECK_FIXED = dict(version="scheme_a_two_scale_v2", policy="CHAIN_DEFER",
    layout_mode="joint", window_start_s=29700, measurement_end_s=30600,
    admission_end_s=30900, last_dispatch_s=30900, physical_drain_limit_s=37800,
    carry_in_lookback_s=300, include_carry_in=True, scenario_timeout_s=900,
    expected_layout_bins=47, expected_av_slots=851, coarse_reference_day_end_s=86434,
    coarse_value_wall_limit_s=5., coarse_value_max_pricing_rounds=6,
    coarse_value_max_columns=20000, coarse_value_max_nonzeros=150000,
    coarse_pickup_grid_s=300, coarse_empty_eta_model="EXISTING_TRAIN_M3_CHORD_PACE",
    solver_fallback="EXPLICIT_FEASIBLE_SERVICE_FIRST",
    coarse_forecast_road_certificate=False, full_day=False, native_execution=True,
    automatic_retry=False)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(root, path=CONFIG):
    """Inherit the original scientific conditions, overriding only this check."""
    root, path = Path(root), Path(path)
    overrides = json.loads((root / path).read_text(encoding="utf-8"))
    base_path = Path(overrides["base_config"])
    if base_path != Path("stage4/config/scheme_a_city_v1.json"):
        raise ValueError("the two-scale check must inherit the frozen city protocol")
    if _sha(root / base_path) != overrides["base_config_sha256"]:
        raise ValueError("frozen base configuration bytes changed")
    base = json.loads((root / base_path).read_text(encoding="utf-8"))
    if any(base.get(key) != value for key, value in _BASE_FIXED.items()):
        raise ValueError("frozen city scientific or resource conditions changed")
    if any(key in overrides and overrides[key] != value for key, value in _BASE_FIXED.items()):
        raise ValueError("two-scale check cannot override the scientific conditions")
    cfg = dict(base, **overrides)
    if (any(cfg.get(key) != value for key, value in _CHECK_FIXED.items())
            or cfg["groups"] != ["TWO_SCALE_NATIVE_CHECK"]
            or cfg["forecast_train_dates"] != ["20161010", "20161017", "20161024"]):
        raise ValueError("outside the one fixed 15-minute two-scale native check")
    if Path(cfg["source_layout"]) != Path("stage4/output/capability_chain_planning/scheme_a_city_v1/layout/placements.json"):
        raise ValueError("only the existing complete 47-bin layout may be reused")
    if Path(cfg["disk_route_cache"]) != OUT / "routing/cache.sqlite3":
        raise ValueError("routing output must remain in the independent two-scale check directory")
    return cfg


def load_frozen_layout(root, cfg):
    """Read-only reuse of all original AV slots; never rerun the layout model."""
    root = Path(root)
    source = root / cfg["source_layout"]
    if _sha(source) != cfg["source_layout_sha256"]:
        raise ValueError("frozen completed layout bytes changed")
    receipt = json.loads(source.read_text(encoding="utf-8"))
    positions = receipt["placements"]
    if (receipt["status"] != "LAYOUT_COMPLETE"
            or len(positions["records"]) != cfg["expected_layout_bins"]
            or len(positions["joint"]) != cfg["expected_av_slots"]
            or set(positions["joint"]) != set(positions["hotspot"])):
        raise ValueError("existing layout is not the complete same-slot 47-bin product")
    protected = dict(receipt["inputs_sha256"])
    for relative, expected in protected.items():
        if _sha(root / relative) != expected:
            raise ValueError(f"frozen layout source changed: {relative}")
    return receipt, protected


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class _HardStop:
    """Process-local administrative limit, not a solver or scientific fallback."""
    def __init__(self, directory, started, seconds, summary):
        self.directory, self.started = Path(directory), started
        self.seconds = float(seconds)
        self.lock = threading.RLock()
        self.published = dict(summary)
        self.timer = threading.Timer(max(.001, seconds - (perf_counter() - started)), self._stop)
        self.timer.daemon = True

    def start(self):
        self.timer.start()

    def publish(self, summary):
        with self.lock:
            _atomic_json(self.directory / "summary.json", summary)
            self.published = json.loads(json.dumps(summary, allow_nan=False))

    def _stop(self):
        try:
            with self.lock:
                marker = dict(status="STOPPED_HARD_TIMEOUT", pid=os.getpid(),
                    runtime_s=perf_counter() - self.started, hard_limit_s=self.seconds,
                    last_durable_progress=self.published.get("durable_progress"),
                    current_call_may_not_have_saved_partial_state=True,
                    final_service_result_available=False, automatic_retry=False)
                _atomic_json(self.directory / "hard_timeout_marker.json", marker)
                _atomic_json(self.directory / "summary.json", dict(self.published, **marker))
                print(json.dumps(marker), flush=True)
        finally:
            os._exit(124)

    def close(self):
        self.timer.cancel()


def _resource_guard(root, cfg, started, *, initial=False):
    import psutil
    resources = dict(rss_mib=psutil.Process().memory_info().rss / 2**20,
        available_ram_mib=psutil.virtual_memory().available / 2**20,
        free_disk_mib=psutil.disk_usage(Path(root).anchor).free / 2**20)
    if perf_counter() - started >= cfg["scenario_timeout_s"]:
        raise TimeoutError(f"fixed {cfg['scenario_timeout_s']:g}-second native execution limit exceeded")
    if resources["rss_mib"] > cfg["maximum_rss_mib"]:
        raise MemoryError("native check exceeds the unchanged 1536-MiB RSS cap")
    if initial and resources["available_ram_mib"] < cfg["minimum_initial_available_ram_mib"]:
        raise MemoryError("initial available RAM is below 1536 MiB; other applications are not closed")
    if resources["free_disk_mib"] < cfg["minimum_disk_reserve_mib"]:
        raise OSError("native check disk reserve is below 512 MiB")
    return resources


def _timing_snapshot(adapter):
    # These are the adapter's own measured fields, not estimated breakdowns.
    return dict(adapter.progress_summary().get("timings_s", {}))


def _timing_delta(after, before):
    return {key: float(value) - float(before.get(key, 0.))
            for key, value in after.items()}


def _run_native(root, fleetpy, config, cfg, *, directory, doc_summary_path,
                completion_status, group=None, execution_binding=None, started=None):
    """Shared physical execution after the calling CLI validates its protocol.

    This private function cannot widen the public short-check CLI. Full-day
    group configuration is supplied only by the separate fixed city protocol.
    """
    started = perf_counter() if started is None else started
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    root = Path(root).resolve()
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("existing native output is retained; no implicit retry or overwrite")
    resources = _resource_guard(root, cfg, started, initial=True)
    full_day = bool(cfg["full_day"])
    summary = dict(status="RUNNING", version=cfg["version"], pid=os.getpid(),
        policy=cfg["policy"], layout_mode=cfg["layout_mode"], window_start_s=cfg["window_start_s"],
        measurement_end_s=cfg["measurement_end_s"], admission_end_s=cfg["admission_end_s"],
        physical_drain_limit_s=cfg["physical_drain_limit_s"], full_day=full_day,
        initialization=("ORIGINAL_HV_START_AND_FROZEN_EARLIER_HISTORY_AV_LAYOUT_FROM_MIDNIGHT"
            if full_day else "COLD_START_FROZEN_JOINT_LAYOUT_NOT_RECOVERED_0830_STATE"),
        strategy_comparison=full_day, scientific_conclusion_from_service_rate=False,
        no_future_test31_requests_in_policy=True, future_values_are_not_realized_service=True,
        algorithm_semantics_changed=True, legacy_realtime_chain_used=False,
        gpu_used=False, dense_matrix=False, layout_recomputed=False,
        hard_limit_s=cfg["scenario_timeout_s"], **resources)
    if group is not None:
        summary["group"] = group
    if execution_binding is not None:
        summary["execution_binding"] = execution_binding
    hard_stop = _HardStop(directory, started, cfg["scenario_timeout_s"], summary)
    hard_stop.publish(summary)
    hard_stop.start()
    routing = bridge = control = adapter = provider = manager = None
    tick = None
    step_rows, phases = [], {}
    try:
        # Heavy imports and cold setup are counted after runner entry, not hidden
        # in the simulation-only timing. Python process/bootstrap is separate.
        imported = perf_counter()
        import pandas as pd
        import psutil
        import pyarrow as pa
        from stage4.analysis.capability_chain_instances import atomic_parquet
        from stage4.analysis.city_joint_validation import code_sha, native_configuration, outcome_table, time_chain_check
        from stage4.analysis.scheme_a_city_experiment import make_routing, load_history_templates, _clock_cfg
        from stage4.dispatch.controlled_routes import ControlledEmptyRouter
        from stage4.dispatch.remaining_time import TrainRemainingTime
        from stage4.dispatch.repositioning_policy import load_train_demand_reference
        from stage4.dispatch.rolling_or_control import create_rolling_or_fleet_control
        from stage4.dispatch.city_pickup_support import CityPickupSupportValidator
        from stage4.dispatch.scheme_a_city_adapter import apply_av_layout
        from stage4.dispatch.scheme_a_city_connections import SchemeACityConnectors, SelectedMovementManager
        from stage4.dispatch.scheme_a_city_graph import CoarseHistoricalLibrary
        from stage4.dispatch.scheme_a_two_scale_native import NativeTwoScaleAdapter
        from stage4.dispatch.scheme_a_two_scale_value import CoarseServiceValue
        from stage4.fleetpy_adapter.mixed_fleet_adapter import create_native_vehicles
        from stage4.fleetpy_adapter.native_network import create_native_network
        from stage4.fleetpy_adapter.research_native_bridge import ResearchNativeBridge
        from stage4.fleetpy_adapter.research_world import load_research_episode
        from stage4.fleetpy_adapter.upstream import CoordinateRegistry, load_fleetpy_bindings
        phases["heavy_import_time_s"] = perf_counter() - imported
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        summary["code_sha"] = code_sha(root)
        if execution_binding is not None and summary["code_sha"] != execution_binding["code_sha"]:
            raise RuntimeError("code commit changed after sequential experiment preparation")
        loaded = perf_counter()
        layout, protected = load_frozen_layout(root, cfg)
        protected.update({Path(config).as_posix(): _sha(root / config),
                          cfg["base_config"]: _sha(root / cfg["base_config"]),
                          cfg["source_layout"]: _sha(root / cfg["source_layout"])})
        if execution_binding is not None and protected != execution_binding["inputs_sha256"]:
            raise RuntimeError("source inputs changed after sequential experiment preparation")
        episode = load_research_episode(root, profile_id=cfg["profile_id"], requested_q_a=cfg["requested_q_a"])
        if full_day:
            # Derive the original source's 33-second tail, not a hard truncation
            # at midnight or a cold-start window masquerading as a full day.
            cfg = _clock_cfg(episode, cfg)
            if (cfg["measurement_end_s"], cfg["admission_end_s"]) != (86434, 86760):
                raise RuntimeError("full-day source clock differs from the frozen Test31 span")
        if apply_av_layout(episode, layout["placements"][cfg["layout_mode"]]) != cfg["expected_av_slots"]:
            raise RuntimeError("reused layout does not cover the exact original AV slots")
        phases["frozen_layout_and_episode_load_s"] = perf_counter() - loaded
        summary.update(inputs_sha256=protected, frozen_protocol=cfg,
            layout_sha256=cfg["source_layout_sha256"], layout_bins_reused=47, AV_slots_reused=851)
        start = pd.Timestamp("2016-10-31T00:00:00+08:00")
        loaded = perf_counter()
        bindings = load_fleetpy_bindings(Path(fleetpy))
        registry = CoordinateRegistry()
        bridge = ResearchNativeBridge(episode, bindings, registry,
            window_start_s=cfg["window_start_s"], measurement_end_s=cfg["measurement_end_s"],
            admission_end_s=cfg["admission_end_s"], include_carry_in=cfg["include_carry_in"])
        if full_day and bridge.input_counts()["window_new_requests"] != len(episode.requests._requests):
            raise RuntimeError("full-day native bridge dropped original common requests")
        network = create_native_network(bindings, registry)
        demand = bridge.create_demand(network, directory)
        fixtures = bridge.native_fixtures()
        vehicles, native_output = create_native_vehicles(fixtures, bindings, registry,
            demand.rq_db, directory / "runtime", native_movement=True, routing_engine=network)
        routing = make_routing(root, cfg)
        native_cfg = native_configuration(root, cfg)
        native_cfg["scheme_a_fast_forecast_hv"] = True
        control = create_rolling_or_fleet_control(bindings, vehicles, [], demand, network, routing,
            start, start + pd.Timedelta(seconds=cfg["physical_drain_limit_s"]), native_cfg)
        control.enable_event_calendar()
        reference, reference_manifest = load_train_demand_reference(root)
        empty_router = ControlledEmptyRouter(root, routing)
        manager = SelectedMovementManager(control, reference, empty_router,
            max_moves=cfg["reposition_max_moves"], radius_m=cfg["reposition_radius_m"],
            max_eta_s=cfg["reposition_max_eta_s"], top_k=cfg["reposition_top_k"],
            day_end_s=cfg["movement_day_end_s"])
        control.repositioning_manager = manager
        bridge.install(control)
        phases["native_environment_setup_s"] = perf_counter() - loaded
        loaded = perf_counter()
        validator = CityPickupSupportValidator(empty_router, geometry_timestamp=start,
            routing_budget_s=cfg["scenario_timeout_s"])
        # Environment-only projection. This scope is never handed to the policy;
        # the existing validator still requires a revealed/committed identity.
        scope = [r.order_id for r in episode.requests._requests
            if r.release_time_s < cfg["measurement_end_s"] and r.expires_at_s > cfg["window_start_s"]
            and r.release_time_s >= cfg["window_start_s"] - cfg["carry_in_lookback_s"]
            and "C" in r.compatible_profiles]
        summary["private_feature_store_setup"] = validator.prepare_private_identity_store(scope)
        del scope
        phases["private_window_identity_projection_s"] = perf_counter() - loaded
        loaded = perf_counter()
        templates, history_inventory = load_history_templates(root, cfg)
        library_cfg = dict(cfg, measurement_end_s=cfg["coarse_reference_day_end_s"])
        library = CoarseHistoricalLibrary(templates, library_cfg)
        connectors = SchemeACityConnectors(control, empty_router, validator, templates, geometry_timestamp=start)
        provider = CoarseServiceValue(library, reference, templates, cfg)
        del templates
        remaining = TrainRemainingTime(json.loads((root / cfg["remaining_time_model"]).read_text(encoding="utf-8")))
        adapter = NativeTwoScaleAdapter(cfg["policy"], provider, reference, connectors, validator, remaining, cfg)
        control.flexibility_adapter = adapter
        simulation = bridge.create_simulation(simulation_end_s=cfg["physical_drain_limit_s"],
            time_step_s=cfg["rolling_step_s"], vehicles=vehicles,
            fleet_control=control, network=network, native_output=native_output)
        phases["historical_value_and_adapter_setup_s"] = perf_counter() - loaded
        setup_s = perf_counter() - started
        summary.update(setup_time_s=setup_s, setup_phases_s=phases,
            native_input_counts=bridge.input_counts(), historical_template_inventory=history_inventory,
            reference_manifest=reference_manifest,
            future_supply_scope=("ORIGINAL_FULL_DAY_SUPPLY_NO_SLOT_TIME_CLIPPING" if full_day
                else "ORIGINAL_WINDOW_OVERLAPPING_SLOTS_NOT_COMPLETE_FUTURE_DAY_SUPPLY"),
            source_slot_times_clipped=False, prediction_horizon_s=1800)
        if not full_day:
            summary["prediction_window_independent_of_short_measurement_end"] = True
        hard_stop.publish(summary)
        simulation_started = perf_counter()
        peak_rss = resources["rss_mib"]
        for tick in range(cfg["window_start_s"], cfg["physical_drain_limit_s"] + 30, 30):
            _resource_guard(root, cfg, started)
            epoch_cursor = len(control.epoch_rows)
            before = _timing_snapshot(adapter)
            step_started = perf_counter()
            step_completed = False
            try:
                simulation.step(tick)
                step_completed = True
            finally:
                step_wall_s = perf_counter() - step_started
                epoch_rows = control.epoch_rows[epoch_cursor:]
                row = dict(simulation_time_s=tick, step_wall_s=step_wall_s,
                    step_completed=step_completed,
                    candidate_generation_time_s=sum(float(r["candidate_generation_time_s"]) for r in epoch_rows),
                    candidate_routing_time_s=sum(float(r["routing_time_s"]) for r in epoch_rows),
                    adapter_timing_s=_timing_delta(_timing_snapshot(adapter), before),
                    completed_native_epoch_records=len(epoch_rows),
                    timings_are_nested_do_not_sum=True)
                step_rows.append(row)
            routing.cache.clear()
            resources = _resource_guard(root, cfg, started)
            peak_rss = max(peak_rss, resources["rss_mib"])
            durable = dict(tick=tick, runtime_s=perf_counter() - started,
                simulation_loop_wall_s=perf_counter() - simulation_started,
                measured_step_count=len(step_rows), last_step_wall_s=step_wall_s,
                matched=len(control.assignment_rows), completed=len(control.completed_rids),
                expired=len(control.expired_rids), **resources)
            summary["durable_progress"] = durable
            progress_interval = cfg.get("progress_interval_s", 300)
            if not full_day or tick % progress_interval == 0:
                hard_stop.publish(summary)
                _atomic_json(directory / "progress.json", dict(durable,
                    performance=adapter.progress_summary(), coarse_value=provider.diagnostics()))
            if tick % progress_interval == 0:
                atomic_parquet(directory / "partial_step_timings.parquet", pd.DataFrame(step_rows))
                if full_day:
                    atomic_parquet(directory / "partial_solver_trace.parquet", pd.DataFrame(adapter.rows))
                    _atomic_json(directory / "coarse_refreshes.json", provider.refresh_records)
                print(json.dumps(durable), flush=True)
            if (tick >= cfg["admission_end_s"]
                    and not (set(control.rid_to_assigned_vid) - control.completed_rids)
                    and not manager.active):
                break
        simulation_loop_s = perf_counter() - simulation_started
        control.reconcile()
        if any((control.position_reconciliation_failures, control.request_state_reconciliation_failures,
                control.vehicle_state_reconciliation_failures, control.av_availability_violations)):
            raise RuntimeError("native physical reconciliation failed")
        if set(control.rid_to_assigned_vid) - control.completed_rids or manager.active:
            raise RuntimeError(f"committed physical work was not drained by the fixed {cfg['physical_drain_limit_s']}s bound")
        assignments = bridge.environment_assignments()
        if len(assignments):
            # Do not export the shared control's legacy generic policy label
            # as a claim of exact city-chain/global lexicographic optimization.
            assignments["dispatch_policy"] = "TWO_SCALE_CURRENT_ACTION_FLOW_COARSE_FUTURE"
        physics = time_chain_check(assignments)
        outcomes = outcome_table(control, assignments, cfg)
        movements = pd.DataFrame(manager.rows)
        if len(movements):
            movements["destination_position"] = movements.destination_position.astype(str)
        for name, frame in (("assignments", assignments), ("cohort_outcomes", outcomes),
                ("solver_trace", pd.DataFrame(adapter.rows)), ("epochs", pd.DataFrame(control.epoch_rows)),
                ("empty_movements", movements), ("step_timings", pd.DataFrame(step_rows))):
            atomic_parquet(directory / f"{name}.parquet", frame)
        if protected != {relative: _sha(root / relative) for relative in protected}:
            raise RuntimeError("frozen inputs changed during the check")
        bridge_diagnostics = bridge.diagnostics()
        if full_day:
            bridge_diagnostics["initial_position_source"] = summary["initialization"]
            _atomic_json(directory / "coarse_refreshes.json", provider.refresh_records)
            summary["coarse_refresh_summary"] = dict(refresh_count=len(provider.refresh_records),
                unavailable_count=sum(not record["available"] for record in provider.refresh_records),
                all_pricing_closed_count=sum(bool(record.get("all_pricing_closed", False))
                    for record in provider.refresh_records),
                final_snapshot_is_not_an_all_day_convergence_certificate=True)
        summary.update(status=completion_status, native_execution_performed=True,
            drain_end_s=tick, setup_time_s=setup_s, simulation_loop_wall_s=simulation_loop_s,
            total_runner_wall_s=perf_counter() - started,
            sum_complete_step_wall_s=sum(r["step_wall_s"] for r in step_rows),
            maximum_complete_step_wall_s=max(r["step_wall_s"] for r in step_rows),
            step_count=len(step_rows), peak_rss_mib=max(peak_rss, psutil.Process().memory_info().peak_wset / 2**20),
            revealed_orders=len(outcomes), committed_orders=len(assignments),
            completed_orders=int(outcomes.completed.sum()), expired_orders=int(outcomes.expired.sum()),
            committed_C_orders=int(assignments.vehicle_type.eq("AV").sum()) if len(assignments) else 0,
            committed_HV_orders=int(assignments.vehicle_type.eq("HV").sum()) if len(assignments) else 0,
            all_day_supply=episode.supply.inventory(), physical_accounting=physics,
            execution_bridge=bridge_diagnostics, current_pickup=validator.diagnostics(),
            adapter_performance=adapter.progress_summary(), coarse_value=provider.diagnostics(),
            connector=connectors.diagnostics(), actual_routing=routing.diagnostics(),
            idle_movement=manager.summary(),
            truth_access="ENVIRONMENT_ONLY_AFTER_IRREVERSIBLE_COMMITMENT",
            real_future_test31_requests_used_for_prediction=False,
            no_full_day_runtime_or_service_advantage_claim=not full_day,
            city_global_optimality_claimed=False)
        _resource_guard(root, cfg, started)
        hard_stop.publish(summary)
        _atomic_json(doc_summary_path, summary)
        print(json.dumps(dict(status=summary["status"], setup_time_s=setup_s,
            simulation_loop_wall_s=simulation_loop_s, total_runner_wall_s=summary["total_runner_wall_s"],
            maximum_step_wall_s=summary["maximum_complete_step_wall_s"], completed_orders=summary["completed_orders"])), flush=True)
        return summary
    except Exception as error:
        summary.update(status="STOPPED", error=repr(error), runtime_s=perf_counter() - started,
            setup_phases_s=phases, last_tick_s=tick, partial_result_not_a_complete_service_result=True)
        if control is not None:
            summary["partial_counts"] = dict(committed=len(control.assignment_rows),
                completed=len(control.completed_rids), expired=len(control.expired_rids))
            from stage4.analysis.capability_chain_instances import atomic_parquet
            import pandas as pd
            if bridge is not None:
                atomic_parquet(directory / "partial_assignments.parquet", bridge.environment_assignments())
            if adapter is not None:
                atomic_parquet(directory / "partial_solver_trace.parquet", pd.DataFrame(adapter.rows))
            atomic_parquet(directory / "partial_step_timings.parquet", pd.DataFrame(step_rows))
        if full_day and provider is not None:
            _atomic_json(directory / "coarse_refreshes.json", provider.refresh_records)
        hard_stop.publish(summary)
        raise
    finally:
        hard_stop.close()
        if routing is not None:
            routing.close()


def run(root, fleetpy=FLEETPY, config=CONFIG):
    """Keep the original strictly fixed 08:15 short-check entry point."""
    started = perf_counter()
    root = Path(root).resolve()
    cfg = load_protocol(root, config)
    return _run_native(root, fleetpy, config, cfg, directory=root / OUT,
        doc_summary_path=root / DOC / "native_check_summary.json",
        completion_status="TWO_SCALE_NATIVE_SHORT_CHECK_COMPLETE", started=started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--fleetpy-root", type=Path, default=FLEETPY)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    run(args.root, args.fleetpy_root, args.config)


if __name__ == "__main__":
    main()
