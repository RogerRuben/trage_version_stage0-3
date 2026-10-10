"""Finite source/forecast/real-routing equivalence check, no native simulation."""
from __future__ import annotations

import json
import gc
from pathlib import Path
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa

from stage3.scripts.traffic_state_batch1 import sha, write_json
from stage4.analysis.acceleration_assets import asset_input_hashes
from stage4.analysis.symmetric_research_prepare import CONFIG, OUT as INPUT
from stage4.dispatch.candidate_graph import SpatialVehicle, SparseCandidateIndex
from stage4.dispatch.deterministic_routing import ArcDeterministicValhallaAdapter, SINGLE_SOURCE_MATRIX
from stage4.dispatch.flexibility_native import ResearchRoutePolicy, TrainDemandForecast
from stage4.dispatch.runtime_assets import RuntimeAssets, CompactResearchRoutePolicy, DrawTapeForecast, routing_context
from stage4.fleetpy_adapter.test31_demand_adapter import load_all_test31_requests


def main():
    root = Path.cwd()
    started = time.perf_counter()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    cfg = json.loads((root / CONFIG).read_text())
    spec_path = root / "stage4/config/symmetric_flexibility_acceleration_v2.json"
    settings = json.loads(spec_path.read_text())["settings"]
    assets = RuntimeAssets(root / settings["offline_assets"], expected_inputs=asset_input_hashes(root, cfg))
    context = routing_context(root)
    if context != assets.manifest["routing_context"]:
        raise ValueError("routing context differs from construction")
    baseline_path = root / "stage4/output/symmetric_flexibility_v1/accelerated_full_day/SERVICE_PRESERVING_LOOKAHEAD/summary.json"
    baseline = json.loads(baseline_path.read_text())
    for name, digest in baseline["inputs_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError(f"completed v1 frozen input changed: {name}")
    start = pd.Timestamp("2016-10-31T00:00:00+08:00")
    raw = load_all_test31_requests(root, start=start, end=start + pd.Timedelta(seconds=cfg["measurement_end_s"]), profile_id="M")
    compact = assets.requests()
    routes = pd.read_parquet(root / INPUT / "test31_research_routes.parquet")
    rows = routes.set_index("order_id")
    for request in raw:
        row = rows.loc[request.order_id]
        if row["common_eligible"]:
            request.predicted_service_time_s = float(row["predicted_route_time_p50_s"])
    if len(raw) != len(compact):
        raise ValueError("compact order population changed")
    for left, right in zip(raw, compact):
        for key, value in left.__dict__.items():
            actual = right.__dict__[key]
            if isinstance(value, float) and np.isnan(value):
                same = isinstance(actual, float) and np.isnan(actual)
            else:
                same = value == actual
            if not same:
                raise ValueError(f"compact request changed: {left.order_id}/{key}")
    profiles = json.loads((root / "stage3/config/stage3_av_capability_profiles.json").read_text())
    reference_policy = ResearchRoutePolicy(routes, "M", profiles)
    compact_policy = CompactResearchRoutePolicy(assets, "M", profiles)
    for request in compact:
        if reference_policy.evaluate(request) != compact_policy.evaluate(request):
            raise ValueError(f"compact route policy changed: {request.order_id}")
    templates = pd.read_parquet(root / INPUT / "train_request_templates.parquet")
    reference = TrainDemandForecast(templates, {**cfg, "fast_forecast": True}, cfg["measurement_end_s"])
    tape = DrawTapeForecast(assets.directory / "forecast")
    source = root / "stage4/output/final_experiments" / cfg["source_scenario"] / "scenario_config.json"
    base = json.loads(source.read_text())["runtime_configuration"]
    ticks = sorted(set(range(0, cfg["last_dispatch_s"] + 1, 1800)) | {0, 30, 60, 86430, 86460, 86760})
    forecast_requests = 0
    for now in ticks:
        expected = reference.scenarios(now, "M", base["passenger_acceptance_rate"], base["passenger_acceptance_seed"])
        actual = tape.scenarios(now, "M", base["passenger_acceptance_rate"], base["passenger_acceptance_seed"])
        if expected != actual:
            raise ValueError(f"forecast tape changed at {now}")
        forecast_requests += sum(len(scene.new_requests) for scene, _ in actual)
    rng = np.random.default_rng(20261005)
    chosen = rng.choice(len(compact), size=12, replace=False)
    timestamp = start + pd.Timedelta(hours=8)
    batches = [([SpatialVehicle(f"QA{i:02d}", i, "HV", compact[int(n)].pickup_lon_wgs84, compact[int(n)].pickup_lat_wgs84)],
        compact[int(n)].dropoff_lon_wgs84, compact[int(n)].dropoff_lat_wgs84, timestamp) for i, n in enumerate(chosen)]
    serial = ArcDeterministicValhallaAdapter(root, routing_mode=SINGLE_SOURCE_MATRIX)
    queued = ArcDeterministicValhallaAdapter(root, routing_mode=SINGLE_SOURCE_MATRIX, route_workers=2,
        persistent_cache_size=32, disk_cache_path=root / "stage4/output/runtime_acceleration_v2/checks" / (sha(spec_path)[:12] + ".sqlite3"),
        routing_context=context, lazy_worker_actor=True)
    try:
        expected = [serial.estimate_many(*batch) for batch in batches]
        serial.actor = None
        gc.collect()
        if queued._actor is not None:
            raise RuntimeError("unused parent actor was constructed")
        actual = queued.estimate_epoch(batches)
        for lhs, rhs in zip(expected, actual):
            if set(lhs) != set(rhs):
                raise ValueError("real routing success/failure changed")
            for vid, value in lhs.items():
                other = rhs[vid]
                if (value.valhalla_time_s, value.corrected_pickup_eta_s, value.route_distance_m, value.beta) != (
                        other.valhalla_time_s, other.corrected_pickup_eta_s, other.route_distance_m, other.beta):
                    raise ValueError("real raw 1x1 query changed")
        queued.cache.clear()
        warm = queued.estimate_epoch(batches)
        for lhs, rhs in zip(expected, warm):
            if [(k, x.valhalla_time_s, x.route_distance_m) for k, x in lhs.items()] != [(k, x.valhalla_time_s, x.route_distance_m) for k, x in rhs.items()]:
                raise ValueError("real exact cache changed query answers")
        resources = queued.process_group_resources()
        group_rss = resources["rss_mib"]
        if queued._actor is not None:
            raise RuntimeError("worker routing unexpectedly created a parent actor")
        if group_rss > 2048:
            raise MemoryError("finite routing QA exceeds the run memory bound")
    finally:
        serial.close()
        queued.close()
    result = dict(status="PASS", scope="OFFLINE_AND_FINITE_EQUIVALENCE_ONLY", full_day_simulation_started=False,
        compact_orders_compared=len(compact), route_policies_compared=len(compact), forecast_epochs_compared=len(ticks),
        forecast_requests_compared=forecast_requests, real_raw_routes_compared=len(batches),
        protected_v1_inputs_unchanged=True, gpu_used=False, dense_matrix=False,
        runtime_s=time.perf_counter() - started, peak_parent_rss_mib=psutil.Process().memory_info().peak_wset / 2**20,
        sampled_route_process_group_rss_mib=group_rss, acceleration_config_sha256=sha(spec_path),
        sampled_route_process_group_private_committed_mib=resources["private_committed_mib"],
        real_raw_backend_evaluations=queued.routing_arc_evaluations, unused_parent_actor_created=False,
        assets_manifest_sha256=sha(assets.directory / "manifest.json"))
    write_json(root / "stage4/docs/flexibility_dispatch/acceleration_v2/checks.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
