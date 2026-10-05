"""Construct reusable acceleration inputs ONLY; never starts an experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd
import psutil
import pyarrow as pa

from stage3.scripts.traffic_state_batch1 import sha, write_json
from stage4.analysis.symmetric_research_prepare import CONFIG, OUT as INPUT
from stage4.dispatch.fleet_normalization import build_fleet_scenario, FLEET_REL, SCALING_REL
from stage4.dispatch.runtime_assets import build_runtime_assets, routing_context, RuntimeAssets
from stage4.fleetpy_adapter.test31_demand_adapter import load_all_test31_requests, ORDER_BASE_REL
from stage4.fleetpy_adapter.valhalla_time_adapter import CALIBRATION_REL


def asset_input_hashes(root, cfg):
    source = Path("stage4/output/final_experiments") / cfg["source_scenario"] / "scenario_config.json"
    paths = [CONFIG, INPUT / "test31_research_routes.parquet", INPUT / "train_request_templates.parquet",
        Path(cfg["remaining_time_model"]), Path("stage3/config/stage3_av_capability_profiles.json"),
        ORDER_BASE_REL, FLEET_REL, SCALING_REL, CALIBRATION_REL, source,
        Path("stage2/output_v5_2/development/M3/epoch_004.pt")]
    return {p.as_posix(): sha(root / p) for p in paths}


def prepare(root, directory):
    root, directory = Path(root).resolve(), Path(directory)
    if not directory.is_absolute():
        directory = root / directory
    if not directory.resolve().is_relative_to(root):
        raise ValueError("offline assets must stay in the workspace")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    started = time.perf_counter()
    cfg = json.loads((root / CONFIG).read_text())
    inputs = asset_input_hashes(root, cfg)
    context = routing_context(root)
    if directory.exists():
        assets = RuntimeAssets(directory, expected_inputs=inputs)
        if assets.manifest["routing_context"] != context:
            raise ValueError("offline routing context is stale")
        print(json.dumps(dict(status="REUSED_COMPLETE", **{k: assets.manifest[k] for k in ("orders", "positions", "forecast_draws", "byte_count")})), flush=True)
        return assets
    source = root / "stage4/output/final_experiments" / cfg["source_scenario"] / "scenario_config.json"
    base = json.loads(source.read_text())["runtime_configuration"]
    start = pd.Timestamp("2016-10-31T00:00:00+08:00")
    raw = load_all_test31_requests(root, start=start, end=start + pd.Timedelta(seconds=cfg["measurement_end_s"]), profile_id="M")
    if len(raw) != 30000:
        raise ValueError("offline Test31 cohort differs from the frozen 30000")
    routes = pd.read_parquet(root / INPUT / "test31_research_routes.parquet")
    templates = pd.read_parquet(root / INPUT / "train_request_templates.parquet")
    fleet = build_fleet_scenario(root, benchmark_start=start, simulation_end=start + pd.Timedelta(seconds=cfg["physical_drain_limit_s"]),
        requested_q_a=base["av_vehicle_hour_share"], seed=base["fleet_sampling_seed"], max_hv_hour_error_pct=base["max_hv_vehicle_hour_error_pct"])
    assets = build_runtime_assets(directory, raw, routes, templates, fleet, cfg, base, inputs, context)
    summary = dict(status="CONSTRUCTED_NO_SIMULATION", runtime_s=time.perf_counter() - started,
        peak_rss_mib=psutil.Process().memory_info().peak_wset / 2**20,
        **{k: assets.manifest[k] for k in ("orders", "common_eligible", "positions", "forecast_draws", "fleet_fixtures", "byte_count")})
    write_json(root / "stage4/docs/flexibility_dispatch/acceleration_v2/assets_summary.json", summary)
    print(json.dumps(summary), flush=True)
    return assets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("stage4/output/runtime_acceleration_v2/assets"))
    args = parser.parse_args()
    prepare(Path.cwd(), args.output)
