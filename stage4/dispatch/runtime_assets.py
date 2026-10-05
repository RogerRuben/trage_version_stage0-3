"""Small read-only offline inputs and an epoch-chunked common draw tape.

There are no future Test outcomes in the planner arrays. Simulation truth is
stored separately and is attached only to the FleetPy physical request.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from stage3.scripts.traffic_state_batch1 import sha, write_json
from stage4.dispatch.acceptance import stable_acceptance_uniform
from stage4.dispatch.exposure import exposure_excess
from stage4.dispatch.flexibility_model import Request, Scenario
from stage4.fleetpy_adapter.test31_demand_adapter import SpikeRequest

ASSET_VERSION = "symmetric_runtime_assets_v2.1"
POSITION_DTYPE = np.dtype([("lon", "f8"), ("lat", "f8"), ("future_position", "U32"), ("x", "f8"), ("y", "f8")])
TEMPLATE_DTYPE = np.dtype([("release", "f8"), ("pickup", "i4"), ("dropoff", "i4"), ("duration", "f8"), ("mask", "u1")])
DRAW_DTYPE = np.dtype([("source", "i4"), ("rid", "i4"), ("release", "f8"), ("acceptance_uniform", "f8")])
DRAW_INDEX_DTYPE = np.dtype([("chunk", "i4"), ("offset", "i4"), ("count", "i4")])
ROUTE_FIELDS = ("route_max_A_c", "route_max_M_c", "route_max_D_c", "route_max_L_c",
                "max_route_speed_domain_kmh", "control_assumption_count", "bearing_fallback_count") + tuple(
    f"{name}_{kind}" for name in ("speed_cv", "acceleration_rms") for kind in ("E", "Q", "C"))


class PositionCatalogBuilder:
    def __init__(self):
        self.ids, self.rows = {}, []

    def add(self, lon, lat):
        key = (float(lon).hex(), float(lat).hex())
        if key not in self.ids:
            from .flexibility_native import position_key, xy
            position = position_key(lon, lat)
            point = xy(position)  # Match existing rounded future-ETA projection.
            self.ids[key] = len(self.rows)
            self.rows.append((float(lon), float(lat), position, float(point[0]), float(point[1])))
        return self.ids[key]

    def write(self, directory):
        np.save(Path(directory) / "positions.npy", np.asarray(self.rows, dtype=POSITION_DTYPE), allow_pickle=False)


def build_forecast_product(directory, templates, cfg, measurement_end_s, acceptance_seed, catalog):
    """Write one bounded chunk at a time; exactly the original RNG draw order."""
    from .flexibility_native import TrainDemandForecast
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    predictor = TrainDemandForecast(templates, {**cfg, "fast_forecast": True}, measurement_end_s)
    records, offsets = [], {}
    for date, day in predictor.days.items():
        left = len(records)
        for source in day.itertuples(index=False):
            mask = sum((1 << i) for i, k in enumerate(("C", "M", "A")) if bool(getattr(source, f"compatible_{k}")))
            records.append((float(source.release_second), catalog.add(source.start_lon_wgs84, source.start_lat_wgs84),
                catalog.add(source.end_lon_wgs84, source.end_lat_wgs84), float(source.predicted_route_time_p50_s), mask))
        offsets[date] = (left, len(records))
    sources = np.asarray(records, dtype=TEMPLATE_DTYPE)
    np.save(directory / "templates.npy", sources, allow_pickle=False)
    step = int(cfg.get("rolling_step_s", 30))
    last = int(cfg.get("last_dispatch_s", measurement_end_s + cfg["patience_s"]))
    epochs = list(range(0, last + 1, step))
    index = np.zeros((len(epochs), len(offsets)), dtype=DRAW_INDEX_DTYPE)
    total_draws = 0
    # 64 epochs = 32 minutes at the frozen step, not a day-sized Python list.
    for chunk_id, left in enumerate(range(0, len(epochs), 64)):
        draws = []
        for epoch in range(left, min(left + 64, len(epochs))):
            now = epochs[epoch]
            horizon_end = min(now + cfg["forecast_horizon_s"], measurement_end_s - 1)
            for scene, (date, (first, stop)) in enumerate(offsets.items()):
                release_times = sources["release"][first:stop]
                pool_left = int(np.searchsorted(release_times, now, side="right"))
                pool_right = max(pool_left, int(np.searchsorted(release_times, horizon_end, side="right")))
                size = pool_right - pool_left
                seed = int.from_bytes(hashlib.sha256(f'{cfg["forecast_seed"]}|{now}|{date}'.encode()).digest()[:8], "little")
                rng = np.random.default_rng(seed)
                count = int(rng.poisson(size * cfg["forecast_sampling_multiplier"])) if size else 0
                index[epoch, scene] = (chunk_id, len(draws), count)
                choices = rng.integers(0, size, size=count) if count else ()
                for number, choice in enumerate(choices):
                    source = first + pool_left + int(choice)
                    release = float(np.clip(sources[source]["release"] + rng.uniform(-15, 15), now + 1, horizon_end))
                    rid = -(scene + 1) * 1_000_000 - number - 1
                    uniform = stable_acceptance_uniform(f"FORECAST|{now}|{date}|{number}", acceptance_seed)
                    draws.append((source, rid, release, uniform))
        np.save(directory / f"draws_{chunk_id:04d}.npy", np.asarray(draws, dtype=DRAW_DTYPE), allow_pickle=False)
        total_draws += len(draws)
    np.save(directory / "draw_index.npy", index, allow_pickle=False)
    metadata = dict(dates=list(offsets), step_s=step, last_dispatch_s=last,
        acceptance_seed=int(acceptance_seed), patience_s=cfg["patience_s"], total_draws=total_draws,
        global_pace=predictor.global_pace, pace_by_slot=predictor.pace_by_slot,
        information_source="TRAIN_TEMPLATES_AND_FROZEN_TRAIN_M3_P50_ONLY")
    write_json(directory / "forecast.json", metadata)
    return metadata


class DrawTapeForecast:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "forecast.json").read_text())
        self.templates = np.load(self.directory / "templates.npy", mmap_mode="r", allow_pickle=False)
        self.index = np.load(self.directory / "draw_index.npy", mmap_mode="r", allow_pickle=False)
        self.positions = np.load(self.directory.parent / "positions.npy", mmap_mode="r", allow_pickle=False)
        self.global_pace = float(self.meta["global_pace"])
        self.pace_by_slot = {int(k): float(v) for k, v in self.meta["pace_by_slot"].items()}
        self.allowed = {mask: frozenset(k for i, k in enumerate(("C", "M", "A")) if mask & (1 << i)) for mask in range(8)}
        self._chunks = OrderedDict()

    def scenarios(self, now, profile, acceptance_rate, acceptance_seed):
        if acceptance_seed != self.meta["acceptance_seed"] or not 0 <= acceptance_rate <= 1:
            raise ValueError("draw tape acceptance seed/rate mismatch")
        if now % self.meta["step_s"] or not 0 <= now <= self.meta["last_dispatch_s"]:
            raise ValueError("draw tape epoch is outside its frozen calendar")
        result = []
        epoch = int(now // self.meta["step_s"])
        for scene, date in enumerate(self.meta["dates"]):
            row = self.index[epoch, scene]
            chunk_id, offset, count = (int(row[k]) for k in ("chunk", "offset", "count"))
            if chunk_id not in self._chunks:
                self._chunks[chunk_id] = np.load(self.directory / f"draws_{chunk_id:04d}.npy", mmap_mode="r", allow_pickle=False)
                if len(self._chunks) > 2:
                    self._chunks.popitem(last=False)
            requests, coordinates = [], {}
            for draw in self._chunks[chunk_id][offset:offset + count]:
                source = self.templates[int(draw["source"])]
                rid, release = int(draw["rid"]), float(draw["release"])
                accepts = acceptance_rate == 1. or float(draw["acceptance_uniform"]) <= acceptance_rate
                requests.append(Request(rid, release, release + self.meta["patience_s"], float(source["duration"]),
                    str(self.positions[int(source["dropoff"])]["future_position"]), self.allowed[int(source["mask"])], bool(accepts)))
                coordinates[rid] = str(self.positions[int(source["pickup"])]["future_position"])
            result.append((Scenario(f"TRAIN_DAY_{date}", 1 / len(self.meta["dates"]), -7 * 86400, tuple(requests), ()), coordinates))
        return result


class RuntimeAssets:
    def __init__(self, directory, *, expected_inputs=None, verify_products=True):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest["version"] != ASSET_VERSION or self.manifest["status"] != "COMPLETE":
            raise ValueError("runtime assets are not complete")
        if expected_inputs is not None and self.manifest["inputs_sha256"] != expected_inputs:
            raise ValueError("runtime assets have stale input sources")
        if verify_products:
            for name, digest in self.manifest["products_sha256"].items():
                if sha(self.directory / name) != digest:
                    raise ValueError(f"runtime asset changed: {name}")
        self.orders = np.load(self.directory / "orders.npy", mmap_mode="r", allow_pickle=False)
        self.metrics = np.load(self.directory / "route_metrics.npy", mmap_mode="r", allow_pickle=False)
        self.positions = np.load(self.directory / "positions.npy", mmap_mode="r", allow_pickle=False)
        self.by_order = {str(row["order_id"]): i for i, row in enumerate(self.orders)}

    def requests(self):
        # Physical truth is not exposed to the forecast/optimization products.
        truth = np.load(self.directory / "simulation_truth.npy", mmap_mode="r", allow_pickle=False)
        result = []
        for i, row in enumerate(self.orders):
            pickup, dropoff = self.positions[int(row["pickup"])], self.positions[int(row["dropoff"])]
            result.append(SpikeRequest(int(row["native_id"]), str(row["order_id"]),
                pd.Timestamp(int(row["request_ns"]), tz="UTC").tz_convert("Asia/Shanghai"), int(row["release"]),
                float(pickup["lon"]), float(pickup["lat"]), float(dropoff["lon"]), float(dropoff["lat"]),
                float(truth[i]), float(row["predicted"]), "M", str(row["hard_state"]), bool(row["evidence"]),
                float(row["rho_static"]), float(row["rho_dynamic"]), float(row["rho_speed"]), str(row["route_type"])))
        return result

    def fleet(self):
        from stage4.dispatch.fleet_normalization import FleetScenario
        from stage4.fleetpy_adapter.mixed_fleet_adapter import VehicleFixture
        record = json.loads((self.directory / "fleet.json").read_text())
        fixtures = []
        for source in record["fixtures"]:
            row = dict(source)
            for key in ("availability_start_time", "availability_end_time"):
                row[key] = pd.Timestamp(row[key])
            fixtures.append(VehicleFixture(**row))
        return FleetScenario(pd.read_parquet(self.directory / "scenario_fleet.parquet"), fixtures, record["accounting"])

    def windows(self):
        frame = np.load(self.directory / "fleet_windows.npy", mmap_mode="r", allow_pickle=False)
        return {int(r["vid"]): (float(r["start"]), float(r["end"])) for r in frame}


class CompactResearchRoutePolicy:
    def __init__(self, assets, profile, frozen_profiles):
        self.assets, self.profile = assets, profile
        self.caps = next(p for p in frozen_profiles["profiles"] if p["profile_id"] == profile)

    def evaluate(self, request):
        i = self.assets.by_order[str(request.order_id)]
        row, metrics = self.assets.orders[i], self.assets.metrics[i]
        static = self.caps["static_caps"]
        ratios = [float(metrics[key]) / static[cap] for key, cap in {
            "route_max_A_c": "external_physical_connection_count", "route_max_M_c": "topological_movement_count",
            "route_max_D_c": "road_class_diversity", "route_max_L_c": "internal_length_m"}.items()]
        var = [float(metrics[f"{name}_{kind}"]) / self.caps["dynamic_caps"][name][kind]
               for name in ("speed_cv", "acceleration_rms") for kind in ("E", "Q", "C")]
        rho_static = max(ratios) if all(np.isfinite(x) for x in ratios) else np.nan
        rho_var = max(var) if all(np.isfinite(x) for x in var) else np.nan
        rho_speed = float(metrics["max_route_speed_domain_kmh"]) / self.caps["speed_domain_max_kmh"]
        exposure = exposure_excess(rho_static, rho_var, rho_speed)
        return dict(research_route_compatible=bool(int(row["mask"]) & (1 << ("C", "M", "A").index(self.profile))),
            research_base_eligible=bool(row["common"]), research_data_ready=bool(row["data_ready"]),
            exposure=exposure, research_exposure_available=exposure is not None,
            research_control_assumption_count=int(metrics["control_assumption_count"]) if np.isfinite(metrics["control_assumption_count"]) else 0,
            research_bearing_fallback_count=int(metrics["bearing_fallback_count"]) if np.isfinite(metrics["bearing_fallback_count"]) else 0)


def routing_context(root):
    root = Path(root)
    source = root / "stage3/config/stage3_finalization.json"
    config_path = Path(json.loads(source.read_text())["valhalla_config"])
    config = json.loads(config_path.read_text())
    tile_directory = Path(config["mjolnir"]["tile_dir"])
    inventory = hashlib.sha256()
    count = 0
    for path in sorted(tile_directory.rglob("*")):
        if path.is_file():
            stat = path.stat()
            inventory.update(f"{path.relative_to(tile_directory).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
            count += 1
    if not count:
        raise ValueError("frozen routing tiles are missing")
    freeze = json.loads((root / "stage0/docs/stage0_v6_freeze_manifest.json").read_text())
    version = "unknown"
    for name in ("pyvalhalla", "valhalla"):
        try:
            version = importlib.metadata.version(name)
            break
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(schema="RAW_VALHALLA_AUTO_MINUTE_V1", costing="auto", coordinate_system="WGS84",
        valhalla_config_sha256=sha(config_path), valhalla_library_version=version,
        frozen_tiles_sha256=freeze["valhalla_tiles_sha"], tile_inventory_stat_sha256=inventory.hexdigest(),
        tile_file_count=count, tile_identity_check="FROZEN_CONTENT_SHA_PLUS_CURRENT_PATH_SIZE_MTIME_INVENTORY")


def build_runtime_assets(directory, raw_requests, routes, templates, fleet, cfg, base, inputs, context):
    """Construction/ETL only: no routing calls, solver, or native simulation."""
    directory = Path(directory)
    if directory.exists():
        return RuntimeAssets(directory, expected_inputs=inputs)
    temp = directory.with_name(directory.name + ".tmp-" + uuid.uuid4().hex[:8])
    temp.mkdir(parents=True)
    catalog = PositionCatalogBuilder()
    order_size = max(len(r.order_id) for r in raw_requests)
    dtype = np.dtype([("native_id", "i4"), ("order_id", f"U{order_size}"), ("request_ns", "i8"),
        ("release", "i4"), ("pickup", "i4"), ("dropoff", "i4"), ("predicted", "f8"),
        ("mask", "u1"), ("common", "?"), ("data_ready", "?"), ("hard_state", "U12"),
        ("evidence", "?"), ("rho_static", "f8"), ("rho_dynamic", "f8"), ("rho_speed", "f8"), ("route_type", "U24")])
    orders = np.zeros(len(raw_requests), dtype=dtype)
    metrics = np.zeros(len(raw_requests), dtype=np.dtype([(name, "f8") for name in ROUTE_FIELDS]))
    truth = np.zeros(len(raw_requests), dtype="f8")
    route_rows = routes.set_index("order_id")
    for i, request in enumerate(raw_requests):
        route = route_rows.loc[request.order_id]
        common = bool(route["common_eligible"])
        mask = sum(1 << j for j, k in enumerate(("C", "M", "A")) if bool(route[f"compatible_{k}"]))
        predicted = float(route["predicted_route_time_p50_s"]) if common else request.predicted_service_time_s
        orders[i] = (request.native_id, request.order_id, request.request_time.value, request.sim_time_s,
            catalog.add(request.pickup_lon_wgs84, request.pickup_lat_wgs84), catalog.add(request.dropoff_lon_wgs84, request.dropoff_lat_wgs84),
            predicted, mask, common, bool(route["research_data_ready"]), request.hard_state, request.evidence_complete,
            request.rho_static, request.rho_dynamic, request.rho_speed, request.selected_route_type)
        metrics[i] = tuple(float(route[name]) for name in ROUTE_FIELDS)
        truth[i] = request.realized_service_time_s
    for name, array in (("orders", orders), ("route_metrics", metrics), ("simulation_truth", truth)):
        np.save(temp / (name + ".npy"), array, allow_pickle=False)
    forecast = build_forecast_product(temp / "forecast", templates, cfg, cfg["measurement_end_s"], base["passenger_acceptance_seed"], catalog)
    start = pd.Timestamp("2016-10-31T00:00:00+08:00")
    windows, events, fixtures = [], [], []
    for fixture in fleet.native_fixtures:
        catalog.add(fixture.initial_lon_wgs84, fixture.initial_lat_wgs84)
        a, b = ((getattr(fixture, k) - start).total_seconds() for k in ("availability_start_time", "availability_end_time"))
        windows.append((fixture.native_id, a, b))
        events.extend(((a, fixture.native_id, 0), (b, fixture.native_id, 1)))
        row = asdict(fixture)
        for key in ("availability_start_time", "availability_end_time"):
            row[key] = row[key].isoformat()
        fixtures.append(row)
    catalog.write(temp)
    np.save(temp / "fleet_windows.npy", np.asarray(windows, dtype=[("vid", "i4"), ("start", "f8"), ("end", "f8")]), allow_pickle=False)
    np.save(temp / "fleet_events.npy", np.asarray(sorted(events), dtype=[("time", "f8"), ("vid", "i4"), ("kind", "u1")]), allow_pickle=False)
    fleet.scenario_fleet.to_parquet(temp / "scenario_fleet.parquet", index=False)
    write_json(temp / "fleet.json", dict(fixtures=fixtures, accounting=fleet.accounting))
    products = {p.relative_to(temp).as_posix(): sha(p) for p in sorted(temp.rglob("*")) if p.is_file()}
    manifest = dict(version=ASSET_VERSION, status="COMPLETE", inputs_sha256=inputs, products_sha256=products,
        orders=len(orders), common_eligible=int(orders["common"].sum()), positions=len(catalog.rows),
        forecast_draws=forecast["total_draws"], fleet_fixtures=len(fixtures), routing_context=context,
        byte_count=sum(p.stat().st_size for p in temp.rglob("*") if p.is_file()),
        simulation_truth_separate=True, dense_matrix=False, gpu_used=False,
        current_spatial_reference="UNCHANGED_RUNTIME_MEAN_AVAILABLE_VEHICLE_LATITUDE",
        future_projection_latitude=34.25, mutable_vehicle_states_precomputed=False)
    write_json(temp / "manifest.json", manifest)
    temp.rename(directory)  # Same-filesystem atomic directory publication.
    return RuntimeAssets(directory, expected_inputs=inputs)
