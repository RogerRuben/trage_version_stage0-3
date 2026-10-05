"""Arc-level deterministic pickup routing modes for Stage4 robustness runs."""

from __future__ import annotations

import time
import json
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage4.fleetpy_adapter.valhalla_time_adapter import PickupEstimate, ValhallaPickupTimeAdapter

from .candidate_graph import SparseValhallaMatrixAdapter, SpatialVehicle

SINGLE_SOURCE_MATRIX = "SINGLE_SOURCE_MATRIX"
SCALAR_ROUTE = "SCALAR_ROUTE"
DETERMINISTIC_ROUTING_MODES = (SINGLE_SOURCE_MATRIX, SCALAR_ROUTE)


class ArcDeterministicValhallaAdapter(SparseValhallaMatrixAdapter):
    """Route each cache-missing arc independently of unrelated candidates."""

    def __init__(self, *args: Any, routing_mode: str, persistent_cache_size: int = 0,
                 route_workers: int = 1, **kwargs: Any) -> None:
        if routing_mode not in DETERMINISTIC_ROUTING_MODES:
            raise ValueError(f"unsupported deterministic routing mode: {routing_mode}")
        if not 1 <= route_workers <= 4 or persistent_cache_size < 0:
            raise ValueError("routing acceleration limits are invalid")
        if route_workers > 1 and kwargs.get("actor") is not None:
            raise ValueError("routing workers require the real frozen actor, not an injected mock")
        super().__init__(*args, **kwargs)
        self.routing_mode = routing_mode
        self.route_workers = int(route_workers)
        self.persistent_cache_size = int(persistent_cache_size)
        self._persistent_cache = OrderedDict()
        self._executor = None
        self.persistent_cache_hits = 0
        self.routing_backend_work_time_s = 0.0
        self.arc_lookup_count = 0

    def _persistent_key(self, vehicle, lon, lat, local, bin_index, beta):
        # Full query precision, rather than the legacy per-epoch rounded key.
        return (self.routing_mode, *(float(x).hex() for x in
                (vehicle.lon_wgs84, vehicle.lat_wgs84, lon, lat)),
                local.strftime("%Y-%m-%dT%H:%M"), int(bin_index), float(beta).hex())

    def _remember(self, key, estimate):
        if self.persistent_cache_size and not estimate.cache_hit:
            self._persistent_cache[key] = estimate
            self._persistent_cache.move_to_end(key)
            while len(self._persistent_cache) > self.persistent_cache_size:
                self._persistent_cache.popitem(last=False)

    def _worker_pool(self):
        if self._executor is None:
            from .routing_workers import initialize_worker
            cfg = json.loads((self.root / "stage3/config/stage3_finalization.json").read_text(encoding="utf-8"))
            self._executor = ProcessPoolExecutor(max_workers=self.route_workers,
                mp_context=multiprocessing.get_context("spawn"), initializer=initialize_worker,
                initargs=(str(Path(cfg["valhalla_config"]).resolve()),))
        return self._executor

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def process_group_rss_mib(self):
        import psutil
        process = psutil.Process()
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return total / 2**20

    @staticmethod
    def _matrix_key(vehicle: SpatialVehicle, pickup_lon: float, pickup_lat: float, local: pd.Timestamp, bin_index: int) -> tuple:
        return (
            round(vehicle.lon_wgs84, 7), round(vehicle.lat_wgs84, 7),
            round(float(pickup_lon), 7), round(float(pickup_lat), 7),
            local.strftime("%Y-%m-%dT%H:%M"), int(bin_index),
        )

    def _single_source_matrix(self, vehicle: SpatialVehicle, pickup_lon: float, pickup_lat: float, local: pd.Timestamp, beta: float, bin_index: int) -> PickupEstimate | None:
        request = {
            "sources": [{"lon": vehicle.lon_wgs84, "lat": vehicle.lat_wgs84}],
            "targets": [{"lon": float(pickup_lon), "lat": float(pickup_lat)}],
            "costing": "auto", "units": "kilometers",
            "date_time": {"type": 1, "value": local.strftime("%Y-%m-%dT%H:%M")},
        }
        self.routing_queries += 1
        self.routing_arc_evaluations += 1
        started = time.perf_counter()
        try:
            matrix = self.actor.matrix(request).get("sources_to_targets", [])
            cell = matrix[0][0]
            raw_time = float(cell["time"])
            distance = float(cell["distance"]) * 1000.0
            if not np.isfinite(raw_time) or raw_time < 0.0:
                raise ValueError("invalid matrix time")
        except Exception as exc:
            self.matrix_failed_arcs += 1
            self.routing_failures += 1
            self._record_failed_arc(vehicle, pickup_lon, pickup_lat, local, f"SINGLE_SOURCE_MATRIX:{type(exc).__name__}")
            return None
        finally:
            self.routing_time_s += time.perf_counter() - started
        return PickupEstimate(raw_time, raw_time * float(beta), distance, float(beta), int(bin_index), False)

    def _scalar_route(self, vehicle: SpatialVehicle, pickup_lon: float, pickup_lat: float, local: pd.Timestamp) -> PickupEstimate | None:
        self.routing_queries += 1
        self.routing_arc_evaluations += 1
        started = time.perf_counter()
        try:
            return ValhallaPickupTimeAdapter.estimate(self, vehicle.lon_wgs84, vehicle.lat_wgs84, pickup_lon, pickup_lat, local)
        except Exception as exc:
            self.routing_failures += 1
            self._record_failed_arc(vehicle, pickup_lon, pickup_lat, local, f"SCALAR_ROUTE:{type(exc).__name__}")
            return None
        finally:
            self.routing_time_s += time.perf_counter() - started

    def estimate_many(self, candidates: list[SpatialVehicle], pickup_lon: float, pickup_lat: float, timestamp: pd.Timestamp) -> dict[int, PickupEstimate]:
        if not candidates:
            return {}
        self.arc_lookup_count += len(candidates)
        local = pd.Timestamp(timestamp).tz_convert("Asia/Shanghai")
        bin_index, beta = self.beta_for(local)
        found: dict[int, PickupEstimate] = {}
        missing: list[tuple[SpatialVehicle, tuple]] = []
        for vehicle in candidates:
            key = self._matrix_key(vehicle, pickup_lon, pickup_lat, local, bin_index)
            cached = self.cache.get(key)
            if cached is None:
                missing.append((vehicle, key))
            else:
                self.cache_hit_count += 1
                found[vehicle.native_vehicle_id] = PickupEstimate(**{**cached.__dict__, "cache_hit": True})
        for vehicle, key in missing:
            self._validate_wgs84(vehicle.lon_wgs84, vehicle.lat_wgs84)
            self._validate_wgs84(pickup_lon, pickup_lat)
        uncached = []
        for vehicle, key in missing:
            exact_key = self._persistent_key(vehicle, pickup_lon, pickup_lat, local, bin_index, beta)
            cached = self._persistent_cache.get(exact_key)
            if cached is not None:
                self._persistent_cache.move_to_end(exact_key)
                self.persistent_cache_hits += 1
                self.cache_hit_count += 1
                self.cache[key] = cached
                found[vehicle.native_vehicle_id] = PickupEstimate(**{**cached.__dict__, "cache_hit": True})
            else:
                uncached.append((vehicle, key, exact_key))
        if self.route_workers > 1 and uncached:
            from .routing_workers import route_query
            minute = local.strftime("%Y-%m-%dT%H:%M")
            payloads = [(self.routing_mode, float(v.lon_wgs84), float(v.lat_wgs84),
                         float(pickup_lon), float(pickup_lat), minute) for v, _, _ in uncached]
            started = time.perf_counter()
            results = list(self._worker_pool().map(route_query, payloads,
                chunksize=max(1, min(8, len(payloads) // (self.route_workers * 2)))))
            # Wall time, not the sum of concurrent workers' durations.
            self.routing_time_s += time.perf_counter() - started
            for (vehicle, key, exact_key), (raw, distance, error, work_s) in zip(uncached, results):
                self.routing_queries += 1
                self.routing_arc_evaluations += 1
                self.routing_backend_work_time_s += work_s
                if error is not None:
                    self.routing_failures += 1
                    self.matrix_failed_arcs += int(self.routing_mode == SINGLE_SOURCE_MATRIX)
                    self._record_failed_arc(vehicle, pickup_lon, pickup_lat, local, error)
                    continue
                estimate = PickupEstimate(raw, raw * beta, distance, beta, bin_index, False)
                self.cache[key] = estimate
                self._remember(exact_key, estimate)
                found[vehicle.native_vehicle_id] = estimate
            return found
        for vehicle, key, exact_key in uncached:
            estimate = (
                self._single_source_matrix(vehicle, pickup_lon, pickup_lat, local, beta, bin_index)
                if self.routing_mode == SINGLE_SOURCE_MATRIX
                else self._scalar_route(vehicle, pickup_lon, pickup_lat, local)
            )
            if estimate is not None:
                self.cache[key] = estimate
                self._remember(exact_key, estimate)
                found[vehicle.native_vehicle_id] = estimate
        return found
