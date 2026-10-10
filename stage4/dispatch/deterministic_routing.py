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
                 route_workers: int = 1, disk_cache_path=None, routing_context=None,
                 disk_cache_limit_mib=512, disk_cache_max_entries=2_000_000,
                 route_queue_chunk_size=256, lazy_worker_actor=False, **kwargs: Any) -> None:
        if routing_mode not in DETERMINISTIC_ROUTING_MODES:
            raise ValueError(f"unsupported deterministic routing mode: {routing_mode}")
        if not 1 <= route_workers <= 4 or persistent_cache_size < 0:
            raise ValueError("routing acceleration limits are invalid")
        if route_workers > 1 and kwargs.get("actor") is not None:
            raise ValueError("routing workers require the real frozen actor, not an injected mock")
        self.lazy_worker_actor = bool(lazy_worker_actor)
        if self.lazy_worker_actor and route_workers > 1:
            kwargs["defer_actor"] = True
        super().__init__(*args, **kwargs)
        self.routing_mode = routing_mode
        self.route_workers = int(route_workers)
        self.persistent_cache_size = int(persistent_cache_size)
        self._persistent_cache = OrderedDict()
        self._executor = None
        self.persistent_cache_hits = 0
        self.routing_backend_work_time_s = 0.0
        self.arc_lookup_count = 0
        if not 1 <= route_queue_chunk_size <= 512:
            raise ValueError("routing queue chunk exceeds the memory bound")
        self.route_queue_chunk_size = int(route_queue_chunk_size)
        self._disk_cache = None
        if disk_cache_path is not None:
            if not routing_context:
                raise ValueError("disk routing cache requires a frozen engine/tiles context")
            from .sparse_route_cache import SparseRouteDiskCache
            self._disk_cache = SparseRouteDiskCache(disk_cache_path, routing_context,
                limit_mib=disk_cache_limit_mib, max_entries=disk_cache_max_entries)
        self.disk_cache_hits = 0
        self.raw_lru_hits = 0
        self.raw_query_deduplications = 0
        self.epoch_route_batches = 0
        self.prefetched_unused_raw_arcs = 0
        self.routing_queue_pipeline_time_s = 0.0
        self.disk_cache_lookup_time_s = 0.0

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
        if self._disk_cache is not None:
            self._disk_cache.close()
            if hasattr(self._disk_cache, "diagnostics"):
                self._closed_disk_cache_diagnostics = self._disk_cache.diagnostics()
            self._disk_cache = None
        if self.lazy_worker_actor:
            self._actor = None

    def _raw_key(self, vehicle, lon, lat, local):
        return (self.routing_mode, *(float(x).hex() for x in
            (vehicle.lon_wgs84, vehicle.lat_wgs84, lon, lat)), local.strftime("%Y-%m-%dT%H:%M"))

    def _raw_query(self, payload):
        mode, olon, olat, tlon, tlat, minute = payload
        request = {"costing": "auto", "units": "kilometers", "date_time": {"type": 1, "value": minute}}
        started = time.perf_counter()
        try:
            if mode == SINGLE_SOURCE_MATRIX:
                request.update(sources=[{"lon": olon, "lat": olat}], targets=[{"lon": tlon, "lat": tlat}])
                cell = self.actor.matrix(request)["sources_to_targets"][0][0]
                raw, distance = float(cell["time"]), float(cell["distance"]) * 1000.
            else:
                request.update(locations=[{"lon": olon, "lat": olat, "type": "break"},
                    {"lon": tlon, "lat": tlat, "type": "break"}], directions_type="none")
                trip = self.actor.route(request)["trip"]
                if int(trip.get("status", 0)) or len(trip.get("legs", [])) != 1:
                    raise ValueError("invalid scalar route")
                raw, distance = float(trip["summary"]["time"]), float(trip["summary"]["length"]) * 1000.
            if not np.isfinite(raw) or raw < 0 or not np.isfinite(distance) or distance < 0:
                raise ValueError("invalid raw routing estimate")
            return raw, distance, None, time.perf_counter() - started
        except Exception as error:
            return None, None, type(error).__name__, time.perf_counter() - started

    def estimate_epoch(self, batches):
        """Bounded sparse queue; replay cache fills in original request order.

        Each primitive stays 1x1. Exact queries can share raw answers; the
        legacy rounded per-epoch aliases are still resolved in original order.
        No all-vehicle/all-request matrix is constructed.
        """
        if not batches:
            return []
        # A queue is bounded even if a future caller presents an unusually
        # bursty day. Blocks retain the original request/cache-fill order.
        if sum(len(batch[0]) for batch in batches) > 50_000:
            found, block, size = [], [], 0
            for batch in batches:
                if len(batch[0]) > 50_000:
                    raise ValueError("single sparse routing batch exceeds 50000 arcs")
                if block and size + len(batch[0]) > 50_000:
                    found.extend(self.estimate_epoch(block))
                    block, size = [], 0
                block.append(batch)
                size += len(batch[0])
            return found + self.estimate_epoch(block)
        self.epoch_route_batches += 1
        started = time.perf_counter()
        plans, unique, exact_keys = [], OrderedDict(), {}
        for candidates, lon, lat, timestamp in batches:
            local = pd.Timestamp(timestamp).tz_convert("Asia/Shanghai")
            bin_index, beta = self.beta_for(local)
            rows = []
            for vehicle in candidates:
                self._validate_wgs84(vehicle.lon_wgs84, vehicle.lat_wgs84)
                self._validate_wgs84(lon, lat)
                key = self._raw_key(vehicle, lon, lat, local)
                matrix_key = self._matrix_key(vehicle, lon, lat, local, bin_index)
                rows.append((vehicle, matrix_key, key))
                exact_keys[key] = self._persistent_key(vehicle, lon, lat, local, bin_index, beta)
                # Existing rounded aliases are already resolved, not promoted
                # into exact persistent entries for a different coordinate.
                if matrix_key not in self.cache:
                    unique.setdefault(key, (self.routing_mode, float(vehicle.lon_wgs84), float(vehicle.lat_wgs84),
                        float(lon), float(lat), local.strftime("%Y-%m-%dT%H:%M")))
            plans.append((rows, local, bin_index, beta, lon, lat))
        answers, origins = {}, {}
        for key in unique:
            cached = self._persistent_cache.get(exact_keys[key])
            if cached is not None:
                answers[key] = (cached.valhalla_time_s, cached.route_distance_m, None)
                origins[key] = "RAM"
        missing = [key for key in unique if key not in answers]
        if self._disk_cache is not None and missing:
            disk_started = time.perf_counter()
            disk_keys = {self._disk_cache.key(key): key for key in missing}
            for digest, value in self._disk_cache.get_many(disk_keys).items():
                key = disk_keys[digest]
                answers[key] = (*value, None)
                origins[key] = "DISK"
            self.disk_cache_lookup_time_s += time.perf_counter() - disk_started
        missing = [key for key in unique if key not in answers]
        backend_wall = 0.0
        for left in range(0, len(missing), self.route_queue_chunk_size):
            keys = missing[left:left + self.route_queue_chunk_size]
            payloads = [unique[key] for key in keys]
            backend_started = time.perf_counter()
            if self.route_workers > 1:
                from .routing_workers import route_query
                results = list(self._worker_pool().map(route_query, payloads, chunksize=8))
            else:
                results = list(map(self._raw_query, payloads))
            backend_wall += time.perf_counter() - backend_started
            for key, (raw, distance, error, work_s) in zip(keys, results):
                self.routing_queries += 1
                self.routing_arc_evaluations += 1
                self.routing_backend_work_time_s += work_s
                answers[key] = (raw, distance, error)
                origins[key] = "COMPUTED"
                if error is None:
                    if self._disk_cache is not None:
                        self._disk_cache.remember(self._disk_cache.key(key), raw, distance)
        found_batches, consumed = [], set()
        for rows, local, bin_index, beta, lon, lat in plans:
            found = {}
            # Snapshot first, like the old estimate_many: within-batch rounded
            # aliases do not depend on the order in which workers complete.
            legacy_hits = {vid.native_vehicle_id: self.cache[key]
                           for vid, key, _ in rows if key in self.cache}
            uncached_rows = []
            for vehicle, matrix_key, raw_key in rows:
                self.arc_lookup_count += 1
                estimate = legacy_hits.get(vehicle.native_vehicle_id)
                if estimate is not None:
                    self.cache_hit_count += 1
                    found[vehicle.native_vehicle_id] = PickupEstimate(**{**estimate.__dict__, "cache_hit": True})
                    continue
                # Same two-phase fill as v1: exact RAM hits first, then all
                # logical misses, irrespective of disk/prefetch answer source.
                # A warm disk cache must not change rounded alias precedence.
                cached = self._persistent_cache.get(exact_keys[raw_key])
                if cached is not None:
                    self._persistent_cache.move_to_end(exact_keys[raw_key])
                    self.persistent_cache_hits += 1
                    self.raw_lru_hits += 1
                    self.cache_hit_count += 1
                    self.cache[matrix_key] = cached
                    found[vehicle.native_vehicle_id] = PickupEstimate(**{**cached.__dict__, "cache_hit": True})
                else:
                    uncached_rows.append((vehicle, matrix_key, raw_key))
            for vehicle, matrix_key, raw_key in uncached_rows:
                raw, distance, error = answers[raw_key]
                origin = origins[raw_key]
                was_consumed = raw_key in consumed
                consumed.add(raw_key)
                if error is not None and was_consumed:
                    # A failed primitive is not reusable. A second logical
                    # lookup behaves as it did in the sequential adapter.
                    retry_started = time.perf_counter()
                    if self.route_workers > 1:
                        from .routing_workers import route_query
                        raw, distance, error, work_s = next(self._worker_pool().map(route_query, [unique[raw_key]], chunksize=1))
                    else:
                        raw, distance, error, work_s = self._raw_query(unique[raw_key])
                    backend_wall += time.perf_counter() - retry_started
                    self.routing_queries += 1
                    self.routing_arc_evaluations += 1
                    self.routing_backend_work_time_s += work_s
                    if error is None:
                        answers[raw_key] = (raw, distance, None)
                        if self._disk_cache is not None:
                            self._disk_cache.remember(self._disk_cache.key(raw_key), raw, distance)
                if error is not None:
                    self.routing_failures += 1
                    self.matrix_failed_arcs += int(self.routing_mode == SINGLE_SOURCE_MATRIX)
                    self._record_failed_arc(vehicle, lon, lat, local, error)
                    continue  # Failures are never saved on disk or in the LRU.
                hit = origin != "COMPUTED"
                self.cache_hit_count += int(hit)
                self.raw_lru_hits += int(origin == "RAM")
                self.disk_cache_hits += int(origin == "DISK")
                self.persistent_cache_hits += int(origin in ("RAM", "DISK"))
                self.raw_query_deduplications += int(origin == "COMPUTED" and was_consumed)
                estimate = PickupEstimate(raw, raw * beta, distance, beta, bin_index, hit)
                stored = PickupEstimate(raw, raw * beta, distance, beta, bin_index, False)
                self.cache[matrix_key] = stored
                self._remember(exact_keys[raw_key], stored)
                found[vehicle.native_vehicle_id] = estimate
            found_batches.append(found)
        self.prefetched_unused_raw_arcs += sum(origins[key] == "COMPUTED" and key not in consumed for key in unique)
        self.routing_time_s += backend_wall  # Same backend-wall unit as v1.
        self.routing_queue_pipeline_time_s += time.perf_counter() - started
        return found_batches

    def process_group_rss_mib(self):
        return self.process_group_resources()["rss_mib"]

    def process_group_resources(self):
        import psutil
        process = psutil.Process()
        total = private = 0
        for child in (process, *process.children(recursive=True)):
            try:
                info = child.memory_info()
                total += info.rss
                private += getattr(info, "private", info.vms)
            except psutil.NoSuchProcess:
                pass
        return dict(rss_mib=total / 2**20, private_committed_mib=private / 2**20)

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
        if self._disk_cache is not None:
            return self.estimate_epoch([(candidates, pickup_lon, pickup_lat, timestamp)])[0]
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
