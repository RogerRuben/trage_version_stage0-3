"""Arc-level deterministic pickup routing modes for Stage4 robustness runs."""

from __future__ import annotations

import time
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

    def __init__(self, *args: Any, routing_mode: str, **kwargs: Any) -> None:
        if routing_mode not in DETERMINISTIC_ROUTING_MODES:
            raise ValueError(f"unsupported deterministic routing mode: {routing_mode}")
        super().__init__(*args, **kwargs)
        self.routing_mode = routing_mode

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
            estimate = (
                self._single_source_matrix(vehicle, pickup_lon, pickup_lat, local, beta, bin_index)
                if self.routing_mode == SINGLE_SOURCE_MATRIX
                else self._scalar_route(vehicle, pickup_lon, pickup_lat, local)
            )
            if estimate is not None:
                self.cache[key] = estimate
                found[vehicle.native_vehicle_id] = estimate
        return found
