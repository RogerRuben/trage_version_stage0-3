"""Sparse geographic candidate graph and Valhalla matrix pickup estimates."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from stage4.fleetpy_adapter.valhalla_time_adapter import (
    PickupEstimate,
    ValhallaPickupTimeAdapter,
)


def search_radius_m(
    failed_round_count: int,
    initial_m: float = 2000.0,
    step_m: float = 1000.0,
    cap_m: float = 8000.0,
) -> float:
    return min(
        float(initial_m) + float(step_m) * max(0, int(failed_round_count)), float(cap_m)
    )


def _xy(lon: np.ndarray, lat: np.ndarray, reference_lat: float) -> np.ndarray:
    scale_x = 111_320.0 * np.cos(np.deg2rad(reference_lat))
    return np.column_stack((lon * scale_x, lat * 110_540.0))


@dataclass(frozen=True)
class SpatialVehicle:
    vehicle_id: str
    native_vehicle_id: int
    vehicle_type: str
    lon_wgs84: float
    lat_wgs84: float


class SparseCandidateIndex:
    """Two cKDTree indexes, preserving the scientific HV/AV eligibility split."""

    def __init__(self, vehicles: Iterable[SpatialVehicle]) -> None:
        self.vehicles = list(vehicles)
        self.reference_lat = (
            float(np.mean([v.lat_wgs84 for v in self.vehicles]))
            if self.vehicles
            else 0.0
        )
        self._groups: dict[str, tuple[list[SpatialVehicle], cKDTree | None]] = {}
        for vehicle_type in ("HV", "AV"):
            group = sorted(
                (v for v in self.vehicles if v.vehicle_type == vehicle_type),
                key=lambda v: v.vehicle_id,
            )
            points = (
                _xy(
                    np.asarray([v.lon_wgs84 for v in group]),
                    np.asarray([v.lat_wgs84 for v in group]),
                    self.reference_lat,
                )
                if group
                else np.empty((0, 2))
            )
            self._groups[vehicle_type] = (group, cKDTree(points) if group else None)

    def query(
        self,
        pickup_lon: float,
        pickup_lat: float,
        radius_m: float,
        top_k: int,
        av_eligible: bool,
    ) -> tuple[list[tuple[SpatialVehicle, float]], int]:
        point = _xy(
            np.asarray([pickup_lon]), np.asarray([pickup_lat]), self.reference_lat
        )[0]
        candidates: list[tuple[SpatialVehicle, float]] = []
        for vehicle_type in ("HV", "AV") if av_eligible else ("HV",):
            group, tree = self._groups[vehicle_type]
            if tree is None:
                continue
            for index in tree.query_ball_point(point, float(radius_m)):
                vehicle = group[int(index)]
                distance = float(
                    np.linalg.norm(
                        _xy(
                            np.asarray([vehicle.lon_wgs84]),
                            np.asarray([vehicle.lat_wgs84]),
                            self.reference_lat,
                        )[0]
                        - point
                    )
                )
                candidates.append((vehicle, distance))
        spatial_count = len(candidates)
        candidates.sort(key=lambda item: (item[1], item[0].vehicle_id))
        return candidates[: int(top_k)], spatial_count

    def count_vehicle_type_within(
        self, pickup_lon: float, pickup_lat: float, radius_m: float, vehicle_type: str
    ) -> int:
        """Count pre-routing candidates for a compact pruning diagnostic."""
        point = _xy(
            np.asarray([pickup_lon]), np.asarray([pickup_lat]), self.reference_lat
        )[0]
        _group, tree = self._groups[vehicle_type]
        return 0 if tree is None else len(tree.query_ball_point(point, float(radius_m)))


class SparseValhallaMatrixAdapter(ValhallaPickupTimeAdapter):
    """Evaluate only sparse arcs, batching K vehicle sources to one pickup target."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.routing_queries = 0
        self.routing_arc_evaluations = 0
        self.routing_failures = 0
        self.matrix_failed_arcs = 0
        self.route_fallback_attempts = 0
        self.route_fallback_successes = 0
        self.route_fallback_failures = 0
        self.routing_time_s = 0.0
        self.failed_arc_records: list[dict[str, Any]] = []

    def _record_failed_arc(
        self,
        vehicle: SpatialVehicle,
        pickup_lon: float,
        pickup_lat: float,
        timestamp: pd.Timestamp,
        reason: str,
    ) -> None:
        self.failed_arc_records.append(
            {
                "origin_lon_wgs84": vehicle.lon_wgs84,
                "origin_lat_wgs84": vehicle.lat_wgs84,
                "pickup_lon_wgs84": float(pickup_lon),
                "pickup_lat_wgs84": float(pickup_lat),
                "timestamp_value": pd.Timestamp(timestamp).strftime("%Y-%m-%dT%H:%M"),
                "matrix_failure_reason": reason,
            }
        )

    def _route_fallback(
        self,
        vehicle: SpatialVehicle,
        pickup_lon: float,
        pickup_lat: float,
        timestamp: pd.Timestamp,
        matrix_cache_key: tuple,
    ) -> PickupEstimate | None:
        self.route_fallback_attempts += 1
        started = time.perf_counter()
        try:
            estimate = ValhallaPickupTimeAdapter.estimate(
                self,
                vehicle.lon_wgs84,
                vehicle.lat_wgs84,
                pickup_lon,
                pickup_lat,
                timestamp,
            )
        except Exception:
            self.route_fallback_failures += 1
            self.routing_failures += 1
            self.routing_time_s += time.perf_counter() - started
            return None
        self.routing_time_s += time.perf_counter() - started
        self.route_fallback_successes += 1
        self.cache[matrix_cache_key] = estimate
        return estimate

    def estimate_many(
        self,
        candidates: list[SpatialVehicle],
        pickup_lon: float,
        pickup_lat: float,
        timestamp: pd.Timestamp,
    ) -> dict[int, PickupEstimate]:
        if not candidates:
            return {}
        local = pd.Timestamp(timestamp).tz_convert("Asia/Shanghai")
        bin_index, beta = self.beta_for(local)
        found: dict[int, PickupEstimate] = {}
        missing: list[tuple[SpatialVehicle, tuple]] = []
        for vehicle in candidates:
            key = (
                round(vehicle.lon_wgs84, 7),
                round(vehicle.lat_wgs84, 7),
                round(float(pickup_lon), 7),
                round(float(pickup_lat), 7),
                local.strftime("%Y-%m-%dT%H:%M"),
                bin_index,
            )
            cached = self.cache.get(key)
            if cached is None:
                missing.append((vehicle, key))
            else:
                self.cache_hit_count += 1
                found[vehicle.native_vehicle_id] = PickupEstimate(
                    **{**cached.__dict__, "cache_hit": True}
                )
        if not missing:
            return found
        for vehicle, _ in missing:
            self._validate_wgs84(vehicle.lon_wgs84, vehicle.lat_wgs84)
        self._validate_wgs84(pickup_lon, pickup_lat)
        request = {
            "sources": [{"lon": v.lon_wgs84, "lat": v.lat_wgs84} for v, _ in missing],
            "targets": [{"lon": float(pickup_lon), "lat": float(pickup_lat)}],
            "costing": "auto",
            "units": "kilometers",
            "date_time": {"type": 1, "value": local.strftime("%Y-%m-%dT%H:%M")},
        }
        started = time.perf_counter()
        self.routing_queries += 1
        self.routing_arc_evaluations += len(missing)
        try:
            matrix = self.actor.matrix(request).get("sources_to_targets", [])
        except Exception as exc:
            self.matrix_failed_arcs += len(missing)
            self.routing_time_s += time.perf_counter() - started
            reason = f"MATRIX_BATCH_EXCEPTION:{type(exc).__name__}"
            for vehicle, key in missing:
                self._record_failed_arc(vehicle, pickup_lon, pickup_lat, local, reason)
                fallback = self._route_fallback(
                    vehicle, pickup_lon, pickup_lat, local, key
                )
                if fallback is not None:
                    found[vehicle.native_vehicle_id] = fallback
            return found
        self.routing_time_s += time.perf_counter() - started
        for source_index, (vehicle, key) in enumerate(missing):
            try:
                cell = matrix[source_index][0]
                raw_time = float(cell["time"])
                distance = float(cell["distance"]) * 1000.0
                if not np.isfinite(raw_time) or raw_time < 0:
                    raise ValueError("invalid matrix time")
            except (IndexError, KeyError, TypeError, ValueError):
                self.matrix_failed_arcs += 1
                self._record_failed_arc(
                    vehicle,
                    pickup_lon,
                    pickup_lat,
                    local,
                    "INVALID_MATRIX_CELL",
                )
                fallback = self._route_fallback(
                    vehicle, pickup_lon, pickup_lat, local, key
                )
                if fallback is not None:
                    found[vehicle.native_vehicle_id] = fallback
                continue
            estimate = PickupEstimate(
                raw_time, raw_time * beta, distance, beta, bin_index, False
            )
            self.cache[key] = estimate
            found[vehicle.native_vehicle_id] = estimate
        return found
