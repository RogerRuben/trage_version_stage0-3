"""Routing facade used by v3 matching and execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EARTH_M = 6_371_000.0


def haversine_m(lon1, lat1, lon2, lat2) -> float:
    lon1 = np.radians(float(lon1))
    lat1 = np.radians(float(lat1))
    lon2 = np.radians(float(lon2))
    lat2 = np.radians(float(lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * EARTH_M * np.arcsin(np.sqrt(a)))


@dataclass
class RouteResult:
    road_distance_m: float
    expected_travel_time_sec: float
    realized_travel_time_sec: float | None
    route_nodes: list[Any]
    route_links: list[Any]
    route_source: str


class RoutingEngine:
    """Two-stage routing facade.

    BallTree/candidate generation may use coarse haversine distances, but final
    pickup ETA and execution legs are obtained through this class.
    """

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.query_count = 0
        self.cache_hit_count = 0
        self._cache: dict[tuple, RouteResult] = {}
        speed = pd.read_parquet(data_root / "pickup_empty_speed_by_zone_time.parquet")
        self.speed_map = {(str(r.origin_zone), int(r.time_bin)): float(r.empty_speed_mps) for r in speed.itertuples(index=False)}
        self.global_speed = float(speed["empty_speed_mps"].median()) if len(speed) else 6.0
        circ_path = data_root / "pickup_circuity_by_zone.parquet"
        if circ_path.exists():
            circ = pd.read_parquet(circ_path)
            self.circuity_map = {str(r.origin_zone): float(r.circuity_factor) for r in circ.itertuples(index=False)}
            self.global_circuity = float(circ["circuity_factor"].median()) if len(circ) else 1.35
        else:
            self.circuity_map = {}
            self.global_circuity = 1.35

    def query(
        self,
        origin: tuple[float, float, str],
        destination: tuple[float, float, str],
        departure_time: pd.Timestamp,
        vehicle_type: str,
        time_bin: int | None = None,
        realized_time_sec: float | None = None,
        route_source_hint: str = "zone_time_empty_speed_prior",
    ) -> RouteResult:
        origin_lon, origin_lat, origin_zone = origin
        dest_lon, dest_lat, dest_zone = destination
        bin_value = int(time_bin if time_bin is not None else pd.Timestamp(departure_time).hour * 2)
        key = (origin_zone, dest_zone, bin_value, vehicle_type, round(float(origin_lon), 4), round(float(origin_lat), 4), round(float(dest_lon), 4), round(float(dest_lat), 4))
        self.query_count += 1
        if key in self._cache:
            self.cache_hit_count += 1
            cached = self._cache[key]
            return RouteResult(
                road_distance_m=cached.road_distance_m,
                expected_travel_time_sec=cached.expected_travel_time_sec,
                realized_travel_time_sec=realized_time_sec if realized_time_sec is not None else cached.realized_travel_time_sec,
                route_nodes=[],
                route_links=[],
                route_source=cached.route_source + "+cache_hit",
            )
        straight = haversine_m(origin_lon, origin_lat, dest_lon, dest_lat)
        circuity = min(max(float(self.circuity_map.get(str(origin_zone), self.global_circuity)), 1.05), 2.5)
        speed = min(max(float(self.speed_map.get((str(origin_zone), bin_value), self.global_speed)), 3.0), 15.0)
        distance = straight * circuity
        result = RouteResult(
            road_distance_m=distance,
            expected_travel_time_sec=distance / speed,
            realized_travel_time_sec=realized_time_sec,
            route_nodes=[],
            route_links=[],
            route_source=route_source_hint,
        )
        self._cache[key] = result
        return result

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hit_count / self.query_count if self.query_count else 0.0

