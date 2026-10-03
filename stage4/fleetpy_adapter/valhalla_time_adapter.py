"""FleetPy-facing pickup-time callback backed by frozen Valhalla + S0 beta."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .upstream import FleetPyCompatibilityError

TIMEZONE = "Asia/Shanghai"
CALIBRATION_REL = Path(
    "stage4/input/replay_foundation/pickup_eta_calibration_15min.parquet"
)
STAGE3_CONFIG_REL = Path("stage3/config/stage3_finalization.json")
ROUTING_COORDINATE_SYSTEM = "WGS84"


@dataclass(frozen=True)
class PickupEstimate:
    valhalla_time_s: float
    corrected_pickup_eta_s: float
    route_distance_m: float
    beta: float
    time_bin_index: int
    cache_hit: bool


class ValhallaPickupTimeAdapter:
    """One deterministic auto route plus the corrected frozen 15-minute beta."""

    def __init__(
        self,
        root: str | Path,
        *,
        actor: Any | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        calibration = pd.read_parquet(self.root / CALIBRATION_REL)
        required = {"time_bin_index", "selected_eta_multiplier"}
        if not required.issubset(calibration.columns) or len(calibration) != 96:
            raise FleetPyCompatibilityError(
                "corrected S0 calibration must contain 96 bins"
            )
        self._beta = (
            calibration.set_index("time_bin_index")["selected_eta_multiplier"]
            .astype(float)
            .to_dict()
        )
        if not np.isfinite(list(self._beta.values())).all():
            raise FleetPyCompatibilityError(
                "S0 pickup ETA beta contains non-finite values"
            )
        if actor is None:
            try:
                from valhalla import Actor
            except ImportError as exc:
                raise FleetPyCompatibilityError(
                    "Valhalla bindings are required"
                ) from exc
            stage3_config = json.loads(
                (self.root / STAGE3_CONFIG_REL).read_text(encoding="utf-8")
            )
            actor = Actor(str(Path(str(stage3_config["valhalla_config"])).resolve()))
        self.actor = actor
        self.cache: dict[tuple, PickupEstimate] = {}
        self.call_log: list[dict[str, Any]] = []
        self.cache_hit_count = 0

    @staticmethod
    def _validate_wgs84(lon: float, lat: float) -> None:
        if not (-180 <= float(lon) <= 180 and -90 <= float(lat) <= 90):
            raise FleetPyCompatibilityError(
                "Valhalla received invalid WGS84 coordinate"
            )

    def beta_for(self, timestamp: pd.Timestamp) -> tuple[int, float]:
        local = pd.Timestamp(timestamp)
        local = (
            local.tz_localize(TIMEZONE)
            if local.tzinfo is None
            else local.tz_convert(TIMEZONE)
        )
        index = int((local.hour * 60 + local.minute) // 15)
        return index, float(self._beta[index])

    def estimate(
        self,
        origin_lon_wgs84: float,
        origin_lat_wgs84: float,
        pickup_lon_wgs84: float,
        pickup_lat_wgs84: float,
        timestamp: pd.Timestamp,
    ) -> PickupEstimate:
        self._validate_wgs84(origin_lon_wgs84, origin_lat_wgs84)
        self._validate_wgs84(pickup_lon_wgs84, pickup_lat_wgs84)
        local = pd.Timestamp(timestamp).tz_convert(TIMEZONE)
        key = (
            round(float(origin_lon_wgs84), 7),
            round(float(origin_lat_wgs84), 7),
            round(float(pickup_lon_wgs84), 7),
            round(float(pickup_lat_wgs84), 7),
            local.strftime("%Y-%m-%dT%H:%M"),
        )
        if key in self.cache:
            self.cache_hit_count += 1
            cached = self.cache[key]
            return PickupEstimate(**{**cached.__dict__, "cache_hit": True})
        request = {
            "locations": [
                {
                    "lon": float(origin_lon_wgs84),
                    "lat": float(origin_lat_wgs84),
                    "type": "break",
                },
                {
                    "lon": float(pickup_lon_wgs84),
                    "lat": float(pickup_lat_wgs84),
                    "type": "break",
                },
            ],
            "costing": "auto",
            "units": "kilometers",
            "directions_type": "none",
            "date_time": {"type": 1, "value": local.strftime("%Y-%m-%dT%H:%M")},
        }
        trip = self.actor.route(request)["trip"]
        if int(trip.get("status", 0)) != 0 or len(trip.get("legs", [])) != 1:
            raise FleetPyCompatibilityError(
                "Valhalla did not return one successful leg"
            )
        summary = trip["summary"]
        raw_time = float(summary["time"])
        distance = float(summary["length"]) * 1000.0
        bin_index, beta = self.beta_for(local)
        estimate = PickupEstimate(
            valhalla_time_s=raw_time,
            corrected_pickup_eta_s=raw_time * beta,
            route_distance_m=distance,
            beta=beta,
            time_bin_index=bin_index,
            cache_hit=False,
        )
        self.cache[key] = estimate
        self.call_log.append(
            {
                "timestamp": local,
                "origin_lon_wgs84": float(origin_lon_wgs84),
                "origin_lat_wgs84": float(origin_lat_wgs84),
                "pickup_lon_wgs84": float(pickup_lon_wgs84),
                "pickup_lat_wgs84": float(pickup_lat_wgs84),
                "routing_coordinate_system": ROUTING_COORDINATE_SYSTEM,
                "valhalla_time_s": raw_time,
                "route_distance_m": distance,
                "beta": beta,
                "corrected_pickup_eta_s": raw_time * beta,
            }
        )
        return estimate
