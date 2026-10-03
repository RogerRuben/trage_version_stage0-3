"""Populate FleetPy Demand.future_requests with the deterministic Test31 fixture."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .replay_service_time_adapter import _haversine_m
from .test31_demand_adapter import SpikeRequest
from .upstream import CoordinateRegistry, FleetPyBindings


def create_native_demand(
    bindings: FleetPyBindings,
    requests: list[SpikeRequest],
    registry: CoordinateRegistry,
    network: Any,
    output_dir: str | Path,
) -> Any:
    """Use FleetPy Demand as the sole request activation database."""
    demand = bindings.demand(
        {"skip_output": 1},
        str(Path(output_dir) / "fleetpy_native_user_stats.csv"),
        routing_engine=network,
    )
    for request in requests:
        registry.set_leg_metrics(
            request.pickup_position,
            request.dropoff_position,
            request.realized_service_time_s,
            _haversine_m(
                request.pickup_lon_wgs84,
                request.pickup_lat_wgs84,
                request.dropoff_lon_wgs84,
                request.dropoff_lat_wgs84,
            ),
        )
        demand.future_requests.setdefault(request.sim_time_s, {})[
            request.native_id
        ] = request.native_request
    return demand
