"""Build deterministic engineering-only HV/AV fixtures from frozen S0 sessions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .upstream import CoordinateRegistry, FleetPyBindings, FleetPyCompatibilityError

TIMEZONE = "Asia/Shanghai"
FLEET_REL = Path("stage4/input/replay_foundation/replay_fleet_template.parquet")


@dataclass(frozen=True)
class VehicleFixture:
    vehicle_id: str
    native_id: int
    vehicle_type: str
    initial_lon_wgs84: float
    initial_lat_wgs84: float
    availability_start_time: pd.Timestamp
    availability_end_time: pd.Timestamp
    source_session_id: str
    av_source_session_end_inherited: bool


@dataclass
class VehicleRuntime:
    fixture: VehicleFixture
    native_vehicle: Any
    state: str
    current_lon_wgs84: float
    current_lat_wgs84: float
    active_order_id: str | None = None
    next_event_time: pd.Timestamp | None = None
    pickup_eta_s: float | None = None

    def available_at(self, timestamp: pd.Timestamp) -> bool:
        return (
            self.state == "AVAILABLE"
            and timestamp >= self.fixture.availability_start_time
            and timestamp < self.fixture.availability_end_time
        )


def _priority(value: str, seed: int, namespace: str) -> str:
    return hashlib.sha256(
        f"{namespace}|{int(seed)}|{value}".encode("utf-8")
    ).hexdigest()


def load_mixed_fleet_fixture(
    root: str | Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    hv_count: int,
    av_count: int,
    seed: int,
) -> list[VehicleFixture]:
    path = Path(root).resolve() / FLEET_REL
    columns = [
        "source_session_id",
        "availability_start_time",
        "availability_end_time",
        "initial_lon_wgs84",
        "initial_lat_wgs84",
    ]
    frame = pd.read_parquet(path, columns=columns)
    for column in ("availability_start_time", "availability_end_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True).dt.tz_convert(TIMEZONE)
    overlap = frame.loc[
        frame["availability_start_time"].lt(end)
        & frame["availability_end_time"].gt(start)
    ].copy()
    overlap["_priority"] = (
        overlap["source_session_id"]
        .astype(str)
        .map(lambda value: _priority(value, seed, "HV"))
    )
    hv = overlap.sort_values(["_priority", "source_session_id"], kind="mergesort").head(
        int(hv_count)
    )
    if len(hv) != int(hv_count):
        raise FleetPyCompatibilityError("insufficient overlapping HV session fixtures")

    remaining = frame.loc[
        ~frame["source_session_id"].isin(hv["source_session_id"])
    ].copy()
    remaining["_priority"] = (
        remaining["source_session_id"]
        .astype(str)
        .map(lambda value: _priority(value, seed, "AV"))
    )
    av = remaining.sort_values(
        ["_priority", "source_session_id"], kind="mergesort"
    ).head(int(av_count))
    if len(av) != int(av_count):
        raise FleetPyCompatibilityError("insufficient empirical AV initial locations")

    fixtures: list[VehicleFixture] = []
    for native_id, row in enumerate(hv.itertuples(index=False)):
        fixtures.append(
            VehicleFixture(
                vehicle_id=f"HV_{native_id:03d}",
                native_id=native_id,
                vehicle_type="HV",
                initial_lon_wgs84=float(row.initial_lon_wgs84),
                initial_lat_wgs84=float(row.initial_lat_wgs84),
                availability_start_time=max(
                    pd.Timestamp(row.availability_start_time), start
                ),
                availability_end_time=min(pd.Timestamp(row.availability_end_time), end),
                source_session_id=str(row.source_session_id),
                av_source_session_end_inherited=False,
            )
        )
    for offset, row in enumerate(av.itertuples(index=False), start=len(fixtures)):
        fixtures.append(
            VehicleFixture(
                vehicle_id=f"AV_{offset - len(hv):03d}",
                native_id=offset,
                vehicle_type="AV",
                initial_lon_wgs84=float(row.initial_lon_wgs84),
                initial_lat_wgs84=float(row.initial_lat_wgs84),
                availability_start_time=start,
                availability_end_time=end,
                source_session_id=str(row.source_session_id),
                av_source_session_end_inherited=False,
            )
        )
    return fixtures


def create_native_vehicles(
    fixtures: list[VehicleFixture],
    bindings: FleetPyBindings,
    registry: CoordinateRegistry,
    request_db: dict[int, Any],
    runtime_dir: str | Path,
    *,
    native_movement: bool = False,
    routing_engine: Any | None = None,
) -> tuple[list[VehicleRuntime], list[dict[str, Any]]]:
    """Instantiate pinned upstream FleetPy vehicle objects."""
    runtime = Path(runtime_dir)
    vehicle_dir = runtime / "vehicles"
    vehicle_dir.mkdir(parents=True, exist_ok=True)
    vehicle_type = "stage4_fleetpy_spike_vehicle"
    pd.Series(
        {
            "vtype_name_full": vehicle_type,
            "maximum_passengers": 1,
            "daily_fix_cost [cent]": 0,
            "per_km_cost [cent]": 0,
            "battery_size [kWh]": 100,
            "range [km]": 100000,
            "source": "Stage4 S1 engineering compatibility fixture",
        }
    ).to_csv(vehicle_dir / f"{vehicle_type}.csv", header=False)

    native_output: list[dict[str, Any]] = []
    result: list[VehicleRuntime] = []
    engine = routing_engine or registry
    for fixture in fixtures:
        vehicle_class = (
            bindings.simulation_vehicle
            if native_movement
            else bindings.external_vehicle
        )
        vehicle = vehicle_class(
            0,
            fixture.native_id,
            str(vehicle_dir),
            vehicle_type,
            engine,
            request_db,
            native_output,
            False,
            False,
        )
        vehicle.pos = registry.position_for(
            fixture.initial_lon_wgs84, fixture.initial_lat_wgs84
        )
        vehicle.soc = 1.0
        state = (
            "AVAILABLE"
            if fixture.availability_start_time
            == min(item.availability_start_time for item in fixtures)
            else "INACTIVE"
        )
        if fixture.vehicle_type == "AV":
            state = "AVAILABLE"
        result.append(
            VehicleRuntime(
                fixture=fixture,
                native_vehicle=vehicle,
                state=state,
                current_lon_wgs84=fixture.initial_lon_wgs84,
                current_lat_wgs84=fixture.initial_lat_wgs84,
            )
        )
    return result, native_output
