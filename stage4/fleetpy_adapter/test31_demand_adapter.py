"""Adapt frozen Test31 replay rows into pinned FleetPy BasicRequest objects."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .upstream import CoordinateRegistry, FleetPyBindings, FleetPyCompatibilityError

TIMEZONE = "Asia/Shanghai"
ORDER_BASE_REL = Path("stage4/input/replay_foundation/stage4_order_replay_base.parquet")


@dataclass
class SpikeRequest:
    native_id: int
    order_id: str
    request_time: pd.Timestamp
    sim_time_s: int
    pickup_lon_wgs84: float
    pickup_lat_wgs84: float
    dropoff_lon_wgs84: float
    dropoff_lat_wgs84: float
    realized_service_time_s: float
    predicted_service_time_s: float
    profile_id: str
    hard_state: str
    evidence_complete: bool
    rho_static: float
    rho_dynamic: float
    rho_speed: float
    native_request: Any = None
    pickup_position: tuple | None = None
    dropoff_position: tuple | None = None

    @property
    def av_smoke_eligible(self) -> bool:
        return self.hard_state == "FEASIBLE" and self.evidence_complete


def _priority(order_id: str, seed: int) -> str:
    return hashlib.sha256(f"{int(seed)}|{order_id}".encode("utf-8")).hexdigest()


def load_test31_requests(
    root: str | Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    profile_id: str,
    request_count: int,
    seed: int,
) -> list[SpikeRequest]:
    """Select a deterministic engineering fixture while preserving historical times."""
    path = Path(root).resolve() / ORDER_BASE_REL
    required = [
        "order_id",
        "request_time",
        "pickup_lon_wgs84",
        "pickup_lat_wgs84",
        "dropoff_lon_wgs84",
        "dropoff_lat_wgs84",
        "realized_service_time_s",
        "predicted_service_time_s",
        "profile_id",
        "hard_state",
        "evidence_complete",
        "rho_static",
        "rho_dynamic",
        "rho_speed",
    ]
    frame = pd.read_parquet(path, columns=required)
    frame["request_time"] = pd.to_datetime(
        frame["request_time"], utc=True
    ).dt.tz_convert(TIMEZONE)
    frame = frame.loc[
        frame["profile_id"].astype(str).eq(str(profile_id))
        & frame["request_time"].ge(start)
        & frame["request_time"].lt(end)
    ].copy()
    numeric_columns = [
        "pickup_lon_wgs84",
        "pickup_lat_wgs84",
        "dropoff_lon_wgs84",
        "dropoff_lat_wgs84",
        "realized_service_time_s",
        "predicted_service_time_s",
    ]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    basic_smoke_eligible = np.isfinite(numeric).all(axis=1)
    basic_smoke_eligible &= numeric["realized_service_time_s"].gt(0)
    basic_smoke_eligible &= numeric["predicted_service_time_s"].gt(0)
    frame = frame.loc[basic_smoke_eligible].copy()
    if frame["order_id"].astype(str).duplicated().any():
        raise FleetPyCompatibilityError(
            "profile-specific Test31 order_id is not unique"
        )
    frame["_priority"] = (
        frame["order_id"].astype(str).map(lambda value: _priority(value, seed))
    )
    frame = frame.sort_values(["_priority", "order_id"], kind="mergesort").head(
        int(request_count)
    )
    if len(frame) != int(request_count):
        raise FleetPyCompatibilityError(
            f"requested {request_count} spike requests, found {len(frame)}"
        )
    frame = frame.sort_values(["request_time", "order_id"], kind="mergesort")
    records: list[SpikeRequest] = []
    for native_id, row in enumerate(frame.itertuples(index=False)):
        seconds = (pd.Timestamp(row.request_time) - start).total_seconds()
        if abs(seconds - round(seconds)) > 1e-6:
            raise FleetPyCompatibilityError(
                "FleetPy spike requires second-aligned requests"
            )
        records.append(
            SpikeRequest(
                native_id=native_id,
                order_id=str(row.order_id),
                request_time=pd.Timestamp(row.request_time),
                sim_time_s=int(round(seconds)),
                pickup_lon_wgs84=float(row.pickup_lon_wgs84),
                pickup_lat_wgs84=float(row.pickup_lat_wgs84),
                dropoff_lon_wgs84=float(row.dropoff_lon_wgs84),
                dropoff_lat_wgs84=float(row.dropoff_lat_wgs84),
                realized_service_time_s=float(row.realized_service_time_s),
                predicted_service_time_s=float(row.predicted_service_time_s),
                profile_id=str(row.profile_id),
                hard_state=str(row.hard_state),
                evidence_complete=bool(row.evidence_complete),
                rho_static=float(row.rho_static),
                rho_dynamic=float(row.rho_dynamic),
                rho_speed=float(row.rho_speed),
            )
        )
    return records


def attach_fleetpy_requests(
    requests: Iterable[SpikeRequest],
    bindings: FleetPyBindings,
    registry: CoordinateRegistry,
) -> dict[int, Any]:
    """Construct real upstream BasicRequest instances with retained capability fields."""
    request_db: dict[int, Any] = {}
    for record in requests:
        record.pickup_position = registry.position_for(
            record.pickup_lon_wgs84, record.pickup_lat_wgs84
        )
        record.dropoff_position = registry.position_for(
            record.dropoff_lon_wgs84, record.dropoff_lat_wgs84
        )
        row = pd.Series(
            {
                "request_id": record.native_id,
                "rq_time": record.sim_time_s,
                "latest_decision_time": record.sim_time_s,
                "start": int(record.pickup_position[0]),
                "end": int(record.dropoff_position[0]),
                "source_order_id": record.order_id,
                "realized_service_time_s": record.realized_service_time_s,
                "predicted_service_time_s": record.predicted_service_time_s,
                "profile_id": record.profile_id,
                "hard_state": record.hard_state,
                "evidence_complete": record.evidence_complete,
                "rho_static": record.rho_static,
                "rho_dynamic": record.rho_dynamic,
                "rho_speed": record.rho_speed,
            }
        )
        native = bindings.basic_request(row, registry, 1, {})
        record.native_request = native
        request_db[record.native_id] = native
    return request_db


def load_all_test31_requests(
    root: str | Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    profile_id: str,
) -> list[SpikeRequest]:
    """Load full bounded Test31 demand without future-outcome filtering.

    Missing decision-time service predictions are retained. They cannot form HV
    admission arcs; realized service time is used only after valid assignment.
    """
    path = Path(root).resolve() / ORDER_BASE_REL
    required = [
        "order_id",
        "request_time",
        "pickup_lon_wgs84",
        "pickup_lat_wgs84",
        "dropoff_lon_wgs84",
        "dropoff_lat_wgs84",
        "realized_service_time_s",
        "predicted_service_time_s",
        "profile_id",
        "hard_state",
        "evidence_complete",
        "rho_static",
        "rho_dynamic",
        "rho_speed",
    ]
    frame = pd.read_parquet(path, columns=required)
    frame["request_time"] = pd.to_datetime(
        frame["request_time"], utc=True
    ).dt.tz_convert(TIMEZONE)
    frame = frame.loc[
        frame["profile_id"].astype(str).eq(str(profile_id))
        & frame["request_time"].ge(start)
        & frame["request_time"].lt(end)
    ].copy()
    coordinate_columns = [
        "pickup_lon_wgs84",
        "pickup_lat_wgs84",
        "dropoff_lon_wgs84",
        "dropoff_lat_wgs84",
    ]
    numeric = frame[coordinate_columns + ["realized_service_time_s"]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = np.isfinite(numeric).all(axis=1) & numeric["realized_service_time_s"].gt(0)
    if not bool(valid.all()):
        raise FleetPyCompatibilityError(
            "bounded replay contains invalid coordinates or realized progression time"
        )
    if frame["order_id"].astype(str).duplicated().any():
        raise FleetPyCompatibilityError(
            "profile-specific Test31 order_id is not unique"
        )
    frame = frame.sort_values(["request_time", "order_id"], kind="mergesort")
    records: list[SpikeRequest] = []
    for native_id, row in enumerate(frame.itertuples(index=False)):
        seconds = (pd.Timestamp(row.request_time) - start).total_seconds()
        predicted = pd.to_numeric(
            pd.Series([row.predicted_service_time_s]), errors="coerce"
        ).iloc[0]
        records.append(
            SpikeRequest(
                native_id=native_id,
                order_id=str(row.order_id),
                request_time=pd.Timestamp(row.request_time),
                sim_time_s=int(round(seconds)),
                pickup_lon_wgs84=float(row.pickup_lon_wgs84),
                pickup_lat_wgs84=float(row.pickup_lat_wgs84),
                dropoff_lon_wgs84=float(row.dropoff_lon_wgs84),
                dropoff_lat_wgs84=float(row.dropoff_lat_wgs84),
                realized_service_time_s=float(row.realized_service_time_s),
                predicted_service_time_s=float(predicted),
                profile_id=str(row.profile_id),
                hard_state=str(row.hard_state),
                evidence_complete=bool(row.evidence_complete),
                rho_static=float(row.rho_static),
                rho_dynamic=float(row.rho_dynamic),
                rho_speed=float(row.rho_speed),
            )
        )
    return records
