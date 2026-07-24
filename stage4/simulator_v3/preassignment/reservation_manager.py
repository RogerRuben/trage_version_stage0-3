"""Reservation lifecycle and invariants for preassignment.

The simulator keeps physical execution in :mod:`vehicle_executor`; this
manager owns the request-to-vehicle reservation invariant and exposes
explicit expiration/invalidation operations for the event engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


def _event_timestamp(value: Any) -> pd.Timestamp:
    """Normalize simulator timestamps to the persisted microsecond contract.

    Pandas arithmetic with floating-point seconds can create nanosecond values
    that Arrow cannot losslessly cast to Parquet's timestamp[us].  The event
    engine itself does not resolve events below one microsecond, so applying
    the same floor at the reservation boundary is deterministic and avoids an
    implicit, writer-specific truncation.
    """

    return pd.Timestamp(value).floor("us")


@dataclass
class ReservationRecord:
    request_id: str
    vehicle_id: str
    create_time: pd.Timestamp
    expiry_time: pd.Timestamp
    expected_release_time: pd.Timestamp | None = None
    safe_release_time: pd.Timestamp | None = None
    buffer_source: str = ""
    buffer_sample_count: int = 0
    buffer_quantile: float | None = None
    residual_quantile_sec: float | None = None
    status: str = "ACTIVE"
    failure_reason: str = ""
    close_time: pd.Timestamp | None = None


@dataclass(frozen=True)
class ReservationValidation:
    feasible: bool
    reason: str
    request_id: str
    vehicle_id: str
    safe_release_time: pd.Timestamp
    projected_pickup_time: pd.Timestamp


class ReservationManager:
    def __init__(self) -> None:
        self.request_to_vehicle: dict[str, str] = {}
        self.vehicle_to_request: dict[str, str] = {}
        self.records: list[ReservationRecord] = []
        self.failure_records: list[dict] = []

    def create(
        self,
        request_id: str,
        vehicle_id: str,
        now: pd.Timestamp,
        expiry_time: pd.Timestamp,
        *,
        expected_release_time: pd.Timestamp | None = None,
        safe_release_time: pd.Timestamp | None = None,
        buffer_source: str = "",
        buffer_sample_count: int = 0,
        buffer_quantile: float | None = None,
        residual_quantile_sec: float | None = None,
    ) -> ReservationRecord:
        if request_id in self.request_to_vehicle:
            raise ValueError(f"Request {request_id} already has an active reservation")
        if vehicle_id in self.vehicle_to_request:
            raise ValueError(f"Vehicle {vehicle_id} already has an active reservation")
        now = _event_timestamp(now)
        expiry_time = _event_timestamp(expiry_time)
        if expiry_time <= now:
            raise ValueError("Reservation expiry_time must be after create_time")
        record = ReservationRecord(
            request_id=request_id,
            vehicle_id=vehicle_id,
            create_time=now,
            expiry_time=expiry_time,
            expected_release_time=_event_timestamp(expected_release_time) if expected_release_time is not None else None,
            safe_release_time=_event_timestamp(safe_release_time) if safe_release_time is not None else None,
            buffer_source=str(buffer_source),
            buffer_sample_count=int(buffer_sample_count),
            buffer_quantile=float(buffer_quantile) if buffer_quantile is not None else None,
            residual_quantile_sec=float(residual_quantile_sec) if residual_quantile_sec is not None else None,
        )
        self.request_to_vehicle[request_id] = vehicle_id
        self.vehicle_to_request[vehicle_id] = request_id
        self.records.append(record)
        return record

    def active_for_request(self, request_id: str) -> ReservationRecord | None:
        if request_id not in self.request_to_vehicle:
            return None
        return next(
            (record for record in reversed(self.records) if record.request_id == request_id and record.status == "ACTIVE"),
            None,
        )

    def active_for_vehicle(self, vehicle_id: str) -> ReservationRecord | None:
        request_id = self.vehicle_to_request.get(vehicle_id)
        return self.active_for_request(request_id) if request_id is not None else None

    def release(
        self,
        request_id: str,
        now: pd.Timestamp,
        reason: str,
        *,
        terminal_status: str = "RELEASED",
    ) -> ReservationRecord | None:
        if terminal_status not in {"RELEASED", "EXPIRED", "INVALIDATED"}:
            raise ValueError(f"Unsupported reservation terminal status: {terminal_status}")
        now = _event_timestamp(now)
        vehicle_id = self.request_to_vehicle.pop(request_id, None)
        if vehicle_id is not None:
            self.vehicle_to_request.pop(vehicle_id, None)
        released: ReservationRecord | None = None
        for record in reversed(self.records):
            if record.request_id == request_id and record.status == "ACTIVE":
                record.status = terminal_status
                record.failure_reason = reason
                record.close_time = now
                released = record
                break
        if released is not None:
            self.failure_records.append({
                "request_id": request_id,
                "vehicle_id": vehicle_id or released.vehicle_id,
                "failure_time": str(now),
                "failure_event": terminal_status,
                "failure_reason": reason,
            })
        return released

    def expire(
        self,
        request_id: str,
        now: pd.Timestamp,
        reason: str = "RESERVATION_EXPIRED",
    ) -> ReservationRecord | None:
        return self.release(request_id, now, reason, terminal_status="EXPIRED")

    def invalidate(
        self,
        request_id: str,
        now: pd.Timestamp,
        reason: str = "PLAN_INVALIDATED",
    ) -> ReservationRecord | None:
        return self.release(request_id, now, reason, terminal_status="INVALIDATED")

    def expire_due(self, now: pd.Timestamp) -> list[ReservationRecord]:
        now = _event_timestamp(now)
        expired: list[ReservationRecord] = []
        for request_id in list(self.request_to_vehicle):
            record = self.active_for_request(request_id)
            if record is not None and record.expiry_time <= now:
                closed = self.expire(request_id, now)
                if closed is not None:
                    expired.append(closed)
        return expired

    def revalidate(
        self,
        request_id: str,
        now: pd.Timestamp,
        safe_release_time: pd.Timestamp,
        pickup_eta_sec: float,
        latest_pickup_time: pd.Timestamp,
        *,
        vehicle_online_end: pd.Timestamp | None = None,
        projected_service_time_sec: float = 0.0,
        release_metadata: dict[str, Any] | None = None,
    ) -> ReservationValidation:
        """Revalidate after release-time, pickup-time, or deadline changes.

        On failure the reservation is atomically removed from both active
        maps and marked ``EXPIRED`` or ``INVALIDATED``. Request/vehicle state
        transitions remain the responsibility of the simulation engine.
        """

        record = self.active_for_request(request_id)
        if record is None:
            raise KeyError(f"No active reservation for request {request_id}")
        now = _event_timestamp(now)
        safe_release_time = _event_timestamp(safe_release_time)
        latest_pickup_time = _event_timestamp(latest_pickup_time)
        projected_pickup = safe_release_time + pd.Timedelta(seconds=float(pickup_eta_sec))
        feasible = True
        reason = "PASS"
        if now >= record.expiry_time:
            feasible = False
            reason = "RESERVATION_EXPIRED"
            self.expire(request_id, now, reason)
        elif projected_pickup > latest_pickup_time:
            feasible = False
            reason = "PLAN_INVALIDATED:PICKUP_DEADLINE"
            self.invalidate(request_id, now, reason)
        elif vehicle_online_end is not None and (
            projected_pickup + pd.Timedelta(seconds=float(projected_service_time_sec))
            > _event_timestamp(vehicle_online_end)
        ):
            feasible = False
            reason = "PLAN_INVALIDATED:SESSION_END"
            self.invalidate(request_id, now, reason)
        else:
            record.safe_release_time = safe_release_time
            if release_metadata:
                expected = release_metadata.get("expected_release_time", record.expected_release_time)
                record.expected_release_time = _event_timestamp(expected) if expected is not None else None
                record.buffer_source = str(release_metadata.get("buffer_source", record.buffer_source))
                record.buffer_sample_count = int(
                    release_metadata.get("buffer_sample_count", record.buffer_sample_count)
                )
                quantile = release_metadata.get("buffer_quantile", record.buffer_quantile)
                record.buffer_quantile = float(quantile) if quantile is not None else None
                residual = release_metadata.get("release_residual_quantile_sec", record.residual_quantile_sec)
                record.residual_quantile_sec = float(residual) if residual is not None else None
        return ReservationValidation(
            feasible=feasible,
            reason=reason,
            request_id=request_id,
            vehicle_id=record.vehicle_id,
            safe_release_time=safe_release_time,
            projected_pickup_time=projected_pickup,
        )

    def complete(self, request_id: str, now: pd.Timestamp | None = None) -> ReservationRecord | None:
        vehicle_id = self.request_to_vehicle.pop(request_id, None)
        if vehicle_id is not None:
            self.vehicle_to_request.pop(vehicle_id, None)
        completed: ReservationRecord | None = None
        for record in reversed(self.records):
            if record.request_id == request_id and record.status == "ACTIVE":
                record.status = "COMPLETED"
                record.close_time = _event_timestamp(now) if now is not None else None
                completed = record
                break
        return completed

    def audit(self) -> dict:
        duplicate_vehicle = len(self.vehicle_to_request) != len(set(self.vehicle_to_request))
        duplicate_request = len(self.request_to_vehicle) != len(set(self.request_to_vehicle))
        inverse_pass = all(
            self.vehicle_to_request.get(vehicle_id) == request_id
            for request_id, vehicle_id in self.request_to_vehicle.items()
        ) and all(
            self.request_to_vehicle.get(request_id) == vehicle_id
            for vehicle_id, request_id in self.vehicle_to_request.items()
        )
        active_records = [record for record in self.records if record.status == "ACTIVE"]
        records_match_maps = {
            (record.request_id, record.vehicle_id) for record in active_records
        } == set(self.request_to_vehicle.items())
        pass_flag = not duplicate_vehicle and not duplicate_request and inverse_pass and records_match_maps
        return {
            "active_reservations": len(self.request_to_vehicle),
            "reservation_records": len(self.records),
            "reservation_failures": len(self.failure_records),
            "duplicate_vehicle_reservation": int(duplicate_vehicle),
            "duplicate_request_reservation": int(duplicate_request),
            "inverse_map_pass": int(inverse_pass),
            "active_records_match_maps": int(records_match_maps),
            "expired_reservations": sum(record.status == "EXPIRED" for record in self.records),
            "invalidated_reservations": sum(record.status == "INVALIDATED" for record in self.records),
            "reservation_invariant_pass": "PASS" if pass_flag else "FAIL",
        }
