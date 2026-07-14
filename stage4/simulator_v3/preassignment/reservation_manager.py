"""Reservation bookkeeping for preassignment.

The simulator keeps assignment execution in :mod:`vehicle_executor`; this
manager only owns the request↔vehicle reservation invariant.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ReservationRecord:
    request_id: str
    vehicle_id: str
    create_time: pd.Timestamp
    expiry_time: pd.Timestamp
    status: str = "ACTIVE"
    failure_reason: str = ""


class ReservationManager:
    def __init__(self) -> None:
        self.request_to_vehicle: dict[str, str] = {}
        self.vehicle_to_request: dict[str, str] = {}
        self.records: list[ReservationRecord] = []
        self.failure_records: list[dict] = []

    def create(self, request_id: str, vehicle_id: str, now: pd.Timestamp, expiry_time: pd.Timestamp) -> ReservationRecord:
        if request_id in self.request_to_vehicle:
            raise ValueError(f"Request {request_id} already has an active reservation")
        if vehicle_id in self.vehicle_to_request:
            raise ValueError(f"Vehicle {vehicle_id} already has an active reservation")
        record = ReservationRecord(request_id=request_id, vehicle_id=vehicle_id, create_time=now, expiry_time=expiry_time)
        self.request_to_vehicle[request_id] = vehicle_id
        self.vehicle_to_request[vehicle_id] = request_id
        self.records.append(record)
        return record

    def release(self, request_id: str, now: pd.Timestamp, reason: str) -> None:
        vehicle_id = self.request_to_vehicle.pop(request_id, None)
        if vehicle_id is not None:
            self.vehicle_to_request.pop(vehicle_id, None)
        for record in reversed(self.records):
            if record.request_id == request_id and record.status == "ACTIVE":
                record.status = "RELEASED"
                record.failure_reason = reason
                break
        self.failure_records.append({
            "request_id": request_id,
            "vehicle_id": vehicle_id or "",
            "failure_time": str(now),
            "failure_reason": reason,
        })

    def complete(self, request_id: str) -> None:
        vehicle_id = self.request_to_vehicle.pop(request_id, None)
        if vehicle_id is not None:
            self.vehicle_to_request.pop(vehicle_id, None)
        for record in reversed(self.records):
            if record.request_id == request_id and record.status == "ACTIVE":
                record.status = "COMPLETED"
                break

    def audit(self) -> dict:
        duplicate_vehicle = len(self.vehicle_to_request) != len(set(self.vehicle_to_request))
        duplicate_request = len(self.request_to_vehicle) != len(set(self.request_to_vehicle))
        return {
            "active_reservations": len(self.request_to_vehicle),
            "reservation_records": len(self.records),
            "reservation_failures": len(self.failure_records),
            "duplicate_vehicle_reservation": int(duplicate_vehicle),
            "duplicate_request_reservation": int(duplicate_request),
            "reservation_invariant_pass": "PASS" if not duplicate_vehicle and not duplicate_request else "FAIL",
        }

