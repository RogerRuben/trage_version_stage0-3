"""Frozen v5.2 tuning, development, rolling, and legacy date protocols."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .contracts import Stage2V52ContractError


def _dates(first: int, last: int) -> tuple[str, ...]:
    return tuple(f"201610{day:02d}" for day in range(first, last + 1))


@dataclass(frozen=True)
class ProtocolSpec:
    protocol_id: str
    train_dates: tuple[str, ...]
    validation_dates: tuple[str, ...]
    calibration_dates: tuple[str, ...] = ()
    evaluation_dates: tuple[str, ...] = ()
    legacy_benchmark_dates: tuple[str, ...] = ()
    rts_role: str = "secondary_frozen_reference_target"

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        raw = json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()


PROTOCOLS = {
    "transfer_tuning": ProtocolSpec(
        "transfer_tuning", _dates(9, 18), _dates(19, 20)
    ),
    "development": ProtocolSpec(
        "development", _dates(9, 21), _dates(22, 23), _dates(24, 24), _dates(25, 27)
    ),
    "fold_1": ProtocolSpec(
        "fold_1", _dates(9, 18), _dates(19, 20), _dates(21, 21), _dates(22, 23)
    ),
    "fold_2": ProtocolSpec(
        "fold_2", _dates(9, 20), _dates(21, 22), _dates(23, 23), _dates(24, 25)
    ),
    "fold_3": ProtocolSpec(
        "fold_3", _dates(9, 22), _dates(23, 24), _dates(25, 25), _dates(26, 27)
    ),
    "legacy_31": ProtocolSpec(
        "legacy_31", _dates(9, 24), _dates(25, 26), _dates(27, 27), (), ("20161031",)
    ),
}


def get_protocol(protocol_id: str) -> ProtocolSpec:
    try:
        return PROTOCOLS[protocol_id]
    except KeyError as exc:
        raise Stage2V52ContractError(f"unknown frozen protocol: {protocol_id}") from exc


def validate_protocols() -> None:
    tuning = PROTOCOLS["transfer_tuning"]
    if tuning.train_dates != _dates(9, 18) or tuning.validation_dates != _dates(19, 20):
        raise Stage2V52ContractError("transfer tuning protocol drifted")
    if any(date >= "20161021" for date in (*tuning.train_dates, *tuning.validation_dates)):
        raise Stage2V52ContractError("20161021+ cannot enter tau tuning")
    for spec in PROTOCOLS.values():
        groups = (
            spec.train_dates, spec.validation_dates, spec.calibration_dates,
            spec.evaluation_dates, spec.legacy_benchmark_dates,
        )
        flattened = tuple(date for group in groups for date in group)
        if len(flattened) != len(set(flattened)) or flattened != tuple(sorted(flattened)):
            raise Stage2V52ContractError(f"protocol {spec.protocol_id} is not strictly temporal")
        if any(date in {"20161028", "20161029", "20161030"} for date in flattened):
            raise Stage2V52ContractError("forbidden unproduced date entered v5.2 protocol")


validate_protocols()
