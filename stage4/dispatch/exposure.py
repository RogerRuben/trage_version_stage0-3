"""Order-level Stage3 reference-envelope excess and cumulative AV state."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping


FAMILIES = ("static", "dynamic", "speed")


@dataclass(frozen=True)
class ExposureExcess:
    static: float
    dynamic: float
    speed: float

    def value(self, family: str) -> float:
        return float(getattr(self, family))


def exposure_excess(
    rho_static: float, rho_dynamic: float, rho_speed: float
) -> ExposureExcess | None:
    values = (float(rho_static), float(rho_dynamic), float(rho_speed))
    if not all(isfinite(value) for value in values):
        return None
    return ExposureExcess(*(max(value - 1.0, 0.0) for value in values))


def parse_gammas(config: Mapping[str, object]) -> dict[str, float | None]:
    parsed: dict[str, float | None] = {}
    for family in FAMILIES:
        raw = config.get(f"gamma_{family}")
        if raw is None:
            parsed[family] = None
            continue
        value = float(raw)
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"gamma_{family} must be null or finite and >= 0")
        parsed[family] = value
    return parsed


@dataclass
class CumulativeExposureState:
    av_assignments: int = 0
    static: float = 0.0
    dynamic: float = 0.0
    speed: float = 0.0

    def update(self, exposures: list[ExposureExcess]) -> ExposureExcess:
        epoch = ExposureExcess(
            sum(item.static for item in exposures),
            sum(item.dynamic for item in exposures),
            sum(item.speed for item in exposures),
        )
        self.av_assignments += len(exposures)
        self.static += epoch.static
        self.dynamic += epoch.dynamic
        self.speed += epoch.speed
        return epoch

    def mean(self, family: str) -> float:
        return (
            float(getattr(self, family)) / self.av_assignments
            if self.av_assignments
            else 0.0
        )

    def validate(self, gammas: Mapping[str, float | None], tolerance: float) -> None:
        for family in FAMILIES:
            gamma = gammas.get(family)
            if gamma is not None and self.mean(family) > float(gamma) + tolerance:
                raise RuntimeError(f"cumulative {family} exposure budget violated")
