"""Conservative allocation of a GPS interval across traversed links."""

from __future__ import annotations

import numpy as np


def allocate_by_projected_mileage(
    duration_sec: float,
    distance_m: float,
    projected_mileages_m: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate time and distance proportionally while preserving exact totals.

    The final element receives the floating-point remainder, making the
    conservation identities deterministic even for many traversed links.
    """

    duration = max(0.0, float(duration_sec))
    distance = max(0.0, float(distance_m))
    mileage = np.asarray(projected_mileages_m, dtype=float)
    if mileage.ndim != 1 or len(mileage) == 0:
        raise ValueError("projected_mileages_m must be a non-empty vector")
    mileage = np.where(np.isfinite(mileage) & (mileage > 0), mileage, 0.0)
    if float(mileage.sum()) <= 0:
        mileage = np.ones(len(mileage), dtype=float)
    shares = mileage / mileage.sum()
    allocated_time = shares * duration
    allocated_distance = shares * distance
    allocated_time[-1] = duration - float(allocated_time[:-1].sum())
    allocated_distance[-1] = distance - float(allocated_distance[:-1].sum())
    return allocated_time, allocated_distance


def directed_endpoint_mileages(
    previous_length_m: float,
    previous_fraction: float,
    previous_direction: str,
    next_length_m: float,
    next_fraction: float,
    next_direction: str,
) -> tuple[float, float]:
    """Mileage from previous projection to exit and entry to next projection."""

    prev_fraction = float(np.clip(previous_fraction, 0.0, 1.0))
    next_fraction = float(np.clip(next_fraction, 0.0, 1.0))
    previous = (
        previous_length_m * (1.0 - prev_fraction)
        if previous_direction == "F"
        else previous_length_m * prev_fraction
    )
    following = (
        next_length_m * next_fraction
        if next_direction == "F"
        else next_length_m * (1.0 - next_fraction)
    )
    return max(0.0, float(previous)), max(0.0, float(following))

