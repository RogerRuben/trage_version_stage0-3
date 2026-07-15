"""Intersection influence-area measurement helpers."""

from __future__ import annotations

import numpy as np


def upstream_influence_share(link_length_m: float, influence_distance_m: float = 75.0) -> float:
    if not np.isfinite(link_length_m) or link_length_m <= 0:
        return 0.0
    return float(np.clip(influence_distance_m / link_length_m, 0.0, 1.0))


def prorate_to_upstream_influence(
    observed_time_sec: float,
    link_length_m: float,
    influence_distance_m: float = 75.0,
) -> float:
    return max(0.0, float(observed_time_sec)) * upstream_influence_share(
        link_length_m, influence_distance_m
    )

