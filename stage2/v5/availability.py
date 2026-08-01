"""Selective-observation masks and stabilized inverse-propensity weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import Stage2V5ContractError


@dataclass(frozen=True)
class IPWDiagnostics:
    observed_count: int
    observed_rate: float
    effective_sample_size: float
    weight_min: float
    weight_p50: float
    weight_p90: float
    weight_p99: float
    weight_max: float
    clipped_count: int


def stabilized_ipw(
    observed: np.ndarray,
    availability_probability: np.ndarray,
    *,
    epsilon: float,
    maximum_weight: float,
) -> tuple[np.ndarray, IPWDiagnostics]:
    mask = np.asarray(observed, dtype=bool)
    probability = np.asarray(availability_probability, dtype=np.float64)
    if mask.shape != probability.shape:
        raise Stage2V5ContractError("availability mask and probability shapes differ")
    if not (0.0 < epsilon < 1.0) or maximum_weight < 1.0:
        raise Stage2V5ContractError("invalid IPW controls")
    if np.any(~np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
        raise Stage2V5ContractError("availability probabilities must be finite in [0,1]")
    prevalence = float(mask.mean()) if len(mask) else 0.0
    denominator = np.maximum(probability, float(epsilon))
    unbounded = np.where(mask, prevalence / denominator, 0.0)
    weights = np.minimum(unbounded, float(maximum_weight))
    positive = weights[mask]
    clipped = int(np.count_nonzero(mask & (unbounded > maximum_weight)))
    if positive.size:
        square_sum = float(np.square(positive).sum())
        ess = float(positive.sum() ** 2 / square_sum) if square_sum > 0 else 0.0
        quantiles = np.quantile(positive, [0.5, 0.9, 0.99])
        minimum = float(positive.min())
        maximum = float(positive.max())
    else:
        ess = minimum = maximum = 0.0
        quantiles = np.zeros(3)
    diagnostics = IPWDiagnostics(
        observed_count=int(mask.sum()),
        observed_rate=prevalence,
        effective_sample_size=ess,
        weight_min=minimum,
        weight_p50=float(quantiles[0]),
        weight_p90=float(quantiles[1]),
        weight_p99=float(quantiles[2]),
        weight_max=maximum,
        clipped_count=clipped,
    )
    return weights, diagnostics


def service_time_target_arrays(
    measurement_source: np.ndarray,
    observed_time_s: np.ndarray,
    observed_distance_m: np.ndarray,
) -> dict[str, np.ndarray]:
    source = np.asarray(measurement_source).astype(str)
    time_s = np.asarray(observed_time_s, dtype=np.float64)
    distance_m = np.asarray(observed_distance_m, dtype=np.float64)
    direct = source == "direct_observed"
    time_valid = direct & np.isfinite(time_s) & (time_s > 0)
    pace_valid = time_valid & np.isfinite(distance_m) & (distance_m > 0)
    pace = np.full(time_s.shape, np.nan, dtype=np.float64)
    np.divide(time_s, distance_m, out=pace, where=pace_valid)
    source_class = np.where(
        direct,
        "direct_raw_gps_interval",
        np.char.add(source.astype(str), "_no_link_time"),
    )
    return {
        "travel_time_target_valid": time_valid,
        "travel_time_direct_valid": time_valid,
        "travel_time_interpolated_valid": np.zeros(time_s.shape, dtype=bool),
        "pace_target_valid": pace_valid,
        "pace_target_sec_per_m": pace,
        "travel_time_source_class": source_class,
    }
