"""Vectorized route aggregation with explicit state and tail semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import Stage2V5ContractError


KEYS = ("split", "date", "order_id")


@dataclass(frozen=True)
class RouteDimension:
    name: str
    percentile_column: str
    tail_probability_column: str
    value_column: str
    weight_column: str


def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def aggregate_route_dimensions(
    frame: pd.DataFrame,
    dimensions: tuple[RouteDimension, ...],
) -> pd.DataFrame:
    required = {*KEYS, "route_sequence"}
    for spec in dimensions:
        required.update((spec.percentile_column, spec.tail_probability_column, spec.value_column, spec.weight_column))
    missing = sorted(required - set(frame.columns))
    if missing:
        raise Stage2V5ContractError(f"route aggregation is missing: {missing}")
    working = frame.loc[:, list(dict.fromkeys([*KEYS, "route_sequence", *[column for spec in dimensions for column in (spec.percentile_column, spec.tail_probability_column, spec.value_column, spec.weight_column)]]))].copy()
    working = working.sort_values([*KEYS, "route_sequence"], kind="stable").reset_index(drop=True)
    grouped = working.groupby(list(KEYS), sort=False, observed=True, dropna=False)
    codes = grouped.ngroup().to_numpy(dtype=np.int64)
    group_count = int(codes.max() + 1) if len(codes) else 0
    identity = working.loc[:, KEYS].groupby(codes, sort=False, observed=True).first().reset_index(drop=True)
    result = identity.copy()
    previous_code = np.concatenate((np.array([-1], dtype=np.int64), codes[:-1]))
    group_change = codes != previous_code
    for spec in dimensions:  # Fixed-size dimension loop; never a per-route loop.
        percentile = pd.to_numeric(working[spec.percentile_column], errors="coerce").to_numpy(dtype=np.float64)
        tail_probability = pd.to_numeric(working[spec.tail_probability_column], errors="coerce").to_numpy(dtype=np.float64)
        value = pd.to_numeric(working[spec.value_column], errors="coerce").to_numpy(dtype=np.float64)
        weight = pd.to_numeric(working[spec.weight_column], errors="coerce").to_numpy(dtype=np.float64)
        physical_weight = np.isfinite(weight) & (weight > 0)
        valid = physical_weight & np.isfinite(value)
        tail_probability_valid = physical_weight & np.isfinite(tail_probability)
        tail = valid & np.isfinite(percentile) & (percentile >= 0.9)
        total_weight = np.bincount(codes, weights=np.where(physical_weight, weight, 0.0), minlength=group_count)
        valid_weight = np.bincount(codes, weights=np.where(valid, weight, 0.0), minlength=group_count)
        state_numerator = np.bincount(codes, weights=np.where(valid, value * weight, 0.0), minlength=group_count)
        tail_probability_denominator = np.bincount(codes, weights=np.where(tail_probability_valid, weight, 0.0), minlength=group_count)
        tail_probability_numerator = np.bincount(codes, weights=np.where(tail_probability_valid, tail_probability * weight, 0.0), minlength=group_count)
        tail_weight = np.bincount(codes, weights=np.where(tail, weight, 0.0), minlength=group_count)
        severity_numerator = np.bincount(codes, weights=np.where(tail, percentile * weight, 0.0), minlength=group_count)
        run_start = tail & (group_change | ~np.concatenate((np.array([False]), tail[:-1])))
        run_id = np.cumsum(run_start, dtype=np.int64) - 1
        maximum_run_weight = np.zeros(group_count, dtype=np.float64)
        if tail.any():
            tail_runs = pd.DataFrame({"group_code": codes[tail], "run_id": run_id[tail], "weight": weight[tail]})
            run_weight = tail_runs.groupby(["group_code", "run_id"], sort=False, observed=True)["weight"].sum()
            per_group_max = run_weight.groupby(level=0, sort=False).max()
            maximum_run_weight[per_group_max.index.to_numpy(dtype=np.int64)] = per_group_max.to_numpy(dtype=np.float64)
        prefix = spec.name
        result[f"{prefix}_mean_state"] = _divide(state_numerator, valid_weight)
        result[f"{prefix}_mean_tail_probability"] = _divide(tail_probability_numerator, tail_probability_denominator)
        result[f"{prefix}_conditional_tail_severity"] = _divide(severity_numerator, tail_weight)
        result[f"{prefix}_weighted_tail_persistence"] = _divide(tail_weight, total_weight)
        result[f"{prefix}_maximum_consecutive_tail_share"] = _divide(maximum_run_weight, total_weight)
        result[f"{prefix}_weighted_coverage_share"] = _divide(valid_weight, total_weight)
        result[f"{prefix}_tail_event_present"] = tail_weight > 0
    return result


def aggregate_route_dimension(
    frame: pd.DataFrame,
    *,
    dimension: str,
    percentile_column: str,
    tail_probability_column: str,
    value_column: str | None = None,
    weight_column: str | None = None,
) -> pd.DataFrame:
    if dimension not in {"lcs", "rts"}:
        raise Stage2V5ContractError("dimension must be lcs or rts")
    return aggregate_route_dimensions(
        frame,
        (
            RouteDimension(
                name=dimension,
                percentile_column=percentile_column,
                tail_probability_column=tail_probability_column,
                value_column=value_column or percentile_column,
                weight_column=weight_column or ("estimated_travel_time_s" if dimension == "lcs" else "route_part_length_m"),
            ),
        ),
    )

