"""Small-data reference implementations; never used by production commands."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .aggregation import KEYS, RouteDimension


def aggregate_route_dimensions_reference(frame: pd.DataFrame, dimensions: tuple[RouteDimension, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered = frame.sort_values([*KEYS, "route_sequence"], kind="stable")
    for identity, group in ordered.groupby(list(KEYS), sort=False, observed=True, dropna=False):
        row: dict[str, object] = dict(zip(KEYS, identity))
        for spec in dimensions:
            value = pd.to_numeric(group[spec.value_column], errors="coerce").to_numpy(dtype=float)
            percentile = pd.to_numeric(group[spec.percentile_column], errors="coerce").to_numpy(dtype=float)
            probability = pd.to_numeric(group[spec.tail_probability_column], errors="coerce").to_numpy(dtype=float)
            weight = pd.to_numeric(group[spec.weight_column], errors="coerce").to_numpy(dtype=float)
            physical = np.isfinite(weight) & (weight > 0)
            valid = physical & np.isfinite(value)
            probability_valid = physical & np.isfinite(probability)
            tail = valid & np.isfinite(percentile) & (percentile >= 0.9)
            total_weight = weight[physical].sum()
            prefix = spec.name
            row[f"{prefix}_mean_state"] = np.average(value[valid], weights=weight[valid]) if valid.any() else np.nan
            row[f"{prefix}_mean_tail_probability"] = np.average(probability[probability_valid], weights=weight[probability_valid]) if probability_valid.any() else np.nan
            row[f"{prefix}_conditional_tail_severity"] = np.average(percentile[tail], weights=weight[tail]) if tail.any() else np.nan
            row[f"{prefix}_weighted_tail_persistence"] = weight[tail].sum() / total_weight if total_weight > 0 else np.nan
            maximum = 0.0
            current = 0.0
            for flag, local_weight in zip(tail, weight):
                current = current + local_weight if flag else 0.0
                maximum = max(maximum, current)
            row[f"{prefix}_maximum_consecutive_tail_share"] = maximum / total_weight if total_weight > 0 else np.nan
            row[f"{prefix}_weighted_coverage_share"] = weight[valid].sum() / total_weight if total_weight > 0 else np.nan
            row[f"{prefix}_tail_event_present"] = bool(tail.any())
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_traversal_scenarios_reference(
    route_inverse: np.ndarray,
    traversal_scenarios: np.ndarray,
    *,
    route_count: int | None = None,
) -> np.ndarray:
    inverse = np.asarray(route_inverse, dtype=np.int64)
    traversal = np.asarray(traversal_scenarios, dtype=np.float64)
    count = int(inverse.max() + 1) if route_count is None and len(inverse) else int(route_count or 0)
    output = np.zeros((count, traversal.shape[1]), dtype=np.float64)
    for row, route in enumerate(inverse):
        output[route] += traversal[row]
    return output
