"""Stage 1 label-schema v2 aggregation and missing-modality semantics."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


DIMENSIONS = ("lcs", "iis", "gns", "rts", "pmis")
# PMIS is an interaction descriptor derived partly from LCS/RTS.  It remains an
# output dimension, but is excluded from the equal-weight core composite to avoid
# counting the same behavior twice.
CORE_DIMENSIONS = ("lcs", "gns", "rts")


def weighted_summary(values: np.ndarray, weights: np.ndarray, threshold: float) -> dict[str, float]:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    if not valid.any():
        return {name: float("nan") for name in ("mean", "max", "tail", "persistence")}
    clean_values = values[valid]
    clean_weights = weights[valid]
    if clean_weights.sum() <= 0:
        clean_weights = np.ones(len(clean_values), dtype=np.float64)
    high = clean_values >= threshold
    return {
        "mean": float(np.average(clean_values, weights=clean_weights)),
        "max": float(clean_values.max()),
        "tail": float(clean_values[high].mean()) if high.any() else float(clean_values.max()),
        "persistence": float(clean_weights[high].sum() / clean_weights.sum()),
    }


def aggregate_order_labels_v2(labels: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """Aggregate links without imputing unavailable dimensions to zero."""

    required = {"order_id"} | {f"{dimension}_pct_link" for dimension in DIMENSIONS}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"missing Stage1 v2 columns: {missing}")
    rows: list[dict[str, object]] = []
    for order_id, group in labels.groupby("order_id", sort=False):
        row: dict[str, object] = {"order_id": order_id}
        mask: dict[str, bool] = {}
        for dimension in DIMENSIONS:
            values = group[f"{dimension}_pct_link"].to_numpy(dtype=np.float64)
            if dimension == "gns":
                weight_name = next(
                    (name for name in ("link_length_m", "observed_distance_m") if name in group.columns),
                    None,
                )
            else:
                weight_name = "travel_time_sec" if "travel_time_sec" in group.columns else None
            weights = (
                group[weight_name].to_numpy(dtype=np.float64)
                if weight_name
                else np.ones(len(group), dtype=np.float64)
            )
            summary = weighted_summary(values, weights, threshold)
            for statistic, value in summary.items():
                row[f"{dimension}_{statistic}"] = value
            available = bool(np.isfinite(values).any())
            row[f"{dimension}_available"] = available
            mask[dimension] = available

        core_available = [dimension for dimension in CORE_DIMENSIONS if mask[dimension]]
        row["dimension_mask"] = json.dumps(mask, sort_keys=True, separators=(",", ":"))
        row["valid_dimension_count"] = len(core_available)
        row["composition_signature"] = "+".join(core_available) if core_available else "NONE"
        for statistic in ("mean", "tail"):
            components = [row[f"{dimension}_{statistic}"] for dimension in core_available]
            row[f"core_composite_{statistic}"] = float(np.mean(components)) if components else float("nan")
        row["pmis_role"] = "interaction_output_excluded_from_core_composite"
        rows.append(row)
    return pd.DataFrame(rows)
