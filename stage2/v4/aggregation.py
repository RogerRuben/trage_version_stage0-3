"""Route/order summaries using decision-time weights only."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .contracts import Stage2V4ContractError


def aggregate_route_dimension(
    frame: pd.DataFrame,
    *,
    dimension: str,
    percentile_column: str,
    tail_probability_column: str,
) -> pd.DataFrame:
    weight_column = (
        "estimated_travel_time_s" if dimension == "lcs" else "route_part_length_m"
    )
    required = {
        "split",
        "date",
        "order_id",
        "route_sequence",
        "route_position_ratio",
        percentile_column,
        tail_probability_column,
        weight_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise Stage2V4ContractError(f"route aggregation is missing: {missing}")
    keys = ["split", "date", "order_id"]
    working = frame.loc[
        :,
        [*keys, "route_sequence", "route_position_ratio", percentile_column,
         tail_probability_column, weight_column],
    ].copy()
    working = working.sort_values(
        [*keys, "route_sequence"],
        kind="stable",
    ).reset_index(drop=True)
    working["_value"] = pd.to_numeric(
        working[percentile_column], errors="coerce"
    )
    working["_tail_probability"] = pd.to_numeric(
        working[tail_probability_column], errors="coerce"
    )
    working["_weight"] = pd.to_numeric(working[weight_column], errors="coerce")
    working["_position"] = pd.to_numeric(
        working["route_position_ratio"], errors="coerce"
    )
    valid = (
        working["_value"].notna()
        & working["_weight"].notna()
        & working["_weight"].gt(0)
    )
    tail_valid = (
        working["_tail_probability"].notna()
        & working["_weight"].notna()
        & working["_weight"].gt(0)
    )
    pickup = valid & working["_position"].le(0.2)
    dropoff = valid & working["_position"].ge(0.8)
    tail_event = valid & working["_value"].ge(0.9)
    working["_valid"] = valid.astype(np.int8)
    working["_weighted_numerator"] = (
        working["_value"] * working["_weight"]
    ).where(valid, 0.0)
    working["_weighted_denominator"] = working["_weight"].where(valid, 0.0)
    working["_tail_numerator"] = (
        working["_tail_probability"] * working["_weight"]
    ).where(tail_valid, 0.0)
    working["_tail_denominator"] = working["_weight"].where(tail_valid, 0.0)
    working["_pickup_numerator"] = (
        working["_value"] * working["_weight"]
    ).where(pickup, 0.0)
    working["_pickup_denominator"] = working["_weight"].where(pickup, 0.0)
    working["_dropoff_numerator"] = (
        working["_value"] * working["_weight"]
    ).where(dropoff, 0.0)
    working["_dropoff_denominator"] = working["_weight"].where(dropoff, 0.0)
    working["_valid_value"] = working["_value"].where(valid)
    working["_tail_event"] = tail_event

    grouped = working.groupby(keys, sort=False, observed=True)
    summary = grouped.agg(
        _row_count=("route_sequence", "size"),
        _valid_count=("_valid", "sum"),
        _weighted_numerator=("_weighted_numerator", "sum"),
        _weighted_denominator=("_weighted_denominator", "sum"),
        _maximum=("_valid_value", "max"),
        _tail_numerator=("_tail_numerator", "sum"),
        _tail_denominator=("_tail_denominator", "sum"),
        _tail_event_present=("_tail_event", "max"),
        _pickup_numerator=("_pickup_numerator", "sum"),
        _pickup_denominator=("_pickup_denominator", "sum"),
        _dropoff_numerator=("_dropoff_numerator", "sum"),
        _dropoff_denominator=("_dropoff_denominator", "sum"),
    )

    group_codes = grouped.ngroup()
    non_tail_blocks = (~tail_event).groupby(group_codes, sort=False).cumsum()
    run_lengths = tail_event.astype(np.int32).groupby(
        [group_codes, non_tail_blocks],
        sort=False,
    ).cumsum()
    maximum_run = run_lengths.groupby(group_codes, sort=False).max()
    summary["_maximum_run"] = maximum_run.reindex(
        np.arange(len(summary)), fill_value=0
    ).to_numpy(dtype=float)

    top = working.loc[valid, [*keys, "route_sequence", "_value"]].sort_values(
        [*keys, "_value"],
        ascending=[True, True, True, False],
        kind="stable",
    )
    top = top.groupby(keys, sort=False, observed=True).head(5)
    top_positions = top.groupby(keys, sort=False, observed=True)[
        "route_sequence"
    ].agg(lambda values: json.dumps(values.astype(int).tolist()))
    summary["_top_positions"] = top_positions.reindex(summary.index).fillna("[]")

    def ratio(numerator: str, denominator: str) -> np.ndarray:
        top_values = summary[numerator].to_numpy(dtype=float)
        bottom_values = summary[denominator].to_numpy(dtype=float)
        return np.divide(
            top_values,
            bottom_values,
            out=np.full(len(summary), np.nan),
            where=bottom_values > 0,
        )

    result = summary.reset_index().loc[:, keys].copy()
    result[f"{dimension}_weighted_mean"] = ratio(
        "_weighted_numerator", "_weighted_denominator"
    )
    result[f"{dimension}_max"] = summary["_maximum"].to_numpy(dtype=float)
    result[f"{dimension}_tail_mean"] = ratio(
        "_tail_numerator", "_tail_denominator"
    )
    valid_count = summary["_valid_count"].to_numpy(dtype=float)
    result[f"{dimension}_tail_persistence"] = np.divide(
        summary["_maximum_run"].to_numpy(dtype=float),
        np.maximum(valid_count, 1.0),
    )
    result[f"{dimension}_coverage_share"] = valid_count / summary[
        "_row_count"
    ].to_numpy(dtype=float)
    result[f"{dimension}_tail_event_present"] = summary[
        "_tail_event_present"
    ].to_numpy(dtype=bool)
    result[f"{dimension}_top5_route_positions"] = summary[
        "_top_positions"
    ].to_numpy(dtype=object)
    result[f"{dimension}_pickup_side_exposure"] = ratio(
        "_pickup_numerator", "_pickup_denominator"
    )
    result[f"{dimension}_dropoff_side_exposure"] = ratio(
        "_dropoff_numerator", "_dropoff_denominator"
    )
    return result
