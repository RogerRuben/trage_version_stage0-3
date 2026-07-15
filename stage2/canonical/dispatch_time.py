"""Strict order-level decision-time semantics for canonical Stage 2.

All links on an assigned-route proxy share the same information cutoff. Future
estimated link-entry timestamps may describe position, but never authorize newer
traffic observations in the dispatch product.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


DISPATCH_FEATURE_WHITELIST = frozenset({
    "order_id", "date", "decision_time", "origin_lon", "origin_lat",
    "destination_lon", "destination_lat", "route_link_id", "route_link_seq",
    "route_link_count", "route_link_length_m", "route_length_m", "position_ratio",
    "distance_to_destination_ratio", "road_class", "area_grid", "time_bin",
    "hour", "weekday", "is_weekend", "activity_intensity_index", "minor_road",
    "curvature_deg_per_km_link", "link_fragmentation", "endpoint_degree",
    "historical_lcs_mean", "historical_pmis_mean", "historical_rts_mean",
    "historical_iis_applicability", "recent_speed_mean", "recent_traversal_count",
    "feature_availability_timestamp", "requested_level_support_count",
    "fallback_level", "fallback_support_count", "fallback_value_source",
})

FORBIDDEN_DISPATCH_FIELDS = frozenset({
    "actual_link_entry_time", "actual_link_exit_time", "enter_time", "exit_time",
    "realized_service_time", "realized_service_time_sec", "travel_time_sec",
    "true_raw", "true_tail", "lcs_raw", "pmis_raw", "rts_raw", "iis_raw",
    "actual_route", "future_traffic_state", "destination_timestamp",
})


def _utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def attach_dispatch_snapshot(
    route_rows: pd.DataFrame,
    state_rows: pd.DataFrame,
    *,
    route_key: str = "route_link_id",
    state_key: str = "link_id",
    state_time: str = "availability_timestamp",
    value_columns: Sequence[str] = ("recent_speed_mean", "recent_traversal_count"),
) -> pd.DataFrame:
    """Backward-asof join using each order's one immutable decision cutoff."""

    required_route = {"order_id", "decision_time", route_key}
    required_state = {state_key, state_time, *value_columns}
    if missing := sorted(required_route - set(route_rows.columns)):
        raise ValueError(f"route rows missing {missing}")
    if missing := sorted(required_state - set(state_rows.columns)):
        raise ValueError(f"state rows missing {missing}")
    route = route_rows.copy()
    state = state_rows.copy()
    route["decision_time"] = _utc(route["decision_time"])
    state[state_time] = _utc(state[state_time])
    if route["decision_time"].isna().any():
        raise ValueError("all canonical dispatch rows require a valid decision_time")
    route[route_key] = route[route_key].astype(str)
    state[state_key] = state[state_key].astype(str)

    # pandas merge_asof has strict global ordering requirements that are brittle
    # under group keys. Groupwise binary search keeps memory bounded and semantics
    # explicit for the smoke-sized canonical rebuild.
    state_groups = {
        key: group.sort_values(state_time, kind="mergesort")
        for key, group in state.groupby(state_key, sort=False)
    }
    output_parts: list[pd.DataFrame] = []
    for key, group in route.groupby(route_key, sort=False):
        left = group.sort_values("decision_time", kind="mergesort").copy()
        right = state_groups.get(str(key))
        if right is None or right.empty:
            for column in value_columns:
                left[column] = np.nan
            left["feature_availability_timestamp"] = pd.NaT
        else:
            joined = pd.merge_asof(
                left,
                right[[state_time, *value_columns]].rename(
                    columns={state_time: "feature_availability_timestamp"}
                ),
                left_on="decision_time",
                right_on="feature_availability_timestamp",
                direction="backward",
                allow_exact_matches=True,
            )
            left = joined
        output_parts.append(left)
    result = pd.concat(output_parts, ignore_index=True) if output_parts else route.iloc[:0].copy()
    result["prediction_mode"] = "dispatch_time"
    result["information_cutoff"] = result["decision_time"]
    return result.sort_values(["order_id", "route_link_seq"], kind="mergesort").reset_index(drop=True)


def hierarchical_fallback(
    requested_keys: pd.Series,
    level_tables: Sequence[tuple[str, pd.DataFrame]],
    *,
    key_column: str = "key",
    value_column: str = "value",
    support_column: str = "sample_size",
    minimum_support: int = 100,
) -> pd.DataFrame:
    """Resolve a historical feature while preserving requested-level support."""

    requested = requested_keys.astype(str)
    result = pd.DataFrame(index=requested.index)
    result["requested_level_support_count"] = 0
    result["fallback_level"] = pd.NA
    result["fallback_support_count"] = 0
    result["fallback_value_source"] = pd.NA
    result[value_column] = np.nan
    unresolved = pd.Series(True, index=requested.index)
    for level_no, (level_name, table) in enumerate(level_tables, start=1):
        lookup = table.copy()
        lookup[key_column] = lookup[key_column].astype(str)
        lookup = lookup.drop_duplicates(key_column).set_index(key_column)
        supports = requested.map(lookup[support_column]).fillna(0).astype(int)
        values = requested.map(lookup[value_column]).astype(float)
        if level_no == 1:
            result["requested_level_support_count"] = supports
        last_level = level_no == len(level_tables)
        eligible = unresolved & values.notna() & ((supports >= minimum_support) | last_level)
        result.loc[eligible, value_column] = values[eligible]
        result.loc[eligible, "fallback_level"] = level_name
        result.loc[eligible, "fallback_support_count"] = supports[eligible]
        result.loc[eligible, "fallback_value_source"] = f"fit_only:{level_name}"
        unresolved.loc[eligible] = False
    return result


def audit_dispatch_features(frame: pd.DataFrame, model_feature_columns: Sequence[str]) -> dict[str, object]:
    features = set(model_feature_columns)
    forbidden = sorted(features & FORBIDDEN_DISPATCH_FIELDS)
    outside_whitelist = sorted(features - DISPATCH_FEATURE_WHITELIST)
    availability = _utc(frame.get("feature_availability_timestamp", pd.Series(pd.NaT, index=frame.index)))
    decision = _utc(frame["decision_time"])
    timestamp_violations = int((availability.notna() & availability.gt(decision)).sum())
    cutoff_counts = frame.groupby("order_id")["decision_time"].nunique(dropna=False)
    multiple_cutoff_orders = int(cutoff_counts.gt(1).sum())
    passed = not forbidden and not outside_whitelist and timestamp_violations == 0 and multiple_cutoff_orders == 0
    return {
        "status": "PASS" if passed else "FAIL",
        "prediction_mode": "dispatch_time",
        "rows": int(len(frame)),
        "orders": int(frame["order_id"].nunique()),
        "forbidden_model_features": forbidden,
        "features_outside_whitelist": outside_whitelist,
        "availability_after_decision_count": timestamp_violations,
        "orders_with_multiple_decision_cutoffs": multiple_cutoff_orders,
    }
