"""Build Stage 1 v3 primitives from direct GPS interval observations.

This module never interprets a traversal-level flag as proof of an observed
travel time.  The only dynamic evidence admitted is an interval satisfying the
frozen ``direct_observed && label_valid`` predicate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .schema import ContractError

if TYPE_CHECKING:
    from .config import Stage1V3Config


DIRECT_REQUIRED_COLUMNS = {
    "order_id",
    "gps_interval_id",
    "traversal_id",
    "canonical_edge_uid",
    "interval_start_time",
    "interval_end_time",
    "observed_travel_time_s",
    "observed_distance_m",
    "observed_speed_mps",
    "measurement_source",
    "label_valid",
}

KNOWN_MEASUREMENT_SOURCES = {
    "direct_observed",
    "interval_supported",
    "engine_interpolated",
    "unresolved",
}

TRAVERSAL_CONTEXT_COLUMNS = {
    "order_id",
    "traversal_id",
    "route_sequence",
    "canonical_edge_uid",
    "observed_directed_edge_uid",
    "observed_from_node",
    "observed_to_node",
    "observed_direction",
    "synthetic_reverse_edge",
    "osm_direction_disagreement",
    "canonical_mapping_available",
    "mapping_status",
    "osm_oneway",
    "allocated_distance_m",
    "measurement_source",
}

ROUTE_CONTEXT_COLUMNS = {
    "order_id",
    "route_sequence",
    "canonical_edge_uid",
    "observed_directed_edge_uid",
    "observed_from_node",
    "observed_to_node",
    "observed_direction",
    "synthetic_reverse_edge",
    "osm_direction_disagreement",
    "canonical_mapping_available",
    "mapping_status",
    "osm_oneway",
    "canonical_highway",
    "length_m",
    "measurement_source",
}

TRAVERSAL_PRIMITIVE_COLUMNS = (
    "order_id",
    "traversal_id",
    "route_sequence",
    "canonical_edge_uid",
    "observed_directed_edge_uid",
    "observed_from_node",
    "observed_to_node",
    "observed_direction",
    "synthetic_reverse_edge",
    "osm_direction_disagreement",
    "canonical_mapping_available",
    "mapping_status",
    "osm_oneway",
    "canonical_highway",
    "observation_window_start_time",
    "observation_window_end_time",
    "direct_interval_count",
    "direct_observed_time_s",
    "direct_observed_distance_m",
    "allocated_distance_m",
    "direct_distance_coverage_share",
    "direct_distance_exceeds_allocated",
    "time_weighted_speed_mean_mps",
    "time_weighted_speed_cv",
    "speed_cv_bounded",
    "low_speed_time_share",
    "stop_time_share",
    "acceleration_rms_mps2",
    "acceleration_rms_bounded",
    "maximum_absolute_acceleration_mps2",
    "acceleration_pair_count",
    "maximum_internal_gap_s",
    "discontinuous_direct_window",
    "lcs_raw",
    "lcs_available",
    "lcs_unavailable_reason",
    "observed_sec_per_m",
    "time_bin_30m",
    "weekday_type",
    "peak_offpeak",
    "measurement_source",
    "label_schema_version",
)


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame[name], errors="coerce")


def _finite(series: pd.Series) -> pd.Series:
    return pd.Series(np.isfinite(series.to_numpy(dtype=np.float64)), index=series.index)


def _section_value(section: dict[str, Any], name: str) -> Any:
    if name not in section:
        raise ContractError(f"Stage1 v3 config section is missing required key: {name}")
    return section[name]


def _minute_of_day(value: str) -> int:
    try:
        hour_text, minute_text = value.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ContractError(f"invalid HH:MM time value: {value!r}") from exc
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ContractError(f"invalid HH:MM time value: {value!r}")
    return hour * 60 + minute


def _lcs_component_weights(lcs_config: dict[str, Any]) -> np.ndarray:
    component_names = (
        "low_speed_time_share",
        "stop_time_share",
        "speed_cv_bounded",
        "acceleration_rms_bounded",
    )
    component_config = _section_value(lcs_config, "components")
    if not isinstance(component_config, dict):
        raise ContractError("lcs.components must be a mapping")
    try:
        weights = np.asarray(
            [
                float(component_config[name]["weight"])
                for name in component_names
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("invalid LCS component weights") from exc
    if (
        not np.isfinite(weights).all()
        or (weights < 0).any()
        or abs(float(weights.sum()) - 1.0) > 1e-12
    ):
        raise ContractError("LCS component weights must be finite and sum to one")
    return weights


def select_direct_observations(
    observations: pd.DataFrame,
    *,
    tolerance_s: float = 1e-6,
    speed_tolerance_mps: float = 1e-6,
) -> pd.DataFrame:
    """Return strictly valid direct observations and reject malformed evidence."""

    missing = sorted(DIRECT_REQUIRED_COLUMNS - set(observations.columns))
    if missing:
        raise ContractError(
            f"link_interval_observations missing Stage1 v3 columns: {missing}"
        )

    if observations["measurement_source"].isna().any():
        raise ContractError("measurement_source must be non-null")
    sources = observations["measurement_source"].astype(str)
    unknown = sorted(set(sources) - KNOWN_MEASUREMENT_SOURCES)
    if unknown:
        raise ContractError(f"unknown measurement_source values: {unknown}")

    duplicate = observations.duplicated(["order_id", "gps_interval_id"], keep=False)
    if duplicate.any():
        keys = observations.loc[
            duplicate, ["order_id", "gps_interval_id"]
        ].head(5)
        raise ContractError(
            "duplicate direct interval keys: "
            f"{keys.astype(str).to_dict(orient='records')}"
        )

    label_valid_raw = observations["label_valid"]
    strict_boolean = label_valid_raw.map(
        lambda value: isinstance(value, (bool, np.bool_))
    )
    if not strict_boolean.all():
        raise ContractError("label_valid must contain only non-null booleans")
    label_valid = label_valid_raw.eq(True)
    selected = observations.loc[
        observations["measurement_source"].eq("direct_observed") & label_valid
    ].copy()
    if selected.empty:
        return selected

    start = _numeric(selected, "interval_start_time")
    end = _numeric(selected, "interval_end_time")
    duration = _numeric(selected, "observed_travel_time_s")
    distance = _numeric(selected, "observed_distance_m")
    valid = (
        _finite(start)
        & _finite(end)
        & _finite(duration)
        & _finite(distance)
        & end.gt(start)
        & duration.gt(0)
        & distance.ge(0)
        & selected["canonical_edge_uid"].notna()
        & (end - start - duration).abs().le(float(tolerance_s))
    )
    if not valid.all():
        bad = selected.loc[
            ~valid,
            [
                "order_id",
                "gps_interval_id",
                "traversal_id",
                "interval_start_time",
                "interval_end_time",
                "observed_travel_time_s",
                "observed_distance_m",
            ],
        ].head(5)
        raise ContractError(
            "invalid direct observation rows: "
            f"{bad.astype(str).to_dict(orient='records')}"
        )

    calculated_speed = distance / duration
    supplied_speed = _numeric(selected, "observed_speed_mps")
    speed_valid = (
        _finite(supplied_speed)
        & supplied_speed.ge(0)
        & (supplied_speed - calculated_speed).abs().le(
            float(speed_tolerance_mps)
        )
    )
    if not speed_valid.all():
        raise ContractError("observed_speed_mps is inconsistent with distance/time")

    selected["interval_start_time"] = start
    selected["interval_end_time"] = end
    selected["observed_travel_time_s"] = duration
    selected["observed_distance_m"] = distance
    selected["observed_speed_mps"] = calculated_speed
    ordered = selected.sort_values(
        [
            "order_id",
            "traversal_id",
            "interval_start_time",
            "interval_end_time",
            "gps_interval_id",
        ],
        kind="stable",
    )
    maximum_end = ordered.groupby(
        ["order_id", "traversal_id"], sort=False
    )["interval_end_time"].cummax()
    previous_maximum_end = maximum_end.groupby(
        [ordered["order_id"], ordered["traversal_id"]], sort=False
    ).shift()
    overlap = previous_maximum_end.notna() & ordered[
        "interval_start_time"
    ].lt(previous_maximum_end - float(tolerance_s))
    if overlap.any():
        raise ContractError("direct observations overlap within a traversal")
    return selected


def build_interval_labels(
    observations: pd.DataFrame,
    traversals: pd.DataFrame,
    route_parts: pd.DataFrame,
    config: "Stage1V3Config",
) -> pd.DataFrame:
    """Enrich direct intervals without crossing traversal or observation gaps."""

    direct_cfg = config.section("direct")
    lcs_cfg = config.section("lcs")
    direct = select_direct_observations(
        observations,
        tolerance_s=float(_section_value(direct_cfg, "duration_tolerance_s")),
        speed_tolerance_mps=float(
            _section_value(direct_cfg, "speed_tolerance_mps")
        ),
    )
    joined = _attach_context(direct, traversals, route_parts)
    joined = joined.sort_values(
        ["order_id", "traversal_id", "interval_start_time", "gps_interval_id"],
        kind="stable",
    ).reset_index(drop=True)

    stop_speed = float(_section_value(lcs_cfg, "stop_speed_mps"))
    low_speed = float(_section_value(lcs_cfg, "low_speed_mps"))
    maximum_speed = float(_section_value(lcs_cfg, "maximum_physical_speed_mps"))
    maximum_acceleration = float(
        _section_value(lcs_cfg, "maximum_absolute_acceleration_mps2")
    )
    _lcs_component_weights(lcs_cfg)
    maximum_gap = float(_section_value(lcs_cfg, "maximum_adjacent_gap_s"))
    grouped = joined.groupby(["order_id", "traversal_id"], sort=False)
    joined["previous_direct_gps_interval_id"] = grouped["gps_interval_id"].shift(1)
    previous_end = grouped["interval_end_time"].shift(1)
    previous_speed = grouped["observed_speed_mps"].shift(1)
    midpoint = (
        joined["interval_start_time"] + joined["interval_end_time"]
    ) / 2.0
    previous_midpoint = midpoint.groupby(
        [joined["order_id"], joined["traversal_id"]], sort=False
    ).shift(1)
    joined["adjacent_gap_s"] = joined["interval_start_time"] - previous_end
    joined["speed_delta_mps"] = joined["observed_speed_mps"] - previous_speed
    midpoint_delta = midpoint - previous_midpoint
    adjacent = (
        joined["previous_direct_gps_interval_id"].notna()
        & joined["adjacent_gap_s"].ge(-1e-6)
        & joined["adjacent_gap_s"].le(maximum_gap)
        & midpoint_delta.gt(0)
    )
    joined["acceleration_mps2"] = np.where(
        adjacent,
        joined["speed_delta_mps"] / midpoint_delta,
        np.nan,
    )
    physical_speed = joined["observed_speed_mps"].between(
        0.0, maximum_speed, inclusive="both"
    )
    physical_acceleration = (
        joined["acceleration_mps2"].isna()
        | joined["acceleration_mps2"].abs().le(maximum_acceleration)
    )
    joined["kinematic_sequence_valid"] = (
        adjacent & physical_speed & physical_acceleration
    )
    joined["is_low_speed"] = joined["observed_speed_mps"].lt(low_speed)
    joined["is_stop"] = joined["observed_speed_mps"].le(stop_speed)
    joined["lcs_component_available"] = physical_speed & physical_acceleration
    joined["lcs_component_unavailable_reason"] = np.where(
        ~physical_speed,
        "IMPOSSIBLE_DIRECT_SPEED",
        np.where(
            ~physical_acceleration,
            "IMPOSSIBLE_DIRECT_ACCELERATION",
            "",
        ),
    )
    joined["interval_duration_s"] = joined["observed_travel_time_s"]
    joined["label_schema_version"] = "stage1_label_schema_v3"
    return joined


def _attach_context(
    observations: pd.DataFrame,
    traversals: pd.DataFrame,
    route_parts: pd.DataFrame,
) -> pd.DataFrame:
    missing_traversal = sorted(TRAVERSAL_CONTEXT_COLUMNS - set(traversals.columns))
    if missing_traversal:
        raise ContractError(
            f"link_traversals missing Stage1 v3 columns: {missing_traversal}"
        )
    missing_route = sorted(ROUTE_CONTEXT_COLUMNS - set(route_parts.columns))
    if missing_route:
        raise ContractError(f"route_parts missing Stage1 v3 columns: {missing_route}")

    if traversals.duplicated(["order_id", "traversal_id"]).any():
        raise ContractError("duplicate (order_id, traversal_id) in link_traversals")
    if route_parts.duplicated(["order_id", "route_sequence"]).any():
        raise ContractError("duplicate (order_id, route_sequence) in route_parts")

    direction_columns = [
        "observed_directed_edge_uid",
        "observed_from_node",
        "observed_to_node",
        "observed_direction",
        "synthetic_reverse_edge",
        "osm_direction_disagreement",
        "canonical_mapping_available",
        "mapping_status",
        "osm_oneway",
    ]
    traversal_context = traversals[
        [
            "order_id",
            "traversal_id",
            "route_sequence",
            "canonical_edge_uid",
            *direction_columns,
            "allocated_distance_m",
            "measurement_source",
        ]
    ].rename(
        columns={
            "canonical_edge_uid": "traversal_edge_uid",
            "measurement_source": "traversal_measurement_source",
            **{
                column: f"traversal_{column}"
                for column in direction_columns
            },
        }
    )
    joined = observations.merge(
        traversal_context,
        on=["order_id", "traversal_id"],
        how="left",
        validate="many_to_one",
        indicator="_traversal_join",
    )
    if joined["_traversal_join"].ne("both").any():
        raise ContractError("direct observation references a missing traversal")
    if (
        joined["canonical_edge_uid"].astype(str)
        != joined["traversal_edge_uid"].astype(str)
    ).any():
        raise ContractError("direct observation canonical edge differs from traversal")
    if not joined["traversal_measurement_source"].eq("direct_observed").all():
        raise ContractError(
            "direct observation references a non-direct traversal"
        )
    for column in direction_columns:
        observation_column = column
        traversal_column = f"traversal_{column}"
        if observation_column in joined:
            matches = (
                joined[observation_column].astype("string").fillna("<NULL>")
                == joined[traversal_column].astype("string").fillna("<NULL>")
            )
            if not matches.all():
                raise ContractError(
                    f"direct observation differs from traversal on {column}"
                )
        else:
            joined[observation_column] = joined[traversal_column]

    route_context = route_parts[
        [
            "order_id",
            "route_sequence",
            "canonical_edge_uid",
            *direction_columns,
            "canonical_highway",
            "length_m",
            "measurement_source",
        ]
    ].rename(
        columns={
            "canonical_edge_uid": "route_edge_uid",
            **{
                column: f"route_{column}"
                for column in direction_columns
            },
            "length_m": "route_part_distance_m",
            "measurement_source": "route_measurement_source",
        }
    )
    joined = joined.merge(
        route_context,
        on=["order_id", "route_sequence"],
        how="left",
        validate="many_to_one",
        indicator="_route_join",
    )
    if joined["_route_join"].ne("both").any():
        raise ContractError("direct traversal references a missing route part")
    if (
        joined["canonical_edge_uid"].astype(str)
        != joined["route_edge_uid"].astype(str)
    ).any():
        raise ContractError("traversal canonical edge differs from route part")
    if not joined["route_measurement_source"].eq("direct_observed").all():
        raise ContractError("direct traversal references a non-direct route part")
    for column in direction_columns:
        matches = (
            joined[column].astype("string").fillna("<NULL>")
            == joined[f"route_{column}"].astype("string").fillna("<NULL>")
        )
        if not matches.all():
            raise ContractError(
                f"traversal actual edge identity differs from route part on {column}"
            )
    return joined.drop(
        columns=[
            "_traversal_join",
            "_route_join",
            "traversal_edge_uid",
            "route_edge_uid",
            "traversal_measurement_source",
            "route_measurement_source",
            *[f"traversal_{column}" for column in direction_columns],
            *[f"route_{column}" for column in direction_columns],
        ]
    )


def _acceleration_stats(
    group: pd.DataFrame,
    *,
    maximum_gap_s: float,
    minimum_pairs: int,
) -> tuple[float, int, float]:
    ordered = group.sort_values(
        ["interval_start_time", "gps_interval_id"], kind="stable"
    )
    starts = ordered["interval_start_time"].to_numpy(dtype=np.float64)
    ends = ordered["interval_end_time"].to_numpy(dtype=np.float64)
    speeds = ordered["observed_speed_mps"].to_numpy(dtype=np.float64)
    if starts.size < 2:
        return float("nan"), 0, float("nan")

    midpoints = (starts + ends) / 2.0
    midpoint_delta = np.diff(midpoints)
    gap = starts[1:] - ends[:-1]
    valid = (
        np.isfinite(midpoint_delta)
        & (midpoint_delta > 0)
        & np.isfinite(gap)
        & (gap >= -1e-6)
        & (gap <= maximum_gap_s)
        & np.isfinite(speeds[:-1])
        & np.isfinite(speeds[1:])
    )
    pair_count = int(valid.sum())
    if pair_count < minimum_pairs:
        return float("nan"), pair_count, float("nan")
    acceleration = np.diff(speeds)[valid] / midpoint_delta[valid]
    return (
        float(np.sqrt(np.mean(np.square(acceleration)))),
        pair_count,
        float(np.max(np.abs(acceleration))),
    )


def _unavailable_reason(
    interval_count: int,
    observed_time_s: float,
    observed_distance_m: float,
    speed_cv: float,
    acceleration_rms: float,
    maximum_observed_acceleration_mps2: float,
    maximum_observed_speed_mps: float,
    discontinuous_direct_window: bool,
    *,
    minimum_intervals: int,
    minimum_time_s: float,
    minimum_distance_m: float,
    maximum_speed_mps: float,
    maximum_acceleration_mps2: float,
) -> str:
    if interval_count < minimum_intervals:
        return "INSUFFICIENT_DIRECT_INTERVALS"
    if observed_time_s < minimum_time_s:
        return "INSUFFICIENT_DIRECT_TIME"
    if observed_distance_m < minimum_distance_m:
        return "INSUFFICIENT_DIRECT_DISTANCE"
    if discontinuous_direct_window:
        return "DISCONTINUOUS_DIRECT_WINDOW"
    if maximum_observed_speed_mps > maximum_speed_mps:
        return "IMPOSSIBLE_DIRECT_SPEED"
    if not np.isfinite(speed_cv):
        return "SPEED_VARIABILITY_UNAVAILABLE"
    if not np.isfinite(acceleration_rms):
        return "ACCELERATION_VARIABILITY_UNAVAILABLE"
    if maximum_observed_acceleration_mps2 > maximum_acceleration_mps2:
        return "IMPOSSIBLE_DIRECT_ACCELERATION"
    return ""


def build_traversal_primitives(
    observations: pd.DataFrame,
    traversals: pd.DataFrame,
    route_parts: pd.DataFrame,
    config: "Stage1V3Config",
) -> pd.DataFrame:
    """Aggregate direct intervals into visit-aware traversal primitives."""

    direct_cfg = config.section("direct")
    lcs_cfg = config.section("lcs")
    rts_cfg = config.section("rts")
    reference_cfg = config.section("reference")
    cohort_cfg = config.section("cohort_reference")
    tolerance_s = float(_section_value(direct_cfg, "duration_tolerance_s"))
    direct = select_direct_observations(
        observations,
        tolerance_s=tolerance_s,
        speed_tolerance_mps=float(
            _section_value(direct_cfg, "speed_tolerance_mps")
        ),
    )
    if direct.empty:
        return pd.DataFrame(columns=TRAVERSAL_PRIMITIVE_COLUMNS)

    joined = _attach_context(direct, traversals, route_parts)
    if joined.duplicated(["order_id", "gps_interval_id"]).any():
        raise ContractError("context joins duplicated a direct GPS interval")

    minimum_intervals = int(
        _section_value(lcs_cfg, "minimum_direct_intervals_per_traversal")
    )
    minimum_time_s = float(_section_value(lcs_cfg, "minimum_observed_time_s"))
    minimum_lcs_distance = float(
        _section_value(lcs_cfg, "minimum_direct_observed_distance_m")
    )
    stop_speed = float(_section_value(lcs_cfg, "stop_speed_mps"))
    low_speed = float(_section_value(lcs_cfg, "low_speed_mps"))
    speed_cv_scale = float(_section_value(lcs_cfg, "speed_cv_scale"))
    acceleration_scale = float(
        _section_value(lcs_cfg, "acceleration_rms_scale_mps2")
    )
    maximum_gap_s = float(_section_value(lcs_cfg, "maximum_adjacent_gap_s"))
    minimum_pairs = int(_section_value(lcs_cfg, "minimum_acceleration_pairs"))
    maximum_speed = float(_section_value(lcs_cfg, "maximum_physical_speed_mps"))
    maximum_acceleration = float(
        _section_value(lcs_cfg, "maximum_absolute_acceleration_mps2")
    )
    minimum_rts_distance = float(
        _section_value(rts_cfg, "minimum_direct_observed_distance_m")
    )
    reference_minimum_distance = float(
        _section_value(reference_cfg, "minimum_observed_distance_m")
    )
    if abs(minimum_rts_distance - reference_minimum_distance) > 1e-12:
        raise ContractError(
            "RTS and reference minimum observed distances must be identical"
        )
    minimum_rts_time = float(
        _section_value(rts_cfg, "minimum_direct_observed_time_s")
    )
    time_bin_minutes = int(
        _section_value(cohort_cfg, "time_bin_minutes")
    )
    peak_windows_raw = _section_value(cohort_cfg, "peak_windows_local")
    if (
        not isinstance(peak_windows_raw, list)
        or any(
            not isinstance(window, list) or len(window) != 2
            for window in peak_windows_raw
        )
    ):
        raise ContractError("cohort_reference.peak_windows_local is invalid")
    peak_windows = tuple(
        (_minute_of_day(window[0]), _minute_of_day(window[1]))
        for window in peak_windows_raw
    )
    if time_bin_minutes <= 0 or any(start > end for start, end in peak_windows):
        raise ContractError("invalid cohort time-bin or peak-window configuration")
    if stop_speed < 0 or low_speed <= stop_speed:
        raise ContractError("LCS speed thresholds must satisfy 0 <= stop < low")
    if speed_cv_scale <= 0 or acceleration_scale <= 0:
        raise ContractError("LCS normalization scales must be positive")
    component_weights = _lcs_component_weights(lcs_cfg)

    rows: list[dict[str, Any]] = []
    group_keys = ["order_id", "traversal_id"]
    for (order_id, traversal_id), group in joined.groupby(
        group_keys, sort=False, dropna=False
    ):
        duration = group["observed_travel_time_s"].to_numpy(dtype=np.float64)
        distance = group["observed_distance_m"].to_numpy(dtype=np.float64)
        speed = group["observed_speed_mps"].to_numpy(dtype=np.float64)
        observed_time_s = float(duration.sum())
        observed_distance_m = float(distance.sum())
        interval_count = int(len(group))
        speed_mean = (
            float(np.average(speed, weights=duration))
            if observed_time_s > 0
            else float("nan")
        )
        speed_variance = (
            float(np.average(np.square(speed - speed_mean), weights=duration))
            if observed_time_s > 0 and np.isfinite(speed_mean)
            else float("nan")
        )
        speed_std = (
            float(np.sqrt(max(speed_variance, 0.0)))
            if np.isfinite(speed_variance)
            else float("nan")
        )
        speed_cv = (
            speed_std / speed_mean
            if np.isfinite(speed_std) and speed_mean > 0
            else float("nan")
        )
        (
            acceleration_rms,
            acceleration_pair_count,
            maximum_observed_acceleration,
        ) = _acceleration_stats(
            group,
            maximum_gap_s=maximum_gap_s,
            minimum_pairs=minimum_pairs,
        )
        ordered = group.sort_values(
            ["interval_start_time", "gps_interval_id"], kind="stable"
        )
        internal_gaps = (
            ordered["interval_start_time"].to_numpy(dtype=np.float64)[1:]
            - ordered["interval_end_time"].to_numpy(dtype=np.float64)[:-1]
        )
        if (internal_gaps < -tolerance_s).any():
            raise ContractError(
                "direct intervals overlap within a traversal primitive"
            )
        maximum_internal_gap = (
            float(np.max(internal_gaps))
            if internal_gaps.size
            else float("nan")
        )
        discontinuous_direct_window = bool(
            internal_gaps.size
            and np.any(internal_gaps > maximum_gap_s)
        )
        low_speed_share = (
            float(duration[speed < low_speed].sum() / observed_time_s)
            if observed_time_s > 0
            else float("nan")
        )
        stop_time_share = (
            float(duration[speed <= stop_speed].sum() / observed_time_s)
            if observed_time_s > 0
            else float("nan")
        )
        reason = _unavailable_reason(
            interval_count,
            observed_time_s,
            observed_distance_m,
            speed_cv,
            acceleration_rms,
            maximum_observed_acceleration,
            float(np.max(speed)) if speed.size else float("nan"),
            discontinuous_direct_window,
            minimum_intervals=minimum_intervals,
            minimum_time_s=minimum_time_s,
            minimum_distance_m=minimum_lcs_distance,
            maximum_speed_mps=maximum_speed,
            maximum_acceleration_mps2=maximum_acceleration,
        )
        lcs_available = reason == ""
        if lcs_available:
            speed_cv_bounded = speed_cv / (speed_cv + speed_cv_scale)
            acceleration_rms_bounded = acceleration_rms / (
                acceleration_rms + acceleration_scale
            )
            components = np.asarray(
                [
                    low_speed_share,
                    stop_time_share,
                    speed_cv_bounded,
                    acceleration_rms_bounded,
                ],
                dtype=np.float64,
            )
            lcs_raw = float(np.average(components, weights=component_weights))
        else:
            speed_cv_bounded = float("nan")
            acceleration_rms_bounded = float("nan")
            lcs_raw = float("nan")

        observed_sec_per_m = (
            observed_time_s / observed_distance_m
            if observed_distance_m >= minimum_rts_distance
            and observed_time_s >= minimum_rts_time
            and not discontinuous_direct_window
            else float("nan")
        )
        first = group.sort_values(
            ["interval_start_time", "gps_interval_id"], kind="stable"
        ).iloc[0]
        local = pd.to_datetime(
            float(first["interval_start_time"]), unit="s", utc=True
        ).tz_convert(str(config.section("time")["timezone"]))
        minute = int(local.hour) * 60 + int(local.minute)
        allocated_distance = float(
            pd.to_numeric(group["allocated_distance_m"], errors="coerce").iloc[0]
        )
        if not np.isfinite(allocated_distance) or allocated_distance < 0:
            raise ContractError(
                "allocated_distance_m must be finite and non-negative"
            )
        direct_distance_coverage_share = (
            observed_distance_m / allocated_distance
            if allocated_distance > 0
            else float("nan")
        )
        direct_distance_exceeds_allocated = bool(
            allocated_distance >= 0
            and observed_distance_m
            > allocated_distance
            + float(
                _section_value(
                    direct_cfg,
                    "distance_identity_tolerance_m",
                )
            )
        )
        rows.append(
            {
                "order_id": str(order_id),
                "traversal_id": int(traversal_id),
                "route_sequence": int(first["route_sequence"]),
                "canonical_edge_uid": str(first["canonical_edge_uid"]),
                "observed_directed_edge_uid": str(
                    first["observed_directed_edge_uid"]
                ),
                "observed_from_node": int(first["observed_from_node"]),
                "observed_to_node": int(first["observed_to_node"]),
                "observed_direction": str(first["observed_direction"]),
                "synthetic_reverse_edge": bool(first["synthetic_reverse_edge"]),
                "osm_direction_disagreement": bool(
                    first["osm_direction_disagreement"]
                ),
                "canonical_mapping_available": bool(
                    first["canonical_mapping_available"]
                ),
                "mapping_status": str(first["mapping_status"]),
                "osm_oneway": first["osm_oneway"],
                "canonical_highway": str(first["canonical_highway"]),
                "observation_window_start_time": float(
                    group["interval_start_time"].min()
                ),
                "observation_window_end_time": float(
                    group["interval_end_time"].max()
                ),
                "direct_interval_count": interval_count,
                "direct_observed_time_s": observed_time_s,
                "direct_observed_distance_m": observed_distance_m,
                "allocated_distance_m": allocated_distance,
                "direct_distance_coverage_share": direct_distance_coverage_share,
                "direct_distance_exceeds_allocated": (
                    direct_distance_exceeds_allocated
                ),
                "time_weighted_speed_mean_mps": speed_mean,
                "time_weighted_speed_cv": speed_cv,
                "speed_cv_bounded": speed_cv_bounded,
                "low_speed_time_share": low_speed_share,
                "stop_time_share": stop_time_share,
                "acceleration_rms_mps2": acceleration_rms,
                "acceleration_rms_bounded": acceleration_rms_bounded,
                "maximum_absolute_acceleration_mps2": maximum_observed_acceleration,
                "acceleration_pair_count": acceleration_pair_count,
                "maximum_internal_gap_s": maximum_internal_gap,
                "discontinuous_direct_window": discontinuous_direct_window,
                "lcs_raw": lcs_raw,
                "lcs_available": lcs_available,
                "lcs_unavailable_reason": reason,
                "observed_sec_per_m": observed_sec_per_m,
                "time_bin_30m": minute // time_bin_minutes,
                "weekday_type": (
                    "weekday" if int(local.dayofweek) < 5 else "weekend"
                ),
                "peak_offpeak": (
                    "peak"
                    if any(
                        start <= minute <= end
                        for start, end in peak_windows
                    )
                    else "offpeak"
                ),
                "measurement_source": "direct_observed",
                "label_schema_version": "stage1_label_schema_v3",
            }
        )
    return pd.DataFrame(rows).reindex(columns=TRAVERSAL_PRIMITIVE_COLUMNS)
