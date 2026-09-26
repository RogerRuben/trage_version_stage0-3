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
    "crawl_time_share",
    "stop_time_share",
    "acceleration_rms_mps2",
    "acceleration_rms_bounded",
    "maximum_absolute_acceleration_mps2",
    "acceleration_pair_count",
    "acceleration_weight_s",
    "maximum_internal_gap_s",
    "discontinuous_direct_window",
    "lcs_raw",
    "lcs_available",
    "lcs_unavailable_reason",
    "rts_measurement_available",
    "rts_measurement_unavailable_reason",
    "rts_direct_speed_valid",
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
        "crawl_time_share",
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
    joined["is_stop"] = joined["observed_speed_mps"].le(stop_speed)
    joined["is_crawl"] = (
        joined["observed_speed_mps"].gt(stop_speed)
        & joined["observed_speed_mps"].lt(low_speed)
    )
    joined["is_low_speed_total"] = joined["is_stop"] | joined["is_crawl"]
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
) -> tuple[float, int, float, float]:
    ordered = group.sort_values(
        ["interval_start_time", "gps_interval_id"], kind="stable"
    )
    starts = ordered["interval_start_time"].to_numpy(dtype=np.float64)
    ends = ordered["interval_end_time"].to_numpy(dtype=np.float64)
    speeds = ordered["observed_speed_mps"].to_numpy(dtype=np.float64)
    if starts.size < 2:
        return float("nan"), 0, 0.0, float("nan")

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
        return float("nan"), pair_count, float(midpoint_delta[valid].sum()), float("nan")
    acceleration = np.diff(speeds)[valid] / midpoint_delta[valid]
    weights = midpoint_delta[valid]
    acceleration_weight_s = float(weights.sum())
    if not np.isfinite(acceleration_weight_s) or acceleration_weight_s <= 0:
        return float("nan"), pair_count, acceleration_weight_s, float("nan")
    return (
        float(
            np.sqrt(
                np.sum(weights * np.square(acceleration))
                / acceleration_weight_s
            )
        ),
        pair_count,
        acceleration_weight_s,
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
    direct_distance_exceeds_allocated: bool,
    *,
    minimum_intervals: int,
    minimum_time_s: float,
    minimum_distance_m: float,
    maximum_speed_mps: float,
    maximum_acceleration_mps2: float,
) -> str:
    if direct_distance_exceeds_allocated:
        return "DIRECT_DISTANCE_EXCEEDS_TRAVERSAL"
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


def _rts_measurement_reason(
    observed_time_s: float,
    observed_distance_m: float,
    *,
    minimum_time_s: float,
    minimum_distance_m: float,
    direct_distance_exceeds_allocated: bool,
    discontinuous_direct_window: bool,
    maximum_observed_speed_mps: float,
    maximum_speed_mps: float,
) -> str:
    if observed_time_s < minimum_time_s:
        return "INSUFFICIENT_DIRECT_TIME"
    if observed_distance_m < minimum_distance_m:
        return "INSUFFICIENT_DIRECT_DISTANCE"
    if direct_distance_exceeds_allocated:
        return "DIRECT_DISTANCE_EXCEEDS_TRAVERSAL"
    if discontinuous_direct_window:
        return "DISCONTINUOUS_DIRECT_WINDOW"
    if (
        not np.isfinite(maximum_observed_speed_mps)
        or maximum_observed_speed_mps > maximum_speed_mps
    ):
        return "IMPOSSIBLE_DIRECT_SPEED"
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

    maximum_rts_speed = float(
        _section_value(rts_cfg, "maximum_direct_speed_mps")
    )
    maximum_reference_speed = float(
        _section_value(
            config.section("reference"),
            "maximum_direct_speed_mps",
        )
    )
    if abs(maximum_rts_speed - maximum_reference_speed) > 1e-12:
        raise ContractError("RTS and reference maximum direct speeds differ")

    group_keys = ["order_id", "traversal_id"]
    work = joined.sort_values(
        [*group_keys, "interval_start_time", "gps_interval_id"],
        kind="stable",
    ).copy()
    for column in (
        "observed_travel_time_s",
        "observed_distance_m",
        "observed_speed_mps",
        "interval_start_time",
        "interval_end_time",
        "allocated_distance_m",
    ):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["_weighted_speed"] = (
        work["observed_speed_mps"] * work["observed_travel_time_s"]
    )
    work["_weighted_speed_sq"] = (
        np.square(work["observed_speed_mps"])
        * work["observed_travel_time_s"]
    )
    work["_crawl_time"] = work["observed_travel_time_s"].where(
        work["observed_speed_mps"].gt(stop_speed)
        & work["observed_speed_mps"].lt(low_speed),
        0.0,
    )
    work["_stop_time"] = work["observed_travel_time_s"].where(
        work["observed_speed_mps"].le(stop_speed), 0.0
    )
    same_group = (
        work["order_id"].astype(str).eq(
            work["order_id"].astype(str).shift()
        )
        & work["traversal_id"].eq(work["traversal_id"].shift())
    )
    previous_end = work["interval_end_time"].shift()
    internal_gap = work["interval_start_time"] - previous_end
    internal_gap = internal_gap.where(same_group)
    if internal_gap.lt(-tolerance_s).any():
        raise ContractError(
            "direct intervals overlap within a traversal primitive"
        )
    work["_internal_gap"] = internal_gap
    midpoint = (
        work["interval_start_time"] + work["interval_end_time"]
    ) / 2.0
    midpoint_delta = (midpoint - midpoint.shift()).where(same_group)
    acceleration_valid = (
        midpoint_delta.gt(0)
        & internal_gap.ge(-1e-6)
        & internal_gap.le(maximum_gap_s)
        & np.isfinite(work["observed_speed_mps"])
        & np.isfinite(work["observed_speed_mps"].shift())
    )
    acceleration = (
        (work["observed_speed_mps"] - work["observed_speed_mps"].shift())
        / midpoint_delta
    ).where(acceleration_valid)
    work["_acceleration_pair"] = acceleration_valid.astype(np.int64)
    work["_acceleration_weight"] = midpoint_delta.where(
        acceleration_valid, 0.0
    )
    work["_weighted_acceleration_sq"] = (
        work["_acceleration_weight"] * np.square(acceleration)
    ).fillna(0.0)
    work["_absolute_acceleration"] = acceleration.abs()

    grouped = work.groupby(group_keys, sort=False, dropna=False)
    aggregate = grouped.agg(
        observation_window_start_time=("interval_start_time", "min"),
        observation_window_end_time=("interval_end_time", "max"),
        direct_interval_count=("gps_interval_id", "size"),
        direct_observed_time_s=("observed_travel_time_s", "sum"),
        direct_observed_distance_m=("observed_distance_m", "sum"),
        allocated_distance_m=("allocated_distance_m", "first"),
        _weighted_speed=("_weighted_speed", "sum"),
        _weighted_speed_sq=("_weighted_speed_sq", "sum"),
        _crawl_time=("_crawl_time", "sum"),
        _stop_time=("_stop_time", "sum"),
        _maximum_speed=("observed_speed_mps", "max"),
        maximum_internal_gap_s=("_internal_gap", "max"),
        acceleration_pair_count=("_acceleration_pair", "sum"),
        acceleration_weight_s=("_acceleration_weight", "sum"),
        _weighted_acceleration_sq=("_weighted_acceleration_sq", "sum"),
        maximum_absolute_acceleration_mps2=(
            "_absolute_acceleration",
            "max",
        ),
    ).reset_index()
    context_columns = [
        *group_keys,
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
    ]
    context = work[context_columns].drop_duplicates(
        group_keys, keep="first"
    )
    result = aggregate.merge(
        context,
        on=group_keys,
        how="left",
        validate="one_to_one",
    )
    if (
        ~np.isfinite(result["allocated_distance_m"])
        | result["allocated_distance_m"].lt(0)
    ).any():
        raise ContractError(
            "allocated_distance_m must be finite and non-negative"
        )

    observed_time = result["direct_observed_time_s"].to_numpy(
        dtype=np.float64
    )
    observed_distance = result["direct_observed_distance_m"].to_numpy(
        dtype=np.float64
    )
    allocated_distance = result["allocated_distance_m"].to_numpy(
        dtype=np.float64
    )
    speed_mean = np.divide(
        result["_weighted_speed"].to_numpy(dtype=np.float64),
        observed_time,
        out=np.full(len(result), np.nan),
        where=observed_time > 0,
    )
    speed_second_moment = np.divide(
        result["_weighted_speed_sq"].to_numpy(dtype=np.float64),
        observed_time,
        out=np.full(len(result), np.nan),
        where=observed_time > 0,
    )
    speed_variance = np.maximum(
        speed_second_moment - np.square(speed_mean), 0.0
    )
    speed_cv = np.divide(
        np.sqrt(speed_variance),
        speed_mean,
        out=np.full(len(result), np.nan),
        where=np.isfinite(speed_mean) & (speed_mean > 0),
    )
    pair_count = result["acceleration_pair_count"].to_numpy(
        dtype=np.int64
    )
    acceleration_weight = result["acceleration_weight_s"].to_numpy(
        dtype=np.float64
    )
    acceleration_rms = np.sqrt(
        np.divide(
            result["_weighted_acceleration_sq"].to_numpy(
                dtype=np.float64
            ),
            acceleration_weight,
            out=np.full(len(result), np.nan),
            where=(
                (pair_count >= minimum_pairs)
                & np.isfinite(acceleration_weight)
                & (acceleration_weight > 0)
            ),
        )
    )
    insufficient_pairs = pair_count < minimum_pairs
    result.loc[
        insufficient_pairs, "maximum_absolute_acceleration_mps2"
    ] = np.nan
    maximum_observed_acceleration = pd.to_numeric(
        result["maximum_absolute_acceleration_mps2"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    maximum_group_speed = result["_maximum_speed"].to_numpy(
        dtype=np.float64
    )
    distance_exceeds = (
        observed_distance
        > allocated_distance
        + float(
            _section_value(
                direct_cfg, "distance_identity_tolerance_m"
            )
        )
    )
    discontinuous = (
        pd.to_numeric(
            result["maximum_internal_gap_s"], errors="coerce"
        )
        .gt(maximum_gap_s)
        .to_numpy(dtype=bool)
    )
    crawl_share = np.divide(
        result["_crawl_time"].to_numpy(dtype=np.float64),
        observed_time,
        out=np.full(len(result), np.nan),
        where=observed_time > 0,
    )
    stop_share = np.divide(
        result["_stop_time"].to_numpy(dtype=np.float64),
        observed_time,
        out=np.full(len(result), np.nan),
        where=observed_time > 0,
    )
    speed_cv_bounded = np.divide(
        speed_cv,
        speed_cv + speed_cv_scale,
        out=np.full(len(result), np.nan),
        where=np.isfinite(speed_cv) & (speed_cv >= 0),
    )
    acceleration_rms_bounded = np.divide(
        acceleration_rms,
        acceleration_rms + acceleration_scale,
        out=np.full(len(result), np.nan),
        where=np.isfinite(acceleration_rms) & (acceleration_rms >= 0),
    )

    def prioritized_reason(
        conditions: list[tuple[np.ndarray, str]],
    ) -> np.ndarray:
        values = np.full(len(result), "", dtype=object)
        for condition, reason in conditions:
            assign = (values == "") & np.asarray(condition, dtype=bool)
            values[assign] = reason
        return values

    lcs_reason = prioritized_reason(
        [
            (distance_exceeds, "DIRECT_DISTANCE_EXCEEDS_TRAVERSAL"),
            (
                result["direct_interval_count"].to_numpy()
                < minimum_intervals,
                "INSUFFICIENT_DIRECT_INTERVALS",
            ),
            (
                observed_time < minimum_time_s,
                "INSUFFICIENT_DIRECT_TIME",
            ),
            (
                observed_distance < minimum_lcs_distance,
                "INSUFFICIENT_DIRECT_DISTANCE",
            ),
            (discontinuous, "DISCONTINUOUS_DIRECT_WINDOW"),
            (
                maximum_group_speed > maximum_speed,
                "IMPOSSIBLE_DIRECT_SPEED",
            ),
            (~np.isfinite(speed_cv), "SPEED_VARIABILITY_UNAVAILABLE"),
            (
                ~np.isfinite(acceleration_rms),
                "ACCELERATION_VARIABILITY_UNAVAILABLE",
            ),
            (
                maximum_observed_acceleration > maximum_acceleration,
                "IMPOSSIBLE_DIRECT_ACCELERATION",
            ),
        ]
    )
    lcs_available = lcs_reason == ""
    components = np.column_stack(
        [
            crawl_share,
            stop_share,
            speed_cv_bounded,
            acceleration_rms_bounded,
        ]
    )
    lcs_raw = components @ component_weights
    lcs_raw[~lcs_available] = np.nan

    rts_speed_valid = (
        np.isfinite(maximum_group_speed)
        & (maximum_group_speed <= maximum_rts_speed)
    )
    rts_reason = prioritized_reason(
        [
            (
                observed_time < minimum_rts_time,
                "INSUFFICIENT_DIRECT_TIME",
            ),
            (
                observed_distance < minimum_rts_distance,
                "INSUFFICIENT_DIRECT_DISTANCE",
            ),
            (distance_exceeds, "DIRECT_DISTANCE_EXCEEDS_TRAVERSAL"),
            (discontinuous, "DISCONTINUOUS_DIRECT_WINDOW"),
            (~rts_speed_valid, "IMPOSSIBLE_DIRECT_SPEED"),
        ]
    )
    rts_measurement_available = rts_reason == ""
    observed_sec_per_m = np.divide(
        observed_time,
        observed_distance,
        out=np.full(len(result), np.nan),
        where=rts_measurement_available,
    )
    local = pd.to_datetime(
        result["observation_window_start_time"], unit="s", utc=True
    ).dt.tz_convert(str(config.section("time")["timezone"]))
    minute = (
        local.dt.hour.to_numpy(dtype=np.int64) * 60
        + local.dt.minute.to_numpy(dtype=np.int64)
    )
    peak = np.zeros(len(result), dtype=bool)
    for start, end in peak_windows:
        peak |= (minute >= start) & (minute <= end)

    result["order_id"] = result["order_id"].astype(str)
    result["traversal_id"] = result["traversal_id"].astype(int)
    result["route_sequence"] = result["route_sequence"].astype(int)
    for column in (
        "canonical_edge_uid",
        "observed_directed_edge_uid",
        "observed_direction",
        "mapping_status",
        "canonical_highway",
    ):
        result[column] = result[column].astype(str)
    for column in (
        "observed_from_node",
        "observed_to_node",
    ):
        result[column] = result[column].astype(int)
    for column in (
        "synthetic_reverse_edge",
        "osm_direction_disagreement",
        "canonical_mapping_available",
    ):
        result[column] = result[column].astype(bool)
    result["direct_distance_coverage_share"] = np.divide(
        observed_distance,
        allocated_distance,
        out=np.full(len(result), np.nan),
        where=allocated_distance > 0,
    )
    result["direct_distance_exceeds_allocated"] = distance_exceeds
    result["time_weighted_speed_mean_mps"] = speed_mean
    result["time_weighted_speed_cv"] = speed_cv
    result["speed_cv_bounded"] = speed_cv_bounded
    result["crawl_time_share"] = crawl_share
    result["stop_time_share"] = stop_share
    result["acceleration_rms_mps2"] = acceleration_rms
    result["acceleration_rms_bounded"] = acceleration_rms_bounded
    result["discontinuous_direct_window"] = discontinuous
    result["lcs_raw"] = lcs_raw
    result["lcs_available"] = lcs_available
    result["lcs_unavailable_reason"] = lcs_reason
    result["rts_measurement_available"] = rts_measurement_available
    result["rts_measurement_unavailable_reason"] = rts_reason
    result["rts_direct_speed_valid"] = rts_speed_valid
    result["observed_sec_per_m"] = observed_sec_per_m
    result["time_bin_30m"] = minute // time_bin_minutes
    result["weekday_type"] = np.where(
        local.dt.dayofweek.to_numpy(dtype=np.int64) < 5,
        "weekday",
        "weekend",
    )
    result["peak_offpeak"] = np.where(peak, "peak", "offpeak")
    result["measurement_source"] = "direct_observed"
    result["label_schema_version"] = "stage1_label_schema_v3"
    return result.reindex(columns=TRAVERSAL_PRIMITIVE_COLUMNS)
