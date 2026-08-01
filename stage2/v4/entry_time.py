"""Two-pass causal estimated entry-time construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Stage2V4Config
from .contracts import ROUTE_PRIMARY_KEY, Stage2V4ContractError, require_columns
from .history_index import TemporalHistoryIndex


def _local_time_fields(
    timestamp: pd.Series | np.ndarray,
    timezone: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local = pd.to_datetime(timestamp, unit="s", utc=True).tz_convert(timezone)
    time_bin = (local.hour.to_numpy(dtype=np.int16) * 2) + (
        local.minute.to_numpy(dtype=np.int16) >= 30
    ).astype(np.int16)
    hour = local.hour.to_numpy(dtype=np.int16)
    weekday = np.where(local.dayofweek.to_numpy(dtype=np.int16) < 5, "weekday", "weekend")
    return time_bin, hour, weekday


def _exclusive_group_cumsum(
    frame: pd.DataFrame,
    values: np.ndarray,
) -> np.ndarray:
    series = pd.Series(values, index=frame.index)
    cumulative = series.groupby(
        [frame["split"], frame["date"], frame["order_id"]],
        sort=False,
    ).cumsum()
    return (cumulative - series).to_numpy(dtype=np.float64)


def _static_pace(frame: pd.DataFrame, config: Stage2V4Config) -> np.ndarray:
    speed_map = config.section("entry_time").get("static_speed_kph")
    if not isinstance(speed_map, dict) or "__default__" not in speed_map:
        raise Stage2V4ContractError("entry_time.static_speed_kph requires __default__")
    default = float(speed_map["__default__"])
    speed_kph = (
        frame["canonical_highway"].astype(str).map(speed_map).fillna(default).to_numpy(float)
    )
    if not np.isfinite(speed_kph).all() or np.any(speed_kph <= 0):
        raise Stage2V4ContractError("static speed fallback must be finite and positive")
    return 3.6 / speed_kph


def estimate_entry_times(
    route_tokens: pd.DataFrame,
    history: TemporalHistoryIndex,
    config: Stage2V4Config,
) -> pd.DataFrame:
    """Estimate all route-token entry times without reading realized labels."""

    required = {
        *ROUTE_PRIMARY_KEY,
        "decision_time",
        "route_part_length_m",
        "canonical_highway",
        "observed_directed_edge_uid",
    }
    require_columns(route_tokens.columns, required, "entry-time route tokens")
    result = route_tokens.copy()
    result["_input_position"] = np.arange(len(result), dtype=np.int64)
    result.sort_values(list(ROUTE_PRIMARY_KEY), kind="stable", inplace=True)
    if result.duplicated(list(ROUTE_PRIMARY_KEY)).any():
        raise Stage2V4ContractError("entry-time route token key is not unique")
    decision = pd.to_numeric(result["decision_time"], errors="coerce").to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    length = pd.to_numeric(result["route_part_length_m"], errors="coerce").to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    if not np.isfinite(decision).all():
        raise Stage2V4ContractError("entry-time input has missing decision_time")
    if not np.isfinite(length).all() or np.any(length < 0):
        raise Stage2V4ContractError("entry-time input has invalid route length")

    entry_config = config.section("entry_time")
    pass_count = int(entry_config.get("passes", 0))
    if pass_count != 2:
        raise Stage2V4ContractError("Stage 2 v4 entry-time estimation requires two passes")
    minimum_speed = float(entry_config["minimum_speed_mps"])
    maximum_speed = float(entry_config["maximum_speed_mps"])
    if not (0 < minimum_speed < maximum_speed):
        raise Stage2V4ContractError("invalid entry-time speed clipping bounds")
    minimum_pace = 1.0 / maximum_speed
    maximum_pace = 1.0 / minimum_speed
    history_config = config.section("history")
    minimum_support = int(history_config["minimum_observations"])
    timezone = str(config.section("causality")["timezone"])
    static_pace = _static_pace(result, config)

    estimated_entry = decision.copy()
    final_profile = pd.DataFrame(index=result.index)
    final_travel_time = np.zeros(len(result), dtype=np.float64)
    final_travel_std = np.zeros(len(result), dtype=np.float64)
    for pass_number in range(1, pass_count + 1):
        time_bin, _hour, weekday = _local_time_fields(estimated_entry, timezone)
        queries = pd.DataFrame(
            {
                "decision_time": decision,
                "observed_directed_edge_uid": result[
                    "observed_directed_edge_uid"
                ].astype(str).to_numpy(),
                "canonical_highway": result["canonical_highway"].astype(str).to_numpy(),
                "profile_time_bin": time_bin,
                "profile_weekday_type": weekday,
            },
            index=result.index,
        )
        profile = history.query_fallback(
            queries,
            metrics=("observed_sec_per_m",),
            minimum_observations=minimum_support,
        )
        pace = profile["observed_sec_per_m_profile_mean"].to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        ).copy()
        pace_std = profile["observed_sec_per_m_profile_std"].to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )
        missing = ~np.isfinite(pace)
        pace[missing] = static_pace[missing]
        pace = np.clip(pace, minimum_pace, maximum_pace)
        travel_time = length * pace
        profile_std = length * np.maximum(pace_std, 0.0)
        sparse = profile["observed_sec_per_m_profile_count"].to_numpy(dtype=np.int64) <= 1
        profile_std[sparse & ~missing] = np.maximum(
            profile_std[sparse & ~missing],
            0.25 * travel_time[sparse & ~missing],
        )
        profile_std[missing] = 0.5 * travel_time[missing]
        offset = _exclusive_group_cumsum(result, travel_time)
        estimated_entry = decision + offset
        final_profile = profile
        final_travel_time = travel_time
        final_travel_std = profile_std
        result[f"entry_time_pass_{pass_number}"] = estimated_entry

    final_time_bin, final_hour, final_weekday = _local_time_fields(
        estimated_entry,
        timezone,
    )
    variance_offset = _exclusive_group_cumsum(result, final_travel_std * final_travel_std)
    entry_std = np.sqrt(np.maximum(variance_offset, 0.0))
    fallback = final_profile[
        "observed_sec_per_m_profile_fallback_level"
    ].astype("string")
    fallback = fallback.fillna("static_speed")
    support = final_profile["observed_sec_per_m_profile_count"].fillna(0).astype("int64")
    source = fallback.map(
        {
            "edge_time": "strict_history_directed_edge_time_bin",
            "edge": "strict_history_directed_edge",
            "highway_time": "strict_history_highway_time_bin",
            "highway": "strict_history_highway",
            "global": "strict_history_global",
            "static_speed": "static_speed_fallback",
        }
    ).fillna("static_speed_fallback")

    result["estimated_entry_time"] = estimated_entry
    result["estimated_time_bin"] = final_time_bin
    result["estimated_hour"] = final_hour
    result["estimated_weekday_type"] = final_weekday
    result["forecast_horizon_s"] = estimated_entry - decision
    result["estimated_travel_time_s"] = final_travel_time
    result["estimated_travel_time_source"] = source.to_numpy()
    result["estimated_travel_time_support"] = support.to_numpy()
    result["estimated_entry_std_s"] = entry_std
    result["entry_time_support"] = support.to_numpy()
    result["entry_time_fallback_level"] = fallback.to_numpy()
    result.sort_values("_input_position", kind="stable", inplace=True)
    result.drop(columns="_input_position", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result
