"""Conservative trajectory cleaning and subtrace segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .coordinates import gcj02_to_wgs84, haversine_m


@dataclass(frozen=True)
class PreprocessResult:
    points: pd.DataFrame
    subtraces: list[pd.DataFrame]
    mapping: pd.DataFrame
    preprocess_breaks: pd.DataFrame
    metrics: dict[str, Any]


REQUIRED_COLUMNS = {"order_id", "timestamp", "lon", "lat"}
BREAK_COLUMNS = [
    "order_id",
    "from_original_point_seq",
    "to_original_point_seq",
    "from_timestamp",
    "to_timestamp",
    "time_gap_s",
    "distance_gap_m",
    "break_reason",
]


def preprocess_order(
    points: pd.DataFrame,
    *,
    coordinate_system: str = "gcj02",
    maximum_time_gap_s: float = 300.0,
    maximum_speed_mps: float = 75.0,
    minimum_subtrace_points: int = 3,
) -> PreprocessResult:
    """Clean one order without discarding different coordinates at equal times."""

    missing = REQUIRED_COLUMNS - set(points.columns)
    if missing:
        raise ValueError(f"points missing required columns: {sorted(missing)}")
    if points.empty:
        return PreprocessResult(
            pd.DataFrame(),
            [],
            pd.DataFrame(columns=["order_id", "subtrace_id", "original_point_seq", "usable"]),
            pd.DataFrame(columns=BREAK_COLUMNS),
            {"input_point_count": 0, "valid_point_count": 0, "subtrace_count": 0},
        )

    frame = points.copy().reset_index(drop=True)
    if frame.order_id.astype(str).nunique() != 1:
        raise ValueError("preprocess_order accepts exactly one order_id")
    if "point_seq" in frame:
        frame["original_point_seq"] = pd.to_numeric(frame["point_seq"], errors="coerce")
        fallback = pd.Series(np.arange(len(frame)), index=frame.index)
        frame["original_point_seq"] = frame["original_point_seq"].fillna(fallback).astype("int64")
    else:
        frame["original_point_seq"] = np.arange(len(frame), dtype="int64")

    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
    frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
    frame["timestamp_reverse_in_input"] = frame.timestamp.diff().lt(0).fillna(False)
    valid = (
        frame.timestamp.notna()
        & frame.lon.between(-180, 180, inclusive="both")
        & frame.lat.between(-90, 90, inclusive="both")
    )
    invalid_count = int((~valid).sum())
    frame = frame.loc[valid].copy()
    frame.sort_values(["timestamp", "original_point_seq"], kind="stable", inplace=True)
    duplicate = frame.duplicated(["timestamp", "lon", "lat"], keep="first")
    duplicate_count = int(duplicate.sum())
    frame = frame.loc[~duplicate].copy().reset_index(drop=True)

    if coordinate_system.lower() == "gcj02":
        wgs_lon, wgs_lat = gcj02_to_wgs84(frame.lon.to_numpy(), frame.lat.to_numpy())
    elif coordinate_system.lower() == "wgs84":
        wgs_lon, wgs_lat = frame.lon.to_numpy(float), frame.lat.to_numpy(float)
    else:
        raise ValueError(f"unsupported coordinate system: {coordinate_system}")
    frame["matching_lon"] = wgs_lon
    frame["matching_lat"] = wgs_lat

    frame["time_gap_s"] = frame.timestamp.diff()
    distances = np.zeros(len(frame), dtype=float)
    if len(frame) > 1:
        distances[1:] = haversine_m(
            wgs_lon[:-1], wgs_lat[:-1], wgs_lon[1:], wgs_lat[1:]
        )
    frame["step_distance_m"] = distances
    next_distances = np.zeros(len(frame), dtype=float)
    if len(frame) > 1:
        next_distances[:-1] = distances[1:]
    frame["next_step_distance_m"] = next_distances
    bridge_distances = np.full(len(frame), np.nan, dtype=float)
    if len(frame) > 2:
        bridge_distances[1:-1] = haversine_m(
            wgs_lon[:-2], wgs_lat[:-2], wgs_lon[2:], wgs_lat[2:]
        )
    frame["neighbor_bridge_distance_m"] = bridge_distances
    positive_dt = frame.time_gap_s.where(frame.time_gap_s > 0)
    frame["step_speed_mps"] = frame.step_distance_m / positive_dt
    frame["time_gap_anomaly"] = frame.time_gap_s.gt(maximum_time_gap_s).fillna(False)
    frame["nonpositive_time_anomaly"] = (
        frame.time_gap_s.le(0) & frame.step_distance_m.gt(0)
    ).fillna(False)
    frame["spatial_jump_anomaly"] = frame.step_speed_mps.gt(maximum_speed_mps).fillna(False)
    isolated_out_and_back = (
        frame.step_distance_m.ge(15.0)
        & frame.next_step_distance_m.ge(15.0)
        & frame.neighbor_bridge_distance_m.le(
            np.maximum(
                5.0,
                0.30
                * (
                    frame.step_distance_m
                    + frame.next_step_distance_m
                ),
            )
        )
    ).fillna(False)
    repeated_time_move = (
        frame.time_gap_s.le(0) & frame.step_distance_m.gt(20.0)
    ).fillna(False)
    local_speed_baseline = (
        frame.step_speed_mps.rolling(5, center=True, min_periods=3).median()
    )
    previous_speed = frame.step_speed_mps.shift()
    next_speed = frame.step_speed_mps.shift(-1)
    local_speed_spike = (
        frame.step_speed_mps.ge(20.0)
        & frame.step_speed_mps.ge(1.60 * local_speed_baseline)
        & (
            (frame.step_speed_mps - previous_speed).abs().ge(7.0)
            | (frame.step_speed_mps - next_speed).abs().ge(7.0)
        )
    ).fillna(False)
    frame["gps_outlier"] = (
        isolated_out_and_back | repeated_time_move | local_speed_spike
    )
    outlier_reason = pd.Series(pd.NA, index=frame.index, dtype="object")
    outlier_reason.loc[isolated_out_and_back] = "isolated_out_and_back"
    outlier_reason.loc[repeated_time_move] = "repeated_time_position_change"
    outlier_reason.loc[local_speed_spike] = "local_speed_acceleration_spike"
    frame["gps_outlier_reason"] = outlier_reason
    outlier_boundary = frame.gps_outlier | frame.gps_outlier.shift(
        fill_value=False
    )
    split_before = (
        frame.time_gap_anomaly
        | frame.nonpositive_time_anomaly
        | frame.spatial_jump_anomaly
        | outlier_boundary
    )
    if len(split_before):
        split_before.iloc[0] = False
    break_reason = pd.Series(pd.NA, index=frame.index, dtype="object")
    break_reason.loc[frame.time_gap_anomaly] = "preprocess_time_gap"
    break_reason.loc[frame.spatial_jump_anomaly] = "preprocess_spatial_jump"
    break_reason.loc[frame.nonpositive_time_anomaly] = "preprocess_nonpositive_time"
    break_reason.loc[outlier_boundary] = "preprocess_gps_outlier"
    frame["preprocess_break_before"] = split_before
    frame["preprocess_break_reason"] = break_reason
    frame["subtrace_number"] = split_before.astype("int64").cumsum()
    order_id = str(frame.order_id.iloc[0]) if len(frame) else str(points.order_id.iloc[0])
    frame["subtrace_id"] = frame.subtrace_number.map(lambda value: f"{order_id}:{int(value):03d}")

    sizes = frame.groupby("subtrace_id", sort=False).size()
    frame["usable_subtrace"] = frame.subtrace_id.map(sizes).ge(minimum_subtrace_points)
    subtraces = [
        group.copy().reset_index(drop=True)
        for _, group in frame.loc[frame.usable_subtrace].groupby("subtrace_id", sort=False)
    ]
    mapping = frame[
        ["order_id", "subtrace_id", "original_point_seq", "usable_subtrace"]
    ].rename(columns={"usable_subtrace": "usable"})
    break_rows: list[dict[str, Any]] = []
    for index in frame.index[frame.preprocess_break_before]:
        if index <= 0:
            continue
        left = frame.iloc[index - 1]
        right = frame.iloc[index]
        break_rows.append(
            {
                "order_id": order_id,
                "from_original_point_seq": int(left.original_point_seq),
                "to_original_point_seq": int(right.original_point_seq),
                "from_timestamp": float(left.timestamp),
                "to_timestamp": float(right.timestamp),
                "time_gap_s": float(right.timestamp - left.timestamp),
                "distance_gap_m": float(right.step_distance_m),
                "break_reason": str(right.preprocess_break_reason),
            }
        )
    preprocess_breaks = pd.DataFrame(break_rows, columns=BREAK_COLUMNS)
    metrics = {
        "input_point_count": int(len(points)),
        "invalid_point_count": invalid_count,
        "duplicate_point_count": duplicate_count,
        "valid_point_count": int(len(frame)),
        "timestamp_reverse_count": int(frame.timestamp_reverse_in_input.sum()),
        "time_gap_anomaly_count": int(frame.time_gap_anomaly.sum()),
        "nonpositive_time_anomaly_count": int(frame.nonpositive_time_anomaly.sum()),
        "spatial_jump_anomaly_count": int(frame.spatial_jump_anomaly.sum()),
        "gps_outlier_count": int(frame.gps_outlier.sum()),
        "subtrace_count": int(frame.subtrace_id.nunique()) if len(frame) else 0,
        "usable_subtrace_count": len(subtraces),
        "preprocess_break_count": int(len(preprocess_breaks)),
        "raw_order_gps_distance_m": float(frame.step_distance_m.sum()),
        "resolved_subtrace_gps_distance_m": float(
            frame.loc[~frame.preprocess_break_before, "step_distance_m"].sum()
        ),
    }
    return PreprocessResult(
        frame,
        subtraces,
        mapping.reset_index(drop=True),
        preprocess_breaks,
        metrics,
    )
