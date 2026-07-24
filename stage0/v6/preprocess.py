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
    metrics: dict[str, Any]


REQUIRED_COLUMNS = {"order_id", "timestamp", "lon", "lat"}


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
    positive_dt = frame.time_gap_s.where(frame.time_gap_s > 0)
    frame["step_speed_mps"] = frame.step_distance_m / positive_dt
    frame["time_gap_anomaly"] = frame.time_gap_s.gt(maximum_time_gap_s).fillna(False)
    frame["nonpositive_time_anomaly"] = (
        frame.time_gap_s.le(0) & frame.step_distance_m.gt(0)
    ).fillna(False)
    frame["spatial_jump_anomaly"] = frame.step_speed_mps.gt(maximum_speed_mps).fillna(False)
    split_before = (
        frame.time_gap_anomaly | frame.nonpositive_time_anomaly | frame.spatial_jump_anomaly
    )
    if len(split_before):
        split_before.iloc[0] = False
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
    metrics = {
        "input_point_count": int(len(points)),
        "invalid_point_count": invalid_count,
        "duplicate_point_count": duplicate_count,
        "valid_point_count": int(len(frame)),
        "timestamp_reverse_count": int(frame.timestamp_reverse_in_input.sum()),
        "time_gap_anomaly_count": int(frame.time_gap_anomaly.sum()),
        "nonpositive_time_anomaly_count": int(frame.nonpositive_time_anomaly.sum()),
        "spatial_jump_anomaly_count": int(frame.spatial_jump_anomaly.sum()),
        "subtrace_count": int(frame.subtrace_id.nunique()) if len(frame) else 0,
        "usable_subtrace_count": len(subtraces),
    }
    return PreprocessResult(frame, subtraces, mapping.reset_index(drop=True), metrics)
