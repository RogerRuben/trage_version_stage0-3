"""Daily, column-projected v5 dataset adapter over frozen v4 features."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stage2.v4.models.baselines import _feature_candidates

from .availability import service_time_target_arrays
from .contracts import Stage2V5ContractError


IDENTITY_COLUMNS = (
    "split", "date", "order_id", "route_sequence", "traversal_id",
    "observed_directed_edge_uid", "canonical_highway", "estimated_time_bin",
    "route_part_length_m", "estimated_travel_time_s", "forecast_horizon_s",
)
AUXILIARY_TARGET_COLUMNS = (
    "crawl_time_share", "stop_time_share", "speed_cv_bounded",
    "acceleration_rms_bounded", "lcs_raw", "lcs_pct", "lcs_tail_event",
    "rts_raw", "rts_pct", "rts_tail_event", "crawl_target_valid",
    "stop_target_valid", "speed_cv_target_valid",
    "acceleration_rms_target_valid", "lcs_target_valid", "rts_target_valid",
)
RECENT_PACE_COLUMNS = tuple(
    f"edge_{minutes}m_observed_sec_per_m_mean" for minutes in (5, 15, 30, 60)
)
PROFILE_PACE_COLUMNS = (
    "observed_sec_per_m_profile_mean",
    "observed_sec_per_m_profile_std",
    "observed_sec_per_m_profile_count",
)


def _input_split(split: str) -> str:
    return "validation" if split in {"validation_model", "calibration"} else split


def _traversal_targets(repo_root: Path, split: str, date: str) -> pd.DataFrame:
    day = repo_root / "stage1/input_v1" / f"split={_input_split(split)}" / f"date={date}"
    paths = sorted(day.glob("bucket=*/link_traversals.parquet"))
    if not paths:
        raise Stage2V5ContractError(f"missing Stage 1 traversal partitions for {date}")
    frames: list[pd.DataFrame] = []
    columns = [
        "order_id", "traversal_id", "measurement_source", "observed_travel_time_s",
        "observed_distance_m", "allocated_distance_m",
    ]
    for path in paths:  # Bounded daily partition streaming; one concat after scan.
        frame = pd.read_parquet(path, columns=columns)
        relative = path.relative_to(repo_root / "stage1/input_v1")
        label_path = repo_root / "stage1/output_v3" / relative.parent / "traversal_labels.parquet"
        labels = pd.read_parquet(
            label_path,
            columns=["order_id", "traversal_id", "observed_sec_per_m", "rts_measurement_available"],
        )
        target = service_time_target_arrays(
            frame["measurement_source"].to_numpy(),
            frame["observed_travel_time_s"].to_numpy(),
            frame["observed_distance_m"].to_numpy(),
        )
        frame = frame.merge(
            labels,
            on=["order_id", "traversal_id"],
            how="left",
            validate="one_to_one",
        )
        pace = pd.to_numeric(frame["observed_sec_per_m"], errors="coerce").to_numpy(float)
        physical_quality = frame["rts_measurement_available"].fillna(False).to_numpy(bool)
        frame["pace_sec_per_m"] = pace
        frame["pace_target_valid"] = target["travel_time_direct_valid"] & physical_quality & np.isfinite(pace) & (pace > 0)
        frame["travel_time_target_valid"] = target["travel_time_target_valid"]
        frame["travel_time_direct_valid"] = target["travel_time_direct_valid"]
        frame["travel_time_interpolated_valid"] = target["travel_time_interpolated_valid"]
        frame["travel_time_source_class"] = target["travel_time_source_class"]
        frames.append(frame)
    target_frame = pd.concat(frames, ignore_index=True)
    if target_frame.duplicated(["order_id", "traversal_id"]).any():
        raise Stage2V5ContractError("Stage 1 traversal target identity is not unique")
    return target_frame


def load_v5_day(
    date: str,
    *,
    split: str,
    repo_root: str | Path = ".",
    extra_columns: Iterable[str] = (),
) -> pd.DataFrame:
    root = Path(repo_root).resolve()
    route_path = root / "stage2/output_v4/route_conditioned_dataset/revealed_route_proxy" / f"day={date}.parquet"
    if not route_path.is_file():
        raise Stage2V5ContractError(f"missing frozen v4 route features for {date}")
    schema = set(pq.read_schema(route_path).names)
    requested = tuple(
        dict.fromkeys(
            [
                *IDENTITY_COLUMNS,
                *AUXILIARY_TARGET_COLUMNS,
                *_feature_candidates(),
                *RECENT_PACE_COLUMNS,
                *PROFILE_PACE_COLUMNS,
                *extra_columns,
            ]
        )
    )
    columns = [column for column in requested if column in schema]
    route = pd.read_parquet(route_path, columns=columns)
    targets = _traversal_targets(root, split, date)
    result = route.merge(
        targets,
        on=["order_id", "traversal_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_stage1"),
    )
    if len(result) != len(route):
        raise Stage2V5ContractError("v5 daily adapter changed route row count")
    result["pace_target_valid"] = result["pace_target_valid"].fillna(False).astype(bool)
    result.loc[~result["pace_target_valid"], "pace_sec_per_m"] = np.nan
    return result
