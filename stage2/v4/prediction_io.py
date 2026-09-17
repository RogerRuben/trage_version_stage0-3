"""Merge overlapping chunk predictions back to physical traversals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import Stage2V4ContractError


PREDICTION_COLUMNS = (
    "pred_crawl_time_share",
    "pred_stop_time_share",
    "pred_speed_cv_bounded",
    "pred_acceleration_rms_bounded",
    "pred_rts_raw",
    "pred_lcs_raw",
    "lcs_tail_score",
    "rts_tail_score",
    "lcs_log_scale",
    "rts_log_scale",
)
TARGET_COLUMNS = (
    "crawl_time_share",
    "stop_time_share",
    "speed_cv_bounded",
    "acceleration_rms_bounded",
    "rts_raw",
    "lcs_raw",
)
MASK_COLUMNS = (
    "crawl_target_valid",
    "stop_target_valid",
    "speed_cv_target_valid",
    "acceleration_rms_target_valid",
    "rts_target_valid",
    "lcs_target_valid",
)


def _flatten_prediction_shard(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as data:
        pad = data["pad_mask"]
        valid = ~pad
        order = np.broadcast_to(data["order_id"][:, None], pad.shape)
        frame = pd.DataFrame(
            {
                "order_id": order[valid].astype(str),
                "traversal_id": data["traversal_id"][valid].astype(np.int64),
                "route_sequence": data["route_sequence"][valid].astype(np.int64),
                "lcs_weight": data["aggregation_weights"][..., 0][valid],
                "rts_weight": data["aggregation_weights"][..., 1][valid],
            }
        )
        for column in PREDICTION_COLUMNS:
            frame[column] = data[column][valid]
        for index, column in enumerate(TARGET_COLUMNS):
            frame[column] = data["targets"][..., index][valid]
        for index, column in enumerate(MASK_COLUMNS):
            frame[column] = data["target_masks"][..., index][valid]
        frame["lcs_tail_event"] = data["tail_targets"][..., 0][valid]
        frame["rts_tail_event"] = data["tail_targets"][..., 1][valid]
        frame["lcs_tail_valid"] = data["tail_masks"][..., 0][valid]
        frame["rts_tail_valid"] = data["tail_masks"][..., 1][valid]
    return frame


def merge_chunk_predictions(
    prediction_root: str | Path,
    *,
    split: str,
    date: str,
) -> pd.DataFrame:
    root = Path(prediction_root) / f"split={split}" / f"date={date}"
    paths = sorted(root.glob("shard=*.npz"))
    if not paths:
        raise Stage2V4ContractError(f"prediction shards are missing: {root}")
    frames = [_flatten_prediction_shard(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    del frames
    key = ["order_id", "traversal_id"]
    consistency_columns = [
        "route_sequence",
        *TARGET_COLUMNS,
        *MASK_COLUMNS,
        "lcs_tail_event",
        "rts_tail_event",
        "lcs_tail_valid",
        "rts_tail_valid",
        "lcs_weight",
        "rts_weight",
    ]
    grouped = frame.groupby(key, sort=False, observed=True)
    for column in consistency_columns:
        if grouped[column].nunique(dropna=False).gt(1).any():
            raise Stage2V4ContractError(
                f"overlap chunks disagree on immutable field {column}"
            )
    means = grouped[list(PREDICTION_COLUMNS)].mean()
    immutable = grouped[consistency_columns].first()
    counts = grouped.size().rename("overlap_prediction_count")
    result = pd.concat([immutable, means, counts], axis=1).reset_index()
    if result.duplicated(key).any():
        raise Stage2V4ContractError("merged traversal prediction key is duplicated")
    return result
