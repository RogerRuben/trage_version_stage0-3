"""Merge overlapping chunk predictions to one row per physical traversal."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import Stage2V5ContractError


PREDICTION_COLUMNS = (
    "pred_crawl_time_share", "pred_stop_time_share", "pred_speed_cv_bounded",
    "pred_acceleration_rms_bounded", "pred_rts_raw", "pred_lcs_raw",
    "lcs_tail_score", "rts_tail_score", "pace_log_mu", "pace_log_scale",
    "pace_pred_mean", "pace_pred_p50", "pace_pred_p90", "pace_pred_p95",
    "history_recent_gate",
    "stop_occurrence_probability", "stop_positive_share",
)
TARGET_COLUMNS = (
    "crawl_time_share", "stop_time_share", "speed_cv_bounded",
    "acceleration_rms_bounded", "rts_raw", "lcs_raw", "pace_sec_per_m",
)
TARGET_MASK_COLUMNS = tuple(f"{name}_target_valid" for name in (
    "crawl", "stop", "speed_cv", "acceleration_rms", "rts", "lcs", "pace",
))


def _weighted(values: np.ndarray, inverse: np.ndarray, weights: np.ndarray, count: int, valid: np.ndarray | None = None) -> np.ndarray:
    usable = np.isfinite(values)
    if valid is not None:
        usable &= valid
    local_weight = np.where(usable, weights, 0.0)
    numerator = np.bincount(inverse, weights=np.where(usable, values * weights, 0.0), minlength=count)
    denominator = np.bincount(inverse, weights=local_weight, minlength=count)
    return np.divide(numerator, denominator, out=np.full(count, np.nan), where=denominator > 0)


def merge_prediction_day(paths: list[Path], *, split: str, date: str) -> pd.DataFrame:
    storage: dict[str, list[np.ndarray]] = {}
    for path in paths:  # Bounded prediction-shard scan; one concat per array.
        with np.load(path, allow_pickle=False) as data:
            valid = ~data["pad_mask"]
            arrays: dict[str, np.ndarray] = {
                "order_id": np.broadcast_to(data["order_id"][:, None], valid.shape)[valid].astype(str),
                "traversal_id": data["traversal_id"][valid].astype(np.int64),
                "route_sequence": data["route_sequence"][valid].astype(np.int64),
                "supervision_weight": data["supervision_weight"][valid].astype(np.float64),
                "allocated_distance_m": data["allocated_distance_m"][valid].astype(np.float64),
            }
            for name in PREDICTION_COLUMNS:
                arrays[name] = data[name][valid].astype(np.float64)
            arrays["availability_probability"] = data["availability_probability"][valid].astype(np.float64)
            for index, name in enumerate(TARGET_COLUMNS):
                arrays[name] = data["targets"][..., index][valid].astype(np.float64)
                arrays[TARGET_MASK_COLUMNS[index]] = data["target_masks"][..., index][valid].astype(bool)
            arrays["lcs_tail_event"] = data["tail_targets"][..., 0][valid].astype(np.float64)
            arrays["rts_tail_event"] = data["tail_targets"][..., 1][valid].astype(np.float64)
            for name, values in arrays.items():
                storage.setdefault(name, []).append(values)
    combined = {name: np.concatenate(parts, axis=0) for name, parts in storage.items()}
    identity = np.rec.fromarrays([combined["order_id"], combined["traversal_id"]], names=("order_id", "traversal_id"))
    unique, inverse = np.unique(identity, return_inverse=True)
    count = len(unique)
    weights = combined["supervision_weight"]
    total_weight = np.bincount(inverse, weights=weights, minlength=count)
    if not np.allclose(total_weight, 1.0, atol=1e-5, rtol=0):
        raise Stage2V5ContractError("overlap prediction weights do not sum to one")
    result = pd.DataFrame({"split": split, "date": date, "order_id": unique["order_id"].astype(str), "traversal_id": unique["traversal_id"].astype(np.int64)})
    result["route_sequence"] = np.rint(_weighted(combined["route_sequence"].astype(float), inverse, weights, count)).astype(np.int64)
    result["allocated_distance_m"] = _weighted(combined["allocated_distance_m"], inverse, weights, count)
    for name in PREDICTION_COLUMNS:
        result[name] = _weighted(combined[name], inverse, weights, count)
    availability = combined["availability_probability"]
    for index, name in enumerate(("service_time", "lcs", "rts", "dynamics")):
        result[f"{name}_availability_probability"] = _weighted(availability[:, index], inverse, weights, count)
    for name, mask_name in zip(TARGET_COLUMNS, TARGET_MASK_COLUMNS):
        mask = combined[mask_name].astype(bool)
        result[name] = _weighted(combined[name], inverse, weights, count, mask)
        result[mask_name] = np.bincount(inverse, weights=mask.astype(float), minlength=count) > 0
    for name, mask_name in (("lcs_tail_event", "lcs_target_valid"), ("rts_tail_event", "rts_target_valid")):
        result[name] = _weighted(combined[name], inverse, weights, count, combined[mask_name].astype(bool))
    result["predicted_traversal_time_mean_s"] = result["pace_pred_mean"] * result["allocated_distance_m"]
    for quantile in ("p50", "p90", "p95"):
        result[f"predicted_traversal_time_{quantile}_s"] = result[f"pace_pred_{quantile}"] * result["allocated_distance_m"]
    result = result.sort_values(["order_id", "route_sequence"], kind="stable", ignore_index=True)
    if np.any(result["pred_crawl_time_share"] + result["pred_stop_time_share"] > 1.0 + 1e-7):
        raise Stage2V5ContractError("merged crawl+stop structural constraint failed")
    if np.any(result["pace_pred_p50"] > result["pace_pred_p90"]) or np.any(result["pace_pred_p90"] > result["pace_pred_p95"]):
        raise Stage2V5ContractError("merged pace quantiles cross")
    return result


def merge_all(prediction_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    source = Path(prediction_root)
    output = Path(output_root)
    days: list[dict[str, Any]] = []
    for date_root in sorted(source.glob("split=*/date=*")):
        split = date_root.parent.name.split("=", 1)[1]
        date = date_root.name.split("=", 1)[1]
        paths = sorted(date_root.glob("shard-*.npz"))
        if not paths:
            continue
        frame = merge_prediction_day(paths, split=split, date=date)
        path = output / f"split={split}" / f"date={date}" / "traversal_predictions.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
        days.append({"split": split, "date": date, "row_count": len(frame), "path": path.as_posix()})
    report = {"schema_version": "stage2_v5_traversal_predictions.1", "status": "PASS", "day_count": len(days), "row_count": int(sum(day["row_count"] for day in days)), "days": days}
    manifest = output / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", default="stage2/output_v5/deep_predictions")
    parser.add_argument("--output-root", default="stage2/output_v5/predictions")
    args = parser.parse_args()
    report = merge_all(args.prediction_root, args.output_root)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
