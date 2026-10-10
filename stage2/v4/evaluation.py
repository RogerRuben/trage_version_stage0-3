"""Frozen Test execution and Stage 2 v4 evaluation."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .aggregation import aggregate_route_dimension
from .calibration import apply_tail_calibrator
from .cdf_adapter import apply_frozen_stage1_cdf
from .config import Stage2V4Config
from .contracts import ROUTE_PRIMARY_KEY, Stage2V4ContractError
from .io import atomic_write_json, atomic_write_parquet, sha256_file
from .metrics import (
    binary_metrics,
    continuous_metrics,
    decision_weighted_order_aggregation,
    order_cluster_bootstrap_ci,
)
from .prediction_io import merge_chunk_predictions


EVALUATION_SCHEMA_VERSION = "stage2_v4_evaluation.1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage2V4ContractError(f"manifest is not an object: {path}")
    return value


def _run_test_prediction(
    config_path: Path,
    config: Stage2V4Config,
    *,
    tensor_root: Path,
    model_root: Path,
    prediction_root: Path,
    test_date: str,
) -> None:
    python = Path(str(config.section("deep")["python_executable"]))
    result = subprocess.run(
        [
            str(python),
            "-m",
            "stage2.v4.models.predict_worker",
            "--config",
            str(config_path),
            "--tensor-root",
            str(tensor_root),
            "--model-root",
            str(model_root),
            "--prediction-root",
            str(prediction_root),
            "--test-date",
            test_date,
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise Stage2V4ContractError(
            "frozen Test prediction worker failed:\n"
            + result.stdout[-2000:]
            + result.stderr[-2000:]
        )


def _interval_metrics(
    truth: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    mask: pd.Series,
) -> dict[str, float | int | None]:
    valid = (
        mask.fillna(False).astype(bool)
        & truth.notna()
        & lower.notna()
        & upper.notna()
    )
    if not valid.any():
        return {"count": 0, "coverage": None, "mean_width": None}
    y = truth.loc[valid].to_numpy(float)
    lo = lower.loc[valid].to_numpy(float)
    hi = upper.loc[valid].to_numpy(float)
    return {
        "count": int(valid.sum()),
        "coverage": float(((y >= lo) & (y <= hi)).mean()),
        "mean_width": float((hi - lo).mean()),
    }


def _slice_report(frame: pd.DataFrame) -> dict[str, Any]:
    definitions = {
        "canonical_highway": frame["canonical_highway"].astype(str),
        "estimated_hour": frame["estimated_hour"].astype(str),
        "synthetic_reverse_edge": frame["synthetic_reverse_edge"].astype(str),
        "osm_direction_disagreement": frame["osm_direction_disagreement"].astype(str),
        "directed_edge_model_scope": frame["directed_edge_model_scope"].astype(str),
        "history_available": frame["dynamic_available_mask"].astype(str),
        "route_position_bucket": pd.cut(
            frame["route_position_ratio"],
            [-np.inf, 0.2, 0.8, np.inf],
            labels=["pickup", "middle", "dropoff"],
        ).astype(str),
        "forecast_horizon_bucket": pd.cut(
            frame["forecast_horizon_s"],
            [-np.inf, 300, 900, 1800, np.inf],
            labels=["0_5m", "5_15m", "15_30m", "30m_plus"],
        ).astype(str),
    }
    report: dict[str, Any] = {}
    for name, values in definitions.items():
        groups: dict[str, Any] = {}
        for value in sorted(values.dropna().unique()):
            selected = values.eq(value)
            if int(selected.sum()) < 100:
                continue
            groups[str(value)] = {
                "row_count": int(selected.sum()),
                "lcs": continuous_metrics(
                    frame.loc[selected, "lcs_raw"],
                    frame.loc[selected, "pred_lcs_raw"],
                    frame.loc[selected, "lcs_target_valid"],
                ),
                "rts": continuous_metrics(
                    frame.loc[selected, "rts_raw"],
                    frame.loc[selected, "pred_rts_raw"],
                    frame.loc[selected, "rts_target_valid"],
                ),
                "lcs_tail": binary_metrics(
                    frame.loc[selected, "lcs_tail_event"].astype("Float64"),
                    frame.loc[selected, "lcs_tail_probability_calibrated"],
                    frame.loc[selected, "lcs_tail_valid"],
                ),
            }
        report[name] = groups
    return report


def evaluate_test(
    config_path: str | Path,
    test_date: str,
    output_root: str | Path,
    config: Stage2V4Config,
    *,
    tensor_root: str | Path = "stage2/output_v4/tensor_shards",
    model_root: str | Path = "stage2/output_v4/models/rc_mstnet_v4",
    calibration_root: str | Path = "stage2/output_v4/models/calibration",
    prediction_root: str | Path = "stage2/output_v4/predictions/uncalibrated",
    dataset_root: str | Path = "stage2/output_v4/route_conditioned_dataset",
    resume: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    if config.section("split")["test_dates"] != [test_date]:
        raise Stage2V4ContractError("evaluation may read only frozen Test date 20161031")
    output = Path(output_root)
    report_path = output / "evaluation_report.json"
    prediction_path = output / "traversal_predictions.parquet"
    if report_path.is_file() and prediction_path.is_file() and resume:
        report = _read_json(report_path)
        if (
            report.get("engineering_status") == "PASS"
            and report.get("stage2_config_sha256") == config.digest
            and report.get("prediction_file_sha256") == sha256_file(prediction_path)
        ):
            return report
    if (report_path.exists() or prediction_path.exists()) and not force:
        raise Stage2V4ContractError("evaluation output exists; use --resume or --force")
    model_manifest = _read_json(Path(model_root) / "model_manifest.json")
    calibration_manifest = _read_json(
        Path(calibration_root) / "calibration_manifest.json"
    )
    if (
        model_manifest.get("stage2_config_sha256") != config.digest
        or calibration_manifest.get("stage2_config_sha256") != config.digest
        or calibration_manifest.get("deep_model_id") != model_manifest.get("model_id")
    ):
        raise Stage2V4ContractError("model/calibration release binding mismatch")
    _run_test_prediction(
        Path(config_path),
        config,
        tensor_root=Path(tensor_root),
        model_root=Path(model_root),
        prediction_root=Path(prediction_root),
        test_date=test_date,
    )
    predictions = merge_chunk_predictions(
        prediction_root,
        split="test",
        date=test_date,
    )
    dataset_path = (
        Path(dataset_root) / "revealed_route_proxy" / f"day={test_date}.parquet"
    )
    metadata_columns = [
        "split",
        "date",
        "order_id",
        "traversal_id",
        "route_sequence",
        "canonical_edge_uid",
        "observed_directed_edge_uid",
        "canonical_highway",
        "observed_direction",
        "synthetic_reverse_edge",
        "osm_direction_disagreement",
        "directed_edge_model_scope",
        "estimated_entry_time",
        "estimated_time_bin",
        "estimated_hour",
        "estimated_weekday_type",
        "forecast_horizon_s",
        "route_position_ratio",
        "route_part_length_m",
        "estimated_travel_time_s",
        "dynamic_available_mask",
        "lcs_pct",
        "rts_pct",
        "lcs_tail_event",
        "rts_tail_event",
        "lcs_target_valid",
        "rts_target_valid",
    ]
    metadata = pd.read_parquet(dataset_path, columns=metadata_columns)
    if metadata.duplicated(["order_id", "traversal_id"]).any():
        raise Stage2V4ContractError("Test dataset traversal key is duplicated")
    frame = predictions.merge(
        metadata,
        on=["order_id", "traversal_id", "route_sequence"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_dataset"),
    )
    if frame["date"].isna().any() or len(frame) != len(metadata):
        raise Stage2V4ContractError("Test prediction/dataset reconciliation failed")
    calibration = joblib.load(Path(calibration_root) / "calibration_bundle.joblib")
    for dimension in ("lcs", "rts"):
        frame[f"{dimension}_tail_probability_calibrated"] = apply_tail_calibrator(
            calibration["tail_calibrators"][dimension],
            frame[f"{dimension}_tail_score"].to_numpy(float),
        )
        radius = float(
            calibration["conformal_absolute_residual_q90"][dimension]
        )
        frame[f"pred_{dimension}_lower_90"] = np.clip(
            frame[f"pred_{dimension}_raw"] - radius,
            0.0,
            1.0,
        )
        frame[f"pred_{dimension}_upper_90"] = np.clip(
            frame[f"pred_{dimension}_raw"] + radius,
            0.0,
            1.0,
        )
    frame = apply_frozen_stage1_cdf(frame, config)

    continuous = {}
    for target, prediction, mask in (
        ("crawl_time_share", "pred_crawl_time_share", "crawl_target_valid"),
        ("stop_time_share", "pred_stop_time_share", "stop_target_valid"),
        ("speed_cv_bounded", "pred_speed_cv_bounded", "speed_cv_target_valid"),
        (
            "acceleration_rms_bounded",
            "pred_acceleration_rms_bounded",
            "acceleration_rms_target_valid",
        ),
        ("lcs_raw", "pred_lcs_raw", "lcs_target_valid"),
        ("rts_raw", "pred_rts_raw", "rts_target_valid"),
        ("lcs_pct", "pred_lcs_pct", "lcs_target_valid"),
        ("rts_pct", "pred_rts_pct", "rts_target_valid"),
    ):
        continuous[target] = continuous_metrics(
            frame[target],
            frame[prediction],
            frame[mask],
        )
    tails = {
        dimension: {
            "uncalibrated": binary_metrics(
                frame[f"{dimension}_tail_event"].astype("Float64"),
                frame[f"{dimension}_tail_score"],
                frame[f"{dimension}_tail_valid"],
            ),
            "calibrated": binary_metrics(
                frame[f"{dimension}_tail_event"].astype("Float64"),
                frame[f"{dimension}_tail_probability_calibrated"],
                frame[f"{dimension}_tail_valid"],
            ),
        }
        for dimension in ("lcs", "rts")
    }
    intervals = {
        dimension: _interval_metrics(
            frame[f"{dimension}_raw"],
            frame[f"pred_{dimension}_lower_90"],
            frame[f"pred_{dimension}_upper_90"],
            frame[f"{dimension}_target_valid"],
        )
        for dimension in ("lcs", "rts")
    }
    order_frames = {}
    for dimension in ("lcs", "rts"):
        prediction_order = decision_weighted_order_aggregation(
            frame,
            value_column=f"pred_{dimension}_pct",
            dimension=dimension,
        ).rename(
            columns={f"pred_{dimension}_pct": f"pred_{dimension}_order_pct"}
        )
        truth_order = decision_weighted_order_aggregation(
            frame,
            value_column=f"{dimension}_pct",
            dimension=dimension,
            mask_column=f"{dimension}_target_valid",
        ).rename(columns={f"{dimension}_pct": f"true_{dimension}_order_pct"})
        order_frames[dimension] = prediction_order.merge(
            truth_order,
            on=["split", "date", "order_id"],
            validate="one_to_one",
        )
    order_metrics = {
        dimension: continuous_metrics(
            values[f"true_{dimension}_order_pct"],
            values[f"pred_{dimension}_order_pct"],
        )
        for dimension, values in order_frames.items()
    }
    bootstrap = {
        dimension: order_cluster_bootstrap_ci(
            values,
            truth_column=f"true_{dimension}_order_pct",
            prediction_column=f"pred_{dimension}_order_pct",
            replicates=500,
            seed=int(config.section("runtime")["random_seed"]),
        )
        for dimension, values in order_frames.items()
    }
    route_summary = aggregate_route_dimension(
        frame,
        dimension="lcs",
        percentile_column="pred_lcs_pct",
        tail_probability_column="lcs_tail_probability_calibrated",
    ).merge(
        aggregate_route_dimension(
            frame,
            dimension="rts",
            percentile_column="pred_rts_pct",
            tail_probability_column="rts_tail_probability_calibrated",
        ),
        on=["split", "date", "order_id"],
        validate="one_to_one",
    )
    atomic_write_parquet(frame, prediction_path)
    atomic_write_parquet(route_summary, output / "order_route_predictions.parquet")
    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "engineering_status": "PASS",
        "stage2_config_sha256": config.digest,
        "test_dates": [test_date],
        "test_tuning_violation_count": 0,
        "deep_model_id": model_manifest["model_id"],
        "calibration_model_id": calibration_manifest["calibration_model_id"],
        "stage1_cdf_model_id": config.section("stage1_release")["model_id"],
        "traversal_row_count": int(len(frame)),
        "order_count": int(frame["order_id"].nunique()),
        "continuous_metrics": continuous,
        "tail_metrics": tails,
        "interval_metrics": intervals,
        "order_metrics": order_metrics,
        "order_cluster_bootstrap": bootstrap,
        "slice_metrics": _slice_report(frame),
        "prediction_file_sha256": sha256_file(prediction_path),
        "runtime_s": time.perf_counter() - started,
    }
    atomic_write_json(report_path, report)
    return report
