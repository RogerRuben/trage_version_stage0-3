"""Causal profile, constant, road-class, and tree baselines."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from ..config import Stage2V4Config
from ..contracts import FORBIDDEN_FORMAL_TARGETS, Stage2V4ContractError
from ..io import (
    atomic_write_json,
    atomic_write_parquet,
    sha256_file,
    stage2_v4_code_identity,
)
from ..metrics import binary_metrics, continuous_metrics


BASELINE_SCHEMA_VERSION = "stage2_v4_baselines.1"
CONTINUOUS_TARGETS = {
    "crawl_time_share": "crawl_target_valid",
    "stop_time_share": "stop_target_valid",
    "speed_cv_bounded": "speed_cv_target_valid",
    "acceleration_rms_bounded": "acceleration_rms_target_valid",
    "lcs_raw": "lcs_target_valid",
    "rts_raw": "rts_target_valid",
}
BINARY_TARGETS = {
    "lcs_tail_event": "lcs_target_valid",
    "rts_tail_event": "rts_target_valid",
}
STATIC_FEATURES = (
    "route_part_length_m",
    "canonical_length_m",
    "bridge",
    "tunnel",
    "synthetic_reverse_edge",
    "osm_direction_disagreement",
    "route_position_ratio",
    "distance_to_destination_ratio",
    "route_token_count",
    "estimated_time_bin",
    "estimated_travel_time_s",
    "forecast_horizon_s",
    "estimated_entry_std_s",
    "entry_time_support",
)


def _feature_candidates() -> tuple[str, ...]:
    columns = list(STATIC_FEATURES)
    source_metrics = (
        "crawl_time_share",
        "stop_time_share",
        "speed_cv_bounded",
        "acceleration_rms_bounded",
        "lcs_raw",
        "rts_raw",
        "observed_sec_per_m",
    )
    for metric in source_metrics:
        columns.extend(
            [
                f"{metric}_profile_mean",
                f"{metric}_profile_count",
                f"edge_15m_{metric}_mean",
                f"edge_60m_{metric}_mean",
                f"highway_15m_{metric}_mean",
                f"global_15m_{metric}_mean",
            ]
        )
    columns.extend(
        [
            "edge_5m_completed_traversal_count",
            "edge_15m_completed_traversal_count",
            "edge_30m_completed_traversal_count",
            "edge_60m_completed_traversal_count",
            "feature_age_s",
            "history_count",
            "dynamic_available_mask",
        ]
    )
    return tuple(dict.fromkeys(columns))


def _dataset_files(dataset_root: Path, dates: tuple[str, ...]) -> list[Path]:
    files = [dataset_root / f"day={date}.parquet" for date in dates]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise Stage2V4ContractError(f"baseline dataset days are missing: {missing}")
    return files


def _stable_cap(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    hash_values = pd.util.hash_pandas_object(
        frame[["order_id", "traversal_id"]],
        index=False,
        hash_key=f"{seed:016d}"[-16:],
    ).to_numpy(dtype=np.uint64)
    selected = np.argpartition(hash_values, limit - 1)[:limit]
    return frame.iloc[selected].copy()


def _encode_features(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    highway_vocabulary: dict[str, int],
) -> np.ndarray:
    numeric = frame.loc[:, feature_columns].copy()
    for column in feature_columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    highway = (
        frame["canonical_highway"]
        .astype(str)
        .map(highway_vocabulary)
        .fillna(-1)
        .to_numpy(dtype=np.float32)
    )
    return np.column_stack(
        [
            numeric.to_numpy(dtype=np.float32, na_value=np.nan),
            highway,
        ]
    )


def _fit_highway_vocabulary(files: list[Path]) -> dict[str, int]:
    values: set[str] = set()
    for path in files:
        frame = pd.read_parquet(path, columns=["canonical_highway"])
        values.update(frame["canonical_highway"].dropna().astype(str).unique())
    return {value: index for index, value in enumerate(sorted(values))}


def _fit_target_model(
    files: list[Path],
    *,
    target: str,
    mask: str,
    feature_columns: tuple[str, ...],
    highway_vocabulary: dict[str, int],
    config: Stage2V4Config,
    binary: bool,
) -> Any:
    baseline = config.section("baseline")
    maximum = int(baseline["tree_max_train_rows_per_target"])
    per_day = int(math.ceil(maximum / len(files)))
    seed = int(config.section("runtime")["random_seed"])
    frames: list[pd.DataFrame] = []
    columns = [
        "order_id",
        "traversal_id",
        "canonical_highway",
        target,
        mask,
        *feature_columns,
    ]
    for day_index, path in enumerate(files):
        frame = pd.read_parquet(path, columns=list(dict.fromkeys(columns)))
        usable = frame[mask].fillna(False).astype(bool)
        values = pd.to_numeric(frame[target], errors="coerce")
        frame = frame.loc[usable & values.notna()].copy()
        frame = _stable_cap(frame, per_day, seed + day_index)
        frames.append(frame)
    training = pd.concat(frames, ignore_index=True)
    training = _stable_cap(training, maximum, seed)
    if training.empty:
        raise Stage2V4ContractError(f"no Train labels for baseline target {target}")
    x = _encode_features(training, feature_columns, highway_vocabulary)
    y = pd.to_numeric(training[target], errors="raise").to_numpy(dtype=np.float64)
    common = {
        "learning_rate": float(baseline["tree_learning_rate"]),
        "max_iter": int(baseline["tree_max_iter"]),
        "max_leaf_nodes": int(baseline["tree_max_leaf_nodes"]),
        "min_samples_leaf": int(baseline["tree_min_samples_leaf"]),
        "random_state": seed,
        "early_stopping": True,
        "validation_fraction": 0.1,
    }
    model = (
        HistGradientBoostingClassifier(loss="log_loss", **common)
        if binary
        else HistGradientBoostingRegressor(loss="squared_error", **common)
    )
    model.fit(x, y.astype(int) if binary else y)
    return model


def _fit_simple_statistics(
    files: list[Path],
) -> tuple[dict[str, float], dict[str, dict[tuple[str, int], float]]]:
    totals = defaultdict(float)
    counts = defaultdict(int)
    group_totals: dict[str, defaultdict[tuple[str, int], float]] = {
        target: defaultdict(float) for target in CONTINUOUS_TARGETS
    }
    group_counts: dict[str, defaultdict[tuple[str, int], int]] = {
        target: defaultdict(int) for target in CONTINUOUS_TARGETS
    }
    columns = [
        "canonical_highway",
        "estimated_time_bin",
        *CONTINUOUS_TARGETS,
        *CONTINUOUS_TARGETS.values(),
    ]
    for path in files:
        frame = pd.read_parquet(path, columns=list(dict.fromkeys(columns)))
        for target, mask in CONTINUOUS_TARGETS.items():
            valid = frame[mask].fillna(False).astype(bool)
            values = pd.to_numeric(frame[target], errors="coerce")
            valid &= values.notna()
            totals[target] += float(values.loc[valid].sum())
            counts[target] += int(valid.sum())
            working = pd.DataFrame(
                {
                    "highway": frame.loc[valid, "canonical_highway"].astype(str),
                    "time_bin": pd.to_numeric(
                        frame.loc[valid, "estimated_time_bin"], errors="coerce"
                    ).astype(int),
                    "value": values.loc[valid],
                }
            )
            grouped = working.groupby(["highway", "time_bin"], observed=True)["value"].agg(
                ["sum", "count"]
            )
            for key, row in grouped.iterrows():
                group_totals[target][key] += float(row["sum"])
                group_counts[target][key] += int(row["count"])
    global_means = {
        target: totals[target] / counts[target] for target in CONTINUOUS_TARGETS
    }
    group_means = {
        target: {
            key: total / group_counts[target][key]
            for key, total in group_totals[target].items()
        }
        for target in CONTINUOUS_TARGETS
    }
    return global_means, group_means


def _tree_predict(model: Any, x: np.ndarray, *, binary: bool) -> np.ndarray:
    return (
        model.predict_proba(x)[:, 1]
        if binary
        else np.clip(model.predict(x), 0.0, 1.0)
    )


def train_baselines(
    dataset_root: str | Path,
    output_root: str | Path,
    config: Stage2V4Config,
    *,
    resume: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    dataset = Path(dataset_root)
    output = Path(output_root)
    model_path = output / "baseline_bundle.joblib"
    manifest_path = output / "baseline_manifest.json"
    if (model_path.exists() or manifest_path.exists()) and not force:
        if resume and model_path.is_file() and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("engineering_status") == "PASS"
                and manifest.get("stage2_config_sha256") == config.digest
                and manifest.get("model_file_sha256") == sha256_file(model_path)
            ):
                return manifest
        raise Stage2V4ContractError("baseline output exists; use --resume or --force")

    split = config.section("split")
    train_files = _dataset_files(dataset, tuple(split["train_dates"]))
    validation_files = _dataset_files(
        dataset,
        tuple(split["validation_model_dates"]),
    )
    schema_columns = set(pq.read_schema(train_files[0]).names)
    if schema_columns & FORBIDDEN_FORMAL_TARGETS:
        raise Stage2V4ContractError("baseline input exposes forbidden legacy targets")
    feature_columns = tuple(
        column for column in _feature_candidates() if column in schema_columns
    )
    highway_vocabulary = _fit_highway_vocabulary(train_files)
    global_means, highway_time_means = _fit_simple_statistics(train_files)
    models: dict[str, Any] = {}
    for target, mask in CONTINUOUS_TARGETS.items():
        models[target] = _fit_target_model(
            train_files,
            target=target,
            mask=mask,
            feature_columns=feature_columns,
            highway_vocabulary=highway_vocabulary,
            config=config,
            binary=False,
        )
    for target, mask in BINARY_TARGETS.items():
        models[target] = _fit_target_model(
            train_files,
            target=target,
            mask=mask,
            feature_columns=feature_columns,
            highway_vocabulary=highway_vocabulary,
            config=config,
            binary=True,
        )
    bundle = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "stage2_config_sha256": config.digest,
        "feature_columns": feature_columns,
        "highway_vocabulary": highway_vocabulary,
        "global_means": global_means,
        "highway_time_means": highway_time_means,
        "models": models,
    }
    output.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".baseline_bundle.tmp-",
        suffix=".joblib",
        dir=output,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(bundle, temporary, compress=3)
        os.replace(temporary, model_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    metric_inputs: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for path in validation_files:
        required = {
            "split",
            "date",
            "order_id",
            "traversal_id",
            "canonical_highway",
            "estimated_time_bin",
            *feature_columns,
            *CONTINUOUS_TARGETS,
            *CONTINUOUS_TARGETS.values(),
            *BINARY_TARGETS,
            *BINARY_TARGETS.values(),
        }
        frame = pd.read_parquet(path, columns=sorted(required))
        x = _encode_features(frame, feature_columns, highway_vocabulary)
        prediction = frame.loc[
            :,
            ["split", "date", "order_id", "traversal_id"],
        ].copy()
        for target, mask in CONTINUOUS_TARGETS.items():
            profile = pd.to_numeric(
                frame.get(f"{target}_profile_mean"),
                errors="coerce",
            ).to_numpy(dtype=float, na_value=np.nan)
            edge = pd.to_numeric(
                frame.get(f"edge_60m_{target}_mean"),
                errors="coerce",
            ).to_numpy(dtype=float, na_value=np.nan)
            global_prediction = np.full(len(frame), global_means[target])
            highway_prediction = np.asarray(
                [
                    highway_time_means[target].get(
                        (str(highway), int(time_bin)),
                        global_means[target],
                    )
                    for highway, time_bin in zip(
                        frame["canonical_highway"],
                        frame["estimated_time_bin"],
                    )
                ],
                dtype=float,
            )
            tree = _tree_predict(models[target], x, binary=False)
            for name, values in (
                ("profile", profile),
                ("global_mean", global_prediction),
                ("highway_time_mean", highway_prediction),
                ("edge_rolling_mean", edge),
                ("tree", tree),
            ):
                prediction[f"pred_{target}_{name}"] = values
                metric_inputs[(target, f"{name}:truth")].append(
                    pd.to_numeric(frame[target], errors="coerce").to_numpy(float)
                )
                metric_inputs[(target, f"{name}:prediction")].append(values)
                metric_inputs[(target, f"{name}:mask")].append(
                    frame[mask].fillna(False).to_numpy(bool)
                )
        for target, mask in BINARY_TARGETS.items():
            tree = _tree_predict(models[target], x, binary=True)
            prediction[f"pred_{target}_tree"] = tree
            metric_inputs[(target, "tree:truth")].append(
                pd.to_numeric(frame[target], errors="coerce").to_numpy(float)
            )
            metric_inputs[(target, "tree:prediction")].append(tree)
            metric_inputs[(target, "tree:mask")].append(
                frame[mask].fillna(False).to_numpy(bool)
            )
        atomic_write_parquet(
            prediction,
            output / "validation_predictions" / path.name,
        )

    metrics: dict[str, dict[str, Any]] = {}
    for target in CONTINUOUS_TARGETS:
        metrics[target] = {}
        for name in (
            "profile",
            "global_mean",
            "highway_time_mean",
            "edge_rolling_mean",
            "tree",
        ):
            metrics[target][name] = continuous_metrics(
                np.concatenate(metric_inputs[(target, f"{name}:truth")]),
                np.concatenate(metric_inputs[(target, f"{name}:prediction")]),
                np.concatenate(metric_inputs[(target, f"{name}:mask")]),
            )
    for target in BINARY_TARGETS:
        metrics[target] = {
            "tree": binary_metrics(
                np.concatenate(metric_inputs[(target, "tree:truth")]),
                np.concatenate(metric_inputs[(target, "tree:prediction")]),
                np.concatenate(metric_inputs[(target, "tree:mask")]),
            )
        }

    manifest = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "engineering_status": "PASS",
        "stage2_config_sha256": config.digest,
        "stage2_code_sha": stage2_v4_code_identity(
            (
                "stage2/v4/models/baselines.py",
                "stage2/v4/metrics.py",
                "stage2/v4/contracts.py",
            )
        ),
        "fit_dates": list(split["train_dates"]),
        "validation_dates": list(split["validation_model_dates"]),
        "test_rows_read": 0,
        "feature_columns": list(feature_columns),
        "forbidden_feature_count": 0,
        "models": sorted(models),
        "validation_metrics": metrics,
        "model_file_sha256": sha256_file(model_path),
        "runtime_s": time.perf_counter() - started,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest
