"""Shared utilities for Stage2 upper-bound modeling experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score


TARGETS = {
    "LCS": ("target_lcs_pct", "lcs_valid", "target_high_lcs_90"),
    "IIS": ("target_iis_pct", "iis_valid", "target_high_iis_90"),
    "RTS": ("target_rts_pct", "rts_valid", "target_high_rts_90"),
    "PMIS": ("target_pmis_pct", "pmis_valid", "target_high_pmis_90"),
}

TARGET_ORDER = ["LCS", "IIS", "RTS", "PMIS"]
ID_COLUMNS = ["order_id", "driver_id", "date", "link_id", "link_seq"]

FEATURE_COLUMNS = [
    "time_bin", "hour", "weekday_type", "peak_offpeak", "is_weekend",
    "road_class", "link_length_m", "curvature_deg_per_km_link", "minor_road",
    "endpoint_degree", "link_fragmentation", "area_grid", "gns_pct_link",
    "activity_intensity_index",
    "poi_density_100m_school", "poi_density_100m_hospital", "poi_density_100m_commercial",
    "poi_density_100m_restaurant", "poi_density_100m_transit", "poi_density_100m_bus_stop",
    "poi_density_100m_residential", "poi_density_100m_office", "poi_density_100m_scenic",
    "poi_density_100m_parking",
    "link_seq", "route_link_count", "position_ratio", "distance_to_destination_ratio",
]

CATEGORICAL_COLUMNS = ["weekday_type", "peak_offpeak", "road_class", "area_grid"]
NUMERIC_COLUMNS = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]

FORBIDDEN_LEAKAGE_COLUMNS = [
    "travel_time_sec", "observed_distance_m", "reference_travel_time_sec", "excess_time_ratio",
    "tail_delay_ratio", "low_speed_ratio_on_poi_link", "stop_time_on_poi_link", "delay_on_poi_link",
    "traversal_quality", "observed_or_inferred", "low_quality_flag",
]


def available_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def unique_existing_columns(path: Path, desired: list[str]) -> list[str]:
    available = set(available_columns(path))
    result = []
    for column in desired:
        if column in available and column not in result:
            result.append(column)
    return result


def parquet_batches(path: Path, columns: list[str], batch_size: int = 250_000):
    parquet = pq.ParquetFile(path)
    selected = unique_existing_columns(path, columns)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=selected):
        yield batch.to_pandas()


def prepare_tabular_features(frame: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    feature_columns = feature_columns or FEATURE_COLUMNS
    features = frame[[column for column in feature_columns if column in frame.columns]].copy()
    for column in CATEGORICAL_COLUMNS:
        if column in features.columns:
            features[column] = features[column].astype("category")
    for column in features.columns:
        if column not in CATEGORICAL_COLUMNS:
            if features[column].dtype == "bool":
                features[column] = features[column].astype("int8")
            else:
                features[column] = pd.to_numeric(features[column], errors="coerce").astype("float32")
    return features


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    if len(y_true) == 0:
        return float("nan")
    y_prob = np.clip(y_prob, 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= low) & (y_prob <= high) if high == 1 else (y_prob >= low) & (y_prob < high)
        if mask.any():
            ece += mask.mean() * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(ece)


def rank_metrics(y: np.ndarray, pred: np.ndarray, max_rows: int | None = None, seed: int = 2026) -> dict[str, float]:
    frame = pd.DataFrame({"y": y, "pred": pred}).dropna()
    if max_rows and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=seed)
    high = frame.y.ge(0.90)
    base_rate = float(high.mean())
    result: dict[str, float] = {"high_rate": base_rate}
    for share, label in [(0.10, "top10"), (0.05, "top5")]:
        n = max(1, int(len(frame) * share))
        top = frame.nlargest(n, "pred")
        precision = float(top.y.ge(0.90).mean())
        recall = float(top.y.ge(0.90).sum() / max(high.sum(), 1))
        result[f"precision_at_{label}pct"] = precision
        result[f"recall_at_{label}pct"] = recall
        result[f"{label}_lift"] = precision / base_rate if base_rate > 0 else float("nan")
    return result


def evaluate_predictions(
    y: np.ndarray,
    pred: np.ndarray,
    high: np.ndarray,
    max_rank_rows: int | None = 1_000_000,
    seed: int = 2026,
) -> dict[str, float]:
    valid = np.isfinite(y) & np.isfinite(pred)
    y = y[valid]
    pred = np.clip(pred[valid], 0, 1)
    high = high[valid].astype(bool)
    metrics = {
        "rows": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred, squared=False)),
        "pearson": float(pd.Series(y).corr(pd.Series(pred), method="pearson")),
        "spearman": float(pd.Series(y).corr(pd.Series(pred), method="spearman")),
        **rank_metrics(y, pred, max_rank_rows, seed),
    }
    if high.any() and (~high).any():
        metrics["auc"] = float(roc_auc_score(high, pred))
        metrics["ap"] = float(average_precision_score(high, pred))
        metrics["brier"] = float(brier_score_loss(high, pred))
        metrics["ece"] = ece_score(high.astype(float), pred)
    else:
        metrics.update({"auc": float("nan"), "ap": float("nan"), "brier": float("nan"), "ece": float("nan")})
    return metrics


def read_split_frame(path: Path, extra_columns: list[str] | None = None) -> pd.DataFrame:
    desired = ID_COLUMNS + FEATURE_COLUMNS
    for target, mask, high in TARGETS.values():
        desired.extend([target, mask, high])
    if extra_columns:
        desired.extend(extra_columns)
    return pd.read_parquet(path, columns=unique_existing_columns(path, desired))


def safe_json_float(value):
    if isinstance(value, (np.floating, float)):
        if np.isfinite(value):
            return float(value)
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {key: safe_json_float(val) for key, val in value.items()}
    if isinstance(value, list):
        return [safe_json_float(item) for item in value]
    return value
