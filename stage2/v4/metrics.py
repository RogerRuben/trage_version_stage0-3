"""Metrics and decision-time aggregation for Stage 2 v4."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .contracts import Stage2V4ContractError


def continuous_metrics(
    truth: np.ndarray | pd.Series,
    prediction: np.ndarray | pd.Series,
    mask: np.ndarray | pd.Series | None = None,
) -> dict[str, float | int | None]:
    y = np.asarray(truth, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    valid = np.isfinite(y) & np.isfinite(pred)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    count = int(valid.sum())
    if not count:
        return {
            "count": 0,
            "mae": None,
            "rmse": None,
            "pearson": None,
            "spearman": None,
        }
    error = pred[valid] - y[valid]
    valid_truth = pd.Series(y[valid])
    valid_prediction = pd.Series(pred[valid])
    has_variation = (
        valid_truth.nunique(dropna=True) > 1
        and valid_prediction.nunique(dropna=True) > 1
    )
    if has_variation:
        ranked_truth = valid_truth.rank(method="average")
        ranked_prediction = valid_prediction.rank(method="average")
        spearman = ranked_truth.corr(ranked_prediction)
        pearson = valid_truth.corr(valid_prediction)
    else:
        spearman = None
        pearson = None
    return {
        "count": count,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "pearson": float(pearson) if pearson is not None and pd.notna(pearson) else None,
        "spearman": (
            float(spearman) if spearman is not None and pd.notna(spearman) else None
        ),
    }


def expected_calibration_error(
    truth: np.ndarray,
    probability: np.ndarray,
    *,
    bins: int = 15,
) -> float | None:
    y = np.asarray(truth, dtype=float)
    probability = np.asarray(probability, dtype=float)
    valid = np.isfinite(y) & np.isfinite(probability)
    y = y[valid]
    probability = np.clip(probability[valid], 0.0, 1.0)
    if not len(y):
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(y[selected].mean()) - float(probability[selected].mean())
            )
    return float(ece)


def binary_metrics(
    truth: np.ndarray | pd.Series,
    probability: np.ndarray | pd.Series,
    mask: np.ndarray | pd.Series | None = None,
) -> dict[str, float | int | None]:
    y = np.asarray(truth, dtype=float)
    probability = np.asarray(probability, dtype=float)
    valid = np.isfinite(y) & np.isfinite(probability)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    y = y[valid]
    probability = np.clip(probability[valid], 0.0, 1.0)
    if not len(y):
        empty = {
            "count": 0,
            "positive_count": 0,
            "average_precision": None,
            "roc_auc": None,
            "brier": None,
            "ece": None,
        }
        for percentage in (5, 10):
            empty[f"ndcg_at_{percentage}pct"] = None
            empty[f"precision_at_{percentage}pct"] = None
            empty[f"recall_at_{percentage}pct"] = None
            empty[f"lift_at_{percentage}pct"] = None
        return empty
    unique = np.unique(y)
    result = {
        "count": int(len(y)),
        "positive_count": int(y.sum()),
        "average_precision": (
            float(average_precision_score(y, probability))
            if len(unique) > 1
            else None
        ),
        "roc_auc": (
            float(roc_auc_score(y, probability))
            if len(unique) > 1
            else None
        ),
        "brier": float(brier_score_loss(y, probability)),
        "ece": expected_calibration_error(y, probability),
    }
    order = np.argsort(-probability, kind="stable")
    positive_rate = float(y.mean())
    total_positive = float(y.sum())
    for percentage in (5, 10):
        k = max(1, int(np.ceil(len(y) * percentage / 100.0)))
        selected = y[order[:k]]
        precision = float(selected.mean())
        recall = float(selected.sum() / total_positive) if total_positive > 0 else None
        discounts = 1.0 / np.log2(np.arange(k) + 2.0)
        dcg = float(np.sum(selected * discounts))
        ideal = np.sort(y)[::-1][:k]
        ideal_dcg = float(np.sum(ideal * discounts))
        result[f"ndcg_at_{percentage}pct"] = (
            dcg / ideal_dcg if ideal_dcg > 0 else None
        )
        result[f"precision_at_{percentage}pct"] = precision
        result[f"recall_at_{percentage}pct"] = recall
        result[f"lift_at_{percentage}pct"] = (
            precision / positive_rate if positive_rate > 0 else None
        )
    return result


def order_cluster_bootstrap_ci(
    frame: pd.DataFrame,
    *,
    truth_column: str,
    prediction_column: str,
    replicates: int = 500,
    seed: int = 20261009,
) -> dict[str, Any]:
    required = {"order_id", truth_column, prediction_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise Stage2V4ContractError(f"bootstrap is missing columns: {missing}")
    working = frame.loc[:, list(required)].dropna()
    orders = working["order_id"].astype(str).unique()
    if not len(orders):
        return {"replicates": 0, "mae_ci95": [None, None]}
    working["_absolute_error"] = np.abs(
        working[prediction_column].to_numpy(float)
        - working[truth_column].to_numpy(float)
    )
    grouped = working.groupby("order_id", sort=False, observed=True)[
        "_absolute_error"
    ].agg(["sum", "count"])
    error_sum = grouped["sum"].to_numpy(float)
    error_count = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = rng.integers(0, len(error_sum), size=len(error_sum))
        values[index] = error_sum[sampled].sum() / max(
            error_count[sampled].sum(),
            1.0,
        )
    return {
        "replicates": replicates,
        "seed": seed,
        "mae_ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def decision_weighted_order_aggregation(
    frame: pd.DataFrame,
    *,
    value_column: str,
    dimension: str,
    mask_column: str | None = None,
) -> pd.DataFrame:
    if dimension == "lcs":
        weight_column = "estimated_travel_time_s"
    elif dimension == "rts":
        weight_column = "route_part_length_m"
    else:
        raise Stage2V4ContractError(f"unknown aggregation dimension: {dimension}")
    required = {"split", "date", "order_id", value_column, weight_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise Stage2V4ContractError(f"aggregation is missing columns: {missing}")
    value = pd.to_numeric(frame[value_column], errors="coerce")
    weight = pd.to_numeric(frame[weight_column], errors="coerce")
    valid = value.notna() & weight.notna() & weight.gt(0)
    if mask_column is not None:
        valid &= frame[mask_column].fillna(False).astype(bool)
    working = frame.loc[:, ["split", "date", "order_id"]].copy()
    working["_weighted"] = (value * weight).where(valid, 0.0)
    working["_weight"] = weight.where(valid, 0.0)
    grouped = working.groupby(["split", "date", "order_id"], sort=False, observed=True)
    result = grouped[["_weighted", "_weight"]].sum().reset_index()
    result[value_column] = np.divide(
        result["_weighted"].to_numpy(dtype=float),
        result["_weight"].to_numpy(dtype=float),
        out=np.full(len(result), np.nan),
        where=result["_weight"].to_numpy(dtype=float) > 0,
    )
    result[f"{dimension}_aggregation_weight"] = result["_weight"]
    return result.drop(columns=["_weighted", "_weight"])
