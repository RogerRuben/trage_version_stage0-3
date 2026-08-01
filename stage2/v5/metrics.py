"""Evaluation metrics for stochastic service state and two-part stopping."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _errors(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if not valid.any():
        return {"count": 0, "mae": None, "rmse": None}
    error = prediction[valid] - truth[valid]
    return {"count": int(valid.sum()), "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean()))}


def evaluate_stop_two_part(
    target_share: np.ndarray,
    occurrence_probability: np.ndarray,
    positive_share_prediction: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> dict[str, object]:
    target = np.asarray(target_share, dtype=np.float64)
    probability = np.asarray(occurrence_probability, dtype=np.float64)
    positive_prediction = np.asarray(positive_share_prediction, dtype=np.float64)
    valid = np.isfinite(target) & np.isfinite(probability) & np.isfinite(positive_prediction)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    target = target[valid]
    probability = np.clip(probability[valid], 0.0, 1.0)
    positive_prediction = np.clip(positive_prediction[valid], 0.0, 1.0)
    occurrence = (target > 0).astype(np.int8)
    prevalence = float(occurrence.mean()) if len(occurrence) else None
    has_classes = len(np.unique(occurrence)) == 2
    expected = probability * positive_prediction
    positive = occurrence == 1
    return {
        "count": int(len(target)),
        "stop_occurrence_prevalence": prevalence,
        "stop_occurrence_average_precision": float(average_precision_score(occurrence, probability)) if has_classes else None,
        "stop_occurrence_roc_auc": float(roc_auc_score(occurrence, probability)) if has_classes else None,
        "stop_occurrence_brier": float(brier_score_loss(occurrence, probability)) if len(target) else None,
        "positive_stop_share": _errors(target[positive], positive_prediction[positive]),
        "expected_stop_share": _errors(target, expected),
        "always_zero_baseline": _errors(target, np.zeros_like(target)),
    }
