"""Formal micro-target metrics and frozen transfer adoption rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .contracts import CORE_TRANSFER_TARGETS, Stage2V52ContractError


REGRESSION_TARGET_COLUMNS = {
    "crawl": ("crawl_time_share", "pred_crawl_share"),
    "speed_cv": ("speed_cv_bounded", "pred_speed_cv_bounded"),
    "acceleration_rms": ("acceleration_rms_bounded", "pred_acceleration_rms_bounded"),
    "rts": ("rts_raw", "pred_rts_raw"),
}


def _valid_pair(truth: Sequence[float], prediction: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(truth, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    return actual[valid], predicted[valid]


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def regression_micro_metrics(
    truth: Sequence[float],
    prediction: Sequence[float],
    *,
    train_top_decile_threshold: float,
) -> dict[str, float | int | None]:
    actual, predicted = _valid_pair(truth, prediction)
    if not len(actual):
        return {key: None if key != "count" else 0 for key in (
            "count", "mae", "rmse", "pearson", "spearman", "top_decile_average_precision"
        )}
    error = predicted - actual
    actual_rank = pd.Series(actual).rank(method="average").to_numpy(np.float64)
    predicted_rank = pd.Series(predicted).rank(method="average").to_numpy(np.float64)
    high = actual >= float(train_top_decile_threshold)
    ap = float(average_precision_score(high.astype(np.int8), predicted)) if len(np.unique(high)) == 2 else None
    return {
        "count": int(len(actual)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "pearson": _correlation(actual, predicted),
        "spearman": _correlation(actual_rank, predicted_rank),
        "top_decile_average_precision": ap,
    }


def stop_two_part_metrics(
    target_share: Sequence[float],
    occurrence_probability: Sequence[float],
    positive_share_prediction: Sequence[float],
) -> dict[str, Any]:
    target = np.asarray(target_share, dtype=np.float64)
    probability = np.asarray(occurrence_probability, dtype=np.float64)
    positive_prediction = np.asarray(positive_share_prediction, dtype=np.float64)
    valid = np.isfinite(target) & np.isfinite(probability) & np.isfinite(positive_prediction)
    target = target[valid]
    probability = np.clip(probability[valid], 0.0, 1.0)
    positive_prediction = np.clip(positive_prediction[valid], 0.0, 1.0)
    occurrence = target > 0
    expected = probability * positive_prediction

    def errors(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
        if not len(actual):
            return {"count": 0, "mae": None, "rmse": None}
        delta = predicted - actual
        return {
            "count": int(len(actual)),
            "mae": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        }

    has_two_classes = len(np.unique(occurrence)) == 2
    return {
        "count": int(len(target)),
        "occurrence_average_precision": float(average_precision_score(occurrence, probability)) if has_two_classes else None,
        "occurrence_roc_auc": float(roc_auc_score(occurrence, probability)) if has_two_classes else None,
        "occurrence_brier": float(brier_score_loss(occurrence, probability)) if len(target) else None,
        "positive_stop_share": errors(target[occurrence], positive_prediction[occurrence]),
        "expected_stop_share": errors(target, expected),
    }


def evaluate_micro_by_support(
    frame: pd.DataFrame,
    *,
    train_top_decile_thresholds: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Evaluate overall and frozen Train-defined support groups."""
    if "support_group" not in frame:
        raise Stage2V52ContractError("evaluation frame requires Train-defined support_group")
    groups = ("overall", "high", "medium", "low", "unseen")
    rows: list[dict[str, Any]] = []
    for group in groups:  # Fixed reporting strata, never data-driven regrouping.
        subset = frame if group == "overall" else frame.loc[frame["support_group"].astype(str).eq(group)]
        for target, (truth_column, prediction_column) in REGRESSION_TARGET_COLUMNS.items():
            if truth_column not in subset or prediction_column not in subset:
                raise Stage2V52ContractError(f"evaluation frame is missing {target} columns")
            metrics = regression_micro_metrics(
                subset[truth_column], subset[prediction_column],
                train_top_decile_threshold=float(train_top_decile_thresholds[target]),
            )
            rows.append({"support_group": group, "target": target, **metrics})
        stop_required = (
            "stop_time_share", "stop_occurrence_probability", "stop_positive_share"
        )
        if any(column not in subset for column in stop_required):
            raise Stage2V52ContractError("evaluation frame is missing two-part stop columns")
        rows.append({
            "support_group": group,
            "target": "stop",
            **stop_two_part_metrics(
                subset["stop_time_share"],
                subset["stop_occurrence_probability"],
                subset["stop_positive_share"],
            ),
        })
    return rows


def target_day_elapsed_slice(decision_time_s: Sequence[float], day_start_s: float) -> np.ndarray:
    """Return the frozen 0-2h, 2-6h, and 6h+ temporal reporting slices."""
    elapsed = (np.asarray(decision_time_s, dtype=np.float64) - float(day_start_s)) / 3600.0
    result = np.full(len(elapsed), "6h_plus", dtype=object)
    result[elapsed < 6.0] = "2_to_6h"
    result[elapsed < 2.0] = "0_to_2h"
    result[~np.isfinite(elapsed) | (elapsed < 0)] = "invalid"
    return result.astype(str)


def relative_error_improvement(baseline: float, candidate: float) -> float:
    if not np.isfinite(baseline) or baseline <= 0 or not np.isfinite(candidate):
        raise Stage2V52ContractError("relative improvement requires finite positive baseline error")
    return float((baseline - candidate) / baseline)


def decide_spatial_transfer(
    *,
    low_support_improvement_by_target: Mapping[str, float],
    overall_improvement_by_target: Mapping[str, float],
    unseen_candidate_error: Mapping[str, float],
    unseen_structure_only_error: Mapping[str, float],
) -> dict[str, Any]:
    if set(low_support_improvement_by_target) != set(CORE_TRANSFER_TARGETS):
        raise Stage2V52ContractError("spatial adoption requires exactly four core micro targets")
    if set(overall_improvement_by_target) != set(CORE_TRANSFER_TARGETS):
        raise Stage2V52ContractError("overall spatial adoption inputs differ from four core targets")
    wins = sum(float(value) > 0 for value in low_support_improvement_by_target.values())
    mean_improvement = float(np.mean(list(low_support_improvement_by_target.values())))
    overall_stable = all(float(value) >= -0.02 for value in overall_improvement_by_target.values())
    unseen_not_worse = all(
        float(unseen_candidate_error[target]) <= float(unseen_structure_only_error[target])
        for target in CORE_TRANSFER_TARGETS
    )
    adopted = wins >= 3 and mean_improvement > 0.02 and overall_stable and unseen_not_worse
    return {
        "adopt": adopted,
        "status": "ADOPT_SUPPORT_AWARE" if adopted else "RETAIN_V5_1_NEGATIVE_OR_INSUFFICIENT_TRANSFER",
        "low_support_target_wins": wins,
        "adoption_target_count": 4,
        "rts_role": "secondary_frozen_reference_target",
        "low_support_mean_relative_improvement": mean_improvement,
        "overall_no_target_degrades_over_2pct": overall_stable,
        "unseen_not_worse_than_structure_only": unseen_not_worse,
    }


def decide_temporal_adapter(
    daily_mean_improvements: Mapping[str, float],
    target_mean_improvements: Mapping[str, float],
) -> dict[str, Any]:
    improved_days = sum(float(value) > 0 for value in daily_mean_improvements.values())
    if set(target_mean_improvements) != set(CORE_TRANSFER_TARGETS):
        raise Stage2V52ContractError("temporal adoption requires exactly four core micro targets")
    target_mean = float(np.mean(list(target_mean_improvements.values())))
    adopted = len(daily_mean_improvements) == 6 and improved_days >= 4 and target_mean > 0.01
    return {
        "adopt": adopted,
        "status": "ADOPT_TEMPORAL_ADAPTER" if adopted else "RETAIN_NO_OR_ZERO_SHOT_ADAPTER",
        "improved_rolling_dates": improved_days,
        "rolling_date_count": len(daily_mean_improvements),
        "four_core_target_mean_relative_improvement": target_mean,
        "rts_role": "secondary_frozen_reference_target",
    }


def pace_stability(candidate_mae: float, v5_1_mae: float) -> dict[str, float | bool | str]:
    degradation = (float(candidate_mae) - float(v5_1_mae)) / float(v5_1_mae)
    passed = degradation <= 0.02
    return {
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "relative_mae_degradation": degradation,
        "maximum_allowed_degradation": 0.02,
    }
