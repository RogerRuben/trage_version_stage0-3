"""Train-only empirical survival timing, not a new safety or demand model.

M3 route time is a sum of traversal medians, not a calibrated joint median.
Normalize observed Train GPS-window duration by that timing proxy. For an
already-booked job still observed busy, condition the empirical ratio on
surviving the elapsed loaded-service time. No Test outcome enters estimation.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class RemainingEstimate:
    remaining_s: float | None
    support_count: int


class TrainRemainingTime:
    def __init__(self, artifact):
        if (artifact.get("method") != "TRAIN_EMPIRICAL_RATIO_SURVIVAL_MEDIAN"
                or artifact.get("test31_used_for_fit") is not False
                or set(artifact.get("train_dates", ())) != {"20161010", "20161017", "20161024"}):
            raise ValueError("remaining-time artifact must be the declared Train-only fit")
        self.ratios = np.asarray(artifact["sorted_duration_prediction_ratios"], dtype=float)
        if (not len(self.ratios) or not np.isfinite(self.ratios).all()
                or np.any(self.ratios <= 0) or np.any(np.diff(self.ratios) < 0)):
            raise ValueError("invalid empirical remaining-time ratio distribution")

    def estimate(self, predicted_service_s, elapsed_service_s):
        if (not isfinite(predicted_service_s) or predicted_service_s <= 0
                or not isfinite(elapsed_service_s) or elapsed_service_s < 0):
            raise ValueError("invalid booked-job timing input")
        first = int(np.searchsorted(self.ratios, elapsed_service_s / predicted_service_s, side="right"))
        support = len(self.ratios) - first
        if not support:
            # No invented 30s finish in an unsupported empirical tail. Native
            # execution continues; only this prospective busy state is omitted.
            return RemainingEstimate(None, 0)
        conditional_ratio = float(np.median(self.ratios[first:]))
        return RemainingEstimate(predicted_service_s * conditional_ratio - elapsed_service_s, support)
