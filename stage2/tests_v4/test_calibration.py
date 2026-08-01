from __future__ import annotations

import numpy as np

from stage2.v4.calibration import _tail_fit, apply_tail_calibrator


def test_tail_calibration_probabilities_stay_in_unit_interval() -> None:
    score = np.asarray([0.05, 0.2, 0.7, 0.9, 0.8, 0.1])
    truth = np.asarray([0, 0, 1, 1, 1, 0])
    model, audit = _tail_fit(score, truth)
    probability = apply_tail_calibrator(model, score)
    assert np.all(probability >= 0.0)
    assert np.all(probability <= 1.0)
    assert audit["selected_method"] in {"platt", "isotonic"}
