from __future__ import annotations

from stage2.v5_2.contracts import CORE_TRANSFER_TARGETS
from stage2.v5_2.protocols import PROTOCOLS, validate_protocols
from stage2.v5_2.support_transfer import select_tau_once


def test_tau_tuning_dates_end_before_20161021() -> None:
    validate_protocols()
    protocol = PROTOCOLS["transfer_tuning"]
    assert protocol.train_dates == tuple(f"201610{day:02d}" for day in range(9, 19))
    assert protocol.validation_dates == ("20161019", "20161020")
    assert all(date < "20161021" for date in (*protocol.train_dates, *protocol.validation_dates))


def test_tau_artifact_records_frozen_dates_and_four_target_metric() -> None:
    support = {"tau_candidates": [1.0, 2.0, 3.0]}
    baseline = {target: 1.0 for target in CORE_TRANSFER_TARGETS}
    candidates = {tau: dict(baseline) for tau in support["tau_candidates"]}
    result = select_tau_once(candidates, baseline, support)
    assert result["validation_dates"] == ["20161019", "20161020"]
    assert result["selected_tau"] == 1.0
    assert result["rts_used"] is False and result["pace_used"] is False
