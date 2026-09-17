from __future__ import annotations

from stage2.v5_2.contracts import CORE_TRANSFER_TARGETS
from stage2.v5_2.training import checkpoint_candidate, select_micro_first_checkpoint


def _candidate(name: str, core: float, pace: float):
    values = {target: core for target in CORE_TRANSFER_TARGETS}
    return checkpoint_candidate(
        checkpoint_id=name, core_mae=values, low_support_core_mae=values,
        v5_1_core_mae={target: 1.0 for target in CORE_TRANSFER_TARGETS},
        all_outputs_finite=True, temporal_leakage_count=0,
        pace_p50_mae=pace, v5_1_pace_p50_mae=1.0,
        rts_metrics={"mae": 999.0 if name == "micro_best" else 0.0},
    )


def test_checkpoint_selection_is_micro_first_and_rts_is_diagnostic_only() -> None:
    result = select_micro_first_checkpoint([
        _candidate("micro_best", 0.8, 1.01), _candidate("pace_and_rts_best", 0.9, 0.8)
    ])
    assert result["selected_checkpoint_id"] == "micro_best"
    assert result["rts_used"] is False


def test_hard_gate_rejects_pace_degradation_over_two_percent() -> None:
    candidate = _candidate("bad_pace", 0.1, 1.021)
    assert candidate["hard_gate_status"] == "FAIL"
