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
    support = {"tau_candidates": [1.0, 2.0, 3.0], "artifact_sha256": "support-hash"}
    baseline = {target: 1.0 for target in CORE_TRANSFER_TARGETS}
    candidates = {tau: dict(baseline) for tau in support["tau_candidates"]}
    metrics = {
        "schema_version": "stage2_v5_2_tau_evaluation.2", "status": "PASS",
        "protocol_id": "transfer_tuning", "protocol_hash": "p" * 64,
        "train_dates": [f"201610{day:02d}" for day in range(9, 19)],
        "validation_dates": ["20161019", "20161020"],
        "support_artifact_embedded_sha256": "support-hash",
        "support_artifact_sha256": "s" * 64, "feature_artifact_sha256": "f" * 64,
        "m1_source_checkpoint_sha256": "a" * 64, "m1_checkpoint_sha256": "b" * 64,
        "m1_evaluation_manifest_sha256": "c" * 64, "evaluation_code_sha256": "d" * 64,
        "evaluation_schema": "fixture", "m1_core_mae": baseline,
        "m4_candidates": {str(tau): {"core_mae": score} for tau, score in candidates.items()},
    }
    result = select_tau_once(metrics, support)
    assert result["validation_dates"] == ["20161019", "20161020"]
    assert result["selected_tau"] == 1.0
    assert result["rts_used"] is False and result["pace_used"] is False
