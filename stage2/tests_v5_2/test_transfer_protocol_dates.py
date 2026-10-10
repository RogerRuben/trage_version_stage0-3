from __future__ import annotations

from stage2.v5_2.contracts import CORE_TRANSFER_TARGETS
from stage2.v5_2.protocols import PROTOCOLS, validate_protocols
from stage2.v5_2.support_transfer import _payload_hash, fit_train_support, select_tau_once


def test_tau_tuning_dates_end_before_20161021() -> None:
    validate_protocols()
    protocol = PROTOCOLS["transfer_tuning"]
    assert protocol.train_dates == tuple(f"201610{day:02d}" for day in range(9, 19))
    assert protocol.validation_dates == ("20161019", "20161020")
    assert all(date < "20161021" for date in (*protocol.train_dates, *protocol.validation_dates))


def test_tau_artifact_records_frozen_dates_and_four_target_metric() -> None:
    support = fit_train_support(
        ["a", "b", "b", "c", "c", "c"], fit_dates=["20161009"]
    ).to_payload()
    baseline = {target: 1.0 for target in CORE_TRANSFER_TARGETS}
    metrics = {
        "schema_version": "stage2_v5_2_tau_evaluation.2", "status": "PASS",
        "protocol_id": "transfer_tuning", "protocol_hash": "p" * 64,
        "train_dates": [f"201610{day:02d}" for day in range(9, 19)],
        "validation_dates": ["20161019", "20161020"],
        "support_artifact_embedded_sha256": support["artifact_sha256"],
        "support_artifact_sha256": "s" * 64, "feature_artifact_sha256": "f" * 64,
        "m1_source_checkpoint_sha256": "a" * 64, "m1_checkpoint_sha256": "b" * 64,
        "m1_evaluation_manifest_sha256": "c" * 64, "evaluation_code_sha256": "d" * 64,
        "evaluation_schema": "fixture", "m1_core_mae": baseline,
        "m4_candidates": {
            label: {
                "support_tau_candidate": label,
                "support_tau_value": support["positive_quantiles"][label],
                "core_mae": dict(baseline),
            }
            for label in ("p25", "p50", "p75")
        },
    }
    metrics["artifact_sha256"] = _payload_hash(metrics)
    result = select_tau_once(metrics, support)
    assert result["validation_dates"] == ["20161019", "20161020"]
    assert result["selected_candidate"] == "p25"
    assert result["rts_used"] is False and result["pace_used"] is False
