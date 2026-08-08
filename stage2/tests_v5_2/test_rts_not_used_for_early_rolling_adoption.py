from __future__ import annotations

import pytest

from stage2.v5_2.contracts import Stage2V52ContractError
from stage2.v5_2.evaluation import validate_evaluation_payload
from stage2.v5_2.protocols import get_protocol


def test_rts_cannot_enter_early_rolling_adoption_targets() -> None:
    payload = {
        "schema_version": "stage2_v5_2_evaluation.2", "status": "PASS",
        "protocol_id": "fold_1", "protocol_hash": get_protocol("fold_1").digest,
        "role": "evaluation", "model_id": "M4",
        "evaluation_dates": ["20161022", "20161023"],
        "rts_role": "secondary_frozen_reference_target",
        "adoption_targets": ["crawl", "stop", "speed_cv", "acceleration_rms", "rts"],
    }
    for key in (
        "checkpoint_sha256", "training_manifest_sha256", "tensor_manifest_sha256",
        "source_checkpoint_sha256", "feature_artifact_sha256", "prediction_sha256",
        "evaluation_code_sha256", "support_artifact_sha256", "static_artifact_sha256",
    ):
        payload[key] = "a" * 64
    with pytest.raises(Stage2V52ContractError, match="exclude RTS"):
        validate_evaluation_payload(payload, protocol_id="fold_1")
