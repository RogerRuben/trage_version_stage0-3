from __future__ import annotations

import pytest

from stage2.v5_2.contracts import Stage2V52ContractError
from stage2.v5_2.evaluation import validate_evaluation_payload


def test_rts_cannot_enter_early_rolling_adoption_targets() -> None:
    payload = {
        "evaluation_dates": ["20161022", "20161023"],
        "rts_role": "secondary_frozen_reference_target",
        "adoption_targets": ["crawl", "stop", "speed_cv", "acceleration_rms", "rts"],
    }
    with pytest.raises(Stage2V52ContractError, match="exclude RTS"):
        validate_evaluation_payload(payload, protocol_id="fold_1")
