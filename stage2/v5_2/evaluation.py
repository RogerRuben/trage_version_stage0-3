"""Protocol-bound evaluation and transfer adoption decisions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contracts import CORE_TRANSFER_TARGETS, Stage2V52ContractError
from .micro_metrics import decide_spatial_transfer, decide_temporal_adapter, pace_stability
from .protocols import get_protocol


def validate_evaluation_payload(payload: Mapping[str, Any], *, protocol_id: str) -> None:
    protocol = get_protocol(protocol_id)
    dates = tuple(sorted(str(value) for value in payload.get("evaluation_dates", ())))
    expected = tuple(sorted((*protocol.evaluation_dates, *protocol.legacy_benchmark_dates)))
    if dates != expected:
        raise Stage2V52ContractError(f"evaluation dates {dates} differ from frozen protocol {expected}")
    if payload.get("rts_role") != "secondary_frozen_reference_target":
        raise Stage2V52ContractError("RTS must remain secondary under frozen Stage 1 reference")
    if set(payload.get("adoption_targets", ())) != set(CORE_TRANSFER_TARGETS):
        raise Stage2V52ContractError("adoption target set must exclude RTS and pace")


def evaluate_spatial_adoption(payload: Mapping[str, Any]) -> dict[str, Any]:
    validate_evaluation_payload(payload, protocol_id=str(payload["protocol_id"]))
    return decide_spatial_transfer(
        low_support_improvement_by_target=payload["low_support_improvement_by_target"],
        overall_improvement_by_target=payload["overall_improvement_by_target"],
        unseen_candidate_error=payload["unseen_candidate_error"],
        unseen_structure_only_error=payload["unseen_structure_only_error"],
    )


def evaluate_temporal_adoption(payload: Mapping[str, Any]) -> dict[str, Any]:
    validate_evaluation_payload(payload, protocol_id=str(payload["protocol_id"]))
    return decide_temporal_adapter(
        payload["daily_mean_improvements"], payload["target_mean_improvements"]
    )


def evaluate_pace_guard(payload: Mapping[str, Any]) -> dict[str, Any]:
    return pace_stability(float(payload["candidate_pace_p50_mae"]), float(payload["v5_1_pace_p50_mae"]))
