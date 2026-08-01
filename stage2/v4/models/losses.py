"""Masked multi-task losses for RC-MSTNet v4."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    usable = mask.to(values.dtype)
    denominator = usable.sum().clamp_min(1.0)
    return (values * usable).sum() / denominator


def masked_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    delta: float = 0.1,
) -> torch.Tensor:
    loss = functional.huber_loss(
        prediction,
        target,
        reduction="none",
        delta=delta,
    )
    return masked_mean(loss, mask)


def masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    loss = functional.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    return masked_mean(loss, mask)


def stop_two_part_loss(
    presence_logits: torch.Tensor,
    positive_share: torch.Tensor,
    target_share: torch.Tensor,
    mask: torch.Tensor,
    *,
    positive_weight: float = 1.0,
) -> torch.Tensor:
    presence_target = (target_share > 0).to(target_share.dtype)
    presence = masked_bce_with_logits(presence_logits, presence_target, mask)
    positive_mask = mask & (target_share > 0)
    positive = masked_huber(positive_share, target_share, positive_mask)
    return presence + float(positive_weight) * positive


def rc_mstnet_v4_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    *,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or {}
    components = {
        "crawl": masked_huber(
            outputs["crawl_share"],
            targets["crawl_time_share"],
            masks["crawl_target_valid"],
        ),
        "stop": stop_two_part_loss(
            outputs["stop_presence_logit"],
            outputs["stop_positive_share"],
            targets["stop_time_share"],
            masks["stop_target_valid"],
        ),
        "speed_cv": masked_huber(
            outputs["speed_cv"],
            targets["speed_cv_bounded"],
            masks["speed_cv_target_valid"],
        ),
        "acceleration": masked_huber(
            outputs["acceleration_rms"],
            targets["acceleration_rms_bounded"],
            masks["acceleration_rms_target_valid"],
        ),
        "rts": masked_huber(
            outputs["rts_raw"],
            targets["rts_raw"],
            masks["rts_target_valid"],
        ),
        "lcs_consistency": masked_huber(
            outputs["lcs_reconstructed_raw"],
            targets["lcs_raw"],
            masks["lcs_target_valid"],
        ),
        "lcs_tail": masked_bce_with_logits(
            outputs["lcs_tail_logit"],
            targets["lcs_tail_event"],
            masks["lcs_target_valid"],
        ),
        "rts_tail": masked_bce_with_logits(
            outputs["rts_tail_logit"],
            targets["rts_tail_event"],
            masks["rts_target_valid"],
        ),
    }
    total = sum(
        float(weights.get(name, 1.0)) * value for name, value in components.items()
    )
    return total, components
