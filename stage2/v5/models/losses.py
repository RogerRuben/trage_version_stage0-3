"""Weighted, masked multi-task losses for RC-MSTNet v5."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


def weighted_masked_mean(values: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    usable = mask.to(values.dtype) * weight.to(values.dtype)
    return (values * usable).sum() / usable.sum().clamp_min(1.0)


def weighted_masked_huber(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor, *, delta: float = 0.1) -> torch.Tensor:
    return weighted_masked_mean(functional.huber_loss(prediction, target, reduction="none", delta=delta), mask, weight)


def weighted_masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return weighted_masked_mean(functional.binary_cross_entropy_with_logits(logits, target, reduction="none"), mask, weight)


def weighted_lognormal_nll(log_mu: torch.Tensor, log_scale: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    safe = target.clamp_min(torch.finfo(target.dtype).tiny)
    sigma = torch.exp(log_scale)
    loss = torch.log(safe) + log_scale + 0.5 * ((torch.log(safe) - log_mu) / sigma).square() + 0.5 * math.log(2.0 * math.pi)
    return weighted_masked_mean(loss, mask & (target > 0), weight)


def stop_two_part_loss(presence_logits: torch.Tensor, positive_share: torch.Tensor, target_share: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    occurrence = (target_share > 0).to(target_share.dtype)
    presence = weighted_masked_bce(presence_logits, occurrence, mask, weight)
    positive = weighted_masked_huber(positive_share, target_share, mask & (target_share > 0), weight)
    return presence, positive


def rc_mstnet_v5_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    supervision_weight: torch.Tensor,
    *,
    ipw: dict[str, torch.Tensor] | None = None,
    component_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ipw = ipw or {}
    component_weights = component_weights or {}
    weight = supervision_weight
    pace_weight = weight * ipw.get("pace", torch.ones_like(weight))
    stop_presence, stop_positive = stop_two_part_loss(
        outputs["stop_presence_logit"], outputs["stop_positive_share"], targets["stop_time_share"], masks["stop_target_valid"], weight
    )
    components = {
        "pace_distribution": weighted_lognormal_nll(outputs["pace_log_mu"], outputs["pace_log_scale"], targets["pace_sec_per_m"], masks["pace_target_valid"], pace_weight),
        "crawl": weighted_masked_huber(outputs["crawl_share"], targets["crawl_time_share"], masks["crawl_target_valid"], weight),
        "stop_occurrence": stop_presence,
        "stop_positive": stop_positive,
        "speed_cv": weighted_masked_huber(outputs["speed_cv"], targets["speed_cv_bounded"], masks["speed_cv_target_valid"], weight),
        "acceleration": weighted_masked_huber(outputs["acceleration_rms"], targets["acceleration_rms_bounded"], masks["acceleration_rms_target_valid"], weight),
        "rts": weighted_masked_huber(outputs["rts_raw"], targets["rts_raw"], masks["rts_target_valid"], weight),
        "lcs_consistency": weighted_masked_huber(outputs["lcs_reconstructed_raw"], targets["lcs_raw"], masks["lcs_target_valid"], weight),
        "lcs_tail": weighted_masked_bce(outputs["lcs_tail_logit"], targets["lcs_tail_event"], masks["lcs_target_valid"], weight),
        "rts_tail": weighted_masked_bce(outputs["rts_tail_logit"], targets["rts_tail_event"], masks["rts_target_valid"], weight),
        "availability": weighted_masked_bce(outputs["availability_logits"], targets["availability"], masks["availability_valid"], weight.unsqueeze(-1)),
    }
    total = sum(float(component_weights.get(name, 1.0)) * value for name, value in components.items())
    return total, components
