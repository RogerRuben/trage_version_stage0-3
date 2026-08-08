"""Additive spatial/temporal transfer wrapper around frozen RC-MSTNet v5.1."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from stage2.v5.models.rc_mstnet_v5 import RCMSTNetV5

from ..contracts import Stage2V52ContractError
from ..support_transfer import SupportAwareEdgeRepresentation
from ..temporal_adapter import TemporalAdapter


class RCMSTNetTransfer(nn.Module):
    """RC-MSTNet v5.1 plus a Train-support edge branch and small time adapter.

    In S0/identity mode the frozen backbone receives exactly its legacy inputs.
    In S1-S3 modes the original categorical edge slot is replaced by the fixed
    UNSEEN token and all edge identity information enters through the explicit
    transfer representation. This ensures n=0 is genuinely structure-only.
    """

    def __init__(
        self,
        *,
        numeric_feature_count: int,
        categorical_sizes: tuple[int, ...],
        static_feature_count: int,
        transfer_embedding_dim: int = 24,
        support_tau: float,
        spatial_mode: str = "support_aware",
        temporal_mode: str = "zero_shot",
        temporal_feature_count: int = 4,
        adapter_bottleneck_dim: int = 4,
        legacy_unseen_index: int = 1,
        backbone_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        if not categorical_sizes:
            raise Stage2V52ContractError("edge identity must be the first categorical feature")
        if temporal_mode not in {"none", "zero_shot", "causal_online"}:
            raise Stage2V52ContractError(f"unknown temporal transfer mode: {temporal_mode}")
        self.spatial_mode = spatial_mode
        self.temporal_mode = temporal_mode
        self.legacy_unseen_index = int(legacy_unseen_index)
        self.backbone = RCMSTNetV5(
            numeric_feature_count=numeric_feature_count,
            categorical_sizes=categorical_sizes,
            **(backbone_kwargs or {}),
        )
        self.edge_representation = SupportAwareEdgeRepresentation(
            edge_vocabulary_size=categorical_sizes[0],
            static_feature_count=static_feature_count,
            embedding_dim=transfer_embedding_dim,
            tau=support_tau,
            mode=spatial_mode,
        )
        self.temporal_adapter = TemporalAdapter(
            transfer_embedding_dim, temporal_feature_count, adapter_bottleneck_dim
        )
        self.edge_numeric_projection = nn.Linear(transfer_embedding_dim, numeric_feature_count)
        self.temporal_adapter.assert_budget(self.backbone, maximum_ratio=0.10)

    def forward(
        self,
        numeric: torch.Tensor,
        numeric_missing: torch.Tensor,
        categorical: torch.Tensor,
        route_sequence: torch.Tensor,
        pad_mask: torch.Tensor,
        *,
        edge_index: torch.Tensor,
        static_edge_features: torch.Tensor,
        edge_train_support: torch.Tensor,
        temporal_features: torch.Tensor,
        **backbone_inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.spatial_mode == "identity" and self.temporal_mode == "none":
            return self.backbone(
                numeric, numeric_missing, categorical, route_sequence, pad_mask, **backbone_inputs
            )
        edge_state = self.edge_representation(edge_index, static_edge_features, edge_train_support)
        if self.temporal_mode != "none":
            edge_state = self.temporal_adapter(edge_state, temporal_features)
        transferred_numeric = numeric + self.edge_numeric_projection(edge_state)
        transferred_categorical = categorical.clone()
        transferred_categorical[..., 0] = self.legacy_unseen_index
        output = self.backbone(
            transferred_numeric,
            numeric_missing,
            transferred_categorical,
            route_sequence,
            pad_mask,
            **backbone_inputs,
        )
        support = edge_train_support.to(transferred_numeric.dtype)
        output["edge_transfer_gate"] = support / (support + self.edge_representation.tau)
        return output

    def transfer_parameter_summary(self) -> dict[str, float | int | str]:
        shared = sum(parameter.numel() for parameter in self.backbone.parameters())
        adapter = self.temporal_adapter.parameter_count()
        return {
            "spatial_mode": self.spatial_mode,
            "temporal_mode": self.temporal_mode,
            "shared_parameter_count": shared,
            "temporal_adapter_parameter_count": adapter,
            "temporal_adapter_parameter_ratio": adapter / max(shared, 1),
            "temporal_adapter_budget_status": "PASS" if adapter / max(shared, 1) <= 0.10 else "FAIL",
        }
