"""Categorical-edge and hidden-state transfer over frozen RC-MSTNet v5.1."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from stage2.v5.models.rc_mstnet_v5 import RCMSTNetV5, sinusoidal_position_encoding

from ..contracts import Stage2V52ContractError
from ..feature_binding import V51FeatureSchemaBinding, V51SourceModelBinding, validate_binding_against_model
from ..support_transfer import SupportAwareEdgeRepresentation
from ..temporal_adapter import TemporalAdapter, stack_temporal_features


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RCMSTNetTransfer(nn.Module):
    """S0-S3 spatial transfer plus a hidden-token temporal adapter.

    S0 delegates directly to the frozen v5.1 model, so loading the same
    checkpoint yields numerical identity. S1-S3 replace only the bound edge
    categorical embedding before the legacy categorical encoder. The temporal
    adapter is inserted after input/history fusion and before local-route and
    Transformer layers.
    """

    def __init__(
        self,
        *,
        numeric_feature_count: int,
        binding: V51FeatureSchemaBinding,
        static_feature_count: int,
        support_tau: float,
        spatial_mode: str = "support_aware",
        temporal_mode: str = "none",
        adapter_bottleneck_dim: int = 4,
        backbone_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        if temporal_mode not in {"none", "zero_shot", "causal_online"}:
            raise Stage2V52ContractError(f"unknown temporal transfer mode: {temporal_mode}")
        if spatial_mode == "identity" and temporal_mode != "none":
            raise Stage2V52ContractError("S0 is reserved for exact frozen v5.1 equivalence")
        options = dict(backbone_kwargs or {})
        embedding_dim = int(options.get("categorical_embedding_dim", 24))
        hidden_dim = int(options.get("hidden_dim", 128))
        self.binding = binding
        self.spatial_mode = spatial_mode
        self.temporal_mode = temporal_mode
        self.backbone = RCMSTNetV5(
            numeric_feature_count=numeric_feature_count,
            categorical_sizes=binding.categorical_sizes,
            **options,
        )
        validate_binding_against_model(binding, self.backbone)
        self.edge_representation = SupportAwareEdgeRepresentation(
            edge_vocabulary_size=binding.categorical_sizes[binding.edge_column_index],
            static_feature_count=static_feature_count,
            embedding_dim=embedding_dim,
            tau=support_tau,
            mode=spatial_mode,
            padding_idx=binding.pad_index,
        )
        self.temporal_adapter = TemporalAdapter(
            hidden_dim=hidden_dim,
            time_feature_dim=4,
            bottleneck_dim=adapter_bottleneck_dim,
        )
        self.temporal_adapter.assert_budget(self.backbone, maximum_ratio=0.10)
        self.source_provenance: dict[str, Any] | None = None

    def initialize_from_v51(
        self,
        checkpoint_path: str | Path,
        *,
        source_binding: V51SourceModelBinding,
    ) -> dict[str, Any]:
        checkpoint_file = Path(checkpoint_path)
        if _sha256(checkpoint_file) != source_binding.source_checkpoint_sha256:
            raise Stage2V52ContractError("v5.1 checkpoint differs from protocol source binding")
        if self.binding.feature_artifact_sha256 != source_binding.feature_artifact_sha256:
            raise Stage2V52ContractError("feature binding differs from protocol source binding")
        saved = torch.load(checkpoint_file, map_location="cpu")
        state = saved.get("model_state_dict", saved)
        self.backbone.load_state_dict(state, strict=True)
        validate_binding_against_model(self.binding, self.backbone)
        legacy_edge = self.backbone.embeddings[self.binding.edge_column_index]
        if legacy_edge.weight.shape != self.edge_representation.id_embedding.weight.shape:
            raise Stage2V52ContractError("transfer ID table shape differs from v5.1 edge embedding")
        with torch.no_grad():
            self.edge_representation.id_embedding.weight.copy_(legacy_edge.weight)
        self.source_provenance = {
            **source_binding.to_payload(),
            "feature_schema_hash": self.binding.feature_artifact_sha256,
            "edge_vocab_hash": self.binding.edge_vocabulary_sha256,
            "initialization_policy": "shared_backbone_and_edge_id_from_frozen_v5_1; structure_and_adapter_fresh",
        }
        return dict(self.source_provenance)

    def initialize_spatial_from_m4(self, checkpoint_path: str | Path) -> dict[str, Any]:
        """Load M4 spatial/shared weights while keeping this M5 adapter fresh."""
        checkpoint_file = Path(checkpoint_path)
        saved = torch.load(checkpoint_file, map_location="cpu")
        state = saved.get("model_state_dict", saved)
        spatial_state = {
            name: value for name, value in state.items()
            if not name.startswith("temporal_adapter.")
        }
        result = self.load_state_dict(spatial_state, strict=False)
        missing = set(result.missing_keys)
        expected_missing = {
            name for name in self.state_dict() if name.startswith("temporal_adapter.")
        }
        if missing != expected_missing or result.unexpected_keys:
            raise Stage2V52ContractError(
                "selected M4 checkpoint is incompatible with the M5 spatial/shared model"
            )
        return {
            "m4_checkpoint_path": checkpoint_file.as_posix(),
            "m4_checkpoint_sha256": _sha256(checkpoint_file),
            "m4_loaded_components": "shared_backbone_support_branch_id_branch",
            "temporal_adapter_initialization": "fresh_after_m4_load",
        }

    def set_shared_backbone_frozen(self, frozen: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen
        for parameter in self.edge_representation.id_embedding.parameters():
            parameter.requires_grad = not frozen

    def optimizer_parameter_groups(
        self,
        *,
        new_branch_lr: float,
        backbone_lr_ratio: float = 0.1,
    ) -> list[dict[str, Any]]:
        if self.source_provenance is None:
            raise Stage2V52ContractError("v5.1 checkpoint must be loaded before optimizer construction")
        if new_branch_lr <= 0 or backbone_lr_ratio <= 0:
            raise Stage2V52ContractError("learning rates must be positive")
        copied_id = list(self.edge_representation.id_embedding.parameters())
        copied_ids = {id(parameter) for parameter in copied_id}
        new_parameters = [
            parameter
            for name, parameter in self.edge_representation.named_parameters()
            if id(parameter) not in copied_ids
        ]
        if self.temporal_mode != "none":
            new_parameters.extend(self.temporal_adapter.parameters())
        return [
            {
                "name": "v5_1_shared_backbone",
                "params": [parameter for parameter in self.backbone.parameters() if parameter.requires_grad],
                "lr": new_branch_lr * backbone_lr_ratio,
            },
            {
                "name": "v5_1_copied_edge_id",
                "params": [parameter for parameter in copied_id if parameter.requires_grad],
                "lr": new_branch_lr * backbone_lr_ratio,
            },
            {
                "name": "new_transfer_branches",
                "params": [parameter for parameter in new_parameters if parameter.requires_grad],
                "lr": new_branch_lr,
            },
        ]

    def _decode(self, h: torch.Tensor, valid: torch.Tensor, gate: torch.Tensor) -> dict[str, torch.Tensor]:
        backbone = self.backbone
        stop_presence_logit = backbone.stop_presence_head(h).squeeze(-1).float()
        stop_positive = torch.sigmoid(backbone.stop_positive_head(h).squeeze(-1).float())
        stop_share = torch.sigmoid(stop_presence_logit) * stop_positive
        crawl = (1.0 - stop_share) * torch.sigmoid(backbone.crawl_head(h).squeeze(-1).float())
        speed_cv = torch.sigmoid(backbone.speed_cv_head(h).squeeze(-1))
        acceleration = torch.sigmoid(backbone.acceleration_head(h).squeeze(-1))
        rts = torch.sigmoid(backbone.rts_head(h).squeeze(-1))
        lcs = (crawl + stop_share + speed_cv + acceleration) / 4.0
        distribution = backbone.pace_distribution_head(h).float()
        z90, z95 = 1.2815515655446004, 1.6448536269514722
        if backbone.distribution_family == "lognormal":
            pace_log_mu = distribution[..., 0]
            pace_log_scale = torch.clamp(
                distribution[..., 1], backbone.minimum_log_scale, backbone.maximum_log_scale
            )
            sigma = torch.exp(pace_log_scale)
            pace_p50 = torch.exp(pace_log_mu)
            pace_p90 = torch.exp(pace_log_mu + z90 * sigma)
            pace_p95 = torch.exp(pace_log_mu + z95 * sigma)
            pace_mean = torch.exp(pace_log_mu + 0.5 * sigma.square())
        else:
            minimum_log_pace = math.log(1.0e-3)
            log_p50 = torch.clamp(distribution[..., 0], minimum_log_pace, backbone.maximum_log_p50)
            log_p90_p50_ratio = torch.clamp(
                torch.nn.functional.softplus(distribution[..., 1]) * 0.1,
                min=1.0e-5,
                max=backbone.maximum_log_p90_p50_ratio,
            )
            log_p95_p90_ratio = torch.clamp(
                torch.nn.functional.softplus(distribution[..., 2]) * 0.05,
                min=1.0e-5,
                max=backbone.maximum_log_p95_p90_ratio,
            )
            pace_log_mu = log_p50
            pace_p50 = torch.exp(log_p50)
            pace_p90 = pace_p50 * torch.exp(log_p90_p50_ratio)
            pace_p95 = pace_p90 * torch.exp(log_p95_p90_ratio)
            effective_sigma = log_p90_p50_ratio / z90
            pace_log_scale = torch.log(effective_sigma.clamp_min(math.exp(backbone.minimum_log_scale)))
            pace_mean = pace_p50
        result = {
            "crawl_share": crawl,
            "stop_presence_logit": stop_presence_logit,
            "stop_positive_share": stop_positive,
            "stop_share": stop_share,
            "speed_cv": speed_cv,
            "acceleration_rms": acceleration,
            "rts_raw": rts,
            "lcs_reconstructed_raw": lcs,
            "lcs_tail_logit": backbone.lcs_tail_head(h).squeeze(-1),
            "rts_tail_logit": backbone.rts_tail_head(h).squeeze(-1),
            "pace_log_mu": pace_log_mu,
            "pace_log_scale": pace_log_scale,
            "pace_pred_mean": pace_mean,
            "pace_pred_p50": pace_p50,
            "pace_pred_p90": pace_p90,
            "pace_pred_p95": pace_p95,
            "availability_logits": backbone.availability_head(h),
            "history_recent_gate": gate,
        }
        if backbone.distribution_family == "monotonic_quantiles":
            result["pace_quantile_family"] = torch.ones((), device=h.device)
        return result

    def forward(
        self,
        numeric: torch.Tensor,
        numeric_missing: torch.Tensor,
        categorical: torch.Tensor,
        route_sequence: torch.Tensor,
        pad_mask: torch.Tensor,
        *,
        static_edge_features: torch.Tensor | None = None,
        edge_train_support: torch.Tensor | None = None,
        temporal_features: Mapping[str, torch.Tensor] | None = None,
        recent_history: torch.Tensor | None = None,
        profile_history: torch.Tensor | None = None,
        forecast_horizon_s: torch.Tensor | None = None,
        history_age_s: torch.Tensor | None = None,
        history_support: torch.Tensor | None = None,
        use_recent: bool = True,
        use_profile: bool = True,
    ) -> dict[str, torch.Tensor]:
        common = {
            "recent_history": recent_history,
            "profile_history": profile_history,
            "forecast_horizon_s": forecast_horizon_s,
            "history_age_s": history_age_s,
            "history_support": history_support,
            "use_recent": use_recent,
            "use_profile": use_profile,
        }
        if self.spatial_mode == "identity":
            return self.backbone(numeric, numeric_missing, categorical, route_sequence, pad_mask, **common)
        if static_edge_features is None or edge_train_support is None:
            raise Stage2V52ContractError("S1-S3 require static features and Train-only support")
        edge_slot = self.binding.edge_column_index
        edge_index = categorical[..., edge_slot]
        edge_state = self.edge_representation(edge_index, static_edge_features, edge_train_support)
        backbone = self.backbone
        numeric_input = torch.cat((numeric, numeric_missing.to(numeric.dtype)), dim=-1)
        numeric_state = backbone.numeric_encoder(numeric_input)
        category_states = [
            edge_state if index == edge_slot else embedding(categorical[..., index])
            for index, embedding in enumerate(backbone.embeddings)
        ]
        categorical_state = backbone.categorical_encoder(torch.cat(category_states, dim=-1))
        h = backbone.input_fusion(torch.cat((numeric_state, categorical_state), dim=-1))
        batch_shape, device, dtype = h.shape[:-1], h.device, h.dtype
        recent_history = torch.zeros((*batch_shape, 4), device=device, dtype=dtype) if recent_history is None else recent_history
        profile_history = torch.zeros((*batch_shape, 3), device=device, dtype=dtype) if profile_history is None else profile_history
        profile_history = torch.cat((
            profile_history[..., :2], torch.log1p(profile_history[..., 2:].clamp_min(0.0))
        ), dim=-1)
        forecast_horizon_s = torch.zeros(batch_shape, device=device, dtype=dtype) if forecast_horizon_s is None else forecast_horizon_s
        history_age_s = torch.zeros(batch_shape, device=device, dtype=dtype) if history_age_s is None else history_age_s
        history_support = torch.zeros(batch_shape, device=device, dtype=dtype) if history_support is None else history_support
        recent_state = backbone.recent_encoder(recent_history)
        profile_state = backbone.profile_encoder(profile_history)
        history, gate = backbone.horizon_gate(
            recent_state, profile_state, forecast_horizon_s, history_age_s, history_support,
            use_recent=use_recent and backbone.history_mode != "without_recent",
            use_profile=use_profile and backbone.history_mode != "without_profile",
        )
        if backbone.history_mode == "ordinary_concatenation":
            history = backbone.ordinary_history_fusion(torch.cat((recent_state, profile_state), dim=-1))
            gate = torch.full_like(gate, 0.5)
        h = backbone.history_fusion(torch.cat((h, history), dim=-1))
        if self.temporal_mode != "none":
            if temporal_features is None:
                raise Stage2V52ContractError("temporal transfer requires the frozen named feature mapping")
            h = self.temporal_adapter(h, stack_temporal_features(temporal_features).to(dtype))
        h = h + sinusoidal_position_encoding(route_sequence, h.shape[-1]).to(dtype)
        valid = (~pad_mask).unsqueeze(-1).to(dtype)
        h = h * valid
        h = (h + backbone.local_route(h.transpose(1, 2)).transpose(1, 2)) * valid
        h = backbone.route_transformer(h, src_key_padding_mask=pad_mask)
        h = backbone.final_norm(h) * valid
        result = self._decode(h, valid, gate)
        support = edge_train_support.to(h.dtype)
        result["edge_transfer_gate"] = support / (support + self.edge_representation.tau)
        return result

    def transfer_manifest_fields(self, *, backbone_lr: float, new_branch_lr: float) -> dict[str, Any]:
        if self.source_provenance is None:
            raise Stage2V52ContractError("source checkpoint provenance is not initialized")
        adapter = self.temporal_adapter.parameter_count()
        shared = sum(parameter.numel() for parameter in self.backbone.parameters())
        return {
            **self.source_provenance,
            "spatial_mode": self.spatial_mode,
            "temporal_mode": self.temporal_mode,
            "adapter_insertion_point": "after_input_history_fusion_before_local_route_transformer",
            "temporal_adapter_parameter_count": adapter,
            "temporal_adapter_parameter_ratio": adapter / max(shared, 1),
            "backbone_lr": float(backbone_lr),
            "new_branch_lr": float(new_branch_lr),
        }
