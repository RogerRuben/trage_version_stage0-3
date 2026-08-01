"""RC-MSTNet v5 with physical constraints and stochastic pace output."""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_position_encoding(positions: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequency = torch.exp(
        torch.arange(half, device=positions.device, dtype=torch.float32)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    angles = positions.to(torch.float32).unsqueeze(-1) * frequency
    encoding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    if encoding.shape[-1] < dimension:
        encoding = torch.nn.functional.pad(encoding, (0, dimension - encoding.shape[-1]))
    return encoding


class HorizonGate(nn.Module):
    """Blend recent and strict profile state using decision-time support."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        del hidden_dim
        self.bias = nn.Parameter(torch.tensor(1.0))
        self.horizon_penalty = nn.Parameter(torch.tensor(1.0))
        self.age_penalty = nn.Parameter(torch.tensor(1.0))
        self.support_reward = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        recent_state: torch.Tensor,
        profile_state: torch.Tensor,
        forecast_horizon_s: torch.Tensor,
        history_age_s: torch.Tensor,
        history_support: torch.Tensor,
        *,
        use_recent: bool = True,
        use_profile: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if recent_state.shape != profile_state.shape:
            raise ValueError("recent/profile state shapes differ")
        horizon = torch.log1p(forecast_horizon_s.clamp_min(0.0)) / 12.0
        age = torch.log1p(history_age_s.clamp_min(0.0)) / 12.0
        support = torch.log1p(history_support.clamp_min(0.0)) / 8.0
        gate_logit = (
            self.bias
            - torch.nn.functional.softplus(self.horizon_penalty) * horizon
            - torch.nn.functional.softplus(self.age_penalty) * age
            + torch.nn.functional.softplus(self.support_reward) * support
        )
        gate = torch.sigmoid(gate_logit)
        if not use_recent:
            gate = torch.zeros_like(gate)
        if not use_profile:
            gate = torch.ones_like(gate)
        fused = gate.unsqueeze(-1) * recent_state + (1.0 - gate).unsqueeze(-1) * profile_state
        return fused, gate


class RCMSTNetV5(nn.Module):
    def __init__(
        self,
        *,
        numeric_feature_count: int,
        categorical_sizes: tuple[int, ...],
        hidden_dim: int = 128,
        categorical_embedding_dim: int = 24,
        transformer_layers: int = 3,
        attention_heads: int = 4,
        dropout: float = 0.1,
        minimum_log_scale: float = -5.0,
        maximum_log_scale: float = 2.0,
    ):
        super().__init__()
        self.minimum_log_scale = float(minimum_log_scale)
        self.maximum_log_scale = float(maximum_log_scale)
        self.numeric_encoder = nn.Sequential(
            nn.Linear(numeric_feature_count * 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, categorical_embedding_dim, padding_idx=0) for size in categorical_sizes]
        )
        self.categorical_encoder = nn.Sequential(
            nn.Linear(categorical_embedding_dim * len(categorical_sizes), hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.input_fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.recent_encoder = nn.Sequential(nn.Linear(4, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.profile_encoder = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.horizon_gate = HorizonGate(hidden_dim)
        self.history_fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.local_route = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.route_transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.crawl_head = nn.Linear(hidden_dim, 1)
        self.stop_presence_head = nn.Linear(hidden_dim, 1)
        self.stop_positive_head = nn.Linear(hidden_dim, 1)
        self.speed_cv_head = nn.Linear(hidden_dim, 1)
        self.acceleration_head = nn.Linear(hidden_dim, 1)
        self.rts_head = nn.Linear(hidden_dim, 1)
        self.lcs_tail_head = nn.Linear(hidden_dim, 1)
        self.rts_tail_head = nn.Linear(hidden_dim, 1)
        self.pace_distribution_head = nn.Linear(hidden_dim, 2)
        self.availability_head = nn.Linear(hidden_dim, 4)

    def forward(
        self,
        numeric: torch.Tensor,
        numeric_missing: torch.Tensor,
        categorical: torch.Tensor,
        route_sequence: torch.Tensor,
        pad_mask: torch.Tensor,
        *,
        recent_history: torch.Tensor | None = None,
        profile_history: torch.Tensor | None = None,
        forecast_horizon_s: torch.Tensor | None = None,
        history_age_s: torch.Tensor | None = None,
        history_support: torch.Tensor | None = None,
        use_recent: bool = True,
        use_profile: bool = True,
    ) -> dict[str, torch.Tensor]:
        numeric_input = torch.cat((numeric, numeric_missing.to(numeric.dtype)), dim=-1)
        numeric_state = self.numeric_encoder(numeric_input)
        categorical_state = self.categorical_encoder(
            torch.cat([embedding(categorical[..., index]) for index, embedding in enumerate(self.embeddings)], dim=-1)
        )
        h = self.input_fusion(torch.cat((numeric_state, categorical_state), dim=-1))
        batch_shape = h.shape[:-1]
        device = h.device
        dtype = h.dtype
        recent_history = torch.zeros((*batch_shape, 4), device=device, dtype=dtype) if recent_history is None else recent_history
        profile_history = torch.zeros((*batch_shape, 3), device=device, dtype=dtype) if profile_history is None else profile_history
        forecast_horizon_s = torch.zeros(batch_shape, device=device, dtype=dtype) if forecast_horizon_s is None else forecast_horizon_s
        history_age_s = torch.zeros(batch_shape, device=device, dtype=dtype) if history_age_s is None else history_age_s
        history_support = torch.zeros(batch_shape, device=device, dtype=dtype) if history_support is None else history_support
        history, gate = self.horizon_gate(
            self.recent_encoder(recent_history),
            self.profile_encoder(profile_history),
            forecast_horizon_s,
            history_age_s,
            history_support,
            use_recent=use_recent,
            use_profile=use_profile,
        )
        h = self.history_fusion(torch.cat((h, history), dim=-1))
        h = h + sinusoidal_position_encoding(route_sequence, h.shape[-1]).to(dtype)
        valid = (~pad_mask).unsqueeze(-1).to(dtype)
        h = h * valid
        h = (h + self.local_route(h.transpose(1, 2)).transpose(1, 2)) * valid
        h = self.route_transformer(h, src_key_padding_mask=pad_mask)
        h = self.final_norm(h) * valid

        stop_presence_logit = self.stop_presence_head(h).squeeze(-1)
        stop_positive = torch.sigmoid(self.stop_positive_head(h).squeeze(-1))
        stop_share = torch.sigmoid(stop_presence_logit) * stop_positive
        crawl = (1.0 - stop_share) * torch.sigmoid(self.crawl_head(h).squeeze(-1))
        speed_cv = torch.sigmoid(self.speed_cv_head(h).squeeze(-1))
        acceleration = torch.sigmoid(self.acceleration_head(h).squeeze(-1))
        rts = torch.sigmoid(self.rts_head(h).squeeze(-1))
        lcs = (crawl + stop_share + speed_cv + acceleration) / 4.0
        distribution = self.pace_distribution_head(h)
        pace_log_mu = distribution[..., 0]
        pace_log_scale = torch.clamp(distribution[..., 1], self.minimum_log_scale, self.maximum_log_scale)
        sigma = torch.exp(pace_log_scale)
        pace_mean = torch.exp(pace_log_mu + 0.5 * sigma.square())
        z90 = 1.2815515655446004
        z95 = 1.6448536269514722
        return {
            "crawl_share": crawl,
            "stop_presence_logit": stop_presence_logit,
            "stop_positive_share": stop_positive,
            "stop_share": stop_share,
            "speed_cv": speed_cv,
            "acceleration_rms": acceleration,
            "rts_raw": rts,
            "lcs_reconstructed_raw": lcs,
            "lcs_tail_logit": self.lcs_tail_head(h).squeeze(-1),
            "rts_tail_logit": self.rts_tail_head(h).squeeze(-1),
            "pace_log_mu": pace_log_mu,
            "pace_log_scale": pace_log_scale,
            "pace_pred_mean": pace_mean,
            "pace_pred_p50": torch.exp(pace_log_mu),
            "pace_pred_p90": torch.exp(pace_log_mu + z90 * sigma),
            "pace_pred_p95": torch.exp(pace_log_mu + z95 * sigma),
            "availability_logits": self.availability_head(h),
            "history_recent_gate": gate,
        }
