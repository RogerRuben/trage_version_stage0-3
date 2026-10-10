"""Small temporal adapter and fail-closed causal update selection."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .contracts import Stage2V52ContractError, require_columns


TEMPORAL_FEATURE_NAMES = (
    "decision_hour_sin",
    "decision_hour_cos",
    "decision_weekday_index",
    "forecast_horizon_log1p",
)

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def causal_update_mask(
    events: pd.DataFrame,
    *,
    adaptation_cutoff_time: float,
    current_order_id: str,
) -> np.ndarray:
    """Select completed orders whose labels were available before adaptation."""
    require_columns(
        events.columns,
        ("order_id", "observation_end_time", "order_completion_time", "label_available_time"),
        product="adapter events",
    )
    end = pd.to_numeric(events["observation_end_time"], errors="coerce").to_numpy(np.float64)
    completion = pd.to_numeric(events["order_completion_time"], errors="coerce").to_numpy(np.float64)
    label_time = pd.to_numeric(events["label_available_time"], errors="coerce").to_numpy(np.float64)
    order = events["order_id"].astype(str).to_numpy()
    cutoff = float(adaptation_cutoff_time)
    return (
        np.isfinite(end) & np.isfinite(completion) & np.isfinite(label_time)
        & (end < cutoff) & (completion < cutoff) & (label_time < cutoff)
        & (order != str(current_order_id))
    )


def causal_update_events(
    events: pd.DataFrame,
    *,
    adaptation_cutoff_time: float,
    current_order_id: str,
) -> pd.DataFrame:
    return events.loc[causal_update_mask(
        events, adaptation_cutoff_time=adaptation_cutoff_time, current_order_id=current_order_id
    )].copy()


def stack_temporal_features(features: Mapping[str, "torch.Tensor"]) -> "torch.Tensor":
    if tuple(features.keys()) != TEMPORAL_FEATURE_NAMES:
        raise Stage2V52ContractError(
            f"temporal feature schema must be exactly {TEMPORAL_FEATURE_NAMES}"
        )
    shapes = {tuple(value.shape) for value in features.values()}
    if len(shapes) != 1:
        raise Stage2V52ContractError("temporal feature tensors have different shapes")
    return torch.stack(tuple(features[name] for name in TEMPORAL_FEATURE_NAMES), dim=-1)


def adapter_parameter_ratio(adapter_parameters: int, shared_parameters: int) -> float:
    if shared_parameters <= 0:
        raise Stage2V52ContractError("shared backbone must have parameters")
    return float(adapter_parameters) / float(shared_parameters)


if nn is not None:
    class TemporalAdapter(nn.Module):
        """Two-layer bottleneck residual adapter conditioned on known time context."""

        def __init__(self, hidden_dim: int, time_feature_dim: int, bottleneck_dim: int):
            super().__init__()
            if min(hidden_dim, time_feature_dim, bottleneck_dim) <= 0:
                raise Stage2V52ContractError("adapter dimensions must be positive")
            self.network = nn.Sequential(
                nn.Linear(hidden_dim + time_feature_dim, bottleneck_dim),
                nn.GELU(),
                nn.Linear(bottleneck_dim, hidden_dim),
            )

        def forward(self, shared_state: "torch.Tensor", time_features: "torch.Tensor") -> "torch.Tensor":
            if shared_state.shape[:-1] != time_features.shape[:-1]:
                raise Stage2V52ContractError("adapter state/time shapes differ")
            return shared_state + self.network(torch.cat((shared_state, time_features), dim=-1))

        def parameter_count(self) -> int:
            return sum(parameter.numel() for parameter in self.parameters())

        def assert_budget(self, shared_module: "nn.Module", maximum_ratio: float = 0.10) -> float:
            shared = sum(parameter.numel() for parameter in shared_module.parameters())
            ratio = adapter_parameter_ratio(self.parameter_count(), shared)
            if ratio > maximum_ratio + 1.0e-12:
                raise Stage2V52ContractError(
                    f"temporal adapter parameter ratio {ratio:.6f} exceeds {maximum_ratio:.6f}"
                )
            return ratio
else:  # pragma: no cover
    TemporalAdapter = None
