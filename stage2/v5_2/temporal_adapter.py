"""Small temporal adapter and fail-closed causal update selection."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .contracts import Stage2V52ContractError, require_columns

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def causal_update_mask(
    events: pd.DataFrame,
    *,
    decision_time: float,
    current_order_id: str,
) -> np.ndarray:
    """Select only completed prior observations, excluding the current order."""
    require_columns(events.columns, ("order_id", "observation_end_time"), product="adapter events")
    end = pd.to_numeric(events["observation_end_time"], errors="coerce").to_numpy(np.float64)
    order = events["order_id"].astype(str).to_numpy()
    return np.isfinite(end) & (end < float(decision_time)) & (order != str(current_order_id))


def causal_update_events(
    events: pd.DataFrame,
    *,
    decision_time: float,
    current_order_id: str,
) -> pd.DataFrame:
    return events.loc[causal_update_mask(
        events, decision_time=decision_time, current_order_id=current_order_id
    )].copy()


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
