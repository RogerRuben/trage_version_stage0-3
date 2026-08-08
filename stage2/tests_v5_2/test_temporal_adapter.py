from __future__ import annotations

import pandas as pd
import pytest

from stage2.v5_2.temporal_adapter import TemporalAdapter, causal_update_events


def test_causal_online_update_excludes_future_and_current_order() -> None:
    events = pd.DataFrame({
        "order_id": ["old", "current", "future", "equal"],
        "observation_end_time": [90.0, 80.0, 110.0, 100.0],
    })
    selected = causal_update_events(events, decision_time=100.0, current_order_id="current")
    assert selected["order_id"].tolist() == ["old"]


def test_adapter_parameter_budget_is_at_most_ten_percent() -> None:
    torch = pytest.importorskip("torch")
    shared = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.Linear(128, 64))
    adapter = TemporalAdapter(hidden_dim=64, time_feature_dim=4, bottleneck_dim=4)
    assert adapter.assert_budget(shared) <= 0.10
