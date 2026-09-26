from __future__ import annotations

import numpy as np
import pandas as pd

from stage2.v4.config import load_config
from stage2.v4.entry_time import estimate_entry_times
from stage2.v4.history_index import TemporalHistoryIndex


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "availability_timestamp": [90.0, 100.0, 110.0],
            "observed_directed_edge_uid": ["e1", "e1", "e1"],
            "canonical_highway": ["primary", "primary", "primary"],
            "profile_time_bin": [0, 0, 0],
            "profile_weekday_type": ["weekday", "weekday", "weekday"],
            "lcs_available": [True, True, True],
            "rts_available": [True, True, True],
            "crawl_time_share": [0.1, 0.2, 0.3],
            "stop_time_share": [0.0, 0.1, 0.2],
            "speed_cv_bounded": [0.2, 0.3, 0.4],
            "acceleration_rms_bounded": [0.2, 0.3, 0.4],
            "lcs_raw": [0.1, 0.2, 0.3],
            "rts_raw": [0.1, 0.2, 0.3],
            "lcs_tail_event": [False, False, True],
            "rts_tail_event": [False, False, True],
            "observed_sec_per_m": [0.1, 0.2, 0.3],
        }
    )


def test_temporal_index_excludes_equal_and_future_events() -> None:
    index = TemporalHistoryIndex(_events())
    query = pd.DataFrame(
        {
            "decision_time": [100.0],
            "observed_directed_edge_uid": ["e1"],
            "canonical_highway": ["primary"],
            "profile_time_bin": [0],
            "profile_weekday_type": ["weekday"],
        }
    )
    result = index.query_metric(query, level="edge", metric="crawl_time_share")
    assert result.count.tolist() == [1]
    assert result.mean.tolist() == [0.1]
    assert result.maximum_event_time.tolist() == [90.0]


class _StaticHistory:
    def query_fallback(self, queries, *, metrics, minimum_observations):
        assert tuple(metrics) == ("observed_sec_per_m",)
        n = len(queries)
        return pd.DataFrame(
            {
                "observed_sec_per_m_profile_mean": np.full(n, np.nan),
                "observed_sec_per_m_profile_std": np.full(n, np.nan),
                "observed_sec_per_m_profile_count": np.zeros(n, dtype=int),
                "observed_sec_per_m_profile_maximum_event_time": np.full(n, np.nan),
                "observed_sec_per_m_profile_fallback_level": pd.Series(
                    [pd.NA] * n,
                    dtype="string",
                ),
            },
            index=queries.index,
        )


def test_two_pass_entry_time_uses_only_route_static_and_decision_time() -> None:
    config = load_config("stage2/config/stage2_v4.json")
    route = pd.DataFrame(
        {
            "split": ["train", "train"],
            "date": ["20161009", "20161009"],
            "order_id": ["o1", "o1"],
            "route_sequence": [0, 1],
            "decision_time": [1475971200.0, 1475971200.0],
            "route_part_length_m": [100.0, 200.0],
            "canonical_highway": ["primary", "primary"],
            "observed_directed_edge_uid": ["e0", "e1"],
        }
    )
    result = estimate_entry_times(route, _StaticHistory(), config)
    expected_first_travel = 100.0 / (45.0 / 3.6)
    assert result["estimated_entry_time"].iloc[0] == route["decision_time"].iloc[0]
    assert np.isclose(
        result["estimated_entry_time"].iloc[1],
        route["decision_time"].iloc[1] + expected_first_travel,
    )
    assert set(result["estimated_travel_time_source"]) == {"static_speed_fallback"}
    assert "entry_time_pass_1" in result
    assert "entry_time_pass_2" in result
