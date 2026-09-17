from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from stage2.v5.aggregation import RouteDimension, aggregate_route_dimensions
from stage2.v5.reference import aggregate_route_dimensions_reference


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["test"] * 7,
            "date": ["20161028"] * 7,
            "order_id": ["a"] * 5 + ["b"] * 2,
            "route_sequence": [0, 1, 2, 3, 4, 0, 1],
            "pct": [0.2, 0.91, 0.95, 0.4, 0.99, 0.2, 0.3],
            "prob": [0.1, 0.8, 0.9, 0.2, 0.95, 0.1, 0.2],
            "value": [0.3, 0.7, 0.8, np.nan, 0.9, 0.2, 0.4],
            "weight": [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 3.0],
        }
    )


def test_vectorized_route_aggregation_matches_reference() -> None:
    spec = (RouteDimension("rts", "pct", "prob", "value", "weight"),)
    actual = aggregate_route_dimensions(_frame(), spec)
    expected = aggregate_route_dimensions_reference(_frame(), spec)
    assert_frame_equal(actual, expected, check_dtype=False, atol=1e-10, rtol=0)
    no_tail = actual.loc[actual["order_id"].eq("b")].iloc[0]
    assert not no_tail["rts_tail_event_present"]
    assert np.isnan(no_tail["rts_conditional_tail_severity"])
    assert np.isclose(actual.iloc[0]["rts_weighted_coverage_share"], 11.0 / 15.0)

