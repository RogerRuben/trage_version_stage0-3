from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stage2.v4.causality import eligible_history_mask
from stage2.v4.contracts import Stage2V4ContractError


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["current"] * 5,
            "history_order_id": ["old", "equal", "future", "current", "missing"],
            "decision_time": [100.0] * 5,
            "availability_timestamp": [99.0, 100.0, 101.0, 90.0, 80.0],
        }
    )


def test_only_strictly_prior_other_order_is_available() -> None:
    values = _candidates().iloc[:4].copy()
    assert eligible_history_mask(values).tolist() == [True, False, False, False]


def test_missing_dynamic_timestamp_fails_closed() -> None:
    values = _candidates()
    values.loc[4, "availability_timestamp"] = np.nan
    with pytest.raises(Stage2V4ContractError, match="missing timestamps fail closed"):
        eligible_history_mask(values)
