"""Strict decision-time availability rules for Stage 2 v4 history."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import Stage2V4ContractError, require_columns


def eligible_history_mask(
    candidates: pd.DataFrame,
    *,
    current_order_column: str = "order_id",
    history_order_column: str = "history_order_id",
    decision_time_column: str = "decision_time",
    availability_column: str = "availability_timestamp",
) -> pd.Series:
    """Return the fail-closed mask for history known strictly before dispatch.

    Equality is intentionally excluded.  The current order is also excluded
    even if malformed timestamps would otherwise make it appear historical.
    """

    required = {
        current_order_column,
        history_order_column,
        decision_time_column,
        availability_column,
    }
    require_columns(candidates.columns, required, "history candidates")
    decision = pd.to_numeric(candidates[decision_time_column], errors="coerce")
    available = pd.to_numeric(candidates[availability_column], errors="coerce")
    finite = (
        np.isfinite(decision.to_numpy(dtype=float, na_value=np.nan))
        & np.isfinite(available.to_numpy(dtype=float, na_value=np.nan))
    )
    if not finite.all():
        raise Stage2V4ContractError(
            "history availability/decision timestamps must be finite; "
            "missing timestamps fail closed"
        )
    different_order = candidates[history_order_column].astype(str).ne(
        candidates[current_order_column].astype(str)
    )
    return pd.Series(
        available.to_numpy() < decision.to_numpy(),
        index=candidates.index,
        dtype=bool,
    ) & different_order


def filter_history_before_decision(candidates: pd.DataFrame) -> pd.DataFrame:
    return candidates.loc[eligible_history_mask(candidates)].copy()
