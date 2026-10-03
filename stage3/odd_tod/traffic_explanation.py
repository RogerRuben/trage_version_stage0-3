"""Additive, profile-neutral explanations; never a dispatch constraint."""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["date", "order_id", "selected_route_reference"]
STATES = ("non_congested_proxy", "congested_proxy", "severe_congestion_proxy",
          "mixed_evidence", "unknown")
SOURCE_COLUMNS = ["pred_total_time_s", "pred_longest_congested_run_s",
                  *[f"pred_{s}_share" for s in STATES]]


def build_explanations(routes: pd.DataFrame) -> pd.DataFrame:
    """Only allowlisted prediction fields cross into the explanation product.

    The input is batch2's complete ORIGINAL Validation-route overlap product,
    not an arbitrary route or an observed-completion-time traffic table.
    """
    x = routes[["date", "order_id", *SOURCE_COLUMNS]].copy()
    x["date"] = x.date.astype(str)
    x["order_id"] = x.order_id.astype(str)
    if x.duplicated(["date", "order_id"]).any():
        raise ValueError("duplicate original route")
    shares = x[[f"pred_{s}_share" for s in STATES]].to_numpy(float)
    time = x.pred_total_time_s.to_numpy(float)
    run = x.pred_longest_congested_run_s.to_numpy(float)
    if (not np.isfinite(shares).all() or (shares < 0).any()
            # Upstream shares are float32; validate without renormalizing them.
            or (shares > 1).any() or not np.allclose(shares.sum(axis=1), 1, atol=1e-6, rtol=0)
            or not np.isfinite(time).all() or (time <= 0).any()
            or not np.isfinite(run).all() or (run < 0).any() or (run > time + 1e-6).any()):
        raise ValueError("invalid predicted exposure partition/time")
    x["selected_route_reference"] = "ORIGINAL:" + x.order_id
    x = x.rename(columns={c: "traffic_" + c.removeprefix("pred_") for c in SOURCE_COLUMNS})
    x["traffic_explanation_status"] = "AVAILABLE"
    x["traffic_explanation_version"] = "1.0"
    x["traffic_source_family"] = "FROZEN_M3_PACE_CRAWL_STOP"
    x["traffic_use_policy"] = "EXPLANATION_ONLY_NO_ADDITIONAL_GATE_OR_PENALTY"
    x["traffic_support_basis"] = "TRAIN_EDGE_REFERENCE_100_ORDER_DAYS_5_DATES"
    x["traffic_current_window_support_assessed"] = False
    x["traffic_explanation_codes"] = [
        "SHARED_SOURCE_WITH_EQC;NOT_INDEPENDENT_EVIDENCE"
        + (";UNKNOWN_EXPOSURE_PRESENT" if u > 0 else "")
        + (";MIXED_EVIDENCE_PRESENT" if m > 0 else "")
        for u, m in zip(x.traffic_unknown_share, x.traffic_mixed_evidence_share)
    ]
    return x


def attach_explanations(decisions: pd.DataFrame, explanations: pd.DataFrame) -> pd.DataFrame:
    """Left attach by date, order and EXACT selected route, preserving all inputs.

    Missing sidecar / fallback route stays unavailable, never zero exposure.
    A caller must supply canonical string keys; this function does not rewrite
    decision identity or reason codes. Existing explanation fields cannot be
    overwritten accidentally.
    """
    if explanations.duplicated(KEYS).any():
        raise ValueError("duplicate explanation identity")
    extra = [c for c in explanations if c not in KEYS]
    if any(not c.startswith("traffic_") for c in extra):
        raise ValueError("non-explanation field in sidecar")
    if set(extra) & set(decisions):
        raise ValueError("would overwrite existing columns")
    if "traffic_explanation_status" not in extra:
        raise ValueError("missing explanation status")
    # Join positions independently so even a nonunique decision index survives.
    left = decisions[KEYS].reset_index(drop=True)
    right = left.merge(explanations, how="left", on=KEYS, sort=False, validate="many_to_one")
    result = decisions.copy()
    for column in extra:
        result[column] = right[column].to_numpy()
    result["traffic_explanation_status"] = result.traffic_explanation_status.fillna("UNAVAILABLE_FOR_SELECTED_ROUTE")
    pd.testing.assert_frame_equal(result[decisions.columns], decisions)
    return result
