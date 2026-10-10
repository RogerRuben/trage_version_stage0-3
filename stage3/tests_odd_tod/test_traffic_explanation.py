import numpy as np
import pandas as pd
import pytest

from stage3.odd_tod.traffic_explanation import build_explanations, attach_explanations, STATES


def example():
    return pd.DataFrame({"date": ["20161025"], "order_id": ["a"],
        "pred_total_time_s": [100.], "pred_longest_congested_run_s": [10.],
        **{f"pred_{s}_share": [v] for s, v in zip(STATES, [.5, .1, .1, .1, .2])},
        "obs_congested_or_severe_share": [.99]})


def test_no_decision_change_or_observed_leakage():
    side = build_explanations(example())
    assert not any(c.startswith("obs_") for c in side)
    original = pd.DataFrame({"date": ["20161025"]*3, "order_id": ["a"]*3,
        "selected_route_reference": ["ORIGINAL:a"]*3, "profile_id": ["C", "M", "A"],
        "hard_state": ["FEASIBLE", "UNKNOWN", "INFEASIBLE"], "rho_dynamic": [2., 1., .5],
        "reason_codes": ["[]"]*3, "crawl_E": [.4]*3}, index=[7, 7, 2])
    result = attach_explanations(original, side)
    pd.testing.assert_frame_equal(result[original.columns], original)
    assert result.traffic_unknown_share.eq(.2).all()


def test_fallback_missing_and_date_identity_do_not_inherit():
    decisions = pd.DataFrame({"date": ["20161025", "20161026", "20161025"],
        "order_id": ["a", "a", "missing"],
        "selected_route_reference": ["FALLBACK:a:123", "ORIGINAL:a", "ORIGINAL:missing"]})
    result = attach_explanations(decisions, build_explanations(example()))
    assert result.traffic_unknown_share.isna().all()
    assert result.traffic_explanation_status.eq("UNAVAILABLE_FOR_SELECTED_ROUTE").all()


def test_invalid_partition_and_duplicate_rejected():
    bad = example()
    bad["pred_unknown_share"] = np.nan
    with pytest.raises(ValueError):
        build_explanations(bad)
    with pytest.raises(ValueError):
        build_explanations(pd.concat([example(), example()]))
    side = build_explanations(example())
    with pytest.raises(ValueError):
        attach_explanations(side, side)
