import numpy as np
import pandas as pd
import pytest

from stage3.scripts.traffic_state_offline_comparison import bounds, classify


def test_bounds_preserve_unknown_for_advanced_profile():
    x = pd.DataFrame({"traffic_unknown_share": [.2], "traffic_mixed_evidence_share": [.1],
                      "traffic_congested_proxy_share": [.3], "traffic_severe_congestion_proxy_share": [.1]})
    assert bounds(x, "C")[0][0] == pytest.approx(.4)
    assert bounds(x, "M")[0][0] == pytest.approx(.1)
    u, z, hi = bounds(x, "A")
    assert u[0] == 0 and hi[0] == pytest.approx(.3)
    assert classify(u, hi, [False], 0)[0] == "UNRESOLVED_BOUND"


def test_replacement_not_stacked_and_retained_family():
    # No old crawl/stop input: old exceedance cannot silently remain a gate.
    states = classify(np.array([0., .2, 0., 0.]), np.array([0., .3, .4, 0.]),
                      [False, False, False, True], .1)
    assert states.tolist() == ["WITHIN_BOUND", "KNOWN_EXPOSURE_EXCEEDS",
                              "UNRESOLVED_BOUND", "RETAINED_VARIABILITY_EXCEEDS"]


def test_budget_monotonicity_and_invalid_inputs():
    u = np.array([0., .1, .5]); hi = np.array([.2, .4, 1.])
    previous = np.zeros(3, bool)
    for b in np.linspace(0, 1, 101):
        within = classify(u, hi, [False]*3, b) == "WITHIN_BOUND"
        assert not (previous & ~within).any()
        previous = within
    assert previous.all()
    with pytest.raises(ValueError):
        classify(np.array([np.nan]), np.array([1.]), [False], .1)
