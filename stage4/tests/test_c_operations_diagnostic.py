"""Small accounting checks; no replay, routing Actor, model or GPU."""
import json

import pandas as pd
import pytest

from stage4.analysis.c_operations_diagnostic import clipped_ledger, request_partition


def test_clipping_and_interrupted_movement_conserve_admission_time():
    events = [(95., 105., "PICKUP"), (105., 145., "LOADED"), (155., 160., "MOVE")]
    row = clipped_ledger(events, 100., 160., 0., 200.)
    assert row == dict(admission_s=60., PICKUP=5., LOADED=40., MOVE=5.,
                      idle_s=10., longest_idle_s=10., idle_intervals=1)
    tail = clipped_ledger([(150., 180., "LOADED")], 100., 170., 160., 175.)
    assert tail["admission_s"] == tail["LOADED"] == 10.
    assert tail["idle_s"] == 0.


def test_overlap_cannot_be_disguised_as_lower_idle_time():
    with pytest.raises(ValueError, match="overlapping"):
        clipped_ledger([(0., 20., "PICKUP"), (10., 30., "MOVE")], 0., 60.)


def test_nearby_C_is_not_claimed_as_a_final_legal_C_arc():
    frame = pd.DataFrame(dict(C_compatible=[True] * 4 + [False],
        passenger_accepts_av=[True] * 5, completed=[True, True, True, False, True],
        expired=[False, False, False, True, False], vehicle_type=["AV", "HV", "HV", None, "HV"],
        spatial_av_opportunities=[0, 0, 4, 0, 9],
        rejection_counts_json=["{}", "{}", json.dumps({"UTURN_OUTSIDE_RESEARCH_PROFILE": 2}), "{}", "{}"]))
    row = request_partition(frame)
    assert row["eligible_orders"] == 4
    assert row["eligible_served_C"] == row["eligible_expired"] == 1
    assert row["eligible_served_HV"] == 2
    assert row["HV_served_with_observed_nearby_C"] == row["HV_served_with_no_observed_nearby_C"] == 1
    assert row["rejection_reason_request_counts"]["UTURN_OUTSIDE_RESEARCH_PROFILE"] == 1
    assert row["rejection_reason_attempt_counts"]["UTURN_OUTSIDE_RESEARCH_PROFILE"] == 2
    assert row["nearby_does_not_prove_final_C_arc_feasibility"]
    assert not row["final_validated_C_arc_count_available"]
