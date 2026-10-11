"""One small accounting check; no replay, routing, or policy selection."""
import pandas as pd

from stage4.analysis.operations_v5_results import typed_partition


def test_final_C_alternative_partition_is_exclusive_and_witnesses_are_separate():
    records = []
    for idle, other, moving in ((0, 0, 0), (1, 1, 1), (0, 2, 0), (0, 1, 1)):
        total = idle + other + moving
        records.append(dict(C_compatible=True, passenger_accepts_av=True, completed=True, expired=False,
            vehicle_type="HV", typed_last_final_AV_count=total,
            typed_HV_assigned_with_final_AV_same_epoch=bool(total),
            typed_AV_alternatives_idle_after_choice=idle, typed_AV_alternatives_serving_other_request=other,
            typed_AV_alternatives_moving_after_choice=moving,
            typed_rejection_counts_json='{"AV:DEPARTURE_JOIN:UTURN_OUTSIDE_RESEARCH_PROFILE": 2}'))
    result = typed_partition(pd.DataFrame(records))
    assert result["eligible_served_HV"] == 4
    assert result["HV_with_final_C_count"] == 3
    assert set(result["exclusive_final_HV_categories"].values()) == {1}
    witness = result["lifetime_rejection_witnesses"]["ELIGIBLE_HV_NO_FINAL_C"]
    assert witness["request_count"] == 1
    assert witness["phase_request_counts"] == {"DEPARTURE_JOIN": 1}
    assert witness["reason_attempt_counts"]["AV:DEPARTURE_JOIN:UTURN_OUTSIDE_RESEARCH_PROFILE"] == 2
