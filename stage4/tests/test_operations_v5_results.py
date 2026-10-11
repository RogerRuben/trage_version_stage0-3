"""One small accounting check; no replay, routing, or policy selection."""
import pandas as pd

from stage4.analysis.operations_v5_results import typed_partition
from stage4.analysis.operations_v5_ablation_results import paired_counts


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


def test_ablation_pair_distinguishes_net_service_from_exchanged_requests():
    common = dict(order_id=["a", "b", "c"], release_s=[0., 1., 2.],
        C_compatible=[True] * 3, passenger_accepts_av=[True] * 3)
    before = pd.DataFrame(dict(common, completed=[True, True, False], vehicle_type=["HV", "HV", None],
        wait_s=[100., 200., None], pickup_distance_m=[1000., 2000., None])).set_index("order_id")
    after = pd.DataFrame(dict(common, completed=[True, False, True], vehicle_type=["AV", None, "HV"],
        wait_s=[80., None, 150.], pickup_distance_m=[800., None, 1500.])).set_index("order_id")
    row = paired_counts(before, after)
    assert row["before_only_served"] == row["after_only_served"] == row["shared_served"] == 1
    assert row["HV_to_C"] == 1 and row["C_to_HV"] == 0
    assert row["shared_served_total_wait_difference_mean_s"] == -20.
    assert row["shared_served_pickup_distance_difference_km"] == -.2
