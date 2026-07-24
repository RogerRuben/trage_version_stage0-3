from stage0.canonical.quality_layers import (
    HARD_FLAGS,
    SOFT_FLAGS,
    canonical_promotion_gate,
    classify_quality_layer,
)


def passing_row():
    return {column: True for column in [*HARD_FLAGS.values(), *SOFT_FLAGS.values()]}


def test_hard_vs_soft_quality_flags():
    row = passing_row()
    row[SOFT_FLAGS["match_confidence"]] = False
    result = classify_quality_layer(row)
    assert result["route_quality_class_v4_final"] == "analysis_set"
    assert result["formal_analysis_eligible"]
    assert not result["strict_evaluation_eligible"]


def test_analysis_set_eligibility_rejects_hard_error():
    row = passing_row()
    row[HARD_FLAGS["direction_gap"]] = False
    result = classify_quality_layer(row)
    assert result["route_quality_class_v4_final"] == "rejected"
    assert not result["formal_analysis_eligible"]


def test_canonical_promotion_without_core_share_gate():
    assert canonical_promotion_gate(
        manual_pass=True, conservation_pass=True, connector_pass=True, full_date_pass=True
    )
