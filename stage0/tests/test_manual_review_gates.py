from stage0.scripts.audit_manual_route_truth import passes_review_gate


def acceptable_metrics():
    return {
        "completed_reviews": 120,
        "core_major_error_rate": 0.15,
        "core_wrong_direction_rate": 0.05,
        "core_wrong_road_level_rate": 0.05,
        "core_unreasonable_detour_rate": 0.10,
    }


def test_manual_major_error_metrics_gate():
    assert passes_review_gate(acceptable_metrics(), 30, 0.80)
    metrics = acceptable_metrics()
    metrics["core_major_error_rate"] = 0.151
    assert not passes_review_gate(metrics, 30, 0.80)


def test_double_review_minimum_agreement():
    assert not passes_review_gate(acceptable_metrics(), 29, 1.0)
    assert not passes_review_gate(acceptable_metrics(), 30, 0.799)
