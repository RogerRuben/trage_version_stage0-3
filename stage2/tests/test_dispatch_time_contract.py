import pandas as pd

from stage2.canonical.dispatch_time import (
    attach_dispatch_snapshot,
    audit_dispatch_features,
    hierarchical_fallback,
)


def test_all_route_links_use_same_order_decision_cutoff():
    route = pd.DataFrame({
        "order_id": ["o1", "o1"],
        "route_link_id": ["a", "b"],
        "route_link_seq": [0, 1],
        "decision_time": ["2016-10-23T01:00:00Z"] * 2,
        "estimated_link_entry_time": ["2016-10-23T01:00:00Z", "2016-10-23T01:30:00Z"],
    })
    state = pd.DataFrame({
        "link_id": ["a", "b", "b"],
        "availability_timestamp": [
            "2016-10-23T00:55:00Z", "2016-10-23T00:55:00Z", "2016-10-23T01:20:00Z"
        ],
        "recent_speed_mean": [1.0, 2.0, 99.0],
        "recent_traversal_count": [10, 10, 10],
    })
    result = attach_dispatch_snapshot(route, state)
    assert result.loc[result.route_link_id.eq("b"), "recent_speed_mean"].iloc[0] == 2.0
    assert result.information_cutoff.nunique() == 1


def test_requested_support_is_not_overwritten_by_global_fallback():
    tables = [
        ("link_time", pd.DataFrame({"key": ["seen"], "value": [1.0], "sample_size": [5]})),
        ("global", pd.DataFrame({"key": ["unseen", "seen"], "value": [3.0, 3.0], "sample_size": [1000, 1000]})),
    ]
    result = hierarchical_fallback(pd.Series(["unseen", "seen"]), tables, minimum_support=100)
    assert result.requested_level_support_count.tolist() == [0, 5]
    assert result.fallback_level.tolist() == ["global", "global"]


def test_dispatch_audit_rejects_realized_feature():
    frame = pd.DataFrame({
        "order_id": ["o"],
        "decision_time": ["2016-10-23T01:00:00Z"],
        "feature_availability_timestamp": ["2016-10-23T00:59:00Z"],
    })
    audit = audit_dispatch_features(frame, ["route_link_id", "travel_time_sec"])
    assert audit["status"] == "FAIL"
    assert audit["forbidden_model_features"] == ["travel_time_sec"]
