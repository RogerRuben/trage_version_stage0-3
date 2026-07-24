from stage2.canonical.dispatch_time import audit_dispatch_features
from stage2.scripts.build_stage2_dispatch_smoke import FEATURES

import pandas as pd


def test_smoke_feature_list_is_exactly_dispatch_safe():
    frame = pd.DataFrame({
        "order_id": ["o", "o"],
        "decision_time": pd.to_datetime(["2016-10-20T00:00:00Z"] * 2),
        "feature_availability_timestamp": pd.to_datetime(["2016-10-20T00:00:00Z"] * 2),
    })
    result = audit_dispatch_features(frame, FEATURES)
    assert result["status"] == "PASS"
    assert result["forbidden_model_features"] == []
    assert result["features_outside_whitelist"] == []


def test_realized_fields_are_not_model_features():
    assert "enter_time" not in FEATURES
    assert "travel_time_sec" not in FEATURES
    assert "actual_link_entry_time" not in FEATURES
