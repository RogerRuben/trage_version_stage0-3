from stage0.scripts.audit_connector_manual_review_v4 import connector_review_pass


def test_connector_review_requires_actual_completed_judgments():
    metrics = {
        "schema_errors": [],
        "completed_connectors": 30,
        "major_error_rate": 0.10,
        "wrong_direction_rate": 0.03,
        "wrong_level_transition_rate": 0.03,
    }
    assert connector_review_pass(metrics)
    metrics["completed_connectors"] = 29
    assert not connector_review_pass(metrics)
