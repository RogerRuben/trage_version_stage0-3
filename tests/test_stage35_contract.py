import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]


def schema():
    return json.loads(
        (ROOT / "stage35/config/stage35_route_output.schema.json").read_text(encoding="utf-8")
    )


def base_row():
    return {
        "order_id": "o1",
        "hv_route_id": "hv1",
        "hv_route_source": "shortest_expected_time",
        "hv_link_sequence": ["l1", "l2"],
        "hv_route_time_sec": 120.0,
        "hv_route_distance_m": 1000.0,
        "hv_route_risk_vector": {"lcs": 0.2},
        "av_route_available": False,
        "odd_profile_id": "moderate_av_v1",
        "prediction_cutoff_time": "2016-10-23T08:00:00",
        "network_version": "network_v3",
        "stage2_model_version": "stage2_dispatch_v2",
        "stage3_model_version": "stage3_route_risk_v2",
        "stage35_version": "stage35_offline_route_selection_v1",
    }


def test_unavailable_av_route_remains_valid_and_explicit():
    Draft202012Validator(schema()).validate(base_row())


def test_available_av_route_requires_route_payload():
    row = base_row()
    row["av_route_available"] = True
    errors = list(Draft202012Validator(schema()).iter_errors(row))
    assert errors
