from __future__ import annotations

import pandas as pd

from stage0.v6.preprocess import preprocess_order
from stage0.v6.valhalla_client import ValhallaMatcher


def test_preprocess_keeps_equal_timestamp_different_coordinates_and_deduplicates_exact():
    points = pd.DataFrame(
        {
            "order_id": ["o"] * 5,
            "timestamp": [3, 1, 1, 1, 2],
            "lon": [108.0, 108.1, 108.1, 108.2, 108.3],
            "lat": [34.0, 34.1, 34.1, 34.2, 34.3],
            "point_seq": [4, 0, 1, 2, 3],
        }
    )
    result = preprocess_order(
        points,
        coordinate_system="wgs84",
        maximum_speed_mps=1_000_000,
        minimum_subtrace_points=1,
    )
    assert len(result.points) == 4
    equal_time = result.points.loc[result.points.timestamp.eq(1)]
    assert len(equal_time) == 2
    assert set(equal_time.lon) == {108.1, 108.2}
    assert result.metrics["duplicate_point_count"] == 1
    assert result.metrics["timestamp_reverse_count"] == 1
    assert "original_point_seq" in result.points


class FakeActor:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def trace_attributes(self, request):
        self.calls.append(request)
        return next(self.responses)


def test_python_actor_is_reused_and_retry_is_bounded():
    success = {
        "edges": [{"id": 1}],
        "matched_points": [
            {"type": "matched"},
            {"type": "interpolated"},
            {"type": "unmatched"},
        ],
    }
    actor = FakeActor([{"edges": [], "matched_points": []}, success, success])
    matcher = ValhallaMatcher(
        {
            "backend": "python",
            "search_radius_m": 80,
            "retry_search_radius_m": 160,
            "controlled_retry": True,
        },
        actor=actor,
    )
    points = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "matching_lon": [108.0, 108.1, 108.2],
            "matching_lat": [34.0, 34.1, 34.2],
        }
    )
    first = matcher.match_order(points)
    second = matcher.match_order(points)
    assert first["status"] == "success"
    assert first["retry_count"] == 1
    assert first["matched_point_count"] == 1
    assert first["interpolated_point_count"] == 1
    assert first["unmatched_point_count"] == 1
    assert len(actor.calls) == 3
    assert actor.calls[0]["trace_options"]["search_radius"] == 80
    assert actor.calls[1]["trace_options"]["search_radius"] == 160
    assert actor.calls[0]["use_timestamps"] is True


def test_http_client_uses_service_url_and_hard_timeout():
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"edges": [{"id": 1}], "matched_points": []}

    class FakeSession:
        def __init__(self):
            self.calls = []

        def post(self, url, json, timeout):
            self.calls.append((url, json, timeout))
            return FakeResponse()

    session = FakeSession()
    matcher = ValhallaMatcher(
        {
            "backend": "http",
            "service_url": "http://127.0.0.1:8002",
            "request_timeout_s": 7.5,
            "controlled_retry": False,
            "search_radius_m": 80,
        },
        session=session,
    )
    points = pd.DataFrame(
        {
            "timestamp": [1, 2],
            "matching_lon": [108.0, 108.1],
            "matching_lat": [34.0, 34.1],
        }
    )
    result = matcher.match_order(points)

    assert result["status"] == "success"
    assert len(session.calls) == 1
    url, payload, timeout = session.calls[0]
    assert url == "http://127.0.0.1:8002/trace_attributes"
    assert payload["trace_options"]["search_radius"] == 80
    assert timeout == 7.5
