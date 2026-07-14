import pandas as pd
import pytest

from stage4.simulator_v3.entities import RequestState
from stage4.simulator_v3.enums import RequestStatus
from stage4.simulator_v3.request_manager import RequestManager


def make_request() -> RequestState:
    t = pd.Timestamp("2016-10-23T00:00:00Z")
    return RequestState(
        order_id="o1",
        request_time=t,
        observed_boarding_time=t,
        origin_lon=108.9,
        origin_lat=34.2,
        origin_zone="z0_0",
        destination_lon=108.91,
        destination_lat=34.21,
        destination_zone="z0_0",
        latest_pickup_time=t + pd.Timedelta(minutes=8),
        condition_available=True,
        predicted_service_time_sec=600,
        realized_service_time_sec=600,
        route_length_m=3000,
        stress_value=0.2,
    )


def test_request_transition_log_and_illegal_terminal_transition():
    req = make_request()
    mgr = RequestManager()
    now = req.request_time
    mgr.transition(req, RequestStatus.PENDING, now, trigger="REQUEST_REVEALED")
    mgr.transition(req, RequestStatus.CANCELLED, now, trigger="TEST_CANCEL")
    assert mgr.transition_records[-1]["old_status"] == "PENDING"
    assert mgr.transition_records[-1]["new_status"] == "CANCELLED"
    with pytest.raises(ValueError):
        mgr.transition(req, RequestStatus.PENDING, now, trigger="ILLEGAL")

