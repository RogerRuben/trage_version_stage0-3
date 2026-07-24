from __future__ import annotations

import pandas as pd

from stage4.simulator_v3.entities import RequestState
from stage4.simulator_v3.logging.request_logger import request_to_record


def test_request_log_persists_independent_odd_decisions() -> None:
    now = pd.Timestamp("2016-10-23T08:00:00Z")
    request = RequestState(
        order_id="o1",
        request_time=now,
        observed_boarding_time=now + pd.Timedelta(minutes=3),
        origin_lon=108.9,
        origin_lat=34.2,
        origin_zone="z1",
        destination_lon=108.95,
        destination_lat=34.25,
        destination_zone="z2",
        latest_pickup_time=now + pd.Timedelta(minutes=8),
        condition_available=True,
        predicted_service_time_sec=600.0,
        realized_service_time_sec=620.0,
        route_length_m=4000.0,
        stress_value=0.4,
    )
    request.assigned_vehicle_id = "AV_1"
    request.metadata.update(
        {
            "assigned_vehicle_type": "AV",
            "pickup_odd_feasible": True,
            "service_odd_feasible": True,
            "combined_odd_feasible": True,
            "capability_profile": "moderate_av",
            "capability_mapping_version": "dimension_specific_scenario_priors_v4_exogenous_no_test_day",
        }
    )
    record = request_to_record(request)
    assert record["assigned_vehicle_type"] == "AV"
    assert record["pickup_odd_feasible"] is True
    assert record["service_odd_feasible"] is True
    assert record["combined_odd_feasible"] is True
    assert record["capability_profile"] == "moderate_av"
    assert record["capability_mapping_version"].endswith("no_test_day")
