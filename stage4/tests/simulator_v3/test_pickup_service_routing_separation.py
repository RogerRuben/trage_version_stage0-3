from pathlib import Path

import pandas as pd

from stage4.simulator_v3.entities import RequestState
from stage4.simulator_v3.routing_engine import RoutingEngine


def test_service_route_uses_request_eta_not_pickup_speed_proxy():
    req = RequestState(
        order_id="o1",
        request_time=pd.Timestamp("2016-10-23T00:00:00Z"),
        observed_boarding_time=pd.Timestamp("2016-10-23T00:00:00Z"),
        origin_lon=108.9,
        origin_lat=34.2,
        origin_zone="z0_0",
        destination_lon=109.0,
        destination_lat=34.3,
        destination_zone="z1_1",
        latest_pickup_time=pd.Timestamp("2016-10-23T00:08:00Z"),
        condition_available=True,
        predicted_service_time_sec=777,
        realized_service_time_sec=888,
        route_length_m=4321,
        stress_value=0.2,
        metadata={"eta_source": "stage2_predicted_eta"},
    )
    route = RoutingEngine(Path("stage4/data/decoupled_abm")).query_service_route(req, req.request_time, "HV")
    assert route.road_distance_m == 4321
    assert route.expected_travel_time_sec == 777
    assert route.realized_travel_time_sec == 888
    assert "historical_service_backend" in route.route_source
