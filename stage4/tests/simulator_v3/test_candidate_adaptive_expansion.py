import pandas as pd

from stage4.simulator_v3.entities import RequestState, VehiclePlan, VehicleState
from stage4.simulator_v3.enums import VehicleExecutionStatus
from stage4.simulator_v3.matching.candidate_generator import CandidateGenerator, CandidatePolicy


def make_vehicle(i: int) -> VehicleState:
    t = pd.Timestamp("2016-10-23T00:00:00Z")
    return VehicleState(
        vehicle_id=f"v{i}",
        vehicle_type="HV",
        current_lon=108.9 + i * 0.00001,
        current_lat=34.2,
        current_zone="z0_0",
        online_start=t,
        online_end=t + pd.Timedelta(hours=1),
        execution_status=VehicleExecutionStatus.IDLE,
        current_leg=None,
        active_plan=VehiclePlan(f"v{i}", 0, [], t, "init", True, 0.0),
        plan_version=0,
    )


def test_candidate_generator_uses_sparse_cap_not_dense_matrix():
    t = pd.Timestamp("2016-10-23T00:00:00Z")
    req = RequestState(
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
    gen = CandidateGenerator(CandidatePolicy(maximum_candidates=20))
    result, stats = gen.generate([req], [make_vehicle(i) for i in range(50)], {"o1": 6000})
    assert len(result["o1"]) == 20
    assert stats["orders_hitting_candidate_cap"] == 1
    assert stats["candidate_truncation_rate"] > 0

