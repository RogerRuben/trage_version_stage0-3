from types import SimpleNamespace as NS

import pandas as pd
import pytest

from stage4.dispatch.flexibility_native import TrainDemandForecast, predicted_vehicle_states, position_key
from stage4.dispatch.remaining_time import TrainRemainingTime


def config():
    return dict(forecast_train_dates=["20161010", "20161017", "20161024"], forecast_horizon_s=600,
                forecast_seed=20261004, forecast_sampling_multiplier=3, patience_s=300)


def templates():
    return pd.DataFrame([dict(date=d, order_id=f"{d}_{i}", release_second=100+i*30,
        start_lon_wgs84=108.93, start_lat_wgs84=34.24, end_lon_wgs84=108.95, end_lat_wgs84=34.25,
        predicted_route_time_p50_s=600, compatible_C=True, compatible_M=True, compatible_A=True)
        for d in config()["forecast_train_dates"] for i in range(4)])


def test_train_only_repeatable_forecasts_and_window_cutoff():
    cfg = config()
    predictor = TrainDemandForecast(templates(), cfg, 160)
    first = predictor.scenarios(90, "C", .7, 123)
    second = predictor.scenarios(90, "C", .7, 123)
    assert first == second
    assert sum(s.probability for s, _ in first) == pytest.approx(1)
    assert all(90 < r.release_time_s < 160 for s, _ in first for r in s.new_requests)
    assert all(s.information_available_at_s < 0 for s, _ in first)
    assert all(not s.new_requests for s, _ in predictor.scenarios(160, "C", .7, 123))
    invalid = templates()
    invalid.loc[0, "date"] = "20161031"
    with pytest.raises(ValueError, match="Train-only"):
        TrainDemandForecast(invalid, cfg, 1000)


def test_busy_ready_state_uses_booked_prediction_not_realized_future():
    class KnownBookedOnly(dict):
        def __getitem__(self, key):
            assert key == 99, "an unbooked future request was accessed"
            return super().__getitem__(key)

    c = NS(config={"profile_id": "C"}, dispatch_interval_s=30, max_pickup_wait_s=300,
           _fixture_seconds=lambda x: x, bindings=NS(states=NS(IDLE=0)),
           _pickup_overhead_s=lambda: 20,
           routing_engine=NS(return_position_coordinates=lambda x: (108.93, 34.24)),
           request_by_rid=KnownBookedOnly({99: NS(dropoff_lon_wgs84=108.95, dropoff_lat_wgs84=34.25)}))
    fixture = lambda kind, start=0: NS(vehicle_type=kind, availability_start_time=start, availability_end_time=1000)
    c.runtime_by_vid = {
        1: NS(fixture=fixture("HV"), native_vehicle=NS(status=1, assigned_route=[1], pos=None)),
        2: NS(fixture=fixture("AV", 100), native_vehicle=NS(status=0, assigned_route=[], pos=None)),
    }
    booked = {1: dict(simulation_time_s=0, pickup_eta_s=10, predicted_service_time_s=120,
                      native_request_id=99, realized_service_time_s=99999, service_end_time=99999)}
    states = predicted_vehicle_states(c, 60, 600, booked)
    assert states[0].ready_time_s == 150
    assert states[0].ready_position == position_key(108.95, 34.25)
    assert states[1].ready_time_s == 100
    booked[1]["predicted_service_time_s"] = 1
    assert predicted_vehicle_states(c, 60, 600, booked)[0].ready_time_s == 90


def timing_model():
    return TrainRemainingTime(dict(method="TRAIN_EMPIRICAL_RATIO_SURVIVAL_MEDIAN",
        train_dates=config()["forecast_train_dates"], test31_used_for_fit=False,
        sorted_duration_prediction_ratios=[1., 2., 3.]))


def test_empirical_survival_remaining_time_and_train_only():
    model = timing_model()
    estimate = model.estimate(100, 150)
    assert estimate.remaining_s == 100 and estimate.support_count == 2
    assert model.estimate(100, 300).remaining_s is None
    with pytest.raises(ValueError, match="Train-only"):
        TrainRemainingTime(dict(method="TRAIN_EMPIRICAL_RATIO_SURVIVAL_MEDIAN",
            train_dates=["20161031"], test31_used_for_fit=True, sorted_duration_prediction_ratios=[1.]))


def test_busy_conditioning_never_reads_realized_end_and_unsupported_tail_omits_forecast_only():
    class NoFuture(dict):
        def __getitem__(self, key):
            assert key not in ("realized_service_time_s", "service_end_time")
            return super().__getitem__(key)
    c = NS(config={"profile_id": "M"}, dispatch_interval_s=30, max_pickup_wait_s=300,
        _fixture_seconds=lambda x: x, bindings=NS(states=NS(IDLE=0)), _pickup_overhead_s=lambda: 0,
        request_by_rid={99:NS(dropoff_lon_wgs84=108.95, dropoff_lat_wgs84=34.25)},
        runtime_by_vid={1:NS(fixture=NS(vehicle_type="HV", availability_start_time=0, availability_end_time=1000),
                            native_vehicle=NS(status=1, assigned_route=[1]))})
    booked = {1:NoFuture(simulation_time_s=0, pickup_eta_s=30, predicted_service_time_s=100, native_request_id=99)}
    assert predicted_vehicle_states(c, 180, 600, booked, timing_model())[0].ready_time_s == 280
    assert predicted_vehicle_states(c, 330, 600, booked, timing_model()) == ()
    assert c.runtime_by_vid[1].native_vehicle.assigned_route == [1]
