"""Research-world data/lifecycle checks with no native actors or live routing."""

from dataclasses import asdict

import pandas as pd
import pytest

from stage4.dispatch.controlled_fleet import build_controlled_fleet
from stage4.dispatch.fleet_normalization import FLEET_REL, FleetScenario
from stage4.fleetpy_adapter.research_world import ControlledSupply, episode_from_frames

START = pd.Timestamp("2016-10-31T00:00:00+08:00")


def make_episode(*, requests=None, patience_s=10., shift_end_s=30., include_truth=True):
    if requests is None:
        requests = [("inside", 10., True), ("outside", 20., False), ("future", 40., True)]
    base = pd.DataFrame([
        dict(order_id=oid, request_time=START + pd.Timedelta(seconds=release),
             pickup_lon_wgs84=108.95 if compatible else 120.0,
             pickup_lat_wgs84=34.25 if compatible else 35.0,
             dropoff_lon_wgs84=109.0, dropoff_lat_wgs84=34.3,
             realized_service_time_s=999., observed_service_time_s=998.)
        for oid, release, compatible in requests
    ])
    routes = pd.DataFrame([
        dict(order_id=oid, common_eligible=True, predicted_route_time_p50_s=50.,
             compatible_C=compatible, compatible_M=True, compatible_A=True, compatible_HV=True)
        for oid, release, compatible in requests
    ])
    fleet = FleetScenario(pd.DataFrame([
        dict(vehicle_id="COMMON_SLOT_00000", native_id=0, slot_id="source-session", vehicle_type="AV",
             availability_start_time=START, availability_end_time=START + pd.Timedelta(seconds=shift_end_s),
             initial_lon_wgs84=108.95, initial_lat_wgs84=34.25,
             availability_policy="STOP_ADMISSION_FINISH_COMMITTED")
    ]), [], {"seed": 20260824})
    return episode_from_frames(base, routes, fleet, benchmark_start=START, profile_id="C",
                               patience_s=patience_s, passenger_acceptance_rate=1.,
                               truth_frame=None if include_truth else base[["order_id"]])


def test_citywide_feed_retains_outside_and_c_incompatible_requests():
    episode = make_episode()
    assert [r.order_id for r in episode.reveal_until(20.)] == ["outside"]
    row = episode.snapshot(20.).requests[0]
    assert row.pickup_lon_wgs84 == 120.0
    assert row.profile_id == "C" and not row.av_compatible
    assert "HV" in row.compatible_profiles
    assert episode.inventory()["requests"]["common_population_requests"] == 3
    assert episode.inventory()["native_multi_vehicle_solver_status"] == "NOT_CONNECTED"


def test_release_committed_expired_and_future_rows_are_isolated():
    episode = make_episode()
    assert episode.snapshot(9.).requests == ()
    assert [r.order_id for r in episode.reveal_until(10.)] == ["inside"]
    episode.commit_request("inside", "COMMON_SLOT_00000", now_s=10.)
    assert episode.snapshot(20.).requests[0].order_id == "outside"
    assert episode.reveal_until(20.) == ()
    assert episode.snapshot(30.).requests == ()  # Expires exactly at the deadline.
    counts = episode.counts()
    assert (counts["revealed"], counts["unreleased"], counts["committed"], counts["expired"]) == (2, 1, 1, 1)
    with pytest.raises(ValueError, match="admission window"):
        episode.commit_request("future", "COMMON_SLOT_00000", now_s=30.)
    with pytest.raises(PermissionError, match="after commitment"):
        episode.execution_truth.lookup("future")
    with pytest.raises(ValueError, match="backwards"):
        episode.snapshot(29.)


def test_execution_truth_never_enters_decision_and_requires_commitment():
    episode = make_episode()
    decision = asdict(episode.snapshot(10.).requests[0])
    assert decision["predicted_route_time_p50_s"] == 50.
    assert not any("realized" in key or "observed" in key or "arrival" in key for key in decision)
    with pytest.raises(PermissionError, match="after commitment"):
        episode.execution_truth.lookup("inside")
    episode.commit_request("inside", "COMMON_SLOT_00000", now_s=10.)
    truth = episode.execution_truth.lookup("inside")
    assert truth.observed_service_time_s == 998.
    assert truth.realized_service_time_s == 999.
    assert truth.observed_arrival_time_s is None


def test_shift_end_stops_admission_and_preserves_commit_until_observed_completion():
    episode = make_episode(requests=[("first", 29., True), ("next", 31., True)], patience_s=300.)
    plan = episode.commit_request("first", "COMMON_SLOT_00000", now_s=29.)
    assert plan.predicted_service_end_s > 30.  # Completion need not fit the admission window.
    vehicle = episode.snapshot(31.).vehicles[0]
    assert vehicle.finishing_committed and not vehicle.can_admit and not vehicle.admission_open
    assert vehicle.active_order_id == "first"
    with pytest.raises(ValueError, match="admission window"):
        episode.commit_request("next", "COMMON_SLOT_00000", now_s=31.)
    episode.complete_request("first", now_s=100.)
    vehicle = episode.snapshot(100.).vehicles[0]
    assert vehicle.availability_state == "EXITED" and not vehicle.can_admit
    assert vehicle.location_source == "OBSERVED_ENVIRONMENT"
    assert vehicle.current_lon_wgs84 == 109.


def test_missing_execution_truth_is_not_replaced_by_p50():
    episode = make_episode(include_truth=False)
    episode.commit_request("inside", "COMMON_SLOT_00000", now_s=10.)
    truth = episode.execution_truth.lookup("inside")
    assert truth.status == "MISSING"
    assert truth.observed_service_time_s is None and truth.realized_service_time_s is None
    assert episode.inventory()["execution_truth"]["p50_fallback"] is False


def test_c_and_m_profiles_reuse_identical_controlled_q_seed_slots(monkeypatch):
    template = pd.DataFrame([
        dict(source_session_id=f"session-{i}", source_driver_id=f"driver-{i}",
             availability_start_time=START, availability_end_time=START + pd.Timedelta(hours=2),
             initial_lon_wgs84=108.95, initial_lat_wgs84=34.25)
        for i in range(8)
    ])
    monkeypatch.setattr("stage4.dispatch.controlled_fleet.pd.read_parquet", lambda path: template.copy())
    scenario = build_controlled_fleet(".", benchmark_start=START,
                                      simulation_end=START + pd.Timedelta(days=2),
                                      requested_q_a=.5, seed=20260824)
    c = ControlledSupply(scenario, START, "C")
    m = ControlledSupply(scenario, START, "M")
    assert [(r.slot_id, r.vehicle_type) for r in c.plan()] == [(r.slot_id, r.vehicle_type) for r in m.plan()]
    assert {r.profile_id for r in c.plan() if r.vehicle_type == "AV"} == {"C"}
    assert c.inventory()["accounting"]["type_label_sha256"] == m.inventory()["accounting"]["type_label_sha256"]
