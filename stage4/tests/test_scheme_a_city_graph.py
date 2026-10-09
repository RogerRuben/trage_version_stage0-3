"""Four focused contracts for the restricted city service-chain graph."""
from dataclasses import replace

import pandas as pd
import pytest

from stage4.dispatch.scheme_a_city_graph import (
    ChainTask, CoarseHistoricalLibrary, RouteState, build_restricted_chains,
)


def point(meters):
    return (108.95 + meters / 92100, 34.25)


def task(rid, release, pickup_m, dropoff_m, *, duration=60, profiles=frozenset({"C", "HV"})):
    return ChainTask(rid, release, release + 300, duration,
        point(pickup_m), point(dropoff_m), profiles)


def cfg(**extra):
    return dict(planning_horizon_s=1800, decision_time_s=0,
        planning_horizon_end_s=1800, **extra)


def certificate(state, target, departure, profile):
    return dict(supported=True, compatible_profiles={"C", "HV"},
        travel_time_s=10, empty_distance_m=100,
        arrival_context=dict(kind="ISSUED_EMPTY_PATH", scalar=True),
        route_geometry=[point(0), point(100)])


def test_direct_evidence_admits_multiple_services_and_only_scalar_path_summaries():
    start = RouteState(point(0), 0, 1800, "C")
    tasks = [task(1, 10, 0, 100), task(2, 100, 100, 200), task(3, 200, 200, 300)]
    chains, stats = build_restricted_chains("wait", 0, "s", start, tasks, [],
        certificate, cfg(), lambda: None)
    assert chains and len(chains[0].request_ids) == 3
    assert stats["max_service_chain_length"] == 3
    assert stats["scope"] == "DECLARED_RESTRICTED_CHAIN_HEURISTIC"
    assert stats["global_upper_bound_claimed"] is False
    assert len(chains) <= 2
    for chain in chains:
        previous_finish = 0
        for activity in chain.payload["activities"]:
            assert activity["departure_s"] >= previous_finish
            assert activity["departure_s"] % 30 == 0
            previous_finish = activity["finish_s"]
            assert "route_geometry" not in activity
        assert len(set(chain.request_ids)) == len(chain.request_ids)


def test_necessary_intermediate_relocation_uses_global_slots_and_propagates_context():
    first, second = task(1, 10, 0, 0), task(2, 1000, 2500, 2600)
    site = dict(site_id="relay", position=point(1500))
    seen = []

    def connector(state, target, departure, profile):
        seen.append((state.context, target, departure))
        result = certificate(state, target, departure, profile)
        if isinstance(target, dict):
            assert departure % 900 == 0
            result["travel_time_s"] = 60
            result["arrival_context"] = {"kind": "RELAY_ROUTE", "departure_s": departure}
        elif target.request_id == 2:
            assert state.context["kind"] == "RELAY_ROUTE"
        return result

    chains, stats = build_restricted_chains("wait", 0, "s", RouteState(point(0), 0, 1800, "C"),
        [first, second], [site], connector, cfg(), lambda: None)
    chain = next(chain for chain in chains if chain.request_ids == (1, 2))
    assert [activity["kind"] for activity in chain.payload["activities"]] == ["SERVICE", "MOVE", "SERVICE"]
    assert chain.relocation_slots == (900,)
    assert stats["max_future_relocation_count"] == 1
    assert chain.payload["service_count"] == 2  # MOVE is not another served customer.
    assert all(activity["finish_s"] < 1800 for activity in chain.payload["activities"])


def test_duplicate_expired_unsupported_and_C_illegal_tasks_never_enter_a_chain():
    good = task(1, 40, 0, 50)
    bad_profile = task(2, 50, 0, 50, profiles=frozenset({"HV"}))
    unsupported = task(3, 70, 0, 50)
    expired = replace(task(4, 0, 0, 50), deadline_s=20)
    rejected = replace(task(5, 60, 0, 50), passenger_accepts_av=False)

    def connector(state, target, departure, profile):
        result = certificate(state, target, departure, profile)
        if target.request_id == 3:
            result["supported"] = False
        return result

    supplied = [good, good, bad_profile, unsupported, expired, rejected]
    start = RouteState(point(0), 30, 1800, "C")
    chains, stats = build_restricted_chains("wait", 1, "s", start, supplied, [],
        connector, cfg(), lambda: None)
    assert chains and all(chain.request_ids == (1,) for chain in chains)
    assert stats["duplicate_input_task_count"] == 1
    assert stats["connector_rejected"] >= 1
    with pytest.raises(RuntimeError, match="budget"):
        build_restricted_chains("wait", 1, "s", start, supplied, [], connector, cfg(), lambda: False)


def test_historical_forecast_refresh_is_stable_between_300_second_boundaries():
    records = []
    for date in ("20161010", "20161017"):
        for index, release in enumerate([10] * 20 + [600] * 20 + [1900] * 20):
            records.append(dict(date=date, order_id=f"{date}-{index}", release_second=release,
                start_lon_wgs84=108.95, start_lat_wgs84=34.25,
                end_lon_wgs84=108.96, end_lat_wgs84=34.25,
                predicted_route_time_p50_s=80, compatible_C=True, compatible_M=True,
                compatible_A=True, research_data_ready=True))
    templates = pd.DataFrame(records)
    settings = dict(forecast_train_dates=["20161010", "20161017"],
        forecast_seed=20261004, forecast_sampling_multiplier=3.0,
        passenger_acceptance_seed=20260827, passenger_acceptance_rate=0.7,
        patience_s=300, coarse_reference_update_s=300, planning_horizon_s=1800)
    library = CoarseHistoricalLibrary(templates, settings)
    first = library.view(0)
    later = library.view(30)
    almost_refresh = library.view(299)
    assert library.refresh_count == 1
    for scene, tasks in first.items():
        original = {task.request_id: task for task in tasks}
        assert all(original[task.request_id] == task for task in later[scene])
        assert all(original[task.request_id] == task for task in almost_refresh[scene])
        assert all(task.release_s > 30 for task in later[scene])
        assert len(later[scene]) < len(tasks)
        assert all(task.deadline_s == task.release_s + 300 for task in tasks)
        assert all(task.source_date <= "20161024" and task.source_order_id for task in tasks)
        assert all(task.request_id < 0 and "HV" in task.compatible_profiles for task in tasks)
    refreshed = library.view(300)
    assert library.refresh_count == 2
    assert any(task.release_s > 1800 for tasks in refreshed.values() for task in tasks)
    assert set(library.weights) == set(refreshed) and sum(library.weights.values()) == 1
    assert all(record["order_id"] not in str(library.diagnostics()) for record in records)
    with pytest.raises(ValueError, match="prediction-only"):
        CoarseHistoricalLibrary(templates.assign(realized_service_time_s=10), settings)
    with pytest.raises(ValueError, match="by 20161024"):
        CoarseHistoricalLibrary(templates, dict(settings, forecast_train_dates=["20161031"]))
