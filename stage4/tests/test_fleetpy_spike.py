from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from stage4.fleetpy_adapter import mixed_fleet_adapter as fleet_adapter
from stage4.fleetpy_adapter import test31_demand_adapter as demand_adapter
from stage4.fleetpy_adapter import valhalla_time_adapter as eta_module
from stage4.fleetpy_adapter.dispatch_stub import MinCorrectedPickupEtaDispatchStub
from stage4.fleetpy_adapter.mixed_fleet_adapter import VehicleFixture, VehicleRuntime
from stage4.fleetpy_adapter.replay_service_time_adapter import FleetPyLifecycleAdapter
from stage4.fleetpy_adapter.spike_runner import load_config
from stage4.fleetpy_adapter.test31_demand_adapter import SpikeRequest
from stage4.fleetpy_adapter.upstream import CoordinateRegistry, load_fleetpy_bindings
from stage4.fleetpy_adapter.valhalla_time_adapter import (
    PickupEstimate,
    ValhallaPickupTimeAdapter,
)

TIMEZONE = "Asia/Shanghai"


def _request(order_id: str = "o1", hard_state: str = "FEASIBLE", evidence: bool = True):
    return SpikeRequest(
        native_id=0,
        order_id=order_id,
        request_time=pd.Timestamp("2016-10-31 08:00", tz=TIMEZONE),
        sim_time_s=0,
        pickup_lon_wgs84=108.90,
        pickup_lat_wgs84=34.20,
        dropoff_lon_wgs84=108.95,
        dropoff_lat_wgs84=34.25,
        realized_service_time_s=300.0,
        predicted_service_time_s=240.0,
        profile_id="M",
        hard_state=hard_state,
        evidence_complete=evidence,
        rho_static=0.5,
        rho_dynamic=0.6,
        rho_speed=0.7,
    )


def test_test31_request_time_maps_exactly_and_selection_is_deterministic(
    tmp_path: Path, monkeypatch
):
    rows = []
    for index in range(6):
        rows.append(
            {
                "order_id": f"o{index}",
                "request_time": pd.Timestamp(f"2016-10-31 08:0{index}:00", tz=TIMEZONE),
                "pickup_lon_wgs84": 108.9,
                "pickup_lat_wgs84": 34.2,
                "dropoff_lon_wgs84": 109.0,
                "dropoff_lat_wgs84": 34.3,
                "realized_service_time_s": 300.0,
                "predicted_service_time_s": 250.0,
                "profile_id": "M",
                "hard_state": "FEASIBLE",
                "evidence_complete": True,
                "rho_static": 0.5,
                "rho_dynamic": 0.6,
                "rho_speed": 0.7,
            }
        )
    pd.DataFrame(rows).to_parquet(tmp_path / "orders.parquet", index=False)
    monkeypatch.setattr(demand_adapter, "ORDER_BASE_REL", Path("orders.parquet"))
    start = pd.Timestamp("2016-10-31 08:00", tz=TIMEZONE)
    end = pd.Timestamp("2016-10-31 09:00", tz=TIMEZONE)
    first = demand_adapter.load_test31_requests(
        tmp_path, start=start, end=end, profile_id="M", request_count=3, seed=7
    )
    second = demand_adapter.load_test31_requests(
        tmp_path, start=start, end=end, profile_id="M", request_count=3, seed=7
    )
    assert [item.order_id for item in first] == [item.order_id for item in second]
    assert all(
        item.request_time == start + pd.Timedelta(seconds=item.sim_time_s)
        for item in first
    )


def test_hv_unavailable_before_start_and_after_end():
    start = pd.Timestamp("2016-10-31 08:10", tz=TIMEZONE)
    end = pd.Timestamp("2016-10-31 08:50", tz=TIMEZONE)
    fixture = VehicleFixture("HV_000", 0, "HV", 108.9, 34.2, start, end, "s1", False)
    runtime = VehicleRuntime(fixture, None, "AVAILABLE", 108.9, 34.2)
    assert not runtime.available_at(start - pd.Timedelta(seconds=1))
    assert runtime.available_at(start)
    assert not runtime.available_at(end)


def test_av_fixture_spans_full_horizon_and_does_not_inherit_session_end(
    tmp_path: Path, monkeypatch
):
    source = pd.DataFrame(
        {
            "source_session_id": ["s1", "s2", "s3"],
            "availability_start_time": pd.to_datetime(
                ["2016-10-31 07:50", "2016-10-31 08:10", "2016-10-31 09:30"]
            ).tz_localize(TIMEZONE),
            "availability_end_time": pd.to_datetime(
                ["2016-10-31 09:10", "2016-10-31 08:30", "2016-10-31 10:00"]
            ).tz_localize(TIMEZONE),
            "initial_lon_wgs84": [108.9, 108.91, 108.92],
            "initial_lat_wgs84": [34.2, 34.21, 34.22],
        }
    )
    source.to_parquet(tmp_path / "fleet.parquet", index=False)
    monkeypatch.setattr(fleet_adapter, "FLEET_REL", Path("fleet.parquet"))
    start = pd.Timestamp("2016-10-31 08:00", tz=TIMEZONE)
    end = pd.Timestamp("2016-10-31 09:00", tz=TIMEZONE)
    fixtures = fleet_adapter.load_mixed_fleet_fixture(
        tmp_path, start=start, end=end, hv_count=1, av_count=1, seed=3
    )
    av = next(item for item in fixtures if item.vehicle_type == "AV")
    assert av.availability_start_time == start
    assert av.availability_end_time == end
    assert av.av_source_session_end_inherited is False


def test_pickup_eta_uses_wgs84_valhalla_and_correct_15min_beta(
    tmp_path: Path, monkeypatch
):
    calibration = pd.DataFrame(
        {
            "time_bin_index": range(96),
            "selected_eta_multiplier": [2.0] * 96,
        }
    )
    calibration.to_parquet(tmp_path / "beta.parquet", index=False)
    monkeypatch.setattr(eta_module, "CALIBRATION_REL", Path("beta.parquet"))

    class Actor:
        request = None

        def route(self, request):
            self.request = request
            return {
                "trip": {
                    "status": 0,
                    "legs": [{}],
                    "summary": {"time": 100.0, "length": 1.5},
                }
            }

    actor = Actor()
    adapter = ValhallaPickupTimeAdapter(tmp_path, actor=actor)
    estimate = adapter.estimate(
        108.90,
        34.20,
        108.95,
        34.25,
        pd.Timestamp("2016-10-31 08:02", tz=TIMEZONE),
    )
    assert actor.request["locations"][0]["lon"] == 108.90
    assert actor.request["locations"][1]["lat"] == 34.25
    assert estimate.time_bin_index == 32
    assert estimate.beta == 2.0
    assert estimate.corrected_pickup_eta_s == 200.0


def test_hv_rejected_when_predicted_completion_exceeds_session_end():
    start = pd.Timestamp("2016-10-31 08:00", tz=TIMEZONE)
    fixture = VehicleFixture(
        "HV_000",
        0,
        "HV",
        108.9,
        34.2,
        start,
        start + pd.Timedelta(minutes=5),
        "s",
        False,
    )
    vehicle = VehicleRuntime(fixture, None, "AVAILABLE", 108.9, 34.2)

    class ETA:
        def estimate(self, *args):
            return PickupEstimate(120.0, 120.0, 1000.0, 1.0, 32, False)

    stub = MinCorrectedPickupEtaDispatchStub(
        start=start,
        end=start + pd.Timedelta(hours=1),
        requests=[_request()],
        vehicles=[vehicle],
        eta_adapter=ETA(),
        lifecycle=None,
        native_output=[],
    )
    assert stub._candidate_estimate(vehicle, _request(), start) is None
    assert len(stub.hv_session_end_exclusions) == 1


def test_av_smoke_filter_is_only_feasible_and_evidence_complete():
    assert _request().av_smoke_eligible
    assert not _request(hard_state="UNKNOWN").av_smoke_eligible
    assert not _request(evidence=False).av_smoke_eligible


def test_config_contains_no_or_gamma_passenger_or_penetration_fields():
    config = load_config(Path("stage4/config/fleetpy_spike.json"))
    forbidden = {
        "Gamma",
        "passenger_acceptance",
        "av_penetration",
        "gurobi",
        "repositioning",
        "charging",
    }
    assert not forbidden & set(config)
    assert config["engineering_hv_count"] == 40
    assert config["engineering_av_count"] == 10


def test_real_fleetpy_native_lifecycle_relocates_to_dropoff(tmp_path: Path):
    fleetpy_root = os.environ.get("FLEETPY_ROOT")
    if not fleetpy_root:
        pytest.skip(
            "FLEETPY_ROOT external checkout is required for native integration test"
        )
    bindings = load_fleetpy_bindings(fleetpy_root)
    registry = CoordinateRegistry()
    request = _request()
    request_db = demand_adapter.attach_fleetpy_requests([request], bindings, registry)
    start = request.request_time
    fixture = VehicleFixture(
        "AV_000",
        0,
        "AV",
        108.89,
        34.19,
        start,
        start + pd.Timedelta(hours=1),
        "source",
        False,
    )
    vehicles, native_output = fleet_adapter.create_native_vehicles(
        [fixture], bindings, registry, request_db, tmp_path
    )
    vehicle = vehicles[0]
    lifecycle = FleetPyLifecycleAdapter(bindings, registry)
    estimate = PickupEstimate(60.0, 60.0, 500.0, 1.0, 32, False)
    lifecycle.assign(vehicle, request, estimate, 0.0)
    lifecycle.arrive_pickup(vehicle, request, 60.0)
    lifecycle.arrive_dropoff(vehicle, request, 360.0)
    assert request.native_request.pu_time == 60.0
    assert request.native_request.do_time == 360.0
    assert vehicle.current_lon_wgs84 == request.dropoff_lon_wgs84
    assert vehicle.current_lat_wgs84 == request.dropoff_lat_wgs84
    assert vehicle.native_vehicle.assigned_route == []
    assert len(native_output) == 4
