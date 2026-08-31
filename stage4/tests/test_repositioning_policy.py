from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from stage4.dispatch.paper_enhancement_repositioning_runner import _execution_config
from stage4.dispatch.repositioning_policy import (
    POLICY_NAME,
    TrainTODRepositioningManager,
    largest_remainder_quotas,
    load_train_demand_reference,
    surplus_to_deficit_plan,
    time_bin_index,
)


ROOT = Path(__file__).resolve().parents[2]


def _reference(weight_a: float = 0.5) -> pd.DataFrame:
    rows = []
    for bin_index in range(96):
        rows.extend(
            [
                {
                    "time_bin_index": bin_index,
                    "node_id": 1,
                    "lon_wgs84": 108.90,
                    "lat_wgs84": 34.20,
                    "train_pickup_count": 1,
                    "demand_share": weight_a,
                },
                {
                    "time_bin_index": bin_index,
                    "node_id": 2,
                    "lon_wgs84": 109.00,
                    "lat_wgs84": 34.20,
                    "train_pickup_count": 1,
                    "demand_share": 1.0 - weight_a,
                },
            ]
        )
    return pd.DataFrame(rows)


class _Registry:
    def __init__(self) -> None:
        self.coords: list[tuple[float, float]] = []

    def position_for(self, lon: float, lat: float) -> tuple[int, None, None]:
        point = (float(lon), float(lat))
        if point not in self.coords:
            self.coords.append(point)
        return (self.coords.index(point), None, None)


class _Network:
    def __init__(self) -> None:
        self.registry = _Registry()
        self.legs = []

    def return_position_coordinates(self, position: tuple) -> tuple[float, float]:
        return self.registry.coords[int(position[0])]

    def register_vehicle_leg(self, *args) -> None:
        self.legs.append(args)


class _Native:
    def __init__(self, pos: tuple, status: str = "IDLE") -> None:
        self.pos = pos
        self.status = status
        self.assigned_route = []

    def assign_vehicle_plan(self, route, simulation_time: int) -> None:
        del simulation_time
        self.assigned_route = list(route)
        self.status = "REPOSITION"


@dataclass
class _Fixture:
    native_id: int
    vehicle_id: str
    vehicle_type: str = "AV"


@dataclass
class _Runtime:
    fixture: _Fixture
    native_vehicle: _Native
    active_order_id: str | None = None
    state: str = "NATIVE_AVAILABLE"


class _ETA:
    def __init__(self, succeeds: bool = True) -> None:
        self.succeeds = succeeds

    def estimate_many(self, candidates, pickup_lon, pickup_lat, timestamp):
        del pickup_lon, pickup_lat, timestamp
        if not self.succeeds:
            return {}
        return {
            item.native_vehicle_id: SimpleNamespace(
                corrected_pickup_eta_s=120.0, route_distance_m=1500.0
            )
            for item in candidates
        }


class _Control:
    def _available(self, runtime, simulation_time: int) -> bool:
        del simulation_time
        return (
            runtime.native_vehicle.status == "IDLE"
            and not runtime.native_vehicle.assigned_route
            and runtime.active_order_id is None
        )


def _manager(runtimes, network, *, succeeds: bool, weight_a: float):
    states = SimpleNamespace(IDLE="IDLE", REPOSITION="REPOSITION")
    bindings = SimpleNamespace(
        states=states,
        vehicle_route_leg=lambda status, destination, requests: SimpleNamespace(
            status=status, destination=destination, requests=requests
        ),
    )
    return TrainTODRepositioningManager(
        bindings=bindings,
        runtimes=runtimes,
        network=network,
        eta_adapter=_ETA(succeeds),
        reference=_reference(weight_a),
        start=pd.Timestamp("2016-10-31T00:00:00+08:00"),
        policy_end=pd.Timestamp("2016-11-01T00:01:00+08:00"),
    )


def test_train_reference_excludes_test31_and_maps_all_nodes() -> None:
    reference, manifest = load_train_demand_reference(ROOT)
    assert manifest["test31_included"] is False
    assert manifest["validation_dates_included"] == []
    assert manifest["train_dates"] == [f"201610{day:02d}" for day in range(9, 25)]
    assert manifest["mapped_pickup_node_count"] == manifest["unique_pickup_node_count"]
    assert set(reference["time_bin_index"]) == set(range(96))


def test_time_bin_and_largest_remainder_are_exact_and_deterministic() -> None:
    assert time_bin_index(pd.Timestamp("2016-10-31T17:44:59+08:00")) == 70
    weights = pd.Series([0.34, 0.33, 0.33], index=[10, 20, 30])
    first = largest_remainder_quotas(weights, 5)
    second = largest_remainder_quotas(weights, 5)
    assert int(first.sum()) == 5
    assert first.equals(second)
    assert first.to_dict() == {10: 2, 20: 2, 30: 1}


def test_surplus_plan_moves_only_excess_and_has_deterministic_ties() -> None:
    idle = [
        {"native_vehicle_id": i, "vehicle_id": f"AV_{i}", "lon_wgs84": 108.90, "lat_wgs84": 34.20}
        for i in range(3)
    ]
    target = _reference(0.34).query("time_bin_index == 0")
    first, distribution = surplus_to_deficit_plan(idle, target)
    second, _ = surplus_to_deficit_plan(idle, target)
    assert [item.vehicle_id for item in first] == ["AV_1", "AV_2"]
    assert first == second
    assert int(distribution["target_quota"].sum()) == len(idle)


def test_only_post_dispatch_idle_av_repositions_and_is_unavailable() -> None:
    network = _Network()
    origin = network.registry.position_for(108.90, 34.20)
    idle_a = _Runtime(_Fixture(0, "AV_0"), _Native(origin))
    idle_b = _Runtime(_Fixture(1, "AV_1"), _Native(origin))
    assigned = _Runtime(_Fixture(2, "AV_2"), _Native(origin, "BUSY"), "order")
    manager = _manager([idle_a, idle_b, assigned], network, succeeds=True, weight_a=0.5)
    control = _Control()

    manager.after_normal_dispatch(control, 0)

    assert idle_a.native_vehicle.assigned_route == []
    assert idle_b.native_vehicle.assigned_route
    assert assigned.native_vehicle.assigned_route == []
    assert control._available(idle_b, 0) is False
    assert manager.trip_rows[0]["deadhead_odd_qualified"] is False

    idle_b.native_vehicle.pos = idle_b.native_vehicle.assigned_route[0].destination
    idle_b.native_vehicle.assigned_route = []
    idle_b.native_vehicle.status = "IDLE"
    manager.before_normal_dispatch(120)
    assert manager.active == {}
    assert manager.trip_rows[0]["status"] == "COMPLETED"
    assert control._available(idle_b, 120) is True


def test_failed_reposition_route_never_teleports_vehicle() -> None:
    network = _Network()
    origin = network.registry.position_for(108.90, 34.20)
    runtime = _Runtime(_Fixture(0, "AV_0"), _Native(origin))
    manager = _manager([runtime], network, succeeds=False, weight_a=0.01)

    manager.after_normal_dispatch(_Control(), 0)

    assert runtime.native_vehicle.pos == origin
    assert runtime.native_vehicle.assigned_route == []
    assert manager.route_failures == 1
    assert manager.trip_rows[0]["status"] == "ROUTING_FAILED_STAYED_IN_PLACE"


def test_repositioning_disabled_uses_isolated_no_reposition_output() -> None:
    config = _execution_config(ROOT, enabled=False)
    assert config["repositioning_enabled"] is False
    assert config["output_root"].endswith("no_reposition_reproduction")
    assert config["repositioning_policy_name"] == POLICY_NAME
