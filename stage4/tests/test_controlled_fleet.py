from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from stage4.dispatch import controlled_fleet
from stage4.dispatch.controlled_fleet import AVAILABILITY_POLICY, build_controlled_fleet
from stage4.dispatch.fleet_normalization import FLEET_REL

ROOT = Path(__file__).resolve().parents[2]
START = pd.Timestamp("2016-10-31T00:00:00+08:00")
END = START + pd.Timedelta(days=1, seconds=360)


@pytest.fixture(scope="module")
def frozen_scenarios():
    return {
        q: build_controlled_fleet(ROOT, benchmark_start=START, simulation_end=END, requested_q_a=q)
        for q in (0.0, 0.1, 1.0)
    }


def test_frozen_slots_preserve_identity_positions_and_exact_hourly_total(frozen_scenarios):
    hv, mixed = frozen_scenarios[0.0], frozen_scenarios[0.1]
    source = pd.read_parquet(ROOT / FLEET_REL).sort_values(
        ["availability_start_time", "source_session_id"], kind="mergesort"
    ).reset_index(drop=True)
    assert len(hv.scenario_fleet) == len(hv.native_fixtures) == len(source) == 8435
    pd.testing.assert_frame_equal(
        hv.scenario_fleet.drop(columns=["vehicle_type", "av_source_session_end_inherited"]),
        mixed.scenario_fleet.drop(columns=["vehicle_type", "av_source_session_end_inherited"]),
    )
    for column in source.columns:
        pd.testing.assert_series_equal(
            hv.scenario_fleet[column].astype(str) if column == "source_session_id" else hv.scenario_fleet[column],
            source[column].astype(str) if column == "source_session_id" else source[column],
            check_names=False,
        )
    assert hv.scenario_fleet["slot_id"].equals(hv.scenario_fleet["source_session_id"])
    assert [fixture.native_id for fixture in hv.native_fixtures] == list(range(8435))
    for original, relabeled in zip(hv.native_fixtures, mixed.native_fixtures):
        physical_original, physical_relabeled = asdict(original), asdict(relabeled)
        for key in ("vehicle_type", "av_source_session_end_inherited"):
            physical_original.pop(key)
            physical_relabeled.pop(key)
        assert physical_original == physical_relabeled
        assert relabeled.availability_policy == AVAILABILITY_POLICY
        assert relabeled.av_source_session_end_inherited == (relabeled.vehicle_type == "AV")
    assert len(mixed.accounting["hourly_supply"]) == 24
    for original, relabeled in zip(hv.accounting["hourly_supply"], mixed.accounting["hourly_supply"]):
        assert original["total_vehicle_hours"] == relabeled["total_vehicle_hours"] == relabeled["baseline_vehicle_hours"]
        assert relabeled["total_supply_error_nanoseconds"] == 0
        assert relabeled["hv_vehicle_hours"] + relabeled["av_vehicle_hours"] == pytest.approx(relabeled["total_vehicle_hours"], abs=1e-12)
        assert relabeled["q_a_error_pp"] == pytest.approx(100.0 * (relabeled["realized_q_a"] - 0.1))
    ledger = mixed.accounting
    assert ledger["h_base_full_template_exact"] == pytest.approx(
        sum(row["total_vehicle_hours"] for row in ledger["hourly_supply"])
        + ledger["pre_day_supply"]["total_vehicle_hours"]
        + ledger["spillover_supply"]["total_vehicle_hours"],
        abs=1e-9,
    )
    assert ledger["spillover_supply"]["slot_count"] > 0
    assert ledger["zero_day_slot_count"] > 0
    assert ledger["achieved_q_a"] == pytest.approx(ledger["achieved_av_vehicle_hours"] / ledger["h_base_day_exact"])


def test_hash_prefix_labels_are_stable_nested_and_ties_take_shorter_prefix(monkeypatch):
    # Four ten-second day slots give exact midpoint ties. Two later slots
    # exercise the separately hashed spillover labels without adding day hours.
    source = pd.DataFrame({
        "source_session_id": [f"slot-{index}" for index in range(6)],
        "source_driver_id": [f"driver-{index}" for index in range(6)],
        "availability_start_time": [START] * 4 + [START + pd.Timedelta(days=1, seconds=value) for value in (0, 30)],
        "availability_end_time": [START + pd.Timedelta(seconds=10)] * 4 + [START + pd.Timedelta(days=1, seconds=value) for value in (10, 40)],
        "initial_lon_wgs84": [108.9] * 6,
        "initial_lat_wgs84": [34.2] * 6,
    })
    monkeypatch.setattr(controlled_fleet.pd, "read_parquet", lambda *args, **kwargs: source.copy())
    previous = set()
    for q, expected_day_av_count in ((0.0, 0), (0.125, 0), (0.375, 1), (0.5, 2), (0.625, 2), (0.875, 3), (1.0, 4)):
        scenario = build_controlled_fleet(ROOT, benchmark_start=START, simulation_end=END, requested_q_a=q)
        fleet = scenario.scenario_fleet
        current = set(fleet.loc[fleet["vehicle_type"].eq("AV"), "slot_id"])
        assert previous.issubset(current)
        assert len(current & {f"slot-{index}" for index in range(4)}) == expected_day_av_count
        for slot in ("slot-4", "slot-5"):
            uniform = int(controlled_fleet._slot_priority(slot, 20260824), 16) / (1 << 256)
            assert (slot in current) == (uniform < q)
        previous = current
    first = build_controlled_fleet(ROOT, benchmark_start=START, simulation_end=END, requested_q_a=0.375)
    source = source.iloc[::-1].reset_index(drop=True)
    second = build_controlled_fleet(ROOT, benchmark_start=START, simulation_end=END, requested_q_a=0.375)
    pd.testing.assert_frame_equal(first.scenario_fleet, second.scenario_fleet)
    assert first.accounting == second.accounting
    assert first.native_fixtures == second.native_fixtures


def test_controlled_endpoints_keep_every_source_window(frozen_scenarios):
    for q, expected_type in ((0.0, "HV"), (1.0, "AV")):
        scenario = frozen_scenarios[q]
        assert scenario.scenario_fleet["vehicle_type"].eq(expected_type).all()
        assert scenario.accounting["achieved_q_a"] == q
        assert scenario.accounting["vehicle_hour_error_pct"] == 0.0
        assert all(fixture.availability_policy == AVAILABILITY_POLICY for fixture in scenario.native_fixtures)
        assert all(fixture.av_source_session_end_inherited == (q == 1.0) for fixture in scenario.native_fixtures)
        for fixture, row in zip(scenario.native_fixtures, scenario.scenario_fleet.itertuples(index=False)):
            assert fixture.availability_start_time == row.availability_start_time
            assert fixture.availability_end_time == row.availability_end_time
        expected_hours_key = "hv_vehicle_hours" if q == 0.0 else "av_vehicle_hours"
        assert all(row[expected_hours_key] == row["baseline_vehicle_hours"] for row in scenario.accounting["hourly_supply"])
    with pytest.raises(controlled_fleet.FleetPyCompatibilityError):
        build_controlled_fleet(ROOT, benchmark_start=START, simulation_end=END, requested_q_a=float("nan"))
