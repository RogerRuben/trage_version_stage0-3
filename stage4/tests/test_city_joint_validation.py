from types import SimpleNamespace

import pandas as pd
import pytest

from stage4.analysis.city_joint_validation import time_chain_check, outcome_table


def test_environment_duration_time_chain_and_overlap():
    start = pd.Timestamp("2016-10-31T00:00:00+08:00")
    frame = pd.DataFrame([
        dict(order_id="one", native_vehicle_id=7, assignment_time=start,
             pickup_time=start+pd.Timedelta(seconds=10),
             service_end_time=start+pd.Timedelta(seconds=110),
             pickup_eta_s=10., realized_service_time_s=100.),
        dict(order_id="two", native_vehicle_id=7, assignment_time=start+pd.Timedelta(seconds=110),
             pickup_time=start+pd.Timedelta(seconds=120),
             service_end_time=start+pd.Timedelta(seconds=180),
             pickup_eta_s=10., realized_service_time_s=60.),
    ])
    assert time_chain_check(frame) == dict(duplicate_orders=0, vehicle_task_overlap=0, maximum_time_error_s=0.)
    frame.loc[1, "assignment_time"] = start+pd.Timedelta(seconds=109)
    with pytest.raises(RuntimeError, match="time-chain"):
        time_chain_check(frame)


def test_carry_in_is_separate_and_expired_orders_reconcile():
    start = pd.Timestamp("2016-10-31T00:00:00+08:00")
    request = lambda oid, release: SimpleNamespace(order_id=oid, sim_time_s=release,
        request_time=start+pd.Timedelta(seconds=release), profile_id="C", compatible_profiles={"HV"},
        passenger_accepts_av=False)
    control = SimpleNamespace(start=start, request_by_rid={1: request("old", 61190), 2: request("new", 61210)},
                              expired_rids={2})
    frame = pd.DataFrame([dict(native_request_id=1, pickup_time=start+pd.Timedelta(seconds=61220),
        service_end_time=start+pd.Timedelta(seconds=61300), completed=True, vehicle_type="HV")])
    outcome = outcome_table(control, frame, {"window_start_s":61200,"measurement_end_s":63000,"patience_s":300})
    assert outcome.cohort.tolist() == ["CARRY_IN", "NEW_WINDOW"]
    assert outcome.matched.tolist() == [True, False]
    assert outcome.expired.tolist() == [False, True]
    assert outcome.iloc[0].wait_s == 30.
