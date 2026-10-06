import json
from pathlib import Path

import pandas as pd
import pytest

from stage4.analysis.replay_calibration_contract import FIXED, load_calibration
from stage4.analysis.replay_calibration_v1 import chain_pairs, stable_sample, witness_flags


def test_calibration_scope_is_one_fixed_condition(tmp_path):
    path=tmp_path/"config.json"
    path.write_text(json.dumps(FIXED))
    assert load_calibration(tmp_path,path)[0] == FIXED
    path.write_text(json.dumps({**FIXED,"patience_s":600}))
    with pytest.raises(ValueError,match="one-condition"):
        load_calibration(tmp_path,path)


def test_chain_uses_running_end_and_retains_missing_predecessor():
    start=pd.Timestamp("2016-10-31T08:00:00+08:00")
    frame=pd.DataFrame(dict(order_id=["a","b","c","d"],driver_id=["x"]*4,
        request_time=[start,start+pd.Timedelta(minutes=10),start+pd.Timedelta(minutes=16),start+pd.Timedelta(hours=3)],
        arrival_time=[start+pd.Timedelta(minutes=20),start+pd.Timedelta(minutes=15),start+pd.Timedelta(minutes=18),start+pd.Timedelta(hours=3,minutes=10)],
        start_lon_wgs84=[108.9]*4,start_lat_wgs84=[34.2]*4,end_lon_wgs84=[108.91]*4,end_lat_wgs84=[34.21]*4))
    got=chain_pairs(frame,5400).set_index("order_id")
    assert not got.loc["a","has_predecessor"]
    assert got.loc["c","inter_trip_gap_s"] == 60
    assert got.loc["c","history_overlap"] and not got.loc["c","chain_routing_eligible"]
    assert got.loc["d","session_break"] and got.loc["d","source_session_id"] == "x__S002"


def test_witness_is_not_actual_pickup_label_and_sampling_is_stable():
    flags=witness_flags(600,200,480)
    assert flags["timing_idle_hold_conflict"] and flags["beta_only_patience_conflict"]
    assert not flags["beta_only_gap_conflict"]
    assert not witness_flags(-1,200,480)["timing_idle_hold_conflict"]
    frame=pd.DataFrame({"order_id":["a","b","c","d"]})
    first=stable_sample(frame,2,20261006).order_id.tolist()
    assert first == stable_sample(frame.iloc[::-1],2,20261006).order_id.tolist()
