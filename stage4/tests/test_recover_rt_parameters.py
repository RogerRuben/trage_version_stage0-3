import argparse

import pandas as pd

from stage4.analysis.recover_rt_parameters import endpoint_frame, fingerprint, PARAMETERS
from stage4.scripts import build_decoupled_abm_environment as original


def test_all_order_endpoint_chain_and_original_rt_transform(tmp_path):
    path = tmp_path / 'stage0/work_v6_final/candidate_manifests/date=20161023.parquet'
    path.parent.mkdir(parents=True)
    base = int(pd.Timestamp('2016-10-23T12:00:00Z').timestamp())
    pd.DataFrame(dict(order_id=['a', 'b', 'c'], driver_id=['d']*3,
        point_count=[2]*3, valid_point_count=[2]*3,
        start_time=[base, base+160, base+380], end_time=[base+100, base+260, base+480],
        start_lon=[108.9]*3, start_lat=[34.2]*3, end_lon=[108.9]*3, end_lat=[34.2]*3)).to_parquet(path)
    frame, source = endpoint_frame(tmp_path, '20161023')
    assert source['orders'] == 3  # two-point orders retained, no Stage1 quality filter
    spec = original.make_zone_spec([frame], .02)
    _, stats = original.build_training_chain_stats({'20161023': frame}, spec)
    assert stats['chain_rows'] == 3 and stats['feasible_chain_rows'] == 2
    assert stats['gap_p25_sec'] == 75 and stats['gap_p50_sec'] == 90
    tables = original.attach_request_times(frame, stats, argparse.Namespace(**PARAMETERS))
    summary = fingerprint(tables)
    assert summary.orders.tolist() == [3, 3, 3]
    assert set(summary.scenario) == {'RT-Low', 'RT-Base', 'RT-High'}
    assert all(f.simulated_request_time.lt(f.observed_boarding_time).all() for f in tables.values())
