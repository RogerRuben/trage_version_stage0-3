from dataclasses import dataclass

import pandas as pd

from stage4.analysis.rt_patience_sensitivity import request_variants
from stage4.analysis.recover_rt_parameters import PARAMETERS


def test_rt_changes_only_release_and_derived_clock():
    @dataclass
    class Request:
        order_id: str
        request_time: pd.Timestamp
        sim_time_s: float
        pickup_lon_wgs84: float = 108.9
        pickup_lat_wgs84: float = 34.2
        dropoff_lon_wgs84: float = 108.91
        dropoff_lat_wgs84: float = 34.21
        rho_dynamic: float = 1.234
    start = pd.Timestamp('2016-10-31T00:00:00+08:00')
    request = Request('same-order', start+pd.Timedelta(hours=12), 43200.)
    stats = dict(gap_p25_sec=420., gap_p50_sec=748., gap_p75_sec=1733.)
    variants, summary = request_variants([request], stats, PARAMETERS, start)
    assert len(summary) == 3
    assert variants['CANONICAL_ZERO_LEAD'][0] is request
    for name, values in variants.items():
        result = values[0]
        assert {k: v for k, v in result.__dict__.items() if k not in ('request_time', 'sim_time_s')} == {
            k: v for k, v in request.__dict__.items() if k not in ('request_time', 'sim_time_s')}
        assert abs((result.request_time-start).total_seconds()-result.sim_time_s) < 1e-6
        if name != 'CANONICAL_ZERO_LEAD':
            assert result.sim_time_s < request.sim_time_s
    again, _ = request_variants([request], stats, PARAMETERS, start)
    assert variants == again
