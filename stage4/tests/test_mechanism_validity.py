from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd

from stage4.analysis.mechanism_validity import graph_metrics, production_neutral_arcs
from stage4.dispatch.candidate_graph import SpatialVehicle


def test_sparse_matching_and_actual_neutral_session_branch():
    assert graph_metrics([(1, 1), (1, 2), (2, 1), (2, 2)]) == {'E': 4, 'U': 2, 'M': 2}
    assert graph_metrics([(1, 1), (2, 1)]) == {'E': 2, 'U': 2, 'M': 1}
    assert graph_metrics([]) == {'E': 0, 'U': 0, 'M': 0}
    @dataclass
    class Fixture:
        native_id: int
        vehicle_type: str
        availability_end_time: pd.Timestamp
    ts = pd.Timestamp('2016-10-31T12:00:00+08:00')
    fixture = Fixture(0, 'HV', ts + pd.Timedelta(seconds=60))
    request = SimpleNamespace(native_id=0, sim_time_s=0, request_time=ts,
        pickup_lon_wgs84=108.9, pickup_lat_wgs84=34.2, predicted_service_time_s=100.)
    adapter = SimpleNamespace(routing_time_s=0., estimate_many=lambda *args:
        {0: SimpleNamespace(corrected_pickup_eta_s=10.)})
    config = dict(search_radius_initial_m=2000, search_radius_step_m=1000,
                  search_radius_cap_m=8000, candidate_top_k=20)
    waiting = [(request, 0, False, False)]
    hv = [SpatialVehicle('same_id', 0, 'HV', 108.9, 34.2)]
    av = [SpatialVehicle('same_id', 0, 'AV', 108.9, 34.2)]
    assert len(production_neutral_arcs(hv, [fixture], waiting, ts, ts, config, adapter)) == 0
    assert len(production_neutral_arcs(av, [fixture], waiting, ts, ts, config, adapter)) == 1
