from types import SimpleNamespace

import pandas as pd

from stage4.analysis.frozen_state_prediction_ablation import _waiting_requests
from stage4.analysis.mechanism_validity import graph_metrics
from stage4.dispatch.rolling_or_control import patience_expired


def test_patience_membership_deadline_and_critical():
    start = pd.Timestamp('2016-10-31T00:00:00+08:00')
    requests = [SimpleNamespace(order_id=str(i), native_id=i, sim_time_s=t,
        request_time=start+pd.Timedelta(seconds=t)) for i,t in enumerate([0,120,121,150,301])]
    assignments = pd.DataFrame(columns=['simulation_time_s','order_id'])
    def ids(p):
        return {r.native_id for r,*_ in _waiting_requests(requests,assignments,300,p)}
    assert ids(180) == {2,3}  # exact deadline is expired
    assert ids(300) == {1,2,3}
    assert ids(600) == {0,1,2,3}
    assert _waiting_requests(requests,assignments,300) == _waiting_requests(requests,assignments,300,300)
    assert all(row[3] for row in _waiting_requests(requests,assignments,300,180))


def test_mixed_capacity_not_sum_of_subgraphs():
    assert graph_metrics([(1,10),(1,20)])['M'] == 1
    assert graph_metrics([(1,10)])['M'] + graph_metrics([(1,20)])['M'] == 2


def test_integer_production_deadline_differs_from_legacy_float_boundary():
    deadline = 300.4
    assert not (300 >= deadline)
    assert patience_expired(300, deadline)
