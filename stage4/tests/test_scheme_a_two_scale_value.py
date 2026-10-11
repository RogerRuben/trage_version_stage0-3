"""Small mathematical checks of the coarse shared-capacity model, not replay."""
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np
import pytest

from stage4.dispatch.scheme_a_city_graph import ChainTask
from stage4.dispatch.scheme_a_two_scale_value import _SceneDAG


POINT = (108.9,34.2)


def task(identity,release,point=POINT,*,profiles=("HV","C"),accepts=True):
    return ChainTask(-identity,release,release+300.,60.,point,point,
        frozenset(profiles),accepts,"20161010",f"historical_{identity}")


def model(tasks,*,now=0,pace=None):
    return _SceneDAG(tasks,[dict(position=POINT)],now,now+1800,
        pace or NS(global_pace=.1,pace_by_slot={}),{},lambda:None)


def test_shared_customer_prices_remove_independent_future_capacity_and_price_zero_supply_state():
    dag=model([task(1,10.)])
    early=("HV",0,0,6)
    late=("HV",0,1,6)
    assert dag.price(early)[0] == 1.
    assert dag.price(late)[0] == 1.
    # Two states each independently claim the same future service, but their
    # shared sparse LP can allocate it only once. Scarcity prices reflect that.
    limit=perf_counter()+5.
    dag.solve_competition({early:2,late:2},lambda:None,lambda:limit-perf_counter(),{})
    assert dag.info["available"] and dag.info["pricing_closed"]
    assert dag.info["lp_service_value"] == pytest.approx(1.)
    assert dag.alpha[0] == pytest.approx(1.)
    assert dag.price(early)[0] == dag.price(late)[0] == pytest.approx(0.)
    unseen=("C",0,0,6)
    assert unseen not in dag.beta and dag.price(unseen)[0] == pytest.approx(0.)


def test_fixed_prices_value_is_ready_time_monotone_even_when_later_pace_is_faster():
    # Earlier-ready resources can wait for later departure slots. Querying
    # the same priced state later cannot increase its feasible continuation.
    remote=(POINT[0]+1000/(111320*np.cos(np.deg2rad(34.25))),POINT[1])
    dag=model([task(1,1850.,remote)],now=1200,
              pace=NS(global_pace=.1,pace_by_slot={0:1.,1:.1}))
    departure=dag._first_feasible_departure(np.array([1700.]),np.array([1000.]),
                                            np.array([2100.]),3000.)
    assert departure[0] == 1800.
    values=[dag.price(("HV",0,ready,6))[0] for ready in range(6)]
    assert values[:3] == [1.,1.,1.]
    assert all(first >= second for first,second in zip(values,values[1:]))
    assert np.all((dag.successors["HV"] >= -1))


def test_coarse_dag_masks_and_predicted_time_chain_and_future_date_boundary():
    dag=model([task(1,10.),task(2,310.),task(3,610.),
               task(4,910.,profiles=("HV",)),task(5,1210.,accepts=False)])
    value,path=dag.price(("C",0,0,6))
    assert value == 3. and len(path)==len(set(path))==3
    assert all(dag.allowed["C"][node] for node in path)
    assert all(dag.finish[first] <= dag.pickup_slot[second]
               for first,second in zip(path,path[1:]))
    assert not dag.allowed["C"][3] and not dag.allowed["C"][4]
    invalid=ChainTask(-9,10.,310.,60.,POINT,POINT,frozenset({"HV"}),
                      True,"20161031","future")
    with pytest.raises(ValueError,match="earlier-history"):
        model([invalid])


def test_nested_profile_successor_domain_prevents_false_hv_value_disadvantage():
    def point(x):
        return (108.95 + x/(111320*np.cos(np.deg2rad(34.25))), 34.25)
    def historical(rid, release, duration, pickup, dropoff, c=True):
        return ChainTask(rid, release, release+300, duration, point(pickup), point(dropoff),
            frozenset({"HV", "C"} if c else {"HV"}), True, "20161010", str(rid))
    tasks = [historical(-1, 1, 100, 0, 2500), historical(-2, 601, 100, 2550, 4000),
        historical(-3, 1201, 100, 4050, 4200),
        *[historical(-i, 601, 1000, 2470+10*i, 2470+10*i, False) for i in (4, 5, 6)]]
    keys = {p: (p, 0, 1, 6) for p in ("HV", "C")}
    def scene(nested):
        return _SceneDAG(tasks, [dict(position=point(0))], 0, 1800,
            NS(global_pace=.1, pace_by_slot={0:.1}),
            dict(coarse_nested_profile_paths=nested), lambda: None)
    old, new = scene(False), scene(True)
    assert [old.price(keys[p])[0] for p in ("HV", "C")] == [2., 3.]
    assert [new.price(keys[p])[0] for p in ("HV", "C")] == [3., 3.]
    alpha = np.array([2., 0., 0., 0., 0., 0.])
    old.set_prices(alpha, {})
    new.set_prices(alpha, {})
    assert [old.price(keys[p])[0] for p in ("HV", "C")] == [0., 1.]
    assert [new.price(keys[p])[0] for p in ("HV", "C")] == [1., 1.]
    assert new.successors["HV"].shape == (6, 6)
    for hv, c in zip(new.successors["HV"], new.successors["C"]):
        assert set(c[c >= 0]) <= set(hv[hv >= 0])
