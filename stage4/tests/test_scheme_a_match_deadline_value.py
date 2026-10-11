"""Three small mathematical time-chain checks; no native replay or source data."""
from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np
import pytest

from stage4.dispatch.scheme_a_city_graph import ChainTask
from stage4.dispatch.scheme_a_match_deadline_value import MatchDeadlineSceneDAG


def point(x):
    return (108.95 + x / (111320 * np.cos(np.deg2rad(34.25))), 34.25)


def task(rid, release, pickup=0, dropoff=0, duration=60, c=True):
    return ChainTask(rid, release, release + 300, duration, point(pickup), point(dropoff),
        frozenset({"HV", "C"} if c else {"HV"}), True, "20161010", str(rid))


def scene(tasks, pace=.5):
    return MatchDeadlineSceneDAG(tasks, [dict(position=point(0))], 0, 1800,
        NS(global_pace=pace, pace_by_slot={0: pace}), {}, lambda: None)


def test_timely_match_can_arrive_after_match_deadline_but_pickup_above_600_is_rejected():
    dag = scene([task(-1, 10, 1000, 1000), task(-2, 10, 1230, 1230)])
    key = ("C", 0, 0, 2)  # Stop admitting at 600, finish committed work later.
    assert dag.dispatch_slot[0] == 300 < dag.deadline[0]
    offsets = {int(node): offset for offset, node in enumerate(dag.site_neighbors[0][0])}
    eta = dag.root_eta[0][offsets[0]]
    assert eta == pytest.approx(500.)
    assert dag.dispatch_slot[0] + eta > dag.deadline[0]
    assert dag.root_finish[0][offsets[0]] == pytest.approx(860.)
    assert dag.root_finish[0][offsets[0]] > 600.
    assert dag.root_eta[0][offsets[1]] > 600.
    assert dag.entries(key).tolist() == [0]
    assert dag.price(key) == (1., (0,))
    assert dag.price(("C", 0, 0, 1)) == (0., ())  # Dispatch exactly at shift end is forbidden.
    long_last = scene([task(-1, 10, 1000, 1000, duration=2000)])
    assert long_last.price(key) == (1., (0,))
    assert long_last.root_finish[0][0] > long_last.horizon
    assert long_last.root_ready_bin[0][0] == 6


def test_second_service_readiness_includes_predecessor_pickup_eta_not_just_m3_service():
    dag = scene([task(-1, 10, 800, 800), task(-2, 310, 800, 800), task(-3, 610, 800, 800)])
    # A is dispatched at 300, reaches the customer at 700, finishes at 760,
    # and becomes coarsely available at 900. B's dispatch at 600 is too early;
    # C at 900 remains feasible. Omitting pickup would falsely allow A-B-C.
    assert dag.finish[0] == 360.  # Explicit ZERO-pickup diagnostic lower bound.
    root_offset = list(dag.site_neighbors[0][0]).index(0)
    assert dag.root_finish[0][root_offset] == pytest.approx(760.)
    assert dag.root_ready_bin[0][root_offset] == 3
    assert dag.price(("C", 0, 0, 6)) == (2., (0, 2))
    successors = dag.successors["C"][0]
    column = list(successors).index(2)
    assert dag.successor_finish["C"][0, column] == pytest.approx(960.)
    assert dag.successor_ready_bin["C"][0, column] == 4
    assert all(values.shape == (4, 7, 3) for values in dag.dp_cache[("C", 6)])
    assert dag.resident_extra_array_bytes() > 0
    # Positive pickup on the SECOND service must also delay its readiness.
    second = scene([task(-1, 10, 0, 800), task(-2, 310, 1500, 1500), task(-3, 610, 1500, 1500)])
    assert second.price(("C", 0, 0, 6)) == (2., (0, 1))
    column = list(second.successors["C"][0]).index(1)
    assert second.successor_eta["C"][0, column] == pytest.approx(350.)
    assert second.successor_finish["C"][0, column] == pytest.approx(1010.)
    assert second.successor_ready_bin["C"][0, column] == 4
    values, _ = second.dp_cache[("C", 6)]
    assert values[1, 3, 1] == 1.  # Would serve C if B were ready at 900.
    assert values[1, 4, 1] == 0.  # Actual B readiness is 1200, after C dispatch.


def test_common_customer_capacity_nested_profiles_and_early_ready_monotonicity():
    tasks = [task(-1, 1, 0, 2500, 100), task(-2, 601, 2550, 4000, 100),
        task(-3, 1201, 4050, 4200, 100),
        *[task(-i, 601, 2470 + 10*i, 2470 + 10*i, 1000, False) for i in (4, 5, 6)]]
    dag = scene(tasks, pace=.1)
    assert dag.price(("HV", 0, 1, 6))[0] == dag.price(("C", 0, 1, 6))[0] == 3.
    dag.set_prices(np.asarray([2., 0., 0., 0., 0., 0.]), {})
    assert dag.price(("HV", 0, 1, 6))[0] == dag.price(("C", 0, 1, 6))[0] == 1.
    for hv, c in zip(dag.successors["HV"], dag.successors["C"]):
        assert set(c[c >= 0]) <= set(hv[hv >= 0])
    for profile in ("HV", "C"):
        values = [dag.price((profile, 0, ready, 6))[0] for ready in range(7)]
        assert all(early >= late for early, late in zip(values, values[1:]))
        for ready in range(7):
            value, path = dag.price((profile, 0, ready, 6))
            assert len(path) == len(set(path)) <= 3
    shared = scene([task(-1, 10)])
    hv, c = ("HV", 0, 0, 6), ("C", 0, 0, 6)
    limit = perf_counter() + 5.
    shared.solve_competition({hv: 2, c: 2}, lambda: None, lambda: limit - perf_counter(), {})
    assert shared.info["available"] and shared.info["pricing_closed"]
    assert shared.info["lp_service_value"] == pytest.approx(1.)
    assert shared.price(hv)[0] == shared.price(c)[0] == pytest.approx(0.)
    assert shared.info["completion_depends_on_predecessor_pickup_eta"] is True
