"""Four bounded construction checks, not a native simulation or performance Gate."""
from fractions import Fraction
from types import SimpleNamespace

import numpy as np

from stage4.dispatch.competitive_chain import (
    ExactPriceOracle, FiniteCatalogue, certified_upper, enumerate_paths, solve_bound_recovery,
)
from stage4.dispatch.scheme_a_city_graph import ChainTask
from stage4.dispatch.scheme_a_match_deadline_value import MatchDeadlineSceneDAG


def fixture():
    sites = [dict(site_id="S0", position=(108.95, 34.25)),
             dict(site_id="S1", position=(108.96, 34.25)),
             dict(site_id="S2", position=(109.03, 34.25))]
    tasks = [
        ChainTask(-1, 1., 301., 50., sites[0]["position"], sites[1]["position"],
                  frozenset({"HV", "C"}), True, "20161010", "a"),
        ChainTask(-2, 310., 610., 50., sites[1]["position"], sites[0]["position"],
                  frozenset({"HV", "C"}), True, "20161010", "b"),
        ChainTask(-3, 620., 920., 50., sites[0]["position"], sites[0]["position"],
                  frozenset({"HV"}), True, "20161010", "c"),
    ]
    model = MatchDeadlineSceneDAG(tasks, sites, 0., 1800.,
        SimpleNamespace(global_pace=.02, pace_by_slot={}), {"max_pickup_eta_s": 600}, lambda: None)
    return {"TRAIN_DAY_20161010": model}, {"TRAIN_DAY_20161010": Fraction(1)}, sites


def test_shared_integer_capacity_and_common_placement():
    models, weights, sites = fixture()
    catalogue = FiniteCatalogue(models, weights, {("HV", 0, 0, 6): 1}, {(0, 6): 2},
                                [s["site_id"] for s in sites])
    vector, info = catalogue.solve_integer({(0, 6): np.array([2, 0, 0])}, 2.)
    checked = catalogue.validate(vector)
    assert checked[2] <= 3 and info["integer_witness_verified"]
    assert int(catalogue.deployment(vector)[(0, 6)].sum()) == 2
    counts = {}
    for index, (root, path) in enumerate(catalogue.columns, len(catalogue.placement)):
        if vector[index]:
            for node in path:
                counts[root.scene, node] = counts.get((root.scene, node), 0)+1
    assert max(counts.values()) == 1


def test_exact_price_upper_includes_unassigned_roots():
    models, weights, sites = fixture()
    model = next(iter(models.values()))
    model.set_prices(np.array([.2, .4, .7]), {})
    oracle = ExactPriceOracle(model)
    for profile in ("C", "HV"):
        for zone in range(3):
            key = profile, zone, 0, 6
            paths = enumerate_paths(model, key)
            brute = max([Fraction(0), *(len(path)-sum(oracle.prices[n] for n in path)
                                      for path in paths)])
            assert oracle.price(key) == brute
    certificate = certified_upper(models, weights, {}, {(0, 6): 1},
                                   [s["site_id"] for s in sites])
    assert certificate["priced_root_count"] == 3
    catalogue = FiniteCatalogue(models, weights, {}, {(0, 6): 1}, [s["site_id"] for s in sites])
    _, reference = catalogue.solve_integer(time_limit_s=2.)
    assert Fraction(*reference["service_exact"]) <= certificate["value"]


def test_recovery_preserves_feasible_seed_and_supply():
    models, weights, sites = fixture()
    catalogue = FiniteCatalogue(models, weights, {}, {(0, 6): 1}, [s["site_id"] for s in sites])
    deployment, vector, info = solve_bound_recovery(models, weights, {}, {(0, 6): 1},
        [s["site_id"] for s in sites],
        {"coarse_value_max_pricing_rounds": 1, "coarse_value_max_columns": 20000,
         "coarse_value_max_nonzeros": 150000}, catalogue)
    assert info["recovered_service_scaled"] >= info["seed_service_scaled"]
    assert int(deployment[(0, 6)].sum()) == 1
    assert catalogue.validate(vector)[2] <= Fraction(*info["upper_bound_exact"])


def test_time_contract_and_nested_chain_domain():
    models, _, _ = fixture()
    model = next(iter(models.values()))
    assert model.dispatch_slot[0] == 300
    assert model.root_finish[1][0] == 300 + model.root_eta[1][0] + 50
    for zone in range(3):
        c = set(enumerate_paths(model, ("C", zone, 0, 6)))
        hv = set(enumerate_paths(model, ("HV", zone, 0, 6)))
        assert c <= hv
        assert len(c) >= len(enumerate_paths(model, ("C", zone, 1, 6)))
