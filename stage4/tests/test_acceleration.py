"""Small grouped equivalence tests; no native scenario sweep."""
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from stage4.analysis.flexibility_small_instances import small_case
from stage4.analysis.symmetric_flexibility_full_day import acceleration_settings
from stage4.dispatch.flexibility_model import solve_dispatch, score_fixed_action
from stage4.dispatch.flexibility_native import TrainDemandForecast
from stage4.dispatch.deterministic_routing import ArcDeterministicValhallaAdapter, SINGLE_SOURCE_MATRIX
from stage4.tests.test_flexibility_native import config, templates
from stage4.tests.test_routing_determinism import BatchSensitiveActor, _vehicles

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "stage4/output/runtime_dependencies/highspy_1_12_0"


def objective_vector(problem, decision, policy):
    requests = {r.request_id: r for r in problem.waiting_requests}
    served = [requests[r] for _, r in decision.selected_pairs]
    eta = sum(a.pickup_eta_s for a in problem.current_pickups
              if (a.vehicle_id, a.request_id) in decision.selected_pairs)
    priority = (sum(r.critical for r in served), len(served), sum(r.carry_over for r in served))
    if policy == "LOOKAHEAD":
        return (priority[0], decision.expected_total_service_count, priority[1], priority[2], -eta)
    return (*priority, decision.expected_next_service_count, -eta)


@pytest.mark.parametrize("backend", ["SCIPY", "HIGHS_PERSISTENT"])
def test_continuous_recourse_preserves_lex_objectives_and_integral_recovery(backend):
    if backend == "HIGHS_PERSISTENT" and not RUNTIME.exists():
        pytest.skip("optional workspace solver runtime is not installed")
    cases = [small_case(name) for name in
             ("flexibility_reservation", "location_counterexample", "forecast_error_counterexample")]
    reserve, realized = cases[0]
    cases.append((replace(reserve, waiting_requests=tuple(replace(r,critical=True,carry_over=True)
                                                         for r in reserve.waiting_requests)), realized))
    location, _ = cases[1]
    cases.append((replace(reserve, scenarios=(replace(reserve.scenarios[0],probability=.75),
                                             replace(location.scenarios[0],probability=.25))), realized))
    for original, realized in cases:
        for policy in ("LOOKAHEAD", "SERVICE_PRESERVING_LOOKAHEAD"):
            baseline = solve_dispatch(original, policy)
            fast = replace(original, limits=replace(original.limits,
                recourse_mode="FLOW_RELAXED", solver_backend=backend, highspy_runtime_dir=str(RUNTIME),
                lock_current_face=True))
            result = solve_dispatch(fast, policy)
            assert objective_vector(original, result, policy) == pytest.approx(objective_vector(original, baseline, policy), abs=1e-6)
            assert result.integer_variable_count < result.variable_count
            for scenario in original.scenarios:
                pairs = [(v, r) for s, v, r in result.recourse_pairs if s == scenario.scenario_id]
                assert len({v for v, _ in pairs}) == len(pairs)
                assert len({r for _, r in pairs}) == len(pairs)
                assert not ({r for _, r in pairs} & {r for _, r in result.selected_pairs})
            slow_score = score_fixed_action(original, result.selected_pairs, realized)
            fast_score = score_fixed_action(fast, result.selected_pairs, realized)
            assert fast_score.expected_total_service_count == pytest.approx(slow_score.expected_total_service_count)


def test_indexed_forecast_is_exact_including_rng_and_boundary_windows():
    frame = templates()
    slow = TrainDemandForecast(frame, config(), 160)
    fast = TrainDemandForecast(frame, {**config(), "fast_forecast": True}, 160)
    for now in (0, 90, 100, 129, 159, 160, 500):
        assert fast.scenarios(now, "M", .7, 123) == slow.scenarios(now, "M", .7, 123)


def test_bounded_exact_cache_and_ordered_independent_worker_path(monkeypatch):
    adapter = ArcDeterministicValhallaAdapter(ROOT, actor=BatchSensitiveActor(),
        routing_mode=SINGLE_SOURCE_MATRIX, persistent_cache_size=2)
    timestamp = pd.Timestamp("2016-10-31T08:00:00+08:00")
    vehicle = _vehicles()[0]
    first = adapter.estimate_many([vehicle], 108.92, 34.22, timestamp)
    adapter.cache.clear()
    again = adapter.estimate_many([vehicle], 108.92, 34.22, timestamp + pd.Timedelta(seconds=30))
    assert first[1].valhalla_time_s == again[1].valhalla_time_s
    assert adapter.routing_queries == 1 and adapter.persistent_cache_hits == 1
    adapter.cache.clear()
    adapter.estimate_many([replace(vehicle, lon_wgs84=vehicle.lon_wgs84 + 1e-9)], 108.92, 34.22, timestamp)
    assert adapter.routing_queries == 2  # No rounded-coordinate cross-epoch alias.
    adapter.cache.clear()
    adapter.estimate_many([vehicle], 108.92, 34.22, timestamp + pd.Timedelta(minutes=1))
    assert adapter.routing_queries == 3 and len(adapter._persistent_cache) == 2
    from stage4.dispatch import routing_workers
    monkeypatch.setattr(routing_workers, "_ACTOR", BatchSensitiveActor())
    class InlinePool:
        def map(self, fn, items, **kwargs):
            return map(fn, items)
    parallel = ArcDeterministicValhallaAdapter(ROOT, actor=BatchSensitiveActor(), routing_mode=SINGLE_SOURCE_MATRIX)
    parallel.route_workers, parallel._executor = 2, InlinePool()
    found = parallel.estimate_many(_vehicles(), 108.92, 34.22, timestamp)
    assert [found[v].valhalla_time_s for v in (1, 2)] == [11., 11.]
    assert parallel.routing_queries == 2 and parallel.routing_failures == 0


def test_acceleration_config_cannot_override_scientific_parameters(tmp_path):
    spec = json.loads((ROOT / "stage4/config/symmetric_flexibility_acceleration_v1.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(spec))
    settings, _, limit = acceleration_settings(tmp_path, Path("config.json"))
    assert settings["route_workers"] == 2 and limit == 2048
    spec["settings"]["patience_s"] = 600
    (tmp_path / "config.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="technical acceleration"):
        acceleration_settings(tmp_path, Path("config.json"))
