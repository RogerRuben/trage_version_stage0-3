"""Focused engineering contracts, without native replay or production data."""
from collections import Counter, OrderedDict
from dataclasses import asdict
from math import ceil
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from stage4.dispatch import scheme_a_city_adapter as adapter
from stage4.dispatch import scheme_a_city_graph as graph
from stage4.dispatch import routing_v4
from stage4.dispatch.deterministic_routing import SINGLE_SOURCE_MATRIX
from stage4.dispatch.scheme_a_city_connections import SchemeACityConnectors, _IndexedJoinRouter
from stage4.dispatch.scheme_a_city_master import CurrentAction
from stage4.fleetpy_adapter.valhalla_time_adapter import CALIBRATION_REL
from stage4.tests.test_scheme_a_city_graph import point, task, certificate, cfg
from stage4.tests.test_scheme_a_city_connections import (
    control_and_evidence, tokens, task as evidence_task, ORIGIN, MIDNIGHT, JoinEvidence,
)


def test_coarse_index_views_and_pending_union_equal_original_filtered_full_scan():
    values = [task(i, (i % 20) * 30, (i % 11) * 200 - 1000, 200,
        profiles=frozenset({"HV", "C"} if i % 3 else {"HV"})) for i in range(120)]
    base = graph._task_index(tuple(values))
    pending = (values[5], values[5], task(1000, 0, 200, 300))
    for now in (0., 30., 60., 270.):
        window = graph.TaskWindow(base, now, 1800.)
        supplied = graph.combine_tasks(pending, window)
        literal = graph._task_index(tuple([*pending, *window]))
        optimized = graph._task_index(supplied)
        assert window.index.base is base
        assert optimized.duplicates == literal.duplicates
        for profile in ("HV", "C", "M"):
            for ready in (now + 30., 599.999999999, 600., 600.000000001):
                state = graph.RouteState(point(0), ready, 1700., profile)
                here = np.asarray(graph._xy(state.position))
                neighbors = literal.tree.query_ball_point(here, 2000.)
                rows = []
                for offset in neighbors:
                    value = literal.tasks[offset]
                    departure = float(ceil(max(ready, value.release_s) / 30) * 30)
                    if (value.request_id in {8, 9} or profile not in value.compatible_profiles
                            or (profile != "HV" and not value.passenger_accepts_av)
                            or departure >= 1700 or departure > value.deadline_s):
                        continue
                    rows.append((float(np.linalg.norm(literal.points[offset] - here)),
                        value.deadline_s, value.release_s, value.request_id, departure, value))
                rows.sort(key=lambda row: row[:4])
                expected = [(row[-2], row[-1]) for row in rows[:3]]
                chosen, spatial, eligible, _ = graph._candidate_tasks(
                    optimized, state, 2000., 3, 30, 1800., {8}, (9,))
                assert chosen == expected
                assert (spatial, eligible) == (len(neighbors), len(rows))
        state = graph.RouteState(point(0), now + 30., 1700., "C")
        old, _ = graph.build_restricted_chains("a", 1, "s", state,
            [*pending, *window], [], certificate, cfg(), lambda: None)
        new, _ = graph.build_restricted_chains("a", 1, "s", state,
            supplied, graph.prepare_sites([]), certificate, cfg(), lambda: None)
        assert old == new
    _, hit = graph._SPATIAL_ORDER_CACHE.ordered(base, point(0), 2000.)
    assert hit


def test_epoch_continuation_reuse_preserves_all_chain_columns_and_grid_boundaries(monkeypatch):
    settings = dict(planning_horizon_s=1800, admission_end_s=1800,
        coarse_reference_update_s=300, layout_graph_cpu_limit_s=60., solver_time_limit_s=10.,
        maximum_model_variables=20000, maximum_model_nonzeros=150000, reposition_max_moves=50)
    source = task(99, 0, 0, 0)
    actions = [CurrentAction(f"serve{i}", i, "SERVE", 99) for i in range(12)]
    states = {action.action_id: graph.RouteState(point(0), 61. + i, 2000. + i, "C",
        dict(kind="CUSTOMER", task=source)) for i, action in enumerate(actions)}
    scene = [task(-1, 150, 0, 100), task(-2, 500, 100, 200), task(-3, 950, 200, 300)]
    captured = []
    def master(first, chains, *args, **kwargs):
        captured.append([asdict(chain) for chain in chains])
        return dict(selected_chain_ids=(), runtime_s=0., model={})
    monkeypatch.setattr(adapter, "solve_city_master", master)
    class Connector:
        def __init__(self):
            self.timings = Counter()
            self.calls = 0
        def __call__(self, *args):
            self.calls += 1
            return certificate(*args)
    original_key = adapter._continuation_key
    monkeypatch.setattr(adapter, "_continuation_key", lambda *args: None)
    old_connector = Connector()
    old = adapter.build_and_solve(actions, states, {"s": scene}, {"s": 1.},
        old_connector, [], settings, 0., "CHAIN_DEFER")
    monkeypatch.setattr(adapter, "_continuation_key", original_key)
    new_connector = Connector()
    new = adapter.build_and_solve(actions, states, {"s": scene}, {"s": 1.},
        new_connector, [], settings, 0., "CHAIN_DEFER")
    assert captured[0] == captured[1]
    assert new_connector.calls < old_connector.calls
    assert new["restricted_graph"]["chain_builds_reused"] == 11
    assert old["restricted_graph"]["generated_columns"] == new["restricted_graph"]["generated_columns"]
    assert new["restricted_graph"]["continuation_cache_bytes"] <= 8 * 1024 * 1024
    # Sub-grid ready times coalesce only while their exact future grids agree.
    other = lambda ready: graph.RouteState(point(0), ready, 2100., "C", dict(kind="CUSTOMER", task=source))
    assert original_key(other(89.999999999), scene, (), 1800.) == original_key(other(90.), scene, (), 1800.)
    assert original_key(other(90.000000001), scene, (), 1800.) != original_key(other(90.), scene, (), 1800.)
    assert original_key(graph.RouteState(point(0), 90., 1800., "C",
        dict(kind="NATIVE_VEHICLE", native_vehicle_id=1)), scene, (), 1800.) is None


def test_future_hv_batch_uses_existing_ordered_raw_queue_and_retimes_beta(tmp_path, monkeypatch):
    calibration = tmp_path / CALIBRATION_REL
    calibration.parent.mkdir(parents=True)
    pd.DataFrame(dict(time_bin_index=range(96), selected_eta_multiplier=[1.] * 69 + [2.] * 27)).to_parquet(calibration)
    monkeypatch.setattr(routing_v4, "static_matrix_certificate", lambda root: dict(schema="UNIT_TEST_STATIC_CONTEXT"))
    class Actor:
        def __init__(self):
            self.calls = []
        def matrix(self, request):
            self.calls.append(request)
            return dict(sources_to_targets=[[dict(time=50. + target["lat"], distance=.3)
                for target in request["targets"]] for source in request["sources"]])
    def make():
        actor = Actor()
        eta = routing_v4.StaticRawRoutingAdapter(tmp_path, actor=actor,
            routing_mode=SINGLE_SOURCE_MATRIX, grouped_sources=True, persistent_cache_size=8)
        control, route, evidence = control_and_evidence(tmp_path)
        control.eta_adapter = route.eta_adapter = eta
        control.config = dict(scheme_a_fast_forecast_hv=True)
        templates = pd.DataFrame([dict(date="20161010", order_id=f"h{i}", compatible_C=False) for i in range(3)])
        connector = SchemeACityConnectors(control, route, NS(seam_evidence=evidence), templates,
            geometry_timestamp=MIDNIGHT)
        return connector, eta, actor
    targets = [evidence_task(f"h{i}", profiles=("HV",)) for i in range(3)]
    targets[1].pickup = (targets[0].pickup[0] + 1e-9, targets[0].pickup[1])  # Original rounded alias.
    targets[2].pickup = (108.902, 34.201)
    state = graph.RouteState(ORIGIN, 61200., 90000., "HV")
    queries = [(state, target, 61200., "HV") for target in targets]
    old, old_eta, old_actor = make()
    new, new_eta, new_actor = make()
    assert [old(*query) for query in queries] == new.evaluate_many(queries)
    assert len(new_actor.calls) < len(old_actor.calls)
    assert new.counts["HV_future_batch_answers_consumed"] == 3
    assert old(state, targets[0], 62100., "HV") == new(state, targets[0], 62100., "HV")
    targets[0].deadline_s = 62101.
    assert new.evaluate_many([(state, targets[0], 62100., "HV")])[0]["reason_codes"] == ["CONNECTOR_EXCEEDS_ORIGINAL_PICKUP_DEADLINE"]
    old_eta.close()
    new_eta.close()


def test_native_prefix_cache_is_complete_and_never_reuses_changed_direction(tmp_path, monkeypatch):
    control, route, evidence = control_and_evidence(tmp_path)
    control.request_by_rid[1] = NS(order_id="actual", sim_time_s=61190.)
    control.runtime_by_vid[2] = NS(native_vehicle=NS(pos=ORIGIN))
    evidence._private_identity = pa.Table.from_pylist(tokens("20161031", "actual"))
    evidence._private_ranges = dict(actual=(0, 12))
    incoming = [f"incoming{i}" for i in range(14)]
    evidence._idle_context = lambda *args: dict(point=ORIGIN,
        edge_uids=tuple(incoming), evidence_kind="ACTUAL_PREFIX")
    parsed = []
    def joined(self, previous, following, routed, tolerance):
        parsed.append(tuple(self.tokens[previous].resolved_stage3_edge_uid))
        return dict(supported=True, compatible_profiles=("C",), C_reason_codes=[])
    monkeypatch.setattr(JoinEvidence, "evaluate", joined)
    connector = SchemeACityConnectors(control, route, NS(seam_evidence=evidence),
        pd.DataFrame(columns=["date", "order_id", "compatible_C"]), geometry_timestamp=MIDNIGHT)
    target = evidence_task("actual", kind="ACTUAL_PENDING", date="20161031")
    state = NS(position=ORIGIN, context=dict(kind="NATIVE_VEHICLE", native_vehicle_id=2, timestamp_s=61200.))
    assert connector(state, target, 61200., "C")["supported"]
    assert connector(state, target, 61230., "C")["supported"]
    assert parsed == [tuple(incoming)]
    control.sim_time = state.context["timestamp_s"] = 61230.
    incoming[-1] = "different-directed-edge"
    assert connector(state, target, 61230., "C")["supported"]
    assert len(parsed) == 2 and parsed[-1][-1] == "different-directed-edge"
    state.context["timestamp_s"] = 61200.
    assert connector(state, target, 61230., "C")["reason_codes"] == ["NATIVE_DEPARTURE_CONTEXT_NOT_CURRENT"]
    assert all(isinstance(key, bytes) and len(key) == 32 for key in connector.certificate_cache)
    assert connector.certificate_bytes <= 32 * 1024 * 1024


def test_static_join_slices_reuse_frozen_inputs_but_virtual_maps_are_private():
    router = _IndexedJoinRouter.__new__(_IndexedJoinRouter)
    router.join_static_cache = OrderedDict()
    router.join_static_bytes = router.join_static_hits = router.join_static_misses = 0
    router.join_parse_hits = router.join_parse_misses = 0
    geometry = {uid: dict(uid=uid, geometry=[ORIGIN, (108.901, 34.2)],
        from_node=f"{uid}-first", to_node=f"{uid}-last") for uid in ("e0", "e1", "e2")}
    router.small_geometry = lambda uids: {uid: geometry[uid] for uid in uids}
    router._boundary = pa.Table.from_pylist([dict(stage3_edge_uid=uid,
        intersection_complex_uid="x", boundary_role=role)
        for uid, role in zip(geometry, ("INCOMING", "INTERNAL", "OUTGOING"))])
    router._movements = pa.Table.from_pylist([dict(intersection_complex_uid="x",
        incoming_stage3_edge_uid="e0", outgoing_stage3_edge_uid="e2", movement_legality_state="LEGAL")])
    router._controls = pa.Table.from_pylist([dict(intersection_complex_uid="x", signalized=True)])
    first = router.join_static_inputs(set(geometry))
    first[0]["virtual"] = dict(uid="virtual")
    first[-1]["virtual"] = (180., 180.)
    second = router.join_static_inputs(set(reversed(geometry)))
    assert "virtual" not in second[0] and "virtual" not in second[-1]
    assert second[1].equals(router._boundary.to_pandas())
    assert second[2].equals(router._movements.to_pandas())
    assert second[3].equals(router._controls.to_pandas())
    assert router.join_static_hits == router.join_static_misses == 1
    assert router.join_static_bytes <= 16 * 1024 * 1024
    frame = pd.DataFrame(tokens("20161010", "body", length=2))
    key = router.join_parse_key([("PREVIOUS", frame)], {}, ("FROZEN",))
    router.join_parse_put(key, dict(supported=True, reason_codes=[]))
    cached = router.join_parse_get(key)
    cached["reason_codes"].append("CALLER_MUTATION")
    assert router.join_parse_get(key)["reason_codes"] == []
    changed = frame.copy()
    changed.loc[0, "route_token_type"] = "UNRESOLVED"
    assert router.join_parse_get(router.join_parse_key([("PREVIOUS", changed)], {}, ("FROZEN",))) is None


def test_persistent_integer_tiers_equal_cold_solves_with_exact_locks_and_warm_start():
    from stage4.dispatch import scheme_a_city_master as city
    from stage4.dispatch.persistent_city_milp import BundledIntegerSession
    receipt = city.contract._solver_options_receipt()
    outcomes = []
    metrics = None
    for persistent in (False, True):
        rows = city._Rows(4, city._NonzeroLedger(1000))
        rows.add([(0, 1), (1, 1)], 0, 1)
        rows.add([(2, 1), (3, 1)], 0, 0)
        if persistent:
            rows._persistent_solver = BundledIntegerSession(receipt)
        budget = city._Budget(10.)
        first, binary, _ = city.contract._run_milp(np.array([-1., -1., 0., 0.]),
            rows, budget, receipt, "service")
        assert first.fun == -1.
        rows.add([(0, 1), (1, 1)], 1, 1, lock=True)
        final, binary, _ = city.contract._run_milp(np.array([4., 1., 0., 0.]),
            rows, budget, receipt, "distance")
        outcomes.append((final.fun, binary.tolist()))
        if persistent:
            metrics = rows._persistent_solver.diagnostics()
    assert outcomes == [(1., [0., 1., 0., 0.])] * 2
    assert metrics["model_loads"] == 1 and metrics["incremental_exact_rows"] == 1
    assert metrics["certified_binary_warm_starts"] == 1 and metrics["solver_runs"] == 2
    assert metrics["incumbent_fallback"] is False and metrics["objective_faces_use_epsilon"] is False


def test_bundled_run_warning_reads_model_status_and_keeps_timeout_failure():
    from stage4.dispatch import scheme_a_city_master as city
    from stage4.dispatch.persistent_city_milp import BundledIntegerSession
    receipt = city.contract._solver_options_receipt()
    for mode in ("optimal_warning", "time_warning", "api_error"):
        limited = mode == "time_warning"
        session = BundledIntegerSession(receipt)
        native, core, calls = session.h, session.core, []
        class WarningResult:
            def __getattr__(self, name):
                return getattr(native, name)
            def run(self):
                native.run()
                calls.append("run")
                return core.HighsStatus.kError if mode == "api_error" else core.HighsStatus.kWarning
            def getModelStatus(self):
                calls.append("model_status")
                return core.HighsModelStatus.kTimeLimit if limited else native.getModelStatus()
        session.h = WarningResult()
        rows = city._Rows(2, city._NonzeroLedger(100))
        rows.add([(0, 1), (1, 1)], 0, 1)
        rows._persistent_solver = session
        if mode == "api_error":
            with pytest.raises(RuntimeError) as error:
                city.contract._run_milp(np.array([-1., -1.]), rows, city._Budget(10.), receipt, "service")
            assert error.value.solver_diagnostics["solver_status"] == 4
            assert "kError" in error.value.solver_diagnostics["highs_run_status"]
        elif limited:
            with pytest.raises(city.contract.domain.DecisionTimeout) as error:
                city.contract._run_milp(np.array([-1., -1.]), rows, city._Budget(10.), receipt, "service")
            diag = error.value.solver_diagnostics
            assert diag["stage"] == "service" and diag["solver_status"] == 1
            assert "kWarning" in diag["highs_run_status"]
            assert "TimeLimit" in diag["highs_model_status"]
        else:
            result, binary, _ = city.contract._run_milp(
                np.array([-1., -1.]), rows, city._Budget(10.), receipt, "service")
            assert result.success and binary.sum() == 1
        assert calls == ["run", "model_status"]


def test_hv_native_auto_key_does_not_bypass_static_cache_or_bypass_c_direction():
    state = graph.RouteState(ORIGIN, 10., 1000., "HV",
        dict(kind="NATIVE_VEHICLE", native_vehicle_id=3, timestamp_s=10.))
    connector = SchemeACityConnectors.__new__(SchemeACityConnectors)
    key, reason = connector._certificate_source_key(state, profile="HV", hv_auto_od=True)
    assert reason is None and key == ("HV_AUTO_OD_ORIGIN", ORIGIN)
    assert connector._certificate_source_key(state, profile="HV") == (None, None)
    # No C token evidence is available: the shortcut cannot manufacture it.
    assert connector._certificate_source_key(state, profile="C", hv_auto_od=True) == (None, None)
