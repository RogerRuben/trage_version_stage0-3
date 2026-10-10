"""Two focused mocked native contracts; no replay/Valhalla or private data."""
from types import SimpleNamespace as NS

import pytest

from stage4.dispatch import scheme_a_city_adapter as old_adapter
from stage4.dispatch import scheme_a_two_scale_native as native
from stage4.dispatch.scheme_a_current_flow import solve_current_flow
from stage4.dispatch.solver import AssignmentArc


ORIGIN = (108.9, 34.2)
MOVING_TARGET = (108.905, 34.2)
DROPOFF = (108.91, 34.2)


def fixture():
    manager = NS(active={1: 0}, rows=[dict(destination_position=MOVING_TARGET,
        start_time_s=900., planned_duration_s=240.)], queued=[])
    manager.queue_actions = lambda moves, now: manager.queued.append((moves, now))
    runtimes = {vid: NS(fixture=NS(vehicle_type="AV" if vid == 1 else "HV"),
                        native_vehicle=NS(pos=ORIGIN), active_order_id=23 if vid == 3 else None)
                for vid in (1, 2, 3, 4)}
    request = NS(order_id="current", sim_time_s=900., predicted_service_time_s=120.,
                 pickup_lon_wgs84=108.901, pickup_lat_wgs84=34.2,
                 dropoff_lon_wgs84=DROPOFF[0], dropoff_lat_wgs84=DROPOFF[1],
                 compatible_profiles=("C",), passenger_accepts_av=True)
    class BookedOnly(dict):
        def __getitem__(self, key):
            if key.startswith(("realized", "observed")) or key == "service_end_time":
                raise AssertionError("future/realized information was read")
            return super().__getitem__(key)
    booked = BookedOnly(simulation_time_s=600., native_vehicle_id=3, native_request_id=23,
                        pickup_eta_s=30., predicted_service_time_s=900.)
    c = NS(gammas=dict(static=None, dynamic=None, speed=None), cost_level_enabled=False,
           assignment_rows=[booked], runtime_by_vid=runtimes,
           config=dict(profile_id="C"), fixture_windows_s={1: (0., 5000.), 2: (0., 5000.),
               3: (0., 5000.), 4: (1500., 5000.)}, request_by_rid={11: request},
           request_meta={11: dict(pickup_deadline_s=1200.)}, repositioning_manager=manager,
           routing_engine=NS(return_position_coordinates=lambda point: point),
           event_calendar=NS(active_ids=lambda now: (1, 2, 3)))
    c._available = lambda runtime, now: runtime in (runtimes[1], runtimes[2])
    cfg = dict(test_date="20161031", planning_horizon_s=1800, solver_time_limit_s=10.,
               layout_candidate_site_count=20, movement_day_end_s=86400)
    return c, cfg


class Value:
    def __init__(self):
        self.refreshes, self.lookups = [], []
        self.available = True

    def refresh(self, now, supply):
        self.refreshes.append((now, supply))
        return dict(available=self.available, status="AVAILABLE" if self.available else "PRICE_NOT_CLOSED",
                    refreshed=True)

    def value(self, state):
        self.lookups.append(state)
        return .9 if state.position == DROPOFF else .5

    def diagnostics(self):
        return dict(refreshes=len(self.refreshes))


def test_current_serve_uses_predicted_after_state_moving_wait_and_only_coarse_supply(monkeypatch):
    c, cfg = fixture()
    value, prediction_calls, flow_inputs = Value(), [], []
    def predict(control, now, horizon, last, remaining, diagnostics):
        prediction_calls.append(now)
        assert set(last[3]) == set(native._BOOKED_FIELDS)
        return [NS(vehicle_id=vid, profile_id="C" if vid == 1 else "HV",
                   ready_position=f"{ORIGIN[0]},{ORIGIN[1]}", ready_time_s=now if vid != 4 else 1500.,
                   availability_end_s=5000.) for vid in (1, 2, 3, 4)]
    def execute(actions, units, **kwargs):
        flow_inputs.append((actions, units, kwargs))
        return solve_current_flow(actions, units, backend="INTEGER_REFERENCE", **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("runtime future enumeration/connector is forbidden")
    monkeypatch.setattr(native, "predicted_vehicle_states", predict)
    monkeypatch.setattr(native, "solve_current_flow", execute)
    monkeypatch.setattr(old_adapter, "build_and_solve", forbidden)
    validator_calls = []
    def validator(control, arcs, now):
        validator_calls.append(now)
        assert control.runtime_by_vid[1].native_vehicle.pos == ORIGIN
        return arcs
    adapter = native.NativeTwoScaleAdapter("CHAIN_DEFER", value, None, forbidden,
                                           validator, None, cfg)
    arc = AssignmentArc(1, 11, 60., False, True,
                        (c.runtime_by_vid[1], c.request_by_rid[11], NS(route_distance_m=400.)), "AV")
    for now in (930., 960.):
        result = adapter.solve(c, [arc], [11], now)
        assert result.selected_indices == (0,) and result.total_matched == 1
        actions, units, kwargs = flow_inputs[-1]
        assert {action.vehicle_id for action in actions} == {1}  # No BUSY/unborn current columns.
        assert {action.kind for action in actions} == {"WAIT", "SERVE"}
        assert units["V1:WAIT"] == 0 and units["V1:SERVE:11"] == 1
        assert kwargs["policy"] == "CHAIN_DEFER"
    assert prediction_calls == [930.] and len(value.refreshes) == 1
    assert value.refreshes[0][1][0].position == MOVING_TARGET
    assert value.refreshes[0][1][0].ready_s == 1140.
    assert any(state.position == MOVING_TARGET and state.ready_s == 1140. for state in value.lookups)
    assert any(state.position == DROPOFF and state.ready_s == 1110. for state in value.lookups)
    assert any(state.position == DROPOFF and state.ready_s == 1140. for state in value.lookups)
    assert validator_calls == [930., 960.]
    assert adapter.rows[-1]["coarse_time_s"] == 0.
    assert adapter.rows[-1]["future_chain_variables"] == adapter.rows[-1]["future_connector_queries"] == 0
    assert c.repositioning_manager.active == {1: 0}  # WAIT/selection never teleports or cancels.
    assert adapter.solve(c, [], [], 990.).selected_indices == ()
    assert prediction_calls == [930.]
    value.available = False
    adapter.solve(c, [arc], [11], 1230.)
    assert adapter.rows[-1]["effective_policy"] == "SERVICE_PRESERVING"
    assert adapter.rows[-1]["continuation_value_unavailable_reason"] == "PRICE_NOT_CLOSED"
    assert all(unit == 0 for unit in flow_inputs[-1][1].values())
    c.gammas["static"] = .2
    with pytest.raises(ValueError, match="Gamma"):
        adapter.solve(c, [arc], [11], 1260.)


def test_only_present_top_three_moves_are_routed_then_queued_and_no_choice_skips_coarse(monkeypatch):
    c, cfg = fixture()
    c.repositioning_manager.active.clear()
    value, queries, prediction_calls = Value(), [], []
    value.value = lambda state: 0. if state.position == ORIGIN else 3.
    def predict(*args):
        prediction_calls.append(args[1])
        return []
    def connector(state, target, now, profile):
        queries.append((state, target, now, profile))
        assert now == 900. and state.ready_s == now and target["require_geometry"] is True
        assert profile in ("C", "HV") and state.context["kind"] == "NATIVE_VEHICLE"
        routed = dict(supported=True, origin_wgs84=state.position, target_wgs84=target["position"],
                      departure_s=now, duration_s=100., distance_m=400.)
        return dict(supported=True, travel_time_s=100., empty_distance_m=400., routed=routed,
                    arrival_context=dict(kind="EMPTY_ARRIVAL", point=target["position"], edge_uids=("edge",)))
    sites = [dict(site_id=f"site{i}", node_id=i, position=(108.9 + (i + 1) * .001, 34.2))
             for i in range(4)]
    monkeypatch.setattr(native, "sites_for", lambda *args: sites)
    monkeypatch.setattr(native, "predicted_vehicle_states", predict)
    monkeypatch.setattr(native, "solve_current_flow", lambda actions, units, **kwargs:
        solve_current_flow(actions, units, backend="INTEGER_REFERENCE", **kwargs))
    adapter = native.NativeTwoScaleAdapter("CHAIN_DEFER", value, None, connector,
                                           lambda *args: [], None, cfg)
    result = adapter.solve(c, [], [], 900.)
    assert result.selected_indices == ()
    assert len(queries) == 6  # Two current vehicles, three current sites; no future query.
    queued, when = c.repositioning_manager.queued[-1]
    assert when == 900. and {move["native_vehicle_id"] for move in queued} == {1, 2}
    assert all(move["routed"]["departure_s"] == 900. for move in queued)
    assert adapter.rows[-1]["current_move_connector_queries"] == 6
    assert adapter.rows[-1]["current_move_count"] == 2
    assert adapter.rows[-1]["future_connector_queries"] == 0
    assert prediction_calls == [900.]
    adapter.solve(c, [], [], 930.)
    assert prediction_calls == [900.] and len(queries) == 6
    assert adapter.rows[-1]["opt_status"] == "FIXED_CURRENT_WAIT"
    assert adapter.progress_summary()["timings_s"]["solve_wall_time_s"] >= 0.
    c.cost_level_enabled = True
    with pytest.raises(ValueError, match="platform-cost"):
        adapter.solve(c, [], [], 960.)


def test_move_value_upper_bound_screens_before_route_and_actual_arrival_screens_after_route(monkeypatch):
    c, cfg = fixture()
    c.repositioning_manager.active.clear()
    value, queries, flow_actions = Value(), [], []
    sites = [dict(site_id=f"site{i}", node_id=i, position=(108.9 + (i + 1) * .001, 34.2))
             for i in range(3)]
    def lookup(state):
        if state.position == ORIGIN:
            return .5  # rounded WAIT value = 2 units.
        site = next(i for i, item in enumerate(sites) if item["position"] == state.position)
        return (.66, .95, 1.5)[site] if state.ready_s == 900. else (.5, .5, 1.2)[site]
    value.value = lookup
    def connector(state, target, now, profile):
        assert value.refreshes  # No optional route is called before fixed prices.
        assert target["site_id"] != "site0"  # Its optimistic rounded delta is zero.
        queries.append(target["site_id"])
        return dict(supported=True, travel_time_s=100., empty_distance_m=400.,
                    routed=dict(supported=True, departure_s=now),
                    arrival_context=dict(kind="EMPTY_ARRIVAL", point=target["position"], edge_uids=("edge",)))
    def execute(actions, units, **kwargs):
        flow_actions.extend(actions)
        return solve_current_flow(actions, units, backend="INTEGER_REFERENCE", **kwargs)
    monkeypatch.setattr(native, "sites_for", lambda *args: sites)
    monkeypatch.setattr(native, "predicted_vehicle_states", lambda *args: [])
    monkeypatch.setattr(native, "solve_current_flow", execute)
    adapter = native.NativeTwoScaleAdapter("CHAIN_DEFER", value, None, connector,
                                           lambda *args: [], None, cfg)
    adapter.solve(c, [], [], 900.)
    row = adapter.rows[-1]
    assert row["current_move_proposed_count"] == 6
    assert row["current_move_upper_screened_count"] == 2
    assert row["current_move_routed_count"] == 4
    assert row["current_move_arrival_screened_count"] == 2
    assert not row["optional_move_budget_cutoff"]
    assert queries == ["site1", "site2", "site1", "site2"]
    assert all(action.kind == "WAIT" or action.payload["move"]["site_id"] == "site2"
               for action in flow_actions)
    assert {move["site_id"] for move in c.repositioning_manager.queued[-1][0]} == {"site2"}


def test_unavailable_price_and_optional_route_budget_cutoff_keep_validated_current_serve(monkeypatch):
    from stage4.dispatch import scheme_a_current_flow as kernel
    c, cfg = fixture()
    c.repositioning_manager.active.clear()
    value, queries, flows = Value(), [], []
    sites = [dict(site_id=f"site{i}", node_id=i, position=(108.9 + (i + 1) * .001, 34.2))
             for i in range(3)]
    clock = [1000.]
    monkeypatch.setattr(native, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(kernel, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(native, "sites_for", lambda *args: sites)
    monkeypatch.setattr(native, "predicted_vehicle_states", lambda *args: [])
    arc = AssignmentArc(1, 11, 60., False, True,
                        (c.runtime_by_vid[1], c.request_by_rid[11], NS(route_distance_m=400.)), "AV")
    def execute(actions, units, **kwargs):
        flows.append((actions, units, kwargs))
        assert all(action.kind != "RELOCATE" for action in actions)
        assert not any(units.values()) and kwargs["policy"] == "SERVICE_PRESERVING"
        return solve_current_flow(actions, units, backend="INTEGER_REFERENCE", **kwargs)
    def connector(state, target, now, profile):
        queries.append(target["site_id"])
        clock[0] += 11.  # An uninterruptible current route overruns adapter wall budget.
        return dict(supported=True, travel_time_s=100., empty_distance_m=400.,
                    routed=dict(supported=True, departure_s=now),
                    arrival_context=dict(kind="EMPTY_ARRIVAL", point=target["position"], edge_uids=("edge",)))
    monkeypatch.setattr(native, "solve_current_flow", execute)
    value.available = False
    adapter = native.NativeTwoScaleAdapter("CHAIN_DEFER", value, None, connector,
                                           lambda control, arcs, now: arcs, None, cfg)
    assert adapter.solve(c, [arc], [11], 900.).selected_indices == (0,)
    row = adapter.rows[-1]
    assert queries == [] and row["current_move_value_unavailable_skipped_count"] == 6
    assert row["optional_move_skip_reason"] == "PRICE_NOT_CLOSED"
    value.available = True
    value.value = lambda state: 0. if state.position == ORIGIN else 3.
    adapter = native.NativeTwoScaleAdapter("CHAIN_DEFER", value, None, connector,
                                           lambda control, arcs, now: arcs, None, cfg)
    assert adapter.solve(c, [arc], [11], 900.).selected_indices == (0,)
    row = adapter.rows[-1]
    assert queries == ["site0"] and row["current_move_routed_count"] == 1
    assert row["current_move_budget_skipped_count"] == 5 and row["optional_move_budget_cutoff"]
    assert row["opt_status"] == "FEASIBLE_BACKUP_SERVICE_FIRST"
    assert row["current_flow_fallback_reason"] == "CURRENT_WALL_DEADLINE"
    assert row["adapter_wall_deadline_exceeded"] and row["current_selected"] == 1
    assert c.repositioning_manager.queued[-1][0] == []
