"""Connect a research-world episode to the pinned FleetPy lifecycle.

No event loop or routing Actor is implemented here. Use the existing native
controller/vehicles/simulation and install these per-instance hooks. Midnight
remains the timestamp anchor; native stepping starts at the cold-start second.
Execution inputs are opened only after irreversible commitment, never during
demand construction, activation, routing queries, or candidate selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import pandas as pd

from .mixed_fleet_adapter import VehicleFixture
from .native_simulation import create_native_simulation
from .replay_service_time_adapter import _haversine_m
from .research_world import CommittedPlan, DecisionRequest, EpisodeSnapshot, ResearchEpisode, _seconds
from .upstream import FleetPyCompatibilityError

BRIDGE_VERSION = "stage4_research_native_bridge.v1"


@dataclass
class NativeDecisionRecord:
    """Native-compatible record with a strict decision-only field set."""

    native_id: int
    decision: DecisionRequest
    request_time: pd.Timestamp
    pickup_position: tuple
    dropoff_position: tuple
    source_cohort: str
    passenger_accepts_av: bool
    acceptance_source: str
    native_request: Any = None

    def __getattr__(self, name):
        return getattr(self.decision, name)

    @property
    def sim_time_s(self):
        return int(round(self.decision.release_time_s))

    @property
    def predicted_service_time_s(self):
        return self.decision.predicted_route_time_p50_s

    @property
    def hard_state(self):
        return "FEASIBLE" if self.decision.av_compatible else "INFEASIBLE"

    @property
    def evidence_complete(self):
        return True  # The already-frozen common input population, not new evidence.

    @property
    def av_smoke_eligible(self):
        return self.decision.av_compatible and self.passenger_accepts_av

    selected_route_type = "RESEARCH_COMMON"
    rho_static = rho_dynamic = rho_speed = float("nan")


class NativeCommonRoutePolicy:
    """Use frozen common/profile flags directly; do not re-evaluate static caps."""

    def evaluate(self, request):
        return {
            "research_base_eligible": True,
            "research_route_compatible": request.profile_id in request.compatible_profiles,
            "research_data_ready": True,
            "research_exposure_available": False,
            "exposure": None,
        }


def _set_direct_prediction(native_request, routing_engine):
    # Demand.get_new_travelers calls this upstream method at activation. Its
    # ordinary implementation reads the physical network, so override this one
    # prediction-only request operation without touching the native event loop.
    del routing_engine
    native_request.direct_route_travel_time = float(native_request.predicted_service_time_s)
    native_request.direct_route_travel_distance = float(native_request.decision_service_distance_m)


class _CommittedExecutionRequest:
    """Private, short-lived input to the existing native _assign implementation."""

    __slots__ = ("_decision", "realized_service_time_s")

    def __init__(self, decision: NativeDecisionRecord, actual_duration_s: float):
        self._decision = decision
        self.realized_service_time_s = actual_duration_s

    def __getattr__(self, name):
        return getattr(self._decision, name)


class _IndexedEpisodeLifecycle:
    """Same commitment checks without scanning all 8,435 supply rows per call.

    Native callbacks are ordered by vehicle, not exact event time. Completion
    therefore records a past observed event without moving the release cursor
    backwards or revealing additional customers in the middle of a fleet update.
    """

    def __init__(self, episode: ResearchEpisode):
        self.episode = episode
        self._vehicle_by_order: dict[str, str] = {}
        episode.commit_request = self.commit_request
        episode.complete_request = self.complete_request

    def commit_request(self, order_id, vehicle_id, *, now_s, pickup_eta_s=0.):
        e = self.episode
        now_s, pickup_eta_s = _seconds(now_s), _seconds(pickup_eta_s)
        if now_s != e.requests._now_s:
            e.requests.reveal_until(now_s)
        request = e.requests._pending.get(str(order_id))
        vehicle = e.supply._by_vehicle.get(str(vehicle_id))
        if (request is None or vehicle is None or str(vehicle_id) in e._commitments
                or not vehicle.activation_s <= now_s < vehicle.admission_end_s):
            raise ValueError("commit requires a pending request and an open uncommitted supply slot")
        if pickup_eta_s < 0 or now_s + pickup_eta_s > request.expires_at_s:
            raise ValueError("pickup must fit remaining passenger patience")
        if vehicle.vehicle_type == "AV" and (not request.av_compatible or not request.passenger_accepts_av):
            raise ValueError("AV commitment requires profile compatibility and passenger acceptance")
        request = e.requests.commit(request.order_id, now_s=now_s)
        plan = CommittedPlan(request.order_id, vehicle.vehicle_id, now_s, now_s + pickup_eta_s,
                             now_s + pickup_eta_s + request.predicted_route_time_p50_s,
                             request.dropoff_lon_wgs84, request.dropoff_lat_wgs84)
        e._commitments[vehicle.vehicle_id] = plan
        self._vehicle_by_order[request.order_id] = vehicle.vehicle_id
        return plan

    def complete_request(self, order_id, *, now_s):
        e = self.episode
        order_id, now_s = str(order_id), _seconds(now_s)
        vehicle_id = self._vehicle_by_order.get(order_id)
        if vehicle_id is None or order_id in e.requests._completed:
            raise ValueError("completion requires an active commitment")
        plan = e._commitments[vehicle_id]
        if now_s < plan.commit_time_s:
            raise ValueError("observed completion cannot precede commitment")
        e.requests._completed.add(order_id)
        e.update_location(vehicle_id, plan.dropoff_lon_wgs84, plan.dropoff_lat_wgs84)
        del e._commitments[vehicle_id]
        del self._vehicle_by_order[order_id]


class ResearchNativeBridge:
    """Prepare native inputs and install hooks, without starting a simulation.

    Construct the native controller with requests=[]; demand activation is the
    only way a request enters its decision record map. Install before stepping.
    The runner owns route validation/selection, physical drain and output files.
    ``requests`` contains only already-activated decision records. Environment
    assignment results are deliberately exposed by a separate results method.
    """

    def __init__(self, episode, bindings, registry, *, window_start_s=61200,
                 measurement_end_s=63000, admission_end_s=63300, include_carry_in=True):
        self.episode, self.bindings, self.registry = episode, bindings, registry
        self.window_start_s = int(window_start_s)
        self.measurement_end_s = int(measurement_end_s)
        self.admission_end_s = int(admission_end_s)
        if not 0 <= self.window_start_s < self.measurement_end_s <= self.admission_end_s:
            raise ValueError("cold-start, measurement and admission bounds must be ordered day seconds")
        if episode.requests._now_s > self.window_start_s or episode._commitments:
            raise ValueError("native bridge requires a fresh episode at or before cold start")
        self.benchmark_start = pd.Timestamp(episode._metadata["benchmark_start"])
        if self.benchmark_start != self.benchmark_start.normalize():
            raise ValueError("native bridge requires midnight as the episode timestamp anchor")
        self._records: dict[int, NativeDecisionRecord] = {}
        self._revealed: dict[int, NativeDecisionRecord] = {}
        self._execution_rows: dict[int, dict] = {}
        self._pickup_geometry: dict[int, tuple[Any, int]] = {}
        self._truth_lookup_calls = 0
        self._pickup_geometry_registrations = 0
        self._control = self._demand = None
        request_class = type("ResearchPredictionOnlyBasicRequest", (bindings.basic_request,),
                             {"set_direct_route_travel_infos": _set_direct_prediction})
        for native_id, decision in enumerate(episode.requests._requests):
            in_window = self.window_start_s <= decision.release_time_s < self.measurement_end_s
            carry_in = include_carry_in and decision.release_time_s < self.window_start_s < decision.expires_at_s
            if not (in_window or carry_in):
                continue
            if abs(decision.release_time_s - round(decision.release_time_s)) > 1e-6:
                raise ValueError("pinned native request activation requires second-aligned releases")
            pickup = registry.position_for(decision.pickup_lon_wgs84, decision.pickup_lat_wgs84)
            dropoff = registry.position_for(decision.dropoff_lon_wgs84, decision.dropoff_lat_wgs84)
            record = NativeDecisionRecord(native_id, decision,
                self.benchmark_start + pd.Timedelta(seconds=decision.release_time_s), pickup, dropoff,
                "CARRY_IN" if carry_in else "WINDOW_RELEASE", decision.passenger_accepts_av, decision.acceptance_source)
            row = pd.Series({
                "request_id": native_id, "rq_time": record.sim_time_s,
                "latest_decision_time": decision.expires_at_s,
                "start": int(pickup[0]), "end": int(dropoff[0]), "source_order_id": decision.order_id,
                "predicted_service_time_s": decision.predicted_route_time_p50_s,
                "profile_id": decision.profile_id, "compatible_profiles": decision.compatible_profiles,
                "passenger_accepts_av": decision.passenger_accepts_av,
                "decision_service_distance_m": _haversine_m(decision.pickup_lon_wgs84, decision.pickup_lat_wgs84,
                                                             decision.dropoff_lon_wgs84, decision.dropoff_lat_wgs84),
            })
            record.native_request = request_class(row, registry, 1, {})
            self._records[native_id] = record
        self._plans = tuple(p for p in episode.supply.plan()
                            if p.admission_end_s > self.window_start_s and p.activation_s < self.admission_end_s)
        self._selected_vids = {p.native_id for p in self._plans}
        self._lifecycle = _IndexedEpisodeLifecycle(episode)

    @property
    def requests(self) -> Mapping[int, NativeDecisionRecord]:
        return MappingProxyType(self._revealed.copy())

    def native_fixtures(self) -> list[VehicleFixture]:
        """Window-overlapping original slots; never relabel/re-sample for the window."""
        return [VehicleFixture(p.vehicle_id, p.native_id, p.vehicle_type, p.initial_lon_wgs84, p.initial_lat_wgs84,
                               self.benchmark_start + pd.Timedelta(seconds=p.activation_s),
                               self.benchmark_start + pd.Timedelta(seconds=p.admission_end_s),
                               p.slot_id, p.vehicle_type == "AV", availability_policy=p.availability_policy)
                for p in self._plans]

    def create_demand(self, network, output_dir):
        """Bypass the old demand factory's unconditional true-duration registration."""
        if self._demand is not None:
            raise ValueError("native demand has already been prepared")
        self._demand = self.bindings.demand({"skip_output": 1},
                                           str(Path(output_dir) / "fleetpy_native_user_stats.csv"), routing_engine=network)
        for rid, record in self._records.items():
            activation = max(self.window_start_s, record.sim_time_s)
            self._demand.future_requests.setdefault(activation, {})[rid] = record.native_request
        return self._demand

    def install(self, control):
        """Wrap native callbacks, preserving native vehicle movement and dispatch."""
        if self._control is not None:
            raise ValueError("native hooks have already been installed")
        if self._demand is None or control.demand is not self._demand:
            raise ValueError("controller must use this bridge's native demand")
        if control.request_by_rid:
            raise ValueError("initialize controller with requests=[] to avoid future/truth decision records")
        if pd.Timestamp(control.start) != self.benchmark_start:
            raise ValueError("native controller must retain the midnight episode timestamp anchor")
        if set(control.runtime_by_vid) != self._selected_vids:
            raise ValueError("native vehicles must be exactly the original window-overlapping supply slots")
        self._control = control
        control.sim_time = self.window_start_s
        control.max_pickup_wait_s = int(self.episode._metadata["patience_s"])
        control.research_route_policy = NativeCommonRoutePolicy()
        if not hasattr(control, "expired_rids"):
            control.expired_rids = set()
        if hasattr(control, "acceptance_rate"):
            expected_rate = self.episode._metadata["passenger_acceptance_rate"]
            expected_seed = self.episode._metadata["passenger_acceptance_seed"]
            if control.acceptance_rate != expected_rate or control.acceptance_seed != expected_seed:
                raise ValueError("native acceptance settings differ from the episode's fixed P70 draw")
        if hasattr(control, "config") and control.config.get("profile_id") != self.episode._metadata["decision_profile_id"]:
            raise ValueError("native profile label differs from the episode decision profile")
        old_user = control.user_request
        def user_request(native_request, simulation_time):
            rid = int(native_request.get_rid_struct())
            record = self._records[rid]
            self.episode.reveal_until(simulation_time)
            if not record.release_time_s <= simulation_time < record.expires_at_s:
                raise ValueError("native activation must retain the original release and unexpired patience")
            self._revealed[rid] = record
            control.request_by_rid[rid] = record
            old_user(native_request, simulation_time)
        control.user_request = user_request
        old_assign = control._assign
        def assign(runtime, request, estimate, simulation_time):
            rid, vid = int(request.native_id), int(runtime.fixture.native_id)
            if self._revealed.get(rid) is not request or not control._available(runtime, simulation_time):
                raise ValueError("native assignment requires an activated decision record and an available slot")
            if simulation_time >= self.admission_end_s:
                raise ValueError("short-window admission has ended")
            self.episode.commit_request(request.order_id, runtime.fixture.vehicle_id, now_s=simulation_time,
                                        pickup_eta_s=estimate.corrected_pickup_eta_s)
            self._truth_lookup_calls += 1
            truth = self.episode.execution_truth.lookup(request.order_id)
            actual = truth.realized_service_time_s
            if actual is None or not isfinite(actual) or actual <= 0:
                raise FleetPyCompatibilityError("MISSING_REALIZED_SERVICE_TRUTH_AFTER_COMMITMENT; P50 fallback prohibited")
            old_assign(runtime, _CommittedExecutionRequest(request, float(actual)), estimate, simulation_time)
            self._execution_rows[rid] = {
                "native_request_id": rid, "order_id": request.order_id, "source_cohort": request.source_cohort,
                "realized_service_time_s": float(actual), "truth_status": truth.status,
                "truth_semantics": "HISTORICAL_OD_DURATION_REUSED_FOR_EXECUTION_NOT_OBSERVED_NEW_ROUTE_TRUTH",
            }
            control.assignment_rows[-1].pop("realized_service_time_s", None)
            control.assignment_rows[-1]["source_cohort"] = request.source_cohort
            paths = getattr(control, "city_pickup_paths", {})
            points = paths.get((vid, rid))
            if points is not None:
                manager = getattr(control, "repositioning_manager", None)
                geometry = getattr(manager, "geometry", None) or getattr(control, "city_geometry_bridge", None)
                if geometry is None:
                    raise FleetPyCompatibilityError("validated pickup shape requires the existing shared geometry bridge")
                # old_assign includes the manager's empty-movement interrupt and
                # release. Use its resulting current native origin, then register.
                geometry.register(vid, runtime.native_vehicle.pos, request.pickup_position, points)
                self._pickup_geometry[vid] = (geometry, rid)
                self._pickup_geometry_registrations += 1
        control._assign = assign
        old_board = control.acknowledge_boarding
        def boarding(rid, vid, simulation_time):
            selected = self._pickup_geometry.get(int(vid))
            if selected is not None and selected[1] == int(rid):
                selected[0].release(int(vid))
                del self._pickup_geometry[int(vid)]
            old_board(rid, vid, simulation_time)
            self._execution_rows[int(rid)]["pickup_time_s"] = float(simulation_time)
            self._sync_location(int(vid))
        control.acknowledge_boarding = boarding
        old_alight = control.acknowledge_alighting
        def alighting(rid, vid, simulation_time):
            old_alight(rid, vid, simulation_time)
            self.episode.complete_request(self._revealed[int(rid)].order_id, now_s=simulation_time)
            self._execution_rows[int(rid)]["service_end_time_s"] = float(simulation_time)
            self._execution_rows[int(rid)]["completed"] = True
            self._sync_location(int(vid))
        control.acknowledge_alighting = alighting
        old_status = control.receive_status_update
        def status(vid, simulation_time, list_finished_vrl, force_update=True):
            old_status(vid, simulation_time, list_finished_vrl, force_update)
            self._sync_location(int(vid))
        control.receive_status_update = status
        old_trigger = control.time_trigger
        def trigger(simulation_time):
            self.episode.reveal_until(simulation_time)
            for rid in list(control.demand.waiting_rq):
                rid = int(rid)
                if rid in control.rid_to_assigned_vid:
                    continue
                if simulation_time >= self._revealed[rid].expires_at_s:
                    if hasattr(control, "_expire"):
                        control._expire(rid, simulation_time)
                    else:
                        control.expired_rids.add(rid)
                        control.cancelled_rids.add(rid)
                        control.demand.waiting_rq.pop(rid, None)
                        control.demand.rq_db.pop(rid, None)
                        control.rq_dict.pop(rid, None)
            if simulation_time < self.admission_end_s:
                old_trigger(simulation_time)
            else:
                control.sim_time = int(simulation_time)
        control.time_trigger = trigger
        return control

    def _sync_location(self, vid):
        runtime = self._control.runtime_by_vid[vid]
        lon, lat = self._control.routing_engine.return_position_coordinates(runtime.native_vehicle.pos)
        self.episode.update_location(runtime.fixture.vehicle_id, lon, lat)

    def decision_snapshot(self, now_s):
        """Only activated native-window demand; no demand after measurement end."""
        self.episode.reveal_until(now_s)
        requests = tuple(r.decision for r in self._revealed.values() if r.order_id in self.episode.requests._pending)
        vehicles = self.episode.supply.snapshot(now_s, commitments=self.episode._commitments,
                                               locations=self.episode._locations)
        return EpisodeSnapshot(float(now_s), requests, tuple(v for v in vehicles if v.plan.native_id in self._selected_vids))

    def environment_assignments(self):
        """Copied committed execution ledger for runner reconciliation/output only."""
        if self._control is None:
            rows = []
        else:
            rows = [{**row, **self._execution_rows[int(row["native_request_id"])]}
                    for row in self._control.assignment_rows]
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
            "native_request_id", "order_id", "source_cohort", "realized_service_time_s",
            "truth_status", "truth_semantics", "completed", "simulation_time_s",
            "native_vehicle_id", "pickup_time", "service_end_time",
        ])

    def input_counts(self):
        """Offline scalar input counts; never return an unreleased request list."""
        return {
            "window_new_requests": sum(r.source_cohort == "WINDOW_RELEASE" for r in self._records.values()),
            "carry_in_requests": sum(r.source_cohort == "CARRY_IN" for r in self._records.values()),
            "native_fixture_count": len(self._plans),
            "full_day_slot_count": len(self.episode.supply.plan()),
        }

    def diagnostics(self):
        truth = self.episode.execution_truth.inventory()
        return {
            "bridge_version": BRIDGE_VERSION,
            "hooks_installed": self._control is not None,
            "deferred_truth_calls": self._truth_lookup_calls,
            "truth_source": truth["source_path"], "truth_source_columns": truth["source_columns"],
            "truth_semantics": "HISTORICAL_OD_DURATION_REUSED_FOR_EXECUTION_NOT_OBSERVED_NEW_ROUTE_TRUTH",
            "truth_access_policy": "AFTER_COMMITMENT_ONLY_NO_P50_FALLBACK",
            "demand_service_metric_registration": "NONE",
            "pickup_geometry_registrations": self._pickup_geometry_registrations,
            "active_pickup_geometry_count": len(self._pickup_geometry),
            "completed_native_requests": sum(row.get("completed", False) for row in self._execution_rows.values()),
            "initial_position_source": "FROZEN_TEMPLATE_COLD_START_NOT_OBSERVED_1700_LOCATION",
        }

    def counts(self, now_s=None):
        if now_s is None:
            now_s = self._control.sim_time if self._control is not None else self.window_start_s
        result = {}
        for cohort in ("WINDOW_RELEASE", "CARRY_IN"):
            planned = [r for r in self._records.values() if r.source_cohort == cohort]
            revealed = [r for r in self._revealed.values() if r.source_cohort == cohort]
            result[cohort] = {
                "planned": len(planned), "revealed": len(revealed),
                "committed": sum(r.order_id in self.episode.requests._committed for r in revealed),
                "completed": sum(r.order_id in self.episode.requests._completed for r in revealed),
                "expired": sum(now_s >= r.expires_at_s and r.order_id not in self.episode.requests._committed for r in revealed),
            }
        return result

    def inventory(self):
        return {
            "bridge_version": BRIDGE_VERSION, "bridge_status": "READY", "hooks_installed": self._control is not None,
            "native_run_executed_by_bridge": False, "clock_anchor": self.benchmark_start.isoformat(),
            "window_start_s": self.window_start_s, "measurement_end_s": self.measurement_end_s,
            "admission_end_s": self.admission_end_s,
            "native_fixture_count": len(self._plans), "full_day_slot_count": len(self.episode.supply.plan()),
            "supply_labels": "UNCHANGED_FULL_DAY_Q_AND_SEED_SELECTION",
            "initial_position_source": "FROZEN_TEMPLATE_COLD_START_NOT_OBSERVED_1700_LOCATION",
            "direct_request_time_source": "M3_P50_DECISION_ONLY",
            "execution_truth_access": "AFTER_COMMITMENT_ONLY_NO_P50_FALLBACK",
            "planned_cohort_counts": self.counts(self.window_start_s),
        }

    def create_simulation(self, *, vehicles, fleet_control, network, native_output,
                          simulation_end_s=70200, time_step_s=30):
        """Reuse pinned FleetPy run/step with cold-start clock, not an event-loop replacement."""
        if fleet_control is not self._control:
            raise ValueError("install this bridge's hooks before creating the simulation")
        if simulation_end_s <= self.admission_end_s:
            raise ValueError("physical drain limit must follow short-window admission")
        simulation = create_native_simulation(self.bindings, simulation_end_s=simulation_end_s,
            time_step_s=time_step_s, demand=self._demand,
            vehicles=[getattr(v, "native_vehicle", v) for v in vehicles],
            fleet_control=fleet_control, network=network, native_output=native_output)
        simulation.start_time = self.window_start_s
        simulation.scenario_parameters["start_time"] = self.window_start_s
        return simulation
