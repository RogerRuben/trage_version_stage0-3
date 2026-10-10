"""Pinned ImmediateDecisionsSimulation shell with FleetPy-owned run and step loops."""

from __future__ import annotations

import time
from typing import Any

from .upstream import FleetPyBindings


class _NativeSimulationCore:
    """Initialization only; run() and step() are inherited unchanged from FleetPy."""

    def __init__(
        self,
        bindings: FleetPyBindings,
        *,
        simulation_end_s: int,
        time_step_s: int,
        demand: Any,
        vehicles: list[Any],
        fleet_control: Any,
        network: Any,
        native_output: list[dict],
    ) -> None:
        self.bindings = bindings
        self.start_time = 0
        self.formal_end_time = int(simulation_end_s)
        # One terminal boundary tick activates requests from the final open minute.
        self.end_time = int(simulation_end_s) + 1
        self.time_step = int(time_step_s)
        self.demand = demand
        self.routing_engine = network
        self.operators = [fleet_control]
        self.n_op = 1
        self.broker = bindings.broker_basic(1, self.operators)
        self.sim_vehicles = {(0, vehicle.vid): vehicle for vehicle in vehicles}
        self.vehicle_update_order = {key: 1 for key in self.sim_vehicles}
        self.charging_operator_dict = {}
        self.op_output = [native_output]
        self.scenario_parameters = {
            "scenario_name": "stage4_s2_native_shell",
            "nr_mod_operators": 1,
            "start_time": self.start_time,
            "end_time": self.formal_end_time,
            "time_step": self.time_step,
        }
        self.scenario_name = "stage4_s2_native_shell"
        self.skip_output = True
        self._started = False
        self.realtime_plot_flag = 0
        self.t_init_start = time.perf_counter()
        self._recorded_ticks: set[int] = set()

    def record_stats(self, force: bool = True) -> None:
        del force
        sim_time = int(self.operators[0].sim_time)
        if sim_time not in self._recorded_ticks:
            self.operators[0].record_tick(sim_time)
            self._recorded_ticks.add(sim_time)

    def save_final_state(self) -> None:
        return None

    def record_remaining_assignments(self) -> None:
        return None

    def add_evaluate(self) -> None:
        return None

    def check_sim_env_spec_inputs(self, scenario_parameters: dict) -> None:
        del scenario_parameters


def create_native_simulation(
    bindings: FleetPyBindings,
    *,
    simulation_end_s: int,
    time_step_s: int,
    demand: Any,
    vehicles: list[Any],
    fleet_control: Any,
    network: Any,
    native_output: list[dict],
) -> Any:
    """Create an ImmediateDecisionsSimulation subclass using upstream run/step."""
    simulation_class = type(
        "Stage4NativeImmediateDecisionsSimulation",
        (_NativeSimulationCore, bindings.immediate_simulation),
        {},
    )
    return simulation_class(
        bindings,
        simulation_end_s=simulation_end_s,
        time_step_s=time_step_s,
        demand=demand,
        vehicles=vehicles,
        fleet_control=fleet_control,
        network=network,
        native_output=native_output,
    )
