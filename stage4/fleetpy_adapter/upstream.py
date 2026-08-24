"""Load and verify the external pinned FleetPy checkout without vendoring it."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FLEETPY_REPOSITORY = "https://github.com/TUM-VT/FleetPy.git"
FLEETPY_COMMIT = "0379f9725a147ff33c674de4884cdf89fd787fa9"


class FleetPyCompatibilityError(RuntimeError):
    """Raised when a compatibility-spike invariant is violated."""


@dataclass(frozen=True)
class FleetPyBindings:
    root: Path
    commit: str
    basic_request: type
    external_vehicle: type
    vehicle_route_leg: type
    states: Any


def load_fleetpy_bindings(fleetpy_root: str | Path) -> FleetPyBindings:
    """Import only the upstream classes exercised by the spike."""
    root = Path(fleetpy_root).resolve()
    if not (root / "src" / "FleetSimulationBase.py").is_file():
        raise FleetPyCompatibilityError(f"FleetPy checkout not found at {root}")
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != FLEETPY_COMMIT:
        raise FleetPyCompatibilityError(
            f"FleetPy commit mismatch: expected {FLEETPY_COMMIT}, got {commit}"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.demand.TravelerModels import BasicRequest
    from src.misc.globals import VRL_STATES
    from src.simulation.Legs import VehicleRouteLeg
    from src.simulation.Vehicles import ExternallyMovingSimulationVehicle

    return FleetPyBindings(
        root=root,
        commit=commit,
        basic_request=BasicRequest,
        external_vehicle=ExternallyMovingSimulationVehicle,
        vehicle_route_leg=VehicleRouteLeg,
        states=VRL_STATES,
    )


class CoordinateRegistry:
    """Map explicit WGS84 points to FleetPy node-position tuples."""

    def __init__(self) -> None:
        self._key_to_node: dict[tuple[float, float], int] = {}
        self._node_to_coord: dict[int, tuple[float, float]] = {}
        self._leg_metrics: dict[tuple[int, int], tuple[float, float]] = {}

    def node_for(self, lon: float, lat: float) -> int:
        key = (round(float(lon), 7), round(float(lat), 7))
        if key not in self._key_to_node:
            node = len(self._key_to_node)
            self._key_to_node[key] = node
            self._node_to_coord[node] = (float(lon), float(lat))
        return self._key_to_node[key]

    def position_for(self, lon: float, lat: float) -> tuple[int, None, None]:
        return (self.node_for(lon, lat), None, None)

    def coordinate_for_position(self, position: tuple) -> tuple[float, float]:
        return self._node_to_coord[int(position[0])]

    def set_leg_metrics(
        self,
        origin_position: tuple,
        destination_position: tuple,
        travel_time_s: float,
        distance_m: float,
    ) -> None:
        self._leg_metrics[(int(origin_position[0]), int(destination_position[0]))] = (
            float(travel_time_s),
            float(distance_m),
        )

    # FleetPy routing/network methods exercised by ExternallyMovingSimulationVehicle.
    def return_node_position(self, node_index: int) -> tuple[int, None, None]:
        return (int(node_index), None, None)

    def return_node_coordinates(self, node_index: int) -> tuple[float, float]:
        return self._node_to_coord[int(node_index)]

    def return_position_coordinates(self, position: tuple) -> tuple[float, float]:
        return self.coordinate_for_position(position)

    def return_positions_lon_lat(
        self, positions: list[tuple]
    ) -> list[tuple[float, float]]:
        return [self.coordinate_for_position(position) for position in positions]

    def return_position_str(self, position: tuple) -> str:
        return str(int(position[0]))

    def return_best_route_1to1(self, origin: tuple, destination: tuple) -> list[int]:
        if int(origin[0]) == int(destination[0]):
            return [int(origin[0])]
        return [int(origin[0]), int(destination[0])]

    def return_route_infos(
        self, route: list[int], rel_start_edge_position: float, start_time: float = 0
    ) -> tuple[float, float]:
        del rel_start_edge_position, start_time
        if len(route) < 2:
            return 0.0, 0.0
        return self._leg_metrics.get((int(route[0]), int(route[-1])), (0.0, 0.0))

    def get_section_infos(self, start_node: int, end_node: int) -> tuple[float, float]:
        return self._leg_metrics.get((int(start_node), int(end_node)), (0.0, 0.0))

    @property
    def node_count(self) -> int:
        return len(self._node_to_coord)
