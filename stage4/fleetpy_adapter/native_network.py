"""FleetPy NetworkBase bridge over registered WGS84 Valhalla leg metrics."""

from __future__ import annotations

from typing import Any

from .upstream import CoordinateRegistry, FleetPyBindings, FleetPyCompatibilityError


class _NativeNetworkCore:
    """Minimal NetworkBase implementation used by native SimulationVehicle movement."""

    def __init__(self, registry: CoordinateRegistry) -> None:
        self.registry = registry
        self.zones = None
        self._vehicle_metrics: dict[tuple[int, int, int, int], tuple[float, float]] = {}

    def register_vehicle_leg(
        self,
        sim_vid_id: tuple[int, int],
        origin: tuple,
        destination: tuple,
        travel_time_s: float,
        distance_m: float,
    ) -> None:
        key = (
            int(sim_vid_id[0]),
            int(sim_vid_id[1]),
            int(origin[0]),
            int(destination[0]),
        )
        values = (float(travel_time_s), float(distance_m))
        if values[0] < 0 or values[1] < 0:
            raise FleetPyCompatibilityError("negative native leg metric")
        self._vehicle_metrics[key] = values
        self.registry.set_leg_metrics(origin, destination, *values)

    def _metrics(
        self,
        origin_node: int,
        destination_node: int,
        sim_vid_id: tuple[int, int] | None = None,
    ) -> tuple[float, float]:
        if origin_node == destination_node:
            return 0.0, 0.0
        if sim_vid_id is not None:
            key = (
                int(sim_vid_id[0]),
                int(sim_vid_id[1]),
                int(origin_node),
                int(destination_node),
            )
            if key in self._vehicle_metrics:
                return self._vehicle_metrics[key]
        values = self.registry.get_section_infos(origin_node, destination_node)
        if values == (0.0, 0.0):
            raise FleetPyCompatibilityError(
                f"unregistered native leg {origin_node}->{destination_node}"
            )
        return values

    def load_tt_file(self, scenario_time: float) -> None:
        del scenario_time

    def update_network(
        self, simulation_time: float, update_state: bool = False
    ) -> bool:
        del simulation_time, update_state
        return False

    def reset_network(self, simulation_time: float) -> None:
        del simulation_time

    def get_number_network_nodes(self) -> int:
        return self.registry.node_count

    def get_must_stop_nodes(self) -> list[int]:
        return []

    def return_node_position(self, node_index: int) -> tuple[int, None, None]:
        return self.registry.return_node_position(node_index)

    def return_node_coordinates(self, node_index: int) -> tuple[float, float]:
        return self.registry.return_node_coordinates(node_index)

    def return_position_coordinates(self, position: tuple) -> tuple[float, float]:
        if position[1] is None:
            return self.registry.coordinate_for_position(position)
        lon0, lat0 = self.registry.return_node_coordinates(int(position[0]))
        lon1, lat1 = self.registry.return_node_coordinates(int(position[1]))
        fraction = float(position[2])
        return lon0 + fraction * (lon1 - lon0), lat0 + fraction * (lat1 - lat0)

    def return_positions_lon_lat(
        self, positions: list[tuple]
    ) -> list[tuple[float, float]]:
        return [self.return_position_coordinates(position) for position in positions]

    def return_position_str(self, position: tuple) -> str:
        if position[1] is None:
            return f"{int(position[0])};-1;-1"
        return f"{int(position[0])};{int(position[1])};{float(position[2]):.7f}"

    def get_section_infos(
        self, start_node_index: int, end_node_index: int
    ) -> tuple[float, float]:
        return self._metrics(int(start_node_index), int(end_node_index))

    def return_route_infos(
        self, route: list[int], rel_start_edge_position: float, start_time: float = 0
    ) -> tuple[float, float]:
        del start_time
        if len(route) < 2:
            return 0.0, 0.0
        travel_time, distance = self._metrics(int(route[0]), int(route[-1]))
        remaining = 1.0 - float(rel_start_edge_position)
        return travel_time * remaining, distance * remaining

    def assign_route_to_network(
        self,
        route: list[int],
        start_time: float,
        end_time: float | None = None,
        number_vehicles: int = 1,
    ) -> None:
        del route, start_time, end_time, number_vehicles

    def return_travel_costs_1to1(
        self,
        origin_position: tuple,
        destination_position: tuple,
        customized_section_cost_function: Any = None,
    ) -> tuple[float, float, float]:
        del customized_section_cost_function
        travel_time, distance = self._metrics(
            int(origin_position[0]), int(destination_position[0])
        )
        return travel_time, travel_time, distance

    def return_travel_costs_Xto1(
        self,
        list_origin_positions: list[tuple],
        destination_position: tuple,
        max_routes: int | None = None,
        max_cost_value: float | None = None,
        customized_section_cost_function: Any = None,
    ) -> list[tuple]:
        del customized_section_cost_function
        rows = []
        for origin in list_origin_positions:
            cost, travel_time, distance = self.return_travel_costs_1to1(
                origin, destination_position
            )
            if max_cost_value is None or cost <= max_cost_value:
                rows.append((origin, cost, travel_time, distance))
        rows.sort(key=lambda row: (row[1], row[0]))
        return rows[:max_routes] if max_routes is not None else rows

    def return_travel_costs_1toX(
        self,
        origin_position: tuple,
        list_destination_positions: list[tuple],
        max_routes: int | None = None,
        max_cost_value: float | None = None,
        customized_section_cost_function: Any = None,
    ) -> list[tuple]:
        del customized_section_cost_function
        rows = []
        for destination in list_destination_positions:
            cost, travel_time, distance = self.return_travel_costs_1to1(
                origin_position, destination
            )
            if max_cost_value is None or cost <= max_cost_value:
                rows.append((destination, cost, travel_time, distance))
        rows.sort(key=lambda row: (row[1], row[0]))
        return rows[:max_routes] if max_routes is not None else rows

    def return_best_route_1to1(
        self,
        origin_position: tuple,
        destination_position: tuple,
        customized_section_cost_function: Any = None,
    ) -> list[int]:
        del customized_section_cost_function
        origin = int(origin_position[0])
        destination = int(destination_position[0])
        return [origin] if origin == destination else [origin, destination]

    def return_best_route_Xto1(
        self,
        list_origin_positions: list[tuple],
        destination_position: tuple,
        max_cost_value: float | None = None,
        customized_section_cost_function: Any = None,
    ) -> list[int]:
        rows = self.return_travel_costs_Xto1(
            list_origin_positions,
            destination_position,
            max_routes=1,
            max_cost_value=max_cost_value,
            customized_section_cost_function=customized_section_cost_function,
        )
        return (
            []
            if not rows
            else self.return_best_route_1to1(rows[0][0], destination_position)
        )

    def return_best_route_1toX(
        self,
        origin_position: tuple,
        list_destination_positions: list[tuple],
        max_cost_value: float | None = None,
        customized_section_cost_function: Any = None,
    ) -> list[int]:
        rows = self.return_travel_costs_1toX(
            origin_position,
            list_destination_positions,
            max_routes=1,
            max_cost_value=max_cost_value,
            customized_section_cost_function=customized_section_cost_function,
        )
        return (
            [] if not rows else self.return_best_route_1to1(origin_position, rows[0][0])
        )

    def return_travel_cost_matrix(
        self,
        list_positions: list[tuple],
        customized_section_cost_function: Any = None,
    ) -> dict[tuple, tuple[float, float, float]]:
        return {
            (origin, destination): self.return_travel_costs_1to1(
                origin, destination, customized_section_cost_function
            )
            for origin in list_positions
            for destination in list_positions
        }

    def move_along_route(
        self,
        route: list[int],
        last_position: tuple,
        time_step: float,
        sim_vid_id: tuple[int, int] | None = None,
        new_sim_time: float | None = None,
        record_node_times: bool = False,
    ) -> tuple[tuple, float, float, list[int], list[float]]:
        current_time = float(new_sim_time or 0.0)
        if last_position[1] is None:
            origin = int(last_position[0])
            destination = int(route[-1]) if route else origin
            fraction = 0.0
        else:
            origin = int(last_position[0])
            destination = int(last_position[1])
            fraction = float(last_position[2])
        if origin == destination:
            return (destination, None, None), 0.0, current_time, [], []
        travel_time, distance = self._metrics(origin, destination, sim_vid_id)
        if travel_time <= 0:
            return (
                (destination, None, None),
                distance * (1 - fraction),
                current_time,
                [destination],
                [current_time] if record_node_times else [],
            )
        remaining_time = travel_time * (1.0 - fraction)
        if float(time_step) + 1e-9 >= remaining_time:
            arrival = current_time + remaining_time
            passed_times = [arrival] if record_node_times else []
            return (
                (destination, None, None),
                distance * (1.0 - fraction),
                arrival,
                [destination],
                passed_times,
            )
        delta = float(time_step) / travel_time
        new_fraction = min(1.0, fraction + delta)
        return (
            (origin, destination, new_fraction),
            distance * delta,
            -1,
            [],
            [],
        )

    def get_zones_external_route_costs(
        self,
        current_time: float,
        tmp_toll_route: list[int],
        park_origin: bool = False,
        park_destination: bool = False,
    ) -> tuple[int, int, int]:
        del current_time, tmp_toll_route, park_origin, park_destination
        return 0, 0, 0


def create_native_network(
    bindings: FleetPyBindings, registry: CoordinateRegistry
) -> Any:
    """Create a concrete official NetworkBase subclass without importing unpinned code."""
    network_class = type(
        "Stage4ValhallaNetworkBridge",
        (_NativeNetworkCore, bindings.network_base),
        {},
    )
    return network_class(registry)
