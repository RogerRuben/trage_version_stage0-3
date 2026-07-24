"""Map Valhalla graph edges to one or more directed canonical OSM segments."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


CANONICAL_COLUMNS = [
    "edge_uid",
    "osm_way_id",
    "segment_seq",
    "direction",
    "from_node",
    "to_node",
    "length_m",
    "highway",
    "bridge",
    "tunnel",
]


@dataclass(frozen=True)
class MappingSummary:
    input_valhalla_edges: int
    output_route_parts: int
    status_counts: dict[str, int]


class CanonicalEdgeMapper:
    """Resolve an edge by way, traversed endpoint nodes, and direction."""

    def __init__(self, canonical_edges: pd.DataFrame) -> None:
        missing = set(CANONICAL_COLUMNS) - set(canonical_edges.columns)
        if missing:
            raise ValueError(f"canonical edges missing columns: {sorted(missing)}")
        self.edges = canonical_edges.reset_index(drop=True).copy()
        self.edges["osm_way_id"] = pd.to_numeric(self.edges.osm_way_id, errors="coerce")
        self._way_direction: dict[tuple[int, str], list[int]] = defaultdict(list)
        for index, way_id, direction in self.edges[
            ["osm_way_id", "direction"]
        ].itertuples(index=True, name=None):
            if pd.notna(way_id):
                self._way_direction[(int(way_id), str(direction))].append(int(index))

    @classmethod
    def from_parquet(cls, path: str | Path) -> "CanonicalEdgeMapper":
        schema = pq.read_schema(path)
        columns = [column for column in CANONICAL_COLUMNS if column in schema.names]
        return cls(pq.read_table(path, columns=columns).to_pandas())

    def _paths(
        self, way_id: int, direction: str, begin_node: int, end_node: int
    ) -> list[list[int]]:
        indices = self._way_direction.get((way_id, direction), [])
        exact = [
            index
            for index in indices
            if int(self.edges.at[index, "from_node"]) == begin_node
            and int(self.edges.at[index, "to_node"]) == end_node
        ]
        if exact:
            return [[index] for index in exact[:2]]
        adjacency: dict[int, list[int]] = defaultdict(list)
        for index in indices:
            adjacency[int(self.edges.at[index, "from_node"])].append(index)
        found: list[list[int]] = []
        stack: list[tuple[int, list[int], set[int]]] = [(begin_node, [], {begin_node})]
        maximum_depth = min(len(indices), 2048)
        while stack and len(found) < 2:
            node, path, visited = stack.pop()
            if len(path) >= maximum_depth:
                continue
            for index in adjacency.get(node, []):
                target = int(self.edges.at[index, "to_node"])
                candidate = [*path, index]
                if target == end_node:
                    found.append(candidate)
                    if len(found) >= 2:
                        break
                elif target not in visited:
                    stack.append((target, candidate, {*visited, target}))
        return found

    def resolve(self, edge: pd.Series) -> tuple[str, list[int]]:
        required = ["osm_way_id", "begin_osm_node_id", "end_osm_node_id", "forward"]
        if any(pd.isna(edge.get(column)) for column in required):
            return "unmapped", []
        way_id = int(edge["osm_way_id"])
        begin_node = int(edge["begin_osm_node_id"])
        end_node = int(edge["end_osm_node_id"])
        direction = "F" if bool(edge["forward"]) else "R"
        paths = self._paths(way_id, direction, begin_node, end_node)
        if not paths:
            return "unmapped", []
        if len(paths) > 1:
            return "ambiguous_mapping", []
        if len(paths[0]) == 1:
            return "exact_edge_mapping", paths[0]
        return "way_and_node_mapping", paths[0]

    def map_route_parts(
        self, route_parts: pd.DataFrame
    ) -> tuple[pd.DataFrame, MappingSummary]:
        if route_parts.empty:
            empty = route_parts.copy()
            for column in (
                "valhalla_route_sequence",
                "canonical_edge_uid",
                "canonical_from_node",
                "canonical_to_node",
                "canonical_highway",
                "canonical_length_m",
                "entry_position_m",
                "exit_position_m",
                "mapping_status",
            ):
                empty[column] = pd.Series(dtype=object)
            return empty, MappingSummary(0, 0, {})
        output: list[dict[str, Any]] = []
        status_counts: dict[str, int] = defaultdict(int)
        for _, route in route_parts.sort_values(
            ["subtrace_id", "route_sequence"], kind="stable"
        ).iterrows():
            status, indices = self.resolve(route)
            status_counts[status] += 1
            base = route.to_dict()
            base["valhalla_route_sequence"] = int(route["route_sequence"])
            if not indices:
                output.append(
                    {
                        **base,
                        "canonical_edge_uid": pd.NA,
                        "canonical_from_node": pd.NA,
                        "canonical_to_node": pd.NA,
                        "canonical_highway": pd.NA,
                        "canonical_length_m": np.nan,
                        "entry_position_m": np.nan,
                        "exit_position_m": np.nan,
                        "mapping_status": status,
                    }
                )
                continue

            canonical = self.edges.loc[indices].copy()
            lengths = pd.to_numeric(canonical.length_m, errors="coerce").fillna(0).to_numpy()
            total = float(lengths.sum())
            source_fraction = float(route.get("source_percent_along", 0.0))
            target_fraction = float(route.get("target_percent_along", 1.0))
            source_m = np.clip(source_fraction, 0, 1) * total
            target_m = np.clip(target_fraction, 0, 1) * total
            matched_span = max(target_m - source_m, 0.0)
            scale = (
                float(route.get("length_m", matched_span)) / matched_span
                if matched_span > 0
                else 0.0
            )
            cursor = 0.0
            emitted = 0
            for (_, canonical_row), segment_length in zip(canonical.iterrows(), lengths):
                segment_start, segment_end = cursor, cursor + float(segment_length)
                overlap_start = max(segment_start, source_m)
                overlap_end = min(segment_end, target_m)
                cursor = segment_end
                if overlap_end <= overlap_start and total > 0:
                    continue
                entry = max(0.0, overlap_start - segment_start)
                exit_ = max(entry, overlap_end - segment_start)
                output.append(
                    {
                        **base,
                        "canonical_edge_uid": canonical_row.edge_uid,
                        "canonical_from_node": int(canonical_row.from_node),
                        "canonical_to_node": int(canonical_row.to_node),
                        "canonical_highway": canonical_row.highway,
                        "canonical_length_m": float(segment_length),
                        "entry_position_m": entry,
                        "exit_position_m": exit_,
                        "length_m": (overlap_end - overlap_start) * scale,
                        "mapping_status": status,
                    }
                )
                emitted += 1
            if emitted == 0:
                output.append(
                    {
                        **base,
                        "canonical_edge_uid": pd.NA,
                        "canonical_from_node": pd.NA,
                        "canonical_to_node": pd.NA,
                        "canonical_highway": pd.NA,
                        "canonical_length_m": np.nan,
                        "entry_position_m": np.nan,
                        "exit_position_m": np.nan,
                        "mapping_status": "unmapped",
                    }
                )
                status_counts[status] -= 1
                status_counts["unmapped"] += 1
        mapped = pd.DataFrame(output)
        if len(mapped):
            mapped["route_sequence"] = np.arange(len(mapped), dtype="int64")
        return mapped, MappingSummary(
            input_valhalla_edges=int(len(route_parts)),
            output_route_parts=int(len(mapped)),
            status_counts=dict(status_counts),
        )
