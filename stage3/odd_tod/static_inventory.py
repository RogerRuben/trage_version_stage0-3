"""Stage 3 S1: read-only inventory of frozen static-data availability.

The module answers what data exists.  It deliberately performs no intersection
clustering, static enrichment, profile calibration, route assessment, or model
inference.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import fiona
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


CONFIG_SCHEMA = "stage3_s1_static_inventory_config.1"
INVENTORY_SCHEMA = "stage3_static_data_inventory.1"
EVIDENCE_SCHEMA = "stage3_s1_static_inventory_evidence.1"
TEST_EVIDENCE_SCHEMA = "stage3_s1_static_inventory_test_evidence.1"
PHASE_STATUS = "STAGE3_S1_STATIC_INVENTORY_COMPLETE"

ROUTE_PART_COLUMNS = (
    "canonical_edge_uid",
    "osm_way_id",
    "begin_osm_node_id",
    "end_osm_node_id",
    "canonical_from_node",
    "canonical_to_node",
    "forward",
    "canonical_traversal_direction",
    "speed_limit",
    "length_m",
    "road_class",
    "canonical_highway",
    "bridge",
    "tunnel",
)

TAG_PATTERN = re.compile(r'"((?:\\.|[^"\\])*)"=>"((?:\\.|[^"\\])*)"')
NUMERIC_SPEED = re.compile(r"^([0-9]+(?:\.[0-9]+)?)$")
UNIT_SPEED = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*(?:km/?h|kmh|kph)$", re.I)
MPH_SPEED = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*mph$", re.I)


class Stage3S1InventoryError(ValueError):
    """Raised when an S1 input or frozen execution boundary is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _payload_hash(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("artifact_sha256", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _descriptor(path: Path, root: Path | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise Stage3S1InventoryError(f"missing S1 source: {path}")
    resolved = path.resolve()
    label = resolved.as_posix()
    if root is not None:
        try:
            label = resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return {
        "path": label,
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Stage3S1InventoryError(f"missing S1 JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Stage3S1InventoryError(f"S1 JSON is not an object: {path}")
    return payload


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_config(path: Path, root: Path) -> dict[str, Any]:
    config = _read_json(path)
    expected_authorizations = {
        "s1": True,
        **{f"s{phase}": False for phase in range(2, 9)},
        "stage4": False,
    }
    boundaries = config.get("analysis_boundaries", {})
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("phase") != "S1_STATIC_DATA_INVENTORY"
        or config.get("execution_authorization") != "S1_ONLY"
        or config.get("authorizations") != expected_authorizations
        or not boundaries
        or any(bool(value) for value in boundaries.values())
    ):
        raise Stage3S1InventoryError("S1 config authorizes work outside static inventory")
    for key in (
        "stage0_freeze_manifest",
        "stage1_release_manifest",
        "stage2_final_release_manifest",
        "stage1_input",
        "pbf",
        "poi",
        "valhalla_build_manifest",
        "output",
    ):
        if key not in config.get("paths", {}):
            raise Stage3S1InventoryError(f"S1 config misses path: {key}")
    return config


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, encoding="utf-8"
    ).strip()


def _decode_fiona_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    text = str(value)
    if (text.startswith("b'") and text.endswith("'")) or (
        text.startswith('b"') and text.endswith('"')
    ):
        try:
            decoded = ast.literal_eval(text)
            if isinstance(decoded, bytes):
                return decoded.decode("utf-8", errors="replace")
        except (SyntaxError, ValueError):
            pass
    return text


def parse_other_tags(value: Any) -> dict[str, str]:
    text = _decode_fiona_text(value)
    return {
        key.replace(r'\"', '"').replace(r"\\", "\\"): val.replace(r'\"', '"').replace(r"\\", "\\")
        for key, val in TAG_PATTERN.findall(text)
    }


def parse_maxspeed(value: str | None) -> dict[str, Any]:
    """Classify an OSM maxspeed string without inventing a legal default."""

    raw = str(value or "").strip()
    if not raw:
        return {"raw": raw, "format": "missing", "parseable": False, "values_kmh": []}
    lowered = raw.lower().strip()
    if "@" in lowered:
        return {"raw": raw, "format": "conditional", "parseable": False, "values_kmh": []}
    pieces = [piece.strip() for piece in re.split(r"[;|]", lowered) if piece.strip()]
    values: list[float] = []
    formats: list[str] = []
    for piece in pieces:
        match = NUMERIC_SPEED.fullmatch(piece)
        if match:
            values.append(float(match.group(1)))
            formats.append("numeric")
            continue
        match = UNIT_SPEED.fullmatch(piece)
        if match:
            values.append(float(match.group(1)))
            formats.append("kmh_unit")
            continue
        match = MPH_SPEED.fullmatch(piece)
        if match:
            values.append(float(match.group(1)) * 1.609344)
            formats.append("mph")
            continue
        return {
            "raw": raw,
            "format": "named_or_non_numeric",
            "parseable": False,
            "values_kmh": [],
        }
    if not pieces:
        return {"raw": raw, "format": "invalid", "parseable": False, "values_kmh": []}
    fmt = formats[0] if len(formats) == 1 else "multi_numeric"
    return {"raw": raw, "format": fmt, "parseable": True, "values_kmh": values}


def _geometry_key(coordinates: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    forward = tuple((round(float(x), 7), round(float(y), 7)) for x, y, *_ in coordinates)
    reverse = tuple(reversed(forward))
    return min(forward, reverse)


def _count_rate(count: int, denominator: int) -> dict[str, Any]:
    return {
        "count": int(count),
        "denominator": int(denominator),
        "share": float(count / denominator) if denominator else None,
    }


def _empty_edge_record() -> dict[str, set[Any]]:
    return {
        "osm_way_id": set(),
        "begin_osm_node_id": set(),
        "end_osm_node_id": set(),
        "canonical_from_node": set(),
        "canonical_to_node": set(),
        "forward": set(),
        "canonical_traversal_direction": set(),
        "speed_limit": set(),
        "road_class": set(),
        "canonical_highway": set(),
        "bridge": set(),
        "tunnel": set(),
    }


def _valid_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _valid_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def inventory_stage0_route_parts(
    input_root: Path, *, low_speed: float, high_speed: float
) -> tuple[dict[str, Any], dict[str, dict[str, set[Any]]], dict[str, Any]]:
    files = sorted(input_root.glob("split=*/date=*/bucket=*/route_parts.parquet"))
    if not files:
        raise Stage3S1InventoryError("no frozen route_parts files found")
    edges: dict[str, dict[str, set[Any]]] = defaultdict(_empty_edge_record)
    row_total = row_null = row_zero = row_negative = row_low = row_high = row_positive = 0
    length_total = length_covered = 0.0
    class_rows: dict[str, Counter[str]] = defaultdict(Counter)

    for path in files:
        parquet = pq.ParquetFile(path)
        missing = set(ROUTE_PART_COLUMNS) - set(parquet.schema_arrow.names)
        if missing:
            raise Stage3S1InventoryError(f"route_parts schema missing {sorted(missing)}: {path}")
        for batch in parquet.iter_batches(columns=list(ROUTE_PART_COLUMNS), batch_size=65536):
            frame = batch.to_pandas()
            speed = pd.to_numeric(frame["speed_limit"], errors="coerce").to_numpy(float)
            length = pd.to_numeric(frame["length_m"], errors="coerce").fillna(0).to_numpy(float)
            finite = np.isfinite(speed)
            positive = finite & (speed > 0)
            row_total += len(frame)
            row_null += int((~finite).sum())
            row_zero += int((finite & (speed == 0)).sum())
            row_negative += int((finite & (speed < 0)).sum())
            row_low += int((positive & (speed < low_speed)).sum())
            row_high += int((finite & (speed > high_speed)).sum())
            row_positive += int(positive.sum())
            length_total += float(np.where(np.isfinite(length), np.maximum(length, 0), 0).sum())
            length_covered += float(
                np.where(positive & np.isfinite(length), np.maximum(length, 0), 0).sum()
            )
            road_classes = frame["road_class"].fillna("<NULL>").astype(str)
            for road_class, indices in road_classes.groupby(road_classes).groups.items():
                idx = np.fromiter(indices, dtype=np.int64)
                class_rows[road_class]["rows"] += int(len(idx))
                class_rows[road_class]["positive"] += int(positive[idx].sum())
                class_rows[road_class]["null"] += int((~finite[idx]).sum())
                class_rows[road_class]["zero"] += int((finite[idx] & (speed[idx] == 0)).sum())
                class_rows[road_class]["abnormal"] += int(
                    ((positive[idx] & (speed[idx] < low_speed)) | (finite[idx] & (speed[idx] > high_speed)) | (finite[idx] & (speed[idx] < 0))).sum()
                )

            # Route parts contain 15.6M traversal rows but only ~12k canonical
            # edges.  Collapse identical static identities inside each batch
            # before updating the edge inventory; row-weighted statistics above
            # still use every row.
            static_rows = frame.drop_duplicates(
                subset=[
                    "canonical_edge_uid", "osm_way_id", "begin_osm_node_id",
                    "end_osm_node_id", "canonical_from_node", "canonical_to_node",
                    "forward", "canonical_traversal_direction", "speed_limit",
                    "road_class", "canonical_highway",
                    "bridge", "tunnel",
                ]
            )
            for row in static_rows.itertuples(index=False):
                uid = _valid_text(row.canonical_edge_uid)
                if uid is None:
                    continue
                record = edges[uid]
                for column in (
                    "osm_way_id",
                    "begin_osm_node_id",
                    "end_osm_node_id",
                    "canonical_from_node",
                    "canonical_to_node",
                ):
                    value = _valid_int(getattr(row, column))
                    if value is not None:
                        record[column].add(value)
                record["forward"].add(bool(row.forward))
                direction = _valid_text(row.canonical_traversal_direction)
                if direction is not None:
                    record["canonical_traversal_direction"].add(direction)
                if row.speed_limit is not None and not pd.isna(row.speed_limit):
                    record["speed_limit"].add(float(row.speed_limit))
                for column in ("road_class", "canonical_highway"):
                    value = _valid_text(getattr(row, column))
                    if value is not None:
                        record[column].add(value)
                record["bridge"].add(bool(row.bridge))
                record["tunnel"].add(bool(row.tunnel))

    edge_total = len(edges)
    edge_positive = edge_null = edge_zero = edge_negative = edge_low = edge_high = 0
    inconsistent_speed = 0
    multi_value_by_field: Counter[str] = Counter()
    missing_value_by_field: Counter[str] = Counter()
    edge_by_class: dict[str, Counter[str]] = defaultdict(Counter)
    for record in edges.values():
        speeds = record["speed_limit"]
        positive_values = [value for value in speeds if math.isfinite(value) and value > 0]
        edge_positive += bool(positive_values)
        edge_null += not speeds
        edge_zero += any(math.isfinite(value) and value == 0 for value in speeds)
        edge_negative += any(math.isfinite(value) and value < 0 for value in speeds)
        edge_low += any(0 < value < low_speed for value in speeds if math.isfinite(value))
        edge_high += any(value > high_speed for value in speeds if math.isfinite(value))
        inconsistent_speed += len(speeds) > 1
        for column in (
            "osm_way_id", "begin_osm_node_id", "end_osm_node_id",
                "canonical_from_node", "canonical_to_node", "forward",
                "canonical_traversal_direction",
        ):
            multi_value_by_field[column] += len(record[column]) > 1
            missing_value_by_field[column] += len(record[column]) == 0
        classes = record["road_class"] or {"<NULL>"}
        for road_class in classes:
            edge_by_class[str(road_class)]["edges"] += 1
            edge_by_class[str(road_class)]["positive"] += bool(positive_values)
            edge_by_class[str(road_class)]["null"] += not speeds
            edge_by_class[str(road_class)]["zero"] += any(value == 0 for value in speeds)
            edge_by_class[str(road_class)]["abnormal"] += (
                any(value < 0 or 0 < value < low_speed or value > high_speed for value in speeds)
            )

    endpoint_nodes = {
        node
        for record in edges.values()
        for column in ("begin_osm_node_id", "end_osm_node_id")
        for node in record[column]
    }
    way_to_nodes: dict[int, set[int]] = defaultdict(set)
    for record in edges.values():
        nodes = record["begin_osm_node_id"] | record["end_osm_node_id"]
        for way in record["osm_way_id"]:
            way_to_nodes[int(way)].update(int(node) for node in nodes)
    directed_identity_count = int(
        sum(
            len(record["canonical_traversal_direction"])
            or len(record["forward"])
            for record in edges.values()
        )
    )
    bidirectionally_observed_segment_count = int(
        sum(len(record["canonical_traversal_direction"]) > 1 for record in edges.values())
    )

    source_set = {
        "file_count": len(files),
        "total_size_bytes": int(sum(path.stat().st_size for path in files)),
        "relative_paths_sha256": hashlib.sha256(
            "\n".join(path.relative_to(input_root).as_posix() for path in files).encode("utf-8")
        ).hexdigest(),
    }
    summary = {
        "population": {
            "route_part_rows": int(row_total),
            "observed_unique_canonical_segment_ids": int(edge_total),
            "observed_unique_directed_canonical_identities": directed_identity_count,
            "canonical_segment_ids_observed_in_both_directions": (
                bidirectionally_observed_segment_count
            ),
            "route_part_file_count": len(files),
            "scope_note": (
                "accepted-order observed subnetwork; not the complete frozen routable graph"
            ),
        },
        "speed_limit": {
            "field_present": True,
            "provenance_field_present": False,
            "unit_assumed_from_valhalla_contract": "km/h",
            "diagnostic_abnormal_definition": {
                "low_positive_below_kmh": low_speed,
                "high_above_kmh": high_speed,
            },
            "row_weighted": {
                "coverage_positive": _count_rate(row_positive, row_total),
                "null": _count_rate(row_null, row_total),
                "zero": _count_rate(row_zero, row_total),
                "negative": _count_rate(row_negative, row_total),
                "low_positive": _count_rate(row_low, row_total),
                "high": _count_rate(row_high, row_total),
                "length_weighted_positive_coverage_share": (
                    float(length_covered / length_total) if length_total else None
                ),
            },
            "unique_edge_weighted": {
                "identity_unit": "canonical_edge_uid base segment; direction stored separately",
                "coverage_positive": _count_rate(edge_positive, edge_total),
                "null": _count_rate(edge_null, edge_total),
                "zero": _count_rate(edge_zero, edge_total),
                "negative": _count_rate(edge_negative, edge_total),
                "low_positive": _count_rate(edge_low, edge_total),
                "high": _count_rate(edge_high, edge_total),
                "multiple_values_for_same_canonical_edge": _count_rate(
                    inconsistent_speed, edge_total
                ),
            },
            "by_road_class_unique_edge": {
                road_class: {
                    **dict(counter),
                    "positive_coverage_share": float(counter["positive"] / counter["edges"]),
                }
                for road_class, counter in sorted(edge_by_class.items())
            },
            "by_road_class_route_part_row": {
                road_class: {
                    **dict(counter),
                    "positive_coverage_share": float(counter["positive"] / counter["rows"]),
                }
                for road_class, counter in sorted(class_rows.items())
            },
        },
        "static_flags": {
            "bridge_true_edges": int(sum(True in record["bridge"] for record in edges.values())),
            "tunnel_true_edges": int(sum(True in record["tunnel"] for record in edges.values())),
            "bridge_null_edges": 0,
            "tunnel_null_edges": 0,
            "layer_field_present": False,
            "junction_field_present": False,
            "signal_field_present": False,
            "roundabout_field_present": False,
            "turn_restriction_field_present": False,
        },
        "graph_identity": {
            "canonical_edge_uid_present": True,
            "canonical_traversal_direction_present": True,
            "directed_identity": [
                "canonical_edge_uid", "canonical_traversal_direction"
            ],
            "osm_way_id_present": True,
            "begin_end_osm_node_id_present": True,
            "canonical_from_to_node_present": True,
            "unique_observed_endpoint_osm_node_count": len(endpoint_nodes),
            "unique_observed_osm_way_count": len(way_to_nodes),
            "multiple_observed_source_values_by_field": {
                field: _count_rate(count, edge_total)
                for field, count in sorted(multi_value_by_field.items())
            },
            "missing_values_by_field": {
                field: _count_rate(count, edge_total)
                for field, count in sorted(missing_value_by_field.items())
            },
            "source_multiplicity_interpretation": (
                "canonical_edge_uid is the base segment identity and direction is separate; "
                "opposite direction/endpoints on one base ID are expected, not conflicts"
            ),
        },
    }
    helpers = {"endpoint_nodes": endpoint_nodes, "way_to_nodes": way_to_nodes}
    return summary, dict(edges), {"source_set": source_set, **helpers}


def inventory_pbf(
    pbf_path: Path,
    *,
    observed_edges: Mapping[str, Mapping[str, set[Any]]],
    endpoint_nodes: set[int],
    way_to_nodes: Mapping[int, set[int]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    layers = fiona.listlayers(pbf_path.as_posix())
    required_layers = {"points", "lines", "other_relations"}
    if not required_layers.issubset(layers):
        raise Stage3S1InventoryError(f"PBF misses GDAL layers: {sorted(required_layers-set(layers))}")

    point_total = signal_count = roundabout_node_count = mini_roundabout_count = 0
    signal_ids: set[int] = set()
    tagged_node_ids: set[int] = set()
    with fiona.open(
        pbf_path.as_posix(), layer="points", allow_unsupported_drivers=True
    ) as source:
        point_schema = dict(source.schema["properties"])
        for feature in source:
            point_total += 1
            properties = feature["properties"]
            node_id = _valid_int(properties.get("osm_id"))
            if node_id is not None:
                tagged_node_ids.add(node_id)
            highway = _decode_fiona_text(properties.get("highway"))
            tags = parse_other_tags(properties.get("other_tags"))
            if highway == "traffic_signals":
                signal_count += 1
                if node_id is not None:
                    signal_ids.add(node_id)
            if tags.get("junction") == "roundabout":
                roundabout_node_count += 1
            if highway == "mini_roundabout":
                mini_roundabout_count += 1

    line_total = highway_total = roundabout_way_count = 0
    highway_way_ids: set[int] = set()
    raw_way_tags: dict[int, dict[str, str]] = {}
    geometry_index: dict[tuple[tuple[float, float], ...], set[int]] = defaultdict(set)
    maxspeed_base = maxspeed_any = maxspeed_parseable = 0
    maxspeed_formats: Counter[str] = Counter()
    maxspeed_values: Counter[str] = Counter()
    maxspeed_by_highway: dict[str, Counter[str]] = defaultdict(Counter)
    directional_only = conditional_count = 0
    layer_present = layer_nonzero = layer_unparseable = 0
    bridge_yes = tunnel_yes = grade_separation_any = 0
    tag_value_counts: dict[str, Counter[str]] = {
        "layer": Counter(), "bridge": Counter(), "tunnel": Counter(), "junction": Counter()
    }

    with fiona.open(
        pbf_path.as_posix(), layer="lines", allow_unsupported_drivers=True
    ) as source:
        line_schema = dict(source.schema["properties"])
        for feature in source:
            line_total += 1
            properties = feature["properties"]
            way_id = _valid_int(properties.get("osm_id"))
            tags = parse_other_tags(properties.get("other_tags"))
            highway = _decode_fiona_text(properties.get("highway"))
            geometry = feature["geometry"]
            if way_id is not None and geometry is not None and geometry.type == "LineString":
                geometry_index[_geometry_key(geometry.coordinates)].add(way_id)
            if not highway or way_id is None:
                continue
            highway_total += 1
            maxspeed_by_highway[highway]["ways"] += 1
            highway_way_ids.add(way_id)
            raw_way_tags[way_id] = {"highway": highway, **tags}
            maxspeed = tags.get("maxspeed")
            directional = tags.get("maxspeed:forward") or tags.get("maxspeed:backward")
            conditional = tags.get("maxspeed:conditional")
            if maxspeed:
                maxspeed_base += 1
                maxspeed_by_highway[highway]["base_tagged"] += 1
                parsed = parse_maxspeed(maxspeed)
                maxspeed_parseable += int(parsed["parseable"])
                maxspeed_by_highway[highway]["base_parseable"] += int(parsed["parseable"])
                maxspeed_formats[parsed["format"]] += 1
                maxspeed_values[maxspeed] += 1
            if maxspeed or directional or conditional:
                maxspeed_any += 1
                maxspeed_by_highway[highway]["any_speed_tag"] += 1
            if not maxspeed and directional:
                directional_only += 1
            if conditional:
                conditional_count += 1
            if tags.get("junction") == "roundabout":
                roundabout_way_count += 1
            for tag in tag_value_counts:
                if tag in tags:
                    tag_value_counts[tag][tags[tag]] += 1
            layer = tags.get("layer")
            if layer is not None:
                layer_present += 1
                try:
                    layer_nonzero += float(layer) != 0
                except ValueError:
                    layer_unparseable += 1
            bridge = str(tags.get("bridge", "")).lower() not in {"", "no", "false", "0"}
            tunnel = str(tags.get("tunnel", "")).lower() not in {"", "no", "false", "0"}
            bridge_yes += bridge
            tunnel_yes += tunnel
            grade_separation_any += bool(bridge or tunnel or (layer is not None and layer != "0"))

    observed_way_ids = {
        int(way)
        for record in observed_edges.values()
        for way in record["osm_way_id"]
    }
    observed_way_raw_match = observed_way_ids & highway_way_ids
    signal_endpoint_match = signal_ids & endpoint_nodes
    roundabout_way_ids = {
        way for way, tags in raw_way_tags.items() if tags.get("junction") == "roundabout"
    }
    observed_roundabout_ways = roundabout_way_ids & observed_way_ids
    signal_incident_segment_count = int(
        sum(
            bool((record["begin_osm_node_id"] | record["end_osm_node_id"]) & signal_ids)
            for record in observed_edges.values()
        )
    )
    signal_incident_directed_count = int(
        sum(
            (
                len(record["canonical_traversal_direction"])
                or len(record["forward"])
            )
            for record in observed_edges.values()
            if (record["begin_osm_node_id"] | record["end_osm_node_id"]) & signal_ids
        )
    )
    total_directed_count = int(
        sum(
            len(record["canonical_traversal_direction"])
            or len(record["forward"])
            for record in observed_edges.values()
        )
    )
    roundabout_segment_count = int(
        sum(bool(record["osm_way_id"] & roundabout_way_ids) for record in observed_edges.values())
    )

    observed_static_comparison = Counter()
    for record in observed_edges.values():
        raw_bridge = raw_tunnel = raw_nonzero_layer = False
        for way in record["osm_way_id"]:
            tags = raw_way_tags.get(int(way), {})
            raw_bridge |= str(tags.get("bridge", "")).lower() not in {"", "no", "false", "0"}
            raw_tunnel |= str(tags.get("tunnel", "")).lower() not in {"", "no", "false", "0"}
            layer = tags.get("layer")
            if layer is not None:
                try:
                    raw_nonzero_layer |= float(layer) != 0
                except ValueError:
                    pass
        stage0_bridge = True in record["bridge"]
        stage0_tunnel = True in record["tunnel"]
        observed_static_comparison["edges"] += 1
        observed_static_comparison["raw_bridge_true"] += raw_bridge
        observed_static_comparison["stage0_bridge_true"] += stage0_bridge
        observed_static_comparison["raw_bridge_true_stage0_false"] += raw_bridge and not stage0_bridge
        observed_static_comparison["stage0_bridge_true_raw_false"] += stage0_bridge and not raw_bridge
        observed_static_comparison["raw_tunnel_true"] += raw_tunnel
        observed_static_comparison["stage0_tunnel_true"] += stage0_tunnel
        observed_static_comparison["raw_tunnel_true_stage0_false"] += raw_tunnel and not stage0_tunnel
        observed_static_comparison["stage0_tunnel_true_raw_false"] += stage0_tunnel and not raw_tunnel
        observed_static_comparison["raw_nonzero_layer"] += raw_nonzero_layer

    edge_osm_comparison = Counter()
    speed_differences: list[float] = []
    for record in observed_edges.values():
        stage0_values = sorted(
            value for value in record["speed_limit"] if math.isfinite(value) and value > 0
        )
        way_values: list[float] = []
        raw_tagged = False
        raw_parseable = False
        directional_info = False
        conditional_info = False
        for way in record["osm_way_id"]:
            tags = raw_way_tags.get(int(way), {})
            raw = tags.get("maxspeed")
            if raw:
                raw_tagged = True
                parsed = parse_maxspeed(raw)
                if parsed["parseable"]:
                    raw_parseable = True
                    way_values.extend(parsed["values_kmh"])
            directional_info |= bool(tags.get("maxspeed:forward") or tags.get("maxspeed:backward"))
            conditional_info |= bool(tags.get("maxspeed:conditional"))
        edge_osm_comparison["edges"] += 1
        edge_osm_comparison["stage0_positive"] += bool(stage0_values)
        edge_osm_comparison["osm_base_tagged"] += raw_tagged
        edge_osm_comparison["osm_base_parseable"] += raw_parseable
        edge_osm_comparison["osm_directional_tagged"] += directional_info
        edge_osm_comparison["osm_conditional_tagged"] += conditional_info
        edge_osm_comparison["osm_parseable_but_stage0_missing"] += (
            raw_parseable and not stage0_values
        )
        if stage0_values and way_values:
            difference = min(abs(left - right) for left in stage0_values for right in way_values)
            speed_differences.append(float(difference))
            edge_osm_comparison["both_numeric"] += 1
            edge_osm_comparison["numeric_exact_within_0_5_kmh"] += difference <= 0.5
            edge_osm_comparison["numeric_difference_over_10_kmh"] += difference > 10

    restriction_total = 0
    restriction_types: Counter[str] = Counter()
    via_types: Counter[str] = Counter()
    relation_line_fully_resolved = 0
    relation_line_ambiguous = 0
    all_member_ways_observed = 0
    node_via_endpoint_join_candidate = 0
    way_via_member_set_observed = 0
    relation_member_counts: Counter[str] = Counter()
    with fiona.open(
        pbf_path.as_posix(), layer="other_relations", allow_unsupported_drivers=True
    ) as source:
        relation_schema = dict(source.schema["properties"])
        for feature in source:
            properties = feature["properties"]
            tags = parse_other_tags(properties.get("other_tags"))
            if _decode_fiona_text(properties.get("type")) != "restriction":
                continue
            restriction_total += 1
            restriction_types[tags.get("restriction", "<MISSING>")] += 1
            geometry = feature["geometry"]
            children = list(geometry.geometries or []) if geometry is not None else []
            lines = [child for child in children if child.type == "LineString"]
            points = [child for child in children if child.type == "Point"]
            if points and len(lines) >= 2:
                via_type = "node_via"
            elif not points and len(lines) >= 3:
                via_type = "way_via"
            else:
                via_type = "ambiguous_geometry"
            via_types[via_type] += 1
            relation_member_counts[f"line_members={len(lines)};point_members={len(points)}"] += 1
            resolved_sets = [geometry_index.get(_geometry_key(line.coordinates), set()) for line in lines]
            uniquely_resolved = bool(lines) and all(len(values) == 1 for values in resolved_sets)
            relation_line_fully_resolved += uniquely_resolved
            relation_line_ambiguous += any(len(values) > 1 for values in resolved_sets)
            resolved_ways = {next(iter(values)) for values in resolved_sets if len(values) == 1}
            all_observed = uniquely_resolved and resolved_ways.issubset(observed_way_ids)
            all_member_ways_observed += all_observed
            if via_type == "node_via" and uniquely_resolved and len(resolved_ways) == 2:
                left, right = sorted(resolved_ways)
                shared = way_to_nodes.get(left, set()) & way_to_nodes.get(right, set())
                node_via_endpoint_join_candidate += len(shared) == 1
            if via_type == "way_via":
                way_via_member_set_observed += all_observed

    difference_summary = {
        "count": len(speed_differences),
        "p50_abs_difference_kmh": (
            float(np.quantile(speed_differences, 0.5)) if speed_differences else None
        ),
        "p90_abs_difference_kmh": (
            float(np.quantile(speed_differences, 0.9)) if speed_differences else None
        ),
        "maximum_abs_difference_kmh": max(speed_differences) if speed_differences else None,
    }
    summary = {
        "reader": {
            "interface": "Fiona/GDAL OSM vector layers",
            "layers": list(layers),
            "limitation": (
                "GDAL exposes restriction geometries but not raw relation member roles/refs; "
                "full exact directed-turn mapping is not certified in S1"
            ),
        },
        "schemas": {
            "points": point_schema,
            "lines": line_schema,
            "other_relations": relation_schema,
        },
        "traffic_signals": {
            "raw_highway_traffic_signals_nodes": signal_count,
            "raw_tagged_point_features": point_total,
            "mapped_to_observed_graph_endpoint_by_osm_node_id": _count_rate(
                len(signal_endpoint_match), signal_count
            ),
            "observed_graph_endpoint_nodes_that_are_signals": _count_rate(
                len(signal_endpoint_match), len(endpoint_nodes)
            ),
            "observed_canonical_segments_incident_to_signal_node": _count_rate(
                signal_incident_segment_count, len(observed_edges)
            ),
            "observed_directed_identities_incident_to_signal_node": _count_rate(
                signal_incident_directed_count, total_directed_count
            ),
            "identity_method": "exact OSM node ID intersection",
            "mapping_scope": "accepted-order observed subnetwork; full frozen graph unavailable",
        },
        "roundabout": {
            "junction_roundabout_way_count": roundabout_way_count,
            "junction_roundabout_node_count": roundabout_node_count,
            "highway_mini_roundabout_node_count": mini_roundabout_count,
            "roundabout_ways_represented_in_observed_graph": _count_rate(
                len(observed_roundabout_ways), roundabout_way_count
            ),
            "observed_canonical_segments_on_roundabout_way": _count_rate(
                roundabout_segment_count, len(observed_edges)
            ),
            "representation": "primarily OSM ways; mini_roundabout is a separate node tag",
        },
        "maxspeed": {
            "highway_way_population": highway_total,
            "base_tag_coverage": _count_rate(maxspeed_base, highway_total),
            "any_base_directional_or_conditional_coverage": _count_rate(
                maxspeed_any, highway_total
            ),
            "base_tag_parseability": _count_rate(maxspeed_parseable, maxspeed_base),
            "directional_only_way_count": directional_only,
            "conditional_way_count": conditional_count,
            "format_counts": dict(maxspeed_formats.most_common()),
            "top_raw_values": dict(maxspeed_values.most_common(30)),
            "by_osm_highway": {
                highway: {
                    **dict(counter),
                    "base_tag_coverage_share": float(counter["base_tagged"] / counter["ways"]),
                    "base_parseable_share_of_tagged": (
                        float(counter["base_parseable"] / counter["base_tagged"])
                        if counter["base_tagged"] else None
                    ),
                }
                for highway, counter in sorted(maxspeed_by_highway.items())
            },
            "tagged_highway_ways_not_in_observed_graph": int(
                len({way for way, tags in raw_way_tags.items() if tags.get("maxspeed")} - observed_way_ids)
            ),
            "observed_edge_comparison": {
                **dict(edge_osm_comparison),
                "osm_parseable_but_stage0_missing_share": (
                    float(edge_osm_comparison["osm_parseable_but_stage0_missing"] / edge_osm_comparison["edges"])
                    if edge_osm_comparison["edges"] else None
                ),
                "numeric_difference": difference_summary,
                "scope": "observed accepted-order canonical edges only",
            },
        },
        "turn_restrictions": {
            "restriction_relation_count": restriction_total,
            "restriction_value_counts": dict(restriction_types.most_common()),
            "via_type_counts_inferred_from_geometry": dict(via_types),
            "member_geometry_shape_counts": dict(relation_member_counts.most_common()),
            "all_line_member_geometries_resolved_to_unique_osm_way": _count_rate(
                relation_line_fully_resolved, restriction_total
            ),
            "relations_with_ambiguous_line_geometry_match": relation_line_ambiguous,
            "all_member_ways_present_in_observed_graph": _count_rate(
                all_member_ways_observed, restriction_total
            ),
            "node_via_single_shared_observed_endpoint_candidate_count": (
                node_via_endpoint_join_candidate
            ),
            "way_via_all_member_ways_observed_count": way_via_member_set_observed,
            "exact_current_directed_network_mapping_certified": False,
            "exact_mapping_blocker": (
                "raw relation roles/member refs are not exposed by the available GDAL reader; "
                "a role-preserving PBF reader is required before S2 enforcement"
            ),
        },
        "topology_and_layers": {
            "highway_way_count": highway_total,
            "layer_tag_coverage": _count_rate(layer_present, highway_total),
            "nonzero_layer": _count_rate(layer_nonzero, highway_total),
            "unparseable_layer_count": layer_unparseable,
            "bridge_tag_true": _count_rate(bridge_yes, highway_total),
            "tunnel_tag_true": _count_rate(tunnel_yes, highway_total),
            "any_grade_separation_tag": _count_rate(grade_separation_any, highway_total),
            "tag_value_counts": {
                name: dict(values.most_common()) for name, values in tag_value_counts.items()
            },
            "observed_edge_osm_vs_stage0": dict(observed_static_comparison),
        },
        "graph_identity": {
            "raw_highway_way_count": highway_total,
            "observed_graph_osm_way_count": len(observed_way_ids),
            "observed_graph_ways_found_as_raw_highway_ways": _count_rate(
                len(observed_way_raw_match), len(observed_way_ids)
            ),
            "raw_tagged_node_count_exposed_by_gdal": len(tagged_node_ids),
            "raw_untagged_way_node_ids_exposed_by_gdal": False,
            "identity_scope_note": (
                "OSM node IDs inside Stage0 are available, but GDAL points includes tagged nodes "
                "rather than the full raw node table"
            ),
        },
    }
    helpers = {
        "signal_ids": signal_ids,
        "highway_way_ids": highway_way_ids,
        "raw_way_tags": raw_way_tags,
    }
    return summary, helpers


def inventory_poi(path: Path, search_terms: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    columns: list[str] | None = None
    total = valid_coordinates = 0
    match_counts = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    matched_major: dict[str, Counter[str]] = defaultdict(Counter)
    matched_middle: dict[str, Counter[str]] = defaultdict(Counter)
    major_counts: Counter[str] = Counter()
    middle_counts: Counter[str] = Counter()
    for chunk in pd.read_csv(path, encoding="utf-8-sig", chunksize=100_000, low_memory=False):
        if columns is None:
            columns = [str(column) for column in chunk.columns]
        total += len(chunk)
        name = chunk.iloc[:, 0].fillna("").astype(str)
        major = chunk.iloc[:, 1].fillna("").astype(str)
        middle = chunk.iloc[:, 2].fillna("").astype(str)
        lon = pd.to_numeric(chunk.iloc[:, 3], errors="coerce")
        lat = pd.to_numeric(chunk.iloc[:, 4], errors="coerce")
        valid_coordinates += int(lon.between(70, 140).mul(lat.between(10, 60)).sum())
        major_counts.update(major.value_counts().to_dict())
        middle_counts.update(middle.value_counts().to_dict())
        text = (name + "|" + major + "|" + middle).str.lower()
        for group, terms in search_terms.items():
            mask = pd.Series(False, index=chunk.index)
            for term in terms:
                mask |= text.str.contains(str(term).lower(), regex=False, na=False)
            match_counts[group] += int(mask.sum())
            matched_major[group].update(major.loc[mask].value_counts().to_dict())
            matched_middle[group].update(middle.loc[mask].value_counts().to_dict())
            if len(examples[group]) < 20:
                examples[group].extend(
                    value for value in name.loc[mask].drop_duplicates().tolist()
                    if value not in examples[group]
                )
                examples[group] = examples[group][:20]
    return {
        "row_count": int(total),
        "columns": columns or [],
        "valid_coordinate_rows": _count_rate(valid_coordinates, total),
        "osm_identity_fields_present": False,
        "signal_or_junction_role": "corroboration_only",
        "search_results": {
            group: {
                "matched_rows": int(match_counts[group]),
                "matched_share": float(match_counts[group] / total) if total else None,
                "top_major_categories": dict(matched_major[group].most_common(10)),
                "top_middle_categories": dict(matched_middle[group].most_common(10)),
                "example_names": examples[group],
                "search_terms": list(terms),
            }
            for group, terms in search_terms.items()
        },
        "top_major_categories": dict(major_counts.most_common(20)),
        "top_middle_categories": dict(middle_counts.most_common(20)),
        "interpretation": (
            "name/category text can only corroborate nearby OSM evidence; no direct OSM node, "
            "way, signal-control, or junction identity is present"
        ),
    }


def _provenance_markdown(inventory: Mapping[str, Any]) -> str:
    stage0 = inventory["stage0_route_parts"]
    pbf = inventory["frozen_osm_pbf"]
    poi = inventory["poi"]
    speed = stage0["speed_limit"]
    maxspeed = pbf["maxspeed"]
    signals = pbf["traffic_signals"]
    restrictions = pbf["turn_restrictions"]
    roundabout = pbf["roundabout"]
    topology = pbf["topology_and_layers"]
    graph = inventory["graph_identity"]
    return f"""# Stage 3 S1 static field provenance

Status: `{inventory['status']}`

S2 authorized: `NO`

This is a read-only inventory. No intersection clustering, tolerance selection,
static enrichment, capability calibration, Stage 2 inference, route assessment,
or Test31 fitting was performed.

## Population boundary

- Frozen Stage0/Stage1 route parts: {stage0['population']['route_part_rows']:,} rows.
- Observed canonical segment IDs: {stage0['population']['observed_unique_canonical_segment_ids']:,}.
- Observed directed identities (`canonical_edge_uid` + direction): {stage0['population']['observed_unique_directed_canonical_identities']:,}.
- This is the accepted-order observed subnetwork, not the complete routable graph.
- Legacy full `canonical_edges.parquet` present: `{str(graph['legacy_full_canonical_edges_present']).lower()}`.

## Priority fields

| Field | Source actually present | Coverage/result | Provenance limit |
|---|---|---|---|
| `speed_limit` | Stage0 route parts / Valhalla edge field | edge positive coverage {speed['unique_edge_weighted']['coverage_positive']['share']:.3%}; null {speed['unique_edge_weighted']['null']['count']:,}; zero {speed['unique_edge_weighted']['zero']['count']:,} | no per-edge source field; cannot call every value DIRECT posted speed |
| OSM `maxspeed` | frozen PBF highway ways | base-tag coverage {maxspeed['base_tag_coverage']['share']:.3%}; parseable among tagged {maxspeed['base_tag_parseability']['share']:.3%} | legal defaults are not inferred in S1 |
| signalization | OSM `highway=traffic_signals` nodes | {signals['raw_highway_traffic_signals_nodes']:,} raw; endpoint-ID mapping {signals['mapped_to_observed_graph_endpoint_by_osm_node_id']['share']:.3%} | mapping denominator is the observed accepted-order subnetwork |
| turn restriction | OSM `type=restriction` relations | {restrictions['restriction_relation_count']:,} relations; all member ways observed {restrictions['all_member_ways_present_in_observed_graph']['share']:.3%} | GDAL does not expose member roles/refs; exact directed enforcement not certified |

## Stage0 speed-limit facts

The field is numeric and assumed km/h by the Valhalla contract. Positive
coverage is reported by route-part row, route length, unique canonical edge, and
road class in the JSON inventory. Values below
{speed['diagnostic_abnormal_definition']['low_positive_below_kmh']:.0f} km/h or above
{speed['diagnostic_abnormal_definition']['high_above_kmh']:.0f} km/h are descriptive
diagnostic flags only, not Stage3 thresholds. Stage0 has no `speed_limit_source`,
so OSM agreement does not prove DIRECT provenance and disagreement does not by
itself prove an error.

## Frozen OSM maxspeed

- Highway ways: {maxspeed['highway_way_population']:,}.
- Base `maxspeed`: {maxspeed['base_tag_coverage']['count']:,}.
- Directional-only ways: {maxspeed['directional_only_way_count']:,}.
- Conditional ways: {maxspeed['conditional_way_count']:,}.
- Parseable base values: {maxspeed['base_tag_parseability']['count']:,}.
- Motorway base coverage: {maxspeed['by_osm_highway']['motorway']['base_tag_coverage_share']:.3%};
  trunk: {maxspeed['by_osm_highway']['trunk']['base_tag_coverage_share']:.3%};
  primary: {maxspeed['by_osm_highway']['primary']['base_tag_coverage_share']:.3%};
  secondary: {maxspeed['by_osm_highway']['secondary']['base_tag_coverage_share']:.3%}.
- Observed canonical edges with parseable OSM base speed but missing positive
  Stage0 speed: {maxspeed['observed_edge_comparison']['osm_parseable_but_stage0_missing']:,}.

OSM `maxspeed`, `maxspeed:forward/backward`, and `maxspeed:conditional` therefore
remain distinct raw evidence. S2 must not replace missing posted limits with road
class or design speed.

## Signals and control

OSM is the primary signal source. Missing `highway=traffic_signals` is not evidence
of `UNSIGNALIZED_CONTROLLED`. POI has {poi['search_results']['signal']['matched_rows']:,}
signal-term rows and {poi['search_results']['junction']['matched_rows']:,}
junction-term rows, of which
{poi['search_results']['junction']['top_middle_categories'].get('公交站', 0):,} are
bus-stop records whose names merely contain “路口”. POI has no OSM identity
columns and is corroboration-only; it is not a signal/control inventory.

## Roundabouts

- `junction=roundabout` ways: {roundabout['junction_roundabout_way_count']:,}.
- `junction=roundabout` nodes: {roundabout['junction_roundabout_node_count']:,}.
- `highway=mini_roundabout` nodes: {roundabout['highway_mini_roundabout_node_count']:,}.
- Roundabout ways represented in the observed graph: {roundabout['roundabout_ways_represented_in_observed_graph']['share']:.3%}.

The dominant representation is way-level. S1 does not consolidate or construct
intersection complexes.

## Turn restrictions

Node-via/way-via counts are inferred only from relation geometry composition:
`{json.dumps(restrictions['via_type_counts_inferred_from_geometry'], ensure_ascii=False)}`.
Geometry members can often be associated with OSM ways, but exact `from`/`via`/`to`
roles and refs are unavailable through the current GDAL reader. A role-preserving
PBF reader is a prerequisite before S2 can enforce directed restrictions.

## Layer, bridge, tunnel, grade separation

- Nonzero OSM layer: {topology['nonzero_layer']['count']:,} highway ways.
- Bridge: {topology['bridge_tag_true']['count']:,}.
- Tunnel: {topology['tunnel_tag_true']['count']:,}.
- Any grade-separation tag: {topology['any_grade_separation_tag']['count']:,}.

Stage0 route parts retain bridge/tunnel booleans but not `layer`. Spatial
proximity alone is therefore insufficient for future intersection merging. On
the observed canonical segments, OSM identifies
{topology['observed_edge_osm_vs_stage0']['raw_bridge_true']:,} bridge segments,
while Stage0 marks {topology['observed_edge_osm_vs_stage0']['stage0_bridge_true']:,};
{topology['observed_edge_osm_vs_stage0']['raw_bridge_true_stage0_false']:,} need
OSM-side enrichment review. Tunnel tags agree on all
{topology['observed_edge_osm_vs_stage0']['raw_tunnel_true']:,} observed segments.

## Graph identity

Observed route parts retain canonical base-segment UID, explicit traversal
direction, OSM way ID, begin/end OSM node IDs, and canonical from/to nodes. Raw tagged signal nodes can
be joined exactly by OSM node ID. The GDAL PBF point layer does not expose all
untagged way nodes, and the cleaned legacy full canonical-edge table is absent.

## S2 decisions required after review

1. Choose a role-preserving OSM relation reader before turn-restriction enforcement.
2. Decide whether to reconstruct/export a complete frozen-network edge table from
   the frozen PBF/Valhalla tiles; do not treat the
   {stage0['population']['observed_unique_canonical_segment_ids']:,} observed route segments
   (or the {stage0['population']['stage0_freeze_unique_direct_observed_edge_count']:,}
   direct-observed subset) as the complete graph.
3. Define a defensible speed-source provenance adapter separating Valhalla,
   OSM base/directional/conditional values, legal defaults, and UNKNOWN.

`NEXT_PHASE_AUTHORIZED = NO`
"""


def run_inventory(
    *, repo_root: str | Path, config_path: str | Path
) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(repo_root).resolve()
    config_file = _resolve(root, str(config_path))
    config = _load_config(config_file, root)
    paths = {name: _resolve(root, value) for name, value in config["paths"].items()}
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=True)

    stage0_freeze = _read_json(paths["stage0_freeze_manifest"])
    stage1_release = _read_json(paths["stage1_release_manifest"])
    stage2_release = _read_json(paths["stage2_final_release_manifest"])
    if (
        stage0_freeze.get("freeze_status") != "FROZEN"
        or stage1_release.get("engineering_status") != "PASS"
        or stage2_release.get("status") != "STAGE2_FINAL_FROZEN"
        or stage2_release.get("stage3_authorized") is not False
    ):
        raise Stage3S1InventoryError("upstream frozen release identity changed")

    speed_config = config["speed_diagnostic_only"]
    route_summary, edges, route_helpers = inventory_stage0_route_parts(
        paths["stage1_input"],
        low_speed=float(speed_config["low_positive_kmh"]),
        high_speed=float(speed_config["high_kmh"]),
    )
    route_summary["population"]["stage0_freeze_unique_direct_observed_edge_count"] = int(
        stage0_freeze.get("coverage", {}).get("unique_canonical_edge_count", 0)
    )
    route_summary["population"]["edge_count_denominator_note"] = (
        "route_parts includes every traversed canonical edge; the Stage0 freeze count is "
        "limited to canonical edges with direct observations"
    )
    pbf_summary, _ = inventory_pbf(
        paths["pbf"],
        observed_edges=edges,
        endpoint_nodes=route_helpers["endpoint_nodes"],
        way_to_nodes=route_helpers["way_to_nodes"],
    )
    poi_summary = inventory_poi(paths["poi"], config["poi_search_terms"])

    pbf_descriptor = _descriptor(paths["pbf"], root)
    if pbf_descriptor["sha256"] != stage0_freeze.get("pbf_sha"):
        raise Stage3S1InventoryError("PBF hash differs from Stage0 freeze manifest")
    valhalla_manifest = _read_json(paths["valhalla_build_manifest"])
    legacy_path = paths.get("legacy_canonical_edges")
    graph_identity = {
        **route_summary["graph_identity"],
        **pbf_summary["graph_identity"],
        "legacy_full_canonical_edges_path": legacy_path.as_posix() if legacy_path else None,
        "legacy_full_canonical_edges_present": bool(legacy_path and legacy_path.is_file()),
        "complete_frozen_network_edge_table_available": bool(
            legacy_path and legacy_path.is_file()
        ),
        "valhalla_tiles_present": (
            Path(config["paths"]["valhalla_build_manifest"]).parent / "tiles"
        ).exists(),
    }

    inventory: dict[str, Any] = {
        "schema_version": INVENTORY_SCHEMA,
        "status": PHASE_STATUS,
        "phase": "S1_STATIC_DATA_INVENTORY",
        "execution_head": _git_head(root),
        "s1_authorized": True,
        "s2_authorized": False,
        "next_phase_authorized": False,
        "authorizations": dict(config["authorizations"]),
        "scope": {
            "question": "What data do we actually have?",
            "intersection_clustering_performed": False,
            "threshold_calibration_performed": False,
            "model_inference_performed": False,
            "route_assessment_performed": False,
        },
        "sources": {
            "config": _descriptor(config_file, root),
            "stage0_freeze_manifest": _descriptor(paths["stage0_freeze_manifest"], root),
            "stage1_release_manifest": _descriptor(paths["stage1_release_manifest"], root),
            "stage2_final_release_manifest": _descriptor(
                paths["stage2_final_release_manifest"], root
            ),
            "frozen_osm_pbf": pbf_descriptor,
            "poi_csv": _descriptor(paths["poi"], root),
            "valhalla_build_manifest": _descriptor(paths["valhalla_build_manifest"], root),
            "route_parts_set": route_helpers["source_set"],
            "stage0_frozen_tiles_sha256": stage0_freeze.get("valhalla_tiles_sha"),
            "valhalla_build_manifest_payload": valhalla_manifest,
        },
        "stage0_route_parts": route_summary,
        "frozen_osm_pbf": pbf_summary,
        "poi": poi_summary,
        "graph_identity": graph_identity,
        "priority_findings": {
            "speed_limit": (
                "Stage0 numeric coverage is measurable but per-edge source provenance is absent; "
                "raw OSM directional/conditional tags remain separate evidence"
            ),
            "signalization": (
                "OSM traffic-signal nodes support exact node-ID joins to observed graph endpoints; "
                "missing tags cannot be interpreted as unsignalized"
            ),
            "turn_restriction": (
                "restriction relations exist, but exact directed mapping is not certified without "
                "member roles/refs"
            ),
        },
        "s2_prerequisites_for_review": [
            "role-preserving PBF relation reader for exact from/via/to restriction mapping",
            "complete frozen-network edge export or an explicitly observed-subnetwork-only S2 scope",
            "speed provenance adapter that does not treat road class/design speed as posted speed",
        ],
        "limitations": [
            "legacy full canonical_edges.parquet is absent",
            "GDAL points exposes tagged nodes, not the complete raw OSM node table",
            "POI has no OSM identity fields and is corroboration-only",
        ],
        "s1_blockers": [],
        "runtime_s": float(time.perf_counter() - started),
    }
    inventory["artifact_sha256"] = _payload_hash(inventory)
    inventory_path = output / "stage3_static_data_inventory.json"
    provenance_path = output / "stage3_static_field_provenance.md"
    _atomic_json(inventory_path, inventory)
    _atomic_text(provenance_path, _provenance_markdown(inventory))
    return inventory


def build_evidence(
    *, repo_root: str | Path, config_path: str | Path
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config_file = _resolve(root, str(config_path))
    config = _load_config(config_file, root)
    output = _resolve(root, config["paths"]["output"])
    inventory_path = output / "stage3_static_data_inventory.json"
    provenance_path = output / "stage3_static_field_provenance.md"
    test_path = output / "stage3_s1_test_evidence.json"
    inventory = _read_json(inventory_path)
    if inventory.get("artifact_sha256") != _payload_hash(inventory):
        raise Stage3S1InventoryError("S1 inventory embedded hash does not resolve")
    evidence = {
        "schema_version": EVIDENCE_SCHEMA,
        "status": "PASS",
        "phase_status": PHASE_STATUS,
        "execution_head": inventory["execution_head"],
        "artifacts": {
            "inventory": _descriptor(inventory_path, root),
            "field_provenance": _descriptor(provenance_path, root),
            "config": _descriptor(config_file, root),
            "implementation": _descriptor(Path(__file__), root),
            "tests": _descriptor(root / "stage3/tests_odd_tod/test_static_inventory.py", root),
            "test_evidence": _descriptor(test_path, root),
        },
        "frozen_upstream": {
            key: value
            for key, value in inventory["sources"].items()
            if key in {
                "stage0_freeze_manifest", "stage1_release_manifest",
                "stage2_final_release_manifest", "frozen_osm_pbf", "poi_csv",
                "valhalla_build_manifest",
            }
        },
        "scope_guards": inventory["scope"],
        "authorizations": inventory["authorizations"],
        "s2_authorized": False,
        "next_phase_authorized": False,
    }
    evidence["artifact_sha256"] = _payload_hash(evidence)
    return evidence


def verify_inventory(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_authorizations = {
        "s1": True,
        **{f"s{phase}": False for phase in range(2, 9)},
        "stage4": False,
    }
    if (
        payload.get("schema_version") != INVENTORY_SCHEMA
        or payload.get("status") != PHASE_STATUS
        or payload.get("artifact_sha256") != _payload_hash(payload)
        or payload.get("s2_authorized") is not False
        or payload.get("next_phase_authorized") is not False
        or payload.get("authorizations") != expected_authorizations
        or payload.get("scope", {}).get("intersection_clustering_performed") is not False
        or payload.get("scope", {}).get("threshold_calibration_performed") is not False
        or payload.get("scope", {}).get("model_inference_performed") is not False
        or payload.get("scope", {}).get("route_assessment_performed") is not False
    ):
        raise Stage3S1InventoryError("invalid S1 inventory identity or scope")
    priority = payload.get("priority_findings", {})
    if set(priority) != {"speed_limit", "signalization", "turn_restriction"}:
        raise Stage3S1InventoryError("S1 priority inventory is incomplete")
    restrictions = payload.get("frozen_osm_pbf", {}).get("turn_restrictions", {})
    if restrictions.get("exact_current_directed_network_mapping_certified") is not False:
        raise Stage3S1InventoryError("S1 overclaims turn-restriction mapping")
    if payload.get("poi", {}).get("signal_or_junction_role") != "corroboration_only":
        raise Stage3S1InventoryError("S1 overclaims POI authority")
    return {
        "schema_version": "stage3_s1_static_inventory_verification.1",
        "status": "PASS",
        "phase_status": PHASE_STATUS,
        "s2_authorized": False,
        "next_phase_authorized": False,
    }


def verify_evidence(
    payload: Mapping[str, Any], *, repo_root: str | Path
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    expected_authorizations = {
        "s1": True,
        **{f"s{phase}": False for phase in range(2, 9)},
        "stage4": False,
    }
    if (
        payload.get("schema_version") != EVIDENCE_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("phase_status") != PHASE_STATUS
        or payload.get("artifact_sha256") != _payload_hash(payload)
        or payload.get("s2_authorized") is not False
        or payload.get("next_phase_authorized") is not False
        or payload.get("authorizations") != expected_authorizations
    ):
        raise Stage3S1InventoryError("invalid S1 evidence identity or authorization")
    resolved = 0
    for section in ("artifacts", "frozen_upstream"):
        for descriptor in payload.get(section, {}).values():
            path = Path(str(descriptor.get("path", "")))
            path = path if path.is_absolute() else root / path
            if (
                not path.is_file()
                or _sha256(path) != descriptor.get("sha256")
                or int(path.stat().st_size) != int(descriptor.get("size_bytes", -1))
            ):
                raise Stage3S1InventoryError(
                    f"S1 evidence descriptor does not resolve: {descriptor.get('path')}"
                )
            resolved += 1
    inventory_descriptor = payload["artifacts"]["inventory"]
    inventory_path = Path(inventory_descriptor["path"])
    inventory_path = inventory_path if inventory_path.is_absolute() else root / inventory_path
    verification = verify_inventory(_read_json(inventory_path))
    test_descriptor = payload["artifacts"]["test_evidence"]
    test_path = Path(test_descriptor["path"])
    test_path = test_path if test_path.is_absolute() else root / test_path
    test_evidence = _read_json(test_path)
    if (
        test_evidence.get("schema_version") != TEST_EVIDENCE_SCHEMA
        or test_evidence.get("status") != "PASS"
        or test_evidence.get("s2_authorized") is not False
        or test_evidence.get("next_phase_authorized") is not False
        or any(check.get("status") != "PASS" for check in test_evidence.get("checks", {}).values())
    ):
        raise Stage3S1InventoryError("S1 test evidence is not passing")
    return {
        "schema_version": "stage3_s1_static_inventory_evidence_verification.1",
        "status": "PASS",
        "phase_status": verification["phase_status"],
        "resolved_descriptor_count": resolved,
        "s2_authorized": False,
        "next_phase_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--config", default="stage3/config/stage3_s1_static_inventory.json"
    )
    parser.add_argument(
        "--build-evidence", action="store_true",
        help="bind completed S1 outputs and tests without rerunning the inventory scan",
    )
    args = parser.parse_args(argv)
    if args.build_evidence:
        root = Path(args.repo_root).resolve()
        config_file = _resolve(root, args.config)
        config = _load_config(config_file, root)
        output = _resolve(root, config["paths"]["output"])
        evidence = build_evidence(repo_root=root, config_path=config_file)
        _atomic_json(output / "stage3_s1_evidence_bundle.json", evidence)
        print(json.dumps(verify_evidence(evidence, repo_root=root), indent=2, sort_keys=True))
        return 0
    inventory = run_inventory(repo_root=args.repo_root, config_path=args.config)
    print(json.dumps(verify_inventory(inventory), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
