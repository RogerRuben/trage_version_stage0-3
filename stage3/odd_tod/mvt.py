"""Minimal, dependency-free Mapbox Vector Tile decoder used for Valhalla tiles."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any


def _varints(data: bytes) -> Iterator[tuple[int, int]]:
    pos = 0
    while pos < len(data):
        value = 0
        shift = 0
        while True:
            byte = data[pos]
            pos += 1
            value |= (byte & 0x7F) << shift
            if byte < 0x80:
                break
            shift += 7
        yield value, pos


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, pos
        shift += 7


def _fields(data: bytes) -> Iterator[tuple[int, int, Any]]:
    pos = 0
    while pos < len(data):
        key, pos = _read_varint(data, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _read_varint(data, pos)
        elif wire == 1:
            value = data[pos : pos + 8]
            pos += 8
        elif wire == 2:
            size, pos = _read_varint(data, pos)
            value = data[pos : pos + size]
            pos += size
        elif wire == 5:
            value = data[pos : pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield number, wire, value


def _packed(data: bytes) -> list[int]:
    values: list[int] = []
    pos = 0
    while pos < len(data):
        value, pos = _read_varint(data, pos)
        values.append(value)
    return values


def _value(data: bytes) -> Any:
    import struct

    for number, wire, raw in _fields(data):
        if number == 1:
            return raw.decode("utf-8")
        if number == 2:
            return struct.unpack("<f", raw)[0]
        if number == 3:
            return struct.unpack("<d", raw)[0]
        if number in (4, 5):
            return int(raw)
        if number == 6:
            return (raw >> 1) ^ -(raw & 1)
        if number == 7:
            return bool(raw)
    return None


def _geometry(commands: list[int], geom_type: int) -> Any:
    x = y = 0
    parts: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    pos = 0
    while pos < len(commands):
        command = commands[pos] & 7
        count = commands[pos] >> 3
        pos += 1
        if command in (1, 2):
            if command == 1 and current:
                parts.append(current)
                current = []
            for _ in range(count):
                dx, dy = commands[pos], commands[pos + 1]
                pos += 2
                x += (dx >> 1) ^ -(dx & 1)
                y += (dy >> 1) ^ -(dy & 1)
                current.append((x, y))
        elif command == 7:
            if current and current[0] != current[-1]:
                current.append(current[0])
        else:
            raise ValueError(f"unsupported MVT geometry command {command}")
    if current:
        parts.append(current)
    if geom_type == 1:
        points = [point for part in parts for point in part]
        return points[0] if len(points) == 1 else points
    return parts[0] if len(parts) == 1 else parts


def decode_tile(data: bytes) -> dict[str, list[dict[str, Any]]]:
    """Decode an MVT payload into layer feature dictionaries."""

    decoded: dict[str, list[dict[str, Any]]] = {}
    for number, wire, layer_raw in _fields(data):
        if number != 3 or wire != 2:
            continue
        name = ""
        keys: list[str] = []
        values: list[Any] = []
        feature_raws: list[bytes] = []
        extent = 4096
        for field, _, raw in _fields(layer_raw):
            if field == 1:
                name = raw.decode("utf-8")
            elif field == 2:
                feature_raws.append(raw)
            elif field == 3:
                keys.append(raw.decode("utf-8"))
            elif field == 4:
                values.append(_value(raw))
            elif field == 5:
                extent = int(raw)
        features: list[dict[str, Any]] = []
        for feature_raw in feature_raws:
            feature_id = None
            tags: list[int] = []
            geom_type = 0
            geometry: list[int] = []
            for field, _, raw in _fields(feature_raw):
                if field == 1:
                    feature_id = int(raw)
                elif field == 2:
                    tags = _packed(raw)
                elif field == 3:
                    geom_type = int(raw)
                elif field == 4:
                    geometry = _packed(raw)
            properties = {
                keys[tags[index]]: values[tags[index + 1]]
                for index in range(0, len(tags), 2)
            }
            features.append(
                {
                    "id": feature_id,
                    "properties": properties,
                    "type": geom_type,
                    "geometry": _geometry(geometry, geom_type),
                    "extent": extent,
                }
            )
        decoded[name] = features
    return decoded


def tile_point_to_lonlat(
    point: tuple[int, int], *, z: int, x: int, y: int, extent: int
) -> tuple[float, float]:
    """Convert a local vector-tile point to WGS84 lon/lat."""

    world_x = (x + point[0] / extent) / (1 << z)
    world_y = (y + point[1] / extent) / (1 << z)
    lon = world_x * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * world_y))))
    return lon, lat
