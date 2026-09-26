"""Coordinate conversion and metric helpers used before map matching."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


def gcj02_to_wgs84(
    lon: np.ndarray | list[float],
    lat: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same explicit GCJ-02 interpretation used by the v5 baseline."""

    lon_array = np.asarray(lon, dtype=float)
    lat_array = np.asarray(lat, dtype=float)
    x, y = lon_array - 105.0, lat_array - 35.0
    dlat = -100 + 2 * x + 3 * y + 0.2 * y**2 + 0.1 * x * y + 0.2 * np.sqrt(np.abs(x))
    dlat += (20 * np.sin(6 * x * np.pi) + 20 * np.sin(2 * x * np.pi)) * 2 / 3
    dlat += (20 * np.sin(y * np.pi) + 40 * np.sin(y / 3 * np.pi)) * 2 / 3
    dlat += (160 * np.sin(y / 12 * np.pi) + 320 * np.sin(y * np.pi / 30)) * 2 / 3
    dlon = 300 + x + 2 * y + 0.1 * x**2 + 0.1 * x * y + 0.1 * np.sqrt(np.abs(x))
    dlon += (20 * np.sin(6 * x * np.pi) + 20 * np.sin(2 * x * np.pi)) * 2 / 3
    dlon += (20 * np.sin(x * np.pi) + 40 * np.sin(x / 3 * np.pi)) * 2 / 3
    dlon += (150 * np.sin(x / 12 * np.pi) + 300 * np.sin(x / 30 * np.pi)) * 2 / 3
    rad = lat_array / 180 * np.pi
    magic = 1 - 0.00669342162296594323 * np.sin(rad) ** 2
    sqrt_magic = np.sqrt(magic)
    dlat = dlat * 180 / ((6_335_552.717000426 / (magic * sqrt_magic)) * np.pi)
    dlon = dlon * 180 / ((6_378_245.0 / sqrt_magic) * np.cos(rad) * np.pi)
    return lon_array - dlon, lat_array - dlat


def haversine_m(
    lon1: np.ndarray | float,
    lat1: np.ndarray | float,
    lon2: np.ndarray | float,
    lat2: np.ndarray | float,
) -> np.ndarray:
    lon1r, lat1r = np.radians(lon1), np.radians(lat1)
    lon2r, lat2r = np.radians(lon2), np.radians(lat2)
    dlon, dlat = lon2r - lon1r, lat2r - lat1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
