"""Coordinate interpretation utilities with explicit provenance."""

from __future__ import annotations

import numpy as np


def gcj02_to_wgs84(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    x, y = lon - 105.0, lat - 35.0
    dlat = -100 + 2 * x + 3 * y + 0.2 * y**2 + 0.1 * x * y + 0.2 * np.sqrt(np.abs(x))
    dlat += (20 * np.sin(6 * x * np.pi) + 20 * np.sin(2 * x * np.pi)) * 2 / 3
    dlat += (20 * np.sin(y * np.pi) + 40 * np.sin(y / 3 * np.pi)) * 2 / 3
    dlat += (160 * np.sin(y / 12 * np.pi) + 320 * np.sin(y * np.pi / 30)) * 2 / 3
    dlon = 300 + x + 2 * y + 0.1 * x**2 + 0.1 * x * y + 0.1 * np.sqrt(np.abs(x))
    dlon += (20 * np.sin(6 * x * np.pi) + 20 * np.sin(2 * x * np.pi)) * 2 / 3
    dlon += (20 * np.sin(x * np.pi) + 40 * np.sin(x / 3 * np.pi)) * 2 / 3
    dlon += (150 * np.sin(x / 12 * np.pi) + 300 * np.sin(x / 30 * np.pi)) * 2 / 3
    rad = lat / 180 * np.pi
    magic = np.sin(rad)
    magic = 1 - 0.00669342162296594323 * magic**2
    sqrt_magic = np.sqrt(magic)
    dlat = dlat * 180 / ((6335552.717000426 / (magic * sqrt_magic)) * np.pi)
    dlon = dlon * 180 / ((6378245.0 / sqrt_magic) * np.cos(rad) * np.pi)
    return lon - dlon, lat - dlat
