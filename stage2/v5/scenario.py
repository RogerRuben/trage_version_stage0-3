"""Reproducible correlated traversal and route service-time scenarios."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .contracts import Stage2V5ContractError


@dataclass(frozen=True)
class ScenarioResult:
    traversal_time_s: np.ndarray
    route_time_s: np.ndarray
    route_codes: np.ndarray
    model_id: str
    seed: int
    input_hash: str


def aggregate_traversal_scenarios(
    route_inverse: np.ndarray,
    traversal_scenarios: np.ndarray,
    *,
    route_count: int | None = None,
) -> np.ndarray:
    inverse = np.asarray(route_inverse, dtype=np.int64)
    traversal = np.asarray(traversal_scenarios, dtype=np.float64)
    if traversal.ndim != 2 or traversal.shape[0] != len(inverse):
        raise Stage2V5ContractError("traversal scenario matrix shape is invalid")
    count = int(inverse.max() + 1) if route_count is None and len(inverse) else int(route_count or 0)
    output = np.zeros((count, traversal.shape[1]), dtype=np.float64)
    np.add.at(output, inverse, traversal)
    return output


def _input_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        contiguous = np.ascontiguousarray(values)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def generate_route_scenarios(
    route_id: np.ndarray,
    pace_log_mu: np.ndarray,
    pace_log_scale: np.ndarray,
    allocated_distance_m: np.ndarray,
    *,
    scenario_count: int,
    seed: int,
    model: str,
    shared_route_rho: float = 0.35,
    residual_block_id: np.ndarray | None = None,
) -> ScenarioResult:
    route = np.asarray(route_id).astype(str)
    mu = np.asarray(pace_log_mu, dtype=np.float64)
    sigma = np.exp(np.asarray(pace_log_scale, dtype=np.float64))
    distance = np.asarray(allocated_distance_m, dtype=np.float64)
    if not (len(route) == len(mu) == len(sigma) == len(distance)):
        raise Stage2V5ContractError("scenario input lengths differ")
    if scenario_count <= 0 or np.any(~np.isfinite(mu)) or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise Stage2V5ContractError("invalid scenario distribution inputs")
    if np.any(~np.isfinite(distance)) or np.any(distance <= 0):
        raise Stage2V5ContractError("scenario distances must be finite and positive")
    unique_routes, inverse = np.unique(route, return_inverse=True)
    route_count = len(unique_routes)
    rng = np.random.default_rng(int(seed))
    independent = rng.standard_normal((len(route), scenario_count))
    if model == "independent":
        latent = independent
    elif model == "shared_route_latent":
        rho = float(shared_route_rho)
        if not 0.0 <= rho < 1.0:
            raise Stage2V5ContractError("shared route rho must be in [0,1)")
        common = rng.standard_normal((route_count, scenario_count))[inverse]
        latent = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * independent
    elif model == "residual_block":
        if residual_block_id is None:
            raise Stage2V5ContractError("residual_block requires residual_block_id")
        block = np.asarray(residual_block_id).astype(str)
        if len(block) != len(route):
            raise Stage2V5ContractError("residual block length differs")
        combined = np.char.add(np.char.add(route, "|"), block)
        _, block_inverse = np.unique(combined, return_inverse=True)
        block_common = rng.standard_normal((int(block_inverse.max() + 1), scenario_count))[block_inverse]
        rho = float(shared_route_rho)
        latent = np.sqrt(rho) * block_common + np.sqrt(1.0 - rho) * independent
    else:
        raise Stage2V5ContractError(f"unknown scenario model: {model}")
    traversal = np.exp(mu[:, None] + sigma[:, None] * latent) * distance[:, None]
    route_time = aggregate_traversal_scenarios(
        inverse,
        traversal,
        route_count=route_count,
    )
    return ScenarioResult(
        traversal_time_s=traversal,
        route_time_s=route_time,
        route_codes=unique_routes,
        model_id=f"stage2_v5_route_scenarios.1:{model}",
        seed=int(seed),
        input_hash=_input_hash(route, mu, sigma, distance),
    )


def summarize_route_scenarios(
    result: ScenarioResult,
    *,
    service_time_threshold_s: np.ndarray | float,
) -> dict[str, np.ndarray]:
    values = result.route_time_s
    threshold = np.asarray(service_time_threshold_s, dtype=np.float64)
    if threshold.ndim == 0:
        threshold = np.full(values.shape[0], float(threshold))
    if threshold.shape != (values.shape[0],):
        raise Stage2V5ContractError("one external timeout threshold is required per route")
    q = np.quantile(values, [0.5, 0.9, 0.95], axis=1)
    p90 = q[1]
    p95 = q[2]
    cvar90 = np.where(values >= p90[:, None], values, np.nan)
    cvar95 = np.where(values >= p95[:, None], values, np.nan)
    return {
        "route_id": result.route_codes,
        "mean": values.mean(axis=1),
        "std": values.std(axis=1),
        "p50": q[0],
        "p90": p90,
        "p95": p95,
        "cvar90": np.nanmean(cvar90, axis=1),
        "cvar95": np.nanmean(cvar95, axis=1),
        "timeout_probability": (values > threshold[:, None]).mean(axis=1),
        "scenario_count": np.full(values.shape[0], values.shape[1], dtype=np.int64),
    }
