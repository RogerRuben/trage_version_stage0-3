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


@dataclass(frozen=True)
class CrossOrderScenarioResult:
    scenario: ScenarioResult
    system_scenario_id: np.ndarray
    network_shock_id: np.ndarray
    route_shock_id: np.ndarray
    correlation_model_id: str


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


def _normal_cdf_approximation(values: np.ndarray) -> np.ndarray:
    """Vectorized normal CDF approximation for Gaussian-copula uniforms."""
    x = np.asarray(values, dtype=np.float64) / np.sqrt(2.0)
    # Winitzki's approximation has sub-1e-3 error in the range relevant to
    # Monte Carlo ranks and avoids adding SciPy to the GPU worker environment.
    erf = np.sign(x) * np.sqrt(
        np.maximum(
            1.0
            - np.exp(
                -np.square(x)
                * (4.0 / np.pi + 0.147 * np.square(x))
                / (1.0 + 0.147 * np.square(x))
            ),
            0.0,
        )
    )
    return np.clip(0.5 * (1.0 + erf), 1.0e-7, 1.0 - 1.0e-7)


def _bounded_quantile_inverse(
    uniforms: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    p95: np.ndarray,
) -> np.ndarray:
    u = np.asarray(uniforms, dtype=np.float64)
    q50 = np.asarray(p50, dtype=np.float64)[:, None]
    q90 = np.asarray(p90, dtype=np.float64)[:, None]
    q95 = np.asarray(p95, dtype=np.float64)[:, None]
    lower = np.maximum(q50 - 2.0 * (q90 - q50), 1.0e-4)
    below = lower + (q50 - lower) * (u / 0.5)
    middle = q50 + (q90 - q50) * ((u - 0.5) / 0.4)
    upper_middle = q90 + (q95 - q90) * ((u - 0.9) / 0.05)
    # The top five percent is explicitly bounded at p95 + 4*(p95-p90).
    # It supports risk ranking without reconstructing the rejected v5 mean.
    upper = q95 + 4.0 * (q95 - q90) * ((u - 0.95) / 0.05)
    return np.where(
        u <= 0.5,
        below,
        np.where(u <= 0.9, middle, np.where(u <= 0.95, upper_middle, upper)),
    )


def generate_quantile_route_scenarios(
    route_id: np.ndarray,
    pace_p50: np.ndarray,
    pace_p90: np.ndarray,
    pace_p95: np.ndarray,
    allocated_distance_m: np.ndarray,
    *,
    scenario_count: int,
    seed: int,
    model: str,
    shared_route_rho: float = 0.35,
    residual_block_id: np.ndarray | None = None,
) -> ScenarioResult:
    """Generate bounded quantile-marginal scenarios with a Gaussian copula."""
    route = np.asarray(route_id).astype(str)
    p50 = np.asarray(pace_p50, dtype=np.float64)
    p90 = np.asarray(pace_p90, dtype=np.float64)
    p95 = np.asarray(pace_p95, dtype=np.float64)
    distance = np.asarray(allocated_distance_m, dtype=np.float64)
    if not (len(route) == len(p50) == len(p90) == len(p95) == len(distance)):
        raise Stage2V5ContractError("quantile scenario input lengths differ")
    if scenario_count <= 0 or np.any(~np.isfinite(np.column_stack((p50, p90, p95)))):
        raise Stage2V5ContractError("quantile scenario inputs must be finite")
    if np.any(p50 <= 0) or np.any(p50 > p90) or np.any(p90 > p95):
        raise Stage2V5ContractError("quantile scenario inputs must be positive and monotonic")
    if np.any(~np.isfinite(distance)) or np.any(distance < 0):
        raise Stage2V5ContractError("scenario distances must be finite and non-negative")
    unique_routes, inverse = np.unique(route, return_inverse=True)
    rng = np.random.default_rng(int(seed))
    independent = rng.standard_normal((len(route), scenario_count))
    rho = float(shared_route_rho)
    if not 0.0 <= rho < 1.0:
        raise Stage2V5ContractError("shared route rho must be in [0,1)")
    if model == "independent_quantile":
        latent = independent
    elif model == "shared_route_quantile":
        common = rng.standard_normal((len(unique_routes), scenario_count))[inverse]
        latent = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * independent
    elif model == "residual_block_quantile":
        if residual_block_id is None:
            raise Stage2V5ContractError("residual_block_quantile requires residual_block_id")
        block = np.asarray(residual_block_id).astype(str)
        if len(block) != len(route):
            raise Stage2V5ContractError("residual block length differs")
        combined = np.char.add(np.char.add(route, "|"), block)
        _, block_inverse = np.unique(combined, return_inverse=True)
        common = rng.standard_normal((int(block_inverse.max() + 1), scenario_count))[block_inverse]
        latent = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * independent
    else:
        raise Stage2V5ContractError(f"unknown quantile scenario model: {model}")
    uniforms = _normal_cdf_approximation(latent)
    traversal = _bounded_quantile_inverse(uniforms, p50, p90, p95) * distance[:, None]
    route_time = aggregate_traversal_scenarios(inverse, traversal, route_count=len(unique_routes))
    return ScenarioResult(
        traversal_time_s=traversal,
        route_time_s=route_time,
        route_codes=unique_routes,
        model_id=f"stage2_v5_1_quantile_route_scenarios.1:{model}",
        seed=int(seed),
        input_hash=_input_hash(route, p50, p90, p95, distance),
    )


def generate_cross_order_quantile_scenarios(
    route_id: np.ndarray,
    pace_p50: np.ndarray,
    pace_p90: np.ndarray,
    pace_p95: np.ndarray,
    allocated_distance_m: np.ndarray,
    *,
    network_time_bin: np.ndarray,
    region_time_bin: np.ndarray,
    highway_time_bin: np.ndarray,
    scenario_count: int,
    seed: int,
    network_weight: float = 0.12,
    region_weight: float = 0.10,
    highway_weight: float = 0.08,
    route_weight: float = 0.20,
) -> CrossOrderScenarioResult:
    """One coherent hierarchical traffic world shared by every route in a batch."""
    route = np.asarray(route_id).astype(str)
    network = np.asarray(network_time_bin).astype(str)
    region = np.asarray(region_time_bin).astype(str)
    highway = np.asarray(highway_time_bin).astype(str)
    if not (len(route) == len(network) == len(region) == len(highway)):
        raise Stage2V5ContractError("cross-order shock label lengths differ")
    weights = np.asarray(
        [network_weight, region_weight, highway_weight, route_weight], dtype=np.float64
    )
    if np.any(weights < 0) or float(weights.sum()) >= 1.0:
        raise Stage2V5ContractError("cross-order shock weights must be non-negative and sum below one")
    unique_routes, route_inverse = np.unique(route, return_inverse=True)
    _, network_inverse = np.unique(network, return_inverse=True)
    _, region_inverse = np.unique(region, return_inverse=True)
    _, highway_inverse = np.unique(highway, return_inverse=True)
    rng = np.random.default_rng(int(seed))
    network_shock = rng.standard_normal((int(network_inverse.max() + 1), scenario_count))[network_inverse]
    region_shock = rng.standard_normal((int(region_inverse.max() + 1), scenario_count))[region_inverse]
    highway_shock = rng.standard_normal((int(highway_inverse.max() + 1), scenario_count))[highway_inverse]
    route_shock = rng.standard_normal((len(unique_routes), scenario_count))[route_inverse]
    residual = rng.standard_normal((len(route), scenario_count))
    latent = (
        np.sqrt(weights[0]) * network_shock
        + np.sqrt(weights[1]) * region_shock
        + np.sqrt(weights[2]) * highway_shock
        + np.sqrt(weights[3]) * route_shock
        + np.sqrt(1.0 - float(weights.sum())) * residual
    )
    p50 = np.asarray(pace_p50, dtype=np.float64)
    p90 = np.asarray(pace_p90, dtype=np.float64)
    p95 = np.asarray(pace_p95, dtype=np.float64)
    distance = np.asarray(allocated_distance_m, dtype=np.float64)
    if np.any(~np.isfinite(np.column_stack((p50, p90, p95, distance)))):
        raise Stage2V5ContractError("cross-order scenario inputs must be finite")
    if np.any(p50 <= 0) or np.any(p50 > p90) or np.any(p90 > p95) or np.any(distance < 0):
        raise Stage2V5ContractError("cross-order scenario quantiles or distances are invalid")
    traversal = _bounded_quantile_inverse(
        _normal_cdf_approximation(latent), p50, p90, p95
    ) * distance[:, None]
    scenario = ScenarioResult(
        traversal_time_s=traversal,
        route_time_s=aggregate_traversal_scenarios(
            route_inverse, traversal, route_count=len(unique_routes)
        ),
        route_codes=unique_routes,
        model_id="stage2_v5_1_cross_order_scenarios.1:hierarchical_quantile_copula",
        seed=int(seed),
        input_hash=_input_hash(route, p50, p90, p95, distance, network, region, highway),
    )
    system_id = np.arange(scenario_count, dtype=np.int64)
    return CrossOrderScenarioResult(
        scenario=scenario,
        system_scenario_id=system_id,
        network_shock_id=system_id.copy(),
        route_shock_id=system_id.copy(),
        correlation_model_id="hierarchical_quantile_copula.network_region_highway_route_residual.1",
    )


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
    if np.any(~np.isfinite(distance)) or np.any(distance < 0):
        raise Stage2V5ContractError("scenario distances must be finite and non-negative")
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
