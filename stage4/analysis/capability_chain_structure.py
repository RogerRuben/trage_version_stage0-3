"""Tiny synthetic chain-packing checks, NOT a FleetPy/replay experiment.

Inputs are declared feasible chain families, not GPS or routed city data.
Only the finite model's primary served-job objective is examined. No claims
about empirical frequency, policy gains, pricing tractability, or novelty.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import itertools
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

MAX_RESOURCES = 6
MAX_JOBS = 12
MAX_CHAINS_PER_RESOURCE = 64
MAX_DP_STATES = 50_000


def subchains(*full_chains):
    """Toy-only closure: skipping a service is declared feasible by the fixture."""
    found = {()}
    for chain in full_chains:
        chain = tuple(chain)
        if len(chain) > MAX_JOBS or len(set(chain)) != len(chain):
            raise ValueError("toy full chain is too long or repeats a job")
        for length in range(1, len(chain) + 1):
            for positions in itertools.combinations(range(len(chain)), length):
                found.add(tuple(chain[i] for i in positions))
                if len(found) > MAX_CHAINS_PER_RESOURCE:
                    raise ValueError("toy chain budget exceeded")
    return tuple(sorted(found, key=lambda path: (len(path), path)))


def _prepare(families, capacities):
    if not families or len(families) > MAX_RESOURCES or set(capacities) != set(families):
        raise ValueError("expected 1..6 named resources and matching capacities")
    names = tuple(sorted(families))
    normalized = {}
    for name in names:
        if type(capacities[name]) is not int or capacities[name] not in (0, 1):
            raise ValueError("tiny check capacities must be integer 0 or 1")
        paths = tuple(tuple(path) for path in families[name])
        if len(paths) > MAX_CHAINS_PER_RESOURCE or any(len(set(p)) != len(p) for p in paths):
            raise ValueError("chain limit exceeded or duplicate service in a chain")
        if any(not isinstance(job, str) or not job for p in paths for job in p):
            raise ValueError("job identities must be nonempty strings")
        normalized[name] = tuple(sorted(set(paths) | {()}, key=lambda p: (len(p), p)))
        if len(normalized[name]) > MAX_CHAINS_PER_RESOURCE:
            raise ValueError("normalized chain budget exceeded")
    jobs = tuple(sorted({job for paths in normalized.values() for p in paths for job in p}))
    if len(jobs) > MAX_JOBS:
        raise ValueError("toy job budget exceeded")
    return names, jobs, normalized


def exact_served(families, capacities):
    """Exact, bounded DP over resource index and already-served job bitmask."""
    names, jobs, normalized = _prepare(families, capacities)
    bits = {job: 1 << index for index, job in enumerate(jobs)}
    choices = {name: tuple((sum(bits[job] for job in p), p) for p in normalized[name]) for name in names}
    visited = 0

    @lru_cache(maxsize=MAX_DP_STATES)
    def solve(index, used):
        nonlocal visited
        visited += 1
        if visited > MAX_DP_STATES:
            raise ValueError("bounded exact DP state limit exceeded")
        if index == len(names):
            return 0, ()
        name = names[index]
        options = choices[name] if capacities[name] else ((0, ()),)
        best = (-1, ())
        for mask, path in options:
            if mask & used:
                continue
            value, suffix = solve(index + 1, used | mask)
            proposed = (value + len(path), ((name, path),) + suffix)
            if proposed[0] > best[0]:
                best = proposed
        return best

    value, selected = solve(0, 0)
    return dict(served=int(value), selected=[dict(resource=name, chain=list(path)) for name, path in selected],
                dp_states=visited, job_count=len(jobs), resource_count=len(names))


def relaxed_served(families, capacities):
    """Complete finite-column LP plus an actual feasible dual certificate."""
    names, jobs, normalized = _prepare(families, capacities)
    columns = [(name, p) for name in names for p in normalized[name] if p]
    if not columns:
        raise ValueError("LP fixture requires a nonempty service column")
    job_row = {job: index for index, job in enumerate(jobs)}
    vehicle_row = {name: len(jobs) + index for index, name in enumerate(names)}
    rows, cols, values = [], [], []
    for index, (name, path) in enumerate(columns):
        for job in path:
            rows.append(job_row[job]); cols.append(index); values.append(1.)
        rows.append(vehicle_row[name]); cols.append(index); values.append(1.)
    matrix = coo_matrix((values, (rows, cols)), shape=(len(jobs) + len(names), len(columns))).tocsr()
    rhs = np.r_[np.ones(len(jobs)), [capacities[name] for name in names]]
    rewards = np.array([len(path) for _, path in columns], float)
    result = linprog(-rewards, A_ub=matrix, b_ub=rhs, bounds=(0., None), method="highs")
    if not result.success:
        raise RuntimeError(f"tiny LP failed: {result.message}")
    prices = -np.asarray(result.ineqlin.marginals)
    dual_slack = np.asarray(matrix.T @ prices).ravel() - rewards
    value = float(-result.fun)
    primal_max_excess = float(np.max(np.asarray(matrix @ result.x).ravel() - rhs))
    dual_value = float(rhs @ prices)
    if (float(prices.min()) < -1e-8 or float(dual_slack.min()) < -1e-8
            or primal_max_excess > 1e-8 or abs(value - dual_value) > 1e-8):
        raise RuntimeError("primal/dual certificate mismatch")
    return dict(served=value, columns=len(columns), rows=matrix.shape[0], nonzeros=matrix.nnz,
                nonzero_solution=[dict(resource=name, chain=list(path), weight=float(weight))
                    for (name, path), weight in zip(columns, result.x) if weight > 1e-8],
                job_prices={job: float(prices[index]) for job, index in job_row.items()},
                resource_prices={name: float(prices[index]) for name, index in vehicle_row.items()},
                certificate=dict(primal_max_excess=primal_max_excess,
                    minimum_dual_slack=float(dual_slack.min()), dual_objective=dual_value,
                    primal_dual_gap=abs(value - dual_value)))


def verify_named_dual(families, capacities, alpha, beta):
    """Check a supplied dual against EVERY declared column, not a solver claim."""
    names, jobs, normalized = _prepare(families, capacities)
    if set(alpha) != set(jobs) or set(beta) != set(names):
        raise ValueError("dual identities do not match the finite model")
    prices = list(alpha.values()) + list(beta.values())
    if any(not np.isfinite(price) or price < 0 for price in prices):
        raise ValueError("dual prices must be finite and nonnegative")
    slacks = [sum(alpha[job] for job in path) + beta[name] - len(path)
              for name in names for path in normalized[name] if path]
    if not slacks or min(slacks) < -1e-8:
        raise ValueError("supplied dual violates a declared chain column")
    return dict(objective=float(sum(alpha.values()) + sum(capacities[name] * beta[name] for name in names)),
                minimum_dual_slack=float(min(slacks)), checked_columns=len(slacks),
                source="EXPLICIT_NAMED_CERTIFICATE_NOT_HIGHS_RETURNED_PRICES")


def run_checks():
    started = perf_counter()
    # A hotspot has more individual jobs, but no multi-job feasible chain.
    hotspot = exact_served({"C": subchains(("a",), ("b",), ("c",))}, {"C": 1})
    connected = exact_served({"C": subchains(("d", "e"))}, {"C": 1})
    assert hotspot["job_count"] > connected["job_count"] and hotspot["served"] < connected["served"]

    # Same initial resource; an extra business fence deletes the long chain.
    unrestricted = exact_served({"C": subchains(("a", "b"))}, {"C": 1})
    fenced = exact_served({"C": subchains(("a",))}, {"C": 1})
    assert unrestricted["served"] >= fenced["served"]

    # Nested INDIVIDUAL-task compatibility, NOT identical conditional chains.
    gap_families = {"C": subchains(("a", "b"), ("c", "d")),
                    "H": subchains(("a", "c"), ("b", "d"))}
    exact = exact_served(gap_families, {"C": 1, "H": 1})
    relaxed = relaxed_served(gap_families, {"C": 1, "H": 1})
    assert exact["served"] == 3 and abs(relaxed["served"] - 4.) < 1e-8

    # Two restricted resources are complementary in freeing the existing H.
    deployment = {"H": subchains(("a", "b", "c"), ("c", "d")),
                  "C1": subchains(("a",)), "C2": subchains(("b",))}
    all_values = {}
    for c1, c2 in itertools.product((0, 1), repeat=2):
        all_values[f"{c1}{c2}"] = exact_served(deployment, {"H": 1, "C1": c1, "C2": c2})["served"]
    marginal_alone = all_values["10"] - all_values["00"]
    marginal_after_other = all_values["11"] - all_values["01"]
    assert all_values == {"00": 3, "01": 3, "10": 3, "11": 4}
    assert marginal_after_other > marginal_alone

    # A zero optimal LP price does NOT imply zero discrete resource value.
    # The candidate's shorter admission window/late A state makes only A
    # feasible; this is not a same-window pure-location comparison.
    dual_families = {**gap_families, "C2": subchains(("a",))}
    before_caps = {"H": 1, "C": 1, "C2": 0}
    before_lp = relaxed_served(dual_families, before_caps)
    after_caps = {**before_caps, "C2": 1}
    after_lp = relaxed_served(dual_families, after_caps)
    before_ip = exact_served(dual_families, before_caps)["served"]
    after_ip = exact_served(dual_families, after_caps)["served"]
    alpha = {job: 1. for job in ("a", "b", "c", "d")}
    beta_prices = {name: 0. for name in dual_families}
    certificate = verify_named_dual(dual_families, before_caps, alpha, beta_prices)
    assert abs(certificate["objective"] - before_lp["served"]) < 1e-8
    beta = beta_prices["C2"]
    lp_increment = after_lp["served"] - before_lp["served"]
    ip_increment = after_ip - before_ip
    ip_gap = before_lp["served"] - before_ip
    assert lp_increment <= beta + 1e-8 and ip_increment <= ip_gap + beta + 1e-8
    assert beta == 0 and ip_increment == 1

    return dict(status="CHECKED_FINITE_SYNTHETIC_MODELS", kind="MATHEMATICAL_COUNTEREXAMPLES_NOT_REPLAY",
        proof_scope="declared complete finite chain families; no city embedding or novelty certification",
        checks=dict(hotspot_ordering=dict(hotspot=hotspot, continuation_friendly=connected),
            geofence=dict(unrestricted=unrestricted, fenced=fenced),
            fractional_relaxation=dict(exact=exact, relaxed=relaxed,
                integer_gap_jobs=relaxed["served"]-exact["served"],
                individual_task_sets_nested=True, equal_start_transition_graphs_claimed=False),
            deployment_complementarity=dict(values=all_values, marginal_alone=marginal_alone,
                marginal_after_other=marginal_after_other, submodularity_counterexample=True),
            dual_increment=dict(before_lp=before_lp["served"], after_lp=after_lp["served"],
                beta=beta, lp_increment=lp_increment, before_ip_gap=ip_gap,
                before_ip=before_ip, after_ip=after_ip, ip_increment=ip_increment,
                valid_ip_increment_upper_bound=ip_gap+beta,
                named_dual_certificate=certificate, beta_alone_is_not_a_valid_ip_gain_bound=True)),
        maximum_columns=max(relaxed["columns"], before_lp["columns"], after_lp["columns"]),
        maximum_nonzeros=max(relaxed["nonzeros"], before_lp["nonzeros"], after_lp["nonzeros"]),
        runtime_s=perf_counter()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=Path("stage4/docs/capability_chain_planning/toy_results.json"))
    args = parser.parse_args()
    result = run_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(dict(status=result["status"], output=str(args.output),
        runtime_s=result["runtime_s"], maximum_columns=result["maximum_columns"],
        maximum_nonzeros=result["maximum_nonzeros"])), flush=True)


if __name__ == "__main__":
    main()
