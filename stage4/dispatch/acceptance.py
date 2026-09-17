"""Exogenous, deterministic passenger acceptance scenarios for AV service."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class AcceptanceDecision:
    passenger_accepts_av: bool
    acceptance_source: str


def stable_acceptance_uniform(order_id: str, seed: int) -> float:
    """Return the frozen common-random-number draw for one order."""
    digest = hashlib.sha256(f"{int(seed)}|{order_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def passenger_acceptance(order_id: str, rate: float, seed: int) -> AcceptanceDecision:
    """Return an outcome-independent acceptance draw keyed only by order and seed."""
    rate = float(rate)
    if not 0.0 <= rate <= 1.0:
        raise ValueError("passenger_acceptance_rate must be in [0, 1]")
    uniform = stable_acceptance_uniform(order_id, seed)
    if rate == 1.0:
        return AcceptanceDecision(True, "ALL_ACCEPT_AV")
    return AcceptanceDecision(uniform <= rate, "HASH_SCENARIO")
