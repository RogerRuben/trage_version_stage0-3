"""Two explicit clocks: request-to-assignment and assignment-to-pickup."""
from __future__ import annotations

from math import isfinite


TOTAL_PICKUP_DEADLINE = "TOTAL_PICKUP_DEADLINE"
MATCH_DEADLINE_AND_PICKUP_CAP = "MATCH_DEADLINE_AND_PICKUP_CAP"


def match_deadline_mode(config):
    mode = config.get("patience_mode", TOTAL_PICKUP_DEADLINE)
    if mode not in (TOTAL_PICKUP_DEADLINE, MATCH_DEADLINE_AND_PICKUP_CAP):
        raise ValueError("unknown passenger waiting contract")
    return mode == MATCH_DEADLINE_AND_PICKUP_CAP


def pickup_eta_limit_s(config, remaining_match_s):
    if not match_deadline_mode(config):
        return float(remaining_match_s)
    limit = float(config["max_pickup_eta_s"])
    if not isfinite(limit) or limit <= 0:
        raise ValueError("independent pickup cap must be finite and positive")
    return limit


def assignment_feasible(config, now_s, request_deadline_s, pickup_eta_s):
    remaining = float(request_deadline_s) - float(now_s)
    eta = float(pickup_eta_s)
    return (isfinite(remaining) and remaining > 0 and isfinite(eta)
        and 0 <= eta <= pickup_eta_limit_s(config, remaining))


def pickup_rejection_code(config, *, cached=False):
    if match_deadline_mode(config):
        return "CACHED_PICKUP_ETA_EXCEEDS_INDEPENDENT_CAP" if cached else "PICKUP_ETA_EXCEEDS_INDEPENDENT_CAP"
    return "CACHED_ETA_EXCEEDS_REMAINING_PATIENCE" if cached else "PICKUP_ETA_EXCEEDS_REMAINING_PATIENCE"
