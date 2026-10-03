"""Explicit nested research assumptions; not an AV safety certification model.

This opt-in interface does not modify the frozen Stage3 profiles. Missing
movement/control inputs are data errors, not a fourth capability category.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


PROFILES = ("C", "M", "A")
TRAFFIC_STATES = ("NC", "ML", "MR", "CG", "SC", "U")
CONTROL_BASES = ("POSITIVE_EVIDENCE", "RESEARCH_ASSUMPTION", "SYNTHETIC")


@dataclass(frozen=True)
class ResearchMovement:
    maneuver: str
    signalized: bool
    control_basis: str
    certified_prohibited: bool = False

    def __post_init__(self):
        if self.maneuver not in ("STRAIGHT", "RIGHT", "LEFT", "UTURN", "ROUNDABOUT"):
            raise ValueError("unresolved maneuver: supply an explicit research input")
        if type(self.signalized) is not bool or self.control_basis not in CONTROL_BASES:
            raise ValueError("control must be explicit, with evidence/assumption provenance")
        if self.control_basis == "POSITIVE_EVIDENCE" and not self.signalized:
            raise ValueError("positive-only evidence cannot prove absence of a signal")


@dataclass(frozen=True)
class ResearchCompatibility:
    profile_id: str
    compatible: bool
    reason_codes: tuple[str, ...]
    outside_time_share: float
    unknown_traffic_share: float


def evaluate_research_compatibility(
    profile_id: str,
    movements: tuple[ResearchMovement, ...],
    traffic_shares: dict[str, float],
    *,
    direction_routable: bool = True,
    outside_budget: float = 0.05,
    conservative_control_policy: str = "SIGNALIZED_CONFLICT_MOVEMENTS",
) -> ResearchCompatibility:
    """C: signalized left turns; M: unsignalized left/roundabout; A: U-turn too.

    Straight/right movements do not automatically require a signal under the
    main rule. ALL_ENCOUNTERS_SIGNALIZED is an explicit stricter sensitivity,
    not an implicit reinterpretation of the main policy. Static/variability/
    speed utilization stays in a separate Stage4 exposure-budget interface.
    """
    if profile_id not in PROFILES:
        raise ValueError("unrecognized profile")
    if conservative_control_policy not in (
        "SIGNALIZED_CONFLICT_MOVEMENTS", "ALL_ENCOUNTERS_SIGNALIZED"
    ):
        raise ValueError("unrecognized C control policy")
    if type(direction_routable) is not bool:
        raise ValueError("direction identity must be resolved")
    if set(traffic_shares) != set(TRAFFIC_STATES):
        raise ValueError("need the complete six-state predicted-time partition")
    values = tuple(float(traffic_shares[s]) for s in TRAFFIC_STATES)
    if (not all(isfinite(v) and 0 <= v <= 1 for v in values)
            or abs(sum(values) - 1) > 1e-6
            or not isfinite(outside_budget) or not 0 <= outside_budget <= 1):
        raise ValueError("invalid traffic partition or budget")
    reasons: set[str] = set()
    if not direction_routable:
        reasons.add("KNOWN_DIRECTION_UNROUTABLE")
    for movement in movements:
        if movement.certified_prohibited:
            reasons.add("CERTIFIED_PROHIBITED_MOVEMENT")
        if movement.maneuver == "UTURN" and profile_id in ("C", "M"):
            reasons.add("UTURN_OUTSIDE_RESEARCH_PROFILE")
        if profile_id == "C":
            if movement.maneuver == "ROUNDABOUT":
                reasons.add("ROUNDABOUT_OUTSIDE_RESEARCH_PROFILE")
            needs_signal = (movement.maneuver == "LEFT"
                            or conservative_control_policy == "ALL_ENCOUNTERS_SIGNALIZED")
            if needs_signal and not movement.signalized:
                reasons.add("UNSIGNALIZED_MOVEMENT_OUTSIDE_RESEARCH_PROFILE")
    outside_states = {"C": ("MR", "CG", "SC"), "M": ("SC",), "A": ()}
    outside = sum(float(traffic_shares[s]) for s in outside_states[profile_id])
    if outside > outside_budget + 1e-6:
        reasons.add("PREDICTED_ENVIRONMENT_OUTSIDE_BUDGET")
    return ResearchCompatibility(
        profile_id, not reasons, tuple(sorted(reasons)), outside,
        float(traffic_shares["U"]),
    )


def compatible_profiles(movements, traffic_shares, **kwargs) -> frozenset[str]:
    return frozenset(
        k for k in PROFILES
        if evaluate_research_compatibility(k, movements, traffic_shares, **kwargs).compatible
    )
