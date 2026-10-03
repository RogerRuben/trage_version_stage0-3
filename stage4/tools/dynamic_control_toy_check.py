"""Nine predeclared synthetic checks; no production imports, data, or writes."""

from fractions import Fraction
from itertools import combinations
import json


def graph(action, scenario):
    positions = {"H": "L", "A": "R"} if action == "x_A" else {"H": "R", "A": "L"}
    restricted_origin = "L" if scenario == 1 else "R"
    return tuple(
        (vehicle, origin)
        for vehicle in ("H", "A")
        for origin in ("L", "R")
        if positions[vehicle] == origin
        and (vehicle == "H" or origin != restricted_origin)
    )


def max_matching(edges):
    best = 0
    for size in range(len(edges) + 1):
        for selected in combinations(edges, size):
            if len({v for v, _ in selected}) == size and len({r for _, r in selected}) == size:
                best = max(best, size)
    return best


def main():
    # Declared before evaluation: 4 action/scenario cells + 5 probability cases.
    expected = {("x_A", 1): 2, ("x_A", 2): 1, ("x_H", 1): 1, ("x_H", 2): 2}
    probabilities = (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1))
    cells = []
    counts = {}
    for (action, scenario), target in expected.items():
        edges = graph(action, scenario)
        actual = max_matching(edges)
        assert actual == target
        counts[action, scenario] = actual
        cells.append({"action": action, "scenario": scenario, "edges": edges, "matching": actual})
    cases = []
    for p in probabilities:
        va = p * counts["x_A", 1] + (1 - p) * counts["x_A", 2]
        vh = p * counts["x_H", 1] + (1 - p) * counts["x_H", 2]
        assert va == 1 + p and vh == 2 - p and va - vh == 2 * p - 1
        optimal = max(va, vh)
        simple_action = "x_A" if p >= Fraction(1, 2) else "x_H"
        simple_value = va if simple_action == "x_A" else vh
        assert simple_value == optimal
        # Same two actions, but scenario is revealed before action: exact finite-model bound.
        clairvoyant = p * max(counts["x_A", 1], counts["x_H", 1]) + (1 - p) * max(counts["x_A", 2], counts["x_H", 2])
        assert clairvoyant == 2 and clairvoyant - optimal == min(p, 1 - p)
        for q in (Fraction(0), Fraction(1, 2), Fraction(1)):
            randomized = q * va + (1 - q) * vh
            assert randomized == 2 - p + q * (2 * p - 1)
            assert randomized <= optimal
        cases.append({"p": str(p), "V_x_A": str(va), "V_x_H": str(vh), "optimal": str(optimal), "simple_action": simple_action, "clairvoyant": str(clairvoyant), "information_relaxation_gap": str(clairvoyant - optimal)})
    print(json.dumps({"status": "PASS", "declared_cases": 9, "synthetic_only": True, "production_imports": False, "cells": cells, "probability_cases": cases}, indent=2))


if __name__ == "__main__":
    main()
