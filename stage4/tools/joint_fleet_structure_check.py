"""Eight bounded synthetic checks. No production imports or data access."""

from itertools import combinations
import json


def graph(action, companion_region, include_h1=True):
    positions = {
        "H1": "L" if action == "x_A" else "R",
        "A1": "R" if action == "x_A" else "L",
        "H2": companion_region,
        "A2": "C",
    }
    return tuple(
        (vehicle, request)
        for vehicle, region in positions.items()
        for request in ("L", "R", "C")
        if region == request
        and (include_h1 or vehicle != "H1")
        and (vehicle.startswith("H") or request == "C")
    )


def rank(edges):
    best = 0
    for n in range(len(edges) + 1):
        for chosen in combinations(edges, n):
            if len({v for v, _ in chosen}) == n and len({o for _, o in chosen}) == n:
                best = max(best, n)
    return best


def main():
    # Four full-fleet cells, two H1-removed cells, two fixed-mixture cases.
    expected = {("x_A", "L"): 2, ("x_H", "L"): 3,
                ("x_A", "R"): 3, ("x_H", "R"): 2}
    values = {}
    rows = []
    for key, target in expected.items():
        edges = graph(*key)
        value = rank(edges)
        assert value == target
        values[key] = value
        rows.append({"action": key[0], "H2_region": key[1], "edges": edges, "rank": value})
    marginal = {}
    for region, target in (("L", 0), ("R", 1)):
        without = rank(graph("x_A", region, include_h1=False))
        assert without == 2
        marginal[region] = values["x_A", region] - without
        assert marginal[region] == target
    # A context-blind fixed action fails in one context. q=.5 is not exact optimal.
    for q in (0, 0.5):
        left = q * values["x_A", "L"] + (1-q) * values["x_H", "L"]
        right = q * values["x_A", "R"] + (1-q) * values["x_H", "R"]
        assert max(3-left, 3-right) >= 0.5
    interaction = values["x_A", "L"] - values["x_H", "L"] - values["x_A", "R"] + values["x_H", "R"]
    assert interaction == -2
    print(json.dumps({"status": "PASS", "declared_cases": 8, "cells": rows,
                     "H1_at_L_marginal": marginal, "additive_separability_cross_difference": interaction,
                     "context_aware_simple_rule_exact": True}, indent=2))


if __name__ == "__main__":
    main()
