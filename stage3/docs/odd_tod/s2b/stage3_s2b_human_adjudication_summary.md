# Stage 3 S2B Human Adjudication Summary

Selected buffer radius: **10 m**.

Overall labels: `{"10_CORRECT": 7, "5_CORRECT": 0, "BOTH_ACCEPTABLE": 61, "NEITHER": 2, "UNCERTAIN": 0}`.

## Stratified diagnostic results

- `signalized`: `{"10_CORRECT": 3, "5_CORRECT": 0, "BOTH_ACCEPTABLE": 16, "NEITHER": 1, "UNCERTAIN": 0}`
- `multi_node_divided_road`: `{"10_CORRECT": 1, "5_CORRECT": 0, "BOTH_ACCEPTABLE": 19, "NEITHER": 0, "UNCERTAIN": 0}`
- `high_degree`: `{"10_CORRECT": 0, "5_CORRECT": 0, "BOTH_ACCEPTABLE": 10, "NEITHER": 0, "UNCERTAIN": 0}`
- `grade_separated`: `{"10_CORRECT": 2, "5_CORRECT": 0, "BOTH_ACCEPTABLE": 7, "NEITHER": 1, "UNCERTAIN": 0}`
- `random_changed`: `{"10_CORRECT": 1, "5_CORRECT": 0, "BOTH_ACCEPTABLE": 9, "NEITHER": 0, "UNCERTAIN": 0}`

## NEITHER cases — reviewer notes preserved verbatim

- `s2b11_008` (signalized): "This is a massive interchange with tons of ramps and entrances, so a 10-meter range just isn't enough
- `s2b11_055` (grade_separated): "This is a massive interchange with tons of ramps and entrances, so a 10-meter range just isn't enough

These two cases indicate that a single 10m radius can still under-consolidate very large interchange systems. They do not prefer 5m, and no manual topology override is added.

This was targeted stratified diagnostic review, not population-random sampling. It supports 10m over 5m but does not estimate population accuracy.
