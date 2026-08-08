# Stage 2 v5.2 performance report

Status: **NOT RUN — performance gate not evaluated**.

The benchmark implementation covers support-aware edge representation, static
structure preprocessing, original-route micro aggregation, weighted quantile,
and maximum consecutive high exposure at 10k, 50k, 100k, and 500k rows. Wall
time, rows/s, peak RSS, and fivefold scaling ratios will be written during Phase
B/C. A ratio above 8 blocks full rolling execution.

The source audit rejects per-row `apply`, `groupby.apply`, row iterators, and
concat inside loops. This report does not claim PASS before those checks run.
