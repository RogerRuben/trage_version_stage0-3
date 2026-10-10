# Stage 2 v5.2 performance report

Status: **PASS — 14/14 real-kernel scaling checks passed**.

The benchmark covers support-aware edge representation, static structure
preprocessing, original-route micro aggregation, weighted quantile, and maximum
consecutive high exposure at 10k, 50k, 100k, and 500k rows. It recorded wall
time, rows/s, peak RSS, and fivefold scaling ratios. The maximum observed ratio
was `7.708`, below the blocking threshold of `8.0`.

The source audit rejects per-row `apply`, `groupby.apply`, row iterators, and
concat inside loops. Detailed measurements are stored in
`stage2_v5_2_performance.json` and `stage2_v5_2_performance.csv`.
