# Stage 2 v5.2 methodology

## Status and scope

This commit completes Phase A implementation only. No bucket PoC, model training,
development evaluation, rolling backtest, legacy benchmark, or performance
benchmark has been run. Accordingly, Stage 2 remains
`NOT_READY_IMPLEMENTATION_ONLY`.

The frozen research question is prediction, at dispatch time and from then-known
history only, of multidimensional micro operating conditions along the historical
original service route. The formal micro targets are crawl share, stop share,
bounded speed CV, bounded acceleration RMS, and raw RTS. Pace/travel-time P50 is
retained as a common operational service-time variable, not as the source of
HV/AV differentiation and not as the only formal target.

HV always uses the historical order's original route; Stage 2 never replans it.
For a future AV candidate, Stage 3 will first evaluate the original route. Only
when that route is unsuitable may Stage 3 search for at most one constrained
fallback. Path search, AV capability thresholds, fallback selection, HV/AV
assignment, and Stage 4 optimization are outside v5.2.

## Frozen upstream and dates

Stage 0, Stage 1, and v5/v5.1 remain frozen. v5.2 uses existing 20161009–20161027
products for development and rolling backtesting and 20161031 only as a legacy
frozen benchmark. It must not produce 20161028–20161030 or call any Stage 0/1
production command. Results are development/rolling backtesting, never an
untouched confirmatory test.

## Products

`micro_condition_tokens` contains one row per physical traversal with the five
micro predictions, P50 pace/time, target-availability masks, Train-only support,
and model/time provenance. `original_route_micro_conditions` contains one row
per original route. Dynamic exposure weights are predicted traversal P50 travel
time. P90 is a weighted empirical quantile. High exposure means the frozen Train
empirical CDF is at least 0.90. Longest consecutive exposure respects
`route_sequence`. Missing predictions reduce coverage and remain missing.

RTS is reported with time and distance weighting; time weighting is the formal
downstream default. No composite grade is created.

Static route complexity is separate. The frozen upstream product provides
`canonical_highway`, `road_class`, `bridge`, and `tunnel`. Intersection, signal,
merge, turn, ramp, speed-limit, and lane fields are unavailable in the formal
upstream schema and are emitted as NA rather than reconstructed or filled with
zero.

## Spatial transfer

S0 is current v5.1 identity, S1 structure-only, S2 ID/structure concatenation,
and S3 support-aware transfer. S3 uses

\[
e_l = \lambda_l e_l^{ID} + (1-\lambda_l)e_l^{STRUCT},\qquad
\lambda_l = n_l/(n_l+\tau).
\]

Support and support groups are fitted on Train only. Unseen edges have `n=0` and
therefore use structure only. Candidate tau values are Train support P25, P50,
and P75; development validation selects once, then rolling folds reuse the
frozen choice.

## Temporal transfer

T0 has no adapter, T1 uses a zero-shot two-layer bottleneck adapter, and T2 is an
optional causal online extension. Adapter parameters are hard-capped at 10% of
the shared backbone. T2 may update every 60 minutes from completed observations
whose `observation_end_time < decision_time`, excluding the current order.
Future, unfinished, whole-day, and current-order labels are prohibited.

## Adoption and stopping

S3 is adopted only if at least three of five low-support targets improve, their
mean relative improvement exceeds 2%, no overall target degrades over 2%, and
unseen performance is no worse than structure-only. Otherwise v5.1 is retained
and negative transfer is reported.

The temporal adapter requires improvement on at least four of six rolling dates
and more than 1% mean improvement across five targets. Optional task pretraining
requires stable improvement both overall and on low-support rows. Pace P50 MAE
may not degrade more than 2% versus v5.1.

P90/P95 strict coverage, mean/std/CVaR, copulas, and route scenarios remain
diagnostic/appendix work and do not block Stage 3.

## Phase gates

Phase B is one fixed Train bucket plus one fixed validation bucket. Phase C is
one development day. Full rolling evaluation is allowed only after product,
leakage, scientific, and 10k/50k/100k/500k performance gates pass. A fivefold
input increase with runtime ratio above 8 is performance-suspicious and blocks
full execution.
