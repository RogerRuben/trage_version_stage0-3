# Stage 2 v5.2 methodology

## Status and scope

This commit completes Phase A.1 implementation only. No tests, bucket PoC,
model training, development evaluation, rolling backtest, legacy benchmark, or
performance benchmark has been run. Stage 2 therefore remains
`NOT_READY_IMPLEMENTATION_ONLY`.

The frozen research question is dispatch-time prediction, using only then-known
history, of multidimensional micro operating conditions along the historical
original service route. The formal targets are crawl share, stop share, bounded
speed CV, bounded acceleration RMS, and raw RTS. Pace/travel-time P50 remains a
common operational variable, not the source of HV/AV differentiation.

HV always uses the historical order's original route. Stage 2 does not replan.
Stage 3 may later assess a constrained AV fallback only when the original route
is unsuitable. Path search, AV thresholds, fallback selection, HV/AV assignment,
and Stage 4 optimization are outside v5.2.

## Frozen upstream and dates

Stage 0, Stage 1, and v5/v5.1 remain frozen. v5.2 uses existing
20161009–20161027 products for development and rolling evaluation; 20161031 is
only a legacy frozen benchmark. It must not produce 20161028–20161030 or invoke
Stage 0/1 production. These results are development/rolling backtests, never an
untouched confirmatory test.

## Products and provenance

`micro_condition_tokens` contains one row per physical traversal with five micro
predictions, P50 pace/time, target-availability masks, Train-only support,
model/time provenance, and frozen original-route provenance. Planned, fallback,
and oracle route inputs fail closed. `original_route_micro_conditions` contains
one row per original route.

Dynamic exposure weights are predicted traversal P50 travel time. P90 is a
weighted empirical quantile. High exposure uses a protocol- and model-specific
frozen Train CDF. Consecutive exposure respects `route_sequence`. Coverage
denominators use full physical route distance. Below minimum pace distance
coverage, `travel_time_p50_s` is NA while `partial_travel_time_p50_s` remains
diagnostic. Every micro dimension has its own distance coverage field.

RTS is reported with time and distance weighting; time weighting is the formal
downstream default. No composite grade is created.

Static route complexity is separate. The bounded frozen-schema audit found
`canonical_highway`, `road_class`, `bridge`, and `tunnel`. Stable joinable
intersection, signal, merge, turn, and ramp semantics are unavailable and are
emitted as NA, never reconstructed or filled with zero.

## Spatial transfer

S0 delegates to v5.1 and must be numerically identical under the same checkpoint
and batch. S1 replaces the edge categorical representation with structure only.
S2 replaces it with ID/structure concatenation and projection. S3 replaces it
with the support-aware representation

\[
e_l = \lambda_l e_l^{ID} + (1-\lambda_l)e_l^{STRUCT},\qquad
\lambda_l = n_l/(n_l+\tau).
\]

Numeric physical features are not modified by edge transfer. The edge slot,
PAD/UNSEEN indices, vocabulary order, and embedding sizes bind to frozen v5.1
preprocessing and are checked against the checkpoint. M2–M5 initialize the
shared backbone and edge-ID table from v5.1; structure and adapter branches are
fresh. After the optional frozen epoch, copied parameters use 0.1 times the
new-branch learning rate.

Support is fitted only from unique physical traversals whose actual split and
dates match protocol Train. Unseen edges have `n=0` and are structure-only. Tau
candidates are Train support P25/P50/P75. Tau is selected once using Train
20161009–20161018 and Validation 20161019–20161020 by macro normalized MAE over
crawl, expected stop share, speed CV, and acceleration RMS; ties prefer smaller
tau. Later dates, pace, and RTS cannot select tau.

## Temporal transfer

T0 has no adapter, T1 is a zero-shot two-layer bottleneck adapter, and T2 is an
optional causal online extension. The adapter acts on the shared hidden token
after input/history fusion and before local-route convolution and the Transformer:
`h_target = h_shared + A(h_shared, x_time)`.

Its named inputs are decision-hour sine/cosine, decision weekday, and log forecast
horizon. Arbitrary four-column tensors are rejected. Adapter parameters are at
most 10% of the shared backbone. T2 may update only from another order whose
observation end, order completion, and label availability all precede the
adaptation cutoff. Future, unfinished, whole-day, and current-order labels are
prohibited.

## Adoption and checkpointing

S3 is adopted only if at least three of four core low-support targets improve,
their mean relative improvement exceeds 2%, no overall core target degrades over
2%, and unseen performance is no worse than structure-only. Otherwise v5.1 is
retained and negative transfer is reported.

The temporal adapter requires improvement on at least four of six rolling dates
and over 1% mean improvement across the same four targets. RTS is a secondary
frozen-reference diagnostic: it cannot tune tau, decide spatial/temporal adoption,
or select a primary checkpoint. Pace P50 MAE may not degrade more than 2%.

Checkpoint selection first requires finite outputs, zero temporal leakage, and
the 2% pace guard. It then minimizes validation macro normalized MAE over the
four core targets, using low-support core MAE and pace P50 only as secondary keys.
RTS and route P90/P95/CVaR do not select early rolling checkpoints.

## Phase gates

Phase B0 is one fixed Train bucket and one fixed Validation bucket. Phase B1 is
the frozen transfer-tuning protocol. Phase C is one development day. Full rolling
is allowed only after product, leakage, scientific, and real-kernel performance
gates pass. This implementation-only commit authorizes none of those runs.
