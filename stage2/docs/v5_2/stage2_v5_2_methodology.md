# Stage 2 v5.2 methodology

## Status and scope

Phase A.2 execution verification, Phase B0, and Phase B1 have passed. Phase B1
is now post-run frozen: its mutating commands are disabled, its evidence bundle
is hash-bound, and tau is write-once at `3.0` (`Train-support p25`). It used
only `20161009-20161018 -> 20161019-20161020`. Phase C/D, rolling folds, the
legacy 20161031 benchmark, and Stage 3 remain unopened and unauthorized.

The frozen research question is dispatch-time prediction, using only then-known
history, of multidimensional micro operating conditions along the historical
original service route. The predicted targets are crawl share, stop share,
bounded speed CV, bounded acceleration RMS, and raw RTS. The first four are the
core transfer targets and the only Stage 3 deployable micro conditions; RTS is a
legacy/descriptive diagnostic. Pace/travel-time P50 remains a common operational
variable, not the source of HV/AV differentiation.

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

Every formal v5.2 run first binds the exact v5.1 feature artifact, resolved
source config, model manifest, and checkpoint to frozen protocol dates and the
model constructor. Transfer tensors are an atomic product under
`protocol=<id>/split=<role>/date=<day>`; their manifests bind Stage 1, route,
feature, support, static, temporal-audit, and shard hashes. No manual leakage
count is accepted.

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
`service_time_complete_flag` requires at least 0.999 pace-distance coverage;
admissible lower coverage is labeled `partial_coverage_estimate`.

RTS is reported with time and distance weighting as a diagnostic only. Its
coverage and summaries do not affect deployable micro coverage, `unknown_flag`,
or any Stage 3 assignment decision. No composite grade is created.

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
Tau metrics come only from the formal checkpoint evaluator and bind M1 plus all
three M4 checkpoint/evaluation hashes, unique-traversal counts, feature/support
artifacts, metric definitions, and evaluator code.

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

Overlapping sequence windows are merged on `(date,order_id,traversal_id)` before
any metric. Copies must agree on support, validity, and truth. Empty overall,
low-support, unseen, or pace groups are `INSUFFICIENT_SUPPORT`, never zero or a
passing metric.

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
M5 starts from the selected M4 spatial/shared state only after a formal M4
adoption PASS; its temporal adapter remains freshly initialized.

M0 is built from a canonical Train-only matrix artifact recording feature
names/order, Train-median missing policy, source/matrix hashes, forbidden-input
audit, and exact target masks. Raw and [0,1] comparison-clipped predictions are
both preserved.

## Phase gates

Phase B0 begins with a Parquet-metadata-only schema audit, then fits protocol-
Train-only support and static artifacts before running one fixed Train bucket
and one fixed Validation bucket. These two preparation commands cannot read
Validation labels for fitting and do not train a model. Phase B1 is the frozen
transfer-tuning protocol and is now complete with a hash-bound tau freeze.
Its low-support and unseen evidence is mixed, so the frozen conclusion is Case
C: no convincing support-aware transfer evidence in B1. A later Phase C
development window requires a separate explicit authorization. Full rolling
remains allowed only after its product, leakage, scientific, and real-kernel
gates pass; completing B1 does not implicitly authorize Phase C or D.
