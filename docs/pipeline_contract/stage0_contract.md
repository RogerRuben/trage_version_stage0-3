# Stage 0 canonical contract

## Research task

Stage 0 converts raw order/GPS observations and a frozen road network into
auditable matched points, directed link traversals, turn movements, order OD,
and quality flags. It is measurement infrastructure, not a predictive model.

## Inputs

- Raw GPS points with order, driver, and observation timestamps.
- Raw order table with observed trip identifiers.
- Versioned directed road network retaining parallel edges and turn topology.
- A manifest that fixes dates, sampling keys, matcher configuration, and source hashes.

## Outputs

- Matched points and per-point confidence.
- Directed link traversals with entry/exit times and allocated distance.
- Valid `from_link -> node -> to_link` movements.
- Order OD with explicit source and coordinate validity.
- Order/link/movement quality flags and a Stage 0 output manifest.
- An order-level route quality class: `core`, `extended`, or `rejected`, with
  gap counts, graph-bridge evidence, confidence, fallback share, and reason.

## Allowed information

Only observations available in the raw trajectory/order records and the frozen
road network. Whole-trip observations may be used for retrospective measurement,
but this fact must be declared and must not be confused with dispatch-time input.

## Forbidden information

- Stage 1 labels or downstream model predictions used to repair matching.
- Silent acceptance of directionally impossible transitions.
- Collapsing parallel road edges without an explicit topology rule.
- Promoting low-confidence trajectories to high-quality truth.
- Choosing outputs by scanning for the newest parquet file.

## Acceptance rules

- Directed-edge, parallel-edge, link-continuity, OD-source, time-allocation, and
  distance-allocation audits are computed from outputs and pass.
- Low-quality and unresolved orders remain explicit.
- `core` contains only fully directed-continuous routes satisfying the frozen
  confidence/fallback contract. `extended` may contain only bounded gaps that
  are demonstrably bridgeable in the frozen directed graph; it is robustness-only.
- `rejected` orders cannot enter formal Stage 1-3 fitting or primary evaluation.
- A manually reviewed truth sample is versioned and reported.
- Every output is reachable from one input manifest and one config hash.

## Time convention

Observation timestamps describe when GPS evidence was produced. Traversal entry
and exit times must conserve the original adjacent-point interval. Stage 0 is a
retrospective measurement product and is not itself a dispatch-time feature.

## Missing and fallback rules

Unmatched points, unresolved topology, absent OD, and low-quality intervals remain
missing or explicitly flagged. A topological inferred path is a fallback with its
own quality value; it is never relabeled as an observed traversal.

## Version and downstream consumers

Contract version: `stage0_canonical_contract_v2`. Consumers are Stage 1 label v2,
Stage 2 planned-route construction, and lineage audits. No later stage may mutate
the Stage 0 artifact.

The paper-pipeline quality extension is `stage0_route_quality_v3`. Promotion of
that extension is blocked until the full-data Core/Extended coverage and manual
truth audit pass; the existing engineering smoke remains a separate artifact.
