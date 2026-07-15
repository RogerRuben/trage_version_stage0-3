# End-to-end canonical pipeline contract

## Purpose

The canonical pipeline is a frozen data lineage, not a directory convention:

`raw -> Stage 0 -> Stage 1 v2 -> Stage 2 dispatch -> Stage 3 calibrated vector -> Stage 4 Safe/O0 smoke`.

## Global invariants

1. Each stage consumes explicit manifest IDs and writes a new manifest.
2. Manifests record artifact version/status, producing commit, config hash, data
   dates, input artifacts, decision-time contract, and known limitations.
3. `canonical` inputs cannot depend on `exploratory` or `deprecated` artifacts.
4. Field roles are one of `pre_dispatch_feature`, `predicted_output`,
   `post_trip_realized_label`, or `evaluation_only`.
5. Test/future/post-trip information cannot flow into a dispatch feature.
6. A failed or missing audit prevents downstream canonical promotion.
7. No script selects an input because it is the newest or only matching file.

## Smoke split

- One training day, one validation day, one test day.
- 1,000–5,000 deterministic order keys per day upstream.
- Stage 4 uses a deterministic 500–1,000-order test subset.
- Sampling keys and seeds are stored before any label/model computation.

## Promotion gate

Formal full-day Stage 4 remains blocked until all are PASS:

- Stage 0 topology/measurement audit;
- Stage 1 label-v2 mathematical audit;
- Stage 2 dispatch-time availability/leakage audit;
- Stage 3 calibration, scale, and leakage audit;
- Stage 4 counterfactual input/state/economy audit;
- end-to-end lineage and reproducibility smoke audit.

## Reproducibility

`python run_pipeline.py --config config/pipeline_canonical.yaml --mode smoke`
is the only canonical smoke entrypoint. It writes a run registry entry and refuses
undeclared artifacts, dirty config drift, noncanonical upstream inputs, or a sample
outside the configured bounds.

