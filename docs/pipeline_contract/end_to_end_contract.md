# End-to-end canonical pipeline contract

## Purpose

The canonical pipeline is a frozen data lineage, not a directory convention.

Engineering smoke v2 remains:

`raw -> Stage 0 -> Stage 1 v2 -> Stage 2 dispatch -> Stage 3 calibrated vector -> Stage 4 Safe/O0 smoke`.

The paper pipeline v3 is:

`raw -> Stage 0 quality sets -> Stage 1 v2 -> Stage 2 dispatch -> Stage 3 route risk/ODD -> Stage 3.5 offline HV/AV routes -> Stage 4 FleetPy dispatch`.

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
8. A Stage 4 algorithm change cannot mutate frozen Stage 0-3.5 artifacts.

## Engineering smoke split

- One upstream fit day, one Stage 3 training day, one validation day, and one test day.
- 1,000-5,000 deterministic order keys per day upstream.
- Stage 4 uses a deterministic 500-1,000-order test subset.
- Sampling keys and seeds are stored before any label/model computation.

## Paper-pipeline split

The proposed frozen chain is 20161019-21 for fit/training support, 20161022 for
validation/calibration, and 20161023 for final testing. Exact model-specific fit
roles are declared in `config/pipeline_research_v3.yaml`. Test data cannot fit a
reference distribution, scaler, calibrator, ODD profile, route weight, or detour cap.

## Promotion gate

Formal full-day Stage 4 remains blocked until all are PASS:

- Stage 0 topology, measurement, Core/Extended quality, and manual-truth audit;
- Stage 1 label-v2 mathematical and train-only-reference audit;
- Stage 2 dispatch-time availability, held-out, and planned/revealed isolation audit;
- Stage 3 calibration, route aggregation, scale, ODD, and leakage audit;
- Stage 3.5 directed-route, detour, ODD, and manifest audit;
- FleetPy adapter and zone-sparse matching validation;
- Stage 4 counterfactual input/state/economy audit;
- end-to-end lineage and reproducibility audit.

## Reproducibility

The v2 engineering smoke entrypoint remains:

`python run_pipeline.py --config config/pipeline_canonical.yaml --mode smoke`

The v3 paper pipeline is intentionally stage-addressable rather than forced into
one long command. Every stage must provide an explicit reproduction command and
manifest. Downstream reruns reuse frozen upstream manifests.

