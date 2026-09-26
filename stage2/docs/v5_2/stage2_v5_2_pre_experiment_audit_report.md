# Stage 2 v5.2 Phase A.2 static-review repair report

Status: `NOT_READY_IMPLEMENTATION_ONLY`.

This change implements the Phase A.2 static-review taskbook. No test,
`compileall`, training, inference, development, rolling, legacy, benchmark, or
data-production workload was run. Phase B remains unauthorized until the user
explicitly authorizes it.

## P0 code review

1. PASS (code): categorical order comes from frozen v5 `CATEGORY_NAMES`; JSON
   object order is ignored and all checkpoint embedding sizes are checked.
2. PASS (code): `V51SourceModelBinding` binds each protocol to exact Train and
   validation dates, resolved source config, feature vocabulary/order, model
   manifest, checkpoint, model parameters, distribution, and history mode.
3. PASS (code): `build-transfer-shards` writes the canonical
   `protocol/split/date` layout with atomic NPZ/JSON files and content hashes.
4. PASS (code): tensor alignment uses `(split,date,order_id,traversal_id)` and
   every builder call accepts exactly one split/date partition.
5. PASS (code): evaluator merges overlap copies by unique physical traversal,
   validates target/support consistency, records raw/unique/duplicate counts,
   and only then computes metrics.
6. PASS (code): empty overall, low-support, unseen, or pace groups produce
   `INSUFFICIENT_SUPPORT`; selection and adoption fail closed.
7. PASS (code): temporal leakage is calculated from hash-bound transfer shards;
   no CLI leakage-count override remains.
8. PASS (code): tau selection accepts only formal evaluator manifests bound to
   protocol dates, M1/M4 checkpoint/evaluation hashes, artifacts, counts,
   definitions, and evaluation code.
9. PASS (code): non-M1 models require a formal same-protocol/source M1 metric
   manifest.
10. PASS (code): M5 requires a passing formal M4 adoption manifest and loads
    selected M4 shared/spatial weights while leaving the temporal adapter fresh.
11. PASS (code): `evaluate-model` performs checkpoint inference, unique-
    traversal evaluation, prediction Parquet output, and provenance manifest
    output.
12. PASS (code): `verify-final` accepts exactly the hard-coded final gate set;
    missing and extra gates fail.
13. PASS (code): release generation reads the real Stage 1 `release_tag` schema
    and binds source config/manifest, tensors, training, checkpoint, evaluation,
    artifacts, and products by SHA-256.

## P1 code review

- PASS (code): support, static, CDF, and M0 artifacts have type-specific
  Train-only/evaluation-row rules.
- PASS (code): static upstream/downstream neighbors use the full
  `(split,date,order_id)` route key and the fitted artifact validates its own
  feature schema.
- PASS (code): the canonical M0 builder freezes feature names/order, Train-only
  median fill, source hashes, forbidden-input audit, targets/masks, and matrix
  hash; M0 preserves raw and clipped predictions.
- PASS (code): thresholds, seeds, loss weights, shard dimensions, benchmark
  sizes, warmups, repetitions, and coverage rules come from frozen config rather
  than CLI overrides.
- PASS (code): benchmark code uses inference mode, two warmups, three repeats,
  median timing, CPU and available GPU, full transfer forward, and baseline /
  peak / delta RSS.
- PASS (code): Phase B0 has a metadata-only upstream schema audit command; it was
  implemented but not run.
- PASS (code): Stage 3 allowed and evaluation-only field masks are separate and
  documented.
- PASS (code): complete service time requires frozen `0.999` route coverage;
  partial estimates remain explicitly marked.
- PASS (code): NumPy Generator, Torch, and CUDA seeds plus policies are recorded.
- PASS (code): raw RTS/LCS masks are independent of tail supervision, while
  early-phase RTS and tail loss weights are frozen to zero.
- PASS (code): config, research contract, token, route, static, training,
  evaluation, release, and transfer products use distinct schema namespaces.

## Execution decision

Phase B allowed now: **NO**.

The next action, only after explicit authorization, is static/unit verification
and the Phase B0 metadata/schema audit. No 20161028-30 production is required.
