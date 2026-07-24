# Canonical smoke reproduction commands

The authoritative configuration is `config/pipeline_canonical.yaml`. All stage
builders require explicit manifests; no canonical command discovers inputs by
searching a legacy output directory.

The executed chain is:

1. Stream a deterministic complete-order sample from each declared raw archive
   with `scripts/extract_canonical_smoke_raw.py` (100,000-row maximum buffer).
2. Run Stage0 geometric processing with eight buckets, then two-worker HMM
   matching, package the exact four dates, and run the direction/conservation
   audit.
3. Run `stage1/scripts/build_stage1_labels_v2.py` with fit date 20161019 and
   targets 20161019/20/22/23.
4. Run `stage2/scripts/build_stage2_dispatch_smoke.py` with fit date 20161019
   and held-out downstream dates 20161020/22/23.
5. Run `stage3/scripts/build_stage3_canonical_smoke.py` with train/validation/
   test 20161020/22/23.
6. Build the training-day-derived smoke environment, then run only
   `stage4/scripts/run_canonical_safe_o0_smoke.py`.
7. Publish every artifact with `scripts/write_artifact_manifest.py` and finish
   with the end-to-end audit.

After the materialized stage artifacts have been produced, the single governed
entrypoint revalidates every manifest hash, reruns the end-to-end audit, and
registers the successful canonical smoke:

```powershell
python run_pipeline.py --config config/pipeline_canonical.yaml --mode smoke --resume-from stage0
```

The entrypoint is intentionally idempotent for the same config hash. It does not
silently retrain or replace a successful canonical run. Rebuilding an individual
stage requires its explicit builder command and a new manifest/config version.
