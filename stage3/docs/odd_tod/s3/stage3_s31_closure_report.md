# Stage 3 S3.1 Scientific Closure

Status: `STAGE3_S31_CLOSURE_COMPLETE`. Reviewed base: `309da4e5164eb99314c34b15ae2652f587a29f0b`.

## Static D correction

`D_c` now equals the number of unique `valhalla_road_class` values on unique `INCOMING` and `OUTGOING` boundary edges of the frozen 10m complex. `INTERNAL` edges are excluded. No clustering, membership, movement, A/M/L definition, speed rule, or dynamic rule changed.

- Train-exposed complexes: 2,425
- Old D caps C/M/A: `1.0` / `1.0` / `2.0`
- New D caps C/M/A: `2.0` / `3.0` / `3.0`
- A/M/L unchanged: `True`

## Frozen dynamic invariance

Dynamic caps and all dynamic products were not recomputed. Before/after hashes are identical: `{"cdf": "a09f284959008890e8a31bab0839bb142f1f66d43bc7012efc3dfeeeec3adc26", "train_descriptors": "4edb5522a224d0cdc4efda1c2e53ba736ff45b6db9c0e2fe67a257a510071b27", "validation_descriptors": "96eda589ad3c055c8b283327cd3830fd0d777f1ef1b202fb87ac6fe8be5c0d05"}`.

## Train M3 cache provenance

- Dates bound: 16
- Prediction rows: 11,432,534
- Model/checkpoint: M3 / `965fc491cd77256f7889961d89932ec6be709bab04adcca358ac1b49f47c2cde`
- All cache hashes, row counts, schemas, and day manifests verified: `true`
- Realized target columns present: `false`
- Inference rerun required: `false`

`pi_k` defines marginal capability caps, not joint route acceptance rates.

Test31 was not read. `S4_AUTHORIZED = NO`; `NEXT_PHASE_AUTHORIZED = NO`.
