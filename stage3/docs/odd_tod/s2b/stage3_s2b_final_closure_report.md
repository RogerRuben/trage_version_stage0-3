# Stage 3 S2B Final Closure

Status: `STAGE3_S2B_INTERSECTION_COMPLEX_FROZEN`. Selected buffer radius: **10m**; candidate-overlap condition: center distance <= 20m.

Human adjudication: `{"10_CORRECT": 7, "5_CORRECT": 0, "BOTH_ACCEPTABLE": 61, "NEITHER": 2, "UNCERTAIN": 0}`. This targeted review supports 10m over 5m and is not a population accuracy estimate.

Final rows: complexes `43,685`, movements `419,907`, membership `59,052`, edge-complex boundary index `289,295`, route-movement lookup `419,907`. Movement, membership, and lookup tables are byte-identical promotions of frozen r10 sources. The complex table retains every frozen r10 source column/value and adds only `buffer_radius_m = tolerance_m = 10` as the contract-required semantic alias.

Historical reverse overlays remain `HISTORICAL_DIRECTION_OVERLAY + AV_ROUTABILITY_VIOLATION`; they are neither missing nor inserted into AV topology. Movement legality and control-state claim boundaries remain conservative.

`S3_AUTHORIZED = NO`; `NEXT_PHASE_AUTHORIZED = NO`.
