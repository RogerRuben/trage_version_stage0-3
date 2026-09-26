# Stage4 routing determinism closure v2

## Frozen decision

- Arc sample: 60 unique Q50 pickup arcs.
- Sample SHA256: `3bfb4d3781e45efb2a09810c599d1b381566c7af5b01f0c09aa038bb86af1e2c`.
- Composition: 1 known divergent, 20 ordinary successful, 20 nearest the 300 s patience boundary, and 19 fixed morning/evening peak arcs.
- Scalar repeats per arc: 5.
- One-source matrix repeats per arc: 5.
- Multi-source contexts: batch size 2, 5, 10, and 20; ten preselected arcs also use first/middle/last focal positions.
- Simulation cache: disabled for the micro-audit and unique-arc performance spike.

All scalar, one-source matrix, and multi-source calls succeeded. The maximum within-arc repeat range was 0 s for scalar and M1. The maximum M1-to-MB difference and maximum source-position spread were both 0 s. The previously divergent 522 m Q50 arc returned 60.0 s on every M1 repeat.

The fixed 5,000-arc spike produced:

| Mode | Wall time (s) | Arc/s | Peak RSS (MB) | Failures |
|---|---:|---:|---:|---:|
| SINGLE_SOURCE_MATRIX | 6.096 | 820.249 | 225.469 | 0 |
| SCALAR_ROUTE | 5.407 | 924.798 | 222.570 | 0 |

The frozen robustness routing mode is `SINGLE_SOURCE_MATRIX`. Correctness has priority over the small observed scalar speed advantage, and the taskbook identifies M1 as the preferred mode when stable. Every cache-missing pickup arc is routed as an independent 1x1 matrix request. Unrelated candidate origins cannot affect its batch or source position. No ETA rounding, patience change, canonical epsilon matching, or arc-specific patch is used.

Canonical 41-scenario outputs remain untouched. Deterministic controls and treatments are written only under `stage4/output/paper_enhancement/repositioning_robustness_det/`.

Q50 clean-process A/B repeatability is the next execution gate. Repositioning remains unauthorized until that gate and the spatial/idle audits pass.
