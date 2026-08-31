# Fleet spatial representativeness

The audit uses a deterministic approximately 1 km WGS84 grid at reference latitude 34.25 degrees and normalized occupied-cell shares. No FleetPy, Valhalla, MILP, or grid tuning is involved.

## Findings

Full Test31 demand and the 30,000-order Profile M replay are closely aligned:

| Window | TVD | Spearman | Top-10% hotspot Jaccard |
|---|---:|---:|---:|
| All day | 0.054 | 0.965 | 0.889 |
| Morning 07:00–08:59 | 0.077 | 0.915 | 0.600 |
| Evening 17:00–18:59 | 0.078 | 0.947 | 0.545 |

Selected replay fleet starts differ more from replay demand, as expected for session starts rather than contemporaneous demand. The all-day replay-to-fleet comparison has TVD 0.318 and Spearman 0.641; the evening comparison has TVD 0.338 and Spearman 0.547. All comparisons remain positively aligned, all-day coverage occupies the same 80 grid cells, and there is no obvious systematic displacement outside the demand footprint.

Decision: `KEEP_CURRENT_FLEET_RECONSTRUCTION = YES`.

The fleet is not forced to equal the demand distribution, and fleet sampling will not be reopened for this paper.
