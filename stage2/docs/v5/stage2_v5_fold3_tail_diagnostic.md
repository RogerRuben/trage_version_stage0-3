# Stage 2 v5 Fold 3 frozen-tail diagnostic

This is a read-only post-evaluation diagnostic. It does not alter predictions, metrics, or admission.

| Date | Valid rows | Frozen mean MAE | Diagnostic MAE with prediction <= 1 s/m | >1 s/m | >10 s/m | Max prediction | Largest-row share of MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20161026 | 238711 | 0.060543924 | 0.028702238 | 4 | 1 | 7584.049805 | 52.47% |
| 20161027 | 237196 | 0.028318721 | 0.028302850 | 3 | 0 | 2.737702 | 0.04% |

The reported Fold 3 metrics remain the frozen values. The diagnostic shows whether a log-normal mean tail, rather than broad daily degradation, explains the instability.
