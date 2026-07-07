# Stage1 prediction labels

Stage1 builds train-fitted, vehicle-neutral link labels for temporal prediction experiments:

- LCS: longitudinal control stress
- IIS: intersection interaction stress
- GNS: geometry/navigation stress
- RTS: reliability/tail-delay stress
- PMIS: POI-mediated interaction stress

The primary 7+1+1 experiment split is defined in [`split_config.json`](../split_config.json). Travel-time references and cohort histograms are fitted on train dates only, then applied to train, validation, and test dates.

Outputs are written under `stage1/output/prediction_split/`; generated labels and models are excluded from Git.

Coverage is recorded separately for every stress dimension, including the
cohort fallback level and sample size used for each row. Inferred route links do
not receive realized LCS/IIS/RTS/PMIS labels; GNS remains available as a static
geometry feature.
