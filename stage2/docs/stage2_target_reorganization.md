# Stage2 target reorganization

Stage2 separates stable expected exposure from conditional anomalies instead of
treating one cohort percentile as the only prediction target.

For target `k` and observation `(o,l,t)`:

```text
Y_raw(k,o,l,t) = baseline_stress(k,l,t) + abnormal_residual(k,o,l,t)
```

The prediction products are:

1. `expected_raw_stress`: conditional expectation under dispatch-time context;
2. `conditional_percentile_stress`: relative position in the train-fitted cohort;
3. `tail_exceedance_probability`: calibrated probability above a train-defined threshold;
4. `stress_uncertainty`: predictive dispersion or conformal interval width;
5. `target_applicability`: especially for IIS movement rows.

Regression percentile scores are not event probabilities. Brier score and ECE
are reported only for a separately calibrated tail probability.

## Modeling units

- LCS/RTS/PMIS: traversal-level models and order aggregation;
- IIS: movement applicability plus conditional movement severity;
- GNS: static context feature and optional route exposure summary;
- order: auxiliary consistency head and decision-facing aggregation.

Stage3 remains blocked until rolling/OOF predictions exist and at least one
target demonstrates stable out-of-day ranking/tail value with cluster-bootstrap
uncertainty. In-sample train predictions and oracle-route results are not valid
calibration inputs.

