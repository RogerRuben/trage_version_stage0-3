# Stage4 Capability Scenario Report

Vehicle capability profiles are dimension-specific scenario priors, not empirical AV capability estimates.

## Profiles

The current profile set is:

- `conservative_av`
- `moderate_av`
- `mature_av`
- `intersection_sensitive_av`
- `uncertainty_sensitive_av`
- `reference_hv`

Each profile specifies:

- dimension sensitivity;
- dimension soft thresholds;
- dimension hard thresholds;
- uncertainty tolerance;
- missing-modality penalty;
- placeholder remote-assistance/fallback costs.

IIS missingness is treated as unknown intersection information, not zero intersection stress.

## Scenario sensitivity summary

Three-fold means:

| profile | AV feasible share | AV mean stress | HV residual stress | ODD margin mean | ODD margin P10 |
|---|---:|---:|---:|---:|---:|
| conservative_av | 0.9734 | 0.2299 | 0.3078 | 0.8373 | 0.6945 |
| intersection_sensitive_av | 0.9768 | 0.2314 | 0.2678 | 0.8291 | 0.6940 |
| mature_av | 0.9992 | 0.2322 | 0.2810 | 0.8403 | 0.6963 |
| moderate_av | 0.9877 | 0.2319 | 0.2611 | 0.8395 | 0.6961 |
| uncertainty_sensitive_av | 0.9851 | 0.2316 | 0.2720 | 0.8381 | 0.6934 |

The profiles produce different feasible sets and residual HV burden. The current thresholds remain scenario priors and should be calibrated when real AV operational data become available.

## Output files

- `stage4/output/capability_mapping_v2/`
- `stage4/output/capability_sensitivity/capability_sensitivity_summary.csv`
