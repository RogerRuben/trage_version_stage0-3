# Simulator v3 ODD profile freeze

## Main profiles

Simulator v3 uses `conservative_av`, `moderate_av`, and `mature_av` as an
explicit low/mid/high scenario grid.  Their probability ceilings are fixed
independently of the 2016-10-23 outcomes:

| Profile | Core tail-probability ceiling | Core uncertainty ceiling | Role |
| --- | ---: | ---: | --- |
| conservative_av | 0.50 | 0.30 | main low-capability scenario |
| moderate_av | 0.70 | 0.40 | main mid-capability scenario |
| mature_av | 0.90 | 0.60 | main high-capability scenario |

These are scenario priors, not empirical ADS capability estimates.  The
threshold source is recorded as
`explicit_exogenous_probability_ceiling_grid_v1`; the test day is not used to
set any main-profile threshold.

The former full-day-calibrated moderate profile is retained only as
`full_day_calibrated_sensitivity_av`.  Its metadata explicitly records
`test_day_used_for_thresholds=true` and the simulator refuses to load it unless
the sensitivity-only command-line override is supplied.

## Full-day evaluation (not calibration)

Applying the frozen profiles to 112,165 condition-known orders gives the
following service-feasible shares:

- conservative: 0.1480%
- moderate: 54.4831%
- mature: 100.0000%
- deprecated full-day-calibrated sensitivity: 72.1303%

These shares evaluate the frozen scenarios; they are not inputs to threshold
selection.  The machine-readable source is
`stage4/docs/results/vehicle_capability_profile_feasibility.csv`.

## Score-scale warning

The fold-3 15,000-order export and the full-day inference have materially
different score distributions.  For the 15,000 overlapping orders, mean LCS
tail probability changes from approximately 0.098 to 0.696.  PMIS changes from
approximately 0.101 to 0.410 and RTS from approximately 0.108 to 0.470.
Consequently, test-day rank remapping and test-day threshold calibration are
prohibited.  This drift remains a model-interface limitation to report in the
final analysis; it is not hidden by the capability profile.

## Conditional IIS

IIS is applied only when it is both available and applicable.  IIS unavailable
or not applicable does not close an AV edge.  Unknown Core condition remains
AV-ineligible, while HV edges continue through all normal operational checks.
