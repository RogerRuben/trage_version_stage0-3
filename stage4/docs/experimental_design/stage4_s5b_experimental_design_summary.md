# Stage4 S5B Experimental Design Freeze

Recommendation: `GO_STAGE4_FINAL_EXPERIMENTS`

## Canonical scientific base

- Commit: `18d03a9df87eb518689f77346cb080ab84f2f402`.
- FleetPy: `0379f9725a147ff33c674de4884cdf89fd787fa9`.
- `H_base_exact = 12279.336389` vehicle-hours.
- `q_A = 24 N_AV / H_base_exact`; the 15-minute supply-profile equivalent is not a penetration denominator.

## Main structural experiment

- `q_A in {0.25, 0.50, 0.75}`.
- Capability profile `k in {C, M, A}`.
- Acceptance probability `p_A in {0.40, 0.70, 1.00}`.
- 27 structural scenarios, plus one all-HV and three all-AV benchmarks.

## Acceptance semantics

- Common-random-number seed: `20260827`.
- `p_A` is a probabilistic scenario parameter.
- `a_o^A = 1(u_o <= p_A)` is the realized binary acceptance indicator.
- The same SHA-256 order draw is reused across scenarios, so acceptance sets are nested.

## ODD exposure policies

- STRICT: `(static, dynamic, speed) = (0, 0, 0)`.
- REFERENCE: `(2.145068, 0.149343, 0)`.
- UNCONSTRAINED: `(null, null, null)`.
- PATH `(2.200874, 0.401175, 0)` remains diagnostic only because dynamic PATH is fixed by assignment rank 1 at 08:01.

## Three-family model status

Static, dynamic, and speed are all retained. Family activity is data-, profile-, and operating-condition-dependent; speed is not deleted or independently swept under Profile M.

## Cost robustness

- `eta_cost_av_to_hv in {0.50, 0.75, 1.00, 1.25}`.
- `epsilon_W in {0, 0.05}`; 5% is a platform-policy sensitivity, not passenger behavior.
- Normalized operating time only; no monetary calibration is claimed.

## Scenario count and compute budget

- Registry rows: 42; reused configurations: 1.
- Unique dispatch scenarios: 41 (hard cap 45).
- One-hour sequential estimate: 1.25-1.42 hours.
- Full-day linear sequential estimate: 30.07-34.17 hours.
- Four-slot ideal full-day wall-clock estimate: 7.52-8.54 hours, subject to Valhalla contention.
- S5B launched no FleetPy run and no scenario matrix.
