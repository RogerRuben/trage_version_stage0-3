# Appendix and supplement plan

## Appendix A. Capability profiles and route-suitability interface

- Formal definitions of the C, M, and A analytical profiles.
- Hard-state construction and evidence-completeness rules.
- Static, dynamic, and speed suitability components.
- Clarification that exposure is reference-envelope utilization rather than safety or failure probability.

## Appendix B. Empirical fleet reconstruction

- Construction of effective HV service sessions.
- Exact continuous-duration accounting and whole-session selection.
- Requested versus achieved `q_A`, AV counts, HV vehicle-hours, and session-selection error.
- Explicit exclusion of the 15-minute supply-profile equivalent from the penetration denominator.

## Appendix C. Sparse rolling-dispatch implementation

- Candidate-radius expansion from 2 to 8 km.
- Top-K=20 filtering and sparse feasible-arc representation.
- Pickup routing adapter and cache/fallback behavior.
- Sequential SciPy/HiGHS MILP implementation and computational diagnostics.

## Appendix D. Complete scenario results

- Full 27-scenario factorial service, wait, assignment, and queue outcomes.
- Descriptive finite differences and interaction contrasts.
- All-HV benchmark and all-AV C/M/A composition extremes.

## Appendix E. Operational-envelope policy details

- STRICT, REFERENCE, and UNCONSTRAINED parameter values.
- REFERENCE calibration provenance.
- Family-specific cumulative exposure and slack trajectories.
- Explanation of why positive slack does not imply an inactive discrete policy.

## Appendix F. Cost robustness

- Four within-`eta_c` comparisons of `epsilon_W=0` versus `0.05`.
- Normalized cost, service, P95 pickup, pickup-objective, and AV-share changes.
- No raw normalized-cost ranking across different `eta_c` values.

## Supplementary reproducibility material

- Frozen scenario registry and configuration identifiers.
- Claim-evidence map and literature-placeholder map.
- Detailed table/figure source mapping.
- Software and solver versions without presenting the study as a software report.
