# D13 closure protocol

Authorized 2026-09-21 after review of c04e623. Freeze before execution.

Only ten registered states, one new condition. D13 uses the frozen D1 route selection and remaining hard/unknown reasons. It bypasses evidence_complete and finite-rho only when that selected route is established and FEASIBLE. It is not D1-eligible OR D3-eligible. UNKNOWN/NONE remain excluded. Gamma/cost OFF. All passenger, spatial, shared Top-K20, pickup/patience and availability rules unchanged. No routing or inference, no new fallback, no profile changes.

Primary A1 and secondary A2 use condition-local count/pickup optima, existing 0/0.5/1/2/5% budgets, exact minimum absolute/relative gaps and 1e-7 numerical tolerance. No new outcome-dependent threshold or pairwise experiment.

Check every required pickup key before any optimization. Missing/unresolved keys STOP; do not fill them. Existing cache preflight found 12,352 unique keys and zero missing. All old products are read-only.

Compare D13 with D0/D1/D3/D4 per state, including graph metrics, reference-order differences, absolute gaps and relative-budget denominator effects. For D4-new-A1 states 0730/0830/1300/1800/1830, inspect the existing D4 baseline and A1 optimal witness against D13. Attribute absent witness arcs to Stage3 hard/unknown/no-route versus shared Top-K versus downstream constraints. Explain that a witness diagnosis is not a unique or exhaustive causal proof. Report remaining D1 hard/unknown reason codes separately, including their co-occurrence.

CPU single process, sparse MILP, each solve <=10 seconds, execution <=600 seconds, RSS guard 2 GiB. Stop after report and aggregate outputs, commit/push. No reverse full-population audit or Stage5 in this execution.
