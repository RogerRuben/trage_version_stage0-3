# Prospective AV opportunity logging protocol

## Purpose and unit

The authorized rerun adds shadow diagnostics only. Every counter uses one common unit:

\[
(o,v,t)=\text{waiting order }o\text{ paired with available AV }v\text{ at decision epoch }t.
\]

The same order–vehicle pair at a later rolling epoch is a new opportunity. Counters are accumulated without storing pair-level rows, so memory use remains proportional to epoch count rather than candidate count.

## Frozen dispatch boundary

The diagnostic call observes the existing spatial index. It does not alter:

- fleet construction or active vehicle-hours;
- vehicle availability or position;
- acceptance realization or seed;
- search-radius expansion;
- shared Top-K candidate selection;
- Valhalla routing/cache behavior;
- patience rules;
- solver variables, constraints, or objective hierarchy;
- assignment or vehicle evolution.

The switch is opt-in (`prospective_gate_logging=true`) and is false on every canonical execution path. Reruns write only below `stage4/output/paper_enhancement/gate_decomposition/reruns/`.

## Gates

| Counter | Definition |
|---|---|
| N0 spatial | Available AV lies within the order's current expanding diagnostic radius, before passenger or route gates |
| N1 passenger-compatible | N0 and the frozen order-level acceptance realization permits AV service |
| N2 structurally ready | N1, a selected route exists, and hard state is `FEASIBLE` |
| N3 evidence-complete | N2 and the frozen evidence flag, all three exposure families, and positive finite AV service-time prediction are present |
| N3a shared Top-K | N3 and the AV survives the actual shared HV/AV sparse Top-K construction |
| N3b route returned | N3a and the unchanged Valhalla adapter returns a pickup estimate |
| N4 pickup within patience | N3b and remaining patience covers corrected pickup ETA |
| N5 solver-eligible | N4 and every remaining unchanged arc-construction condition passes |
| N6 selected | N5 and the unchanged lexicographic solver selects the AV arc |

N3a and N3b are explicit because the production solver does not route every spatial opportunity. Omitting them would incorrectly label shared Top-K truncation or routing failure as patience loss.

## Mutually exclusive structural attribution

The N1→N2 loss is partitioned in this order:

1. `NO_SELECTED_ROUTE`;
2. `HARD_INFEASIBLE`;
3. `HARD_UNKNOWN`.

N2→N3 is `EVIDENCE_INCOMPLETE`. Existing evidence suggests it may be substantial for fallback routes even though the older `missing_exposure` counter was zero; the older counter was conditioned on a different dispatch state and must not be treated as this gate's answer.

## Interpretation

- Eligibility conversion: \(N_5/N_0\).
- Assignment conversion: \(N_6/N_0\).
- N5→N6 is dispatch competition/selection, not capacity attrition.
- Gamma is excluded because all four authorized anchors are `UNCONSTRAINED`; it remains a policy-level frontier experiment.

## Aggregation and invariants

Each run writes epoch counters, scenario totals, and 15-minute bins. Every epoch and aggregate must satisfy:

\[
N_0\ge N_1\ge N_2\ge N_3\ge N_{3a}\ge N_{3b}\ge N_4\ge N_5\ge N_6.
\]

Each adjacent difference must equal its recorded loss attribution. Scientific interpretation is blocked unless request outcomes, assignment fingerprints, and the frozen summary metrics reproduce their canonical anchor exactly.
