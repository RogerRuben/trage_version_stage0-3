# Traffic mechanism v1: preregistered bounded extension

## Material Passport

academic-research-suite / experiment-agent; 2026-09-28; planned execution, UNVERIFIED.
User authorized offline diagnosis -> fixed-state F/V/VT -> small dynamic windows.

## Fixed choices before solving

Configuration: `stage4/config/traffic_mechanism_v1.json`.
Test31 10:30 and 17:30, 15-minute release cohorts; clock-selected, not selected by service differences.
Do not label either window empirically congested before measuring its descriptors.
q50 baseline-normalized AV active-hour level; P70; existing seeds, TopK20, 30s ticks,
300s arrival patience; no dwell/lead/repositioning additions; known outside budget 5%.
F=frozen, V=only CV/acceleration replacement, VT=V plus known-outside traffic gate.
Unknown stays diagnostic; no new rejection, penalty, or denominator change.

## Phase 1: offline independent action

Read existing policy rows 25/26/27/31 sequentially. Report date/profile distributions,
outside component shares and unknown upper bound, never feed the bound into control.
For Test31 join ORIGINAL identity to replay foundation, report baseline structural/evidence
eligibility vs new traffic rejection, hourly and fixed 0.02-degree pickup-grid summaries.
Validation has traffic descriptors but no equivalent Stage4 replay eligibility product:
do not fabricate Validation hard-state/dispatch qualification. Spatial/hourly joint results are Test31 only.
Gamma is cumulative, not an order-level pass/fail classifier. No realized outcome columns read.

## Phase 2: six fixed states, three policies

Use each profile's own frozen scenario history, not an M state relabeled C.
C/A sources have Gamma disabled; retain that. M source is ODD_Q50_M_P70_REFERENCE with
frozen reference Gamma. Cross-profile contrasts are not controlled profile-only effects.
Forced recorded prehistory uses native FleetPy motion; no prehistory route solving/training.
Validate strict pre-decision available/busy states, exposure ledger, serialization and source hashes.
Keep historical tasks fixed; V/VT re-express past dynamic exposure using the same new definition.
Traffic budget applies only to new decisions, not retrospectively to past assignments.

At each checkpoint intercept production solver to capture sparse arcs without committing an action.
Report AV/mixed E,U,maximum matching; Gamma-constrained maximum-cardinality count separately
from actual critical-first lexicographic selections. Use the unchanged production solver.
F/V candidate identities must match; V/VT may differ through traffic eligibility BEFORE shared TopK.
Do not assert mixed candidate-set nesting: removing AV choices can admit replacement HV TopK arcs.
Additionally evaluate a fixed-V-arc deletion diagnostic to isolate pure gate removal from TopK refill.
This diagnostic is not a replacement dispatch policy.

## Phase 3: twelve bounded continuations

C/M x two clocks x F/V/VT. Identical per-profile per-clock physical checkpoints.
New requests stop at cohort end; dispatch until cohort end+300s; drain existing/accepted tasks.
Compare paired order identities, AV/HV transfers, waits, service completion, exposure and gate losses.
Enable existing shadow gate aggregates and check conservation. Do not store dense graphs.
Frozen M10:30 assignments must equal previous retry1 F output, despite extra logging.
Do not call any continuation canonical full-day reproduction; deterministic routing is retained.

## Execution, QA and stopping

Sequential CPU, 2GiB advisory, 1800s per bounded unit, no GPU. Cache cleared per epoch.
Keep runtime/status and failures; no automatic experiment retry after a crash.
Check V/VT shared exposure definition, disabled default, source/config/profile/M3 immutability,
native reconciliation, no overlapping tasks, pickup/service conservation, and cohort completeness.
Do not retune thresholds, Gamma, selection, timing or windows based on results.
Stop after these products, report null effects too, commit/push. No full-day/41-scenario rerun.
