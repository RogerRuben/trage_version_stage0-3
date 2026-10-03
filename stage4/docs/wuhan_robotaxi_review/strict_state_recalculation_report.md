# Strict pre-decision correction — 2026-09-26

## Material Passport and scope

Origin skill: academic-research-suite / experiment-agent. User authorized the
ordered sequence: corrected fixed-state analyses, complete native checkpoint
validation, then six finite-window dwell comparisons. This report covers the
first step (COMPLETE) and the second step's startup blocker. No manuscript,
frozen model, capability profile, or formal 41-scenario product was changed.
Execution base: ff6ec07. One process; no GPU or dense request-by-vehicle matrix.

The previous helper used assignment_time <= decision_time, so it removed
vehicles assigned at the current decision while keeping pre-decision waiting
orders. The corrected state uses assignment_time < decision_time. Completions
at the decision time remain available at their dropoff location. Historical
outputs remain untouched; corrected products are under strict_pre_epoch.

## Completed execution

- Prediction P/H/D0: 10 states, 40.24 s, zero routing failures.
- Neutral identity: 10/10 PASS; mechanism/K analysis: 74.59 s, zero routing failures.
- RT: 4 physical states × 4 release variants completed.
- Lead × patience: 48/48 cells; all 16 corrected 300-s reference cells reproduced;
  222.10 s, 87,519 routed arcs, zero routing failures.
- Whole sequential execution: 357.38 s; observed process peak 837.89 MiB
  (cumulative process high-water mark, not factorial-only incremental memory).
- Seven targeted tests passed. New checkpoint module compiles but is not yet
  execution-validated.

## Corrected findings

Ten-state AV graph maximum matching sums are now:

| Gate | Maximum matching sum |
|---|---:|
| Spatial | 718 |
| Passenger | 493 |
| Structural | 238 |
| Evidence | 124 |
| Shared Top-K | 124 |
| Routing returned | 124 |
| Patience / solver input | 14 |

The old final sum was 12. K=10/20/40/80 still gives identical final matching
cardinality within each of the ten sampled states. This is a sampled result,
not a global guarantee. Prediction remains DECISION-RELEVANT: P/H selections
differ in 10/10 states; mean selected AV-arc Jaccard is 0.025. A common outcome
evaluator was not run, so this does not establish decision superiority.

Four-state AV matching sums at 300 s are zero/Low/Base/High = 11/29/26/40
(old: 8/26/25/40). Corresponding aggregate patience retentions are
11/53=20.75%, 29/130=22.31%, 26/128=20.31%, 40/156=25.64%.

The 48-cell corrected aggregate results (state-summed, not daily capacity):

| Lead | Patience s | AV graph M | Mixed graph M |
|---|---:|---:|---:|
| Zero | 180 | 9 | 22 |
| Zero | 300 | 11 | 52 |
| Zero | 600 | 27 | 144 |
| Low | 180 | 13 | 34 |
| Low | 300 | 29 | 89 |
| Low | 600 | 103 | 251 |
| Base | 180 | 7 | 19 |
| Base | 300 | 26 | 68 |
| Base | 600 | 127 | 284 |
| High | 180 | 13 | 35 |
| High | 300 | 40 | 102 |
| High | 600 | 157 | 326 |

**The previous 24/24 cross-fleet ordering statement must be withdrawn.** It is
now 23/24: at 17:30, RT-High, patience=300 s, mixed graph M is 23 for q50 and
24 for q75. Thus cross-fleet ordering is not invariant even in these sampled
states. This strengthens the timing/patience qualification; it does not revise
the separately generated formal full-day results.

These are conditional fixed-state analyses, not alternative timing trajectories.
Canonical prior actions and vehicle locations are held fixed; earlier release
does not rebuild M3 features. They must not be described as causal full-day
counterfactuals or proof of early-release information availability. AV subgraph
capacity is not an additive decomposition of mixed-system service. Graph M is
before cumulative Gamma constraints and lexicographic selection. AV mirror
metrics retain fractional deadlines while mixed metrics use production integer
deadlines; the cells retain explicit discrepancy counts.

## Native checkpoint — NOT VERIFIED

`stage4.analysis.native_checkpoint_closure` reconstructs prehistory using logged
actions and native FleetPy motion (no prehistory routing/optimization). It is
designed to save route legs, onboard requests, waiting/expired demand, request
attempt metadata, and cumulative exposure, then check serialization roundtrip
and completion timestamps of all existing tasks. It uses the installed joblib
vendored cloudpickle; checkpoint pickle files are trusted-local artifacts only.

The first invocation exited before simulation because `load_fleetpy_bindings`
ran Git against a checkout owned by another account (dubious ownership). A
separate read-only check using an exact-path, command-local safe.directory
override confirms the pinned commit remains
0379f9725a147ff33c674de4884cdf89fd787fa9. No global Git safety setting was changed.

Per the experiment runner's no-automatic-retry rule, no retry was launched.
Retain `stage4/output/paper_enhancement/native_checkpoint_closure.log`.
An authorized retry should use a process-local exact-path safe.directory
override and `--attempt retry1`; do not overwrite the first attempt. The native
checkpoint is not yet produced or verified. Six dynamic conditions are NOT RUN.

## Reproduction and artifacts

First-step command (refuses overwriting completed outputs):

```powershell
& D:/anaconda/envs/stage0-valhalla/python.exe -u -m stage4.analysis.strict_state_recalculation
```

Small results are copied alongside this report under `strict_state_results/`.
Local ignored outputs retain registries, selected-pair diagnostics and runtime
logs. All legacy result directories are preserved. The next allowed work remains
checkpoint validation first, then the six zero-lead dynamic dwell conditions;
no additional model fitting, earlier-release inference, or full-day experiment
has been added.
