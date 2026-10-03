# Native checkpoint validation

## Material Passport

- Date: 2026-09-26; academic-research-suite / experiment-agent.
- User authorized retry after Git ownership startup failure.
- Scope: exact canonical prehistory reconstruction and native-state validation;
  no model training, full-day optimization or manuscript modification.
- Construction/roundtrip status: VERIFIED. Dynamic continuation is a separate gate.

## Completed reconstruction

The exact-path `safe.directory` exception was supplied only through the launch
process environment. No global Git configuration was changed. Pinned FleetPy
commit: 0379f9725a147ff33c674de4884cdf89fd787fa9.

Logged assignment actions before 10:30 were replayed through native FleetPy
motion. Routing and optimization were not rerun for prehistory. The checkpoint
is after movement and demand activation at 10:30, but before that epoch's
assignment. Consequently resume calls `control.time_trigger(37800)` once,
then `sim.step(37830)` onwards; repeating `sim.step(37800)` would be incorrect.

| Condition | Prior assignments | Available vehicles | In-progress tasks | Checkpoint bytes |
|---|---:|---:|---:|---:|
| q50 | 3,856 | 367 | 247 | 32,682,211 |
| q75 | 2,663 | 396 | 157 | 29,419,768 |

Both checkpoints passed state-signature roundtrip equality. Both original and
deserialized objects were advanced without new dispatch until all existing
tasks finished. Native-state signatures remained equal; pickup and completion
timestamps for every pre-checkpoint assignment matched canonical logs with
maximum absolute error 0 s. Available vehicle identities/locations and
cumulative exposure also matched the strict pre-decision reference.

Runtime: 92.30 s. Peak process memory: 602.99 MiB. CPU only.
Machine-readable result: `strict_state_results/native_checkpoint_summary.json`.
Full serialized objects remain local in
`stage4/output/paper_enhancement/native_checkpoint_closure/retry1/`.
These are trusted-local Python pickle artifacts, not portable/untrusted inputs.

## Zero-continuation gate: baseline divergence

`stage4.analysis.native_zero_continuation` compares every epoch's selected
order–vehicle pairs from 10:30 through 11:59:30 against the canonical source,
using the restored native lifecycle and single-source deterministic matrix
routing. A first divergence stops the gate and is recorded rather than hidden
by matching only assignment counts. The 60/120-s treatments must not start
unless this gate passes or an explicit revised baseline is authorized.

Executed q50: 10:30 through 11:30:30 matched exactly, including all 1,082
selected pairs and their pickup ETAs. At 11:31 (simulation second 41,460), both
runs selected six requests, but one vehicle choice differed (pair symmetric
difference 2). Waiting=75, available=327, Top-K pairs=1,362 and valid arcs=12
were identical. The gate stopped immediately; q75 continuation and all dwell
treatments were not run. Runtime 256.38 s, peak 426.70 MiB, routing failures 0.

A bounded diagnostic on the same canonical 11:31 physical state, with cold
adapters and the production candidate constructor, identified a routing API
difference for native request 7975:

| Vehicle | Single-source matrix corrected ETA s | Batch matrix corrected ETA s |
|---|---:|---:|
| 1055 | 273.3415958451 | 9.2658168083 |
| 1056 | 171.4176109537 | 171.4176109537 |
| 1357 | 240.9112370161 | 240.9112370161 |

The continuation selected vehicle 1056; canonical selected 1055. This is not a
solver tie: the same nominal arc has materially different routing cost under
the two API modes. The cold batch reproduces the recorded canonical ETA, so
prior cache history is not required to reproduce this specific discrepancy.
The underlying Valhalla algorithm cause was not diagnosed here.

This does not invalidate the native checkpoint. It blocks a claim that
single-source continuation exactly reproduces the original batch-routing
benchmark. Before treatments, choose either (a) retain original batch routing
and verify its zero control, or (b) preregister a new deterministic-routing zero
control shared by all dwell conditions and report it separately from the
canonical benchmark. No choice has been silently applied.

The experiment skill's ordered validation discipline kept downstream treatments
off after the failed equivalence gate; no result-dependent tuning was performed.
Machine-readable result: `strict_state_results/native_zero_continuation_summary.json`.
Local diagnostic log: `stage4/output/paper_enhancement/native_zero_routing_probe.log`.
