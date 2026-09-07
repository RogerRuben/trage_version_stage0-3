# Neutral AV identity test

**HIDDEN_ASYMMETRY.** The first preregistered state fails candidate-graph identity. Per taskbook B5, testing stops at this counterexample; the other nine states were not run.

Source: ODD_Q50_M_P70_REFERENCE, 2016-10-31 07:30:00+08:00. Frozen state SHA256: 4ed0e2fba8c6130471b3c682e154159db10bc296eae1c37e61c462a81d4f221a. Restoring the prior snapshot from canonical logs reproduces this hash exactly. The state contains 55 waiting orders and 310 available vehicles (234 AV, 76 HV).

| Variant | Candidate E | Unique orders U | Maximum matching M | Selected count | Pickup objective (s) |
|---|---:|---:|---:|---:|---:|
| Original labels; explicit AV restrictions disabled | 109 | 25 | 24 | 24 | 2814.493941 |
| Same physical vehicles all labeled AV; explicit AV restrictions disabled | 110 | 25 | 24 | 24 | 2814.493941 |

Candidate graphs differ by one added arc. Selected assignment identities are exactly equal. This is a concrete **edge increase without matching-capacity increase** in this state, not a full gate-funnel result.

## Actual production comparison

The diagnostic calls the production _RollingORFleetControlCore.time_trigger, intercepts its actual solver input, and solves the captured sparse arcs using the existing solver. It does not substitute a hand-written candidate builder. Passenger acceptance is universal, AV readiness is enabled equally in both variants, exposures are zero, Gamma and cost are disabled. Coordinates, IDs, arrival/patience state, service duration, session endpoints, search, Top-K, and routing are held fixed. Available vehicles are the already-restored frozen available set. No simulated time advances or vehicle assignments are executed.

The sole requested type change is original HV -> AV. Crucially, an implicit HV-only session check remains in production: rolling_or_control.py checks finite predicted duration and predicted_end <= availability_end_time only when vehicle_type == HV. There is no neutral-mode switch that preserves this inherited session rule independently of the type label. Silently adding that switch inside the test would conceal the asymmetry under examination.

## Counterexample

- Order: 855699a09cd4dd73d3829ba3829e1c54 (native request 1800).
- Vehicle: HV_S3_00361 (native vehicle 361), with unchanged session end 07:33:19.
- State time: 07:30:00; remaining session: 199 seconds.
- Predicted service duration: 730.3849370479584 seconds, already longer than the remaining session even before pickup.
- As HV: rejected by the production session-end gate.
- As AV with the same session endpoint: that gate is bypassed and the arc is admitted.

The counterexample points toward **relaxing**, not tightening, eligibility when the label becomes AV. It therefore cannot by itself explain the reported AV-heavy service decline. It does invalidate the stronger assertion that the label alone is neutral under fixed session semantics. Canonical AV fixtures normally have full-day availability while HVs have empirical sessions; the desired identity test requires an explicit decision about inherited session behavior under relabeling.

## Bounded validation

One focused test passes, including a synthetic 60-second session / 100-second service counterexample through the real production candidate path. Real-state runtime: 10.11 seconds, 1,099 routed arcs, zero routing failures, no GPU, sparse graph only. Candidate and selected identities are retained in mechanism_validity/neutral_av_identity.csv.

No production policy or canonical output was changed. Recommended next correction for review: represent the session-end/service-evidence policy independently of HV/AV label, then rerun B under inherited identical session semantics before interpreting C/D. The taskbook does not authorize silently changing canonical vehicle-session policy to force this test to pass.
