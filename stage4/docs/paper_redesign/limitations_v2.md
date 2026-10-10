# Limitations v2

- Operational suitability is not AV safety, legal compliance, or commercial certification.
- The C/M/A profiles and Gamma values are frozen operational envelopes, not universal capability standards or safety probabilities.
- Test31 is one temporal/network context; numerical effects do not automatically generalize to other cities, fleets, or demand regimes.
- Passenger acceptance is a controlled scenario input rather than an estimated behavioral choice model.
- The canonical baseline has no proactive idle-vehicle rebalancing. A deterministic robustness run exceeded its computational threshold, so repositioning effectiveness is not identified.
- Fixed-state prediction ablation preserves physical states and reveals local decision consequence only. It cannot identify full-day service or future-state effects.
- P does not dominate the Train-only historical baseline on every raw prediction metric; the supported claim is decision relevance of the multivariate interface.
- Same-epoch Gamma monotonicity does not imply monotonic full-day outcomes.
- Fleet reconstruction is strongly spatially representative by audited summaries but is not an optimized or experimentally randomized fleet.
- Map matching and upstream filters substantially improve route quality but cannot guarantee correctness for every retained order.
