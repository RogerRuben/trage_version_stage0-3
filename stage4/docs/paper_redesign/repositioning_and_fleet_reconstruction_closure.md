# Repositioning and fleet-reconstruction closure

The canonical mixed-fleet counterfactual uses a frozen empirical fleet reconstruction. Its all-day spatial distribution is close to the full Test31 supply record (TVD 0.054, Spearman correlation 0.965, and top-decile hotspot overlap 0.889), with the separately audited evening comparison supporting the same conclusion. The reconstruction is therefore retained: `KEEP_CURRENT_FLEET_RECONSTRUCTION = YES`.

Idle vehicles remain at their current or most recent drop-off position. The baseline contains neither random idle roaming nor proactive rebalancing. Consequently, the estimated fleet-transition effects are conditional on a no-active-rebalancing operating policy.

A deterministic, Train-only demand-based repositioning policy was preregistered as a robustness analysis. Reproducibility required independent `SCALAR_ROUTE` calls. The first deterministic Q50 full-day attempt exceeded the preregistered computational-cost threshold before producing a valid scientific outcome. The exercise was stopped rather than replacing deterministic routing or tuning the policy after seeing outcomes.

The repositioning effect is therefore **not identified due to computational cost under reproducibility-preserving routing**. This result is neither evidence that repositioning is ineffective nor evidence that it would remedy the observed transition. Proactive rebalancing remains an explicit model limitation and future extension; it is not part of the canonical treatment contrast.
