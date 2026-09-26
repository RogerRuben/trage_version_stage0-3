# Stage 3 S3 Capability-Envelope Methodology

This phase freezes three nested hypothetical capability scenarios `C ⊆ M ⊆ A`; it does not estimate safety, legality, failure, disengagement, or accident probability.

1. Historical tokens are resolved into typed full-network, reverse-overlay, or unresolved identities. Broken identity breaks continuity.
2. The production complex parser recognizes incoming → zero or more internal → outgoing edges and never splices across a gap.
3. Static A/M/D/L caps use one observation per unique Train-exposed complex and `higher` quantiles at 0.75/0.90/0.975. `D_c` counts unique `valhalla_road_class` on INCOMING/OUTGOING boundary edges; INTERNAL edges are excluded.
4. Dynamic inputs are frozen-M3 decision-time predictions. Predicted P50 travel time advances/weights exposure; realized future timing is forbidden.
5. Each dimension uses a global Train predicted-time-weighted mid-CDF. Tail is strict `z > 0.90`. Route E/Q/C preserve token order; threshold fitting gives every complete route one vote. `pi_k` freezes marginal dimension caps, not joint route acceptance rates.
6. Speed caps remain 60/80/120 km/h. Maneuver, roundabout, restriction, and unknown rules are categorical and non-compensatory. Certified prohibition is incompatible; non-certification is not legal permission. Grade separation, bridge, and tunnel are descriptive only.
7. Validation 25–27 is sanity only. The profile is hash-bound before and after. Test31 aliases are hard rejected.

Frozen profile: `stage3/config/stage3_av_capability_profiles.json` SHA-256 `bea54c12a0c013995c3644a0bab84b35413d5df8a6785dfdd6a03fd49f32978e`.

`S4_AUTHORIZED = NO`; `NEXT_PHASE_AUTHORIZED = NO`.
