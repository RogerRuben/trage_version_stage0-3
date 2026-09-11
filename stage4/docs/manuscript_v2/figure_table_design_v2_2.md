# V2.2 targeted figure and table design

This specification supersedes the F4 row of figure_callout_plan.csv and the older Figure 4 mechanism specification in ../paper_redesign/figure_plan_v2.md. Those CSV plans remain historical records, not the current V2.2 mechanism design. Unrelated figure/table designs are unchanged. No new experiment, figure rendering, or workbook is required in this revision.

## Figure 4: three distinct evidence panels

**4A — Canonical full-day empirical result.** Retain existing factorial service-rate versus q_A values (means .7258/.5984/.3924). Label the panel: “Canonical zero lead; 300-s patience; 27 factorial scenarios within 41 frozen scenarios; 30,000 requests.” Y-axis is full-day service rate, not M or arc survival. Source: existing factorial_effects.csv and benchmark_comparison.csv used by the main manuscript. Do not update any outcome values.

**4B — Fixed-state mechanism validation.** Use aligned gate plots: a small upper E panel with a clearly labeled logarithmic count axis, and a larger lower U/M panel with a shared linear count axis. Gate order: spatial, passenger, structural, evidence, Top-K, route returned, patience, solver eligible. Source: stage4/output/paper_enhancement/mechanism_validity/matching_capacity_summary.csv. E=95180/65444/31153/15775/1371/1371/48/48; U=720/493/238/124/124/124/12/12; M=718/493/238/124/124/124/12/12. Label “sums over ten fixed zero-lead states; not daily unique orders.” Shade Top-K as algorithmic compression; annotate 91.31% edge removal, zero M loss. Add the existing K10/20/40/80 result as a caption note: final M is identical within each sampled state. Do not plot selected assignments as another gate or imply E and M share a denominator.

**4C — Conditional RT qualification.** Four small multiples for q50/q75 at noon/evening, using RT variant on x and final M on y. Source: stage4/output/paper_enhancement/mechanism_validity/request_time_sensitivity.csv. Keep zero/Low/Base/High order. Values: q50 noon 4/9/7/10; q50 evening 1/3/4/8; q75 noon 3/9/8/11; q75 evening 0/5/6/11. Caption states “four fixed physical states; canonical history and descriptors retained; waiting cohorts vary.” Do not connect panels 4A–C with causal arrows, fit a trend over RT categories, or imply that local ordering reverses full-day service ordering.

**Draft combined caption.** Canonical service, matching-capacity validation, and timing sensitivity are distinct evidence layers. (A) Full-day service under zero lead and 300-second patience. (B) Ten-state gate counts show genuine maximum-matchable-order losses while Top-K removes redundant connectivity. (C) Four-state timing sensitivity changes capacity magnitudes and local fleet ordering. M counts simultaneous AV feasibility before assignment; neither state-summed M nor repeated-epoch edge survival is a daily service rate. No timing-invariant effect or counterfactual full-day RT outcome is identified.

The old .0939%/.0720%/.0445% opportunity survival figures remain supporting prose or an appendix inset only; they are not the primary capacity evidence.

## Tables in the revised results

- **Table MC1**, section 6.3.1: exact E/U/M/loss-of-M rows from matching_capacity_summary.csv. Gate loss is relative to the preceding row. Counts are summed over ten states. Stable temporary label MC1 avoids collision with legacy table numbering.
- **Table RT1**, section 6.3.3: zero/Low/Base/High pre-patience M 53/130/128/156, final M 8/26/25/40, and retention 15.09%/20.00%/19.53%/25.64%. Retention is ratio of sums, not mean of percentages. Population is four states and must not be pooled with MC1.
- **Experimental-design timing statement**, opening section 5.1: observed boarding/departure proxy as release; zero lead; 300-second patience; RT-sensitive magnitudes. This is prominent text, not hidden only in a limitation footnote.

## Interpretation boundaries

Recovery caption terminology: fingerprint-verified reconstruction. Missing original manifest and per-order hashes were not recovered. RT is latent scenario timing, not measured passenger requests. Prediction remains DECISION-RELEVANT; no common evaluator or decision-superiority panel is added. Do not add uncertainty bars from these few deterministic states or invent new statistical replications.
