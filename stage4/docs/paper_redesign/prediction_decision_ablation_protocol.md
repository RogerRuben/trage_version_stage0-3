# Frozen-state prediction-to-decision ablation protocol

Source trajectory: frozen `ODD_Q50_M_P70_REFERENCE` (q50, profile M, p70, three REFERENCE Gamma constraints). Ten timestamps were fixed before viewing outcomes: 07:30, 08:30, 12:00, 13:00, 17:00, 17:30, 18:00, 18:30, 21:00, and 23:00 on Test31.

At each timestamp the analysis reconstructs, from frozen inputs, waiting orders, patience/carry state, available HV/AV identities and positions, passenger acceptance, and preceding cumulative exposure. Candidate radii, Top-K=20, HV session admission, pickup-ETA correction, and the sequential exact solver are unchanged. Sparse arcs are routed once with audited `SINGLE_SOURCE_MATRIX` and reused. No state advances and no assignment enters a later timestamp.

Variants are: **P**, frozen M3 decision-time predictions and current interface; **H**, a Train-only 09–24 historical 15-minute × route-token-quartile median of observed crawl, stop, speed-CV, and acceleration-RMS labels, transformed through Train empirical mid-CDFs; and **D0**, the P hard/static/speed interface with the dynamic Gamma row removed. H never reads Test31 realizations when predicting. Test31 labels are opened afterward for evaluation only.

Outputs are same-unit opportunity counts, AV solver-input arcs, selected HV/AV assignments, selected family exposures, pickup objective, order overlap, and AV-arc overlap. No full-day statistic is computed. `DECISION-RELEVANT` requires assignment changes in at least 25% of snapshots and mean AV-arc Jaccard below 0.90; otherwise the implementation labels evidence modest or negligible.
