# Stage 0 v4 known data limitations

Status: provisional register; finalize after the v2 route and connector reviews.

- Sparse GPS traces can make parallel roads and complex interchange movements
  observationally indistinguishable.
- The source road network contains missing, conflicting, or incomplete layer,
  bridge, tunnel, and one-way attributes. v4 preserves these distinctions where they
  are encoded but cannot infer absent civil-engineering detail.
- Zero-length graph-only connectors represent co-located legal endpoint transfers;
  they do not prove the physical presence of a ramp. Ambiguous cases are explicitly
  routed to human review.
- Some valid trips may trigger conservative U-turn or detour diagnostics when the
  basemap is incomplete. Such flags remain separate from adjudicated algorithmic
  errors.
- Low-quality and failed routes are accounted for and excluded rather than silently
  promoted. The final geographic limitation list requires full-date diagnostics.

These limitations are research constraints, not claims that all remaining routes are
correct. They may be reopened only with new evidence of a systematic error described
in the Stage 0 contract.
