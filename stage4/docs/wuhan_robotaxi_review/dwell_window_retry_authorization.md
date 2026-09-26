# Execution-budget amendment — 2026-09-26

User explicitly authorized retry of q75 / 120-s additional pickup overhead with
a 60-minute per-condition wall-time limit. No scientific parameter changes.
Original configuration remains byte-for-byte unchanged; the retry overrides only
`scenario_timeout_s` from 1800 to 3600 in memory. The other five conditions are
reused, not rerun. Original timeout and completed products remain untouched.

Runner: `stage4.analysis.dwell_window_retry`. New output directory:
`stage4/output/paper_enhancement/dwell_deterministic_window_retry1`.
It verifies the frozen config and five-condition inventory, hashes all original
files before and after, and emits a separate combined summary only on success.
The underlying condition runner is unchanged except for displaying the actual
configured timeout in its error message. Single CPU process, sparse routing,
same cache release, same checkpoint, same integrity checks; no GPU.

Origin skill: academic-research-suite / experiment-agent. This is the explicitly
authorized retry, not an automatic retry after the timeout. A further failure
will be reported without silently rerunning again.
