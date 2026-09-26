# Stage 4 Manuscript V2 rewrite report

## Scope and frozen basis

- Branch: codex/stage2-v5-micro-transfer
- Authorized base: f333b0fa1e1949d71676a5731fbf24f74b7d99b3
- Scaffold commit: 78e8d7f
- Target journal family: primary *Transportation Research Part C*; secondary *Transportation Research Part E*
- New experiment, simulation, literature search, final figure production, and DOCX generation: none
- Canonical experiment products modified: no

## Manuscript

- Main-text word count: 8,052 by whitespace-token audit, including title, keywords, headings, and callouts
- Appendix word count: 1,556 by whitespace-token audit
- Working title: **From Ride-Hailing Trajectories to ODD-Aware Dispatch: Effective Service Capacity in Mixed Human-Driven and Autonomous Fleets**
- Alternative title 1: **Nominal Supply, Effective Capacity: ODD-Aware Rolling Dispatch for Mixed Human-Driven and Autonomous Ride-Hailing Fleets**
- Alternative title 2: **When an AV Hour Is Not an HV Hour: Empirical Capability Gates in Mixed-Fleet Ride-Hailing Dispatch**

Section word counts:

| Section | Words |
|---|---:|
| Abstract | 245 |
| 1. Introduction | 980 |
| 2. Data and empirical grounding | 1,141 |
| 3. Decision-time operational representation | 829 |
| 4. Rolling mixed-fleet assignment | 905 |
| 5. Experimental design | 772 |
| 6. Results | 1,671 |
| 7. Discussion and limitations | 1,278 |
| 8. Conclusion | 197 |
| References placeholder | 18 |

The section rows sum to 8,036; the title and keyword line contribute the remaining 16 tokens.

## Architecture

- Data-to-decision chain visible: **YES**
- Stage 4 tunnel vision removed: **YES**
- Three RQs present: **YES**
- Four contributions present: **YES**
- Central mechanism visible in abstract, introduction, results, and discussion: **YES**
- Hard feasibility separated from continuous suitability: **YES**
- Passenger acceptance separated from inferred behavior: **YES**
- Test31 described as legacy frozen benchmark rather than never-viewed final test: **YES**

## Claims and numbers

- Claim-audit PASS: 15
- Claim REWRITE: 0
- Claim REMOVE: 0
- Unsupported main claims: 0
- Number-audit PASS: 41
- Derived values among PASS rows: 2 (absolute and relative q_A contrast)
- Removed/not traceable: 1 (earlier approximate 99.4% mapping figure)
- Unresolved quoted numbers remaining in manuscript: 0

The rounded 99.4% mapping claim was removed because no single authoritative denominator was recovered during this rewrite. The manuscript retains the supported qualitative statement of near-complete identity coverage and explicitly records why the percentage is absent.

## Theory

- Propositions 1–3 present in main text: **YES**
- Proposition 4 placement: Appendix G
- Proofs for Propositions 1–4 present: **YES**
- Eligibility monotonicity distinguished from realized-service monotonicity: **YES**
- All capability and safety caveats preserved: **YES**

## Prediction

- Decision-relevant wording preserved: **YES**
- Universal prediction-superiority claim present: **NO**
- Mixed target-wise MAE reported: **YES**
- Fixed-state evidence extrapolated to full-day benefit: **NO**
- Decision-time leakage restriction explicit: **YES**

## Limitations

- No-active-rebalancing baseline disclosed: **YES**
- Repositioning effect correctly not identified by the main experiment: **YES**
- One-city and one-period scope disclosed: **YES**
- Passenger acceptance identified as exogenous rather than estimated: **YES**
- Capability envelope distinguished from safety certification: **YES**
- Fixed-route/no-joint-rerouting boundary disclosed: **YES**

## Literature

- Verified citations reused: 0
- Invented bibliographic entries: 0
- Semantic citation placeholders in main manuscript: 9
- Registered critical literature gaps: 3
- Registered total literature gaps: 9

No literature verification was authorized in this phase. The References section therefore remains intentionally unpopulated, and every required external support location is represented by a semantic placeholder and the gap registry.

## Figures and tables

- Main figure callouts planned: 5
- Main table callouts planned: 5
- Appendix figure callouts planned: 3
- Appendix table callouts planned: 3
- Final figure or table art generated: 0

## QA

- Semantic gates: PASS
- Main word-count gate (8,000–9,500): PASS
- Markdown heading and display-equation review: PASS
- Number audit: PASS with one explicitly removed untraceable value
- Claim audit: PASS
- Equation traceability: 12/12 registered
- Canonical output mutation check: PASS
- Git diff check: PASS
- Clean worktree: expected after Commit B; verified again after commit

## Final decision

**GO_LITERATURE_VERIFICATION**

The manuscript architecture is coherent, the primary mechanism is visible, the frozen numerical claims are audited, and missing external scholarship is explicitly isolated. The next authorized phase should verify literature and replace semantic placeholders; it should not reopen canonical experiments.
