# Theory notes v2

## Notation

At epoch (t), (x_{vo}\in\{0,1\}) selects sparse vehicle–order arc ((v,o)). The AV subset is \(\mathcal A_t^A\). For family \(f\in\{static,dynamic,speed\}\), \(e_{vo}^f\ge0\) is arc exposure excess, \(Z_t^f\) accumulated exposure, and \(N_t^A\) accumulated AV assignments. The exact solver maximizes critical, total, and carry-over matches, then minimizes pickup ETA; its optional last level minimizes operating cost.

## Proposition 1 — cumulative family-exposure guarantee

**Statement.** If
\[
Z_t^f+\sum_{(v,o)\in\mathcal A_t^A}e_{vo}^f x_{vo}
\le \Gamma_f\left(N_t^A+\sum_{(v,o)\in\mathcal A_t^A}x_{vo}\right),
\]
then \(Z_{t+1}^f\le\Gamma_fN_{t+1}^A\), and, when \(N_{t+1}^A>0\), \(Z_{t+1}^f/N_{t+1}^A\le\Gamma_f\).

**Assumptions.** State and selected AV exposures update additively exactly as in `CumulativeExposureState`; the family row is enabled.

**Proof.** Substitute \(Z_{t+1}^f=Z_t^f+\sum e_{vo}^fx_{vo}\) and \(N_{t+1}^A=N_t^A+\sum x_{vo}\) into the premise, then divide by positive \(N_{t+1}^A\). ∎

**Guarantee:** day-to-date mean exposure-excess cap. **Not guaranteed:** per-trip cap, safety probability, or certification. **Placement:** Methods after the rolling formulation.

## Proposition 2 — same-epoch feasible-set monotonicity

**Statement.** Fix fleet state, waiting orders, sparse candidates, acceptance, pickup estimates, exposure values, and all non-Gamma constraints. If \(\Gamma_f'\ge\Gamma_f\) for every family, then \(\mathcal F_t(\Gamma)\subseteq\mathcal F_t(\Gamma')\); relaxation cannot worsen the highest-priority objective in that state.

**Assumptions.** The current state satisfies both budgets, exposure is nonnegative, and only Gamma changes.

**Proof.** The implemented row is equivalent to \(Z_t^f+\sum e_{vo}^fx_{vo}\le\Gamma_f(N_t^A+\sum x_{vo})\). Its right side weakly increases under \(\Gamma_f'\), with all other rows fixed. Every old feasible point remains feasible; maximization over a superset cannot reduce the highest-priority optimum. ∎

**Guarantee:** local same-epoch monotonicity. **Not guaranteed:** monotonic full-day service, because current decisions alter later positions and queues. **Placement:** Methods or theory appendix.

## Proposition 3 — separate budgets versus a weighted scalar

**Statement.** Bounds \(E_f\le\Gamma_f\) imply \(\sum_fw_fE_f\le\sum_fw_f\Gamma_f\) for every \(w_f\ge0\). The reverse fails in general.

**Proof.** Multiply and sum the family inequalities. Conversely, with two families, \(w_1=w_2=1\), unit family caps, and scalar budget 2, \((E_1,E_2)=(2,0)\) satisfies \(E_1+E_2\le2\) but violates \(E_1\le1\). ∎

**Guarantee:** distinct operational meaning without cross-family compensation. **Not guaranteed:** safety interpretation or globally optimal caps. **Placement:** justification for the three Gamma rows.

## Proposition 4 — local epsilon pickup-quality guarantee

**Statement.** Conditional on the preceding critical-, total-, and carry-over-match optima, let \(P_t^*\) be the minimum aggregate pickup ETA. When the optional cost level is enabled, its solution obeys \(P_t\le(1+\epsilon_W)P_t^*+\delta\), where \(\delta\) is solver tolerance.

**Proof.** The implementation appends exactly this upper-bound row before minimizing operating cost while retaining the earlier lexicographic equalities. ∎

**Guarantee:** epoch-level relaxation relative to the conditional aggregate pickup optimum. **Not guaranteed:** per-passenger wait or full-day mean/P95 pickup. **Placement:** cost-robustness appendix.
