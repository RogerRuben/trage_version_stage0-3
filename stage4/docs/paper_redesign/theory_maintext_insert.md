### Structural properties of family-specific rolling control

The three operational families are controlled separately. Before epoch (t), let (Z_t^f) and (N_t^A) denote cumulative family-(f) exposure excess and AV assignments. The optimization imposes

\[
Z_t^f+\sum e_{vo}^f x_{vo}\le\Gamma_f\left(N_t^A+\sum x_{vo}\right).
\]

By direct substitution of the additive state update, every selected solution satisfies (Z_{t+1}^f/N_{t+1}^A\le\Gamma_f) whenever at least one AV assignment has accumulated. Gamma therefore caps day-to-date mean exposure excess; it is neither a per-trip prohibition nor a safety probability.

For a fixed epoch state, increasing every Gamma weakly enlarges the feasible set. The highest-priority lexicographic service objective therefore cannot deteriorate in that same state. This local result does not imply monotonic full-day outcomes because assignments change future vehicle positions and queues.

Separate family limits also prevent compensation hidden by a scalar score. They imply every nonnegative weighted aggregate bound, whereas the converse fails: with two unit caps, the vector ((2,0)) satisfies the scalar bound (E_1+E_2\le2) but violates the first family cap. Finally, in cost-enabled runs, the last lexicographic level constrains aggregate epoch pickup ETA to ((1+\epsilon_W)P_t^*) (plus solver tolerance) before minimizing operating cost; this is not a full-day wait guarantee.
