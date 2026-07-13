# Single-day agent-based Stage4 design

This Stage4 bottom layer is fixed to the 2016-10-23 test day.  The demand stream is not the old 15,000-order Stage3 subset; it is the full Stage0-valid day, with the deployable simulation universe restricted only by the real model-inference coverage audit.

Pipeline:

1. full-day Stage0 OD/order universe;
2. full-day route-conditioned estimated-entry inputs;
3. held-out RC-MSTNet Stage2 inference;
4. Stage3 Core and Core+IIS inference;
5. Stage4 condition-vector export;
6. full-day HV agent/session reconstruction from observed drivers;
7. training-data depot AV initialization with AV/HV ratio capped at 5%;
8. dynamic-radius candidate generation;
9. window-level Hungarian matching.

The old multi-fold synthetic-supply Stage4 outputs are now treated as mechanism regression tests, not as the final operational simulation.

