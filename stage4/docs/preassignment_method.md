# Preassignment method

Preassignment is off by default.  In O2 and O3, busy or near-free vehicles may
be considered if their safe release time plus pickup ETA satisfies the order
deadline.  Each vehicle can reserve at most one next order.

The current implementation uses a safe-release proxy and records preassignment
status in order logs.  Full residual-quantile calibration remains an extension
point.

