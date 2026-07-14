# Preassignment method

Preassignment is off by default.  O2 and O3 do not activate it unless the
simulator is explicitly run with `--enable-preassignment`.

The previously implemented safe-release proxy is retained only as an
experimental code path.  It is not yet a valid formal preassignment mechanism
because the simulator still needs a two-layer vehicle state that keeps the
current service and the reserved next order separate.  Full residual-quantile
safe-release calibration remains an extension point.
