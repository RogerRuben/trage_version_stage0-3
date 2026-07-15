# Simulator v3 preassignment plan insertion

Simulator v3 implements a two-layer vehicle plan:

```text
locked current pickup/service
→ reserved next pickup
→ reserved next service
```

The current request and the reserved request are stored separately.  A
reservation never changes vehicle coordinates and becomes executable only
after the locked service finishes and the release-time/deadline validation is
repeated.

Safe release uses the validation-date residual table and records the Q0.9
source, sample count, quantile and residual seconds.  HV reservations require
an asynchronous `DRIVER_RESPONSE` event; AV reservations are controlled by the
platform and can be created immediately.  Expired or invalid plans remove the
future stops, clear both reservation maps and return the request to `PENDING`.

Formal evidence is written by `audit_simulator_v3.py` to
`simulator_v3_preassignment_audit.json`; the audit joins every reservation
cycle to its transition and failure records instead of relying on an
architecture assertion.
