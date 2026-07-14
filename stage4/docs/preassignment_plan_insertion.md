# Preassignment plan insertion

Preassignment is intentionally not enabled in the Phase 1 v3 smoke run.

The v3 data model now supports the correct future insertion pattern:

```text
locked current service
→ future pickup
→ future service
```

However, the actual insertion and invalidation logic remains pending Phase 3:

- one reserved request per vehicle;
- one reserved vehicle per request;
- Q0.9 safe-release buffer;
- HV offer-response before reservation;
- reservation invalidation after ETA or pickup-deadline changes.

The current audit reports:

```text
preassignment_audit_status = NOT_RUN_PHASE_1
```

