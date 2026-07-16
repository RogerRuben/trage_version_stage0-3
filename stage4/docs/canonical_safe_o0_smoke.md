# Canonical Stage 4 Safe/O0 engineering smoke

This is the only Stage 4 run permitted during pipeline rebaseline: 1,000
2016-10-23 orders, Safe GlobalMatch-MinPickup, O0/Stay, preassignment off, and
replication 1. It is a functional test and is not a scientific result.

Demand conditions come from the canonical Stage 3 dispatch vector. Initial HV
locations and AV depots come from the 2016-10-20 training-day sample, while the
online schedule is an explicit engineering-smoke scenario. No 2016-10-23
driver supply or future demand is used to initialize supply.

Service execution uses the Stage 2 predicted distribution plus a pre-generated
residual stream. Historical realized service duration is absent from dispatch
and execution. The AV profile is an external scenario prior and is not calibrated
on the test day.
