# HV idle repositioning method

The current idle setting is a lightweight joint idle-management proxy.  It
moves a small reproducible share of idle vehicles to nearby grid cells and
records time, distance, and cost.  It is not yet the intended training-derived
`P(z'|z,t)` empirical repositioning model, and it is not a recovered true driver
cruising trajectory.

O1 and O3 combine HV repositioning and AV rebalancing as a joint idle-management
package.  They should not be interpreted as separately identifying HV-only and
AV-only effects.
