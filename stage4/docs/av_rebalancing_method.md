# AV rebalancing method

AVs start from training-data depots and, after service, remain in the field.
The current O1/O3 implementation adds a lightweight zone-level rebalancing
proxy that consumes empty distance, time, and cost.  It does not use
2016-10-23 future demand.

