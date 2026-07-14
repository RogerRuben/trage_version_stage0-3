# Pickup ETA calibration method

Pickup ETA no longer uses straight-line distance divided by a fixed speed.
It uses:

`pickupRoadDistance = HaversineDistance × CircuityFactor`

`pickupETA = pickupRoadDistance / EmptySpeed(zone,time)`

Circuity and empty-speed priors are estimated only from training-day data.  If a
zone-time cell is sparse, the simulator falls back to a training global prior,
not to test-day future information.

