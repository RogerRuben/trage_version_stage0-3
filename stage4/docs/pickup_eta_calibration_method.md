# Pickup ETA calibration method

Pickup ETA no longer uses straight-line distance divided by a fixed speed.
It uses:

`pickupRoadDistance = HaversineDistance × CircuityFactor`

`pickupETA = pickupRoadDistance / EmptySpeed(zone,time)`

Circuity is estimated from training-day matched route length relative to
haversine distance.  Empty-speed priors are derived from training-day loaded
route speeds with a conservative multiplier, not from full inter-trip gaps.  If
a zone-time cell is sparse, the simulator falls back to a training global prior,
not to test-day future information.
