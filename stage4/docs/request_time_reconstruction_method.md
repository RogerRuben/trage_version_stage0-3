# Request-time reconstruction method

The observed trajectory start is treated as `observed_boarding_time`, not as a
true passenger request time.  The simulator creates scenario request times:
RT-Low, RT-Base, and RT-High.

Training-day driver chains from 20161019-20161022 are used only to bound
plausible request lead times.  They are not interpreted as the true passenger
pre-pickup distribution because inter-trip gaps include waiting, cruising,
other-platform activity, matching, response, and pickup.

The lower bound is at least minimum matching/response time plus minimum pickup
time.  The upper bound is clipped by the business-day warm-up boundary and by
training-chain feasibility.  Clipping is recorded in the generated demand
tables.

