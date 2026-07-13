# HV agent reconstruction method

For 2016-10-23, each observed `driver_id` is treated as one HV agent.  Online sessions are reconstructed from the driver's full-day historical order sequence, using observed order times only to infer existence, online windows, and session initial locations.

Current reconstruction:

- historical orders: 114,356
- HV agents: 18,301
- HV sessions: 31,627
- selected session gap threshold: 90 minutes

Future historical locations are not used after simulation starts.  Vehicle position, availability, income, and stress burden are updated only by simulated assignments.

