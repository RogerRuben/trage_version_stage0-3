"""Stage4 Simulator v3.

The v3 simulator separates request state, operator plans, and physical vehicle
execution.  Fleet controllers may publish :class:`VehiclePlan` objects, but
only :class:`VehicleExecutor` mutates vehicle locations and active legs.
"""

