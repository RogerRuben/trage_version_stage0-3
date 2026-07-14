# FleetController design

`FleetController` reads a system snapshot at each decision epoch and returns plans. It does not directly mutate vehicle physical state.

Current Phase 1 flow:

1. read pending requests and controllable vehicles;
2. generate coarse BallTree candidates;
3. query RoutingEngine for candidate pickup ETA;
4. validate session, pickup deadline, pickup ODD, and service ODD;
5. compute edge economics;
6. solve Safe GlobalMatch on sparse candidate edges;
7. publish VehiclePlan objects;
8. VehicleExecutor executes legs and updates physical state.

Price-Aware and Balanced strategy modules exist, but formal Balanced global constraints are pending Phase 5.

