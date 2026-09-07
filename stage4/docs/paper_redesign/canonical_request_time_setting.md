# Canonical request-time setting

Base: b2e3f9cabfff9855cc50bbf23be17a0436a84685.

The frozen 41-scenario FleetPy experiment uses **observed trajectory departure as request release**, with **zero request lead** and **300 seconds of pickup patience**. It does not use RT-Low, RT-Base, or RT-High.

Evidence:

- stage4/replay_foundation.py, load_replay_orders: request_time = timestamp(departure_time).
- stage4/fleetpy_adapter/test31_demand_adapter.py, load_all_test31_requests: reads request_time from the frozen replay base without subtracting lead time.
- All 41 stage4/output/final_experiments/*/scenario_config.json runtime configurations specify max_pickup_wait_s=300; none contains request_time_scenario.
- Direct one-to-one comparison of the 30,000 Stage1 Test31 order_base rows against Moderate-profile replay rows: 30,000/30,000 request timestamps equal their source departure timestamps; all lead differences are 0 seconds.
- stage4/dispatch/rolling_or_control.py: deadline = sim_time_s + max_pickup_wait_s; an order expires at simulation_time >= deadline. Routed pickup ETA must fit remaining patience.

If observed departure is interpreted as observed boarding, the canonical request-time proxy equals observed boarding; the canonical replay does **not** reconstruct a latent pre-boarding request time. Simulated pickup happens after this release.

The separate stage4/docs/request_time_reconstruction_method.md refers to the decoupled ABM pipeline. Its implementation is stage4/scripts/build_decoupled_abm_environment.py, attach_request_times, whose default study date is 20161023. It uses training-chain quantiles, response/pickup lower bounds, scenario fractions 0.25/0.50/0.75, order-specific jitter, and clipping. That method must not be attributed to the canonical Test31 FleetPy runs. The available RT summary contains 114,356 orders, not the canonical 30,000-order sample.

RT sensitivity was not launched following the B5 hidden-asymmetry stop. A future RT comparison needs the actual frozen chain-bound parameters and the existing transformation applied to the same Test31 physical state; neither a guessed RT-Base label nor the old aggregate mean leads is an acceptable replacement.
