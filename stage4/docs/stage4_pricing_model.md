# Stage4 Pricing Model

Stage4 uses explicit passenger fares, HV driver payments, AV operating costs,
and platform profit accounting. The pricing scenarios are configured in
`stage4/config/pricing_scenarios.json`.

Passenger fare for order `o` and vehicle mode `m` is:

```text
fare_om = base_fare + distance_fare + time_fare
        + surge_adjustment
        + vehicle_type_adjustment
        + passenger-funded stress surcharge
```

Stress surcharge uses the deployable Stage3 condition vector:

```text
stress = mean(lcs_expected, pmis_expected, rts_expected)
```

IIS is optional and availability-aware; missing IIS is not interpreted as zero
stress. Surge is bounded by configured floor/cap values. Compensation funding
is explicitly split between passenger and platform shares.

The current experiment includes:

- `P0_uniform`
- `P1_platform_funded_comp`
- `P2_passenger_funded_comp`
- `P3_shared_comp`
- `P4_av_discount_hv_comp`
- `P5_three_stakeholder_balanced`

These are scenario mechanisms, not observed platform prices.
