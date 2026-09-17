# Stage 3 S4 Test31 Static Report

All encountered physical complexes are evaluated. `D_c` is recomputed over unique INCOMING/OUTGOING boundary-edge Valhalla road classes for the full frozen 43,685-complex network; the legacy S2B INTERNAL-edge QA field is not used.

- Test31 complex encounters: 538,355
- Unique exposed complexes: 2,107
- Grade-separated complex encounters (descriptive only): 9,077

## Unique-complex distributions

| Dimension | n | min | p25 | p50 | p75 | p90 | p97.5 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_c | 2,107 | 2 | 3 | 3 | 4 | 6 | 10 | 16 |
| M_c | 2,107 | 1 | 4 | 9 | 9 | 16 | 25 | 64 |
| D_c | 2,107 | 1 | 2 | 2 | 2 | 3 | 3 | 5 |
| L_c | 2,107 | 0 | 0 | 0 | 12 | 36 | 74 | 456 |

## Frozen-cap exceedance

| Profile | A_c | M_c | D_c | L_c | All four pass |
|---|---:|---:|---:|---:|---:|
| C | 444 (21.0726%) | 355 (16.8486%) | 280 (13.2890%) | 536 (25.4390%) | 1,434 (68.0589%) |
| M | 226 (10.7262%) | 101 (4.7935%) | 22 (1.0441%) | 211 (10.0142%) | 1,808 (85.8092%) |
| A | 55 (2.6103%) | 49 (2.3256%) | 22 (1.0441%) | 55 (2.6103%) | 2,012 (95.4912%) |

Frozen caps are applied per encountered complex without averaging or Test31 refitting.
