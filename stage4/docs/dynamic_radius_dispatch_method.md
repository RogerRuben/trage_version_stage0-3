# Dynamic-radius dispatch method

The single-day ABM no longer uses a fixed 6 km pickup radius.  Each pending order expands its own search radius according to accumulated waiting time:

| accumulated wait | radius |
| ---: | ---: |
| 0–2 min | 2.0 km |
| 2–4 min | 3.0 km |
| 4–6 min | 4.5 km |
| 6–8 min | 6.0 km |

Passenger patience is 8 minutes.  The dynamic-radius audit confirms that served orders are never matched outside their current radius stage and no order is served after patience expiry.

