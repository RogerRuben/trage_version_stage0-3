# Stage 3 S4 Test31 Dynamic Report

- Frozen M3 checkpoint: `965fc491cd77256f7889961d89932ec6be709bab04adcca358ac1b49f47c2cde`
- Prediction rows: 2,116,712
- Dynamic-complete routes: 29,864 / 30,000 (99.546667%)
- Validation dynamic-complete share: 99.503333%
- Decision-time only: `true`
- Predicted progression only: `true`
- Realized future time used: `false`
- Realized target columns persisted: `false`

## Test31 complete-route E/Q/C distributions

| Dimension | n | min | p25 | p50 | p75 | p90 | p97.5 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| crawl_E | 29,864 | 0.0238548 | 0.426122 | 0.495769 | 0.558709 | 0.618975 | 0.689766 | 0.850887 |
| crawl_Q | 29,864 | 0 | 0.0636401 | 0.095872 | 0.136561 | 0.183634 | 0.250254 | 0.580098 |
| crawl_C | 29,864 | 0 | 8.18861 | 12.3261 | 18.3744 | 27.6198 | 42.84 | 97.672 |
| stop_E | 29,864 | 0.177708 | 0.435302 | 0.487906 | 0.540741 | 0.590692 | 0.653094 | 0.869335 |
| stop_Q | 29,864 | 0 | 0.0537461 | 0.0772084 | 0.107585 | 0.146511 | 0.206225 | 0.501072 |
| stop_C | 29,864 | 0 | 6.24301 | 8.98135 | 13.5691 | 20.7274 | 34.6743 | 119.245 |
| speed_cv_E | 29,864 | 0.104132 | 0.43864 | 0.49369 | 0.544873 | 0.590183 | 0.643864 | 0.784899 |
| speed_cv_Q | 29,864 | 0 | 0.0444125 | 0.0679017 | 0.0988117 | 0.138661 | 0.205645 | 0.559794 |
| speed_cv_C | 29,864 | 0 | 5.63973 | 8.18621 | 12.7 | 19.1942 | 33.9169 | 119.245 |
| acceleration_rms_E | 29,864 | 0.164625 | 0.441617 | 0.503915 | 0.56924 | 0.630647 | 0.699396 | 0.887093 |
| acceleration_rms_Q | 29,864 | 0 | 0 | 0.0691304 | 0.159412 | 0.25852 | 0.380646 | 0.801827 |
| acceleration_rms_C | 29,864 | 0 | 0 | 24.8064 | 35.6939 | 49.6599 | 70.0992 | 123.323 |

## Frozen-cap exceedance

| Profile | Dimension/metric | Test31 exceed | Validation exceed | Frozen cap |
|---|---|---:|---:|---:|
| C | crawl_E | 21.8156% | 33.1647% | 0.568208 |
| C | crawl_Q | 29.4535% | 32.5550% | 0.127292 |
| C | crawl_C | 28.9312% | 32.0291% | 17.3873 |
| C | stop_E | 18.9559% | 30.1196% | 0.557009 |
| C | stop_Q | 17.4290% | 16.8872% | 0.122778 |
| C | stop_C | 19.3678% | 19.6107% | 15.3902 |
| C | speed_cv_E | 19.5051% | 20.4583% | 0.558161 |
| C | speed_cv_Q | 14.2178% | 13.2089% | 0.12348 |
| C | speed_cv_C | 16.4312% | 16.1335% | 15.2462 |
| C | acceleration_rms_E | 25.0536% | 21.9591% | 0.569089 |
| C | acceleration_rms_Q | 25.2913% | 22.8569% | 0.158209 |
| C | acceleration_rms_C | 25.4119% | 23.1584% | 35.4304 |
| C | **all 12 pass** | **5,833 (19.4433%)** | **4,761 (15.8700%)** | N/A |
| M | crawl_E | 9.1180% | 16.2474% | 0.624139 |
| M | crawl_Q | 13.3472% | 14.6729% | 0.17003 |
| M | crawl_C | 12.0011% | 13.9024% | 25.0873 |
| M | stop_E | 7.1357% | 14.4116% | 0.606859 |
| M | stop_Q | 6.1211% | 6.0768% | 0.167253 |
| M | stop_C | 5.4078% | 5.9562% | 27.6067 |
| M | speed_cv_E | 7.3098% | 7.7217% | 0.603634 |
| M | speed_cv_Q | 4.8955% | 4.8441% | 0.172142 |
| M | speed_cv_C | 4.7750% | 4.7670% | 27.4885 |
| M | acceleration_rms_E | 10.0087% | 8.8841% | 0.63062 |
| M | acceleration_rms_Q | 9.6571% | 9.0449% | 0.262157 |
| M | acceleration_rms_C | 10.0388% | 9.2057% | 49.602 |
| M | **all 12 pass** | **15,611 (52.0367%)** | **13,973 (46.5767%)** | N/A |
| A | crawl_E | 2.5047% | 5.2226% | 0.689676 |
| A | crawl_Q | 3.8642% | 4.4153% | 0.228902 |
| A | crawl_C | 3.1309% | 3.7553% | 39.3476 |
| A | stop_E | 1.7245% | 4.6029% | 0.667146 |
| A | stop_Q | 1.2724% | 1.3031% | 0.237905 |
| A | stop_C | 0.7836% | 1.2328% | 50.9794 |
| A | speed_cv_E | 1.6910% | 1.8626% | 0.655918 |
| A | speed_cv_Q | 1.0715% | 1.0686% | 0.249523 |
| A | speed_cv_C | 0.6195% | 0.8207% | 49.362 |
| A | acceleration_rms_E | 2.5080% | 2.3316% | 0.699274 |
| A | acceleration_rms_Q | 2.3674% | 2.4455% | 0.384615 |
| A | acceleration_rms_C | 2.4746% | 2.3751% | 70.3135 |
| A | **all 12 pass** | **25,080 (83.6000%)** | **23,978 (79.9267%)** | N/A |

The exact frozen Train CDF is applied; Test31 is not appended and no Test31 percentile is fitted. Incomplete routes remain dynamic `UNKNOWN` with null E/Q/C.
The Test31-versus-Validation comparison is descriptive only; it defines no shift threshold and triggers no retuning.
