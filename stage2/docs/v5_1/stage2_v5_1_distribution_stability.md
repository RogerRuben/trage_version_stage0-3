# Stage 2 v5.1 distribution stability audit

This audit is read-only over frozen v5 predictions and scenarios. It does not trim rows or replace the frozen rolling mean metric with P50.

| Protocol | Split | Date | Status | Pace mean max | Mean/P50 max | P99.9 mean | Max-row MAE share | Route mean RMSE | P95 width |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| development | calibration | 20161024 | FAIL | 1.659189 | 3.176 | 0.367849 | 0.1594% | 282.25 | 535.43 |
| development | evaluation | 20161025 | FAIL | 3.026368 | 4.722 | 0.368669 | 0.1373% | 335.82 | 549.24 |
| development | evaluation | 20161026 | FAIL | 1.388765 | 2.790 | 0.367296 | 0.2240% | 385.21 | 543.57 |
| development | evaluation | 20161027 | PASS | 1.868844 | 3.330 | 0.365653 | 0.0666% | 312.74 | 540.83 |
| development | validation_model | 20161022 | FAIL | 2.773353 | 4.502 | 0.374043 | 0.1889% | 349.39 | 563.26 |
| development | validation_model | 20161023 | FAIL | 2.319215 | 3.945 | 0.371709 | 0.0723% | 228.75 | 523.68 |
| fold_1 | calibration | 20161021 | FAIL | 6.147446 | 4.968 | 0.456293 | 0.0670% | 274.91 | 481.13 |
| fold_1 | evaluation | 20161022 | FAIL | 98.090706 | 51.842 | 0.476035 | 0.1882% | 354.44 | 494.29 |
| fold_1 | evaluation | 20161023 | FAIL | 35.173031 | 19.924 | 0.463846 | 0.0720% | 231.41 | 457.98 |
| fold_1 | validation_model | 20161019 | FAIL | 13.092847 | 9.219 | 0.449199 | 0.0934% | 227.43 | 459.10 |
| fold_1 | validation_model | 20161020 | FAIL | 76.697678 | 52.649 | 0.448963 | 0.1850% | 239.91 | 463.06 |
| fold_2 | calibration | 20161023 | PASS | 0.542985 | 1.318 | 0.383931 | 0.0695% | 231.70 | 417.24 |
| fold_2 | evaluation | 20161024 | FAIL | 0.546084 | 1.328 | 0.382818 | 0.1540% | 290.19 | 426.12 |
| fold_2 | evaluation | 20161025 | FAIL | 0.512996 | 1.364 | 0.382705 | 0.1336% | 346.13 | 435.89 |
| fold_2 | validation_model | 20161021 | PASS | 0.563371 | 1.428 | 0.384087 | 0.0650% | 281.12 | 435.45 |
| fold_2 | validation_model | 20161022 | FAIL | 0.516179 | 1.336 | 0.380418 | 0.1838% | 362.41 | 445.94 |
| fold_3 | calibration | 20161025 | FAIL | 253.722122 | 209.370 | 0.456737 | 3.4173% | 342.09 | 562.50 |
| fold_3 | evaluation | 20161026 | FAIL | 7584.049805 | 3807.918 | 0.459252 | 52.4742% | 4432.48 | 554.46 |
| fold_3 | evaluation | 20161027 | FAIL | 12.499694 | 14.729 | 0.454031 | 0.0667% | 308.59 | 548.59 |
| fold_3 | validation_model | 20161023 | FAIL | 2096.867920 | 1078.848 | 0.448212 | 0.0723% | 4085.59 | 530.67 |
| fold_3 | validation_model | 20161024 | FAIL | 33267.058594 | 14245.245 | 0.455237 | 83.1658% | 11992.55 | 545.54 |
| legacy | calibration | 20161027 | PASS | 1.222037 | 1.818 | 0.376133 | 0.0664% | 312.95 | 521.65 |
| legacy | legacy | 20161031 | PASS | 1.285911 | 2.280 | 0.370758 | 0.0263% | 243.81 | 511.80 |
| legacy | validation_model | 20161025 | PASS | 2.828800 | 3.750 | 0.376276 | 0.1018% | 335.56 | 529.70 |
| legacy | validation_model | 20161026 | FAIL | 10.791489 | 9.804 | 0.373765 | 0.2238% | 385.60 | 525.69 |

Overall frozen-v5 stability status: `FAIL`; failed protocol-days: 19/25.
