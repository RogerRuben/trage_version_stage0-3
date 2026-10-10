# Stage 3 S3 Validation Sanity Report

Validation dates `20161025`–`20161027` were processed only after Train profile freeze.

- Orders: 30,000; tokens: 2,100,209
- Full-network share: 95.852556%
- Reverse-overlay share: 4.102496%
- Unresolved share: 0.044948%
- Complete dynamic routes: 29,851/30,000 (99.503333%)
- Nestedness sanity: `PASS`
- Profile SHA before: `bea54c12a0c013995c3644a0bab84b35413d5df8a6785dfdd6a03fd49f32978e`
- Profile SHA after: `bea54c12a0c013995c3644a0bab84b35413d5df8a6785dfdd6a03fd49f32978e`

Static frozen-cap exceedance: `{"A_A_c": 0.026697892271662763, "A_D_c": 0.010304449648711944, "A_L_c": 0.026697892271662763, "A_M_c": 0.024355971896955503, "A_all_caps_pass_count": 2037, "C_A_c": 0.21124121779859484, "C_D_c": 0.13114754098360656, "C_L_c": 0.2571428571428571, "C_M_c": 0.1718969555035129, "C_all_caps_pass_count": 1447, "M_A_c": 0.10725995316159251, "M_D_c": 0.010304449648711944, "M_L_c": 0.10491803278688525, "M_M_c": 0.05011709601873536, "M_all_caps_pass_count": 1823}`.

Dynamic frozen-cap exceedance: `{"A_acceleration_rms_C": 0.02375129811396603, "A_acceleration_rms_E": 0.023315801815684566, "A_acceleration_rms_Q": 0.024454792134266858, "A_all_caps_pass_count": 23978, "A_crawl_C": 0.037553180797963215, "A_crawl_E": 0.052226056078523334, "A_crawl_Q": 0.044152624702690026, "A_speed_cv_C": 0.008207430236842986, "A_speed_cv_E": 0.018625841680345718, "A_speed_cv_Q": 0.010686409165522093, "A_stop_C": 0.01232789521289069, "A_stop_E": 0.046028608756825566, "A_stop_Q": 0.013031389233191517, "C_acceleration_rms_C": 0.2315835315399819, "C_acceleration_rms_E": 0.2195906334796154, "C_acceleration_rms_Q": 0.22856855716726407, "C_all_caps_pass_count": 4761, "C_crawl_C": 0.320290777528391, "C_crawl_E": 0.3316471809989615, "C_crawl_Q": 0.325550232823021, "C_speed_cv_C": 0.16133462865565643, "C_speed_cv_E": 0.20458276104653111, "C_speed_cv_Q": 0.13208937724029346, "C_stop_C": 0.19610733308766876, "C_stop_E": 0.30119593983451143, "C_stop_Q": 0.168872064587451, "M_acceleration_rms_C": 0.09205721751365113, "M_acceleration_rms_E": 0.08884124484941879, "M_acceleration_rms_Q": 0.09044923118153496, "M_all_caps_pass_count": 13973, "M_crawl_C": 0.13902381829754448, "M_crawl_E": 0.16247361897423873, "M_crawl_Q": 0.14672875280560116, "M_speed_cv_C": 0.047670094804194166, "M_speed_cv_E": 0.07721684365682892, "M_speed_cv_Q": 0.04844058825499983, "M_stop_C": 0.05956249371880339, "M_stop_E": 0.14411577501591236, "M_stop_Q": 0.06076848346789052}`.

The C/M/A dynamic all-12-dimension pass counts are joint non-compensatory results, not estimates of the marginal `pi` anchors.

Validation did not select thresholds, retune profiles, or emit an AV-serviceable-order set. Test31 remained untouched.
