# Stage 3 S4 Methodology

S4 evaluates the exact frozen `20161031` historical route under the frozen hypothetical C/M/A capability scenarios. It performs no rerouting, fallback, nearest-geometry repair, profile retuning, CDF fitting, or Test31 calibration.

Atomic evidence is non-compensatory. A known incompatibility has precedence over unknown evidence, and both known and unknown cause vectors are retained. This is operational route compatibility, not safety, legal certification, failure probability, accident probability, disengagement probability, or ODD approval.

Dynamic inference uses frozen M3 checkpoint `965fc491cd77256f7889961d89932ec6be709bab04adcca358ac1b49f47c2cde` and the frozen Train weighted mid-CDF. Dynamic E/Q/C is calculated only when every original-route token has complete prediction-side evidence; tail membership uses strict `z > 0.9`.
