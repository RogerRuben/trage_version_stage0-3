# AV depot initialization method

AV depots are generated from pre-test-day historical demand, not from the 2016-10-23 future demand stream.

Current configuration:

- training dates: 20161019–20161022
- depot method: training-origin KMeans with medoid depot locations
- depot count: 8
- AV/HV ratio: 0.05
- AV count: 915

The AV ratio is capped by configuration: `0 <= av_ratio_to_hv <= 0.05`.  AVs start from their assigned depot and remain in the field after completing service.  Relocation, charging, and automatic depot return are not modeled in this stage.

