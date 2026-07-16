# Stage 3 canonical condition vector v2

This engineering-smoke vector consumes only held-out Stage 2 dispatch-time
predictions. Stage 1 realized labels are isolated targets and evaluation fields;
the feature list contains no realized label.

`*_expected` is a continuous regression expectation. `*_tail_probability` is a
binary probability calibrated using the validation day only. These fields are
not q90 aliases. The core overall probability is trained against the OR of the
three training-defined dimension-tail labels.

The smoke has no canonical dispatch-time IIS model. IIS availability is false
and all IIS values remain NA. Consequently, the extended overall probability is
also NA; it is not synthesized with `max(core, IIS)`. Stage 4 smoke consumes the
core vector only.

This artifact validates semantics, timing, and lineage. Its lightweight models
are not formal scientific replacements for the existing RC-MSTNet or DeepSets.
