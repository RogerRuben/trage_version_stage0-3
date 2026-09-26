# Stage 3 S2B to S3 Contract

Status: `STAGE3_S2B_INTERSECTION_COMPLEX_FROZEN`. S3 remains unauthorized.

The selected **buffer radius is 10m**. Because candidate-node buffers overlap, the spatial candidate-overlap condition is center distance <= 20m; 10m is not a direct pairwise node-distance threshold. POI evidence remains excluded.

S3 may join static network identity (`stage3_edge_uid`, `intersection_complex_uid`), intersection descriptors (`external_physical_connection_count`, `topological_movement_count`, `internal_length_m`, `road_class_diversity`, `signal_state`, roundabout and grade-separation descriptors), and route-specific movement descriptors (incoming/outgoing edges, geometric turn type/angle, control state, and restriction uncertainty). Frozen speed-domain data joins by `stage3_edge_uid`.

The movement table describes **topological movements, not legally certified movements**. Missing control evidence is `UNKNOWN_CONTROL`, not unsignalized. Grade-separation fields are network-structure and anti-merge evidence, not automatic AV infeasibility. QA flags remain `QA_ONLY` and are not AV capability thresholds.

Historical reverse identities remain historical observations. They are not missing, are not inserted into the AV topology, and return `AV_ROUTABILITY_VIOLATION`. Of 6,502 overlays, 6,388 retain a mapped forward physical reference; no reference is fabricated for the remaining 114. S2B-2 does not propagate this into route F/U/I. The deterministic adapter accepts an ordered edge sequence plus the frozen movement, complex, boundary, reverse-overlay, and full-network identity sets; full-network edges with no complex transition return `NO_COMPLEX_TRANSITION`.

S3 must not interpret topological movement as legal movement, operational stress as AV safety probability, or missing control evidence as unsignalized.
