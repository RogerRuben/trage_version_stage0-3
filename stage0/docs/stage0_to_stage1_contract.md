# Stage 0 to Stage 1 Contract

Stage 1 input is partitioned by `split/date/bucket`. Only rows in `order_base`
with `stage1_core_eligible=true` are core orders. Core orders require
`route_pass`, eligible GPS, resolved canonical identity, usable dynamic status,
valid conservation audits, at least eight direct intervals, and at least five
unique timed edges.

Direct link supervision comes only from `link_interval_observations` rows where
`measurement_source == "direct_observed"` and `label_valid == true`.
`interval_measurements` retains all four provenance classes:
`direct_observed`, `interval_supported`, `engine_interpolated`, and
`unresolved`. Non-direct rows never carry observed time.

`link_traversals` is the physical route-access table: one continuous access per
row. `route_parts` is the directed canonical route sequence, and
`route_segments` carries segment-level GPS, route, dynamic, and canonical
statuses. Failed candidates are absent from core products and retained only in
the lightweight rejection manifests.

Stage 0 is frozen after verification. It may be reopened only for data
loss/duplication, conservation defects, large-scale systematic route mismatch,
or a Stage 1 schema-read failure.
