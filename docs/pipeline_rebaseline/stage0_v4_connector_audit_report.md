# Stage 0 v4 direction-aware connector audit

Status: code and ablation diagnostics complete; 50-connector human review pending.

## Direction correction

The earlier engineering diagnostic emitted every graph-only connector as
bidirectional. The current v4 implementation derives each connector direction from
the permitted entry/exit direction of its two incident road endpoints. The rebuilt
network contains 2,267 graph-only connectors: 652 bidirectional and 1,615
unidirectional. Connectors remain ineligible as GPS/HMM candidates.

## Usage on the fixed 2016-10-23 sample

- 984 orders reconstructed; 627 (63.72%) used at least one connector.
- 4,879 connector occurrences and 254 unique connector links were used.
- Mean/P95 connector occurrences among using orders were 7.78/19.
- All connector link lengths were zero because they connect co-located terminal
  nodes; 4,339 uses were internal and 540 were near an OD endpoint.
- The most frequently used connector appeared in 73 orders (7.42%); no single
  connector dominates the sample.

## Connector-off ablation

| Metric | Direction-aware connectors | Connectors disabled |
|---|---:|---:|
| Directed OD reachability | 99.80% | 71.75% |
| Reconstructed / input | 984 / 1,000 | 984 / 1,000 |
| Orders with direction gap | 169 | 607 |
| Mean direction gaps | 0.224 | 2.624 |
| Strict Core | 139 | 136 |
| Mean matched route length | 5,891 m | 5,514 m |

The equal reconstruction count is expected because the HMM can still emit a
disconnected sequence when connectors are absent; the route classifier then exposes
the missing graph transitions. The longer connector-enabled route length reflects
recovered between-link paths, not positive connector distance.

## Human gate

`manual_truth/stage0_connector_review_v1.csv` and its GeoJSON contain 50 actual used
connectors, stratified toward high-frequency and one-way cases. Until those judgments
are completed, the audit cannot claim that elevated/ground or bridge/tunnel
transitions are free of systematic semantic error. This is the remaining connector
promotion gate.
