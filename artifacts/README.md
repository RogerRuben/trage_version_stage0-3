# Artifact registry

This directory stores small manifests and governance records, not large data.

- `canonical/`: the only artifacts allowed in formal downstream runs.
- `exploratory/`: reproducible research candidates, never formal inputs.
- `deprecated/`: retained lineage for superseded approaches.

Large files stay in ignored stage output roots. A manifest contains their relative
paths, sizes, hashes, row counts, config hash, audit result, and limitations.
Scripts must receive `--input-manifest`; directory scanning is forbidden.

