## Provenance Model

Spindle provenance tracks where extracted graph facts come from and which source
evidence supports them.

## Purpose

- Link entities/relationships to source documents.
- Preserve evidence spans for traceability.
- Support audits and downstream trust scoring.

## Implementation

- Module: `spindle.provenance`
- Store: SQLite-backed `ProvenanceStore`
- Migration utility: `python -m spindle.provenance.migration`

## Typical linkage flow

1. Extraction produces triples with source metadata and supporting text.
2. Graph mutations can be paired with provenance writes.
3. Query and review workflows can traverse graph facts back to evidence.

## Data boundaries

- Graph structure lives in graph store.
- Provenance detail lives in provenance store.
- Both can be joined logically at the application layer.
