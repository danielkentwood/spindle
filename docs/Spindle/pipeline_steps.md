## Pipeline Steps

Spindle is organized as composable stages. The default pipeline factory is
`spindle.get_pipeline_definition(...)`, which returns `list[StageDef]`.

## Default stage flow

1. **Preprocessing**
   - Module: `spindle.stages.preprocessing`
   - Runtime: `SpindlePreprocessor` (`spindle.preprocessing`)
   - Output: chunked document units with metadata (and optional coref updates).

2. **KOS Extraction** (optional via `include_kos=True`)
   - Module: `spindle.stages.kos_extraction`
   - Runtime dependency: `KOSService`
   - Output: updated KOS artifacts and extraction metrics.

3. **Ontology Synthesis** (when KOS is enabled)
   - Module: `spindle.stages.ontology_synthesis`
   - Output: ontology and SHACL artifacts plus graph payload.

4. **Retrieval** (when KOS is enabled)
   - Module: `spindle.stages.retrieval`
   - Inputs: ontology synthesis output + optional graph/vector stores
   - Output: contexts consumed by generation.

5. **Generation** (optional via `include_generation=True`)
   - Module: `spindle.stages.generation`
   - Runtime: `SpindleExtractor`
   - Output: extracted triples.

## Optional post-processing stage

- **Entity Resolution**
  - Module: `spindle.stages.entity_resolution`
  - Runtime: `EntityResolver`
  - Not included in `get_pipeline_definition(...)` by default; run explicitly when needed.

## API-level operational flow

For REST usage (`spindle.api`), common end-to-end sequence is:

1. Create session (`POST /api/sessions`) or use stateless endpoints.
2. Extract triples (`/api/extraction/...`).
3. Persist/query via graph store workflows.
4. Run deduplication (`/api/resolution/...`) if needed.
5. Use `/kos/...` endpoints for KOS search/CRUD/validation operations.
