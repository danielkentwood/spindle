## Package Structure

`spindle` is organized around extraction, runtime stages, and storage services.

## Top-level modules

- `spindle.__init__`: public package exports.
- `spindle.extraction`: `SpindleExtractor`, ontology helpers, triple utilities.
- `spindle.preprocessing`: ingestion, chunking, coreference preprocessing pipeline.
- `spindle.kos`: KOS runtime (`KOSService`) and extraction/synthesis support.
- `spindle.stages`: stage wrappers used by pipeline/eval orchestration.
- `spindle.eval_bridge`: `get_pipeline_definition(...)` factory.
- `spindle.configuration`: `SpindleConfig` and storage/runtime settings.
- `spindle.api`: FastAPI app and routers.
- `spindle.graph_store`: graph persistence and query facade.
- `spindle.vector_store`: Chroma-backed vector store + embedding factory helpers.
- `spindle.entity_resolution`: blocking, matching, and merge orchestration.
- `spindle.provenance`: provenance data model and SQLite store.
- `spindle.observability`: structured event model and optional persistence.
- `spindle.conf`: Hydra config groups shipped with the package.
- `spindle.baml_src`: source BAML schemas/prompts.
- `spindle.baml_client`: generated BAML client/types (do not edit directly).

## Runtime pipeline shape

Typical ordering:

1. Preprocessing
2. KOS extraction
3. Ontology synthesis
4. Retrieval
5. Generation
6. Entity resolution (optional post-processing stage)

`get_pipeline_definition(...)` currently returns stages 1-5 by default.
