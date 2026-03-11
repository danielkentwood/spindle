## Glossary

This glossary defines terms as used by the current Spindle codebase.

- **Spindle**: an ontology-first toolkit for extracting and maintaining knowledge graphs from unstructured documents.
- **Extractor (`SpindleExtractor`)**: the LLM-backed component that produces typed triples from text using an ontology.
- **Ontology**: the schema constraining valid entity and relation types during extraction.
- **KOS (Knowledge Organization System)**: the in-process SKOS/OWL/SHACL runtime in `spindle.kos`, exposed through `KOSService`.
- **KOS extraction**: stage that derives/updates KOS artifacts from processed chunks.
- **Ontology synthesis**: stage that transforms KOS artifacts into ontology and SHACL outputs.
- **Preprocessing**: document conversion, chunking, and optional coreference resolution (`spindle.preprocessing`).
- **Stage definition (`StageDef`)**: executable stage metadata object returned by `get_pipeline_definition(...)`.
- **Graph store (`GraphStore`)**: persistence and query facade for KG nodes/edges/triples (Kuzu backend by default).
- **Vector store (`ChromaVectorStore`)**: embedding-backed similarity storage used by retrieval and entity resolution.
- **Entity resolution**: deduplication pipeline (`EntityResolver`) that blocks, matches, and links duplicates with `SAME_AS`.
- **Provenance**: source/evidence tracking for graph facts (`spindle.provenance`).
- **Observability events**: structured runtime events emitted by services/stages and optionally persisted.
- **Session API**: REST mode where ontology/config/triples are retained across calls in a session context.
- **Stateless API**: REST mode where each request provides all required inputs.
- **BAML source (`spindle/baml_src`)**: editable prompt/schema definitions.
- **BAML client (`spindle/baml_client`)**: generated client/types; regenerate instead of editing manually.
