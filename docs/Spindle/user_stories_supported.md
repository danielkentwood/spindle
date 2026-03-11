## User Stories Supported

This page describes the workflows currently supported by the package and API.

## Package user stories

- As a Python user, I can define an ontology and extract triples from text with `SpindleExtractor`.
- As a pipeline user, I can assemble staged processing via `get_pipeline_definition(...)`.
- As a storage user, I can persist and query KG data with `GraphStore`.
- As a retrieval user, I can use embedding-backed vector storage with `ChromaVectorStore`.
- As a data quality user, I can run entity deduplication with `EntityResolver`.
- As a KOS user, I can search/resolve/manage concepts through `KOSService`.
- As an audit user, I can track source evidence with provenance and observability components.

## API user stories

- As an API consumer, I can run stateless extraction by providing text and ontology.
- As an API consumer, I can create sessions and perform stateful extraction across multiple documents.
- As an API consumer, I can run stateless or session-based entity resolution.
- As an API consumer, I can perform KOS operations:
  - lexical and ANN search
  - multistep mention resolution
  - concept CRUD
  - validation and SPARQL queries

## Contributor user stories

- As a contributor, I can edit BAML source definitions in `spindle/baml_src` and regenerate typed clients.
- As a contributor, I can test isolated subsystems and integration paths with `uv run pytest`.
- As a contributor, I can configure storage/runtime behavior with `SpindleConfig` and Hydra groups.
