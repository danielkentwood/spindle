# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spindle is an LLM-powered, ontology-first knowledge graph extraction toolkit. It processes unstructured documents through a multi-stage pipeline: preprocessing (Docling/Chonkie/fastcoref), KOS extraction (pyoxigraph + NER cascade), ontology synthesis, LLM-powered triple extraction (BAML), entity resolution, and graph storage (Kùzu). Stages are composable and integrate with spindle-eval via Hydra configuration.

## Essential Commands

**All Python commands must use `uv`:**

```bash
# Setup
uv venv && uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/ -m "not integration"        # Unit tests (no API)
uv run pytest tests/ -m integration              # Integration tests (needs API key)
uv run pytest tests/test_extractor.py -v         # Single file
uv run pytest tests/test_helpers.py::TestFindSpanIndices::test_exact_match  # Single test

# Regenerate BAML client after editing .baml files
cd spindle && uv run baml-cli generate

# CLI tools
uv run spindle-api
```

## Architecture

### Pipeline Stages
```
Documents → PreprocessingStage (Docling → Chonkie → fastcoref)
         → KOSExtractionStage (cold-start LLM | incremental 3-pass NER)
         → OntologySynthesisStage (KOS → ontology + SHACL)
         → RetrievalStage (KOS + GraphStore + ChromaDB)
         → GenerationStage (SpindleExtractor → triples)
         → EntityResolutionStage (semantic deduplication)
```

### Key Components

- **`spindle/preprocessing/`**: `SpindlePreprocessor` — Docling document conversion, Chonkie chunking, fastcoref coreference resolution
- **`spindle/kos/`**: `KOSService` — in-process SKOS/OWL/SHACL runtime (pyoxigraph), Aho-Corasick NER, ANN search, SPARQL; `KOSExtractionPipeline` for cold-start and incremental extraction
- **`spindle/stages/`**: Pipeline stage wrappers (`PreprocessingStage`, `KOSExtractionStage`, `OntologySynthesisStage`, `RetrievalStage`, `GenerationStage`, `EntityResolutionStage`) implementing the spindle-eval stage protocol
- **`spindle/extraction/`**: `SpindleExtractor` (main extraction engine, ontology required), helpers, utils
- **`spindle/graph_store/`**: `GraphStore` facade over Kùzu backend for triple persistence and querying
- **`spindle/entity_resolution/`**: `EntityResolver`, `SemanticBlocker`, `SemanticMatcher` for LLM-based entity deduplication
- **`spindle/vector_store/`**: `ChromaVectorStore` with embedding factories (OpenAI, HuggingFace, Google, local)
- **`spindle/provenance/`**: `ProvenanceStore` — SQLite provenance tracking (objects ↔ documents ↔ evidence spans)
- **`spindle/observability/`**: `ServiceEvent` and `EventRecorder` for structured logging (SQLite persistence)
- **`spindle/api/`**: FastAPI REST endpoints (extraction, resolution)
- **`spindle/eval_bridge.py`**: `get_pipeline_definition()` factory for spindle-eval integration; `PipelineDefinition`, `StageDef`
- **`spindle/conf/`**: Hydra YAML config groups (preprocessing, kos_extraction, ontology_synthesis, retrieval, generation)

### BAML Files (LLM Prompts)

Located in `spindle/baml_src/`:
- `spindle.baml` - Core extraction schemas (Entity, Triple, Ontology, ExtractionResult, ExtractTriples)
- `entity_resolution.baml` - Entity matching functions
- `clients.baml` - LLM client configuration

**Important**: `spindle/baml_client/` is auto-generated. Edit `.baml` files, then run `cd spindle && uv run baml-cli generate`.

### Configuration

- Hydra YAML configs under `spindle/conf/` for each pipeline stage
- `SpindleConfig` in `configuration.py` for programmatic config
- Load programmatically: `from spindle.configuration import load_config_from_file`

## BAML Patterns

When editing `.baml` files:
- Include `{{ ctx.output_format }}` in prompts (renders schema instructions)
- Use `{{ _.role("user") }}` to mark user inputs
- Don't repeat output schema fields in prompt text
- Use "high"/"medium"/"low" for confidence, not numbers
- Run `cd spindle && uv run baml-cli generate` after changes

## Testing

- Unit tests mock LLM calls and run fast
- Integration tests (marked `@pytest.mark.integration`) require `ANTHROPIC_API_KEY`
- Coverage target: ≥80% for core modules
- Fixtures in `tests/conftest.py`: `simple_ontology`, `sample_triples`, `mock_extraction_result`

## Environment Variables

Required:
- `ANTHROPIC_API_KEY` - Anthropic Claude API

Optional:
- `OPENAI_API_KEY` - OpenAI embeddings
- `HF_API_KEY` - HuggingFace embeddings
- `GEMINI_API_KEY` - Google embeddings
- Langfuse keys for observability (see `.env.example`)

## Key Types

From `spindle/baml_client/types.py` (auto-generated from BAML):
- `Entity`, `Triple`, `Ontology`
- `EntityType`, `RelationType`, `AttributeDefinition`
- `ExtractionResult`

## Public API

Main exports from `spindle/__init__.py`:
- `SpindleExtractor`, `GraphStore`, `ChromaVectorStore`
- `EntityResolver`, `ResolutionConfig`
- `KOSService`
- `get_pipeline_definition`, `PipelineDefinition`, `StageDef`
- Factory functions: `create_ontology()`, `create_source_metadata()`
- Utilities: `triples_to_dict()`, `filter_triples_by_source()`
