# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spindle is an LLM-powered ontology-first knowledge graph extraction toolkit. It extracts structured knowledge (triples) from unstructured text using BAML-defined LLM prompts, stores them in a graph database (Kùzu), and provides entity resolution, vector search, and observability.

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

### Core Data Flow
```
Documents → SpindleExtractor (ontology required, BAML) → Triples → EntityResolver → GraphStore/VectorStore
```

### Key Components

- **`spindle/extraction/`**: `SpindleExtractor` (main extraction engine, ontology required), helpers, utils
- **`spindle/graph_store/`**: `GraphStore` facade over Kùzu backend for triple persistence and querying
- **`spindle/entity_resolution/`**: `EntityResolver`, `SemanticBlocker`, `SemanticMatcher` for LLM-based entity deduplication
- **`spindle/vector_store/`**: `ChromaVectorStore` with embedding factories (OpenAI, HuggingFace, Google, local)
- **`spindle/api/`**: FastAPI REST endpoints (extraction, resolution)
- **`spindle/observability/`**: `ServiceEvent` and `EventRecorder` for structured logging (SQLite persistence)

### BAML Files (LLM Prompts)

Located in `spindle/baml_src/`:
- `spindle.baml` - Core extraction schemas (Entity, Triple, Ontology, ExtractionResult, ExtractTriples)
- `entity_resolution.baml` - Entity matching functions
- `clients.baml` - LLM client configuration

**Important**: `spindle/baml_client/` is auto-generated. Edit `.baml` files, then run `cd spindle && uv run baml-cli generate`.

### Configuration

- `SpindleConfig` in `configuration.py` is the unified config system
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
- `SpindleExtractor`
- `GraphStore`, `ChromaVectorStore`
- `EntityResolver`, `ResolutionConfig`
- Factory functions: `create_ontology()`, `create_source_metadata()`
- Utilities: `triples_to_dict()`, `filter_triples_by_source()`
