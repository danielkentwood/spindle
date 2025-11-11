<p align="center">
  <img src="assets/spindle_logo.svg" alt="Spindle Logo" width="400">
</p>


# Spindle

LLM-powered ontology-first extraction of knowledge graph triples from text.

## What It Does

- Recommends ontologies (entities, relations, attributes) from raw text using BAML-driven LLM prompts
- Extracts triples with source metadata, evidence spans, and timestamps
- Keeps entities consistent across documents and merges duplicate facts
- Persists results in an embedded Kùzu-backed `GraphStore`
- Emits structured service events across ingestion, extraction, and storage with optional persistence (see `docs/OBSERVABILITY.md`)
- Ships with example workflows and a test suite covering API and persistence layers

## Quick Start

Prerequisites: Python 3.11+, Anthropic API key, and [`uv`](https://github.com/astral-sh/uv) (preferred installer).

```bash
git clone <repository-url>
cd spindle
uv venv
uv pip install -e ".[dev]"
cp .env.example .env  # add ANTHROPIC_API_KEY
```

See `docs/UV_SETUP.md` if you hit environment issues.

## Basic Usage

```python
from spindle import SpindleExtractor

extractor = SpindleExtractor()  # auto-recommends ontology on first extract
result = extractor.extract(
    text="Alice Johnson leads research at TechCorp.",
    source_name="Company Blog"
)

for triple in result.triples:
    print(triple.subject.name, triple.predicate, triple.object.name)
```

- Control ontology scope with `ontology_scope="minimal" | "balanced" | "comprehensive"`
- Use `OntologyRecommender` for explicit recommendations and conservative ontology extension
- Persist extractions with `GraphStore()`; demos live in `demos/`

### Unified Storage & Configuration

- Scaffold a runtime config (defines storage root, catalog, graph DB, vector store, logs) with `uv run spindle-ingest config init`.
- Edit the generated `config.py` to relocate storage or add template directories, then pass it to tooling via `--config /path/to/config.py`.
- Programmatic consumers can load the same file with `from spindle.configuration import load_config_from_file`.
- See `docs/CONFIGURATION.md` for the full schema and usage patterns.

## Project Layout

```
spindle/
├── spindle/          # Package code (extractor, GraphStore, BAML client)
├── demos/            # Example scripts covering core workflows
├── docs/             # Additional guides (graph store, observability, testing, uv setup)
├── tests/            # Unit + integration tests
└── README.md
```

## Testing

```bash
uv run pytest tests/ -m "not integration"
uv run pytest tests/ -m integration  # requires API access
```

Coverage reports and further details: `docs/TESTING.md`.

## Contributing & License

Contribution guidelines and license details are tracked in the `/docs` folder. PRs welcome.

