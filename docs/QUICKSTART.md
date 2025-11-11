# Spindle Quick Start Guide

Spin up Spindle, extract your first triples, and explore the tooling in a few minutes.

## Prerequisites

- Python 3.11 (managed automatically by `uv` via `.python-version`)
- [uv](https://github.com/astral-sh/uv) installed and on your `PATH`
- Anthropic API key (required) plus any optional embedding API keys you plan to use

## 1. Bootstrap the Environment (≈1 min)

```bash
cd /Users/thalamus/Repos/spindle

# Create the virtual environment defined in .python-version
uv venv

# Install project + dev dependencies in editable mode
uv pip install -e ".[dev]"
```

> Need uv? Install it with `curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`, then restart your shell.

## 2. Configure API Keys (≈1 min)

Create a `.env` file in the repository root so `dotenv` and the demos can load credentials automatically:

```bash
cat <<'EOF' > .env
# Required for extraction and ontology workflows
ANTHROPIC_API_KEY=sk-ant-...

# Optional: enable remote embedding providers when using VectorStore utilities
OPENAI_API_KEY=sk-openai-...
HF_API_KEY=hf_...
GEMINI_API_KEY=AIza...
EOF
```

Keep the file out of version control (already covered by `.gitignore`).

## 3. Generate Unified Storage Config (≈1 min)

Spindle persists its catalog, graph database, vector store, logs, and template
artifacts relative to a single storage root. Scaffold the default layout with:

```bash
uv run spindle-ingest config init
```

This writes `config.py` in the current directory and creates the storage tree
under `./spindle_storage/`. Customize the root or destination at any time:

```bash
uv run spindle-ingest config init my-config.py --root ~/projects/spindle_data --force
```

The `config.py` exports `SPINDLE_CONFIG`, which you can load with
`from spindle.configuration import load_config_from_file`. See
`docs/CONFIGURATION.md` for the full schema.

## 4. Run a Built-in Demo (≈2 min)

```bash
# Automatic ontology recommendation + extraction on the first call
uv run python demos/example_auto_ontology.py
```

What you get:
- Detects the ontology scope automatically (`minimal`/`balanced`/`comprehensive`)
- Reuses the recommended ontology on subsequent extracts
- Shows entity consistency, supporting evidence, and reasoning strings

Prefer to start from a hand-authored ontology? Try the classic example instead:

```bash
uv run python demos/example.py
```

Additional demos:
- `uv run python demos/example_scope_comparison.py` — compare scope levels side-by-side
- `uv run python demos/example_ontology_extension.py` — conservative ontology extension flow

## 5. Create Your Own Script (≈3 min)

```python
from spindle import SpindleExtractor, create_ontology

# 1. Describe your domain
ontology = create_ontology(
    entity_types=[
        {
            "name": "Person",
            "description": "A human being",
            "attributes": [
                {"name": "title", "type": "string", "description": "Job title"}
            ],
        },
        {
            "name": "Company",
            "description": "Business organization",
            "attributes": [
                {"name": "founded_year", "type": "int", "description": "Year founded"},
            ],
        },
    ],
    relation_types=[
        {
            "name": "founded",
            "description": "Created or established",
            "domain": "Person",
            "range": "Company",
        }
    ],
)

# 2. Instantiate the extractor (auto-loads Anthropic credentials from .env)
extractor = SpindleExtractor(ontology)

# 3. Extract triples with provenance
result = extractor.extract(
    text="Elon Musk founded SpaceX in 2002.",
    source_name="SpaceX Wikipedia",
    source_url="https://en.wikipedia.org/wiki/SpaceX",
)

print(f"Found {len(result.triples)} triples")
for triple in result.triples:
    print(f"• {triple.subject.name} ({triple.subject.type})"
          f" --[{triple.predicate}]--> {triple.object.name} ({triple.object.type})")
    print(f"  Evidence: {triple.supporting_spans[0].text if triple.supporting_spans else 'n/a'}")
    print(f"  Source: {triple.source.source_name}")
print(f"Reasoning: {result.reasoning}")
```

Run it with:

```bash
uv run python my_example.py
```

### Keep Building

- Pass `existing_triples=result.triples` on subsequent extracts to maintain entity consistency across sources.
- Use helpers such as `filter_triples_by_source`, `parse_extraction_datetime`, and `triples_to_dict` (all exported from `spindle`).

## 6. Explore Graph Persistence & Embeddings

- Read `docs/GRAPH_STORE.md` for storing triples in the embedded Kùzu database (`GraphStore`).
- Check `spindle/vector_store.py` and the `embeddings`/`embeddings-api` extras for semantic search workflows.

## Troubleshooting

- **`ANTHROPIC_API_KEY not set`** → Confirm `.env` exists and rerun the command with `uv run`.
- **`ImportError: No module named 'kuzu'`** → Install extras if you trimmed dependencies: `uv pip install kuzu>=0.7.0`.
- **`ValueError: OpenAI API key required`** → Provide `OPENAI_API_KEY` (or pass `api_key=`) before using OpenAI embeddings.
- **Slow first request** → The first LLM call is cold; subsequent requests reuse warm connections.

## Next Steps

- Review `docs/ONTOLOGY_RECOMMENDER.md` for deeper control over recommendation + extension flows.
- Consult `docs/TESTING.md` for the full testing strategy and `docs/TESTING_QUICK_REF.md` for everyday commands.
- Pair this guide with `docs/UV_SETUP.md` if you need more detail on the uv workflow.

Happy knowledge graph building! 🚀

