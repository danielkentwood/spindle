# Ingestion Templates

`spindle` ships with a small set of LangChain-driven ingestion templates you can
customise or extend. Templates define how documents are loaded, preprocessed,
split into chunks, enriched with metadata, and connected to the document graph.

## File format

Templates are stored as YAML, TOML, or JSON files. Each file can contain a
single template or a list under the `templates` key:

```yaml
name: contracts-pdf
description: Legal contract ingestion
selector:
  extensions: [".pdf"]
loader: langchain_community.document_loaders.PDFMinerLoader
preprocessors:
  - spindle.ingestion.templates.hooks.strip_empty_text
splitter:
  name: langchain_text_splitters.RecursiveCharacterTextSplitter
  params:
    chunk_size: 600
    chunk_overlap: 120
metadata_extractors:
  - mypackage.extractors.extract_parties
graph_hooks:
  - mypackage.graph.attach_clause_nodes
```

### Selector matching

Templates apply to documents using any combination of:

- `mime`: list of MIME types.
- `extensions`: file extensions such as `.pdf` or `.txt`.
- `glob`: glob patterns resolved against the absolute path.

When multiple templates match, the most specific selector wins.

## Callables

Template fields accept either dotted-path strings (e.g. `package.module.func`) or
direct callables. The following stages are available:

- `loader`: LangChain document loader class or callable returning `Document` objects.
- `preprocessors`: callables transforming `list[Document]`.
- `splitter`: either a dotted path to a splitter class or a mapping with `name`
  and `params` for instantiation.
- `metadata_extractors`: callables producing additional metadata dictionaries.
- `postprocessors`: callables for final chunk-level adjustments.
- `graph_hooks`: callables receiving the document, chunks, and graph builder to
  attach relationships.

## Default templates

Two templates are bundled:

1. `default-text` for plain text and markdown files.
2. `default-pdf` for PDF files with simple preprocessing.

You can extend or replace them by placing new template files inside a directory
and pointing `spindle` to it using `template_search_paths` in your ingestion
configuration. When you generate a `config.py`, add template directories to
`SPINDLE_CONFIG.templates.search_paths` so every ingestion run (CLI or API) picks
them up automatically.

## Process DAG Extraction

The ingestion graph (`DocumentGraph`) intentionally focuses on documents and
their chunk relationships only. If you need to extract structured process DAGs
from text, use the high-level helper in `spindle.extractor` instead of the
ingestion pipeline:

```python
from spindle import extractor

result = extractor.extract_process_graph(
    text=procedure_text,
    process_hint="Customer onboarding workflow",
)

if result.graph:
    print("Found steps:", [step.step_id for step in result.graph.steps])
    print("Dependencies:", [(d.from_step, d.to_step) for d in result.graph.dependencies])
```

The helper wraps the new BAML template (`ExtractProcessGraph`) and performs
post-processing such as evidence span indexing, DAG validation, and merging with
existing graphs. This keeps ingestion focused on document topology while still
enabling process reasoning when needed.

## Hot reloading

The template registry watches for changes every time you invoke an ingestion
run. Changes to template files take effect immediately without restarting the
Python process.

