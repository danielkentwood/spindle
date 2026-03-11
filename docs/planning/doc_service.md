# Corpus: Document Ingestion & Preprocessing

This document specifies the design for Spindle's document ingestion and preprocessing pipeline — everything that happens before KOS extraction or KG extraction. The pipeline has three stages: **Ingestion & Version Control**, **Semantic Chunking**, and **Coreference Resolution**.

The output of this pipeline is a list of enriched chunks ready for downstream KOS extraction and KG extraction. Each chunk carries the raw text plus structured metadata annotations (coreference resolutions) that downstream stages can consume.

## Architecture Overview

```
                         spindle-eval boundary
                    ┌────────────────────────────────┐
                    │  SpindlePreprocessor.__call__() │
                    │  (cfg) → list[Chunk]            │
                    │                                 │
  ┌──────────────┐  │  ┌──────────────────────────┐   │
  │ Source docs   │──┼─▶│ Stage 1: Ingestion       │   │
  │ (files, URLs) │  │  │ Docling → JSON → Catalog │   │
  └──────────────┘  │  └───────────┬──────────────┘   │
                    │              ▼                   │
                    │  ┌──────────────────────────┐   │
                    │  │ Stage 2: Chunking         │   │
                    │  │ Chonkie pipeline          │   │
                    │  └───────────┬──────────────┘   │
                    │              ▼                   │
                    │  ┌──────────────────────────┐   │
                    │  │ Stage 3: Coref Resolution │   │
                    │  │ (fastcoref)                │   │
                    │  └───────────┬──────────────┘   │
                    │              ▼                   │
                    │        list[Chunk]               │
                    └────────────────────────────────┘
```

All three stages are orchestrated inside a single `SpindlePreprocessor` class that satisfies spindle-eval's `Preprocessor` protocol: `__call__(self, cfg) -> list[Chunk]`.

---

## Stage 1: Document Ingestion & Version Control

### Purpose

Accept documents from an external source, detect changes since the last run, convert changed documents into a structured JSON representation, and register metadata in a catalog.

### Steps

| Step | Description | Tool | Input | Output |
|------|-------------|------|-------|--------|
| 1a | **Mirror corpus** | File sync / configurable backend | External source (local dir, GCS, S3) | Local mirror of original documents |
| 1b | **Detect changes** | `deepdiff` content hash comparison | Current mirror vs. previous content hashes | List of changed/new/deleted document IDs |
| 1c | **Convert to structured JSON** | [Docling](https://github.com/docling-project/docling) | Changed source documents | Versioned structured JSON per document |
| 1d | **Register metadata** | `DocumentCatalog` (SQLite) | Docling output + file metadata | Document records with content hash, extraction timestamp, version |

### Design Details

**Docling** is the primary document conversion engine. It produces structured JSON that preserves document hierarchy — section headings, nesting, tables, lists, page boundaries. This structural information feeds directly into semantic chunking (Stage 2). Docling handles PDFs, DOCX, PPTX, HTML, Markdown, and other formats natively.

**Diff detection** uses content hashes (SHA-256 of raw bytes) stored in the catalog. On each run:
- New documents (no existing hash) are processed in full.
- Changed documents (hash mismatch) are re-converted and re-chunked.
- Unchanged documents are skipped entirely.
- Deleted documents (present in catalog but absent from mirror) are flagged for downstream handling (soft-delete in the KG).

This is the mechanism that enables the incremental processing described in the KG maintenance design (`kg_maintenance_updating.md`).

**Document catalog** tracks per-document state:

```python
@dataclass
class DocumentRecord:
    doc_id: str               # Stable identifier
    source_path: str          # Path in the mirror
    content_hash: str         # SHA-256 of raw bytes
    docling_json_path: str    # Path to versioned JSON output
    last_ingested: datetime   # Timestamp of last successful ingestion
    version: int              # Monotonically increasing version counter
    metadata: dict[str, Any]  # Title, author, date, source URL, etc.
```

This record also serves spindle-eval's staleness monitoring hook (`DocumentExtractionState`), which needs `doc_id`, `content_hash`, and `last_ingested` to compute staleness.

**Fallback loaders**: For file types that Docling handles poorly, maintain a registry of fallback converters (plain text passthrough, custom parsers). The template system from the current ingestion pipeline can be adapted to select between Docling pipelines and fallback paths based on file type.

### Output

A list of `DocumentRecord` objects plus their corresponding Docling JSON files on disk.

---

## Stage 2: Semantic Chunking

### Purpose

Split each document's structured content into chunks that are semantically coherent, sentence-boundary-aligned, and carry positional metadata for provenance tracking.

### Tool: Chonkie

We use [Chonkie](https://github.com/chonkie-inc/chonkie) for chunking. Chonkie is a lightweight, fast chunking library purpose-built for RAG pipelines. Key advantages over the current LangChain `RecursiveCharacterTextSplitter` approach:

- **Semantic chunking**: `SemanticChunker` groups text by embedding similarity, keeping topically related content together.
- **Recursive chunking with recipes**: `RecursiveChunker` supports a `markdown` recipe that respects document structure (headings, lists, code blocks).
- **Pipeline composability**: Chonkie's `Pipeline` API chains fetching, processing, chunking, and refinement in a single declarative flow.
- **Overlap refinement**: `OverlapRefinery` adds configurable context overlap between adjacent chunks without duplicating full chunk content.
- **Positional metadata**: Every chunk carries `start_index` and `end_index` back to the original text, making the chunking invertible — essential for provenance.
- **Lightweight**: ~500KB wheel, ~49MB installed. No bloat.

### Chunking Strategy

The default strategy uses Chonkie's recursive chunker with the character-based tokenizer:

1. **Recursive chunking** with the `markdown` recipe to respect document structure (section boundaries, paragraph breaks).
2. **Overlap refinement** to add configurable context overlap between adjacent chunks.

```python
from chonkie import Pipeline

doc = (
    Pipeline()
    .chunk_with("recursive", chunk_size=cfg_chunk_size, recipe="markdown", tokenizer="character")
    .refine_with("overlap", context_size=cfg_overlap, method="suffix")
    .run(texts=docling_text)
)
```

The `chunk_size` and `overlap` (expressed as fractional `context_size` in Chonkie's `OverlapRefinery`) are read from the Hydra config (`cfg.chunk_size`, `cfg.overlap`) to support spindle-eval's chunk size sweep. The character-based tokenizer matches spindle-eval's default (character count).

### Chunk Data Model

Each Chonkie chunk is mapped to spindle-eval's `Chunk` dataclass:

```python
Chunk(
    text=chonkie_chunk.text,
    source_id=doc_record.doc_id,
    metadata={
        "chunk_index": i,
        "start_index": chonkie_chunk.start_index,
        "end_index": chonkie_chunk.end_index,
        "token_count": chonkie_chunk.token_count,
        "section_path": section_path,        # from Docling JSON hierarchy
        "doc_content_hash": doc_record.content_hash,
        "doc_version": doc_record.version,
    },
)
```

Important properties:
- **Invertibility**: `start_index` / `end_index` allow reconstructing the exact source text span from the Docling output.
- **Sentence-boundary alignment**: Chonkie's recursive chunker with the markdown recipe splits at structural boundaries (headings, paragraphs), satisfying spindle-eval's `chunk_boundary_coherence` metric (fraction of chunks ending with sentence-terminal punctuation).
- **Section context**: `section_path` (derived from Docling's heading hierarchy) gives downstream extractors structural context without re-parsing the document.

### Configuration

| Parameter | Source | Default | Notes |
|-----------|--------|---------|-------|
| `chunk_size` | `cfg.chunk_size` | 600 | Target characters per chunk (character-based tokenizer). Swept by spindle-eval. |
| `overlap` | `cfg.overlap` | 0.10 | Fractional overlap between chunks. Swept by spindle-eval. Maps to Chonkie's `context_size` parameter. |
| `strategy` | `cfg.strategy` | `"recursive"` | Chunking strategy. Options: `"recursive"` (default pipeline above). |

### Output

A list of `Chunk` objects with positional and structural metadata, but not yet enriched with coref annotations (that happens in Stage 3).

---

## Stage 3: Coreference Resolution

### Purpose

Annotate chunks with coreference resolutions before they reach the KOS and KG extraction pipelines. Resolved references improve extraction quality by giving downstream stages (entity linking, NER) consistent entity mentions to work with.

All annotations are stored in `Chunk.metadata`, preserving the spindle-eval `Chunk` contract without modification. **The chunk text itself is never modified** — it always contains the original document text. Downstream consumers that need coref-resolved text (e.g., the Aho-Corasick NER pass in [kos_extraction.md](kos_extraction.md)) construct it at runtime from the annotations. See the [Coref-Resolved Text Reconstruction](#coref-resolved-text-reconstruction) section in kos_extraction.md for the reconstruction algorithm.

### Resolution Scope: Per-Document

Coreference resolution runs on the **full document text** (the complete Docling output), not per-chunk. This is critical because coreference chains frequently span chunk boundaries — an entity introduced by name in paragraph 1 may be referenced by pronoun in paragraph 5, which could be in a different chunk. Per-chunk resolution would miss these cross-chunk references entirely.

**Process**:
1. Concatenate the full Docling text output for the document.
2. Run fastcoref on the full document text to produce coreference chains.
3. For each chain, select the most informative mention (typically the longest named mention) as the representative.
4. Project chain memberships onto chunks: for each non-representative mention, determine which chunk it falls in using the chunk's `start_index` / `end_index`, and add the annotation to that chunk's `coref_annotations` with chunk-relative offsets.

This means all chunks for a given document share the same underlying coref chain assignments, and each chunk's annotations include only the mentions that fall within its text span.

### Coreference Resolution (fastcoref)

**Goal**: Resolve pronouns and nominal references ("the company", "he", "this agreement") to their antecedent named entities within each document.

**Approach**: Use [fastcoref](https://github.com/shon-otmazgin/fastcoref) for within-document coreference. fastcoref clusters mentions into coreference chains, selects the most informative mention as the representative, and tags each non-representative mention with its resolved representative.

```python
# Added to chunk.metadata["coref_annotations"]
[
    {
        "mention": "he",
        "span_start": 142,        # offset within chunk text
        "span_end": 144,
        "resolved_to": "John Smith",
        "chain_id": "chain_0",    # groups mentions referring to the same entity
        "confidence": 0.92,
    },
    ...
]
```

**`chain_id`** groups all mentions (across all chunks of a document) that refer to the same real-world entity. This enables downstream consumers to:
- Identify all chunks that mention a given entity (by querying for a shared `chain_id` across chunks with the same `source_id`).
- Distinguish between two different entities that happen to share a pronoun (e.g., two different "he" references in the same chunk that belong to different chains).
- Build cross-chunk entity mention graphs for entity resolution.

**Model selection**: fastcoref offers two models:
- **FCoref** (default): Fast (~230 tokens/sec). Recommended for production.
- **LingMessCoref**: More accurate (~45 tokens/sec). Use via config toggle when accuracy is prioritized over speed.

### Blacklist Filtering

The blacklist is defined in `kos/blacklist.txt` (see [kos_data_model.md](kos_data_model.md) section 6). During coreference resolution, mentions that resolve to blacklisted terms are excluded from `coref_annotations`. The blacklist is loaded by `SpindlePreprocessor` at init time and passed to the coref stage.

---

## Implementation: SpindlePreprocessor

The `SpindlePreprocessor` class orchestrates all three stages behind a single callable interface.

```python
from spindle_eval.protocols import Chunk

class SpindlePreprocessor:
    """Orchestrates ingestion → chunking → coref resolution.

    Satisfies spindle-eval's Preprocessor protocol:
        __call__(cfg) -> list[Chunk]
    """

    def __init__(self, documents: list[str] | None = None, tracker=None):
        self._documents = documents
        self._tracker = tracker or NoOpTracker()

    def __call__(self, cfg) -> list[Chunk]:
        doc_records = self._ingest(cfg)
        chunks_by_doc = self._chunk(doc_records, cfg)         # dict[doc_id, list[Chunk]]
        chunks_by_doc = self._resolve_coreferences_per_doc(chunks_by_doc, cfg)
        chunks = [c for doc_chunks in chunks_by_doc.values() for c in doc_chunks]
        chunks = self._filter_blacklist(chunks)

        self._tracker.log_metrics({
            "num_documents": len(doc_records),
            "num_chunks": len(chunks),
        })

        return chunks
```

### Incremental Processing

On subsequent runs (not the initial ingestion), the preprocessor re-processes only changed documents:

1. Stage 1 identifies changed docs via hash comparison.
2. Changed docs go through Stage 2 (Docling conversion, Chonkie chunking). Unchanged docs are skipped entirely.
3. Changed docs go through Stage 3 (coref resolution on full document text, projected onto chunks). Because coref runs per-document and chains span chunk boundaries, all chunks of a changed document are re-annotated — even if only one paragraph changed, the coreference chains may have shifted.
4. The full chunk list (cached unchanged docs + freshly processed changed docs) is returned.

The chunk cache is keyed by `(doc_id, doc_content_hash)` — all chunks for a document are invalidated together when the document changes.

---

## Dependencies

| Dependency | Purpose | Install |
|------------|---------|---------|
| `docling` | Document conversion to structured JSON | `uv pip install docling` |
| `chonkie` | Chunking (recursive, character-based) | `uv pip install chonkie` |
| `deepdiff` | Content diff detection | `uv pip install deepdiff` |
| `fastcoref` | Coreference resolution | `uv pip install fastcoref` |

Core dependencies (`docling`, `chonkie`, `deepdiff`, `fastcoref`) should be in spindle's main dependencies.

---

## Open Questions

1. **Section path extraction**: Docling's JSON schema for representing document hierarchy needs to be mapped to a `section_path` list. This mapping depends on the specific Docling output format and may need per-document-type adapters.

2. **Chunk caching strategy**: For incremental processing, we need to decide on the caching backend (filesystem, SQLite, or in-memory) and invalidation strategy. The cache is keyed by `(doc_id, doc_content_hash)` — all chunks for a document are invalidated together when the document changes, since coref runs per-document.