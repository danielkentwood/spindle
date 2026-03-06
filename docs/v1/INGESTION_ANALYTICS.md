# Ingestion Analytics Taxonomy

This document defines the baseline attributes that Spindle records while
ingesting a document. The goal is to surface corpus-wide statistics that inform
ontology recommendation, graph extraction, and future pipeline optimizations.

## Overview

- **Scope** – Each observation captures the processing of a single source
  document and any derived chunk or span level metrics gathered during
  ingestion.
- **Granularity** – Metrics are collected at document, structural summary,
  chunk-window, and semantic segment levels so downstream systems can reason
  about the right context size.
- **Observability integration** – All `ServiceEvent` records produced during the
  ingestion run are persisted alongside analytics payloads for time-aligned
  diagnostics.

## Attribute Families

### Document Metadata

- `document_id` – Unique identifier assigned during ingestion.
- `source_uri` – Canonical URI or file path of the ingested asset.
- `source_type` – Enumerated source origin (e.g., `file`, `url`, `api`).
- `content_type` – MIME-like classification (`text/plain`, `application/pdf`).
- `language` – ISO-639 language code when reliably detected.
- `ingested_at` – UTC timestamp when ingestion completed.
- `hash_signature` – Stable content hash for deduplication.

### Structural Metrics

- `token_count` – Total tokens measured with the active tokenizer.
- `character_count` – Unicode character length of the raw document.
- `page_count` – Page total if supplied by the loader (e.g., PDF).
- `section_count` – Number of top-level semantic or structural sections.
- `average_tokens_per_section` – Mean token length across sections.
- `chunk_count` – Number of chunks emitted by the splitter.
- `chunk_token_summary` – Min/median/p95 token counts for produced chunks.

### Chunk Window Analytics

- `sliding_window_size` – Count of contiguous chunks grouped for analysis.
- `window_overlap` – Token overlap between adjacent windows (absolute & %).
- `window_token_summary` – Summary statistics over window token totals.
- `cross_chunk_link_rate` – Ratio of extracted triples referencing multiple
  chunks within the same window.
- `context_limit_risk` – Categorical assessment (`low`, `medium`, `high`) of
  exceeding target model context when operating on the window.

### Semantic Stability

- `embedding_dispersion` – Aggregate distance (mean + variance) across chunk
  embeddings for the document.
- `topic_transition_score` – Normalized measure of topic drift between adjacent
  chunks or sections.
- `segment_boundaries` – Token indices marking inferred semantic segments.
- `segment_token_summary` – Min/median/p95 tokens per semantic segment.

### Downstream Readiness Signals

- `ontology_candidate_terms` – Top surfaced candidate concepts for ontology
  expansion.
- `coverage_estimate` – Estimated fraction of ontology classes present in the
  document.
- `graph_density_estimate` – Predicted node/edge counts obtainable from the
  document.
- `context_window_strategy` – Recommended processing unit (`chunk`,
  `window`, `segment`, `document`) for ontology/triple extraction.

### Observability Correlation

- `service_events` – All `ServiceEvent` payloads emitted during ingestion, keyed
  by service namespace.
- `error_signals` – Structured summary of failures or warnings encountered.
- `latency_breakdown` – Per-stage stopwatch metrics aligned to service events.

## Usage Guidance

1. **Ingestion Pipeline** – Emit analytics observations once chunking and
   preliminary ontology recommendations complete. Attach any scoped
   `ServiceEvent` emissions for visibility.
2. **Persistence Layer** – Store the observation in the shared analytics SQLite
   database so dashboards and offline jobs can query trends.
3. **Dashboard** – Visualize corpus distributions (tokens, chunk sizes, semantic
   drift) and correlating ServiceEvents (latency, failures) to tune processing
   strategies.
4. **Iteration** – Extend this taxonomy as new downstream needs arise, keeping
   backward compatibility by versioning analytics payloads through the
   accompanying Pydantic models.

