# Saga vs. Spindle: Comparative Analysis

## 1. Areas of Overlap

### 1.1 Data Model: Triple-Based Knowledge Representation

| | Saga | Spindle |
|---|---|---|
| **Core model** | RDF `<S, P, O>` triples, extended to capture one-hop relationships ("extended triples") for flat relational access | `Triple` objects (subject `Entity`, predicate, object `Entity`) stored as property graph in Kuzu |
| **Serialization** | JSON-LD variant optimized for industry-scale querying | Python dataclasses (BAML-generated), JSON serialization via utils |
| **Metadata per fact** | Source array, locale, trustworthiness score array | Source metadata (name, URL, type), supporting evidence spans, extraction datetime |

**Assessment**: Saga's "extended triples" avoid expensive self-joins for one-hop queries by materializing neighbor data inline -- an optimization Spindle lacks. However, Spindle's property-graph model in Kuzu is inherently more flexible for traversals. The key gap is that Spindle lacks **trustworthiness scores** and **locale** on facts.

**Recommendation**: **Partially adopt.** Add a `confidence` or `trust_score` field to triples/edges in Spindle. Locale may be relevant depending on use case. The extended-triple optimization is a performance concern Spindle doesn't need at its current scale.

---

### 1.2 Entity Resolution

| | Saga | Spindle |
|---|---|---|
| **Blocking** | Entity-type-specific blocking functions (domain-tuned) | `SemanticBlocker` -- embedding-based clustering (type-agnostic) |
| **Matching** | Domain-specific ML or rule-based models; neural string similarity with per-type encoders | `SemanticMatcher` -- LLM-based via BAML (`MatchEntities`, `MatchEdges`) |
| **Resolution** | Correlation clustering | Threshold-based clustering (`ResolutionConfig.clustering_method`) |
| **Output** | Merged canonical entities | `same_as` edges; canonical entity lookups via `get_canonical_entity` |

**Assessment**: Saga's approach is more mature and cost-efficient at scale -- entity-type-specific blockers reduce false candidates, and trained ML models are cheaper to run than LLM calls per pair. Spindle's LLM-based matching is more flexible (works across domains without per-type model training) but significantly more expensive and slower.

**Recommendation**: **Adopt selectively.** Spindle's LLM approach is a reasonable tradeoff for its scale, but it should add: (1) entity-type-aware blocking to reduce candidate pairs before LLM matching, (2) neural string similarity as a fast pre-filter before LLM calls, and (3) correlation clustering as an option alongside threshold-based methods. These changes would reduce LLM costs significantly.

---

### 1.3 Provenance & Trust

| | Saga | Spindle |
|---|---|---|
| **Provenance tracking** | Every fact carries source array | `ProvenanceStore` (SQLite) with objects -> docs -> evidence spans |
| **Granularity** | Source-level (which sources contributed) | Span-level (exact text evidence with character offsets and section paths) |
| **Trustworthiness** | Score array per source; truth discovery for correctness probabilities | None |
| **Compliance features** | License-filtered views, on-demand deletion propagation | `delete_provenance()` exists but no cascading to graph or downstream views |
| **Reverse lookup** | Not described | `get_affected_objects(doc_id)` -- all graph objects sourced from a document |

**Assessment**: Spindle actually has **finer-grained provenance** than Saga at the text level (exact character spans, section paths). But Saga has **richer metadata** on provenance (trust scores, locale) and **stronger compliance capabilities** (license-filtered views, deletion propagation).

**Recommendation**: **Adopt trust scoring and deletion propagation.** Spindle's span-level provenance is a genuine strength. Layer on: (1) a trust/confidence score per source in `ProvenanceDoc`, (2) cascading deletion -- when `delete_provenance` is called, propagate to GraphStore edges/nodes, and (3) a "source filter" query mode on GraphStore (partially present via `query_by_source` but not enforced as a view-level constraint).

---

### 1.4 Named Entity Recognition / Disambiguation

| | Saga | Spindle |
|---|---|---|
| **NER** | Full NERD stack: neural string similarity + transformer disambiguation | Aho-Corasick over KOS vocabulary terms |
| **Disambiguation** | Transformer-based classification against NERD Entity View (names, types, descriptions, relationships, importance) | Multi-step resolution: exact label match, then ANN semantic search |
| **Training** | Weak supervision (annotated text, curated logs, template snippets) | No training -- vocabulary-driven |
| **Object resolution** | Resolves string literals in object fields to KG entity IDs | Not implemented |

**Assessment**: Saga's NERD is a production-grade, ML-heavy system designed for disambiguation at scale. Spindle's Aho-Corasick + ANN approach is faster and simpler but fundamentally limited -- it can find known vocabulary terms but cannot disambiguate between entities with similar names or handle novel entity mentions.

**Recommendation**: **Adopt concept, not implementation.** Spindle's LLM-based extraction already handles entity identification in the generation stage, partially sidestepping the need for a separate NERD stack. However, Spindle should add: (1) **object resolution** -- when a triple's object is a string that matches a known entity, link it, and (2) **disambiguation signals** -- pass the KOS entity descriptions/types to the LLM during extraction (some of this is already done via ontology context, but explicit candidate entities would improve accuracy).

---

### 1.5 Graph Embeddings

| | Saga | Spindle |
|---|---|---|
| **Models** | TransE, DistMult, etc. via Marius system | Node2Vec-style graph walks |
| **Uses** | Fact ranking, fact verification (outlier detection), missing fact imputation | Node embeddings stored in Kuzu; used for entity resolution blocking |
| **Scale** | Billion-scale, multi-GPU training (~1 day) | Single-process, small-graph oriented |

**Assessment**: Saga uses KG embeddings as a quality tool (find wrong facts, impute missing ones). Spindle uses them only for entity resolution blocking.

**Recommendation**: **Adopt fact verification use case.** KG embedding-based outlier detection is a high-value, low-cost addition. After extraction, compute embeddings and flag triples that are statistically anomalous as candidates for review. This could be added as a post-generation quality check stage.

---

### 1.6 Ontology / Schema Management

| | Saga | Spindle |
|---|---|---|
| **Ontology source** | Manually defined via Predicate Generation Functions (config-driven) | LLM-powered cold-start extraction + incremental NER; SKOS/OWL/SHACL synthesis |
| **Alignment** | Manual per-source configuration files | Automated via KOS synthesis pipeline |
| **Validation** | Not described | SHACL constraint generation and validation |

**Assessment**: This is an area where **Spindle is ahead**. Saga requires manual ontology alignment per data source, while Spindle can synthesize ontologies from text and generate SHACL validation constraints. Saga's config-driven approach is more predictable but higher maintenance.

**Recommendation**: **Keep Spindle's approach.** This is a genuine differentiator. Consider adding config-based overrides for cases where automated synthesis produces suboptimal ontologies, giving users Saga-style control when needed.

---

### 1.7 Graph Storage & Querying

| | Saga | Spindle |
|---|---|---|
| **Architecture** | Federated polystore (analytics warehouse, entity index, text index, vector DB, compute cluster) | Kuzu (property graph) + ChromaDB (vector) + SQLite (provenance, catalog) |
| **Query language** | KGQ (custom, bounded-performance graph query language) | Raw Cypher via `query_cypher()`, plus pattern-based helpers |
| **Query orchestration** | Federated orchestration agent framework across stores | Manual -- caller chooses which store to query |

**Assessment**: Saga's polystore is designed for massive scale and diverse query patterns. Spindle's simpler stack is appropriate for its scale but lacks query-level orchestration across stores.

**Recommendation**: **Don't adopt polystore architecture.** It's overkill for Spindle's scale. However, **adopt the orchestration concept** -- add a unified query interface that can transparently combine GraphStore, VectorStore, and ProvenanceStore results. A lightweight query planner that routes to the right backend would be valuable as the system grows.

---

## 2. Areas Where Saga Has Functionality Spindle is Missing

### 2.1 Delta / Incremental Processing

**Saga**: Computes deltas against previous snapshots. Only Added/Deleted/Updated payloads are processed. Updated/Deleted payloads skip full linking and only require ID lookups. Volatile properties (popularity scores) have optimized partition-overwrite fusion.

**Spindle**: Has content-hash change detection in `DocumentCatalog` (preprocessing), but the extraction pipeline reprocesses entire chunks. No delta computation at the triple level.

**Recommendation**: **Adopt -- high priority.** This is one of Saga's most impactful design decisions. Spindle should: (1) track extracted triples per chunk (already partially possible via provenance), (2) on re-extraction, diff new triples against previous triples for that chunk, (3) only update/delete/add the changed triples in GraphStore. The `get_affected_objects(doc_id)` reverse lookup in ProvenanceStore is already the foundation for this.

---

### 2.2 Data Fusion / Truth Discovery

**Saga**: When multiple sources provide facts about the same entity, performs outer joins for simple facts and similarity-based merging for composite relationships. Uses truth discovery to estimate probability of correctness for consolidated facts.

**Spindle**: Has evidence merging (`merge_evidence` in `edges.py`) that deduplicates supporting evidence spans when the same edge comes from multiple sources, but no truth discovery or conflict resolution.

**Recommendation**: **Adopt -- medium priority.** As Spindle processes more sources about the same entities, conflicting facts will emerge. Add: (1) conflict detection (same subject+predicate, different objects), (2) a simple truth discovery heuristic (e.g., source reliability weights, recency, agreement count), (3) expose conflicts via a "fact conflicts" query API.

---

### 2.3 Parallel Pipeline Execution

**Saga**: Inter-source parallelism (sources run in parallel, synchronize only at fusion), intra-source parallelism (Add/Update/Delete payloads in parallel), intra-block parallelism.

**Spindle**: Sequential pipeline execution. `extract_batch` processes chunks sequentially (though `extract_batch_stream` provides async iteration). Stages in the eval bridge run sequentially.

**Recommendation**: **Adopt -- medium priority.** Source-level parallelism is the highest-value addition: process multiple documents concurrently through extraction, synchronize at entity resolution. Spindle's async extraction support (`extract_async`) provides the building blocks. The eval bridge pipeline stages could support concurrent execution where dependencies allow.

---

### 2.4 Knowledge Graph Views

**Saga**: A View Catalog with dependency tracking. Views can be sub-graphs, aggregates, PageRank computations, embeddings, or schematized relational tables. View Manager orchestrates the dependency graph. 26% runtime improvement from reusing common views.

**Spindle**: No concept of views. Consumers query the raw graph directly.

**Recommendation**: **Adopt concept -- low priority for now.** At Spindle's current scale, views aren't critical. But the concept of "registered derived artifacts that refresh when the underlying graph changes" is powerful. A lightweight implementation could be: (1) named queries saved in GraphStore, (2) a staleness flag updated when underlying data changes, (3) materialized results cached for repeated access.

---

### 2.5 Entity Importance Scoring

**Saga**: Aggregates in-degree, out-degree, number of source identities, and PageRank. Covers head, torso, and tail entities.

**Spindle**: `compute_graph_embeddings` does Node2Vec walks but no importance/centrality scoring.

**Recommendation**: **Adopt -- low effort, high value.** Kuzu can compute degree centrality trivially via Cypher. Adding an `importance_score` property to Entity nodes would enable better ranking in downstream applications. PageRank can be computed via the existing graph embedding infrastructure.

---

### 2.6 Live / Streaming Knowledge Graph

**Saga**: Union of stable KG with real-time streaming sources. Live sources bypass full linking/fusion; ambiguous references resolved via NERD against stable graph. Handles billions of queries/day at <20ms p95.

**Spindle**: No streaming capability. Batch-only processing.

**Recommendation**: **Do not adopt at this stage.** Saga's live KG is an enterprise-scale feature for sub-second freshness (sports scores, stock prices). This is far outside Spindle's current use case. If real-time needs emerge, a simpler approach (webhook-triggered re-extraction of specific documents) would suffice.

---

### 2.7 Human-in-the-Loop Curation

**Saga**: Detects potential errors/vandalism, quarantines facts for human review, curations stream back to both live index and stable KG.

**Spindle**: No curation or quality review workflow.

**Recommendation**: **Adopt concept -- medium priority.** A review queue for low-confidence triples is high value. Implementation: (1) flag triples below a confidence threshold during extraction, (2) store a "review_status" field on edges (pending/approved/rejected), (3) expose a review API endpoint. The BAML extraction already returns confidence levels that could drive this.

---

### 2.8 Self-Serve Data Source Onboarding

**Saga**: Pluggable adapters with standard interfaces (Importer, Transformer, PGFs). Templates for custom source ingestion. Config-driven ontology alignment.

**Spindle**: Preprocessing is through Docling (document conversion) + Chonkie (chunking) + fastcoref (coreference). No pluggable adapter framework for diverse source types.

**Recommendation**: **Partially adopt.** Spindle's Docling-based preprocessing already handles many document formats. But for non-document sources (databases, APIs, structured feeds), a pluggable adapter interface would be valuable. Define a `DataSource` protocol with `fetch() -> Iterator[RawRecord]` and `transform(record) -> Chunk` methods.

---

## 3. Global Considerations: Saga's Ecosystem and Lessons for Spindle

### 3.1 The Delta-First Architecture

Saga's most fundamental design principle is **"always operate on diffs, never reprocess."** This permeates every layer: ingestion computes deltas against snapshots, construction only processes changed payloads, views refresh incrementally, and even volatile properties get special-case optimized paths.

**Lesson for Spindle**: Spindle's current architecture is batch-oriented and stateless per run. The `DocumentCatalog` content-hash detection is a start, but the delta principle should extend to: chunk-level change tracking, triple-level diffing, and incremental GraphStore updates. The `ProvenanceStore.get_affected_objects()` method is a perfect foundation -- it already maps documents to graph objects, enabling targeted re-extraction.

### 3.2 The Separation of Construction and Serving

Saga cleanly separates **KG construction** (batch pipelines, entity resolution, fusion) from **KG serving** (Graph Engine, Live KG, query APIs). Construction is throughput-optimized; serving is latency-optimized. They communicate through the shared log.

**Lesson for Spindle**: Spindle currently conflates construction and querying in the same process. The `GraphStore` both writes triples and serves queries. As Spindle grows, consider separating the write path (extraction pipeline writing to Kuzu) from the read path (API layer querying Kuzu). This doesn't require a distributed architecture -- even process-level separation with a shared database would improve reliability.

### 3.3 The Quality Feedback Loop

Saga has a closed loop: extraction -> storage -> embedding -> quality signals (fact verification, importance scoring) -> curation -> back to storage. Embeddings aren't just for similarity search -- they're a **quality assurance tool** (outlier detection, missing fact imputation).

**Lesson for Spindle**: Spindle's pipeline is currently open-loop: extract -> store -> done. Adding a feedback mechanism would significantly improve quality. The simplest version: after extraction, compute graph embeddings, identify statistical outliers, flag them for review. This could be added as a post-generation stage in the eval bridge pipeline.

### 3.4 The Multi-Source Reconciliation Problem

Saga's entire architecture assumes multiple sources providing overlapping, conflicting data about the same entities. Every component is designed around this: blocking groups cross-source entities, matching finds duplicates, fusion resolves conflicts, provenance tracks which source said what, and trust scoring weighs them.

**Lesson for Spindle**: Spindle currently handles multi-source scenarios through evidence merging (appending evidence from multiple sources to the same edge). But it lacks **conflict resolution** -- when Source A says "CEO of X is Alice" and Source B says "CEO of X is Bob", Spindle stores both without flagging the conflict. Adding conflict detection and resolution logic would be the single highest-impact improvement for multi-source use cases.

### 3.5 What Spindle Has That Saga Doesn't

It's worth noting Spindle's advantages:

1. **LLM-powered extraction from unstructured text** -- Saga explicitly consumes pre-extracted structured/semi-structured sources. Spindle extracts triples from raw text via BAML, which is a capability Saga doesn't have (the paper predates the ChatGPT era).
2. **Automated ontology synthesis** -- Saga requires manual ontology configuration per source. Spindle can bootstrap ontologies from text.
3. **Span-level provenance** -- Saga tracks which source provided a fact; Spindle tracks the exact text span that evidences it, with character offsets.
4. **Coreference resolution** -- Spindle resolves pronouns/references before extraction, improving triple quality from narrative text.

These are genuine differentiators that should be preserved and emphasized.

### 3.6 Prioritized Recommendations Summary

| Priority | Recommendation | Effort | Impact |
|---|---|---|---|
| **High** | Delta/incremental processing at triple level | Medium | Major efficiency gain for re-extraction |
| **High** | Trust/confidence scores on provenance | Low | Enables quality-based filtering |
| **High** | Entity-type-aware blocking in ER | Low | Reduces LLM matching costs |
| **Medium** | Conflict detection for multi-source facts | Medium | Critical for multi-source quality |
| **Medium** | Human-in-the-loop review queue for low-confidence triples | Medium | Improves overall KG quality |
| **Medium** | Parallel source processing | Medium | Throughput improvement |
| **Low** | Entity importance scoring (degree + PageRank) | Low | Useful for downstream ranking |
| **Low** | KG embedding-based fact verification | Medium | Automated quality assurance |
| **Low** | Unified cross-store query interface | Medium | Better developer experience |
| **Skip** | Live/streaming KG | High | Not relevant to current use case |
| **Skip** | Federated polystore | High | Overkill at current scale |
| **Skip** | Custom query language (KGQ) | High | Cypher is sufficient |