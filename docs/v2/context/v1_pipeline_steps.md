# Spindle End-to-End Pipeline Steps

## Phase A — Setup & Configuration

| # | Step | Description |
|---|------|-------------|
| **1** | **Project Configuration** | Load or generate a `SpindleConfig` (storage paths, template settings, LLM config, vector/graph store settings, observability) from a Python config file or programmatic defaults. Auto-detects available LLM API keys. |
| **2** | **Observability Initialization** | Instantiate the central `EventRecorder`, optionally attach a persistent `EventLogStore` (SQLite-backed) and/or Langfuse observer so that every subsequent step emits structured `ServiceEvent` records. |
| **3** | **Storage Backend Initialization** | Create or connect to the storage backends: `DocumentCatalog` (SQLite for document/chunk metadata), `ChromaVectorStore` (embeddings), and `GraphStore` (Kùzu graph database). Factory function `create_storage_backends()` wires these up from config. |

## Phase B — Document Ingestion

| # | Step | Description |
|---|------|-------------|
| **4** | **Template Resolution** | Match each input file to an ingestion template (by MIME type, file extension, or path glob) via the `TemplateRegistry`. Templates define which loader, splitter, metadata extractor, and pre/post-processing hooks to use. Built-in templates cover plain text and PDF. |
| **5** | **Document Loading** | Convert raw files to LangChain `Document` objects using the loader specified by the resolved template (e.g., text file reader, PDF parser). Produces `DocumentArtifact` records with source metadata (title, author, date, source URL). |
| **6** | **Document Preprocessing** | Apply template-defined preprocessing hooks (e.g., `strip_empty_text`) to clean and normalize loaded document content before splitting. |
| **7** | **Document Splitting / Chunking** | Split documents into smaller chunks using the template-specified splitter (e.g., recursive character splitter with configurable chunk size/overlap). Each chunk becomes a `ChunkArtifact` with positional metadata linking it back to its parent document. |
| **8** | **Chunk Metadata Enrichment** | Attach per-chunk metadata: document ID, chunk index, character offsets, token estimates, and any custom metadata produced by the template's metadata extractor. |
| **9** | **Checksum & Deduplication** | Compute content checksums for documents and chunks; skip re-processing of documents already present in the catalog with matching hashes. |
| **10** | **Document Graph Construction** | Build a `DocumentGraph` (DAG) capturing structural relationships between documents and their chunks — document-contains-chunk edges plus any cross-document relationships. Handled by `DocumentGraphBuilder`. |
| **11** | **Catalog Persistence** | Persist `DocumentArtifact`, `ChunkArtifact`, graph nodes/edges, and the `IngestionRun` record to the SQLite `DocumentCatalog`. |
| **12** | **Vector Store Indexing** | Embed chunk text and store embeddings in the `ChromaVectorStore`, enabling downstream semantic search and entity-resolution blocking. |
| **13** | **Corpus Management** | Organize ingested documents into named `Corpus` collections via `CorpusManager`. A corpus groups documents for pipeline processing, tracks pipeline state per stage, and supports CRUD operations. |
| **14** | **Ingestion Analytics** | The `IngestionAnalyticsEmitter` computes per-document observations: structural metrics (token/char counts), sliding chunk-window summaries, context-strategy recommendations, and risk signals. Persists to `AnalyticsStore`. |

## Phase C — Knowledge Organization System (KOS) Pipeline

The `PipelineOrchestrator` drives six stages sequentially (or via batch-consolidate / sample-based strategies). Each stage processes corpus chunks, merges per-chunk results, and persists artifacts.

| # | Step | Description |
|---|------|-------------|
| **15** | **Stage 1 — Controlled Vocabulary Extraction** | The `VocabularyStage` calls the BAML `ExtractControlledVocabulary` prompt per chunk to identify domain-specific terms (preferred labels, definitions, scope notes). Results are consolidated with `ConsolidateVocabulary` to deduplicate and normalize terms into `VocabularyTerm` records. |
| **16** | **Stage 2 — Metadata Schema Extraction** | The `MetadataStage` calls `ExtractMetadataSchema` to identify structural, descriptive, and administrative metadata elements present in the corpus. Produces `MetadataElement` records typed as structural/descriptive/administrative. |
| **17** | **Stage 3 — Taxonomy Extraction** | The `TaxonomyStage` calls `ExtractTaxonomy` to discover hierarchical (broader/narrower) relationships among vocabulary terms. Outputs `TaxonomyNode` and `TaxonomyRelation` objects forming a polyhierarchical tree. |
| **18** | **Stage 4 — Thesaurus Extraction** | The `ThesaurusStage` calls `ExtractThesaurus` to produce ISO 25964/SKOS-style associative relationships (USE, BT, NT, RT, UF) between terms. Outputs `ThesaurusEntry` records with cross-references. |
| **19** | **Stage 5 — Ontology Synthesis** | The `OntologyStage` calls `EnhanceOntologyFromPipeline`, feeding in accumulated vocabulary, taxonomy, and thesaurus data to generate a formal `Ontology` (entity types, relation types, attribute definitions). This ontology governs what the knowledge-graph extractor will look for. |

## Phase D — Knowledge Graph Extraction

| # | Step | Description |
|---|------|-------------|
| **20** | **Ontology Recommendation (Standalone Path)** | When no KOS pipeline is run, the `OntologyRecommender` can auto-generate an ontology directly from sample text, with configurable scope (minimal / balanced / comprehensive). Supports iterative extension via `analyze_extension()` and `extend_ontology()`. |
| **21** | **Triple Extraction** | The `SpindleExtractor` calls the BAML `ExtractTriples` prompt against each chunk (or batch), using the provided/generated ontology to extract `Triple` objects (subject, predicate, object, confidence, evidence spans, source metadata). Supports sync, async, batch, and streaming modes. |
| **22** | **Span Index Computation** | Post-extraction helper `_compute_all_span_indices` maps each triple's evidence text spans back to character offsets in the source chunk, enabling provenance tracing. Handles whitespace normalization and fuzzy matching. |
| **23** | **Process Graph Extraction (Optional)** | `extract_process_graph()` calls the BAML `ExtractProcessDAG` prompt to identify process steps, temporal ordering, and control flow (sequence, parallel, conditional, loop). Results are rehydrated, merged across chunks, validated for cycles, and boundary-recalculated. |
| **24** | **LLM Metrics & Cost Collection** | After each LLM call, BAML metrics collectors capture model name, input/output token counts, and latency. The extractor computes estimated cost using `spindle/llm_pricing.py` and records it via the `EventRecorder`. |

## Phase E — Entity Resolution

| # | Step | Description |
|---|------|-------------|
| **25** | **Graph Embedding for Blocking** | Compute embeddings for all nodes (and optionally edges) in the graph store using the `ChromaVectorStore`. Serializes node/edge attributes into text, then embeds for similarity computation. |
| **26** | **Semantic Blocking** | The `SemanticBlocker` clusters candidate entities into blocks using cosine similarity on embeddings. Supports multiple clustering methods: k-means, hierarchical, HDBSCAN, and simple threshold. Reduces the O(n²) matching problem to within-block comparisons. |
| **27** | **Semantic Matching** | The `SemanticMatcher` sends each block of candidate duplicates to an LLM (BAML `MatchEntities` / `MatchEdges`) which returns pairwise `EntityMatch` or `EdgeMatch` verdicts with confidence scores and reasoning. |
| **28** | **Merge / SAME_AS Edge Creation** | `create_same_as_edges()` writes `SAME_AS` relationships into the graph for confirmed duplicate nodes (and edges). `get_duplicate_clusters()` then computes connected components to find transitive duplicate groups. |
| **29** | **Canonical Entity Resolution** | `get_canonical_entity()` and `query_with_resolution()` allow querying the graph with transparent deduplication — returning results through canonical entities while respecting SAME_AS chains. |

## Phase F — Graph Storage & Enrichment

| # | Step | Description |
|---|------|-------------|
| **30** | **Node & Edge Persistence** | `GraphStore.add_triples()` extracts subject/object nodes and relationship edges from triples and persists them to the Kùzu backend. Duplicate edges trigger evidence merging (combining source references and spans). |
| **31** | **Graph Embedding (Node2Vec)** | `GraphEmbeddingGenerator` extracts the graph topology into a NetworkX graph, runs Node2Vec random walks to produce structural embeddings, and stores them back into both the graph (node properties) and vector store. |
| **32** | **Graph Querying** | The `GraphStore` exposes multiple query interfaces: pattern-based (`query_by_pattern`), source-based (`query_by_source`), date-range (`query_by_date_range`), and raw Cypher (`query_cypher`). All support optional duplicate resolution. |

## Phase G — Serving & Visualization

| # | Step | Description |
|---|------|-------------|
| **33** | **REST API** | The FastAPI application (`spindle/api/`) exposes seven router groups — corpus, ingestion, extraction, ontology, pipeline, process, and resolution — providing HTTP endpoints for all pipeline operations. Session management shares `GraphStore`/`VectorStore` instances across requests. |
| **34** | **Dashboard** | The Streamlit dashboard (`spindle/dashboard/app.py`) provides interactive visualization of the graph, extraction results, and analytics data. |
| **35** | **Analytics Views** | Pre-built analytical views (`analytics/views.py`) aggregate stored observations into corpus overviews, document size tables, chunk-window risk reports, ontology recommendation metrics, triple extraction metrics, and entity resolution metrics. |

---

**Summary of the flow:** Configuration (1–3) → Document Ingestion (4–14) → KOS Development (15–19) → Knowledge Extraction (20–24) → Entity Resolution (25–29) → Graph Storage & Enrichment (30–32) → Serving & Visualization (33–35).

Steps 15–19 and 20 represent two alternative paths to obtain an ontology: the full KOS pipeline builds one bottom-up from the corpus, while the standalone `OntologyRecommender` generates one directly. Both converge at step 21 (Triple Extraction).
