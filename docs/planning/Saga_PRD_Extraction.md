# Product Requirements Document: Saga Knowledge Platform

**Source:** Ilyas, Rekatsinas, Konda, Pound, Qi, Soliman (Apple)
**Published:** SIGMOD 2022 — arXiv:2204.07309v1
**PRD Extracted:** March 2026

> **Evidence Classification Legend**
> - **[STATED]** — Directly described in the article
> - **[INFERRED]** — Reasonably deduced from context
> - **[GAP]** — Not addressed; best guess included where possible, marked as speculative

---

## 1. Problem Statement

### 1.1 Core Problem

- **[STATED]** Accurate and up-to-date knowledge about real-world entities is needed in many applications, but constructing a central knowledge graph (KG) that serves these needs is a challenging problem.
- **[STATED]** Data about entities comes from multiple heterogeneous sources with different update rates, formats, licensing constraints, and trustworthiness levels, requiring complex integration.
- **[STATED]** Entries of data sources used to construct the KG are continuously changing: new entities appear, entities might be deleted, and facts about existing entities change at different frequencies.
- **[INFERRED]** Without a centralized platform, each application team would need to independently build and maintain its own entity resolution, data fusion, and knowledge serving infrastructure — resulting in duplicated effort and inconsistent data.

### 1.2 Cost of the Status Quo

- **[STATED]** Legacy view computation systems (custom Spark jobs) required nearly 10x more hardware than Saga's optimized Graph Engine for equivalent workloads.
- **[STATED]** Existing entity disambiguation solutions failed on tail (less popular) entities because they relied on training data correlations rather than relational KG context.
- **[INFERRED]** Before Saga, knowledge was fragmented across applications, leading to inconsistent entity representations, stale facts, and poor coverage for niche entities.
- **[GAP]** No explicit dollar-cost or engineering-time figures are given for the pre-Saga state. Speculative: significant operational cost from maintaining parallel pipelines across teams.

### 1.3 Existing Solutions / Workarounds Referenced

- **[STATED]** Earlier knowledge graph projects: DBpedia, Freebase, KnowItAll, YAGO, Wikidata, and DeepDive.
- **[STATED]** Industrial KGs from other organizations including general purpose and vertical deployments.
- **[STATED]** Custom Spark jobs were used as a legacy approach for computing KG views prior to the Graph Engine.
- **[STATED]** An alternative deployed Entity Disambiguation solution (not using relational KG context) was the baseline NERD was measured against.

---

## 2. Target Users & Personas

### 2.1 Primary User Segments

- **[STATED]** Search and assistant services requiring open-domain knowledge for question answering (e.g., Siri).
- **[STATED]** Entity-centric experience teams needing rich entity data for Entity Cards and similar surface features.
- **[STATED]** Machine learning teams needing training datasets with entity information and relationships.
- **[STATED]** Domain teams (music, media, sports, celebrities, nutrition, etc.) who onboard specialized data sources.
- **[INFERRED]** Internal platform/infrastructure engineers who maintain and extend the Saga pipelines.

### 2.2 User Goals & Pain Points

- **[STATED]** Consumers need data freshness (sports scores within seconds), accuracy (correct entity linking), and availability (billions of queries daily at <20ms p95 latency).
- **[STATED]** Domain teams need low-effort onboarding of new data sources through self-serve APIs.
- **[STATED]** Production use cases require provenance annotations on all facts for data governance, license compliance, and on-demand deletion.
- **[STATED]** ML teams need custom KG views, computed artifacts (e.g., entity importance, embeddings), and consistent training data.

### 2.3 Buyer vs. End-User Distinction

- **[INFERRED]** Saga is an internal Apple platform; there is no external buyer. The "buyer" is Apple's organizational leadership investing in centralized knowledge infrastructure.
- **[INFERRED]** End-users are internal Apple teams and, indirectly, Apple's consumer product users who interact with Siri, Search, Entity Cards, etc.
- **[GAP]** No information on team size, organizational structure, or internal adoption/rollout process.

---

## 3. Product Vision & Positioning

### 3.1 High-Level Value Proposition

- **[STATED]** Saga is a next-generation knowledge construction and serving platform for continuously integrating billions of facts about real-world entities and powering experiences across a variety of production use cases.
- **[STATED]** It provides a shared KG construction and serving solution across applications, eliminating redundant infrastructure.

### 3.2 Differentiation from Alternatives

- **[STATED]** Hybrid batch-incremental design supporting both batch processing (daily source dumps) and streaming (sports scores, stock prices) with SLAs around freshness, latency, and availability.
- **[STATED]** Non-destructive data integration with full provenance tracking on every fact, enabling license compliance, attribution, and on-demand deletion.
- **[STATED]** Federated polystore approach using specialized engines for each workload rather than forcing a single system to handle all query patterns.
- **[STATED]** NERD stack that uses KG relational context for disambiguation, achieving ~70% recall improvement on tail entities vs. deployed alternative at 0.9 confidence.

### 3.3 Design Philosophy & Principles

- **[STATED]** Delta-based continuous construction: always operate on diffs rather than full reprocessing.
- **[STATED]** Self-serve-centric and modular APIs for extensibility and ease of onboarding.
- **[STATED]** Config-driven development paradigm (e.g., Predicate Generation Functions) to minimize custom code.
- **[STATED]** Parallelism at every level: inter-source, intra-source (Added/Updated/Deleted payloads), and intra-block.
- **[INFERRED]** Domain agnosticism: the same pipelines process both open-domain and domain-specific data through ontology alignment.

---

## 4. Features & Capabilities

### 4.1 Data Source Ingestion

- **[STATED]** Pluggable, configurable adapters that implement Import → Entity Transform → Ontology Alignment → Delta Computation → Export pipeline stages.
- **[STATED]** Supports heterogeneous formats: Parquet/HDFS, CSV, JSON, HTTP endpoints.
- **[STATED]** Delta Computation eagerly detects changes (Added/Deleted/Updated) against previously consumed snapshots to minimize downstream processing.

> **User Story:** As a domain data engineer, I want to onboard a new specialized data source by implementing standard interfaces (Importer, Transformer, PGFs), so that my domain's entities are continuously integrated into the central KG without building custom end-to-end pipelines.

**Priority:** Core

**Open Questions:**
- What is the average time to onboard a new data source? No SLA or timeline given.
- How are schema evolution conflicts handled when a source's format changes?
- What monitoring/observability exists for ingestion pipeline health?

---

### 4.2 Knowledge Construction Pipeline

- **[STATED]** Four-stage process: Linking (blocking → pair generation → matching → resolution), Object Resolution, and Fusion.
- **[STATED]** Linking uses entity-type-specific blocking functions, domain-specific matching models (ML-based or rule-based), and correlation clustering for resolution.
- **[STATED]** Object Resolution uses NERD to map string literals in object fields to KG entity identifiers.
- **[STATED]** Fusion performs outer joins for simple facts and similarity-based merging for composite relationship nodes, with truth discovery for correctness probabilities.

> **User Story:** As a KG platform operator, I want new source data to be automatically deduplicated, linked to existing entities, and fused into a consistent KG state, so that downstream consumers always see an accurate, unified view of entities.

**Priority:** Core

**Open Questions:**
- What are the precision/recall targets for entity linking across different entity types?
- How are linking errors surfaced and corrected?
- What is the end-to-end latency from source change to KG update for batch sources?

---

### 4.3 Parallel Knowledge Graph Construction

- **[STATED]** Inter-source parallelism: source ingestion pipelines run in parallel; synchronization only at fusion.
- **[STATED]** Intra-source parallelism: Added, Updated, and Deleted payloads processed in parallel within each source.
- **[STATED]** Updated/Deleted payloads skip full linking and only require ID lookups and object resolution.
- **[STATED]** Volatile properties (e.g., popularity scores) are factored out into separate payloads with optimized partition-overwrite fusion.

> **User Story:** As a platform operator, I want KG construction to run in parallel across all sources and payload types, so that end-to-end construction time is minimized even as the number of sources grows.

**Priority:** Core

**Open Questions:**
- What is the total construction cycle time? Only relative improvements mentioned.
- How is ordering/consistency maintained across parallel fusion operations?

---

### 4.4 Knowledge Graph Query Engine (Graph Engine)

- **[STATED]** Serves as primary KG store, computes knowledge views, and exposes query APIs.
- **[STATED]** Federated polystore: analytics warehouse, entity index, text index, vector DB, plus elastic compute cluster.
- **[STATED]** Distributed shared log coordinates continuous ingest across all stores for eventual consistency.
- **[STATED]** Extensible orchestration agent framework for integrating new storage/compute engines.
- **[STATED]** Log sequence numbers (LSN) used as distributed synchronization primitives to track store freshness.

> **User Story:** As a KG consumer, I want to query the graph through specialized APIs (entity retrieval, text search, analytics, NN search) with low latency, so that my application can serve real-time user experiences.

**Priority:** Core

**Open Questions:**
- Which specific technologies back the analytics warehouse, text index, entity index, and vector DB?
- What is the consistency model for cross-store reads?
- What are the storage costs/requirements at current scale?

---

### 4.5 Knowledge Graph Views

- **[STATED]** Views can be any transformation: sub-graph, schematized relational, aggregates, iterative algorithms (PageRank), or learned representations (embeddings).
- **[STATED]** View definitions scripted against target engine native APIs; stored in a central View Catalog with dependency tracking.
- **[STATED]** View Manager orchestrates execution of the dependency graph, including cross-engine view dependencies.
- **[STATED]** 26% run-time improvement achieved by reusing common views via dependency graph optimization.
- **[STATED]** Up to 14x performance improvement (avg 5x) vs. legacy Spark-based view computation, using ~10x less hardware.

> **User Story:** As a data consumer, I want to register custom views of the KG with specific freshness SLAs, so that my application receives exactly the derived data it needs, automatically maintained as the KG updates.

**Priority:** Core

**Open Questions:**
- What is the maximum view complexity supported? Are there limits on dependency chain depth?
- How are view failures handled and retried?
- What is the view registration and approval workflow?

---

### 4.6 Entity Importance Scoring

- **[STATED]** Aggregates four structural metrics: in-degree, out-degree, number of identities (source count), and PageRank.
- **[STATED]** Designed to cover head, torso, and tail entities, unlike external popularity signals which only cover head entities.
- **[STATED]** Modeled as a KG view, automatically maintained by the analytics engine as the graph changes.

> **User Story:** As a ranking engineer, I want a graph-structure-based importance score for every entity, so that I can rank entities for search, QA, and display even when external popularity signals are absent.

**Priority:** Supporting

**Open Questions:**
- How are the four metrics weighted/aggregated? Is it a learned model or fixed formula?
- How does importance scoring handle newly added entities with minimal connectivity?

---

### 4.7 Live Knowledge Graph

- **[STATED]** Union of a stable KG view with real-time streaming sources (sports scores, stock prices, flight data).
- **[STATED]** Live sources are uniquely identifiable and do not require full linking/fusion; ambiguous entity references resolved via NERD against the stable graph.
- **[STATED]** Indexed via scalable inverted index and key-value store, sharded and replicated for scale-out.
- **[STATED]** Handles billions of queries daily with 95th percentile latency under 20ms.

> **User Story:** As a Siri/search engineer, I want to query live-updating facts (game scores, stock prices) that are linked to stable entity information, so that I can answer time-sensitive user questions with sub-second freshness.

**Priority:** Core

**Open Questions:**
- What is the end-to-end latency from streaming source event to queryable index update?
- How many concurrent live streaming sources are supported?

---

### 4.8 Live Graph Query Engine

- **[STATED]** Supports ad-hoc structured graph queries via KGQ, a custom graph query language balancing expressiveness with bounded performance.
- **[STATED]** Intent handling routes annotated NL queries to correct KGQ executions based on entity semantics (e.g., prime minister vs. mayor).
- **[STATED]** Maintains context graphs for multi-turn query sequences, allowing follow-up queries to reference prior entities and intents.
- **[STATED]** Supports virtual operators for encapsulating complex expressions as reusable operators.

> **User Story:** As a conversational AI developer, I want to issue multi-turn structured queries against the KG where follow-up questions automatically resolve references from prior turns, so that users can have natural multi-step information-seeking conversations.

**Priority:** Core

**Open Questions:**
- What is the KGQ language specification? Only high-level description given.
- How long are context graphs retained? Per-session or persistent?
- What are the query compilation and optimization strategies beyond operator push-down and intra-query parallelism?

---

### 4.9 Live Graph Curation (Human-in-the-Loop)

- **[STATED]** Facts containing potential errors or vandalism are detected and quarantined for human curation.
- **[STATED]** Curations treated as a streaming source by live graph construction, enabling hot-fixing of live indexes.
- **[STATED]** Curations also flow back into stable KG construction as a source.

> **User Story:** As a content quality curator, I want detected errors to be flagged for my review and, once corrected, immediately reflected in the live index and eventually in the stable KG, so that users never see stale or incorrect information for long.

**Priority:** Supporting

**Open Questions:**
- What detection methods identify potential errors and vandalism? Rule-based? ML-based?
- What is the mean time from detection to curation resolution?
- What tooling exists for the curation team?

---

### 4.10 Named Entity Recognition & Disambiguation (NERD)

- **[STATED]** Complete NERD stack for identifying and disambiguating entity mentions in text against the KG.
- **[STATED]** Uses NERD Entity View: comprehensive summary per entity including names/aliases, types, descriptions, relationships, neighbor info, and importance scores.
- **[STATED]** Candidate retrieval uses neural string similarity and optional entity type constraints; contextual disambiguation uses transformer-based classification with rejection.
- **[STATED]** Elastic deployment for large batch jobs; high-performance low-latency variant for online workloads.
- **[STATED]** Models trained via weak supervision combining annotated text, curated query logs, and template-generated snippets.

> **User Story:** As a content understanding engineer, I want to annotate arbitrary text or semi-structured records with disambiguated KG entities, so that downstream applications can leverage entity-centric information for categorization, search, and recommendation.

**Priority:** Core

**Open Questions:**
- What is the online NERD latency for a single disambiguation request?
- How frequently are NERD models retrained?
- What is the entity coverage (percentage of KG entities in the NERD Entity View)?

---

### 4.11 Neural String Similarities

- **[STATED]** Neural network-based character-sequence encoders producing high-dimensional vectors; cosine similarity for comparison.
- **[STATED]** Capture synonyms (e.g., Robert/Bob) and typos beyond deterministic similarity functions.
- **[STATED]** Separate encoders per string type (human names, locations, album titles, etc.) trained via distant supervision with triplet loss.
- **[STATED]** Recall improvements of more than 20 basis points in entity matching when using learned similarities.

> **User Story:** As a matching model developer, I want pre-trained neural similarity functions for different string types, so that I can improve recall in entity matching without building custom encoders from scratch.

**Priority:** Supporting

**Open Questions:**
- What encoder architecture is used (RNN, CNN, Transformer)?
- How large are the embeddings and what is inference latency?

---

### 4.12 Knowledge Graph Embeddings

- **[STATED]** Supports multiple embedding models (TransE, DistMult, etc.) via a generalizable architecture.
- **[STATED]** Used for fact ranking, fact verification (outlier detection), and missing fact imputation via vector similarity search.
- **[STATED]** Trained using Marius system on single-node multi-GPU with external memory; completes in ~1 day for billion-scale KGs.
- **[STATED]** Embeddings stored in Vector DB for similarity search functionalities.

> **User Story:** As a KG quality engineer, I want graph embeddings that enable automated fact ranking, verification, and imputation, so that I can improve KG completeness and surface erroneous facts for auditing.

**Priority:** Supporting

**Open Questions:**
- How are embedding quality metrics evaluated?
- How frequently are embeddings retrained?
- What embedding dimensionality is used in production?

---

### 4.13 Provenance & Trust Management

- **[STATED]** Every fact annotated with source array, locale, and trustworthiness score array.
- **[STATED]** Enables attribution, license compliance, KG views conforming to licensing agreements, and on-demand data deletion.
- **[STATED]** Truth discovery and source reliability methods estimate probability of correctness for consolidated facts.

> **User Story:** As a compliance officer, I want every fact in the KG to carry full provenance and trust metadata, so that I can enforce license agreements, respond to data deletion requests, and audit fact origins.

**Priority:** Core

**Open Questions:**
- How are deletion requests propagated to all downstream views and indexes?
- What is the latency for on-demand deletion?

---

## 5. System Architecture & Technical Details

### 5.1 High-Level Architecture

- **[STATED]** Three-layer architecture: (1) Batch Processing (Stable Sources → KG Construction → KG → Graph Engine → ML Training), (2) Edge Serving (Streaming Sources → Live Graph Construction → Live KG → Live Graph Query Engine), (3) Consumers (KG Views → Models → Batch/Live ML Services).

### 5.2 Data Model

- **[STATED]** RDF-based `<subject, predicate, object>` triples, extended to capture one-hop relationships (extended triples).
- **[STATED]** Extended triples provide a flat relational model avoiding expensive self-joins for one-hop lookups.
- **[STATED]** Variation of JSON-LD format adopted for efficient querying at industry scale.
- **[STATED]** Metadata fields: provenance (source array), locale, and trustworthiness (score array per source).

### 5.3 Storage Infrastructure

- **[STATED]** Federated polystore: analytics warehouse (relational, read-optimized), entity index, text index, vector DB, distributed object store.
- **[STATED]** Distributed shared log for coordinating continuous ingest across stores.
- **[STATED]** Elastic compute cluster for ML workloads (graph embeddings, etc.).
- **[STATED]** GPU cluster for training embedding models and running NERD disambiguation.
- **[GAP]** Specific technologies/products for each store not named (e.g., which relational warehouse, which vector DB, which message bus).

### 5.4 Performance & Scale Claims

- **[STATED]** Billions of facts and entities in the KG; 33x increase in facts and 6.5x increase in entities since 2018.
- **[STATED]** Live KG engine handles billions of queries daily at <20ms p95 latency.
- **[STATED]** Graph Engine views: avg 5x performance improvement (up to 14x) vs. legacy Spark, using ~10x less hardware.
- **[STATED]** 26% runtime improvement from view dependency reuse.
- **[STATED]** KG embedding training: ~1 day for billion-scale graph (Marius system).
- **[STATED]** NERD: ~70% recall improvement at 0.9 confidence threshold vs. deployed baseline for text annotations.

### 5.5 Security, Privacy & Compliance

- **[STATED]** All facts carry provenance enabling license compliance and on-demand data deletion.
- **[STATED]** KG views can be configured to conform to licensing agreements for specific consumers.
- **[STATED]** Privacy policy compliance enforced for different registered views.
- **[STATED]** Changes to licensing or privacy/trustworthiness requirements dynamically affect admissible data sources.
- **[GAP]** No mention of encryption at rest/in transit, access control models, audit logging, or GDPR/CCPA-specific compliance mechanisms.

---

## 6. User Experience & Workflows

### 6.1 Described User Journeys

- **[STATED]** Open-Domain QA flow: User utterance → NERD entity disambiguation → Intent inference → Structured KGQ query → Live KG Query Engine → Answer retrieval.
- **[STATED]** Multi-turn QA: Context graphs maintain intent and entity references across sequential queries (e.g., "Who is Beyoncé married to?" → "How about Tom Hanks?" → "Where is she from?").
- **[STATED]** Entity Card rendering: Query KG for entity facts + neighbors ranked by popularity → compile rich entity view for display.
- **[STATED]** Data onboarding: Engineer implements Importer, Transformer, and PGFs via configuration → pipeline auto-integrates source data into KG.
- **[STATED]** Curation: Errors/vandalism detected → quarantined → human review → correction streamed to live index + fed back to stable KG.

### 6.2 UI/UX Details

- **[GAP]** No UI screenshots, wireframes, or interaction patterns described. The paper is an engineering/systems paper, not a product design document.
- **[INFERRED]** End-user UX is mediated through Apple products (Siri, search results, Entity Cards); Saga itself is backend infrastructure.
- **[STATED]** Custom-built curation tooling exists for the human curation team, but no details provided.

### 6.3 Onboarding / Adoption

- **[STATED]** Self-serve data onboarding via modular APIs: engineers implement Data Source Importer, Data Transformer, and Predicate Generation Functions.
- **[STATED]** Saga provides importer templates that can be altered for custom source ingestion.
- **[STATED]** Config-driven development paradigm reduces custom code needed for ontology alignment.
- **[GAP]** No documentation quality, tutorials, developer portal, or training program details.

---

## 7. Success Metrics & Outcomes

### 7.1 Quantitative Claims

| Metric | Value | Evidence |
|--------|-------|----------|
| KG fact growth | 33x increase since 2018 | STATED |
| KG entity growth | 6.5x increase since 2018 | STATED |
| Live KG query volume | Billions of queries/day | STATED |
| Live KG query latency (p95) | <20ms | STATED |
| View computation speedup | Avg 5x, up to 14x vs. legacy Spark | STATED |
| Hardware efficiency | Legacy used ~10x more hardware | STATED |
| View dependency reuse gain | 26% runtime improvement | STATED |
| NERD recall improvement (text) | ~70% at 0.9 confidence | STATED |
| NERD precision improvement (text) | Up to 3.4% at ≥0.8 confidence | STATED |
| NERD precision (OBR, w/ type hints) | ~10% improvement | STATED |
| NERD recall (OBR) | ~25% improvement | STATED |
| Neural similarity recall gain | >20 basis points for matching | STATED |
| Embedding training time | ~1 day for billion-scale KG | STATED |

### 7.2 Stated KPIs / Success Criteria

- **[STATED]** Data freshness SLAs: streaming updates within seconds; batch updates on daily cadence.
- **[STATED]** Accuracy SLAs driven by confidence scores on facts.
- **[STATED]** Availability SLAs for the Live KG engine supporting interactive user-facing services.
- **[GAP]** Specific numeric SLA thresholds not disclosed.

### 7.3 Customer Quotes / Case Studies

- **[GAP]** No customer quotes. The paper describes internal Apple deployments. Case studies provided for QA, Entity Cards, and Semantic Annotations, but without external customer testimonials.

---

## 8. Scope & Constraints

### 8.1 What the Product Does NOT Do

- **[STATED]** KGQ intentionally limits expressiveness vs. general graph query languages to bound query performance.
- **[STATED]** Live sources bypass the complex linking/fusion pipeline — only stable sources go through full entity resolution.
- **[INFERRED]** Saga does not appear to handle unstructured data extraction directly; it consumes pre-extracted structured/semi-structured sources (DeepDive-style extraction referenced as prior work, not as a Saga capability).
- **[INFERRED]** No NL-to-KGQ generation described within Saga itself; NL understanding happens upstream in the search/assistant stack.

### 8.2 Implied Constraints

- **[STATED]** Internal Apple platform — not a commercial product or open-source offering.
- **[STATED]** Ontology alignment is manually defined via configuration files, implying human effort for each source.
- **[STATED]** Blocking and matching models are entity-type-specific, requiring per-domain development.
- **[STATED]** Graph embedding training requires GPU cluster resources; memory requirements for billion-scale graphs exceed single-GPU and even single-node main memory.
- **[INFERRED]** The system assumes data sources provide entity-level identifiers (each entity must have an ID predicate), which may not always be the case for raw web data.

### 8.3 Dependencies

- **[STATED]** Marius system for graph embedding training.
- **[STATED]** External data sources (Wikipedia, Wikidata, specialized providers) with varying license agreements.
- **[STATED]** GPU cluster for NERD model training and inference, and embedding training.
- **[INFERRED]** Distributed computing infrastructure (Spark mentioned as legacy; current batch processing framework unspecified).
- **[GAP]** No details on cloud vs. on-premise deployment, CI/CD pipeline, or operational runbook.

---

## 9. Roadmap Signals

- **[STATED]** Figure 12 shows continuous growth trajectory for KG facts and entities, suggesting ongoing investment in source onboarding.
- **[STATED]** The paper describes Saga as "next-generation," implying it replaced or evolved from a prior system (the "legacy system" referenced in benchmarks).
- **[INFERRED]** Product maturity: established in production, not a v1 prototype. Powers multiple critical Apple services (Siri QA, Entity Cards, semantic annotations).
- **[INFERRED]** The dashed line in the growth chart (Figure 12) marking Saga's introduction suggests it launched roughly mid-timeline (circa 2019–2020), meaning 2–3 years of production hardening by the time of publication.
- **[GAP]** No explicit roadmap items, planned features, or future directions mentioned.
- **[GAP]** No indication of whether LLM integration for extraction, QA augmentation, or ontology automation is planned (paper predates the ChatGPT era).

---

## 10. Gap Analysis

The following unknowns are ranked by importance for a team looking to build or integrate a comparable system.

| # | Gap Area | What's Missing | Impact | Criticality |
|---|----------|----------------|--------|-------------|
| 1 | End-to-End Latency SLAs | No stated targets for batch source change → KG update → view refresh → consumer availability. Only streaming p95 (<20ms) is given. | Capacity planning, SLA negotiation | **Critical** |
| 2 | Technology Stack | Specific products/systems for analytics warehouse, vector DB, text index, entity store, message bus, and batch compute framework are not named. | Build vs. buy decisions | **Critical** |
| 3 | Error Handling & Recovery | No discussion of failure modes, rollback procedures, data corruption recovery, or pipeline monitoring/alerting. | Operational reliability | **Critical** |
| 4 | Access Control & Security | No details on authentication, authorization, encryption, audit logging, or specific privacy regulation compliance (GDPR, CCPA). | Compliance, governance | **High** |
| 5 | KGQ Specification | The custom query language is described conceptually but no formal grammar, operator catalog, or documentation is provided. | Developer adoption | **High** |
| 6 | Onboarding Effort | No metrics on time/effort to onboard a new source. Is it days, weeks, or months? What skills are required? | Scalability of source growth | **High** |
| 7 | Ontology Management | No details on how the ontology evolves, how schema migrations are handled, or how type conflicts are resolved. | Long-term maintainability | **High** |
| 8 | Cost / Resource Model | No information on infrastructure costs, compute/storage requirements, or team sizing. | Budget justification | **Medium** |
| 9 | Testing & Validation | No mention of regression testing, accuracy monitoring, A/B testing frameworks, or data quality dashboards. | Quality assurance | **Medium** |
| 10 | Multi-tenancy & Isolation | How are different consumer teams isolated? Can one team's heavy query load affect another's SLAs? | Service reliability | **Medium** |

---

## Confidence Summary

| Section | Confidence | Rationale |
|---------|------------|-----------|
| 1. Problem Statement | **High** | Well-articulated in the introduction with clear challenge enumeration. |
| 2. Target Users & Personas | **Medium** | Use cases are described but personas are inferred from an academic paper, not a product spec. |
| 3. Product Vision & Positioning | **High** | Design philosophy and differentiation are explicit throughout. |
| 4. Features & Capabilities | **High** | Richest section; most features are directly described with architectural detail. |
| 5. Architecture & Technical Details | **Medium–High** | Architecture is well-described at the conceptual level; specific technology names are absent. |
| 6. User Experience & Workflows | **Low–Medium** | Backend-focused paper; no UI/UX details. Workflows are described at the system level, not the user interaction level. |
| 7. Success Metrics & Outcomes | **Medium–High** | Strong quantitative claims but absolute numbers (scale, cost) are often relative, not absolute. |
| 8. Scope & Constraints | **Medium** | Limitations are partially inferred from what's described vs. what's omitted. |
| 9. Roadmap Signals | **Low** | No explicit roadmap. Signals are inferred from growth trends and publication timing. |
| 10. Gap Analysis | **High** | Gaps are clearly identifiable from what the paper does not address. |
