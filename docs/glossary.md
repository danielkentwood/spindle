# Spindle Glossary

A comprehensive reference of all terms, classes, entities, schemas, and artifact types used in Spindle.

## Table of Contents

- [Core Concepts](#core-concepts)
- [Main Classes](#main-classes)
- [Data Models & Schemas](#data-models--schemas)
- [BAML Types](#baml-types)
- [Ingestion System](#ingestion-system)
- [Ontology Pipeline](#ontology-pipeline)
- [Entity Resolution](#entity-resolution)
- [Graph Store](#graph-store)
- [Vector Store](#vector-store)
- [Analytics & Observability](#analytics--observability)
- [Configuration](#configuration)
- [API Models](#api-models)
- [Enumerations](#enumerations)
- [CLI Commands](#cli-commands)

---

## Core Concepts

### Knowledge Graph Triple
A fundamental unit of knowledge representation consisting of a **subject** (entity), **predicate** (relationship), and **object** (entity). In Spindle, triples are extracted from text using LLM-powered extraction.

### Ontology
A formal specification of entity types and relation types that defines the schema for knowledge extraction. Ontologies can be user-defined or automatically recommended by Spindle.

### Entity
A named object or concept in the knowledge graph. Each entity has a name, type, description, and optional custom attributes.

### Relation
A directed relationship between two entities, defined by a predicate with domain (subject type) and range (object type) constraints.

### Source Metadata
Information about the origin of extracted knowledge, including source name, URL, and extraction timestamp.

### Evidence Span
A text span (with start and end character positions) that provides supporting evidence for an extracted triple.

---

## Main Classes

### `SpindleExtractor`
**Module:** `spindle.extraction.extractor`

The primary class for extracting knowledge graph triples from text using LLM-powered extraction.

**Key Methods:**
- `extract(text, source_name)` - Extract triples from text
- `batch_extract(texts)` - Extract from multiple texts concurrently

**Key Attributes:**
- `ontology` - The ontology schema used for extraction
- `ontology_scope` - Scope level: "minimal", "balanced", or "comprehensive"

### `OntologyRecommender`
**Module:** `spindle.extraction.recommender`

Automatically recommends ontologies by analyzing sample text and determining appropriate entity and relation types.

**Key Methods:**
- `recommend(text, scope)` - Recommend an ontology from text
- `analyze_extension_need(text, current_ontology)` - Check if ontology needs extension
- `apply_extension(current_ontology, extension)` - Extend an existing ontology

### `GraphStore`
**Module:** `spindle.graph_store.store`

Persistent graph database storage using Kùzu as the backend. Provides CRUD operations for nodes, edges, and queries.

**Key Methods:**
- `add_node(name, entity_type, metadata)` - Add a node to the graph
- `add_edge(subject, predicate, obj, metadata)` - Add an edge
- `add_triples(triples)` - Batch add triples from extraction
- `query_by_pattern(subject, predicate, obj)` - Pattern-based queries
- `query_cypher(query)` - Execute raw Cypher queries

### `EntityResolver`
**Module:** `spindle.entity_resolution.resolver`

Performs semantic entity resolution to identify and deduplicate entities across the graph using embeddings and LLM-based matching.

**Key Methods:**
- `resolve_entities(nodes)` - Resolve node duplicates
- `resolve_edges(edges)` - Resolve edge duplicates

### `ChromaVectorStore`
**Module:** `spindle.vector_store.chroma`

Vector database for storing and searching embeddings, backed by ChromaDB.

**Key Methods:**
- `add_documents(texts, metadatas, ids)` - Add documents with embeddings
- `similarity_search(query, k)` - Find similar documents
- `delete_collection()` - Remove all documents

---

## Data Models & Schemas

### `Triple`
**Module:** `spindle.baml_client.types`

Represents a single knowledge graph triple extracted from text.

**Fields:**
- `subject: Entity` - The subject entity
- `predicate: str` - The relationship type
- `object: Entity` - The object entity
- `source: SourceMetadata` - Source information
- `supporting_spans: List[CharacterSpan]` - Evidence spans
- `extraction_datetime: Optional[str]` - When it was extracted

### `Entity`
**Module:** `spindle.baml_client.types`

An entity extracted from text with type and attributes.

**Fields:**
- `name: str` - Entity name (canonical form)
- `type: str` - Entity type from ontology
- `description: str` - Brief description
- `custom_atts: Dict[str, AttributeValue]` - Domain-specific attributes

### `Ontology`
**Module:** `spindle.baml_client.types`

Schema defining entity and relation types for extraction.

**Fields:**
- `entity_types: List[EntityType]` - Allowed entity types
- `relation_types: List[RelationType]` - Allowed relation types

### `EntityType`
**Module:** `spindle.baml_client.types`

Definition of an entity type in the ontology.

**Fields:**
- `name: str` - Type name (e.g., "Person", "Organization")
- `description: str` - What this type represents
- `attributes: List[AttributeDefinition]` - Allowed attributes

### `RelationType`
**Module:** `spindle.baml_client.types`

Definition of a relation type in the ontology.

**Fields:**
- `name: str` - Relation name (e.g., "works_at", "manages")
- `description: str` - What this relation represents
- `domain: str` - Subject entity type constraint
- `range: str` - Object entity type constraint

### `SourceMetadata`
**Module:** `spindle.baml_client.types`

Metadata about the source of extracted knowledge.

**Fields:**
- `source_name: str` - Name of the source document
- `source_url: Optional[str]` - URL if available

### `CharacterSpan` / `EvidenceSpan`
**Module:** `spindle.baml_client.types`

A text span providing evidence for an extraction.

**Fields:**
- `text: str` - The text content of the span
- `start: Optional[int]` - Start character position
- `end: Optional[int]` - End character position

### `AttributeDefinition`
**Module:** `spindle.baml_client.types`

Definition of an entity attribute in the ontology.

**Fields:**
- `name: str` - Attribute name
- `type: str` - Data type (string, int, date, etc.)
- `description: str` - What this attribute represents

### `AttributeValue`
**Module:** `spindle.baml_client.types`

A value for an entity attribute.

**Fields:**
- `value: Optional[str]` - The attribute value
- `type: str` - The data type

---

## BAML Types

### `ExtractionResult`
**Module:** `spindle.baml_client.types`

Result from the LLM extraction function.

**Fields:**
- `triples: List[Triple]` - Extracted triples
- `reasoning: str` - LLM's reasoning for the extraction

### `OntologyRecommendation`
**Module:** `spindle.baml_client.types`

Result from ontology recommendation.

**Fields:**
- `ontology: Ontology` - The recommended ontology
- `text_purpose: str` - Identified purpose of the text
- `reasoning: str` - Why this ontology was chosen

### `OntologyExtension`
**Module:** `spindle.baml_client.types`

Analysis of whether an ontology needs extension.

**Fields:**
- `needs_extension: bool` - Whether extension is needed
- `new_entity_types: List[EntityType]` - New entities to add
- `new_relation_types: List[RelationType]` - New relations to add
- `critical_information_at_risk: Optional[str]` - Info that would be lost
- `reasoning: str` - Explanation

### `EntityMatchingResult`
**Module:** `spindle.baml_client.types`

Result from entity matching for resolution.

**Fields:**
- `matches: List[EntityMatch]` - Matched entity pairs
- `reasoning: str` - Overall matching reasoning

### `EntityMatch` (BAML)
**Module:** `spindle.baml_client.types`

A single entity match from LLM.

**Fields:**
- `entity1_id: str` - First entity ID
- `entity2_id: str` - Second entity ID
- `is_duplicate: bool` - Whether they're duplicates
- `confidence_level: str` - Confidence assessment
- `reasoning: str` - Why they match/don't match

### `EdgeMatchingResult`
**Module:** `spindle.baml_client.types`

Result from edge matching for resolution.

**Fields:**
- `matches: List[EdgeMatch]` - Matched edge pairs
- `reasoning: str` - Overall matching reasoning

### `EdgeMatch` (BAML)
**Module:** `spindle.baml_client.types`

A single edge match from LLM.

**Fields:**
- `edge1_id: str` - First edge ID
- `edge2_id: str` - Second edge ID
- `is_duplicate: bool` - Whether they're duplicates
- `confidence_level: str` - Confidence assessment
- `reasoning: str` - Why they match/don't match

### `ProcessGraph`
**Module:** `spindle.baml_client.types`

A process workflow extracted from text.

**Fields:**
- `process_name: Optional[str]` - Name of the process
- `scope: Optional[str]` - Scope description
- `primary_goal: str` - Main objective
- `start_step_ids: List[str]` - Entry point steps
- `end_step_ids: List[str]` - Terminal steps
- `steps: List[ProcessStep]` - All process steps
- `dependencies: List[ProcessDependency]` - Step dependencies

### `ProcessStep`
**Module:** `spindle.baml_client.types`

A single step in a process graph.

**Fields:**
- `step_id: str` - Unique identifier
- `title: str` - Step name
- `summary: str` - Description
- `step_type: ProcessStepType` - Type (ACTIVITY, DECISION, EVENT, etc.)
- `actors: List[str]` - Who performs this step
- `inputs: List[str]` - Required inputs
- `outputs: List[str]` - Produced outputs
- `duration: Optional[str]` - Time estimate
- `prerequisites: List[str]` - Required conditions
- `evidence: List[EvidenceSpan]` - Supporting text

### `ProcessDependency`
**Module:** `spindle.baml_client.types`

Dependency between process steps.

**Fields:**
- `from_step: str` - Source step ID
- `to_step: str` - Target step ID
- `relation: str` - Type of dependency
- `condition: Optional[str]` - Conditional logic
- `evidence: List[EvidenceSpan]` - Supporting text

---

## Ingestion System

### `DocumentArtifact`
**Module:** `spindle.ingestion.types`

Represents a source document after loading.

**Fields:**
- `document_id: str` - Unique document identifier
- `source_path: Path` - Original file path
- `checksum: str` - Content hash for deduplication
- `loader_name: str` - Loader used to read the file
- `template_name: str` - Template used for processing
- `metadata: Metadata` - Extracted metadata
- `raw_bytes: Optional[bytes]` - Raw document content
- `created_at: datetime` - Ingestion timestamp

### `ChunkArtifact`
**Module:** `spindle.ingestion.types`

A text chunk from document splitting.

**Fields:**
- `chunk_id: str` - Unique chunk identifier
- `document_id: str` - Parent document ID
- `text: str` - Chunk text content
- `metadata: Metadata` - Chunk-level metadata
- `embedding: Optional[Sequence[float]]` - Vector embedding

### `DocumentGraph`
**Module:** `spindle.ingestion.types`

Graph structure created during document ingestion.

**Fields:**
- `nodes: List[DocumentGraphNode]` - Graph nodes
- `edges: List[DocumentGraphEdge]` - Graph edges

### `DocumentGraphNode`
**Module:** `spindle.ingestion.types`

A node in the document graph.

**Fields:**
- `node_id: str` - Unique node ID
- `document_id: str` - Source document
- `label: str` - Node type label
- `attributes: Metadata` - Node properties

### `DocumentGraphEdge`
**Module:** `spindle.ingestion.types`

An edge in the document graph.

**Fields:**
- `edge_id: str` - Unique edge ID
- `source_id: str` - Source node ID
- `target_id: str` - Target node ID
- `relation: str` - Edge type
- `attributes: Metadata` - Edge properties

### `Corpus`
**Module:** `spindle.ingestion.types`

A collection of related documents for pipeline processing.

**Fields:**
- `corpus_id: str` - Unique corpus identifier
- `name: str` - Corpus name
- `description: str` - Purpose description
- `created_at: datetime` - Creation timestamp
- `updated_at: datetime` - Last modification
- `pipeline_state: Metadata` - Pipeline execution state

### `CorpusDocument`
**Module:** `spindle.ingestion.types`

Links a document to a corpus.

**Fields:**
- `corpus_id: str` - Corpus identifier
- `document_id: str` - Document identifier
- `added_at: datetime` - When added to corpus

### `TemplateSpec`
**Module:** `spindle.ingestion.types`

Configuration for a document ingestion template.

**Fields:**
- `name: str` - Template name
- `selector: TemplateSelector` - Matching criteria
- `loader: str | Callable` - Document loader
- `preprocessors: Sequence[Callable]` - Pre-processing steps
- `splitter: str | Mapping | Callable` - Text splitting strategy
- `metadata_extractors: Sequence[Callable]` - Metadata extractors
- `postprocessors: Sequence[Callable]` - Post-processing steps
- `graph_hooks: Sequence[Callable]` - Graph construction hooks
- `description: Optional[str]` - Template description

### `TemplateSelector`
**Module:** `spindle.ingestion.types`

Criteria for matching documents to templates.

**Fields:**
- `mime_types: Sequence[str]` - MIME type patterns
- `path_globs: Sequence[str]` - File path patterns
- `file_extensions: Sequence[str]` - File extension patterns

### `IngestionConfig`
**Module:** `spindle.ingestion.types`

Configuration for an ingestion run.

**Fields:**
- `template_specs: Sequence[TemplateSpec]` - Available templates
- `template_search_paths: Sequence[Path]` - Template directories
- `catalog_url: Optional[str]` - Document catalog database URL
- `vector_store_uri: Optional[str]` - Vector store location
- `cache_dir: Optional[Path]` - Cache directory
- `allow_network_requests: bool` - Whether to allow network access
- `spindle_config: Optional[SpindleConfig]` - Global Spindle config

### `IngestionResult`
**Module:** `spindle.ingestion.types`

Final result of an ingestion run.

**Fields:**
- `documents: List[DocumentArtifact]` - Processed documents
- `chunks: List[ChunkArtifact]` - Generated chunks
- `document_graph: DocumentGraph` - Document relationships
- `metrics: IngestionRunMetrics` - Performance metrics
- `events: Iterable[IngestionEvent]` - Emitted events

### `IngestionRunMetrics`
**Module:** `spindle.ingestion.types`

Metrics collected during ingestion.

**Fields:**
- `started_at: datetime` - Start timestamp
- `finished_at: Optional[datetime]` - End timestamp
- `processed_documents: int` - Document count
- `processed_chunks: int` - Chunk count
- `bytes_read: int` - Total bytes processed
- `errors: List[str]` - Error messages
- `extra: Metadata` - Additional metrics

### `IngestionEvent`
**Module:** `spindle.ingestion.types`

Event emitted during ingestion for observability.

**Fields:**
- `timestamp: datetime` - Event time
- `name: str` - Event name
- `payload: Metadata` - Event data

### `IngestionContext`
**Module:** `spindle.ingestion.types`

Context passed between pipeline stages.

**Fields:**
- `config: IngestionConfig` - Ingestion configuration
- `active_template: Optional[TemplateSpec]` - Current template
- `run_metadata: Metadata` - Run-level metadata

---

## Ontology Pipeline

The Ontology Pipeline is a six-stage process for building semantic knowledge systems.

### Pipeline Stages

#### 1. Controlled Vocabulary (`PipelineStage.VOCABULARY`)
**Purpose:** Extract and define clean, disambiguated terms.

**Artifact:** `VocabularyTerm`

**Fields:**
- `term_id: str` - Unique term identifier
- `preferred_label: str` - Canonical term label
- `definition: str` - Term definition
- `synonyms: List[str]` - Alternative terms
- `domain: Optional[str]` - Domain context
- `source_document_ids: List[str]` - Source documents
- `usage_count: int` - Frequency in corpus
- `created_at: datetime` - Creation timestamp
- `updated_at: datetime` - Last update

#### 2. Metadata Standards (`PipelineStage.METADATA`)
**Purpose:** Define schema-based controls for data description.

**Artifact:** `MetadataElement`

**Fields:**
- `element_id: str` - Unique element ID
- `name: str` - Element name
- `element_type: MetadataElementType` - STRUCTURAL, DESCRIPTIVE, or ADMINISTRATIVE
- `description: str` - Element description
- `data_type: str` - Data type (string, int, date, etc.)
- `required: bool` - Whether required
- `allowed_values: Optional[List[str]]` - Value constraints
- `default_value: Optional[str]` - Default value
- `examples: List[str]` - Example values
- `created_at: datetime` - Creation timestamp

#### 3. Taxonomy (`PipelineStage.TAXONOMY`)
**Purpose:** Build hierarchical parent-child relationships.

**Artifacts:** `TaxonomyNode`, `TaxonomyRelation`

**TaxonomyNode Fields:**
- `node_id: str` - Unique node ID
- `term_id: str` - References VocabularyTerm
- `label: str` - Node label
- `level: int` - Depth in hierarchy (0 = root)
- `parent_node_id: Optional[str]` - Parent node reference
- `child_count: int` - Number of children
- `created_at: datetime` - Creation timestamp

**TaxonomyRelation Fields:**
- `relation_id: str` - Unique relation ID
- `parent_node_id: str` - Parent node
- `child_node_id: str` - Child node
- `relation_type: str` - "broader" or "narrower"
- `created_at: datetime` - Creation timestamp

#### 4. Thesaurus (`PipelineStage.THESAURUS`)
**Purpose:** Define semantic relationships following ISO 25964/SKOS.

**Artifact:** `ThesaurusEntry`

**Fields:**
- `entry_id: str` - Unique entry ID
- `term_id: str` - References VocabularyTerm
- `preferred_label: str` - Preferred term
- `use_for: List[str]` - Non-preferred synonyms (UF)
- `broader_terms: List[str]` - Parent concepts (BT)
- `narrower_terms: List[str]` - Child concepts (NT)
- `related_terms: List[str]` - Associated concepts (RT)
- `scope_note: Optional[str]` - Definition/clarification (SN)
- `history_note: Optional[str]` - Historical context (HN)
- `created_at: datetime` - Creation timestamp
- `updated_at: datetime` - Last update

#### 5. Ontology (`PipelineStage.ONTOLOGY`)
**Purpose:** Define domain-specific entity and relation types.

**Artifact:** `Ontology` (from BAML types)

#### 6. Knowledge Graph (`PipelineStage.KNOWLEDGE_GRAPH`)
**Purpose:** Synthesize all stages into a queryable knowledge graph.

**Artifact:** Stored in `GraphStore`

### Pipeline Types

### `PipelineState`
**Module:** `spindle.pipeline.types`

Tracks pipeline execution state for a corpus.

**Fields:**
- `corpus_id: str` - Corpus identifier
- `current_stage: Optional[PipelineStage]` - Active stage
- `completed_stages: List[PipelineStage]` - Completed stages
- `stage_results: Dict[str, PipelineStageResult]` - Results by stage
- `started_at: Optional[datetime]` - Pipeline start
- `finished_at: Optional[datetime]` - Pipeline end
- `strategy: ExtractionStrategyType` - Extraction strategy used

### `PipelineStageResult`
**Module:** `spindle.pipeline.types`

Result from executing a pipeline stage.

**Fields:**
- `stage: PipelineStage` - Stage that was executed
- `corpus_id: str` - Corpus identifier
- `success: bool` - Whether stage succeeded
- `started_at: datetime` - Stage start time
- `finished_at: datetime` - Stage end time
- `artifact_count: int` - Number of artifacts produced
- `error_message: Optional[str]` - Error if failed
- `metrics: Dict[str, Any]` - Stage-specific metrics

### Pipeline Classes

### `PipelineOrchestrator`
**Module:** `spindle.pipeline.orchestrator`

Coordinates execution of pipeline stages.

**Key Methods:**
- `register_stage(stage_type, stage_instance)` - Register a stage
- `register_default_stages()` - Register all built-in stages
- `run_stage(corpus, stage)` - Execute a single stage
- `run_all(corpus)` - Execute all stages sequentially

### `BasePipelineStage`
**Module:** `spindle.pipeline.base`

Abstract base class for pipeline stages.

**Key Methods:**
- `extract_from_text(text, document_id)` - Extract artifacts from text
- `merge_artifacts(artifacts)` - Merge/deduplicate artifacts
- `persist_artifacts(corpus_id, artifacts)` - Save to storage
- `load_artifacts(corpus_id)` - Load from storage

### Stage Implementations

- `VocabularyStage` - Extracts vocabulary terms
- `MetadataStage` - Extracts metadata elements
- `TaxonomyStage` - Builds taxonomy hierarchies
- `ThesaurusStage` - Creates thesaurus entries
- `OntologyStage` - Generates ontology schemas
- `KnowledgeGraphStage` - Builds knowledge graph

### Extraction Strategies

### `ExtractionStrategy`
**Module:** `spindle.pipeline.strategies`

Abstract base for extraction strategies.

**Implementations:**
- `SequentialStrategy` - Process documents one at a time
- `BatchConsolidateStrategy` - Process in batches, consolidate results
- `SampleBasedStrategy` - Extract from representative samples

---

## Entity Resolution

### `EntityResolver`
**Module:** `spindle.entity_resolution.resolver`

Main class for entity resolution pipeline.

### `SemanticBlocker`
**Module:** `spindle.entity_resolution.blocking`

Creates candidate blocks using embedding-based clustering.

**Key Methods:**
- `create_blocks(nodes)` - Cluster similar entities

### `SemanticMatcher`
**Module:** `spindle.entity_resolution.matching`

Uses LLM to match entities within blocks.

**Key Methods:**
- `match_nodes(node_pairs, context)` - Match node pairs
- `match_edges(edge_pairs, context)` - Match edge pairs

### `ResolutionConfig`
**Module:** `spindle.entity_resolution.config`

Configuration for resolution pipeline.

**Fields:**
- `blocking_threshold: float` - Cosine similarity threshold (0-1)
- `matching_threshold: float` - LLM confidence threshold (0-1)
- `clustering_method: str` - Clustering algorithm
- `batch_size: int` - Entities per LLM call
- `merge_strategy: str` - How to handle duplicates
- `max_cluster_size: int` - Max entities in a cluster
- `min_cluster_size: int` - Min entities in a cluster

### `ResolutionResult`
**Module:** `spindle.entity_resolution.models`

Results from entity resolution.

**Fields:**
- `total_nodes_processed: int` - Node count
- `total_edges_processed: int` - Edge count
- `blocks_created: int` - Clustering blocks
- `same_as_edges_created: int` - SAME_AS relationships added
- `duplicate_clusters: int` - Connected components
- `node_matches: List[EntityMatch]` - Node match results
- `edge_matches: List[EdgeMatch]` - Edge match results
- `execution_time_seconds: float` - Total time
- `config: Optional[ResolutionConfig]` - Config used

### `EntityMatch` (Resolution)
**Module:** `spindle.entity_resolution.models`

A matched entity pair.

**Fields:**
- `entity1_id: str` - First entity
- `entity2_id: str` - Second entity
- `is_duplicate: bool` - Match result
- `confidence: float` - Confidence score (0-1)
- `reasoning: str` - LLM explanation

### `EdgeMatch` (Resolution)
**Module:** `spindle.entity_resolution.models`

A matched edge pair.

**Fields:**
- `edge1_id: str` - First edge
- `edge2_id: str` - Second edge
- `is_duplicate: bool` - Match result
- `confidence: float` - Confidence score (0-1)
- `reasoning: str` - LLM explanation

---

## Graph Store

### `GraphStore`
**Module:** `spindle.graph_store.store`

Facade class for graph database operations.

### `GraphStoreBackend`
**Module:** `spindle.graph_store.base`

Abstract base class for backend implementations.

**Implementations:**
- `KuzuBackend` - Kùzu embedded graph database

### Backend Operations

**Node Operations:**
- `add_node(name, entity_type, metadata)` - Add a node
- `get_node(name)` - Retrieve a node
- `nodes()` - Get all nodes
- `update_node(name, updates)` - Update node properties
- `delete_node(name)` - Delete a node

**Edge Operations:**
- `add_edge(subject, predicate, obj, metadata)` - Add an edge
- `get_edge(subject, predicate, obj)` - Retrieve an edge
- `edges()` - Get all edges
- `update_edge(subject, predicate, obj, updates)` - Update edge
- `delete_edge(subject, predicate, obj)` - Delete an edge

**Query Operations:**
- `query_by_pattern(subject, predicate, obj)` - Pattern matching
- `query_by_source(source_name)` - Filter by source
- `query_by_date_range(start, end)` - Filter by date
- `query_cypher(query)` - Execute Cypher query
- `get_statistics()` - Get graph statistics

---

## Vector Store

### `VectorStore`
**Module:** `spindle.vector_store.base`

Abstract base class for vector store implementations.

### `ChromaVectorStore`
**Module:** `spindle.vector_store.chroma`

ChromaDB-based vector store implementation.

**Key Methods:**
- `add_documents(texts, metadatas, ids)` - Add documents
- `similarity_search(query, k)` - Find similar documents
- `delete_collection()` - Clear all documents

### `GraphEmbeddingGenerator`
**Module:** `spindle.vector_store.graph_embeddings`

Generates embeddings for graph nodes using Node2Vec.

**Key Methods:**
- `generate_embeddings(graph_store, dimensions)` - Generate embeddings
- `store_embeddings(graph_store, vector_store)` - Store in vector store

### Embedding Functions

**Module:** `spindle.vector_store.embeddings`

- `create_openai_embedding_function()` - OpenAI embeddings
- `create_huggingface_embedding_function()` - HuggingFace embeddings
- `create_gemini_embedding_function()` - Google Gemini embeddings
- `get_default_embedding_function()` - Auto-select best available

---

## Analytics & Observability

### `ServiceEvent`
**Module:** `spindle.observability.events`

Event emitted from Spindle services.

**Fields:**
- `timestamp: datetime` - Event timestamp
- `service: str` - Service namespace (e.g., "extraction.extractor")
- `name: str` - Event name (e.g., "extract.start")
- `payload: Metadata` - Event data

### `EventRecorder`
**Module:** `spindle.observability.events`

Central dispatcher for service events.

**Key Methods:**
- `record(name, payload)` - Record an event
- `emit(event)` - Forward an existing event
- `register(observer)` - Add event observer
- `scoped(service)` - Create scoped child recorder

### `EventLogStore`
**Module:** `spindle.observability.storage`

SQLite-based persistence for service events.

**Key Methods:**
- `store_event(event)` - Persist an event
- `replay_events(service_filter)` - Retrieve events
- `attach_observer(recorder)` - Auto-persist events

### Analytics Schema

**Module:** `spindle.analytics.schema`

### `DocumentObservation`
Top-level analytics record for ingested documents.

**Fields:**
- `schema_version: str` - Schema version
- `metadata: DocumentMetadata` - Document identifiers
- `structural: StructuralMetrics` - Structural metrics
- `chunk_windows: List[ChunkWindowSummary]` - Chunk analysis
- `segments: Optional[SemanticSegmentSummary]` - Segment analysis
- `ontology: Optional[OntologySignal]` - Ontology signals
- `context: Optional[ContextWindowAssessment]` - Context recommendations
- `observability: ObservabilitySignals` - Event logs

### `DocumentMetadata`
Document identification metadata.

**Fields:**
- `document_id: str` - Unique ID
- `source_uri: Optional[str]` - Source location
- `source_type: SourceType` - FILE, URL, API, STREAM, OTHER
- `content_type: Optional[str]` - MIME type
- `language: Optional[str]` - ISO-639 language code
- `ingested_at: datetime` - Ingestion time
- `hash_signature: Optional[str]` - Content hash

### `StructuralMetrics`
Aggregate structural statistics.

**Fields:**
- `token_count: int` - Total tokens
- `character_count: Optional[int]` - Character count
- `page_count: Optional[int]` - Page count
- `section_count: Optional[int]` - Section count
- `average_tokens_per_section: Optional[float]` - Avg tokens/section
- `chunk_count: int` - Number of chunks
- `chunk_token_summary: QuantileSummary` - Chunk token distribution

### `QuantileSummary`
Descriptive statistics for numeric distributions.

**Fields:**
- `minimum: float` - Min value
- `maximum: float` - Max value
- `median: Optional[float]` - Median value
- `mean: Optional[float]` - Mean value
- `p95: Optional[float]` - 95th percentile

### `ChunkWindowSummary`
Statistics over sliding windows of chunks.

**Fields:**
- `window_size: int` - Window size
- `token_summary: QuantileSummary` - Token distribution
- `overlap_tokens: Optional[int]` - Overlap tokens
- `overlap_ratio: Optional[float]` - Overlap percentage
- `cross_chunk_link_rate: Optional[float]` - Cross-chunk reference rate
- `context_limit_risk: RiskLevel` - Risk assessment

### `SemanticSegmentSummary`
Analysis of semantic segments.

**Fields:**
- `segment_boundaries: List[int]` - Segment start positions
- `segment_token_summary: Optional[QuantileSummary]` - Segment sizes
- `embedding_dispersion: Optional[float]` - Embedding variance
- `topic_transition_score: Optional[float]` - Topic drift measure

### `OntologySignal`
Signals for ontology recommendation.

**Fields:**
- `ontology_candidate_terms: List[str]` - Candidate terms
- `coverage_estimate: Optional[float]` - Ontology coverage
- `graph_density_estimate: Optional[float]` - Predicted graph density

### `ContextWindowAssessment`
Recommendation for processing window size.

**Fields:**
- `recommended_strategy: ContextStrategy` - CHUNK, WINDOW, SEGMENT, or DOCUMENT
- `supporting_risk: RiskLevel` - Risk level
- `estimated_token_usage: Optional[int]` - Estimated tokens
- `target_token_budget: Optional[int]` - Token budget

---

## Configuration

### `SpindleConfig`
**Module:** `spindle.configuration`

Root configuration for Spindle.

**Fields:**
- `storage: StoragePaths` - Storage locations
- `templates: TemplateSettings` - Template search paths
- `extras: Mapping[str, Any]` - User-defined metadata
- `observability: ObservabilitySettings` - Logging config
- `ingestion: IngestionSettings` - Ingestion defaults
- `vector_store: VectorStoreSettings` - Vector store config
- `graph_store: GraphStoreSettings` - Graph store config
- `llm: Optional[LLMConfig]` - LLM configuration

**Key Methods:**
- `with_root(root)` - Create config from root directory
- `create_extractor()` - Create configured SpindleExtractor
- `create_recommender()` - Create configured OntologyRecommender
- `get_llm_config()` - Get LLM configuration

### `StoragePaths`
**Module:** `spindle.configuration`

Filesystem locations for Spindle storage.

**Fields:**
- `root: Path` - Root storage directory
- `vector_store_dir: Path` - Vector store location
- `graph_store_path: Path` - Graph database file
- `document_store_dir: Path` - Document cache
- `log_dir: Path` - Log directory
- `catalog_path: Path` - Ingestion catalog database
- `template_root: Optional[Path]` - Template directory

### `TemplateSettings`
**Module:** `spindle.configuration`

Template registry configuration.

**Fields:**
- `search_paths: tuple[Path, ...]` - Template search paths

### `ObservabilitySettings`
**Module:** `spindle.configuration`

Observability and logging configuration.

**Fields:**
- `event_log_url: Optional[str]` - Event log database URL
- `log_level: str` - Logging level
- `enable_pipeline_events: bool` - Enable pipeline event logging

### `IngestionSettings`
**Module:** `spindle.configuration`

Default ingestion settings.

**Fields:**
- `catalog_url: Optional[str]` - Catalog database URL
- `vector_store_uri: Optional[str]` - Vector store location
- `cache_dir: Optional[Path]` - Cache directory
- `allow_network_requests: bool` - Allow network access
- `recursive: bool` - Recursive directory ingestion

### `VectorStoreSettings`
**Module:** `spindle.configuration`

Vector store preferences.

**Fields:**
- `collection_name: str` - Collection name
- `embedding_model: Optional[str]` - Embedding model name
- `use_api_fallback: bool` - Fall back to API embeddings
- `prefer_local_embeddings: bool` - Prefer local embeddings

### `GraphStoreSettings`
**Module:** `spindle.configuration`

Graph store persistence settings.

**Fields:**
- `db_path_override: Optional[Path]` - Override database path
- `auto_snapshot: bool` - Enable automatic snapshots
- `snapshot_dir: Optional[Path]` - Snapshot directory
- `embedding_dimensions: int` - Embedding dimensions
- `auto_compute_embeddings: bool` - Auto-compute graph embeddings

### `LLMConfig`
**Module:** `spindle.llm_config`

LLM authentication and configuration.

**Fields:**
- `gcp_project_id: Optional[str]` - GCP project ID
- `gcp_region: Optional[str]` - GCP region
- `anthropic_api_key: Optional[str]` - Anthropic API key
- `openai_api_key: Optional[str]` - OpenAI API key
- `google_api_key: Optional[str]` - Google API key
- `preferred_auth_method: Optional[AuthMethod]` - Preferred auth
- `available_auth_methods: List[AuthMethod]` - Available methods
- `gcp_credentials_path: Optional[str]` - GCP credentials file

---

## API Models

REST API request/response models (FastAPI/Pydantic).

### Session Models

- `SessionCreate` - Create session request
- `SessionInfo` - Session information response
- `SessionUpdate` - Update session request
- `OntologyUpdate` - Update ontology request

### Ingestion Models

- `IngestionRequest` - Stateless ingestion request
- `IngestionSessionRequest` - Session ingestion request
- `DocumentInfo` - Document information
- `ChunkInfo` - Chunk information
- `IngestionMetrics` - Ingestion metrics
- `IngestionResponse` - Ingestion response
- `IngestionStreamChunk` - Streaming progress chunk

### Extraction Models

- `ExtractionRequest` - Extract triples request
- `BatchExtractionRequest` - Batch extraction request
- `SessionExtractionRequest` - Session extraction request
- `ExtractionResponse` - Extraction response
- `BatchExtractionResponse` - Batch extraction response

### Ontology Models

- `OntologyRecommendationRequest` - Recommend ontology request
- `OntologyRecommendationResponse` - Recommendation response
- `OntologyExtensionAnalysisRequest` - Analyze extension request
- `OntologyExtensionAnalysisResponse` - Extension analysis response
- `OntologyExtensionApplyRequest` - Apply extension request
- `OntologyExtensionApplyResponse` - Apply extension response
- `RecommendAndExtractRequest` - Combined recommend+extract request
- `RecommendAndExtractResponse` - Combined response

### Resolution Models

- `ResolutionRequest` - Stateless resolution request
- `ResolutionSessionRequest` - Session resolution request
- `ResolutionResponse` - Resolution results response

### Process Models

- `ProcessExtractionRequest` - Extract process graph request
- `ProcessStepInfo` - Process step information
- `ProcessDependencyInfo` - Process dependency information
- `ProcessGraphInfo` - Process graph information
- `ProcessExtractionResponse` - Process extraction response

### Common Models

- `ErrorResponse` - Standard error response
- `HealthResponse` - Health check response

---

## Enumerations

### `PipelineStage`
**Module:** `spindle.pipeline.types`

Pipeline stage enumeration.

**Values:**
- `VOCABULARY` - Controlled vocabulary stage
- `METADATA` - Metadata standards stage
- `TAXONOMY` - Taxonomy hierarchy stage
- `THESAURUS` - Thesaurus relationships stage
- `ONTOLOGY` - Ontology definition stage
- `KNOWLEDGE_GRAPH` - Knowledge graph synthesis stage

### `ExtractionStrategyType`
**Module:** `spindle.pipeline.types`

Extraction strategy enumeration.

**Values:**
- `SEQUENTIAL` - Process documents sequentially
- `BATCH_CONSOLIDATE` - Process in batches with consolidation
- `SAMPLE_BASED` - Extract from representative samples

### `MetadataElementType`
**Module:** `spindle.pipeline.types`

Metadata element type classification.

**Values:**
- `STRUCTURAL` - Machine readability (format, encoding)
- `DESCRIPTIVE` - Context (title, author, subject)
- `ADMINISTRATIVE` - Maintenance and lineage

### `ThesaurusRelationType`
**Module:** `spindle.pipeline.types`

Thesaurus relationship types (ISO 25964/SKOS).

**Values:**
- `USE` - Preferred term indicator
- `USE_FOR` (UF) - Non-preferred term indicator
- `BROADER_TERM` (BT) - Hierarchical broader concept
- `NARROWER_TERM` (NT) - Hierarchical narrower concept
- `RELATED_TERM` (RT) - Associative/related concept
- `SCOPE_NOTE` (SN) - Definition or scope clarification
- `HISTORY_NOTE` (HN) - Historical context

### `ProcessStepType`
**Module:** `spindle.baml_client.types`

Process step type enumeration.

**Values:**
- `ACTIVITY` - Action or task
- `DECISION` - Decision point
- `EVENT` - Event or trigger
- `PARALLEL_GATEWAY` - Parallel execution split/join
- `SUBPROCESS` - Nested subprocess

### `RiskLevel`
**Module:** `spindle.analytics.schema`

Risk assessment levels.

**Values:**
- `LOW` - Low risk
- `MEDIUM` - Medium risk
- `HIGH` - High risk

### `ContextStrategy`
**Module:** `spindle.analytics.schema`

Recommended context granularity.

**Values:**
- `CHUNK` - Single chunk context
- `WINDOW` - Sliding window context
- `SEGMENT` - Semantic segment context
- `DOCUMENT` - Full document context

### `SourceType`
**Module:** `spindle.analytics.schema`

Document source types.

**Values:**
- `FILE` - Local file
- `URL` - Web URL
- `API` - API endpoint
- `STREAM` - Streaming source
- `OTHER` - Other source type

### `OntologyScope`
**Module:** `spindle.api.models`

Ontology recommendation scope levels.

**Values:**
- `MINIMAL` - Minimal ontology (fewest types)
- `BALANCED` - Balanced ontology (recommended default)
- `COMPREHENSIVE` - Comprehensive ontology (most detailed)

### `AuthMethod`
**Module:** `spindle.llm_config`

LLM authentication methods.

**Values:**
- `VERTEX_AI` - GCP Vertex AI (Anthropic/Gemini)
- `DIRECT_API` - Direct API keys
- `VERTEX_MAAS` - Vertex AI Model-as-a-Service

### `Provider`
**Module:** `spindle.llm_config`

LLM providers.

**Values:**
- `ANTHROPIC` - Anthropic (Claude)
- `OPENAI` - OpenAI (GPT)
- `GOOGLE` - Google (Gemini)

---

## CLI Commands

### `spindle-ingest`
**Module:** `spindle.ingestion.cli`

Document ingestion CLI.

**Subcommands:**
- `config init` - Initialize configuration file
  - `--output PATH` - Output path for config.py

**Main command:**
- `spindle-ingest [options] inputs...` - Ingest documents

**Options:**
- `--config PATH` - Path to config.py file
- `inputs` - Files or directories to ingest
- `--templates DIR [DIR...]` - Template search directories
- `--catalog-url URL` - Document catalog database URL
- `--vector-store PATH` - Vector store directory
- `--recursive` - Recurse into directories
- `--no-vector-store` - Disable vector store
- `--no-catalog` - Disable catalog persistence
- `--event-log URL` - Event log database URL

### `spindle-dashboard`
**Module:** `spindle.dashboard.app`

Launch analytics dashboard.

**Options:**
- `--database URL` - Analytics database URL
- `--port PORT` - Server port (default: 8501)

### `spindle-pipeline`
**Module:** `spindle.pipeline.cli`

Ontology pipeline CLI (future implementation).

**Planned commands:**
- `run` - Run pipeline stages
- `status` - Check pipeline status
- `reset` - Reset pipeline state

---

## Utility Functions

### Extraction Utilities

**Module:** `spindle.extraction.utils`

- `create_ontology(entity_types, relation_types)` - Create Ontology object
- `create_source_metadata(source_name, source_url)` - Create SourceMetadata
- `triples_to_dict(triples)` - Serialize triples to dictionary
- `dict_to_triples(data)` - Deserialize triples from dictionary
- `ontology_to_dict(ontology)` - Serialize ontology
- `recommendation_to_dict(recommendation)` - Serialize recommendation
- `extension_to_dict(extension)` - Serialize extension
- `get_supporting_text(triples)` - Extract all supporting text
- `filter_triples_by_source(triples, source_name)` - Filter by source
- `parse_extraction_datetime(date_string)` - Parse extraction datetime
- `filter_triples_by_date_range(triples, start, end)` - Filter by date

### Entity Resolution Utilities

**Module:** `spindle.entity_resolution`

- `resolve_entities(nodes, config)` - Resolve entity duplicates
- `create_same_as_edges(matches)` - Create SAME_AS edges from matches
- `create_same_as_edges_for_edges(matches)` - Create SAME_AS for edges
- `get_duplicate_clusters(same_as_edges)` - Get connected components
- `find_connected_components(same_as_edges)` - Find duplicate clusters
- `serialize_node_for_embedding(node)` - Serialize node for embedding
- `serialize_edge_for_embedding(edge)` - Serialize edge for embedding
- `merge_node_metadata(nodes)` - Merge metadata from duplicate nodes
- `merge_edge_metadata(edges)` - Merge metadata from duplicate edges

### Configuration Utilities

**Module:** `spindle.configuration`

- `default_config(root)` - Create default configuration
- `render_default_config(root)` - Render config.py template
- `load_config_from_file(path)` - Load SpindleConfig from file

### LLM Configuration Utilities

**Module:** `spindle.llm_config`

- `detect_available_auth()` - Auto-detect available authentication
- `from_env()` - Create LLMConfig from environment variables

---

## Storage & Persistence

### Document Catalog

**Module:** `spindle.ingestion.storage.catalog`

**Class:** `DocumentCatalog`

SQLite-based document catalog for tracking ingested documents.

### Corpus Manager

**Module:** `spindle.ingestion.storage.corpus`

**Class:** `CorpusManager`

Manages corpus collections and their relationships to documents.

### Vector Storage Manager

**Module:** `spindle.ingestion.storage.vector`

**Class:** `VectorStoreManager`

Manages vector store operations during ingestion.

---

## Process Extraction

### Process Extraction Result

**Module:** `spindle.baml_client.types`

**Class:** `ProcessExtractionResult`

Result from process graph extraction.

**Fields:**
- `status: Union["process_found", "no_process", "incomplete"]` - Extraction status
- `graph: Optional[ProcessGraph]` - Extracted process graph
- `reasoning: str` - Extraction reasoning
- `issues: List[ProcessExtractionIssue]` - Extraction issues

### Process Extraction Issue

**Module:** `spindle.baml_client.types`

**Class:** `ProcessExtractionIssue`

Issue encountered during process extraction.

**Fields:**
- `code: str` - Issue code
- `message: str` - Issue description
- `related_step_ids: List[str]` - Affected steps

---

## BAML Functions

BAML-defined LLM functions for extraction and recommendation.

### Extraction Functions

**Module:** `spindle.baml_src.spindle.baml`

- `ExtractTriples` - Extract knowledge graph triples from text
- `RecommendOntology` - Recommend an ontology for text
- `AnalyzeExtensionNeed` - Analyze if ontology needs extension
- `EnhanceOntology` - Enhance/extend an existing ontology

### Entity Resolution Functions

**Module:** `spindle.baml_src.entity_resolution.baml`

- `MatchEntities` - Match entity pairs for deduplication
- `MatchEdges` - Match edge pairs for deduplication

### Process Extraction Functions

**Module:** `spindle.baml_src.process.baml`

- `ExtractProcess` - Extract process graph from text

### Pipeline Functions

**Module:** `spindle.baml_src.pipeline.baml`

- `ExtractVocabulary` - Extract vocabulary terms
- `ExtractMetadata` - Extract metadata elements
- `ExtractTaxonomy` - Extract taxonomy relationships
- `ExtractThesaurus` - Extract thesaurus entries

---

## Version Information

**Current Version:** `0.1.0`

**Module:** `spindle.__init__`

---

## Additional Resources

- **Documentation:** `/docs/` directory
- **Examples:** `/demos/` directory  
- **Tests:** `/tests/` directory
- **Configuration:** `docs/CONFIGURATION.md`
- **Testing Guide:** `docs/TESTING.md`
- **Quick Start:** `docs/QUICKSTART.md`

---

*This glossary covers all major classes, types, schemas, and artifacts in the Spindle knowledge graph extraction system. For detailed API documentation, see the module docstrings and inline code documentation.*

