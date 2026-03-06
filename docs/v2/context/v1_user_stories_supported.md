## User Stories Supported by Spindle

### 1. Knowledge Graph Extraction

**US-1: Extract triples from text**
- Extract knowledge graph triples (subject-predicate-object) from unstructured text
- Auto-recommend ontology on first extraction if none provided
- Support manual ontology definition
- Track source metadata (name, URL) and extraction timestamps
- Include supporting evidence spans with character positions

**US-2: Batch extraction**
- Extract triples from multiple texts in batch
- Maintain entity consistency across batch extractions
- Support concurrent processing with configurable concurrency

**US-3: Streaming extraction**
- Stream extraction results as they complete (Server-Sent Events)
- Real-time progress for batch operations

**US-4: Session-based extraction**
- Maintain session state with accumulated triples and ontology
- Extract incrementally within a session context
- Preserve entity consistency across session extractions

### 2. Ontology Management

**US-5: Auto-recommend ontology**
- Analyze text and recommend appropriate ontology (entity types, relation types)
- Control granularity via scope: `minimal`, `balanced`, `comprehensive`
- Provide reasoning for recommendations

**US-6: Ontology extension**
- Analyze if existing ontology needs extension for new text
- Conservatively identify missing entity/relation types
- Apply extensions to create extended ontology

**US-7: Combined recommend-and-extract**
- Recommend ontology and extract triples in one operation
- Convenience operation for quick workflows

### 3. Process Extraction

**US-8: Extract process graphs**
- Extract process DAGs (directed acyclic graphs) from procedural text
- Capture process steps, dependencies, actors, resources
- Support incremental extension of existing process graphs
- Identify process issues and validation problems

### 4. Graph Storage & Persistence

**US-9: Store triples in graph database**
- Persistent storage using embedded Kùzu database
- Multi-source evidence consolidation (same fact from multiple sources)
- Case normalization (uppercase) for consistency
- Full CRUD operations on nodes and edges

**US-10: Query graph database**
- Pattern matching queries (subject, predicate, object)
- Query by source document
- Query by date range
- Direct Cypher query support
- Query with/without entity resolution

**US-11: Graph statistics and analytics**
- Get node/edge counts, sources, predicates
- Date range analysis
- Graph growth tracking

### 5. Entity Resolution

**US-12: Deduplicate entities**
- Semantic blocking using embeddings to reduce comparisons
- LLM-based matching with confidence scores
- Create SAME_AS edges to link duplicates
- Preserve provenance (original entities not merged)

**US-13: Query resolved graphs**
- Get duplicate clusters
- Get canonical entity names
- Query with automatic duplicate resolution

### 6. Document Ingestion Pipeline

**US-14: Ingest documents**
- Process documents from filesystem paths
- Template-based document processing
- Support multiple file formats (via loaders)
- Chunking strategies for large documents
- Metadata extraction

**US-15: Streaming ingestion**
- Stream ingestion progress events (SSE)
- Real-time updates during processing

**US-16: Session-based ingestion**
- Ingest documents into a session
- Use session configuration for storage paths

### 7. Ontology Pipeline (Multi-Stage Extraction)

**US-17: Run ontology pipeline stages**
- Vocabulary extraction
- Metadata extraction
- Taxonomy building
- Thesaurus construction
- Ontology generation
- Knowledge graph extraction

**US-18: Pipeline orchestration**
- Run individual stages or all stages
- Track pipeline state and progress
- Support different extraction strategies (sequential, batch_consolidate, sample_based)

**US-19: Access pipeline artifacts**
- Retrieve vocabulary terms
- Retrieve taxonomy nodes
- Retrieve thesaurus entries
- Retrieve generated ontology
- Query knowledge graph statistics

### 8. Corpus Management

**US-20: Manage document corpora**
- Create, list, update, delete corpora
- Add/remove documents from corpora
- Track corpus document counts
- Manage pipeline state per corpus

### 9. Vector Store Integration

**US-21: Semantic search**
- ChromaDB-based vector storage
- Multiple embedding providers (OpenAI, HuggingFace, Google)
- Semantic search over extracted content
- Graph structure-aware embeddings (Node2Vec)

### 10. Analytics & Observability

**US-22: View analytics dashboard**
- Streamlit-based dashboard
- Visualize ingestion metrics
- View extraction statistics
- Monitor entity resolution results
- Track LLM usage and costs

**US-23: Event logging**
- Structured event logging across ingestion, extraction, storage
- Optional SQLite persistence
- Service event tracking
- Latency breakdowns

### 11. Configuration Management

**US-24: Unified configuration**
- Scaffold configuration with `spindle-ingest config init`
- Customize storage paths, graph DB, vector store
- Configure observability settings
- Load configuration programmatically

### 12. API Access

**US-25: REST API access**
- FastAPI-based REST API
- Stateless and stateful (session-based) modes
- Streaming support (SSE)
- Interactive API documentation (Swagger/ReDoc)
- All core services exposed via API

### 13. Programmatic Utilities

**US-26: Helper functions**
- Filter triples by source
- Parse extraction datetimes
- Convert triples to/from dictionaries
- Get supporting text spans
- Serialization utilities

---

## Summary by User Type

**Data Scientist/Researcher:**
- Extract knowledge graphs from text (US-1, US-2, US-3)
- Auto-recommend ontologies (US-5)
- Store and query graphs (US-9, US-10)
- Resolve entity duplicates (US-12, US-13)

**Knowledge Engineer:**
- Define custom ontologies (US-1)
- Extend ontologies incrementally (US-6)
- Run multi-stage ontology pipeline (US-17, US-18)
- Manage document corpora (US-20)

**Software Developer:**
- REST API integration (US-25)
- Programmatic extraction (US-1, US-2)
- Session management (US-4, US-16)
- Configuration management (US-24)

**Process Analyst:**
- Extract process graphs (US-8)
- Analyze workflows and dependencies

**Operations/DevOps:**
- Monitor analytics dashboard (US-22)
- Track observability events (US-23)
- Configure storage and services (US-24)

**End User (via CLI):**
- Document ingestion pipeline (US-14, US-15)
- Configuration scaffolding (US-24)
- Dashboard viewing (US-22)

---

This covers the current capabilities. The redesign document (`SPINDLE_REDESIGN.md`) outlines future enhancements (governance, versioning, continuous learning) that are not yet implemented.