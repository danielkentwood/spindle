# GraphStore: Persistent Graph Database for Spindle

## Overview

GraphStore provides persistent storage and querying capabilities for Spindle knowledge graphs using the Kùzu embedded graph database. It enables you to store extracted triples, query them efficiently, and maintain provenance information across multiple sources.

## Key Features

- **Embedded Database**: No separate server needed - Kùzu runs in-process like SQLite
- **Full CRUD Operations**: Create, read, update, and delete nodes and edges
- **Case Normalization**: All node names and edge predicates automatically converted to UPPERCASE for consistency
- **Case-Insensitive Queries**: Query with any case - automatically converted to uppercase
- **Multi-Source Evidence Consolidation**: Duplicate triples from different sources merge into single edges
- **Intelligent Deduplication**: Automatically detects and merges evidence from multiple sources
- **Pattern Matching**: Query with wildcards for flexible graph exploration
- **Source Tracking**: Filter triples by their source document
- **Temporal Queries**: Filter by extraction date ranges
- **Cypher Support**: Execute direct Cypher queries for advanced use cases
- **Triple Fidelity**: Full roundtrip support - export triples exactly as they were imported
- **Context Manager**: Clean resource management with Python's `with` statement

## Evidence Structure

GraphStore uses a **nested evidence structure** that consolidates facts from multiple sources into single edges. This enables cross-source validation and evidence aggregation.

### Structure Overview

Each edge stores evidence grouped by source:

```python
{
  'subject': 'Alice Johnson',
  'predicate': 'works_at',
  'object': 'TechCorp',
  'supporting_evidence': [
    {
      'source_nm': 'Company Directory',
      'source_url': 'https://directory.example.com',
      'spans': [
        {
          'text': 'Alice Johnson works at TechCorp',
          'start': 0,
          'end': 31,
          'extraction_datetime': '2024-01-15T10:00:00Z'
        },
        {
          'text': 'Alice is a senior engineer at TechCorp',
          'start': 50,
          'end': 89,
          'extraction_datetime': '2024-01-15T10:05:00Z'
        }
      ]
    },
    {
      'source_nm': 'HR Database',
      'source_url': 'https://hr.example.com',
      'spans': [
        {
          'text': 'Alice Johnson - Employee at TechCorp',
          'start': 0,
          'end': 36,
          'extraction_datetime': '2024-01-16T09:30:00Z'
        }
      ]
    }
  ]
}
```

### Key Concepts

- **Source Grouping**: All evidence from the same source is grouped together
- **Span-Level Timestamps**: Each text span has its own `extraction_datetime`
- **Automatic Merging**: When you add the same fact from different sources, they consolidate into one edge
- **Cross-Validation**: Multiple sources supporting the same fact increases confidence

### Deduplication Behavior

When adding evidence to existing edges:

1. **Same source + same span text** → Skipped (already exists)
2. **Same source + different span** → New span added to existing source
3. **Different source** → New source entry created

Example:

```python
# First extraction from Company Directory
store.add_triples([
    Triple(subject="Alice", predicate="works_at", object="TechCorp",
           source=SourceMetadata(source_name="Company Directory"),
           supporting_spans=[...])
])

# Second extraction from HR Database with same fact
store.add_triples([
    Triple(subject="Alice", predicate="works_at", object="TechCorp",
           source=SourceMetadata(source_name="HR Database"),
           supporting_spans=[...])
])

# Result: ONE edge with evidence from BOTH sources
edges = store.get_edge("Alice", "works_at", "TechCorp")
assert len(edges) == 1
assert len(edges[0]['supporting_evidence']) == 2  # Two sources
```

## Case Normalization

**Important:** GraphStore automatically converts all node names and edge predicates (relationship types) to UPPERCASE for consistency and improved matching across different sources.

### Why Uppercase?

- **Consistency**: Ensures "Alice Johnson", "alice johnson", and "ALICE JOHNSON" all refer to the same entity
- **Deduplication**: Better merging of facts from different sources with inconsistent casing
- **Query Reliability**: Eliminates case-sensitivity issues in pattern matching

### How It Works

```python
# Input with mixed case
store.add_node(name="Alice Johnson", entity_type="Person")
store.add_edge(subject="Alice Johnson", predicate="works_at", obj="TechCorp")

# Stored as uppercase
node = store.get_node("alice johnson")  # Case-insensitive lookup
print(node["name"])  # Output: "ALICE JOHNSON"

edges = store.query_by_pattern(predicate="Works_At")  # Case-insensitive
print(edges[0]["predicate"])  # Output: "WORKS_AT"
```

### What Gets Converted

✅ **Converted to uppercase:**

- Node names (entity identifiers)
- Edge predicates (relationship types)

❌ **NOT converted:**

- Entity types (e.g., "Person", "Organization")
- Source names
- Metadata values
- Supporting evidence text

### Query Examples

All queries are case-insensitive - you can use any case when querying:

```python
# All of these work identically:
store.get_node("alice johnson")
store.get_node("Alice Johnson")
store.get_node("ALICE JOHNSON")

# All return the same results:
store.query_by_pattern(predicate="works_at")
store.query_by_pattern(predicate="Works_At")
store.query_by_pattern(predicate="WORKS_AT")
```

### Direct Cypher Queries

When writing direct Cypher queries, remember to use uppercase for names and predicates:

```python
# Correct - use uppercase in Cypher queries
query = """
MATCH (p:Entity {name: 'ALICE JOHNSON'})-[r:Relationship {predicate: 'WORKS_AT'}]->(c:Entity)
RETURN p.name, c.name
"""
results = store.query_cypher(query)
```

## Installation

GraphStore requires the Kùzu Python package:

```bash
uv pip install kuzu>=0.7.0
```

Or install all Spindle dependencies including GraphStore support:

```bash
uv pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from spindle import SpindleExtractor, create_ontology, GraphStore

# Create extractor and extract triples
entity_types = [
    {"name": "Person", "description": "A human being"},
    {"name": "Organization", "description": "A company"}
]
relation_types = [
    {
        "name": "works_at",
        "description": "Employment relationship",
        "domain": "Person",
        "range": "Organization"
    }
]

ontology = create_ontology(entity_types, relation_types)
extractor = SpindleExtractor(ontology)

text = "Alice Johnson works at TechCorp in San Francisco."
result = extractor.extract(text, source_name="Company Directory")

# Store in graph database
with GraphStore() as store:
    # Add triples
    store.add_triples(result.triples)
    
    # Query by pattern
    employees = store.query_by_pattern(predicate="works_at")
    
    # Get statistics
    stats = store.get_statistics()
    print(f"Stored {stats['edge_count']} relationships")
```

## Configuration

GraphStore participates in the unified storage framework driven by
`SpindleConfig`. When you pass a config object, the default database path comes
from `config.storage.graph_store_path` (created automatically by
`StoragePaths.ensure_directories()` or the `spindle-ingest config init`
generator).

### Constructor Parameters

```python
from spindle.configuration import load_config_from_file
from spindle import GraphStore

config = load_config_from_file("config.py")

# Uses config.storage.graph_store_path
store = GraphStore(config=config)

# Override with an explicit path (absolute or relative)
store = GraphStore(db_path="/tmp/my_graph.db", config=config)
```

If you omit `config=`, GraphStore still accepts string paths. Relative names
(`"custom_graph"`) resolve into a `graphs/` directory beneath the package root
for backward compatibility, but new deployments should rely on
`SpindleConfig.with_root(...)` so all storage lives under a single directory.

## Schema

GraphStore uses a simple but effective schema:

### Entity (Node Table)

```
Entity(
    name STRING PRIMARY KEY,    # Unique entity identifier
    type STRING,                 # Entity type (e.g., "Person", "Organization")
    metadata STRING              # JSON string with additional metadata
)
```

### Relationship (Edge Table)

```
Relationship(
    FROM Entity TO Entity,
    predicate STRING,            # Relationship type (e.g., "works_at")
    source STRING,               # Source document name
    extraction_datetime STRING,  # ISO 8601 timestamp
    supporting_evidence STRING,  # JSON array of character spans
    metadata STRING              # JSON string with additional metadata
)
```

## API Reference

### Graph Management

#### `__init__(db_path: Optional[str] = None)`

Initialize GraphStore with optional database path.

```python
# Use environment variable or default
store = GraphStore()

# Explicit path
store = GraphStore(db_path="./my_graph.db")
```

#### `create_graph(db_path: Optional[str] = None)`

Create a new graph database, optionally at a different path.

```python
store.create_graph()  # Reinitialize at current path
store.create_graph(db_path="./new_graph.db")  # Create at new location
```

#### `delete_graph()`

Delete the entire graph database (irreversible!).

```python
store.delete_graph()
# Database directory is removed
```

#### `close()`

Close database connection. Automatically called when using context manager.

```python
store = GraphStore()
# ... use store ...
store.close()

# Or use context manager (recommended)
with GraphStore() as store:
    # ... use store ...
    pass  # Automatically closed
```

### Node Operations

#### `add_node(name: str, entity_type: str, metadata: Dict = None) -> bool`

Add a single node to the graph.

```python
store.add_node(
    name="Alice Johnson",
    entity_type="Person",
    metadata={"employee_id": "E12345", "verified": True}
)
```

#### `add_nodes(nodes: List[Dict]) -> int`

Bulk add multiple nodes.

```python
nodes = [
    {"name": "Alice", "type": "Person", "metadata": {}},
    {"name": "Bob", "type": "Person", "metadata": {}},
    {"name": "TechCorp", "type": "Organization", "metadata": {}}
]
count = store.add_nodes(nodes)
print(f"Added {count} nodes")
```

#### `get_node(name: str) -> Optional[Dict]`

Retrieve a node by name.

```python
node = store.get_node("Alice Johnson")
if node:
    print(f"Type: {node['type']}")
    print(f"Metadata: {node['metadata']}")
```

#### `update_node(name: str, updates: Dict) -> bool`

Update node properties.

```python
store.update_node(
    "Alice Johnson",
    updates={
        "type": "Engineer",
        "metadata": {"employee_id": "E12345", "verified": True, "department": "Engineering"}
    }
)
```

#### `delete_node(name: str) -> bool`

Delete a node and all its edges.

```python
success = store.delete_node("Alice Johnson")
```

### Edge Operations

#### `add_edge(subject: str, predicate: str, obj: str, metadata: Dict = None) -> Dict[str, Any]`

Add a single edge with intelligent evidence merging.

**Returns:** Dictionary with `success` (bool) and `message` (str) keys.

**Evidence Structure:** Edges now consolidate evidence from multiple sources using a nested structure:
- Each source has a `source_nm`, `source_url`, and list of `spans`
- Each span includes `text`, `start`, `end`, and `extraction_datetime`
- Duplicate triples from different sources merge into a single edge

**Deduplication Rules:**
- Same source + same span text → Skip with message
- Same source + different span → Add span to existing source  
- New source → Add as new source entry

```python
# Ensure nodes exist first
store.add_node("Alice", "Person", {})
store.add_node("TechCorp", "Organization", {})

# Add edge with new nested evidence format
result = store.add_edge(
    subject="Alice",
    predicate="works_at",
    obj="TechCorp",
    metadata={
        "supporting_evidence": [{
            "source_nm": "HR Database",
            "source_url": "https://hr.example.com",
            "spans": [{
                "text": "Alice works at TechCorp",
                "start": 0,
                "end": 23,
                "extraction_datetime": "2024-01-15T10:00:00Z"
            }]
        }]
    }
)

if result["success"]:
    print(result["message"])  # "Created new edge" or merge message
    
# Adding from another source merges into the same edge
result2 = store.add_edge(
    subject="Alice",
    predicate="works_at", 
    obj="TechCorp",
    metadata={
        "supporting_evidence": [{
            "source_nm": "Company Directory",
            "source_url": "https://directory.example.com",
            "spans": [{
                "text": "Alice Johnson is employed at TechCorp",
                "start": 0,
                "end": 37,
                "extraction_datetime": "2024-01-16T09:30:00Z"
            }]
        }]
    }
)
# result2["message"] will be "Added new source: Company Directory"
```

#### `add_edges(edges: List[Dict]) -> int`

Bulk add multiple edges.

```python
edges = [
    {"subject": "Alice", "predicate": "works_at", "object": "TechCorp", "metadata": {}},
    {"subject": "Bob", "predicate": "works_at", "object": "TechCorp", "metadata": {}}
]
count = store.add_edges(edges)
```

#### `get_edge(subject: str, predicate: str, obj: str) -> Optional[List[Dict]]`

Retrieve edges matching exact pattern. Returns a list with a single edge (containing consolidated evidence from all sources).

```python
edges = store.get_edge("Alice", "works_at", "TechCorp")
if edges:
    edge = edges[0]  # Single edge with all evidence
    print(f"Subject: {edge['subject']}")
    print(f"Predicate: {edge['predicate']}")
    print(f"Object: {edge['object']}")
    
    # Iterate through all sources
    for source in edge['supporting_evidence']:
        print(f"\nSource: {source['source_nm']}")
        print(f"URL: {source['source_url']}")
        print(f"Number of spans: {len(source['spans'])}")
        
        # Check each span
        for span in source['spans']:
            print(f"  - '{span['text']}' at [{span['start']}:{span['end']}]")
            print(f"    Extracted: {span['extraction_datetime']}")
```

#### `update_edge(subject: str, predicate: str, obj: str, updates: Dict) -> bool`

Update edge properties. Updates ALL edges matching the pattern.

```python
store.update_edge(
    "Alice", "works_at", "TechCorp",
    updates={"metadata": {"verified": True}}
)
```

#### `delete_edge(subject: str, predicate: str, obj: str) -> bool`

Delete all edges matching the pattern.

```python
store.delete_edge("Alice", "works_at", "TechCorp")
```

### Triple Integration

#### `add_triples(triples: List[Triple]) -> int`

Bulk import triples from Spindle extraction.

```python
result = extractor.extract(text, "Source Name")
count = store.add_triples(result.triples)
print(f"Stored {count} triples")
```

#### `get_triples() -> List[Triple]`

Export all edges as Triple objects. Since edges now consolidate evidence from multiple sources, this creates one Triple per source for backward compatibility.

```python
all_triples = store.get_triples()

# If an edge has evidence from 2 sources, you'll get 2 Triples
# Each Triple represents the fact from one specific source
for triple in all_triples:
    print(f"{triple.subject} {triple.predicate} {triple.object}")
    print(f"  Source: {triple.source.source_name}")
    print(f"  Spans: {len(triple.supporting_spans)}")

# Use with any Spindle function
from spindle import filter_triples_by_source
source_triples = filter_triples_by_source(all_triples, "My Source")
```

#### `add_edge_from_triple(triple: Triple) -> bool`

Create edge from a single Triple object.

```python
success = store.add_edge_from_triple(triple)
```

#### `add_nodes_from_triple(triple: Triple) -> Tuple[bool, bool]`

Extract and add subject and object nodes from a triple.

```python
subject_added, object_added = store.add_nodes_from_triple(triple)
```

### Node Embeddings

GraphStore supports Node2Vec-based graph structure-aware embeddings that capture the structural relationships between nodes. Unlike text-based embeddings, Node2Vec embeddings are computed from the graph structure itself, making nodes with similar connection patterns have similar embeddings.

#### Why Node2Vec?

Node2Vec embeddings are structure-aware, meaning they consider:
- **Node connections**: Nodes with similar neighborhoods get similar embeddings
- **Graph topology**: Structural roles are captured (e.g., hub nodes, bridge nodes)
- **Relationship patterns**: Nodes connected through similar paths have similar representations

This is particularly valuable for knowledge graphs where the structure contains semantic information beyond just node attributes.

#### Computing Embeddings

Embeddings are computed on-demand using the `compute_graph_embeddings()` method. You must first build your graph, then compute embeddings:

```python
from spindle.configuration import load_config_from_file
from spindle import GraphStore, ChromaVectorStore

config = load_config_from_file("config.py")

# Create graph store and vector store using unified storage paths
store = GraphStore(config=config)
vector_store = ChromaVectorStore(collection_name="my_embeddings", config=config)

# Build your graph first
store.add_node("Alice", "Person")
store.add_node("Bob", "Person")
store.add_node("TechCorp", "Organization")
store.add_edge("Alice", "works_at", "TechCorp")
store.add_edge("Bob", "works_at", "TechCorp")

# Compute embeddings after graph is built
embeddings = store.compute_graph_embeddings(
    vector_store,
    dimensions=128,      # Embedding dimension
    walk_length=80,      # Length of random walks
    num_walks=10,        # Number of walks per node
    p=1.0,               # Return parameter
    q=1.0                # In-out parameter
)

# embeddings is a dict mapping node names to vector_index UIDs
print(f"Computed embeddings for {len(embeddings)} nodes")
```

#### Parameters

- **dimensions** (default: 128): Dimensionality of embedding vectors
- **walk_length** (default: 80): Length of each random walk
- **num_walks** (default: 10): Number of random walks per node
- **p** (default: 1.0): Return parameter - controls likelihood of immediately revisiting a node
- **q** (default: 1.0): In-out parameter - controls exploration vs exploitation
- **workers** (default: 1): Number of worker threads

#### Using Embeddings

After computing embeddings, nodes are automatically updated with their `vector_index` values. You can then use the VectorStore to query for similar nodes:

```python
# Query for nodes similar to a specific node
alice = store.get_node("Alice")
if alice and alice.get("vector_index"):
    # Find similar nodes using vector store
    similar = vector_store.query(
        text=alice["vector_index"],  # Query by embedding UID
        top_k=5,
        metadata_filter={"type": "node"}
    )
```

#### Edge Cases

- **Empty graph**: Raises `ValueError` - cannot compute embeddings for empty graphs
- **Single node**: Returns zero vector embedding
- **Isolated nodes**: Handled correctly - nodes with no edges get embeddings based on their isolation
- **Disconnected components**: Each component is processed independently within the random walk

#### Requirements

Node2Vec embeddings require additional dependencies:
```bash
uv pip install node2vec>=0.4.5 networkx>=3.0
```

### Query Operations

#### `query_by_pattern(subject: Optional[str] = None, predicate: Optional[str] = None, obj: Optional[str] = None) -> List[Dict]`

Query edges by pattern with wildcards (`None` = match anything).

```python
# All "works_at" relationships
employees = store.query_by_pattern(predicate="works_at")

# All relationships involving Alice
alice_rels = store.query_by_pattern(subject="Alice Johnson")

# All relationships pointing to TechCorp
techcorp_rels = store.query_by_pattern(obj="TechCorp")

# Specific relationship
specific = store.query_by_pattern(
    subject="Alice Johnson",
    predicate="works_at",
    obj="TechCorp"
)

# All relationships (wildcard)
all_edges = store.query_by_pattern()
```

#### `query_by_source(source_name: str) -> List[Dict]`

Filter edges by source document. Searches within the nested evidence structure to find edges that include the specified source.

```python
# All edges that have evidence from a specific source
hr_data = store.query_by_source("HR Database")

# An edge appears in results if ANY of its sources match
# Even if it also has evidence from other sources
for edge in hr_data:
    print(f"{edge['subject']} -> {edge['predicate']} -> {edge['object']}")
    # Check all sources in this edge
    for source in edge['supporting_evidence']:
        print(f"  From: {source['source_nm']}")

# Compare data from different sources
source1 = store.query_by_source("Source A")
source2 = store.query_by_source("Source B")
# Note: An edge can appear in both if it has evidence from both sources
```

#### `query_by_date_range(start: Optional[datetime] = None, end: Optional[datetime] = None) -> List[Dict]`

Filter edges by extraction date range. Since `extraction_datetime` is now at the span level, this returns edges that have **at least one span** within the specified date range.

```python
from datetime import datetime, timedelta, timezone

# Edges with at least one span from the last 7 days
week_ago = datetime.now(timezone.utc) - timedelta(days=7)
recent = store.query_by_date_range(start=week_ago)

# Edges with spans from specific time window  
start = datetime(2024, 1, 1, tzinfo=timezone.utc)
end = datetime(2024, 1, 31, tzinfo=timezone.utc)
january = store.query_by_date_range(start, end)

# All edges with spans before a date
before_date = store.query_by_date_range(end=datetime(2024, 6, 1, tzinfo=timezone.utc))

# Results include edges where ANY span matches the date criteria
for edge in recent:
    for source in edge['supporting_evidence']:
        for span in source['spans']:
            # Each span has its own extraction_datetime
            print(f"Span extracted at: {span['extraction_datetime']}")
```

#### `query_cypher(cypher_query: str) -> List[Dict]`

Execute raw Cypher query for advanced use cases.

```python
# Find all people and their companies
query = """
MATCH (p:Entity)-[r:Relationship {predicate: 'works_at'}]->(c:Entity)
RETURN p.name AS person, c.name AS company
"""
results = store.query_cypher(query)

# Complex traversal: people, companies, and locations
query = """
MATCH (p:Entity)-[:Relationship {predicate: 'works_at'}]->(c:Entity)
MATCH (c)-[:Relationship {predicate: 'located_in'}]->(loc:Entity)
RETURN p.name, c.name, loc.name
"""
results = store.query_cypher(query)

# Aggregation
query = """
MATCH (e:Entity)-[r:Relationship {predicate: 'works_at'}]->()
RETURN e.name, count(r) as job_count
ORDER BY job_count DESC
"""
results = store.query_cypher(query)
```

### Utility Methods

#### `get_statistics() -> Dict`

Get graph statistics.

```python
stats = store.get_statistics()

print(f"Nodes: {stats['node_count']}")
print(f"Edges: {stats['edge_count']}")
print(f"Sources: {', '.join(stats['sources'])}")
print(f"Predicates: {', '.join(stats['predicates'])}")

if stats['date_range']:
    print(f"Earliest: {stats['date_range']['earliest']}")
    print(f"Latest: {stats['date_range']['latest']}")
```

## Advanced Usage Patterns

### Multi-Source Knowledge Graph

Build a knowledge graph from multiple sources and track provenance:

```python
with GraphStore() as store:
    # Extract from first source
    result1 = extractor.extract(text1, "Wikipedia")
    store.add_triples(result1.triples)
    
    # Extract from second source
    result2 = extractor.extract(text2, "Company Website")
    store.add_triples(result2.triples)
    
    # Find facts confirmed by multiple sources
    all_edges = store.query_by_pattern()
    
    fact_to_sources = {}
    for edge in all_edges:
        fact = (edge['subject'], edge['predicate'], edge['object'])
        if fact not in fact_to_sources:
            fact_to_sources[fact] = []
        fact_to_sources[fact].append(edge['source'])
    
    # Facts with multiple sources are more trustworthy
    confirmed = {f: s for f, s in fact_to_sources.items() if len(s) > 1}
```

### Incremental Knowledge Graph Building

Build your knowledge graph incrementally over time:

```python
with GraphStore() as store:
    # Export existing triples for entity consistency
    existing_triples = store.get_triples()
    
    # Extract from new document
    new_result = extractor.extract(
        new_text,
        "New Source",
        existing_triples=existing_triples
    )
    
    # Add to graph
    store.add_triples(new_result.triples)
```

### Temporal Analysis

Analyze how your knowledge graph evolves over time:

```python
from datetime import datetime, timedelta

with GraphStore() as store:
    # Triples added today
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0)
    today_triples = store.query_by_date_range(start=today_start)
    
    # Triples added this week
    week_start = datetime.utcnow() - timedelta(days=7)
    week_triples = store.query_by_date_range(start=week_start)
    
    # Compare growth
    print(f"Today: {len(today_triples)} new triples")
    print(f"This week: {len(week_triples)} total")
```

### Custom Node Types and Metadata

Enrich nodes with custom metadata:

```python
with GraphStore() as store:
    # Add node with rich metadata
    store.add_node(
        name="Alice Johnson",
        entity_type="Person",
        metadata={
            "employee_id": "E12345",
            "department": "Engineering",
            "verified": True,
            "confidence_score": 0.95,
            "first_seen": "2024-01-15T10:00:00Z",
            "sources": ["HR Database", "LinkedIn"]
        }
    )
    
    # Update metadata as you learn more
    alice = store.get_node("Alice Johnson")
    alice_meta = alice['metadata']
    alice_meta['title'] = "Senior Engineer"
    alice_meta['sources'].append("Company Website")
    
    store.update_node("Alice Johnson", updates={"metadata": alice_meta})
```

### Export and Import

Export your graph for backup or sharing:

```python
import json
from spindle import triples_to_dict

with GraphStore() as store:
    # Export all triples
    triples = store.get_triples()
    
    # Convert to JSON-serializable format
    triples_dict = triples_to_dict(triples)
    
    # Save to file
    with open("knowledge_graph_export.json", "w") as f:
        json.dump(triples_dict, f, indent=2)
```

Import from backup:

```python
from spindle import dict_to_triples

with open("knowledge_graph_export.json", "r") as f:
    triples_dict = json.load(f)

# Convert back to Triple objects
triples = dict_to_triples(triples_dict)

# Import into new database
with GraphStore(db_path="./restored_graph.db") as store:
    store.add_triples(triples)
```

## Performance Considerations

### Batch Operations

Use bulk operations for better performance:

```python
# Good: Bulk insert
store.add_triples(all_triples)  # Efficient

# Avoid: Individual inserts in loop
for triple in all_triples:  # Less efficient
    store.add_edge_from_triple(triple)
```

### Query Optimization

Use specific patterns instead of wildcards when possible:

```python
# More efficient: Specific predicate
employees = store.query_by_pattern(predicate="works_at")

# Less efficient: Full scan with post-filtering
all_edges = store.query_by_pattern()
employees = [e for e in all_edges if e['predicate'] == 'works_at']
```

### Database Size

Monitor database size and consider partitioning for very large graphs:

```python
import os

db_size = sum(
    os.path.getsize(os.path.join(dirpath, filename))
    for dirpath, _, filenames in os.walk(store.db_path)
    for filename in filenames
)
print(f"Database size: {db_size / (1024*1024):.2f} MB")
```

## Best Practices

### 1. Use Context Managers

Always use `with` statement for automatic resource cleanup:

```python
with GraphStore() as store:
    # Your code here
    pass
# Connection automatically closed
```

### 2. Validate Data Before Storage

Check extraction quality before persisting:

```python
result = extractor.extract(text, "Source")

# Check quality
if len(result.triples) > 0 and result.reasoning:
    store.add_triples(result.triples)
else:
    print("Low quality extraction, skipping storage")
```

### 3. Track Source Metadata

Always include source information:

```python
result = extractor.extract(
    text,
    source_name="Document Name",
    source_url="https://example.com/doc"  # Include URL for provenance
)
```

### 4. Regular Backups

Export your graph regularly:

```python
def backup_graph(store, backup_path):
    triples = store.get_triples()
    triples_dict = triples_to_dict(triples)
    with open(backup_path, "w") as f:
        json.dump(triples_dict, f)
```

### 5. Monitor Statistics

Track graph growth over time:

```python
def log_statistics(store):
    stats = store.get_statistics()
    timestamp = datetime.utcnow().isoformat()
    
    log_entry = {
        "timestamp": timestamp,
        "nodes": stats['node_count'],
        "edges": stats['edge_count'],
        "sources": len(stats['sources']),
        "predicates": len(stats['predicates'])
    }
    
    # Log to file or monitoring system
    print(json.dumps(log_entry))
```

## Migration Strategies

### Migrating Existing Data

If you have existing triple dictionaries (from JSON files):

```python
import json
from spindle import dict_to_triples

# Load existing data
with open("old_data.json", "r") as f:
    triples_dict = json.load(f)

# Convert to Triple objects
triples = dict_to_triples(triples_dict)

# Import into GraphStore
with GraphStore() as store:
    count = store.add_triples(triples)
    print(f"Migrated {count} triples")
```

### Upgrading Schema

To modify the schema (e.g., add fields), export, transform, and reimport:

```python
# Export existing data
with GraphStore(db_path="./old_graph.db") as old_store:
    old_triples = old_store.get_triples()

# Transform data (example: add confidence scores)
for triple in old_triples:
    # Add custom processing
    pass

# Import into new database
with GraphStore(db_path="./new_graph.db") as new_store:
    new_store.add_triples(old_triples)
```

## Troubleshooting

### Database Locked Error

If you get a "database locked" error:

```python
# Ensure previous connection is closed
store1 = GraphStore()
store1.close()  # Must close before opening again

# Or use context manager to ensure cleanup
with GraphStore() as store:
    pass  # Automatically closed
```

### Import Errors

If GraphStore import fails:

```bash
uv pip install kuzu>=0.7.0
```

### Memory Issues

For very large graphs, process in batches:

```python
batch_size = 1000
for i in range(0, len(all_triples), batch_size):
    batch = all_triples[i:i+batch_size]
    store.add_triples(batch)
```

## Examples

See `spindle/notebooks/example_graph_store.ipynb` for a comprehensive walkthrough of GraphStore features (launch with `uv run jupyter lab` or view directly in your IDE).

## Further Reading

- [Kùzu Documentation](https://kuzudb.com/docs/)
- [Cypher Query Language](https://neo4j.com/docs/cypher-manual/current/)
- [Spindle README](../README.md)
- [Testing Guide](./TESTING.md)

