# Entity Resolution

Semantic entity resolution for knowledge graphs using embeddings and LLMs.

## Overview

Entity resolution (also called entity deduplication or record linkage) is the process of identifying and merging duplicate entities in a knowledge graph. Spindle's entity resolution uses:

- **Semantic Blocking**: Clustering embeddings to group similar entities, reducing O(n²) comparisons
- **Semantic Matching**: LLM-based duplicate detection with confidence scores and reasoning
- **Merging**: Creating SAME_AS edges to preserve provenance while linking duplicates

This approach is based on the techniques described in ["The Rise of Semantic Entity Resolution"](https://towardsdatascience.com/the-rise-of-semantic-entity-resolution/) by Russell Jurney.

## Key Concepts

### 1. Semantic Blocking

Traditional entity resolution requires comparing every entity pair (O(n²) complexity). Semantic blocking uses embedding-based clustering to create smaller "blocks" of similar entities:

```python
# Without blocking: 1000 entities = 499,500 comparisons
# With blocking (10 blocks of 100): 10 × 4,950 = 49,500 comparisons
```

**How it works:**
1. Serialize entities to text representations
2. Compute embeddings using sentence transformers
3. Cluster embeddings using hierarchical/k-means/HDBSCAN
4. Only compare entities within the same cluster

**Supported clustering methods:**
- `hierarchical`: Agglomerative clustering with cosine distance (default)
- `kmeans`: K-means clustering for faster processing
- `hdbscan`: Density-based clustering for variable cluster sizes

### 2. Semantic Matching

Within each block, an LLM (Claude Sonnet 4) analyzes entities to determine if they're duplicates:

**Matching criteria:**
- **Name variations**: "TechCorp" vs "Tech Corp" vs "TechCorp Inc."
- **Semantic similarity**: "NYC" vs "New York City"
- **Type consistency**: Must have compatible entity types
- **Attribute alignment**: Similar or complementary attributes
- **Description consistency**: Non-contradictory descriptions

**Confidence levels:**
- `high` (0.95): Clear matches, obvious name variations
- `medium` (0.75): Probable matches, semantic similarity
- `low` (0.50): Possible matches, weak signals

### 3. SAME_AS Edges

Rather than merging entities immediately, we create bidirectional SAME_AS edges:

```
TechCorp --[SAME_AS]--> Tech Corp
Tech Corp --[SAME_AS]--> TechCorp
```

**Benefits:**
- **Provenance**: Original entities and sources preserved
- **Reversibility**: Can remove SAME_AS edges if incorrect
- **Auditability**: Track reasoning for each match
- **Connected components**: Query clusters via graph traversal

## Usage

### Basic Example

```python
from spindle.configuration import load_config_from_file
from spindle import GraphStore, VectorStore, ChromaVectorStore
from spindle.entity_resolution import EntityResolver, ResolutionConfig
from spindle.vector_store import get_default_embedding_function

# Setup
config = load_config_from_file("config.py")

store = GraphStore(config=config)
vector_store = ChromaVectorStore(
    collection_name="my_embeddings",
    embedding_function=get_default_embedding_function(),
    config=config,
)

# Configure resolution
config = ResolutionConfig(
    blocking_threshold=0.85,      # Cosine similarity for clustering
    matching_threshold=0.8,       # LLM confidence for duplicates
    clustering_method='hierarchical',
    batch_size=20,
    min_cluster_size=2
)

# Run resolution
resolver = EntityResolver(config)
result = resolver.resolve_entities(
    graph_store=store,
    vector_store=vector_store,
    apply_to_nodes=True,
    apply_to_edges=True,
    context="Knowledge graph about companies and people"
)

print(f"Created {result.same_as_edges_created} SAME_AS edges")
print(f"Found {result.duplicate_clusters} duplicate clusters")
```

> Tip: The unified `SpindleConfig` keeps the graph database, vector store, and
> document catalog under a single root. See `docs/CONFIGURATION.md` for
> details on customizing paths before running resolution jobs.

### Configuration Options

```python
ResolutionConfig(
    blocking_threshold=0.85,      # Default: 0.85
    # Higher = stricter clustering, fewer comparisons
    # Lower = looser clustering, more comparisons
    
    matching_threshold=0.8,       # Default: 0.8
    # Minimum confidence to create SAME_AS edge
    # 0.95 (high), 0.75 (medium), 0.50 (low)
    
    clustering_method='hierarchical',  # Default: 'hierarchical'
    # Options: 'hierarchical', 'kmeans', 'hdbscan'
    
    batch_size=20,                # Default: 20
    # Entities per LLM matching call
    
    max_cluster_size=50,          # Default: 50
    # Skip clusters larger than this
    
    min_cluster_size=2,           # Default: 2
    # Skip clusters smaller than this
    
    merge_strategy='preserve'     # Default: 'preserve'
    # Currently only 'preserve' (SAME_AS edges)
)
```

### Querying Resolved Graphs

#### Get Duplicate Clusters

```python
clusters = store.get_duplicate_clusters()
for cluster in clusters:
    print(f"Cluster: {cluster}")
    # Example: ['TechCorp', 'Tech Corp', 'TechCorp Inc.']
```

#### Get Canonical Entity

```python
canonical = store.get_canonical_entity("Tech Corp")
print(canonical)  # Output: "TechCorp" (alphabetically first)
```

#### Query with Resolution

```python
# Regular query (includes duplicates)
edges = store.query_by_pattern(predicate="works_at")

# Resolved query (deduplicates entities)
resolved = store.query_with_resolution(
    predicate="works_at",
    resolve_duplicates=True
)
```

### Convenience Function

```python
from spindle.entity_resolution import resolve_entities

result = resolve_entities(
    graph_store=store,
    vector_store=vector_store,
    config=ResolutionConfig(blocking_threshold=0.9)
)
```

## Performance Considerations

### Blocking Efficiency

| Entities | Without Blocking | With Blocking (10 clusters) | Speedup |
|----------|------------------|----------------------------|---------|
| 100      | 4,950           | 495                        | 10×     |
| 1,000    | 499,500         | 49,950                     | 10×     |
| 10,000   | 49,995,000      | 4,999,500                  | 10×     |

### LLM Costs

Matching uses Claude Sonnet 4. Costs depend on:
- **Block sizes**: Smaller blocks = more API calls
- **Batch size**: Process multiple entities per call
- **Number of blocks**: More clusters = more API calls

**Example costs (approximate):**
- 100 entities → 5 blocks → ~$0.10
- 1,000 entities → 50 blocks → ~$1.00
- 10,000 entities → 500 blocks → ~$10.00

### Optimization Tips

1. **Tune blocking_threshold**: Start high (0.90), lower if missing matches
2. **Increase batch_size**: Process more entities per LLM call
3. **Use faster clustering**: Try `kmeans` for large graphs
4. **Pre-filter entities**: Remove obvious non-duplicates before resolution
5. **Cache embeddings**: Store embeddings in vector store for reuse

## Best Practices

### When to Use Entity Resolution

✅ **Good use cases:**
- Knowledge graphs extracted from multiple sources
- Graphs with name variations (abbreviations, typos)
- Merging data from different databases
- Cleaning user-generated content

❌ **Not recommended:**
- Small graphs (<50 entities)
- Graphs with strict naming conventions
- Financial/medical data requiring determinism
- Real-time applications (resolution is batch-oriented)

### Configuration Guidelines

**Conservative (high precision):**
```python
config = ResolutionConfig(
    blocking_threshold=0.90,
    matching_threshold=0.85,
    min_cluster_size=3
)
```

**Aggressive (high recall):**
```python
config = ResolutionConfig(
    blocking_threshold=0.75,
    matching_threshold=0.65,
    min_cluster_size=2
)
```

**Balanced (default):**
```python
config = ResolutionConfig()  # Uses defaults
```

### Error Handling

```python
try:
    result = resolver.resolve_entities(store, vector_store)
except Exception as e:
    print(f"Resolution failed: {e}")
    # Common issues:
    # - Missing ANTHROPIC_API_KEY
    # - Vector store not configured
    # - Empty graph
```

### Validation

After resolution, validate results:

```python
# Check cluster sizes
clusters = store.get_duplicate_clusters()
large_clusters = [c for c in clusters if len(c) > 10]
if large_clusters:
    print(f"Warning: {len(large_clusters)} clusters with >10 entities")

# Check confidence distribution
low_conf = [m for m in result.node_matches if m.confidence < 0.7]
print(f"Low confidence matches: {len(low_conf)}")

# Spot check reasoning
for match in result.node_matches[:5]:
    print(f"{match.entity1_id} → {match.entity2_id}")
    print(f"  Reason: {match.reasoning}")
```

## Troubleshooting

### No Matches Found

**Possible causes:**
- `blocking_threshold` too high (clusters too small)
- `matching_threshold` too high (LLM too conservative)
- `min_cluster_size` too large (skipping valid clusters)
- Entities actually aren't duplicates

**Solutions:**
- Lower `blocking_threshold` to 0.75-0.80
- Lower `matching_threshold` to 0.70-0.75
- Reduce `min_cluster_size` to 2

### Too Many False Positives

**Possible causes:**
- `blocking_threshold` too low (clusters too large)
- `matching_threshold` too low (accepting weak matches)
- Insufficient context for LLM

**Solutions:**
- Raise `blocking_threshold` to 0.85-0.90
- Raise `matching_threshold` to 0.80-0.85
- Provide better `context` parameter with domain information

### Poor Clustering

**Possible causes:**
- Wrong clustering method for data distribution
- Embeddings not capturing semantic similarity
- Threshold not tuned for domain

**Solutions:**
- Try different clustering methods (`hierarchical`, `kmeans`, `hdbscan`)
- Use domain-specific embeddings if available
- Visualize embedding space to understand distribution

### LLM Errors

**Possible causes:**
- Missing or invalid API key
- Rate limiting
- Malformed entity data

**Solutions:**
- Verify `ANTHROPIC_API_KEY` in environment
- Reduce `batch_size` to avoid rate limits
- Check entity serialization (names, types, descriptions)

## Advanced Usage

### Custom Context

Provide domain context for better matching:

```python
context = """
This is a knowledge graph about technology companies.
Common name variations:
- Corp, Corporation, Inc., LLC are interchangeable
- Street addresses: St., Street, Ave., Avenue
- Abbreviations: SF = San Francisco, NYC = New York City
"""

result = resolver.resolve_entities(
    graph_store=store,
    vector_store=vector_store,
    context=context
)
```

### Incremental Resolution

Resolve new entities without re-processing everything:

```python
# Initial resolution
result1 = resolver.resolve_entities(store, vector_store)

# Add new entities
store.add_triples(new_triples)

# Resolve only new entities
# (Implementation note: Currently resolves all entities.
#  Incremental resolution is a future enhancement.)
```

### Export Results

```python
result_dict = result.to_dict()
import json
with open('resolution_results.json', 'w') as f:
    json.dump(result_dict, f, indent=2)
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   EntityResolver                        │
│                                                         │
│  1. Get entities from GraphStore                        │
│  2. Serialize entities to text                          │
│  3. Compute embeddings (VectorStore)                    │
│  4. Semantic blocking (SemanticBlocker)                 │
│  5. Semantic matching (SemanticMatcher + BAML/Claude)   │
│  6. Create SAME_AS edges (GraphStore)                   │
│  7. Find connected components                           │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    GraphStore                           │
│                                                         │
│  • Original entities preserved                          │
│  • SAME_AS edges connect duplicates                     │
│  • Query with/without resolution                        │
│  • Get canonical entity names                           │
│  • Find duplicate clusters                              │
└─────────────────────────────────────────────────────────┘
```

## Related Resources

- [Demo Script](../demos/example_entity_resolution.py)
- [Test Suite](../tests/test_entity_resolution.py)
- [BAML Schema](../spindle/baml_src/entity_resolution.baml)
- [The Rise of Semantic Entity Resolution](https://towardsdatascience.com/the-rise-of-semantic-entity-resolution/)

## Future Enhancements

Planned improvements:
- [ ] Incremental resolution for new entities only
- [ ] Property merging strategies beyond SAME_AS
- [ ] Active learning for threshold tuning
- [ ] Alternative LLM providers
- [ ] Entity resolution UI/visualization
- [ ] Performance profiling and optimization

