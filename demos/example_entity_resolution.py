"""
Entity Resolution Demo

This demo shows how to use semantic entity resolution to deduplicate
a knowledge graph. It demonstrates:
1. Creating a graph with known duplicates
2. Running entity resolution
3. Querying the resolved graph
4. Viewing duplicate clusters
"""

from spindle import (
    GraphStore,
    SpindleExtractor,
    create_ontology,
    VectorStore,
    ChromaVectorStore,
)
from spindle.entity_resolution import (
    EntityResolver,
    ResolutionConfig,
    resolve_entities,
)
from spindle.vector_store import get_default_embedding_function


def main():
    """Run entity resolution demo."""
    print("=" * 80)
    print("Semantic Entity Resolution Demo")
    print("=" * 80)
    print()
    
    # Step 1: Create a knowledge graph with intentional duplicates
    print("Step 1: Creating knowledge graph with duplicate entities...")
    print("-" * 80)
    
    # Define ontology
    entity_types = [
        {
            "name": "Person",
            "description": "A human being",
            "attributes": []
        },
        {
            "name": "Organization",
            "description": "A company or institution",
            "attributes": []
        },
        {
            "name": "Location",
            "description": "A geographic location",
            "attributes": []
        }
    ]
    
    relation_types = [
        {
            "name": "works_at",
            "description": "Employment relationship",
            "domain": "Person",
            "range": "Organization"
        },
        {
            "name": "employed_by",
            "description": "Employment relationship (alternate)",
            "domain": "Person",
            "range": "Organization"
        },
        {
            "name": "located_in",
            "description": "Location relationship",
            "domain": "Organization",
            "range": "Location"
        },
        {
            "name": "based_in",
            "description": "Location relationship (alternate)",
            "domain": "Organization",
            "range": "Location"
        }
    ]
    
    ontology = create_ontology(entity_types, relation_types)
    
    # Create extractor and graph store
    extractor = SpindleExtractor(ontology=ontology)
    store = GraphStore(db_path="entity_resolution_demo")
    
    # Extract triples from texts with duplicate entities
    texts = [
        ("Alice Johnson works at TechCorp, a company based in San Francisco.", "doc1", None),
        ("Alice J. is employed by Tech Corp in SF.", "doc2", None),
        ("Bob Smith works at TechCorp.", "doc3", None),
        ("Robert Smith is employed by Tech Corp.", "doc4", None),
        ("TechCorp is located in San Francisco, California.", "doc5", None),
        ("Tech Corporation is based in San Francisco.", "doc6", None),
    ]
    
    print(f"Extracting triples from {len(texts)} documents...")
    for text, source_name, source_url in texts:
        result = extractor.extract(text, source_name, source_url)
        store.add_triples(result.triples)
        print(f"  ✓ Extracted from {source_name}: {len(result.triples)} triples")
    
    # Show initial statistics
    stats = store.get_statistics()
    print(f"\nInitial graph statistics:")
    print(f"  Nodes: {stats['node_count']}")
    print(f"  Edges: {stats['edge_count']}")
    print()
    
    # Step 2: Setup entity resolution
    print("Step 2: Configuring entity resolution...")
    print("-" * 80)
    
    # Create vector store for embeddings
    embedding_function = get_default_embedding_function()
    vector_store = ChromaVectorStore(
        collection_name="entity_resolution_demo",
        embedding_function=embedding_function
    )
    
    # Configure resolution
    config = ResolutionConfig(
        blocking_threshold=0.80,      # Lower threshold to catch more variations
        matching_threshold=0.75,       # Medium confidence required
        clustering_method='hierarchical',
        batch_size=20,
        min_cluster_size=2
    )
    
    print(f"Configuration:")
    print(f"  Blocking threshold: {config.blocking_threshold}")
    print(f"  Matching threshold: {config.matching_threshold}")
    print(f"  Clustering method: {config.clustering_method}")
    print()
    
    # Step 3: Run entity resolution
    print("Step 3: Running entity resolution...")
    print("-" * 80)
    print("This will:")
    print("  1. Cluster similar entities using embeddings")
    print("  2. Use LLM to match duplicates within clusters")
    print("  3. Create SAME_AS edges between duplicates")
    print()
    
    resolver = EntityResolver(config=config)
    
    try:
        result = resolver.resolve_entities(
            graph_store=store,
            vector_store=vector_store,
            apply_to_nodes=True,
            apply_to_edges=True,
            context="Knowledge graph about people and companies"
        )
        
        print(f"Resolution complete!")
        print(f"  Nodes processed: {result.total_nodes_processed}")
        print(f"  Edges processed: {result.total_edges_processed}")
        print(f"  Blocks created: {result.blocks_created}")
        print(f"  SAME_AS edges created: {result.same_as_edges_created}")
        print(f"  Duplicate clusters found: {result.duplicate_clusters}")
        print(f"  Execution time: {result.execution_time_seconds:.2f} seconds")
        print()
        
        # Show matches found
        if result.node_matches:
            print(f"Entity matches found ({len(result.node_matches)}):")
            for match in result.node_matches[:5]:  # Show first 5
                if match.is_duplicate:
                    print(f"  • {match.entity1_id} ≈ {match.entity2_id}")
                    print(f"    Confidence: {match.confidence:.2f}")
                    print(f"    Reasoning: {match.reasoning[:80]}...")
        print()
        
    except Exception as e:
        print(f"Error during resolution: {e}")
        print("Note: This demo requires an ANTHROPIC_API_KEY for LLM matching.")
        print("Set it in your .env file or environment variables.")
        store.close()
        return
    
    # Step 4: Query resolved graph
    print("Step 4: Querying resolved graph...")
    print("-" * 80)
    
    # Get duplicate clusters
    clusters = store.get_duplicate_clusters()
    print(f"Duplicate clusters ({len(clusters)}):")
    for i, cluster in enumerate(clusters, 1):
        canonical = sorted(cluster)[0]
        print(f"  Cluster {i} (canonical: {canonical}):")
        for entity in sorted(cluster):
            marker = "★" if entity == canonical else " "
            print(f"    {marker} {entity}")
    print()
    
    # Query with resolution
    print("Querying for 'works_at' relationships:")
    print("  Without resolution:")
    edges = store.query_by_pattern(predicate="works_at")
    for edge in edges[:3]:
        print(f"    {edge['subject']} -[works_at]-> {edge['object']}")
    
    print("  With resolution:")
    resolved_edges = store.query_with_resolution(predicate="works_at", resolve_duplicates=True)
    for edge in resolved_edges[:3]:
        print(f"    {edge['subject']} -[works_at]-> {edge['object']}")
    print()
    
    # Step 5: Demonstrate canonical entity lookup
    print("Step 5: Canonical entity lookup...")
    print("-" * 80)
    
    test_entities = ["Alice Johnson", "TechCorp", "Tech Corp", "San Francisco"]
    print("Looking up canonical forms:")
    for entity in test_entities:
        canonical = store.get_canonical_entity(entity)
        if canonical and canonical != entity:
            print(f"  {entity} → {canonical}")
        elif canonical:
            print(f"  {entity} (already canonical)")
        else:
            print(f"  {entity} (not found)")
    print()
    
    # Cleanup
    print("=" * 80)
    print("Demo complete!")
    print()
    print("To explore further:")
    print("  - Check the graph database in graphs/entity_resolution_demo/")
    print("  - Modify texts to add more duplicates")
    print("  - Adjust configuration thresholds")
    print("  - Try different clustering methods")
    print("=" * 80)
    
    store.close()


if __name__ == "__main__":
    main()

