"""
Graph embedding computation utilities for graph stores.

This module provides utilities for computing and storing graph embeddings.
"""

from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from spindle.graph_store.store import GraphStore
    from spindle.vector_store import VectorStore


def compute_graph_embeddings(
    graph_store: 'GraphStore',
    vector_store: 'VectorStore',
    dimensions: int = 128,
    walk_length: int = 80,
    num_walks: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    workers: int = 1
) -> Dict[str, str]:
    """
    Compute Node2Vec embeddings for all nodes in the graph and store them.
    
    This function:
    1. Extracts the graph structure from the graph store
    2. Computes Node2Vec embeddings that capture structural relationships
    3. Stores embeddings in the provided VectorStore
    4. Returns mapping of node names to vector_index UIDs
    
    Args:
        graph_store: GraphStore instance (has query_cypher method)
        vector_store: VectorStore instance to store embeddings in
        dimensions: Dimensionality of embedding vectors (default: 128)
        walk_length: Length of each random walk (default: 80)
        num_walks: Number of random walks per node (default: 10)
        p: Return parameter - controls likelihood of revisiting a node (default: 1.0)
        q: In-out parameter - controls exploration vs exploitation (default: 1.0)
        workers: Number of worker threads (default: 1)
    
    Returns:
        Dictionary mapping node names to vector_index UIDs in VectorStore
    
    Raises:
        ImportError: If required dependencies (networkx, node2vec) are not installed
        ValueError: If vector_store is None
    """
    if vector_store is None:
        raise ValueError("vector_store is required for computing embeddings")
    
    try:
        from spindle.vector_store import GraphEmbeddingGenerator
    except ImportError:
        raise ImportError(
            "Graph embedding computation requires optional dependencies. "
            "Ensure all dependencies are installed: pip install node2vec networkx"
        )
    
    # Extract graph structure (GraphEmbeddingGenerator expects GraphStore with query_cypher)
    graph = GraphEmbeddingGenerator.extract_graph_structure(graph_store)
    
    # Compute embeddings
    embeddings = GraphEmbeddingGenerator.compute_node2vec_embeddings(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers
    )
    
    # Store embeddings and create mapping
    vector_index_map = {}
    
    for node_name, embedding_vector in embeddings.items():
        node = graph_store.get_node(node_name)
        node_type = node.get('type', 'Unknown') if node else 'Unknown'
        
        embedding_list = embedding_vector.tolist()
        
        text_representation = node_name
        
        vector_index = vector_store.add_embedding(
            embedding=embedding_list,
            text=text_representation,
            metadata={
                "type": "node",
                "entity_type": node_type,
                "name": node_name,
                "embedding_method": "node2vec",
                "dimensions": dimensions
            }
        )
        
        vector_index_map[node_name] = vector_index
    
    return vector_index_map


def update_node_embeddings(
    graph_store: 'GraphStore',
    embeddings: Dict[str, str]
) -> int:
    """
    Update nodes with their computed vector_index values.
    
    Args:
        graph_store: GraphStore instance
        embeddings: Dictionary mapping node names to vector_index UIDs
    
    Returns:
        Number of nodes successfully updated
    """
    updated_count = 0
    
    for node_name, vector_index in embeddings.items():
        try:
            success = graph_store.update_node(
                node_name,
                updates={"vector_index": vector_index}
            )
            if success:
                updated_count += 1
        except Exception:
            # Skip nodes that can't be updated
            continue
    
    return updated_count

