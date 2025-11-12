"""
GraphEmbeddingGenerator: Node2Vec-based graph structure-aware embeddings.

This module provides functionality to extract graph structure from a GraphStore
and compute Node2Vec embeddings that capture the structural relationships
between nodes.
"""

from typing import TYPE_CHECKING, Dict

import numpy as np

try:
    import networkx as nx
    _NETWORKX_AVAILABLE = True
except ImportError:
    _NETWORKX_AVAILABLE = False

try:
    from node2vec import Node2Vec
    _NODE2VEC_AVAILABLE = True
except ImportError:
    _NODE2VEC_AVAILABLE = False

if TYPE_CHECKING:
    from spindle.graph_store import GraphStore
    from spindle.vector_store.base import VectorStore


class GraphEmbeddingGenerator:
    """
    Generator for Node2Vec-based graph structure-aware embeddings.

    This class extracts graph structure from a GraphStore and computes
    Node2Vec embeddings that capture the structural relationships between nodes.

    Example:
        >>> from spindle import GraphStore, ChromaVectorStore, GraphEmbeddingGenerator
        >>>
        >>> store = GraphStore(db_path="my_graph")
        >>> # ... add nodes and edges ...
        >>>
        >>> vector_store = ChromaVectorStore()
        >>> generator = GraphEmbeddingGenerator()
        >>>
        >>> embeddings = generator.compute_and_store_embeddings(
        ...     store, vector_store, dimensions=128
        ... )
    """

    @staticmethod
    def extract_graph_structure(store: 'GraphStore') -> 'nx.Graph':
        """
        Extract graph structure from Kùzu GraphStore as NetworkX format.

        Args:
            store: GraphStore instance to extract from

        Returns:
            NetworkX Graph object with nodes and edges from the store

        Raises:
            ImportError: If networkx is not installed
        """
        if not _NETWORKX_AVAILABLE:
            raise ImportError(
                "networkx is required for graph embedding computation. "
                "Install it with: pip install networkx>=3.0"
            )

        G = nx.Graph()

        # Get all nodes
        nodes_query = """
        MATCH (e:Entity)
        RETURN e.name, e.type, e.description
        """
        nodes = store.query_cypher(nodes_query)

        # Add nodes with attributes
        for node in nodes:
            node_name = node['e.name']
            node_type = node.get('e.type', 'Unknown')
            description = node.get('e.description', '') or ''

            G.add_node(
                node_name,
                type=node_type,
                description=description
            )

        # Get all edges
        edges_query = """
        MATCH (s:Entity)-[r:Relationship]->(o:Entity)
        RETURN s.name, r.predicate, o.name
        """
        edges = store.query_cypher(edges_query)

        # Add edges (with predicate information)
        edge_counts = {}
        for edge in edges:
            source = edge['s.name']
            target = edge['o.name']
            predicate = edge.get('r.predicate', 'RELATED')

            edge_key = (source, target)
            if edge_key in edge_counts:
                edge_counts[edge_key] += 1
                if 'predicates' in G[source][target]:
                    G[source][target]['predicates'].append(predicate)
            else:
                edge_counts[edge_key] = 1
                G.add_edge(
                    source,
                    target,
                    weight=1,
                    predicates=[predicate]
                )

        # Update edge weights based on frequency
        for (source, target), count in edge_counts.items():
            if G.has_edge(source, target):
                G[source][target]['weight'] = count

        return G

    @staticmethod
    def compute_node2vec_embeddings(
        graph: 'nx.Graph',
        dimensions: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        workers: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Compute Node2Vec embeddings for nodes in the graph.

        Args:
            graph: NetworkX graph to compute embeddings for
            dimensions: Dimensionality of the embedding vectors
            walk_length: Length of each random walk
            num_walks: Number of random walks per node
            p: Return parameter (controls likelihood of immediately revisiting a node)
            q: In-out parameter (controls likelihood of exploring away from start node)
            workers: Number of worker threads to use

        Returns:
            Dictionary mapping node names to numpy array embeddings

        Raises:
            ImportError: If node2vec is not installed
            ValueError: If graph is empty
        """
        if not _NODE2VEC_AVAILABLE:
            raise ImportError(
                "node2vec is required for embedding computation. "
                "Install it with: pip install node2vec>=0.4.5"
            )

        if len(graph.nodes()) == 0:
            raise ValueError("Cannot compute embeddings for empty graph")

        # For single node graphs, return a zero vector
        if len(graph.nodes()) == 1:
            node_name = list(graph.nodes())[0]
            return {node_name: np.zeros(dimensions)}

        # For graphs with no edges, return zero vectors for all nodes
        if len(graph.edges()) == 0:
            embeddings = {}
            for node in graph.nodes():
                embeddings[node] = np.zeros(dimensions)
            return embeddings

        node2vec = Node2Vec(
            graph,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            p=p,
            q=q,
            workers=workers
        )

        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        embeddings = {}
        for node in graph.nodes():
            embeddings[node] = model.wv[node]

        return embeddings

    @staticmethod
    def compute_and_store_embeddings(
        store: 'GraphStore',
        vector_store: 'VectorStore',
        dimensions: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        workers: int = 1
    ) -> Dict[str, str]:
        """
        Compute Node2Vec embeddings and store them in VectorStore.

        This is the main method that:
        1. Extracts graph structure from GraphStore
        2. Computes Node2Vec embeddings
        3. Stores embeddings in VectorStore
        4. Returns mapping of node names to vector_index UIDs

        Args:
            store: GraphStore instance to extract graph from
            vector_store: VectorStore instance to store embeddings in
            dimensions: Dimensionality of embedding vectors
            walk_length: Length of each random walk
            num_walks: Number of random walks per node
            p: Return parameter for Node2Vec
            q: In-out parameter for Node2Vec
            workers: Number of worker threads

        Returns:
            Dictionary mapping node names to vector_index UIDs in VectorStore
        """
        graph = GraphEmbeddingGenerator.extract_graph_structure(store)

        embeddings = GraphEmbeddingGenerator.compute_node2vec_embeddings(
            graph,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            p=p,
            q=q,
            workers=workers
        )

        vector_index_map = {}

        for node_name, embedding_vector in embeddings.items():
            node = store.get_node(node_name)
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

