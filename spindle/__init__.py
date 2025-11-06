"""
Spindle: Knowledge Graph Triple Extraction Tool

A tool for real-time extraction of knowledge graphs from multimodal data using BAML and LLMs.

Main Components:
- SpindleExtractor: Extract triples from text using a predefined ontology
- OntologyRecommender: Automatically recommend ontologies by analyzing text
- GraphStore: Persistent graph database storage using Kùzu
- Helper functions for ontology creation, serialization, and filtering

Example:
    >>> from spindle import SpindleExtractor, create_ontology
    >>> 
    >>> entity_types = [{"name": "Person", "description": "A human"}]
    >>> relation_types = [{"name": "knows", "description": "Knows", "domain": "Person", "range": "Person"}]
    >>> ontology = create_ontology(entity_types, relation_types)
    >>> 
    >>> extractor = SpindleExtractor(ontology)
    >>> result = extractor.extract("Alice knows Bob", source_name="Test")
"""

from spindle.extractor import (
    SpindleExtractor,
    OntologyRecommender,
    create_ontology,
    create_source_metadata,
    triples_to_dict,
    dict_to_triples,
    get_supporting_text,
    filter_triples_by_source,
    parse_extraction_datetime,
    filter_triples_by_date_range,
    ontology_to_dict,
    recommendation_to_dict,
    extension_to_dict,
    _find_span_indices,
)

# Import GraphStore (optional dependency)
try:
    from spindle.graph_store import GraphStore
    _GRAPH_STORE_AVAILABLE = True
except ImportError:
    GraphStore = None
    _GRAPH_STORE_AVAILABLE = False

# Import VectorStore classes (optional dependency)
try:
    from spindle.vector_store import (
        VectorStore,
        ChromaVectorStore,
        create_openai_embedding_function,
        create_huggingface_embedding_function,
        get_default_embedding_function,
    )
    _VECTOR_STORE_AVAILABLE = True
except ImportError:
    VectorStore = None
    ChromaVectorStore = None
    create_openai_embedding_function = None
    create_huggingface_embedding_function = None
    get_default_embedding_function = None
    _VECTOR_STORE_AVAILABLE = False

__version__ = "0.1.0"

__all__ = [
    # Main classes
    "SpindleExtractor",
    "OntologyRecommender",
    "GraphStore",
    "VectorStore",
    "ChromaVectorStore",
    # Factory functions
    "create_ontology",
    "create_source_metadata",
    # Embedding functions
    "create_openai_embedding_function",
    "create_huggingface_embedding_function",
    "get_default_embedding_function",
    # Serialization functions
    "triples_to_dict",
    "dict_to_triples",
    "ontology_to_dict",
    "recommendation_to_dict",
    "extension_to_dict",
    # Query/filter functions
    "get_supporting_text",
    "filter_triples_by_source",
    "parse_extraction_datetime",
    "filter_triples_by_date_range",
    # Internal (for testing)
    "_find_span_indices",
]

