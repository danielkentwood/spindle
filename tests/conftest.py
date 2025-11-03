"""Pytest configuration and shared fixtures for Spindle tests."""

import os
import pytest
import tempfile
import shutil
from datetime import datetime
from spindle.baml_client.types import (
    Triple,
    Entity,
    EntityType,
    AttributeDefinition,
    AttributeValue,
    RelationType,
    Ontology,
    ExtractionResult,
    OntologyRecommendation,
    OntologyExtension,
    SourceMetadata,
    CharacterSpan
)
from tests.fixtures.sample_ontologies import (
    create_simple_ontology,
    create_complex_ontology
)

# Try to import GraphStore (may not be available if kuzu not installed)
try:
    from spindle import GraphStore
    GRAPH_STORE_AVAILABLE = True
except ImportError:
    GraphStore = None
    GRAPH_STORE_AVAILABLE = False


def _create_entity(name: str, entity_type: str, description: str = "", **attrs):
    """Helper to create test Entity objects."""
    custom_atts = {
        attr_name: AttributeValue(value=attr_val, type="string")
        for attr_name, attr_val in attrs.items()
    }
    return Entity(
        name=name,
        type=entity_type,
        description=description,
        custom_atts=custom_atts
    )


# Pytest configuration
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires API key)"
    )


@pytest.fixture
def simple_ontology():
    """Provide a simple ontology for testing."""
    return create_simple_ontology()


@pytest.fixture
def complex_ontology():
    """Provide a complex ontology for testing."""
    return create_complex_ontology()


@pytest.fixture
def sample_source_metadata():
    """Provide sample source metadata."""
    return SourceMetadata(
        source_name="Test Document",
        source_url="https://example.com/test"
    )


@pytest.fixture
def sample_character_span():
    """Provide a sample character span."""
    return CharacterSpan(
        text="Alice works at TechCorp",
        start=0,
        end=23
    )


@pytest.fixture
def sample_triple(sample_source_metadata, sample_character_span):
    """Provide a sample triple with metadata."""
    return Triple(
        subject=_create_entity("Alice Johnson", "Person", "A software engineer"),
        predicate="works_at",
        object=_create_entity("TechCorp", "Organization", "A technology company"),
        source=sample_source_metadata,
        supporting_spans=[sample_character_span],
        extraction_datetime="2024-01-15T10:30:00Z"
    )


@pytest.fixture
def sample_triples(sample_source_metadata):
    """Provide a list of sample triples."""
    return [
        Triple(
            subject=_create_entity("Alice Johnson", "Person", "A software engineer"),
            predicate="works_at",
            object=_create_entity("TechCorp", "Organization", "A technology company"),
            source=sample_source_metadata,
            supporting_spans=[
                CharacterSpan(text="Alice Johnson works at TechCorp", start=0, end=32)
            ],
            extraction_datetime="2024-01-15T10:30:00Z"
        ),
        Triple(
            subject=_create_entity("TechCorp", "Organization", "A technology company"),
            predicate="located_in",
            object=_create_entity("San Francisco", "Location", "A city in California"),
            source=sample_source_metadata,
            supporting_spans=[
                CharacterSpan(text="TechCorp is located in San Francisco", start=34, end=71)
            ],
            extraction_datetime="2024-01-15T10:30:00Z"
        )
    ]


@pytest.fixture
def mock_extraction_result(sample_triples):
    """Provide a mock ExtractionResult."""
    return ExtractionResult(
        triples=sample_triples,
        reasoning="Extracted entities and relations based on the ontology."
    )


@pytest.fixture
def mock_ontology_recommendation(simple_ontology):
    """Provide a mock OntologyRecommendation."""
    return OntologyRecommendation(
        ontology=simple_ontology,
        text_purpose="To describe employment relationships in a technology company.",
        reasoning="The text focuses on people working at companies, so Person and Organization entity types are appropriate."
    )


@pytest.fixture
def mock_ontology_extension_needed():
    """Provide a mock OntologyExtension indicating extension is needed."""
    return OntologyExtension(
        needs_extension=True,
        new_entity_types=[
            EntityType(
                name="Medication",
                description="A pharmaceutical drug or treatment",
                attributes=[
                    AttributeDefinition(name="dosage", type="string", description="Medication dosage")
                ]
            ),
            EntityType(
                name="Condition",
                description="A medical condition or disease",
                attributes=[]
            )
        ],
        new_relation_types=[
            RelationType(
                name="treats",
                description="Treatment relationship",
                domain="Medication",
                range="Condition"
            )
        ],
        critical_information_at_risk="Medical entities and treatment relationships cannot be captured with current ontology.",
        reasoning="The text discusses medical treatments which require specialized entity types."
    )


@pytest.fixture
def mock_ontology_extension_not_needed():
    """Provide a mock OntologyExtension indicating no extension is needed."""
    return OntologyExtension(
        needs_extension=False,
        new_entity_types=[],
        new_relation_types=[],
        critical_information_at_risk="",
        reasoning="All entities can be represented with existing Person and Organization types."
    )


@pytest.fixture
def sample_text():
    """Provide sample text for extraction."""
    return "Alice Johnson works at TechCorp. TechCorp is located in San Francisco."


@pytest.fixture
def sample_text_with_newlines():
    """Provide text with newlines for span matching tests."""
    return """Alice Johnson
works at
TechCorp in
San Francisco."""


@pytest.fixture
def sample_text_with_whitespace():
    """Provide text with extra whitespace."""
    return "Alice   Johnson    works    at    TechCorp."


@pytest.fixture
def fixed_datetime():
    """Provide a fixed datetime for testing."""
    return datetime(2024, 1, 15, 10, 30, 0)


@pytest.fixture
def api_key_available():
    """Check if API key is available for integration tests."""
    return os.getenv("ANTHROPIC_API_KEY") is not None


@pytest.fixture
def skip_if_no_api_key(api_key_available):
    """Skip test if no API key is available."""
    if not api_key_available:
        pytest.skip("ANTHROPIC_API_KEY not set - skipping integration test")


# ========== GraphStore Fixtures ==========

@pytest.fixture
def skip_if_no_graph_store():
    """Skip test if GraphStore is not available (kuzu not installed)."""
    if not GRAPH_STORE_AVAILABLE:
        pytest.skip("GraphStore not available - kuzu not installed")


@pytest.fixture
def temp_db_path():
    """Provide a temporary database path and clean it up after test."""
    temp_dir = tempfile.mkdtemp(prefix="spindle_test_")
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def temp_graph_store(temp_db_path, skip_if_no_graph_store):
    """Provide a temporary GraphStore for isolated tests."""
    store = GraphStore(db_path=temp_db_path)
    yield store
    store.close()


@pytest.fixture
def populated_graph_store(temp_graph_store, sample_triples):
    """Provide a GraphStore pre-populated with sample triples."""
    temp_graph_store.add_triples(sample_triples)
    return temp_graph_store


@pytest.fixture
def diverse_triples():
    """Provide a diverse set of triples for query testing."""
    source1 = SourceMetadata(
        source_name="Source A",
        source_url="https://example.com/a"
    )
    source2 = SourceMetadata(
        source_name="Source B",
        source_url="https://example.com/b"
    )
    
    return [
        # Employment relationships
        Triple(
            subject=_create_entity("Alice Johnson", "Person", "Software engineer"),
            predicate="works_at",
            object=_create_entity("TechCorp", "Organization", "Technology company"),
            source=source1,
            supporting_spans=[CharacterSpan(text="Alice Johnson works at TechCorp", start=0, end=32)],
            extraction_datetime="2024-01-15T10:00:00Z"
        ),
        Triple(
            subject=_create_entity("Bob Smith", "Person", "Senior developer"),
            predicate="works_at",
            object=_create_entity("TechCorp", "Organization", "Technology company"),
            source=source1,
            supporting_spans=[CharacterSpan(text="Bob Smith works at TechCorp", start=34, end=62)],
            extraction_datetime="2024-01-15T10:30:00Z"
        ),
        Triple(
            subject=_create_entity("Carol Davis", "Person", "Data scientist"),
            predicate="works_at",
            object=_create_entity("DataCorp", "Organization", "Data analytics company"),
            source=source2,
            supporting_spans=[CharacterSpan(text="Carol Davis works at DataCorp", start=0, end=29)],
            extraction_datetime="2024-01-15T11:00:00Z"
        ),
        # Location relationships
        Triple(
            subject=_create_entity("TechCorp", "Organization", "Technology company"),
            predicate="located_in",
            object=_create_entity("San Francisco", "Location", "City in California"),
            source=source1,
            supporting_spans=[CharacterSpan(text="TechCorp is in San Francisco", start=64, end=92)],
            extraction_datetime="2024-01-15T10:00:00Z"
        ),
        Triple(
            subject=_create_entity("DataCorp", "Organization", "Data analytics company"),
            predicate="located_in",
            object=_create_entity("New York", "Location", "City in New York state"),
            source=source2,
            supporting_spans=[CharacterSpan(text="DataCorp is in New York", start=31, end=54)],
            extraction_datetime="2024-01-15T11:00:00Z"
        ),
        # Technology usage
        Triple(
            subject=_create_entity("Alice Johnson", "Person", "Software engineer"),
            predicate="uses",
            object=_create_entity("Python", "Technology", "Programming language"),
            source=source1,
            supporting_spans=[CharacterSpan(text="Alice uses Python", start=94, end=111)],
            extraction_datetime="2024-01-15T10:00:00Z"
        ),
        Triple(
            subject=_create_entity("Bob Smith", "Person", "Senior developer"),
            predicate="uses",
            object=_create_entity("TypeScript", "Technology", "Programming language"),
            source=source1,
            supporting_spans=[CharacterSpan(text="Bob uses TypeScript", start=113, end=132)],
            extraction_datetime="2024-01-15T10:30:00Z"
        ),
    ]


@pytest.fixture
def graph_store_with_diverse_data(temp_graph_store, diverse_triples):
    """Provide a GraphStore populated with diverse triples for complex queries."""
    temp_graph_store.add_triples(diverse_triples)
    return temp_graph_store

