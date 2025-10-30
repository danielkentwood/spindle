"""Pytest configuration and shared fixtures for Spindle tests."""

import os
import pytest
from datetime import datetime
from baml_client.types import (
    Triple,
    EntityType,
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
        subject="Alice Johnson",
        predicate="works_at",
        object="TechCorp",
        source=sample_source_metadata,
        supporting_spans=[sample_character_span],
        extraction_datetime="2024-01-15T10:30:00Z"
    )


@pytest.fixture
def sample_triples(sample_source_metadata):
    """Provide a list of sample triples."""
    return [
        Triple(
            subject="Alice Johnson",
            predicate="works_at",
            object="TechCorp",
            source=sample_source_metadata,
            supporting_spans=[
                CharacterSpan(text="Alice Johnson works at TechCorp", start=0, end=32)
            ],
            extraction_datetime="2024-01-15T10:30:00Z"
        ),
        Triple(
            subject="TechCorp",
            predicate="located_in",
            object="San Francisco",
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
            EntityType(name="Medication", description="A pharmaceutical drug or treatment"),
            EntityType(name="Condition", description="A medical condition or disease")
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

