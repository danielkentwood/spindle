"""Unit tests for serialization functions in spindle.py."""

import pytest
from spindle.baml_client.types import (
    Triple,
    Entity,
    AttributeValue,
    SourceMetadata,
    CharacterSpan,
    EntityType,
    RelationType,
    Ontology,
    OntologyRecommendation,
    OntologyExtension
)

from spindle import (
    triples_to_dict,
    dict_to_triples,
    ontology_to_dict,
    recommendation_to_dict,
    extension_to_dict
)


def _create_entity(name: str, entity_type: str, description: str = ""):
    """Helper to create test Entity."""
    return Entity(name=name, type=entity_type, description=description, custom_atts={})


class TestTriplesSerializatio:
    """Tests for triples_to_dict and dict_to_triples functions."""
    
    def test_triples_to_dict_single(self, sample_triple):
        """Test converting a single triple to dict."""
        dicts = triples_to_dict([sample_triple])
        
        assert len(dicts) == 1
        d = dicts[0]
        assert d["subject"]["name"] == "Alice Johnson"
        assert d["subject"]["type"] == "Person"
        assert d["predicate"] == "works_at"
        assert d["object"]["name"] == "TechCorp"
        assert d["object"]["type"] == "Organization"
        assert d["source"]["source_name"] == "Test Document"
        assert d["source"]["source_url"] == "https://example.com/test"
        assert d["extraction_datetime"] == "2024-01-15T10:30:00Z"
        assert len(d["supporting_spans"]) == 1
    
    def test_triples_to_dict_multiple(self, sample_triples):
        """Test converting multiple triples to dict."""
        dicts = triples_to_dict(sample_triples)
        
        assert len(dicts) == 2
        assert dicts[0]["subject"]["name"] == "Alice Johnson"
        assert dicts[1]["subject"]["name"] == "TechCorp"
    
    def test_triples_to_dict_empty(self):
        """Test converting empty list to dict."""
        dicts = triples_to_dict([])
        
        assert len(dicts) == 0
    
    def test_triples_to_dict_with_no_spans(self):
        """Test converting triple with no supporting spans."""
        triple = Triple(
            subject=_create_entity("Alice", "Person"),
            predicate="works_at",
            object=_create_entity("TechCorp", "Organization"),
            source=SourceMetadata(source_name="Test"),
            supporting_spans=[],
            extraction_datetime="2024-01-15T10:30:00Z"
        )
        
        dicts = triples_to_dict([triple])
        
        assert len(dicts[0]["supporting_spans"]) == 0
    
    def test_triples_to_dict_with_none_indices(self):
        """Test converting triple with None span indices."""
        triple = Triple(
            subject=_create_entity("Alice", "Person"),
            predicate="works_at",
            object=_create_entity("TechCorp", "Organization"),
            source=SourceMetadata(source_name="Test"),
            supporting_spans=[
                CharacterSpan(text="Alice works at TechCorp", start=None, end=None)
            ],
            extraction_datetime="2024-01-15T10:30:00Z"
        )
        
        dicts = triples_to_dict([triple])
        
        assert dicts[0]["supporting_spans"][0]["start"] == -1
        assert dicts[0]["supporting_spans"][0]["end"] == -1
    
    def test_dict_to_triples_single(self):
        """Test converting a single dict back to triple."""
        dict_data = {
            "subject": "Alice Johnson",  # Old format - will be converted
            "predicate": "works_at",
            "object": "TechCorp",
            "source": {
                "source_name": "Test Document",
                "source_url": "https://example.com/test"
            },
            "supporting_spans": [
                {
                    "text": "Alice works at TechCorp",
                    "start": 0,
                    "end": 23
                }
            ],
            "extraction_datetime": "2024-01-15T10:30:00Z"
        }
        
        triples = dict_to_triples([dict_data])
        
        assert len(triples) == 1
        triple = triples[0]
        # Old format is converted to Entity with Unknown type
        assert triple.subject.name == "Alice Johnson"
        assert triple.subject.type == "Unknown"
        assert triple.predicate == "works_at"
        assert triple.object.name == "TechCorp"
        assert triple.object.type == "Unknown"
        assert triple.source.source_name == "Test Document"
        assert triple.source.source_url == "https://example.com/test"
        assert triple.extraction_datetime == "2024-01-15T10:30:00Z"
        assert len(triple.supporting_spans) == 1
    
    def test_dict_to_triples_with_missing_url(self):
        """Test converting dict without source URL."""
        dict_data = {
            "subject": "Alice",
            "predicate": "works_at",
            "object": "TechCorp",
            "source": {
                "source_name": "Test Document"
            },
            "supporting_spans": [],
            "extraction_datetime": "2024-01-15T10:30:00Z"
        }
        
        triples = dict_to_triples([dict_data])
        
        assert triples[0].source.source_url is None
    
    def test_dict_to_triples_with_invalid_indices(self):
        """Test converting dict with -1 indices (not found)."""
        dict_data = {
            "subject": "Alice",
            "predicate": "works_at",
            "object": "TechCorp",
            "source": {"source_name": "Test"},
            "supporting_spans": [
                {
                    "text": "Alice works at TechCorp",
                    "start": -1,
                    "end": -1
                }
            ],
            "extraction_datetime": ""
        }
        
        triples = dict_to_triples([dict_data])
        
        # Indices of -1 should be converted to None
        span = triples[0].supporting_spans[0]
        assert span.start is None
        assert span.end is None
    
    def test_round_trip_serialization(self, sample_triples):
        """Test round-trip serialization preserves data."""
        # Serialize to dict
        dicts = triples_to_dict(sample_triples)
        
        # Deserialize back to triples
        restored_triples = dict_to_triples(dicts)
        
        # Compare
        assert len(restored_triples) == len(sample_triples)
        for orig, restored in zip(sample_triples, restored_triples):
            assert orig.subject == restored.subject
            assert orig.predicate == restored.predicate
            assert orig.object == restored.object
            assert orig.source.source_name == restored.source.source_name
            assert orig.extraction_datetime == restored.extraction_datetime


class TestOntologySerialization:
    """Tests for ontology_to_dict function."""
    
    def test_ontology_to_dict_simple(self, simple_ontology):
        """Test converting simple ontology to dict."""
        d = ontology_to_dict(simple_ontology)
        
        assert "entity_types" in d
        assert "relation_types" in d
        assert len(d["entity_types"]) == 2
        assert len(d["relation_types"]) == 1
    
    def test_ontology_to_dict_entity_types(self, simple_ontology):
        """Test entity types structure in dict."""
        d = ontology_to_dict(simple_ontology)
        
        entity = d["entity_types"][0]
        assert "name" in entity
        assert "description" in entity
        assert entity["name"] == "Person"
    
    def test_ontology_to_dict_relation_types(self, simple_ontology):
        """Test relation types structure in dict."""
        d = ontology_to_dict(simple_ontology)
        
        relation = d["relation_types"][0]
        assert "name" in relation
        assert "description" in relation
        assert "domain" in relation
        assert "range" in relation
        assert relation["name"] == "works_at"
        assert relation["domain"] == "Person"
        assert relation["range"] == "Organization"
    
    def test_ontology_to_dict_empty(self):
        """Test converting empty ontology to dict."""
        ontology = Ontology(entity_types=[], relation_types=[])
        d = ontology_to_dict(ontology)
        
        assert len(d["entity_types"]) == 0
        assert len(d["relation_types"]) == 0
    
    def test_ontology_to_dict_complex(self, complex_ontology):
        """Test converting complex ontology to dict."""
        d = ontology_to_dict(complex_ontology)
        
        assert len(d["entity_types"]) == 5
        assert len(d["relation_types"]) == 5


class TestRecommendationSerialization:
    """Tests for recommendation_to_dict function."""
    
    def test_recommendation_to_dict(self, mock_ontology_recommendation):
        """Test converting recommendation to dict."""
        d = recommendation_to_dict(mock_ontology_recommendation)
        
        assert "ontology" in d
        assert "text_purpose" in d
        assert "reasoning" in d
    
    def test_recommendation_to_dict_ontology(self, mock_ontology_recommendation):
        """Test ontology structure in recommendation dict."""
        d = recommendation_to_dict(mock_ontology_recommendation)
        
        ontology = d["ontology"]
        assert "entity_types" in ontology
        assert "relation_types" in ontology
        assert len(ontology["entity_types"]) == 2
        assert len(ontology["relation_types"]) == 1
    
    def test_recommendation_to_dict_text_purpose(self, mock_ontology_recommendation):
        """Test text_purpose field in dict."""
        d = recommendation_to_dict(mock_ontology_recommendation)
        
        assert d["text_purpose"] == "To describe employment relationships in a technology company."
    
    def test_recommendation_to_dict_reasoning(self, mock_ontology_recommendation):
        """Test reasoning field in dict."""
        d = recommendation_to_dict(mock_ontology_recommendation)
        
        assert "Person" in d["reasoning"]
        assert "Organization" in d["reasoning"]


class TestExtensionSerialization:
    """Tests for extension_to_dict function."""
    
    def test_extension_to_dict_needed(self, mock_ontology_extension_needed):
        """Test converting extension (needed) to dict."""
        d = extension_to_dict(mock_ontology_extension_needed)
        
        assert d["needs_extension"] is True
        assert len(d["new_entity_types"]) == 2
        assert len(d["new_relation_types"]) == 1
        assert "medical" in d["critical_information_at_risk"].lower()
        assert len(d["reasoning"]) > 0
    
    def test_extension_to_dict_not_needed(self, mock_ontology_extension_not_needed):
        """Test converting extension (not needed) to dict."""
        d = extension_to_dict(mock_ontology_extension_not_needed)
        
        assert d["needs_extension"] is False
        assert len(d["new_entity_types"]) == 0
        assert len(d["new_relation_types"]) == 0
        assert d["critical_information_at_risk"] == ""
    
    def test_extension_to_dict_entity_structure(self, mock_ontology_extension_needed):
        """Test new entity types structure in dict."""
        d = extension_to_dict(mock_ontology_extension_needed)
        
        entity = d["new_entity_types"][0]
        assert "name" in entity
        assert "description" in entity
        assert entity["name"] == "Medication"
    
    def test_extension_to_dict_relation_structure(self, mock_ontology_extension_needed):
        """Test new relation types structure in dict."""
        d = extension_to_dict(mock_ontology_extension_needed)
        
        relation = d["new_relation_types"][0]
        assert "name" in relation
        assert "description" in relation
        assert "domain" in relation
        assert "range" in relation
        assert relation["name"] == "treats"

