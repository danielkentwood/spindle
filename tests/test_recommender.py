"""Unit tests for OntologyRecommender class."""

import pytest
from unittest.mock import patch, MagicMock
from baml_client.types import OntologyRecommendation, OntologyExtension, EntityType, RelationType

from spindle import OntologyRecommender, SpindleExtractor
from tests.fixtures.sample_texts import SIMPLE_TEXT, MEDICAL_TEXT


class TestOntologyRecommenderRecommend:
    """Tests for OntologyRecommender.recommend method."""
    
    @patch('spindle.extractor.b.RecommendOntology')
    def test_recommend_basic(self, mock_baml_recommend, mock_ontology_recommendation):
        """Test basic ontology recommendation."""
        mock_baml_recommend.return_value = mock_ontology_recommendation
        
        recommender = OntologyRecommender()
        result = recommender.recommend(text=SIMPLE_TEXT)
        
        # Verify result
        assert result == mock_ontology_recommendation
        assert result.ontology is not None
        assert result.text_purpose is not None
        assert result.reasoning is not None
        
        # Verify BAML was called
        mock_baml_recommend.assert_called_once_with(
            text=SIMPLE_TEXT,
            scope="balanced"
        )
    
    @patch('spindle.extractor.b.RecommendOntology')
    def test_recommend_with_minimal_scope(self, mock_baml_recommend, mock_ontology_recommendation):
        """Test recommendation with minimal scope."""
        mock_baml_recommend.return_value = mock_ontology_recommendation
        
        recommender = OntologyRecommender()
        result = recommender.recommend(text=SIMPLE_TEXT, scope="minimal")
        
        # Verify BAML was called with correct scope
        mock_baml_recommend.assert_called_once_with(
            text=SIMPLE_TEXT,
            scope="minimal"
        )
    
    @patch('spindle.extractor.b.RecommendOntology')
    def test_recommend_with_comprehensive_scope(self, mock_baml_recommend, mock_ontology_recommendation):
        """Test recommendation with comprehensive scope."""
        mock_baml_recommend.return_value = mock_ontology_recommendation
        
        recommender = OntologyRecommender()
        result = recommender.recommend(text=SIMPLE_TEXT, scope="comprehensive")
        
        # Verify BAML was called with correct scope
        mock_baml_recommend.assert_called_once_with(
            text=SIMPLE_TEXT,
            scope="comprehensive"
        )


class TestOntologyRecommenderRecommendAndExtract:
    """Tests for OntologyRecommender.recommend_and_extract method."""
    
    @patch('spindle.SpindleExtractor.extract')
    @patch('spindle.extractor.b.RecommendOntology')
    def test_recommend_and_extract_basic(self, mock_baml_recommend, mock_extract, 
                                         mock_ontology_recommendation, mock_extraction_result):
        """Test combined recommendation and extraction."""
        mock_baml_recommend.return_value = mock_ontology_recommendation
        mock_extract.return_value = mock_extraction_result
        
        recommender = OntologyRecommender()
        recommendation, extraction = recommender.recommend_and_extract(
            text=SIMPLE_TEXT,
            source_name="Test"
        )
        
        # Verify results
        assert recommendation == mock_ontology_recommendation
        assert extraction == mock_extraction_result
        
        # Verify calls
        mock_baml_recommend.assert_called_once()
        mock_extract.assert_called_once()
    
    @patch('spindle.SpindleExtractor.extract')
    @patch('spindle.extractor.b.RecommendOntology')
    def test_recommend_and_extract_with_url(self, mock_baml_recommend, mock_extract,
                                            mock_ontology_recommendation, mock_extraction_result):
        """Test recommend and extract with source URL."""
        mock_baml_recommend.return_value = mock_ontology_recommendation
        mock_extract.return_value = mock_extraction_result
        
        recommender = OntologyRecommender()
        recommendation, extraction = recommender.recommend_and_extract(
            text=SIMPLE_TEXT,
            source_name="Test",
            source_url="https://example.com/test"
        )
        
        # Verify extract was called with URL
        call_args = mock_extract.call_args
        assert call_args[1]["source_url"] == "https://example.com/test"
    
    @patch('spindle.SpindleExtractor.extract')
    @patch('spindle.extractor.b.RecommendOntology')
    def test_recommend_and_extract_with_existing_triples(self, mock_baml_recommend, mock_extract,
                                                          mock_ontology_recommendation, 
                                                          mock_extraction_result, sample_triples):
        """Test recommend and extract with existing triples."""
        mock_baml_recommend.return_value = mock_ontology_recommendation
        mock_extract.return_value = mock_extraction_result
        
        recommender = OntologyRecommender()
        recommendation, extraction = recommender.recommend_and_extract(
            text=SIMPLE_TEXT,
            source_name="Test",
            existing_triples=sample_triples
        )
        
        # Verify extract was called with existing triples
        call_args = mock_extract.call_args
        assert call_args[1]["existing_triples"] == sample_triples
    
    @patch('spindle.SpindleExtractor.extract')
    @patch('spindle.extractor.b.RecommendOntology')
    def test_recommend_and_extract_uses_recommended_ontology(self, mock_baml_recommend, mock_extract,
                                                              mock_ontology_recommendation, 
                                                              mock_extraction_result):
        """Test that extraction uses the recommended ontology."""
        mock_baml_recommend.return_value = mock_ontology_recommendation
        mock_extract.return_value = mock_extraction_result
        
        recommender = OntologyRecommender()
        recommendation, extraction = recommender.recommend_and_extract(
            text=SIMPLE_TEXT,
            source_name="Test",
            scope="comprehensive"
        )
        
        # Verify recommend was called with correct scope
        mock_baml_recommend.assert_called_once_with(text=SIMPLE_TEXT, scope="comprehensive")


class TestOntologyRecommenderAnalyzeExtension:
    """Tests for OntologyRecommender.analyze_extension method."""
    
    @patch('spindle.extractor.b.AnalyzeOntologyExtension')
    def test_analyze_extension_needed(self, mock_baml_analyze, mock_ontology_extension_needed, simple_ontology):
        """Test analyzing when extension is needed."""
        mock_baml_analyze.return_value = mock_ontology_extension_needed
        
        recommender = OntologyRecommender()
        result = recommender.analyze_extension(
            text=MEDICAL_TEXT,
            current_ontology=simple_ontology
        )
        
        # Verify result
        assert result.needs_extension is True
        assert len(result.new_entity_types) > 0
        assert len(result.new_relation_types) > 0
        assert len(result.critical_information_at_risk) > 0
        
        # Verify BAML was called
        mock_baml_analyze.assert_called_once_with(
            text=MEDICAL_TEXT,
            current_ontology=simple_ontology,
            scope="balanced"
        )
    
    @patch('spindle.extractor.b.AnalyzeOntologyExtension')
    def test_analyze_extension_not_needed(self, mock_baml_analyze, mock_ontology_extension_not_needed, simple_ontology):
        """Test analyzing when extension is not needed."""
        mock_baml_analyze.return_value = mock_ontology_extension_not_needed
        
        recommender = OntologyRecommender()
        result = recommender.analyze_extension(
            text=SIMPLE_TEXT,
            current_ontology=simple_ontology
        )
        
        # Verify result
        assert result.needs_extension is False
        assert len(result.new_entity_types) == 0
        assert len(result.new_relation_types) == 0
    
    @patch('spindle.extractor.b.AnalyzeOntologyExtension')
    def test_analyze_extension_with_scope(self, mock_baml_analyze, mock_ontology_extension_needed, simple_ontology):
        """Test analyzing with custom scope."""
        mock_baml_analyze.return_value = mock_ontology_extension_needed
        
        recommender = OntologyRecommender()
        result = recommender.analyze_extension(
            text=MEDICAL_TEXT,
            current_ontology=simple_ontology,
            scope="comprehensive"
        )
        
        # Verify BAML was called with correct scope
        mock_baml_analyze.assert_called_once_with(
            text=MEDICAL_TEXT,
            current_ontology=simple_ontology,
            scope="comprehensive"
        )


class TestOntologyRecommenderExtendOntology:
    """Tests for OntologyRecommender.extend_ontology method."""
    
    def test_extend_ontology_adds_types(self, simple_ontology, mock_ontology_extension_needed):
        """Test that extend_ontology adds new types."""
        recommender = OntologyRecommender()
        extended = recommender.extend_ontology(simple_ontology, mock_ontology_extension_needed)
        
        # Verify new types were added
        original_entity_count = len(simple_ontology.entity_types)
        original_relation_count = len(simple_ontology.relation_types)
        
        assert len(extended.entity_types) == original_entity_count + 2  # Added 2 new entity types
        assert len(extended.relation_types) == original_relation_count + 1  # Added 1 new relation type
    
    def test_extend_ontology_preserves_original(self, simple_ontology, mock_ontology_extension_needed):
        """Test that extend_ontology preserves original types."""
        recommender = OntologyRecommender()
        extended = recommender.extend_ontology(simple_ontology, mock_ontology_extension_needed)
        
        # Verify original types are present
        original_entity_names = {et.name for et in simple_ontology.entity_types}
        extended_entity_names = {et.name for et in extended.entity_types}
        
        assert original_entity_names.issubset(extended_entity_names)
    
    def test_extend_ontology_no_extension_needed(self, simple_ontology, mock_ontology_extension_not_needed):
        """Test extending when no extension is needed."""
        recommender = OntologyRecommender()
        extended = recommender.extend_ontology(simple_ontology, mock_ontology_extension_not_needed)
        
        # Should have same number of types
        assert len(extended.entity_types) == len(simple_ontology.entity_types)
        assert len(extended.relation_types) == len(simple_ontology.relation_types)
    
    def test_extend_ontology_does_not_modify_original(self, simple_ontology, mock_ontology_extension_needed):
        """Test that extending doesn't modify the original ontology."""
        original_entity_count = len(simple_ontology.entity_types)
        original_relation_count = len(simple_ontology.relation_types)
        
        recommender = OntologyRecommender()
        extended = recommender.extend_ontology(simple_ontology, mock_ontology_extension_needed)
        
        # Original should be unchanged
        assert len(simple_ontology.entity_types) == original_entity_count
        assert len(simple_ontology.relation_types) == original_relation_count


class TestOntologyRecommenderAnalyzeAndExtend:
    """Tests for OntologyRecommender.analyze_and_extend method."""
    
    @patch('spindle.extractor.b.AnalyzeOntologyExtension')
    def test_analyze_and_extend_with_auto_apply(self, mock_baml_analyze, 
                                                 mock_ontology_extension_needed, simple_ontology):
        """Test analyze and extend with auto_apply=True."""
        mock_baml_analyze.return_value = mock_ontology_extension_needed
        
        recommender = OntologyRecommender()
        extension, extended_ontology = recommender.analyze_and_extend(
            text=MEDICAL_TEXT,
            current_ontology=simple_ontology,
            auto_apply=True
        )
        
        # Verify results
        assert extension.needs_extension is True
        assert extended_ontology is not None
        assert len(extended_ontology.entity_types) > len(simple_ontology.entity_types)
    
    @patch('spindle.extractor.b.AnalyzeOntologyExtension')
    def test_analyze_and_extend_without_auto_apply(self, mock_baml_analyze,
                                                    mock_ontology_extension_needed, simple_ontology):
        """Test analyze and extend with auto_apply=False."""
        mock_baml_analyze.return_value = mock_ontology_extension_needed
        
        recommender = OntologyRecommender()
        extension, extended_ontology = recommender.analyze_and_extend(
            text=MEDICAL_TEXT,
            current_ontology=simple_ontology,
            auto_apply=False
        )
        
        # Verify results
        assert extension.needs_extension is True
        assert extended_ontology is None  # Not applied
    
    @patch('spindle.extractor.b.AnalyzeOntologyExtension')
    def test_analyze_and_extend_no_extension_needed(self, mock_baml_analyze,
                                                     mock_ontology_extension_not_needed, simple_ontology):
        """Test analyze and extend when no extension is needed."""
        mock_baml_analyze.return_value = mock_ontology_extension_not_needed
        
        recommender = OntologyRecommender()
        extension, extended_ontology = recommender.analyze_and_extend(
            text=SIMPLE_TEXT,
            current_ontology=simple_ontology,
            auto_apply=True
        )
        
        # Verify results
        assert extension.needs_extension is False
        assert extended_ontology is None  # No extension applied
    
    @patch('spindle.extractor.b.AnalyzeOntologyExtension')
    def test_analyze_and_extend_with_scope(self, mock_baml_analyze,
                                           mock_ontology_extension_needed, simple_ontology):
        """Test analyze and extend with custom scope."""
        mock_baml_analyze.return_value = mock_ontology_extension_needed
        
        recommender = OntologyRecommender()
        extension, extended_ontology = recommender.analyze_and_extend(
            text=MEDICAL_TEXT,
            current_ontology=simple_ontology,
            scope="minimal"
        )
        
        # Verify BAML was called with correct scope
        mock_baml_analyze.assert_called_once_with(
            text=MEDICAL_TEXT,
            current_ontology=simple_ontology,
            scope="minimal"
        )

