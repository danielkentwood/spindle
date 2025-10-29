# Changes Summary: Ontology Recommender Service

## Date
October 28, 2025

## Overview
Added a new service to Spindle that automatically recommends ontologies by analyzing text and inferring its purpose, domain, and relevant entity/relation types.

## New Features

### 1. OntologyRecommendation Type
- **File**: `baml_src/spindle.baml`
- **Description**: New BAML class to represent ontology recommendations
- **Fields**:
  - `ontology`: The recommended Ontology object
  - `text_purpose`: Analysis of the text's overarching purpose
  - `reasoning`: Explanation of recommendation choices

### 2. RecommendOntology BAML Function
- **File**: `baml_src/spindle.baml`
- **Description**: LLM function that analyzes text and recommends appropriate ontology
- **Parameters**:
  - `text`: Text to analyze
  - `max_entity_types`: Maximum number of entity types to recommend
  - `max_relation_types`: Maximum number of relation types to recommend
- **Returns**: `OntologyRecommendation`
- **Features**:
  - Analyzes text domain and purpose
  - Recommends entity types that are general yet meaningful
  - Designs relation types with proper domain/range constraints
  - Follows ontology design best practices
  - Provides detailed reasoning

### 3. OntologyRecommender Python Class
- **File**: `spindle.py`
- **Description**: Python wrapper for the ontology recommendation service
- **Methods**:
  - `recommend()`: Analyze text and recommend ontology
  - `recommend_and_extract()`: Recommend ontology and extract triples in one step

### 4. Helper Functions
- **File**: `spindle.py`
- **Functions**:
  - `ontology_to_dict()`: Serialize Ontology to dictionary
  - `recommendation_to_dict()`: Serialize OntologyRecommendation to dictionary

### 5. Example Scripts
- **example_ontology_recommender.py**: Comprehensive example demonstrating:
  - Basic ontology recommendation
  - Using recommended ontology for extraction
  - One-step recommendation + extraction workflow
  - JSON serialization
  - Medical and business domain examples
  
- **test_ontology_recommender.py**: Quick test script for verification

### 6. Documentation
- **ONTOLOGY_RECOMMENDER.md**: Complete documentation of the service
- **README.md**: Updated with ontology recommender usage examples and API reference
- **spindle.py**: Updated module docstring

## Modified Files

### baml_src/spindle.baml
- Added `OntologyRecommendation` class definition
- Added `RecommendOntology` function with comprehensive prompt

### spindle.py
- Imported `OntologyRecommendation` type
- Added `OntologyRecommender` class
- Added `ontology_to_dict()` function
- Added `recommendation_to_dict()` function
- Updated module docstring

### README.md
- Added ontology recommendation to MVP features
- Added "Automatic Ontology Recommendation" usage section
- Added "One-Step Recommendation + Extraction" section
- Updated "Running the Examples" section
- Updated API Reference with `OntologyRecommender` documentation
- Added new helper functions to API reference
- Updated project structure

### baml_client/ (auto-generated)
- Regenerated with new types and functions:
  - `OntologyRecommendation` class in types.py
  - `RecommendOntology()` method in sync_client.py and async_client.py

## New Files

1. **example_ontology_recommender.py** (126 lines)
   - Full-featured example script
   - Medical research text example
   - Business/VC investment text example
   - Demonstrates both recommendation workflows

2. **test_ontology_recommender.py** (68 lines)
   - Quick test for basic functionality
   - Simple text example
   - Tests serialization

3. **ONTOLOGY_RECOMMENDER.md** (259 lines)
   - Complete service documentation
   - Usage guide with examples
   - API reference
   - Implementation details
   - When to use guidance

4. **CHANGES_ONTOLOGY_RECOMMENDER.md** (this file)
   - Summary of all changes

## Integration

The new service integrates seamlessly with existing Spindle components:

1. **Input/Output Compatibility**: Returns standard `Ontology` objects that work directly with `SpindleExtractor`
2. **Type Safety**: Uses BAML's type-safe generation for Python client
3. **Consistent API**: Follows same patterns as `SpindleExtractor` 
4. **No Breaking Changes**: All existing code continues to work unchanged

## Usage Patterns

### Pattern 1: Separate Recommendation and Extraction
```python
recommender = OntologyRecommender()
recommendation = recommender.recommend(text)
extractor = SpindleExtractor(recommendation.ontology)
result = extractor.extract(text, source_name="Doc")
```

### Pattern 2: Combined Workflow
```python
recommender = OntologyRecommender()
recommendation, extraction = recommender.recommend_and_extract(
    text=text, 
    source_name="Doc"
)
```

### Pattern 3: Manual Ontology (Existing)
```python
ontology = create_ontology(entity_types, relation_types)
extractor = SpindleExtractor(ontology)
result = extractor.extract(text, source_name="Doc")
```

## Testing

To test the new service:
```bash
# Quick test
python test_ontology_recommender.py

# Full example
python example_ontology_recommender.py
```

## Dependencies

No new dependencies added. Uses existing:
- baml-py==0.211.2
- python-dotenv==1.0.0

## Next Steps / Future Enhancements

Potential improvements:
- Support for providing seed concepts or example texts
- Ontology merging/combining capabilities
- Refinement based on extraction quality
- Hierarchical entity type support
- Export to standard formats (OWL, RDF)
- Caching of recommendations for similar texts
- Interactive ontology refinement

## Notes

- The service uses Claude Sonnet 4 (via CustomSonnet4 client)
- Recommendations are stochastic - same text may yield slightly different ontologies
- Quality depends on text clarity and domain coverage
- Works best with focused, domain-specific texts
- Can handle various domains: medical, business, technical, academic, etc.

