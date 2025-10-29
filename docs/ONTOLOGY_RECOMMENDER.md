# Ontology Recommender Service

## Overview

The **OntologyRecommender** is a new service in Spindle that automatically analyzes text to recommend appropriate ontologies for knowledge graph extraction. Instead of manually defining entity and relation types, you can let the system infer the best ontology structure based on the content and domain of your text.

## Key Features

- **Automatic Domain Detection**: Analyzes text to understand its overarching purpose and domain
- **Smart Type Inference**: Recommends entity and relation types that are relevant to the text
- **Ready-to-Use Output**: Returns an `Ontology` object that can be directly used with `SpindleExtractor`
- **Flexible Constraints**: Control the complexity by setting max entity and relation types
- **One-Step Workflow**: Optional combined recommendation + extraction method

## How It Works

The service uses an LLM to:

1. **Analyze the text** to understand its domain, purpose, and key concepts
2. **Identify entities** that appear or are implied in the text
3. **Determine relationships** between these entities
4. **Design entity types** that are general enough for reuse but specific enough to be meaningful
5. **Design relation types** with proper domain/range constraints
6. **Provide reasoning** explaining the ontology design choices

## Usage

### Basic Recommendation

```python
from spindle import OntologyRecommender

recommender = OntologyRecommender()

text = """
Dr. Sarah Chen led a clinical trial at Massachusetts General Hospital 
to evaluate Medication A for treating chronic migraines...
"""

recommendation = recommender.recommend(
    text=text,
    max_entity_types=8,
    max_relation_types=10
)

# Examine the recommendation
print(f"Purpose: {recommendation.text_purpose}")
print(f"Entity Types: {[et.name for et in recommendation.ontology.entity_types]}")
print(f"Reasoning: {recommendation.reasoning}")

# Use the recommended ontology
from spindle import SpindleExtractor
extractor = SpindleExtractor(recommendation.ontology)
result = extractor.extract(text, source_name="Medical Research")
```

### One-Step Recommendation + Extraction

```python
recommender = OntologyRecommender()

recommendation, extraction = recommender.recommend_and_extract(
    text=your_text,
    source_name="Document Name",
    max_entity_types=6,
    max_relation_types=8
)

print(f"Purpose: {recommendation.text_purpose}")
print(f"Extracted {len(extraction.triples)} triples")
```

## API

### `OntologyRecommender.recommend()`

```python
def recommend(
    text: str,
    max_entity_types: int = 10,
    max_relation_types: int = 15
) -> OntologyRecommendation
```

**Parameters:**
- `text`: The text to analyze for ontology recommendation
- `max_entity_types`: Maximum number of entity types to recommend (default: 10)
- `max_relation_types`: Maximum number of relation types to recommend (default: 15)

**Returns:** `OntologyRecommendation` with:
- `ontology`: The recommended `Ontology` object
- `text_purpose`: Analysis of the text's overarching purpose/goal
- `reasoning`: Explanation of the recommendation choices

### `OntologyRecommender.recommend_and_extract()`

```python
def recommend_and_extract(
    text: str,
    source_name: str,
    source_url: Optional[str] = None,
    max_entity_types: int = 10,
    max_relation_types: int = 15,
    existing_triples: List[Triple] = None
) -> Tuple[OntologyRecommendation, ExtractionResult]
```

**Parameters:**
- `text`: The text to analyze and extract from
- `source_name`: Name or identifier of the source document
- `source_url`: Optional URL of the source document
- `max_entity_types`: Maximum number of entity types to recommend
- `max_relation_types`: Maximum number of relation types to recommend
- `existing_triples`: Optional list of previously extracted triples

**Returns:** Tuple of (`OntologyRecommendation`, `ExtractionResult`)

## When to Use

### Use OntologyRecommender when:

- You're exploring a new domain and don't know what ontology to use
- You want to quickly prototype knowledge extraction for different text types
- The text domain is clear but you want to save time on ontology design
- You're processing diverse texts and want domain-specific ontologies for each

### Use Manual Ontology when:

- You have a well-established domain model or schema
- You need precise control over entity and relation types
- You're working with a standardized knowledge representation
- You want consistent extraction across a large corpus of similar texts

## Implementation Details

### BAML Function

The service is powered by the `RecommendOntology` BAML function defined in `baml_src/spindle.baml`:

```baml
class OntologyRecommendation {
  ontology Ontology
  text_purpose string
  reasoning string
}

function RecommendOntology(
  text: string,
  max_entity_types: int,
  max_relation_types: int
) -> OntologyRecommendation
```

### Python Wrapper

The `OntologyRecommender` class in `spindle.py` provides a clean interface:
- Wraps the BAML function for easy use
- Provides the convenience `recommend_and_extract()` method
- Handles integration with `SpindleExtractor`

### Output Format

The ontology is returned in the exact same format as manually created ontologies:
- `EntityType` objects with name and description
- `RelationType` objects with name, description, domain, and range
- Can be directly used with `SpindleExtractor`

## Examples

### Example 1: Medical Research Text

**Input:** Clinical trial description

**Recommended Entity Types:**
- Patient
- Medication
- Hospital
- Researcher
- Condition

**Recommended Relation Types:**
- treats (Medication → Condition)
- led_by (Trial → Researcher)
- conducted_at (Trial → Hospital)
- enrolled_in (Patient → Trial)

### Example 2: Business/Technology Text

**Input:** Startup funding announcement

**Recommended Entity Types:**
- Person
- Company
- Investment
- Location
- Technology

**Recommended Relation Types:**
- founded (Person → Company)
- invested_in (Company → Company)
- located_in (Company → Location)
- develops (Company → Technology)

## Files

- `baml_src/spindle.baml`: BAML schema with `RecommendOntology` function
- `spindle.py`: Python wrapper with `OntologyRecommender` class
- `example_ontology_recommender.py`: Complete usage example
- `test_ontology_recommender.py`: Quick test script

## Testing

Run the example:
```bash
python example_ontology_recommender.py
```

Run the quick test:
```bash
python test_ontology_recommender.py
```

## Future Enhancements

Potential improvements for the ontology recommender:
- Support for providing example texts or seed concepts
- Ability to merge/combine recommended ontologies
- Ontology refinement based on extraction results
- Support for hierarchical entity types
- Integration with standard ontology formats (OWL, RDF)

