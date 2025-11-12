# Ontology Recommender Service

## Overview

The **OntologyRecommender** is a new service in Spindle that automatically analyzes text to recommend appropriate ontologies for knowledge graph extraction. Instead of manually defining entity and relation types, you can let the system infer the best ontology structure based on the content and domain of your text.

## Key Features

- **Automatic Domain Detection**: Analyzes text to understand its overarching purpose and domain
- **Smart Type Inference**: Recommends entity and relation types that are relevant to the text
- **Ready-to-Use Output**: Returns an `Ontology` object that can be directly used with `SpindleExtractor`
- **Scope Controls**: Choose `minimal`, `balanced`, or `comprehensive` ontologies depending on your needs
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
    scope="balanced"  # "minimal", "balanced", or "comprehensive"
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

> Tip: Instantiating `SpindleExtractor()` without providing an ontology automatically calls `OntologyRecommender.recommend()` on the first `extract()`.

### One-Step Recommendation + Extraction

```python
recommender = OntologyRecommender()

recommendation, extraction = recommender.recommend_and_extract(
    text=your_text,
    source_name="Document Name",
    scope="minimal"
)

print(f"Purpose: {recommendation.text_purpose}")
print(f"Extracted {len(extraction.triples)} triples")
```

## API

### `OntologyRecommender.recommend()`

```python
def recommend(
    text: str,
    scope: str = "balanced"
) -> OntologyRecommendation
```

**Parameters:**
- `text`: The text to analyze for ontology recommendation
- `scope`: Granularity level for the ontology (`"minimal"`, `"balanced"`, or `"comprehensive"`)

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
    scope: str = "balanced",
    existing_triples: Optional[List[Triple]] = None
) -> Tuple[OntologyRecommendation, ExtractionResult]
```

**Parameters:**
- `text`: The text to analyze and extract from
- `source_name`: Name or identifier of the source document
- `source_url`: Optional URL of the source document
- `scope`: Granularity level (`"minimal"`, `"balanced"`, `"comprehensive"`)
- `existing_triples`: Optional list of previously extracted triples for entity consistency

**Returns:** Tuple of (`OntologyRecommendation`, `ExtractionResult`)

### `OntologyRecommender.analyze_extension()`

```python
def analyze_extension(
    text: str,
    current_ontology: Ontology,
    scope: str = "balanced"
) -> OntologyExtension
```

Assesses whether new text would cause information loss with the current ontology.

**Returns:** `OntologyExtension` detailing:
- `needs_extension`: Whether new types are required
- `new_entity_types` / `new_relation_types`: Suggested additions (if any)
- `critical_information_at_risk`: Rationale for extending
- `reasoning`: Full explanation from the model

### `OntologyRecommender.extend_ontology()`

```python
def extend_ontology(
    current_ontology: Ontology,
    extension: OntologyExtension
) -> Ontology
```

Applies an `OntologyExtension`, producing a new ontology that merges existing and new types.

### `OntologyRecommender.analyze_and_extend()`

```python
def analyze_and_extend(
    text: str,
    current_ontology: Ontology,
    scope: str = "balanced",
    auto_apply: bool = True
) -> Tuple[OntologyExtension, Optional[Ontology]]
```

Combines analysis + application. When `auto_apply=True`, a new ontology is returned if an extension is required; otherwise the second item is `None` and you can keep the original ontology.

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
  scope: string
) -> OntologyRecommendation
```

### Python Wrapper

The `OntologyRecommender` class in `spindle/extraction/recommender.py` provides a clean interface:
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

- `spindle/baml_src/spindle.baml`: BAML schema with ontology + extension functions
- `spindle/extraction/recommender.py`: Python implementation of `OntologyRecommender`
- `demos/example_auto_ontology.py`: Automatic recommendation inside `SpindleExtractor`
- `demos/example_scope_comparison.py`: Demonstrates scope levels
- `demos/example_ontology_extension.py`: Conservative extension workflow
- `tests/test_recommender.py`: Unit tests covering the recommender API

## Testing

Run the demos:
```bash
# Auto-ontology workflow (recommender invoked inside SpindleExtractor)
uv run python demos/example_auto_ontology.py

# Scope comparison across minimal/balanced/comprehensive
uv run python demos/example_scope_comparison.py

# Conservative ontology extension flow
uv run python demos/example_ontology_extension.py
```

Run recommender-specific unit tests:
```bash
uv run pytest tests/test_recommender.py -v
```

## Future Enhancements

Potential improvements for the ontology recommender:
- Support for providing example texts or seed concepts
- Ability to merge/combine recommended ontologies
- Ontology refinement based on extraction results
- Support for hierarchical entity types
- Integration with standard ontology formats (OWL, RDF)

