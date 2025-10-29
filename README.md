# Spindle

A tool for real-time extraction of knowledge graphs from multimodal data using BAML and LLMs.

## Overview

Spindle extracts structured knowledge graph triples (subject-predicate-object relationships) from text using Large Language Models. It supports ontology-based extraction with entity consistency across multiple extraction runs.

## MVP Features

The current MVP provides:

- **Automatic ontology recommendation**: Analyze text to automatically suggest appropriate entity and relation types
- **Conservative ontology extension**: Automatically analyze and extend existing ontologies when processing new sources, only adding types when critical information would be lost
- **Ontology-driven extraction**: Define entity and relation types to guide extraction
- **Source metadata tracking**: Each triple includes source name and optional URL
- **Supporting evidence**: Character spans with text and indices computed in post-processing (whitespace-normalized matching)
- **Temporal tracking**: Extraction datetime automatically set for each triple in ISO 8601 format
- **Incremental extraction**: Build knowledge graphs across multiple texts while maintaining entity consistency
- **Multi-source support**: Duplicate triples allowed from different sources for cross-validation
- **Entity consistency**: Recognizes when entities in new text match existing entities
- **Source filtering**: Query triples by their source
- **Date-based filtering**: Query triples by extraction date range
- **BAML-powered**: Uses the BAML framework for type-safe LLM interactions
- **Claude Sonnet 4**: Leverages Anthropic's Claude for high-quality extraction

## Installation

### Prerequisites

- Python 3.8 or higher
- Anaconda/Miniconda (recommended)
- Anthropic API key

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd spindle
```

2. Create and activate the conda environment (or use existing kgx environment):
```bash
conda create -n spindle python=3.11
conda activate spindle
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your API key:
```bash
# Create a .env file with your Anthropic API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

## Usage

### Automatic Ontology Recommendation (New!)

#### Option 1: Automatic Recommendation in SpindleExtractor (Simplest!)

The easiest way - just create a `SpindleExtractor` without an ontology and it will automatically recommend one based on your text:

```python
from spindle import SpindleExtractor

# Create extractor WITHOUT an ontology
extractor = SpindleExtractor()

# Extract - ontology will be automatically recommended from the text
text = """
Dr. Sarah Chen led a clinical trial at Massachusetts General Hospital 
to evaluate Medication A for treating chronic migraines...
"""

result = extractor.extract(
    text=text,
    source_name="Research Paper",
    ontology_scope="balanced"  # "minimal", "balanced" (default), or "comprehensive"
)

# The ontology is now available and will be reused for future extractions
print(extractor.ontology.entity_types)  # Auto-recommended entity types
```

**Scope Levels:**
- `"minimal"`: Essential concepts only (3-8 entity types, 4-10 relations) - for quick exploration
- `"balanced"`: Standard analysis (6-12 entity types, 8-15 relations) - recommended default
- `"comprehensive"`: Detailed ontology (10-20 entity types, 12-25 relations) - for research/expertise

#### Option 2: Explicit Recommendation with OntologyRecommender

For more control, use the `OntologyRecommender` directly:

```python
from spindle import OntologyRecommender, SpindleExtractor

# Create recommender
recommender = OntologyRecommender()

# Analyze text and get ontology recommendation
text = """
Dr. Sarah Chen led a clinical trial at Massachusetts General Hospital 
to evaluate Medication A for treating chronic migraines...
"""

recommendation = recommender.recommend(text, scope="balanced")

# View the analysis
print(f"Text Purpose: {recommendation.text_purpose}")
print(f"Entity Types: {[et.name for et in recommendation.ontology.entity_types]}")
print(f"Relation Types: {[rt.name for rt in recommendation.ontology.relation_types]}")

# Use the recommended ontology for extraction
extractor = SpindleExtractor(recommendation.ontology)
result = extractor.extract(text, source_name="Research Paper")
```

### One-Step Recommendation + Extraction

For maximum convenience, recommend and extract in one call:

```python
from spindle import OntologyRecommender

recommender = OntologyRecommender()
recommendation, extraction = recommender.recommend_and_extract(
    text=your_text,
    source_name="Document Name",
    scope="balanced"  # Or "minimal" / "comprehensive"
)

print(f"Auto-detected purpose: {recommendation.text_purpose}")
print(f"Extracted {len(extraction.triples)} triples")
```

### Conservative Ontology Extension (New!)

When processing multiple sources, conservatively extend your ontology only when necessary:

```python
from spindle import OntologyRecommender, SpindleExtractor, create_ontology

# Start with an existing ontology (e.g., for business domain)
ontology = create_ontology(business_entities, business_relations)

# Process first document
extractor = SpindleExtractor(ontology)
result1 = extractor.extract(business_text, "Business News")

# New document from different domain - analyze if extension needed
recommender = OntologyRecommender()
new_text = "Dr. Chen prescribed Medication A for treating hypertension..."

extension, extended_ontology = recommender.analyze_and_extend(
    text=new_text,
    current_ontology=ontology,
    scope="balanced",
    auto_apply=True  # Automatically apply if needed
)

if extended_ontology:
    print(f"Extension needed: {extension.critical_information_at_risk}")
    print(f"Added types: {[et.name for et in extension.new_entity_types]}")
    # Use extended ontology for extraction
    extractor = SpindleExtractor(extended_ontology)
else:
    print("No extension needed - using original ontology")
    # Continue with original ontology

result2 = extractor.extract(new_text, "Medical Research")
```

**Key Principles:**
- Extensions are **conservative** - only when critical information would be lost
- Existing types are preferred over creating new ones
- Extensions are backward-compatible additions
- Original ontology is never modified - new ontology is created

### Basic Example with Manual Ontology

```python
from spindle import SpindleExtractor, create_ontology

# Define your ontology
entity_types = [
    {"name": "Person", "description": "A human being"},
    {"name": "Organization", "description": "A company or institution"}
]

relation_types = [
    {
        "name": "works_at",
        "description": "Employment relationship",
        "domain": "Person",
        "range": "Organization"
    }
]

ontology = create_ontology(entity_types, relation_types)

# Create extractor
extractor = SpindleExtractor(ontology)

# Extract triples with source metadata
text = "Alice Johnson works at TechCorp."
result = extractor.extract(
    text=text,
    source_name="Company Directory 2024",
    source_url="https://example.com/directory"
)

# Access triples with metadata
for triple in result.triples:
    print(f"{triple.subject} -> {triple.predicate} -> {triple.object}")
    print(f"  Source: {triple.source.source_name}")
    print(f"  Extracted: {triple.extraction_datetime}")
    print(f"  Evidence: {triple.supporting_spans[0].text}")
```

### Incremental Extraction with Multi-Source Support

```python
# First extraction from source 1
result1 = extractor.extract(
    text="Alice works at TechCorp.",
    source_name="Employee Database",
    existing_triples=[]
)

# Second extraction from source 2 with entity consistency
# Duplicate triples are allowed since it's a different source
result2 = extractor.extract(
    text="Alice Johnson works at TechCorp and uses Python.",
    source_name="Company Blog",
    source_url="https://example.com/blog/team",
    existing_triples=result1.triples
)

# Alice will be recognized as the same entity ("Alice Johnson" used consistently)
# If the same fact appears in both sources, both triples are kept for validation

# Filter by source
from spindle import filter_triples_by_source
blog_triples = filter_triples_by_source(result2.triples, "Company Blog")
```

### Running the Examples

Two complete examples are provided:

**Manual Ontology (`example.py`):**
```bash
python example.py
```

This demonstrates:
- Defining a custom ontology
- Extracting triples from multiple texts with source metadata
- Maintaining entity consistency across sources
- Supporting character spans for evidence
- Handling duplicate triples from different sources
- Filtering triples by source
- Serialization of triples with metadata

**Automatic Ontology Recommendation (`example_ontology_recommender.py`):**
```bash
python example_ontology_recommender.py
```

This demonstrates:
- Automatic ontology recommendation from text analysis
- Text purpose/goal inference
- Domain-appropriate entity and relation type generation
- Using recommended ontologies for extraction
- One-step recommendation + extraction workflow
- JSON serialization of recommendations

**Auto-Ontology in SpindleExtractor (`example_auto_ontology.py`):**
```bash
python example_auto_ontology.py
```

This demonstrates:
- SpindleExtractor without providing an ontology
- Automatic ontology recommendation on first extract() call
- Reusing the same ontology for subsequent extractions
- Entity consistency across multiple texts with auto-recommended ontology

**Scope Comparison (`example_scope_comparison.py`):**
```bash
python example_scope_comparison.py
```

This demonstrates:
- Comparing minimal, balanced, and comprehensive scopes
- How the same text produces different ontologies at different granularities
- Guidance on when to use each scope level
- Extraction comparison across scopes

**Conservative Ontology Extension (`example_ontology_extension.py`):**
```bash
python example_ontology_extension.py
```

This demonstrates:
- Starting with an existing ontology
- Analyzing new text to determine if extension is needed
- Conservative extension principles (only when critical information at risk)
- Applying extensions and extracting with evolved ontology
- When extensions are/aren't needed

## Defining Custom Ontologies

An ontology consists of entity types and relation types:

### Entity Types

```python
entity_types = [
    {
        "name": "Person",
        "description": "A human being, identified by their name"
    },
    {
        "name": "Location",
        "description": "A geographic place, city, or address"
    }
]
```

### Relation Types

```python
relation_types = [
    {
        "name": "located_in",
        "description": "Physical location relationship",
        "domain": "Person",      # Subject must be this entity type
        "range": "Location"      # Object must be this entity type
    }
]
```

## Project Structure

```
spindle/
├── baml_src/                        # BAML schema definitions
│   ├── clients.baml                 # LLM client configurations
│   ├── generators.baml              # Code generation config
│   ├── resume.baml                  # Example BAML function
│   └── spindle.baml                 # Knowledge graph extraction and ontology recommendation
├── baml_client/                     # Auto-generated BAML Python client
├── spindle.py                       # Main Python interface
├── example.py                       # Manual ontology usage example
├── example_ontology_recommender.py  # Explicit ontology recommendation example
├── example_auto_ontology.py         # Auto-ontology in SpindleExtractor example
├── example_scope_comparison.py      # Comparing scope levels (minimal/balanced/comprehensive)
├── example_ontology_extension.py    # Conservative ontology extension for new sources
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## API Reference

### `OntologyRecommender`

Service for automatically recommending ontologies based on text analysis using principled design guidelines.

**Methods:**
- `recommend(text: str, scope: str = "balanced") -> OntologyRecommendation`: Analyze text and recommend an appropriate ontology. Scope must be one of "minimal", "balanced", or "comprehensive".
- `recommend_and_extract(text: str, source_name: str, source_url: Optional[str] = None, scope: str = "balanced", existing_triples: List[Triple] = None) -> Tuple[OntologyRecommendation, ExtractionResult]`: Recommend ontology and extract triples in one step.
- `analyze_extension(text: str, current_ontology: Ontology, scope: str = "balanced") -> OntologyExtension`: Conservatively analyze if an existing ontology needs extension for new text.
- `extend_ontology(current_ontology: Ontology, extension: OntologyExtension) -> Ontology`: Apply an extension to create an extended ontology.
- `analyze_and_extend(text: str, current_ontology: Ontology, scope: str = "balanced", auto_apply: bool = True) -> Tuple[OntologyExtension, Optional[Ontology]]`: Analyze and optionally apply extensions in one step.

**Scope Levels:**
- `"minimal"`: Essential concepts only (typically 3-8 entity types, 4-10 relation types). Use for quick exploration, simple queries, and broad pattern identification.
- `"balanced"`: Standard analysis (typically 6-12 entity types, 8-15 relation types). Recommended default for general-purpose extraction.
- `"comprehensive"`: Detailed ontology (typically 10-20 entity types, 12-25 relation types). Use for detailed analysis, domain expertise, and research.

Note: These are guidelines, not hard limits. The LLM determines the actual number of types based on principled ontology design.

**Returns (OntologyRecommendation):**
- `ontology`: The recommended Ontology object (ready for use with SpindleExtractor)
- `text_purpose`: Analysis of the text's overarching purpose/goal
- `reasoning`: Explanation of why these entity and relation types were recommended

### `SpindleExtractor`

Main interface for triple extraction with a predefined or auto-recommended ontology.

**Methods:**
- `__init__(ontology: Optional[Ontology] = None, ontology_scope: str = "balanced")`: Initialize with an ontology. If None, an ontology will be automatically recommended from the text when extract() is first called, using the specified scope.
- `extract(text: str, source_name: str, source_url: Optional[str] = None, existing_triples: List[Triple] = None, ontology_scope: Optional[str] = None) -> ExtractionResult`: Extract triples from text with source metadata. If no ontology was provided at init, one will be automatically recommended. The `ontology_scope` parameter can override the default scope for this specific extraction.

### `create_ontology()`

Factory function to create ontology objects.

**Parameters:**
- `entity_types`: List of dicts with 'name' and 'description'
- `relation_types`: List of dicts with 'name', 'description', 'domain', and 'range'

**Returns:** `Ontology` object

### `create_source_metadata()`

Create a SourceMetadata object.

**Parameters:**
- `source_name`: Name or identifier of the source
- `source_url`: Optional URL of the source

**Returns:** `SourceMetadata` object

### `triples_to_dict()`

Convert Triple objects to dictionaries for serialization.

**Parameters:**
- `triples`: List of Triple objects

**Returns:** List of dictionaries with all triple fields including source metadata and supporting spans

### `dict_to_triples()`

Convert dictionaries back to Triple objects.

**Parameters:**
- `dicts`: List of dictionaries with triple fields

**Returns:** List of Triple objects

### `get_supporting_text()`

Extract supporting text snippets from a triple's character spans.

**Parameters:**
- `triple`: A Triple object

**Returns:** List of text strings from the supporting spans

### `filter_triples_by_source()`

Filter triples to only those from a specific source.

**Parameters:**
- `triples`: List of Triple objects
- `source_name`: Name of the source to filter by

**Returns:** List of triples from the specified source

### `parse_extraction_datetime()`

Parse the extraction datetime string into a datetime object.

**Parameters:**
- `triple`: A Triple object

**Returns:** `datetime` object, or None if parsing fails

### `filter_triples_by_date_range()`

Filter triples by extraction date range.

**Parameters:**
- `triples`: List of Triple objects
- `start_date`: Optional start datetime (inclusive)
- `end_date`: Optional end datetime (inclusive)

**Returns:** List of triples extracted within the date range

### `ontology_to_dict()`

Convert an Ontology object to a dictionary for serialization.

**Parameters:**
- `ontology`: An Ontology object

**Returns:** Dictionary with 'entity_types' and 'relation_types' keys

### `recommendation_to_dict()`

Convert an OntologyRecommendation to a dictionary for serialization.

**Parameters:**
- `recommendation`: An OntologyRecommendation object

**Returns:** Dictionary with ontology, text_purpose, and reasoning

### `extension_to_dict()`

Convert an OntologyExtension to a dictionary for serialization.

**Parameters:**
- `extension`: An OntologyExtension object

**Returns:** Dictionary with needs_extension, new_entity_types, new_relation_types, critical_information_at_risk, and reasoning

## Architecture

Spindle uses the BAML (Basically, A Made-up Language) framework to define type-safe LLM interactions:

1. **BAML Schema** (`baml_src/spindle.baml`): Defines data structures and the extraction function
2. **Generated Client** (`baml_client/`): Auto-generated Python client for type-safe LLM calls
3. **Python Wrapper** (`spindle.py`): High-level interface for easy use

The extraction process:
1. User defines an ontology (entity and relation types)
2. Text is passed to the LLM along with:
   - The ontology defining valid types
   - Source metadata (name and optional URL)
   - Existing triples from other sources for entity consistency
3. LLM extracts triples conforming to the ontology with:
   - Source metadata attached to each triple
   - Text spans providing supporting evidence (exact quotes from source)
   - Entity names consistent with existing triples
4. Post-processing automatically:
   - Computes accurate character indices for each text span
   - Sets extraction datetime (ISO 8601) for all triples
5. Duplicate triples from different sources are preserved for cross-validation

## Future Work

The MVP is a foundation for the full Spindle system. Future enhancements:

- Multimodal support (images, PDFs, audio)
- Web API/microservice interface
- Multiple LLM providers (OpenAI, local models)
- Graph storage and querying
- Visualization tools
- Real-time streaming extraction

## Requirements

- `baml-py==0.211.2`: BAML framework for Python
- `python-dotenv==1.0.0`: Environment variable management

## License

[Add license information]

## Contributing

[Add contribution guidelines]

