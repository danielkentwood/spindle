# Spindle

A tool for real-time extraction of knowledge graphs from multimodal data using BAML and LLMs.

## Overview

Spindle extracts structured knowledge graph triples (subject-predicate-object relationships) from text using Large Language Models. It supports ontology-based extraction with entity consistency across multiple extraction runs.

## MVP Features

The current MVP provides:

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

### Basic Example

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

### Running the Example

A complete example is provided in `example.py`:

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
├── baml_src/              # BAML schema definitions
│   ├── clients.baml       # LLM client configurations
│   ├── generators.baml    # Code generation config
│   ├── resume.baml        # Example BAML function
│   └── spindle.baml       # Knowledge graph extraction schema
├── baml_client/           # Auto-generated BAML Python client
├── spindle.py             # Main Python interface
├── example.py             # Usage example
├── requirements.txt       # Python dependencies
├── overview.md            # Project overview
└── README.md              # This file
```

## API Reference

### `SpindleExtractor`

Main interface for triple extraction.

**Methods:**
- `__init__(ontology: Ontology)`: Initialize with an ontology
- `extract(text: str, source_name: str, source_url: Optional[str] = None, existing_triples: List[Triple] = None) -> ExtractionResult`: Extract triples from text with source metadata

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

