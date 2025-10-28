# Spindle Changes - Source Metadata & Supporting Spans

## Overview

Major update to add provenance tracking and evidence support to extracted triples.

## Key Changes

### 1. Source Metadata Tracking

Every triple now includes information about where it came from:

- **Source Name**: Identifier for the document/source (required)
- **Source URL**: Optional URL of the source document

**Benefits:**
- Track provenance of every fact
- Enable source-based filtering and queries
- Support cross-validation across multiple sources

### 2. Supporting Character Spans

Each triple includes character spans pointing to the text that supports it:

- **Start index**: Beginning of the evidence text (0-based)
- **End index**: End of the evidence text (exclusive, like Python slicing)
- **Text content**: The actual supporting text

**Benefits:**
- Verify extraction accuracy
- Provide citations and evidence
- Enable fact-checking and validation
- Support explainable AI requirements

### 3. Multi-Source Duplicate Handling

**New behavior:** Duplicate triples from different sources are now **allowed**.

**Rationale:**
- Cross-validation: Same fact from multiple sources increases confidence
- Provenance: Track all occurrences across different documents
- Evidence aggregation: Collect all supporting evidence for a fact

**Example:**
```python
# Source 1: Company website
triple1 = (Alice, works_at, TechCorp) from "Company Website"

# Source 2: LinkedIn profile  
triple2 = (Alice, works_at, TechCorp) from "LinkedIn"

# Both are kept! They confirm each other.
```

## Technical Changes

### BAML Schema (`baml_src/spindle.baml`)

**New Classes:**
```baml
class SourceMetadata {
  source_name string
  source_url string?
}

class CharacterSpan {
  start int
  end int
  text string
}
```

**Updated Triple Class:**
```baml
class Triple {
  subject string
  predicate string
  object string
  source SourceMetadata              // NEW
  supporting_spans CharacterSpan[]    // NEW
}
```

**Updated Function Signature:**
```baml
function ExtractTriples(
  text: string,
  ontology: Ontology,
  source_metadata: SourceMetadata,   // NEW
  existing_triples: Triple[]
) -> ExtractionResult
```

### Python API (`spindle.py`)

**Updated `SpindleExtractor.extract()` signature:**
```python
def extract(
    self,
    text: str,
    source_name: str,              # NEW - required
    source_url: Optional[str] = None,  # NEW - optional
    existing_triples: List[Triple] = None
) -> ExtractionResult
```

**New Helper Functions:**
- `create_source_metadata(source_name, source_url)`: Create SourceMetadata objects
- `get_supporting_text(triple)`: Extract text from character spans
- `filter_triples_by_source(triples, source_name)`: Filter by source

**Updated Helper Functions:**
- `triples_to_dict()`: Now includes source and supporting_spans
- `dict_to_triples()`: Now handles source and supporting_spans

## Migration Guide

### Before (Old API)

```python
result = extractor.extract(
    text="Alice works at TechCorp."
)
```

### After (New API)

```python
result = extractor.extract(
    text="Alice works at TechCorp.",
    source_name="Employee Database",
    source_url="https://example.com/db/employees"  # optional
)
```

### Accessing New Fields

```python
for triple in result.triples:
    # Basic triple info (unchanged)
    print(f"{triple.subject} -> {triple.predicate} -> {triple.object}")
    
    # NEW: Source information
    print(f"Source: {triple.source.source_name}")
    if triple.source.source_url:
        print(f"URL: {triple.source.source_url}")
    
    # NEW: Supporting evidence
    for span in triple.supporting_spans:
        print(f"Evidence [{span.start}:{span.end}]: {span.text}")
```

## Behavioral Changes

### 1. Duplicate Handling

**Old Behavior:**
- Duplicates were prevented across all extractions
- Same fact extracted once even from multiple sources

**New Behavior:**
- Duplicates allowed if from different sources
- Same fact can appear multiple times with different source metadata
- Use `source_name` to distinguish sources

### 2. Entity Consistency

**Unchanged:**
- Entity names remain consistent across sources
- If "Alice Johnson" appears in source 1, source 2 uses "Alice Johnson" not "Alice"

**Enhanced:**
- Consistency maintained even when allowing duplicates
- LLM recognizes entities across all existing triples regardless of source

## Use Cases Enabled

### 1. Cross-Source Validation

```python
# Extract from multiple sources
wiki_result = extractor.extract(text1, "Wikipedia", existing_triples=[])
news_result = extractor.extract(text2, "News Article", existing_triples=wiki_result.triples)
blog_result = extractor.extract(text3, "Blog Post", existing_triples=wiki_result.triples + news_result.triples)

# Find facts confirmed by multiple sources
all_triples = wiki_result.triples + news_result.triples + blog_result.triples
fact_counts = {}
for triple in all_triples:
    fact = (triple.subject, triple.predicate, triple.object)
    fact_counts[fact] = fact_counts.get(fact, 0) + 1

confirmed_facts = {f: count for f, count in fact_counts.items() if count > 1}
print(f"Facts confirmed by multiple sources: {len(confirmed_facts)}")
```

### 2. Provenance Tracking

```python
# Query: Where did we learn that Alice works at TechCorp?
for triple in all_triples:
    if (triple.subject == "Alice Johnson" and 
        triple.predicate == "works_at" and 
        triple.object == "TechCorp"):
        print(f"Found in: {triple.source.source_name}")
        if triple.source.source_url:
            print(f"  URL: {triple.source.source_url}")
```

### 3. Evidence-Based Fact Checking

```python
# Show evidence for each extracted fact
for triple in result.triples:
    print(f"\nFact: {triple.subject} {triple.predicate} {triple.object}")
    print("Evidence:")
    for i, span in enumerate(triple.supporting_spans, 1):
        # Extract context around the span
        context_start = max(0, span.start - 50)
        context_end = min(len(text), span.end + 50)
        context = text[context_start:context_end]
        print(f"  {i}. ...{context}...")
```

### 4. Source-Specific Analysis

```python
from spindle import filter_triples_by_source

# Analyze coverage by source
sources = set(t.source.source_name for t in all_triples)
for source in sources:
    source_triples = filter_triples_by_source(all_triples, source)
    print(f"{source}: {len(source_triples)} triples")
```

## Example Output

```
Triple: Alice Johnson --[works_at]--> TechCorp
  Source: TechCorp Company Profile 2024
  URL: https://example.com/techcorp/profile
  Supporting evidence (2 spans):
    1. [5:85] "Alice Johnson is a senior software engineer who works at TechCorp"
    2. [300:350] "Alice has been with TechCorp since its early days"
```

## Performance Considerations

- **Storage**: Triples now include additional metadata (source + spans)
- **Serialization**: JSON output is larger due to character spans
- **LLM Processing**: Slightly longer prompts due to source metadata
- **Extraction Time**: Character span identification may add small overhead

## Breaking Changes

⚠️ **API Change**: `extract()` now requires `source_name` parameter

**Migration:**
1. Update all `extract()` calls to include `source_name`
2. Optionally add `source_url` for better provenance
3. Update code that accesses triple fields to handle new structure
4. Regenerate BAML client: `baml-cli generate`

## Backward Compatibility

**Not backward compatible** - this is a breaking change requiring code updates.

**Why:** The benefits of provenance tracking and evidence support are fundamental to knowledge graph quality and justify the breaking change.

## Testing Recommendations

1. **Unit tests**: Verify source metadata is correctly attached
2. **Integration tests**: Test multi-source extraction with duplicates
3. **Character span tests**: Validate span accuracy and coverage
4. **Source filtering tests**: Verify filter_triples_by_source()
5. **Serialization tests**: Test triples_to_dict/dict_to_triples roundtrip

## Documentation Updates

- ✅ README.md: Updated with new features and API
- ✅ QUICKSTART.md: Updated examples with source metadata
- ✅ example.py: Comprehensive demonstration of new features
- ✅ spindle.py: Updated docstrings
- ✅ CHANGES.md: This document

## Future Enhancements

Building on this foundation:

1. **Confidence scores**: Add confidence to each triple based on evidence quality
2. **Span highlighting**: UI components for visualizing evidence
3. **Source reliability**: Weight facts by source trustworthiness  
4. **Temporal tracking**: Add extraction timestamps
5. **Conflict resolution**: Handle contradictory facts from different sources
6. **Evidence aggregation**: Combine evidence across duplicate triples
7. **Citation generation**: Auto-generate citations from source metadata

## Questions?

For issues or questions about these changes, please refer to:
- README.md for usage examples
- example.py for complete demonstrations
- QUICKSTART.md for getting started

