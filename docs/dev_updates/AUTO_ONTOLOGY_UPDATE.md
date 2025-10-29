# Update: Auto-Ontology in SpindleExtractor

## Date
October 28, 2025

## Overview
Enhanced `SpindleExtractor` to support automatic ontology recommendation when no ontology is provided at initialization. This makes Spindle even easier to use - you can now extract triples without defining an ontology upfront.

## Changes

### Modified: `SpindleExtractor` Class

**File:** `spindle.py`

#### 1. Optional Ontology Parameter

**Before:**
```python
def __init__(self, ontology: Ontology):
    """Initialize the extractor with an ontology."""
    self.ontology = ontology
```

**After:**
```python
def __init__(self, ontology: Optional[Ontology] = None):
    """
    Initialize the extractor with an ontology.
    
    Args:
        ontology: Optional Ontology object. If None, an ontology will be 
                 automatically recommended from the text when extract() is first called.
    """
    self.ontology = ontology
    self._ontology_recommender = None if ontology is not None else OntologyRecommender()
```

**Key Changes:**
- `ontology` parameter is now `Optional` and defaults to `None`
- If no ontology provided, creates an internal `OntologyRecommender` instance
- Non-breaking change: existing code still works exactly as before

#### 2. Auto-Recommendation in extract() Method

**Added Parameters:**
```python
def extract(
    self,
    text: str,
    source_name: str,
    source_url: Optional[str] = None,
    existing_triples: List[Triple] = None,
    max_entity_types: int = 10,          # NEW
    max_relation_types: int = 15         # NEW
) -> ExtractionResult:
```

**New Logic:**
```python
# Auto-recommend ontology if not provided
if self.ontology is None:
    recommendation = self._ontology_recommender.recommend(
        text=text,
        max_entity_types=max_entity_types,
        max_relation_types=max_relation_types
    )
    self.ontology = recommendation.ontology
```

**Key Features:**
- Checks if `self.ontology` is `None` before extraction
- If `None`, automatically recommends an ontology from the text
- Uses `max_entity_types` and `max_relation_types` to control ontology size
- Stores the recommended ontology in `self.ontology` for reuse
- All subsequent `extract()` calls use the same ontology

## Usage Examples

### Example 1: Simplest Usage (New!)

```python
from spindle import SpindleExtractor

# No ontology needed!
extractor = SpindleExtractor()

# Just extract - ontology will be auto-recommended
result = extractor.extract(
    text="Alice works at Google in Mountain View...",
    source_name="Company Data"
)

# Ontology is now available for inspection
print(extractor.ontology.entity_types)
```

### Example 2: With Custom Ontology (Still Works!)

```python
from spindle import SpindleExtractor, create_ontology

# Define ontology manually (existing workflow)
ontology = create_ontology(entity_types, relation_types)
extractor = SpindleExtractor(ontology)

# Works exactly as before
result = extractor.extract(text, source_name="Doc")
```

### Example 3: Control Auto-Recommendation Size

```python
from spindle import SpindleExtractor

extractor = SpindleExtractor()

# Control the complexity of the auto-recommended ontology
result = extractor.extract(
    text=complex_text,
    source_name="Doc",
    max_entity_types=5,      # Keep it simple
    max_relation_types=8
)
```

### Example 4: Multi-Document with Auto-Ontology

```python
from spindle import SpindleExtractor

extractor = SpindleExtractor()

# First document - ontology auto-recommended
result1 = extractor.extract(text1, source_name="Doc1")

# Second document - reuses same ontology, maintains entity consistency
result2 = extractor.extract(
    text2, 
    source_name="Doc2",
    existing_triples=result1.triples
)

# All using the same auto-recommended ontology!
```

## Benefits

### 1. Ease of Use
- **Before**: Had to manually define entity and relation types
- **After**: Can start extracting immediately with zero configuration

### 2. Flexibility
- **Option A**: No ontology → Auto-recommend from text
- **Option B**: Provide ontology → Use as specified
- **Option C**: Use `OntologyRecommender` explicitly → More control

### 3. Consistency
- Auto-recommended ontology is stored and reused
- Maintains entity consistency across multiple documents
- Same behavior as manually-defined ontologies

### 4. Backward Compatibility
- Existing code works without any changes
- All parameters are optional with sensible defaults
- No breaking changes to the API

## Comparison of Approaches

### Approach 1: Auto-Ontology in SpindleExtractor (NEW)
```python
extractor = SpindleExtractor()
result = extractor.extract(text, source_name="Doc")
```
**Best for:** Quick prototyping, exploring new domains, simple use cases

### Approach 2: Explicit OntologyRecommender
```python
recommender = OntologyRecommender()
recommendation = recommender.recommend(text)
print(recommendation.text_purpose)  # View analysis
extractor = SpindleExtractor(recommendation.ontology)
result = extractor.extract(text, source_name="Doc")
```
**Best for:** When you want to inspect/understand the recommendation first

### Approach 3: One-Step recommend_and_extract()
```python
recommender = OntologyRecommender()
recommendation, extraction = recommender.recommend_and_extract(
    text, source_name="Doc"
)
```
**Best for:** Quick one-off extractions with analysis details

### Approach 4: Manual Ontology (ORIGINAL)
```python
ontology = create_ontology(entity_types, relation_types)
extractor = SpindleExtractor(ontology)
result = extractor.extract(text, source_name="Doc")
```
**Best for:** Production systems, standardized schemas, precise control

## Implementation Details

### Internal Flow

1. User creates `SpindleExtractor()` without ontology
2. `__init__` sets `self.ontology = None` and creates `self._ontology_recommender`
3. User calls `extract(text, ...)`
4. Method checks: `if self.ontology is None:`
5. If true, calls `self._ontology_recommender.recommend(text, ...)`
6. Stores result in `self.ontology`
7. Proceeds with normal extraction using the ontology
8. Future calls to `extract()` skip steps 4-6 (ontology already set)

### Performance Considerations

- **First extract() call**: Slightly slower (ontology recommendation + extraction)
- **Subsequent extract() calls**: Same speed as manual ontology
- **Recommendation overhead**: One additional LLM call per SpindleExtractor instance

### Memory

- Stores one `OntologyRecommender` instance if no ontology provided
- Stores the recommended `Ontology` after first extraction
- Minimal overhead (just the ontology object)

## Files Modified

1. **spindle.py**
   - Modified `SpindleExtractor.__init__()` signature
   - Modified `SpindleExtractor.extract()` signature and logic
   - Updated class docstring

2. **README.md**
   - Added new usage section for auto-ontology
   - Updated API reference
   - Added new example to project structure
   - Updated SpindleExtractor documentation

## New Files

1. **example_auto_ontology.py** (106 lines)
   - Demonstrates SpindleExtractor without ontology
   - Shows auto-recommendation on first extract()
   - Shows reuse for subsequent documents
   - Full example with art museum domain

2. **AUTO_ONTOLOGY_UPDATE.md** (this file)
   - Comprehensive documentation of the update
   - Usage examples and comparisons
   - Implementation details

## Testing

Test the new feature:

```bash
python example_auto_ontology.py
```

## Migration Guide

### No Migration Needed!

Existing code continues to work without changes:

```python
# This still works exactly as before
ontology = create_ontology(entity_types, relation_types)
extractor = SpindleExtractor(ontology)
result = extractor.extract(text, source_name="Doc")
```

### Optional: Simplify Your Code

If you want to use the new feature:

**Before:**
```python
recommender = OntologyRecommender()
recommendation = recommender.recommend(text)
extractor = SpindleExtractor(recommendation.ontology)
result = extractor.extract(text, source_name="Doc")
```

**After:**
```python
extractor = SpindleExtractor()
result = extractor.extract(text, source_name="Doc")
```

## Summary

This update makes Spindle even more user-friendly by:
- ✅ Eliminating the need to manually define ontologies for quick tasks
- ✅ Maintaining backward compatibility (no breaking changes)
- ✅ Providing flexible options (auto, explicit, or manual)
- ✅ Ensuring consistent behavior across all approaches
- ✅ Simplifying the API while keeping power-user features

You can now use Spindle with as little as 3 lines of code:
```python
extractor = SpindleExtractor()
result = extractor.extract(text, source_name="Doc")
print(result.triples)
```

