# Scope-Based Ontology Design: Principle Over Numbers

## Overview

Spindle uses a **principle-based approach** to ontology recommendation instead of hard numerical limits. Rather than specifying "max 10 entity types", you specify a **scope level** that guides the LLM to make intelligent decisions about granularity using ontological design principles.

## The Problem with Hard Limits

**Old Approach (Problematic):**
```python
# What does "10" really mean?
recommendation = recommender.recommend(
    text=text,
    max_entity_types=10,
    max_relation_types=15
)
```

**Issues:**
- Numbers are arbitrary and context-independent
- Same limit for a simple email vs. a complex research paper
- Doesn't reflect how ontologists actually think
- User has to guess appropriate numbers
- No semantic guidance on granularity

## The Scope-Based Solution

**New Approach (Principle-Based):**
```python
# Clear semantic intent
recommendation = recommender.recommend(
    text=text,
    scope="balanced"  # or "minimal" or "comprehensive"
)
```

**Benefits:**
- **Semantic clarity**: Intent is clear from the scope name
- **Adaptive**: LLM adjusts to text complexity within scope guidelines
- **Principled**: Based on ontology design best practices
- **Intuitive**: Users think "I need detailed analysis" not "I need 17 types"

## The Three Scopes

### Minimal
**Purpose**: Quick exploration, simple analysis, broad patterns

**Characteristics:**
- Essential concepts only
- Broader categories that combine similar concepts
- Most frequent relationships only
- Typically: 3-8 entity types, 4-10 relation types

**When to Use:**
- Quick data exploration
- Simple dashboards and visualizations
- Prototyping and experimentation
- When you want broad patterns without details
- Resource-constrained environments

**Example:**
```python
# For a business article
extractor = SpindleExtractor(ontology_scope="minimal")
# Might produce: Person, Organization, Location, works_at, located_in
```

### Balanced (Default)
**Purpose**: Standard analysis, general-purpose extraction

**Characteristics:**
- All significant domain concepts included
- Main relationship patterns captured
- Balance between specificity and reusability
- Typically: 6-12 entity types, 8-15 relation types

**When to Use:**
- Most common use case (default)
- General-purpose knowledge extraction
- Standard analytical queries
- When you want good detail without overwhelming complexity
- Production systems with diverse content

**Example:**
```python
# For a business article (default)
extractor = SpindleExtractor()  # Uses "balanced" by default
# Might produce: Person, Organization, Location, Product, Investment, 
#                works_at, located_in, founded, invests_in, develops
```

### Comprehensive
**Purpose**: Detailed analysis, domain expertise, research

**Characteristics:**
- All distinct and meaningful concepts
- Nuanced relationship types
- Domain-specific and specialized types
- Typically: 10-20 entity types, 12-25 relation types

**When to Use:**
- Research and academic analysis
- Domain expertise and specialized knowledge
- Detailed querying and complex analytics
- When nuances and distinctions matter
- Building knowledge bases for specific domains

**Example:**
```python
# For a research paper
extractor = SpindleExtractor(ontology_scope="comprehensive")
# Might produce: Person, Organization, Location, Product, Investment, 
#                Technology, Market, Role, Metric, Event, works_at, 
#                located_in, founded, invests_in, develops, competes_with, 
#                targets, acquired_by, partners_with, measures, participates_in
```

## Design Principles Embedded in Prompts

The scope levels are implemented through comprehensive guidelines in the BAML prompt that teach the LLM how to think about ontology design:

### Entity Type Granularity Principles

1. **Abstraction Level**
   - Too broad: "Thing", "Entity" ❌
   - Too narrow: "SoftwareEngineerAtGoogle" ❌
   - Just right: "Person", "Organization", "JobRole" ✓

2. **Domain Relevance**
   - Include types central to the domain
   - Don't include rare or irrelevant types
   - Focus on what matters for this text

3. **Distinctiveness Test**
   - Only separate if different attributes, relationships, or analytical value
   - Example: "Professor" vs "Student" (distinct relationships) ✓
   - Example: "CEO" and "CTO" → might be "ExecutiveRole" if similar

4. **Generalization**
   - Combine similar concepts when appropriate
   - But keep separate if distinction is analytically important
   - Example: "Device" vs {"Laptop", "Desktop", "Tablet"}

5. **Coverage**
   - Ensure types cover main concepts
   - Don't create types for one-off mentions
   - Do create types for structurally important concepts

### Relation Type Granularity Principles

1. **Semantic Precision**
   - Too vague: "related_to" ❌
   - Too specific: "works_at_as_senior_engineer_since_2020" ❌
   - Just right: "works_at", "manages" ✓

2. **Directionality**
   - Choose direction reflecting real-world semantics
   - "Person works_at Organization" (natural) ✓
   - Be consistent across the ontology

3. **Relationship Patterns**
   - Include relations that appear multiple times
   - Focus on relationships that matter for the domain
   - Every relation should have analytical purpose

4. **Avoid Redundancy**
   - Don't create near-synonyms: "works_at" vs "employed_by" (pick one)
   - Do have semantic distinctions: "founded" vs "acquired" ✓

5. **Completeness**
   - Think: "What questions would users ask?"
   - Include relations that support those questions
   - Balance coverage with simplicity

## Quality Over Quantity

Key principle embedded throughout:

> "It's better to have fewer, well-defined types than many poorly-defined ones"

The LLM is explicitly instructed to:
- Prioritize clarity and usefulness
- Make each type distinct and well-described
- Ensure consistent classification
- Support practical analysis

## Self-Check Questions

The prompt includes reflection questions for the LLM:

1. Can I clearly explain when to use each entity type vs. others?
2. Would two different people classify entities consistently?
3. Are my relation types capturing distinct semantic relationships?
4. Can I extract useful, actionable information with this ontology?
5. Is this ontology reusable for similar texts?
6. Does the granularity match the requested scope level?

## Example Comparison

**Same Text, Three Scopes:**

Medical research abstract about a clinical trial...

### Minimal Scope
```
Entities (5): Person, Organization, Medication, Condition, Study
Relations (6): works_at, conducted_at, treats, enrolled_in, funded_by, reported_by
```

### Balanced Scope
```
Entities (9): Person, Organization, Medication, Condition, Study, Location, 
              Institution, Symptom, ResearchRole
Relations (12): works_at, conducted_at, treats, enrolled_in, funded_by, 
                reported_by, located_in, leads, causes, prescribes, 
                presents_with, affiliated_with
```

### Comprehensive Scope
```
Entities (15): Person, Organization, Medication, Condition, Study, Location,
               Institution, Symptom, Procedure, Equipment, Metric, 
               Publication, FundingSource, EthicsBoard, ResearchRole
Relations (20): works_at, conducted_at, treats, enrolled_in, funded_by,
                reported_by, located_in, leads, causes, prescribes,
                presents_with, affiliated_with, measures, uses, publishes,
                approved_by, monitors, collaborates_with, cites, administers
```

## Implementation Details

### BAML Prompt Structure

```baml
function RecommendOntology(
  text: string,
  scope: string  // "minimal", "balanced", "comprehensive"
) -> OntologyRecommendation {
  prompt #"
    ONTOLOGY SCOPE: {{ scope }}
    
    GRANULARITY GUIDELINES:
    [Comprehensive principles about entity and relation granularity]
    
    SCOPE-SPECIFIC GUIDANCE:
    **If scope is "minimal":**
    - Create only the most essential entity types
    - Typical result: 3-8 entity types, 4-10 relation types
    [Detailed guidance...]
    
    **If scope is "balanced":**
    [Detailed guidance...]
    
    **If scope is "comprehensive":**
    [Detailed guidance...]
    
    QUALITY OVER QUANTITY:
    [Principles emphasizing thoughtful design over hitting numbers]
  "#
}
```

### Python API

```python
class OntologyRecommender:
    def recommend(self, text: str, scope: str = "balanced") -> OntologyRecommendation:
        """
        scope: "minimal", "balanced", or "comprehensive"
        
        LLM determines appropriate number of types based on:
        - Text complexity and domain
        - Scope-specific guidelines
        - Ontology design principles
        """

class SpindleExtractor:
    def __init__(self, ontology: Optional[Ontology] = None, ontology_scope: str = "balanced"):
        """
        ontology_scope: Only used if ontology is None
        Default "balanced" works for most cases
        """
    
    def extract(self, ..., ontology_scope: Optional[str] = None):
        """
        Can override scope per extraction if needed
        """
```

## Migration Guide

### From Old API (max_entity_types, max_relation_types)

**Before:**
```python
# Had to guess numbers
recommendation = recommender.recommend(
    text=text,
    max_entity_types=10,  # Is this right? 🤷
    max_relation_types=15
)
```

**After:**
```python
# Think about analytical needs
recommendation = recommender.recommend(
    text=text,
    scope="balanced"  # Clear intent ✓
)
```

### Mapping (Approximate)

- Old: `max_entity_types=3-5` → New: `scope="minimal"`
- Old: `max_entity_types=8-12` → New: `scope="balanced"`
- Old: `max_entity_types=15-20` → New: `scope="comprehensive"`

But note: New approach is smarter and adapts to text, not just numbers!

## Advantages Summary

1. **More Intelligent**: LLM uses domain understanding, not arbitrary limits
2. **Clearer Semantics**: "balanced" vs "10" - intent is obvious
3. **Better Defaults**: "balanced" works for most cases naturally
4. **Self-Documenting**: Code reads like natural language
5. **Future-Proof**: Principles are timeless, numbers are arbitrary
6. **Adaptive**: Same scope produces appropriate results for simple and complex texts
7. **Aligned with Practice**: How ontologists actually think about design

## Philosophy

> "Instead of asking 'How many types should I have?', ask 'What level of detail do I need?'"

This shift from quantitative to qualitative thinking produces better ontologies that are:
- More appropriate for the domain
- Better aligned with analytical needs
- Easier to understand and maintain
- More consistent across similar texts

## Related Files

- `baml_src/spindle.baml` - Full prompt with all principles
- `spindle.py` - Python implementation
- `example_scope_comparison.py` - Side-by-side comparison
- `README.md` - User-facing documentation

