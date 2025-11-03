# Spindle Quick Start Guide

Get up and running with Spindle in 5 minutes!

## Step 1: Environment Setup (30 seconds)

```bash
# Activate the kgx conda environment
conda activate kgx

# Verify dependencies are installed
python -c "import baml_py; print('BAML installed:', baml_py.__version__)"
```

## Step 2: Set Up API Key (1 minute)

Create a `.env` file in the spindle directory:

```bash
cd /Users/thalamus/Repos/spindle
echo "ANTHROPIC_API_KEY=your_actual_api_key_here" > .env
```

💡 **Get your API key**: https://console.anthropic.com/

## Step 3: Run the Example (30 seconds)

```bash
python example.py
```

This will:
- Define a sample ontology
- Extract triples from two texts
- Show entity consistency in action
- Display the complete knowledge graph

## Step 4: Try Your Own Example (2 minutes)

Create a file `my_example.py`:

```python
from spindle import SpindleExtractor, create_ontology

# 1. Define your domain (with optional custom attributes)
ontology = create_ontology(
    entity_types=[
        {
            "name": "Person", 
            "description": "A human being",
            "attributes": []  # Optional: add custom attributes to extract
        },
        {
            "name": "Company", 
            "description": "A business organization",
            "attributes": [
                {
                    "name": "founded_year",
                    "type": "int",
                    "description": "The year the company was founded"
                }
            ]
        }
    ],
    relation_types=[
        {
            "name": "founded",
            "description": "Created or established",
            "domain": "Person",
            "range": "Company"
        }
    ]
)

# 2. Create extractor
extractor = SpindleExtractor(ontology)

# 3. Extract triples with source tracking
result = extractor.extract(
    text="Elon Musk founded SpaceX in 2002.",
    source_name="SpaceX Wikipedia",
    source_url="https://en.wikipedia.org/wiki/SpaceX"
)

# 4. View results with evidence and entity metadata
print(f"Found {len(result.triples)} triples:")
for triple in result.triples:
    print(f"  • {triple.subject.name} ({triple.subject.type}) → {triple.predicate} → {triple.object.name} ({triple.object.type})")
    print(f"    Subject: {triple.subject.description}")
    print(f"    Object: {triple.object.description}")
    if triple.object.custom_atts:
        print(f"    Attributes: {triple.object.custom_atts}")
    print(f"    Source: {triple.source.source_name}")
    if triple.supporting_spans:
        print(f"    Evidence: \"{triple.supporting_spans[0].text}\"")

print(f"\nReasoning: {result.reasoning}")
```

Run it:
```bash
python my_example.py
```

## Step 5: Build Incrementally with Multi-Source Support (1 minute)

```python
# First extraction from Wikipedia
result1 = extractor.extract(
    text="Elon Musk founded SpaceX.",
    source_name="Wikipedia",
    source_url="https://en.wikipedia.org/wiki/Elon_Musk"
)

# Second extraction from news article
# Note: Duplicate facts from different sources are allowed!
result2 = extractor.extract(
    text="Elon Musk also founded Tesla and SpaceX.",
    source_name="Tech News",
    source_url="https://example.com/news/elon-musk",
    existing_triples=result1.triples  # Maintains entity consistency!
)

# Combine results - duplicates from different sources preserved
all_triples = result1.triples + result2.triples
print(f"Total knowledge graph: {len(all_triples)} triples")

# Check which facts are confirmed by multiple sources
from spindle import filter_triples_by_source
wiki_triples = filter_triples_by_source(all_triples, "Wikipedia")
news_triples = filter_triples_by_source(all_triples, "Tech News")
print(f"From Wikipedia: {len(wiki_triples)}, From News: {len(news_triples)}")
```

## Common Use Cases

### Research Paper Analysis
```python
ontology = create_ontology(
    entity_types=[
        {
            "name": "Researcher", 
            "description": "A scientist or academic",
            "attributes": [
                {"name": "affiliation", "type": "string", "description": "University or institution"},
                {"name": "email", "type": "string", "description": "Contact email"}
            ]
        },
        {
            "name": "Paper", 
            "description": "A research publication",
            "attributes": [
                {"name": "publication_year", "type": "int", "description": "Year published"},
                {"name": "journal", "type": "string", "description": "Journal name"},
                {"name": "doi", "type": "string", "description": "Digital Object Identifier"}
            ]
        },
        {
            "name": "Method", 
            "description": "A research method or technique",
            "attributes": []
        }
    ],
    relation_types=[
        {
            "name": "authored",
            "description": "Wrote or co-authored",
            "domain": "Researcher",
            "range": "Paper"
        },
        {
            "name": "uses",
            "description": "Employs or utilizes",
            "domain": "Paper",
            "range": "Method"
        }
    ]
)
```

### Business Intelligence
```python
ontology = create_ontology(
    entity_types=[
        {
            "name": "Company", 
            "description": "Business organization",
            "attributes": [
                {"name": "founded_year", "type": "int", "description": "Year company was founded"},
                {"name": "headquarters", "type": "string", "description": "Location of HQ"},
                {"name": "revenue", "type": "float", "description": "Annual revenue in millions"}
            ]
        },
        {
            "name": "Product", 
            "description": "Product or service",
            "attributes": [
                {"name": "launch_date", "type": "date", "description": "Date product was launched"},
                {"name": "price", "type": "float", "description": "Product price"}
            ]
        },
        {
            "name": "Market", 
            "description": "Market or industry sector",
            "attributes": []
        }
    ],
    relation_types=[
        {
            "name": "produces",
            "description": "Manufactures or offers",
            "domain": "Company",
            "range": "Product"
        },
        {
            "name": "competes_in",
            "description": "Operates in market",
            "domain": "Company",
            "range": "Market"
        }
    ]
)
```

### Medical Records
```python
ontology = create_ontology(
    entity_types=[
        {
            "name": "Patient", 
            "description": "Medical patient",
            "attributes": [
                {"name": "age", "type": "int", "description": "Patient age in years"},
                {"name": "gender", "type": "string", "description": "Patient gender"}
            ]
        },
        {
            "name": "Condition", 
            "description": "Medical condition or diagnosis",
            "attributes": [
                {"name": "diagnosis_date", "type": "date", "description": "Date of diagnosis"},
                {"name": "severity", "type": "string", "description": "Condition severity (mild/moderate/severe)"}
            ]
        },
        {
            "name": "Treatment", 
            "description": "Medical treatment or medication",
            "attributes": [
                {"name": "dosage", "type": "string", "description": "Medication dosage"},
                {"name": "frequency", "type": "string", "description": "How often to take"}
            ]
        }
    ],
    relation_types=[
        {
            "name": "diagnosed_with",
            "description": "Has been diagnosed with",
            "domain": "Patient",
            "range": "Condition"
        },
        {
            "name": "prescribed",
            "description": "Given as treatment",
            "domain": "Patient",
            "range": "Treatment"
        }
    ]
)
```

## Tips for Best Results

1. **Be Specific in Ontology**: Clear descriptions help the LLM extract accurately
2. **Use Consistent Entity Names**: Define how entities should be named (e.g., "full name" vs "first name")
3. **Define Custom Attributes**: Add type-specific attributes to extract structured data from entities (e.g., dates, amounts, locations)
4. **Pass Previous Triples**: Always pass `existing_triples` for entity consistency
5. **Check Reasoning**: The `reasoning` field explains extraction decisions
6. **Iterate on Ontology**: Refine your ontology based on extraction results
7. **Start Simple**: Begin with empty attributes lists and add them as you identify patterns in your data

## Troubleshooting

### Error: "ANTHROPIC_API_KEY not set"
- Make sure you created `.env` file with your API key
- Verify the key starts with `sk-ant-`

### Error: "ModuleNotFoundError: No module named 'baml_client'"
- Run `baml-cli generate` to generate the client
- Make sure you're in the spindle directory

### Error: "BamlValidationError"
- Check that your ontology is properly defined
- Verify entity_types and relation_types are lists of dicts

### Slow Extractions
- Normal for first call (cold start)
- Subsequent calls should be faster
- Claude Sonnet 4 may take 5-10 seconds per extraction

## Next Steps

- Read `README.md` for full documentation
- Check `IMPLEMENTATION_SUMMARY.md` for technical details
- Explore `spindle/baml_src/spindle.baml` to understand the BAML schema
- Customize `spindle.py` for your specific needs

## Getting Help

- BAML Documentation: https://docs.boundaryml.com/
- Anthropic API: https://docs.anthropic.com/
- Project Issues: [Add your issue tracker]

Happy knowledge graph building! 🚀

