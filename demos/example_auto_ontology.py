"""
Example usage of SpindleExtractor with automatic ontology recommendation.

This script demonstrates the new feature where SpindleExtractor can be
initialized without an ontology and will automatically recommend one
based on the text when extract() is called.
"""

import os
from dotenv import load_dotenv
from spindle import SpindleExtractor, triples_to_dict

# Load environment variables (API keys)
load_dotenv()

# Check if API key is set
if not os.getenv("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY environment variable not set.")
    print("Please set it in a .env file or as an environment variable.")
    exit(1)


def main():
    print("=" * 70)
    print("SpindleExtractor with Automatic Ontology Recommendation")
    print("=" * 70)
    print()
    
    # Example text about a different domain
    text = """
    The Metropolitan Museum of Art acquired a rare 15th-century painting by 
    Johannes Vermeer at Sotheby's auction in New York. The artwork, titled 
    "Girl with a Pearl Earring Redux," was purchased for $45 million by the 
    museum's curator Dr. Elizabeth Sterling. The painting will be displayed 
    in the Dutch Masters gallery alongside other works from the Golden Age 
    period. Art historian Professor Michael Chen from Yale University 
    authenticated the piece and confirmed its provenance. The acquisition was 
    funded by the Patterson Foundation's endowment for European art.
    """
    
    print("Step 1: Create SpindleExtractor WITHOUT an ontology")
    print("-" * 70)
    print("When extract() is called, it will automatically recommend an ontology")
    print("based on the text content.")
    print()
    
    # Create extractor WITHOUT providing an ontology
    extractor = SpindleExtractor()
    print("✓ SpindleExtractor created without ontology")
    print()
    
    print("Step 2: Extract triples (ontology will be auto-recommended)")
    print("-" * 70)
    print(f"Input text:\n{text}\n")
    
    # Call extract - this will automatically recommend an ontology first
    result = extractor.extract(
        text=text,
        source_name="Art Museum News",
        source_url="https://example.com/museum/news/acquisition",
        ontology_scope="balanced"  # Optional: "minimal", "balanced", or "comprehensive"
    )
    
    print(f"✓ Ontology automatically recommended and applied!")
    print()
    
    print("Step 3: Inspect the auto-recommended ontology")
    print("-" * 70)
    
    print(f"Entity Types ({len(extractor.ontology.entity_types)}):")
    for i, et in enumerate(extractor.ontology.entity_types, 1):
        print(f"  {i}. {et.name}: {et.description}")
    print()
    
    print(f"Relation Types ({len(extractor.ontology.relation_types)}):")
    for i, rt in enumerate(extractor.ontology.relation_types, 1):
        print(f"  {i}. {rt.name}: {rt.description}")
        print(f"     ({rt.domain} → {rt.range})")
    print()
    
    print("Step 4: View extracted triples")
    print("-" * 70)
    
    print(f"Extracted {len(result.triples)} triples:\n")
    for i, triple in enumerate(result.triples, 1):
        print(f"  {i}. ({triple.subject}) --[{triple.predicate}]--> ({triple.object})")
        print(f"     Source: {triple.source.source_name}")
        print(f"     Evidence snippets: {len(triple.supporting_spans)}")
        if triple.supporting_spans:
            snippet = triple.supporting_spans[0].text[:60] + "..."
            print(f"     \"{snippet}\"")
    print()
    
    print(f"Extraction Reasoning:\n  {result.reasoning}\n")
    
    print("Step 5: Subsequent extractions use the same ontology")
    print("-" * 70)
    
    text2 = """
    The Louvre Museum in Paris announced that it will loan three Monet 
    paintings to the National Gallery in London for a special exhibition. 
    The curator Marie Dubois coordinated the arrangement with her British 
    counterpart. These impressionist works will be featured alongside pieces 
    from the Tate Modern's collection.
    """
    
    print(f"Second text:\n{text2}\n")
    
    # Subsequent calls will use the already-recommended ontology
    result2 = extractor.extract(
        text=text2,
        source_name="International Art News",
        existing_triples=result.triples  # Maintain entity consistency
    )
    
    print(f"✓ Extracted {len(result2.triples)} more triples using same ontology")
    print()
    
    for i, triple in enumerate(result2.triples, 1):
        print(f"  {i}. ({triple.subject}) --[{triple.predicate}]--> ({triple.object})")
    print()
    
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print()
    print("Key features demonstrated:")
    print("  ✓ SpindleExtractor initialized without ontology")
    print("  ✓ Ontology automatically recommended from text on first extract()")
    print("  ✓ Auto-recommended ontology used for all subsequent extractions")
    print("  ✓ Principled ontology design with scope levels (minimal/balanced/comprehensive)")
    print("  ✓ Maintains entity consistency across multiple texts")
    print("=" * 70)


if __name__ == "__main__":
    main()

