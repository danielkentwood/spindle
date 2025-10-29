"""
Example demonstrating the three ontology scope levels: minimal, balanced, and comprehensive.

This script shows how the same text can be analyzed with different granularity
levels, and compares the resulting ontologies.
"""

import os
from dotenv import load_dotenv
from spindle import OntologyRecommender, SpindleExtractor

# Load environment variables (API keys)
load_dotenv()

# Check if API key is set
if not os.getenv("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY environment variable not set.")
    print("Please set it in a .env file or as an environment variable.")
    exit(1)


def main():
    print("=" * 70)
    print("Ontology Scope Comparison: Minimal vs Balanced vs Comprehensive")
    print("=" * 70)
    print()
    
    # Complex text with multiple domains and concepts
    text = """
    The Metropolitan Museum of Art in New York recently acquired a rare 
    15th-century painting by Johannes Vermeer at a Sotheby's auction. The 
    artwork, valued at $45 million, was purchased using funds from the 
    Patterson Foundation's endowment. Dr. Elizabeth Sterling, the museum's 
    chief curator, led the acquisition process and authenticated the piece 
    with Professor Michael Chen from Yale University's art history department.
    
    The painting depicts a domestic scene common in Dutch Golden Age art, 
    showing a young woman reading a letter by a window. The composition 
    demonstrates Vermeer's masterful use of light and his characteristic 
    attention to detail in rendering fabrics and interior spaces. The work 
    will be displayed in the museum's European Paintings gallery alongside 
    other Dutch Masters including Rembrandt and Frans Hals.
    
    The acquisition was funded through a competitive bidding process at 
    Sotheby's headquarters in Manhattan. The museum competed against the 
    Rijksmuseum in Amsterdam and a private collector from London. The 
    Patterson Foundation, established in 1962 by industrialist Robert 
    Patterson, has been a major supporter of the Metropolitan Museum's 
    European art collection for decades.
    """
    
    print("Analyzing the same text with three different scope levels...")
    print("-" * 70)
    print(f"Text:\n{text}\n")
    
    recommender = OntologyRecommender()
    
    # Test all three scopes
    scopes = ["minimal", "balanced", "comprehensive"]
    recommendations = {}
    
    for scope in scopes:
        print(f"\n{'=' * 70}")
        print(f"SCOPE: {scope.upper()}")
        print('=' * 70)
        
        recommendation = recommender.recommend(text=text, scope=scope)
        recommendations[scope] = recommendation
        
        print(f"\nText Purpose:\n  {recommendation.text_purpose}\n")
        
        print(f"Entity Types ({len(recommendation.ontology.entity_types)}):")
        for i, et in enumerate(recommendation.ontology.entity_types, 1):
            print(f"  {i}. {et.name}")
            print(f"     └─ {et.description}")
        
        print(f"\nRelation Types ({len(recommendation.ontology.relation_types)}):")
        for i, rt in enumerate(recommendation.ontology.relation_types, 1):
            print(f"  {i}. {rt.name}: {rt.description}")
            print(f"     └─ ({rt.domain} → {rt.range})")
        
        print(f"\nReasoning:\n  {recommendation.reasoning}")
    
    # Comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print()
    
    print("Ontology Sizes:")
    for scope in scopes:
        rec = recommendations[scope]
        entity_count = len(rec.ontology.entity_types)
        relation_count = len(rec.ontology.relation_types)
        print(f"  {scope.capitalize():14} : {entity_count:2} entity types, {relation_count:2} relation types")
    
    print("\n" + "-" * 70)
    print("Use Case Guidance:")
    print("-" * 70)
    print()
    print("MINIMAL:")
    print("  • Best for: Quick exploration, simple dashboards, broad patterns")
    print("  • Pros: Fast extraction, easy to understand, less maintenance")
    print("  • Cons: May miss nuances, less detailed queries")
    print()
    print("BALANCED (Default):")
    print("  • Best for: Standard analysis, general-purpose extraction")
    print("  • Pros: Good detail/simplicity balance, versatile")
    print("  • Cons: May need adjustment for very simple or very complex domains")
    print()
    print("COMPREHENSIVE:")
    print("  • Best for: Research, domain expertise, detailed analysis")
    print("  • Pros: Captures nuances, supports complex queries")
    print("  • Cons: More types to maintain, longer extraction time")
    print()
    
    # Now demonstrate extraction with different scopes
    print("=" * 70)
    print("EXTRACTION COMPARISON")
    print("=" * 70)
    print()
    
    short_text = """
    The Louvre Museum announced that curator Marie Dubois will lead a new 
    exhibition featuring Impressionist paintings from the Musée d'Orsay.
    """
    
    print(f"Extracting from shorter text:\n{short_text}\n")
    
    for scope in ["minimal", "comprehensive"]:
        print(f"\n{'-' * 70}")
        print(f"Using {scope.upper()} scope:")
        print('-' * 70)
        
        extractor = SpindleExtractor(ontology_scope=scope)
        result = extractor.extract(short_text, source_name=f"News ({scope})")
        
        print(f"Ontology: {len(extractor.ontology.entity_types)} entities, "
              f"{len(extractor.ontology.relation_types)} relations")
        print(f"Extracted: {len(result.triples)} triples\n")
        
        for i, triple in enumerate(result.triples, 1):
            print(f"  {i}. ({triple.subject}) --[{triple.predicate}]--> ({triple.object})")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print()
    print("Key Insights:")
    print("  • Same text → Different ontologies based on scope")
    print("  • Minimal: Fewest types, broader categories")
    print("  • Balanced: Practical middle ground (recommended default)")
    print("  • Comprehensive: Most types, finer distinctions")
    print("  • Choose scope based on your analytical needs, not text length")
    print("=" * 70)


if __name__ == "__main__":
    main()

