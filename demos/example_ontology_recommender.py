"""
Example usage of the OntologyRecommender service.

This script demonstrates:
1. Analyzing text to automatically recommend an ontology
2. Using the recommended ontology to extract triples
3. Combining recommendation and extraction in one step
"""

import os
import json
from dotenv import load_dotenv
from spindle import (
    OntologyRecommender,
    SpindleExtractor,
    recommendation_to_dict,
    triples_to_dict
)

# Load environment variables (API keys)
load_dotenv()

# Check if API key is set
if not os.getenv("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY environment variable not set.")
    print("Please set it in a .env file or as an environment variable.")
    exit(1)


def main():
    print("=" * 70)
    print("Ontology Recommender Example")
    print("=" * 70)
    print()
    
    # Example text: Medical research abstract
    medical_text = """
    A recent clinical trial evaluated the efficacy of Medication A in treating 
    patients with chronic migraines. The study, conducted at Massachusetts General 
    Hospital, enrolled 250 patients aged 18-65 who experienced at least 8 migraine 
    days per month. Dr. Sarah Chen, the principal investigator, led a team of 
    neurologists who administered the drug over a 12-week period.
    
    Results showed that Medication A reduced migraine frequency by an average of 
    50% compared to the placebo group. Common side effects included nausea and 
    dizziness, which affected approximately 15% of participants. The medication 
    works by inhibiting CGRP receptors, which are known to play a role in migraine 
    pathophysiology.
    
    Dr. Chen reported the findings at the American Academy of Neurology conference 
    in Seattle, where the research was well-received by the medical community. The 
    FDA is expected to review the data for potential approval in the coming year. 
    Massachusetts General Hospital has been a leading research institution in 
    neurology for decades and continues to conduct groundbreaking studies in 
    headache disorders.
    """
    
    print("Step 1: Recommend ontology from medical text")
    print("-" * 70)
    print(f"Input text:\n{medical_text}\n")
    
    # Create the recommender
    recommender = OntologyRecommender()
    
    # Get ontology recommendation
    recommendation = recommender.recommend(
        text=medical_text,
        scope="balanced"  # "minimal", "balanced", or "comprehensive"
    )
    
    print("Text Purpose:")
    print(f"  {recommendation.text_purpose}\n")
    
    print("Recommended Entity Types:")
    for i, entity_type in enumerate(recommendation.ontology.entity_types, 1):
        print(f"  {i}. {entity_type.name}: {entity_type.description}")
    print()
    
    print("Recommended Relation Types:")
    for i, relation_type in enumerate(recommendation.ontology.relation_types, 1):
        print(f"  {i}. {relation_type.name}: {relation_type.description}")
        print(f"     ({relation_type.domain} → {relation_type.range})")
    print()
    
    print("Reasoning:")
    print(f"  {recommendation.reasoning}\n")
    
    # Step 2: Use the recommended ontology to extract triples
    print("Step 2: Extract triples using recommended ontology")
    print("-" * 70)
    
    extractor = SpindleExtractor(recommendation.ontology)
    extraction_result = extractor.extract(
        text=medical_text,
        source_name="Medical Research Abstract 2024",
        source_url="https://example.com/research/abstract-001"
    )
    
    print(f"Extracted {len(extraction_result.triples)} triples:\n")
    for i, triple in enumerate(extraction_result.triples, 1):
        print(f"  {i}. ({triple.subject}) --[{triple.predicate}]--> ({triple.object})")
        print(f"     Evidence: {len(triple.supporting_spans)} span(s)")
    print()
    
    # Step 3: Demonstrate the combined approach
    print("Step 3: Recommend and extract in one step")
    print("-" * 70)
    
    business_text = """
    TechVentures Inc., a venture capital firm based in Palo Alto, recently 
    announced a $50 million Series B investment in CloudScale, a rapidly growing 
    cloud infrastructure startup. The investment was led by Jennifer Martinez, 
    managing partner at TechVentures, who will join CloudScale's board of directors.
    
    CloudScale, founded in 2020 by former Google engineers Tom Wilson and Lisa 
    Park, has developed proprietary technology for optimizing cloud resource 
    allocation. The company, headquartered in Austin, Texas, has already secured 
    major clients including Fortune 500 companies in the financial services and 
    healthcare sectors.
    
    Jennifer Martinez stated that CloudScale's innovative approach to solving 
    cloud efficiency challenges aligns perfectly with TechVentures' investment 
    thesis. The funding will be used to expand CloudScale's engineering team and 
    accelerate product development. Tom Wilson and Lisa Park plan to use the 
    capital to open new offices in San Francisco and New York, targeting a headcount 
    of 200 employees by the end of next year.
    """
    
    print(f"Input text:\n{business_text}\n")
    
    # Recommend and extract in one call
    recommendation2, extraction2 = recommender.recommend_and_extract(
        text=business_text,
        source_name="TechVentures Investment Announcement",
        source_url="https://example.com/news/techventures-investment",
        scope="balanced"  # Could also use "minimal" or "comprehensive"
    )
    
    print("Auto-detected Purpose:")
    print(f"  {recommendation2.text_purpose}\n")
    
    print("Auto-generated Ontology:")
    print(f"  Entity Types: {', '.join([et.name for et in recommendation2.ontology.entity_types])}")
    print(f"  Relation Types: {', '.join([rt.name for rt in recommendation2.ontology.relation_types])}")
    print()
    
    print(f"Extracted {len(extraction2.triples)} triples:\n")
    for i, triple in enumerate(extraction2.triples, 1):
        print(f"  {i}. ({triple.subject}) --[{triple.predicate}]--> ({triple.object})")
    print()
    
    # Step 4: Demonstrate serialization
    print("Step 4: Serialize recommendation to JSON")
    print("-" * 70)
    
    recommendation_dict = recommendation_to_dict(recommendation2)
    print(json.dumps(recommendation_dict, indent=2))
    print()
    
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print()
    print("Key features demonstrated:")
    print("  ✓ Automatic ontology recommendation from text analysis")
    print("  ✓ Text purpose/goal inference")
    print("  ✓ Domain-appropriate entity and relation type generation")
    print("  ✓ Direct use of recommended ontology for extraction")
    print("  ✓ One-step recommendation + extraction workflow")
    print("  ✓ JSON serialization of recommendations")
    print("=" * 70)


if __name__ == "__main__":
    main()

