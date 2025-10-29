"""
Example demonstrating conservative ontology extension for new sources.

This script shows how to:
1. Start with an existing ontology
2. Analyze new text to see if extension is needed
3. Apply extensions only when critical information would be lost
4. Continue extracting with the evolved ontology
"""

import os
import json
from dotenv import load_dotenv
from spindle import (
    OntologyRecommender,
    SpindleExtractor,
    create_ontology,
    extension_to_dict,
    ontology_to_dict
)

# Load environment variables (API keys)
load_dotenv()

# Check if API key is set
if not os.getenv("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY environment variable not set.")
    print("Please set it in a .env file or as an environment variable.")
    exit(1)


def main():
    print("=" * 80)
    print("Conservative Ontology Extension Example")
    print("=" * 80)
    print()
    
    # Step 1: Start with a business-focused ontology
    print("Step 1: Create Initial Business Ontology")
    print("-" * 80)
    
    entity_types = [
        {"name": "Person", "description": "An individual person"},
        {"name": "Organization", "description": "A company, institution, or business entity"},
        {"name": "Location", "description": "A geographic place or address"},
        {"name": "Product", "description": "A product or service offered by an organization"}
    ]
    
    relation_types = [
        {
            "name": "works_at",
            "description": "Employment relationship between a person and organization",
            "domain": "Person",
            "range": "Organization"
        },
        {
            "name": "located_in",
            "description": "Physical location of an organization or person",
            "domain": "Organization",
            "range": "Location"
        },
        {
            "name": "develops",
            "description": "Relationship where an organization creates a product",
            "domain": "Organization",
            "range": "Product"
        },
        {
            "name": "founded",
            "description": "Founding relationship",
            "domain": "Person",
            "range": "Organization"
        }
    ]
    
    initial_ontology = create_ontology(entity_types, relation_types)
    
    print(f"Initial Ontology:")
    print(f"  Entity Types: {[et.name for et in initial_ontology.entity_types]}")
    print(f"  Relation Types: {[rt.name for rt in initial_ontology.relation_types]}")
    print()
    
    recommender = OntologyRecommender()
    
    # Step 2: Test with similar domain text (should NOT need extension)
    print("Step 2: Analyze Text from Similar Domain")
    print("-" * 80)
    
    similar_text = """
    TechVentures Inc., a venture capital firm based in San Jose, recently
    invested in CloudScale, a cloud infrastructure startup. The investment
    was led by Jennifer Martinez, managing partner at TechVentures. CloudScale
    develops cloud optimization software and was founded by Tom Wilson in 2020.
    """
    
    print(f"Text:\n{similar_text}\n")
    
    extension1 = recommender.analyze_extension(
        text=similar_text,
        current_ontology=initial_ontology,
        scope="balanced"
    )
    
    print(f"Extension Needed: {extension1.needs_extension}")
    if extension1.needs_extension:
        print(f"New Types:")
        print(f"  Entities: {[et.name for et in extension1.new_entity_types]}")
        print(f"  Relations: {[rt.name for rt in extension1.new_relation_types]}")
        print(f"\nCritical Information at Risk:")
        print(f"  {extension1.critical_information_at_risk}")
    print(f"\nReasoning:")
    print(f"  {extension1.reasoning}")
    print()
    
    # Step 3: Test with text from different domain (MIGHT need extension)
    print("Step 3: Analyze Text from Medical Domain")
    print("-" * 80)
    
    medical_text = """
    Dr. Sarah Chen, a cardiologist at Stanford Medical Center, recently published
    research on the efficacy of Medication Beta in treating hypertension. The
    clinical trial involved 500 patients and showed significant improvement in
    blood pressure control. The study was funded by the National Institutes of
    Health and published in the Journal of Cardiology.
    """
    
    print(f"Text:\n{medical_text}\n")
    
    extension2 = recommender.analyze_extension(
        text=medical_text,
        current_ontology=initial_ontology,
        scope="balanced"
    )
    
    print(f"Extension Needed: {extension2.needs_extension}")
    if extension2.needs_extension:
        print(f"\nNew Entity Types:")
        for et in extension2.new_entity_types:
            print(f"  - {et.name}: {et.description}")
        print(f"\nNew Relation Types:")
        for rt in extension2.new_relation_types:
            print(f"  - {rt.name}: {rt.description}")
            print(f"    ({rt.domain} → {rt.range})")
        print(f"\nCritical Information at Risk:")
        print(f"  {extension2.critical_information_at_risk}")
    print(f"\nReasoning:")
    print(f"  {extension2.reasoning}")
    print()
    
    # Step 4: Apply extension if needed
    if extension2.needs_extension:
        print("Step 4: Apply Extension to Ontology")
        print("-" * 80)
        
        extended_ontology = recommender.extend_ontology(initial_ontology, extension2)
        
        print(f"Original Ontology:")
        print(f"  {len(initial_ontology.entity_types)} entity types, "
              f"{len(initial_ontology.relation_types)} relation types")
        print(f"\nExtended Ontology:")
        print(f"  {len(extended_ontology.entity_types)} entity types, "
              f"{len(extended_ontology.relation_types)} relation types")
        print(f"\nAll Entity Types: {[et.name for et in extended_ontology.entity_types]}")
        print(f"All Relation Types: {[rt.name for rt in extended_ontology.relation_types]}")
        print()
        
        # Now extract with the extended ontology
        print("Step 5: Extract Triples with Extended Ontology")
        print("-" * 80)
        
        extractor = SpindleExtractor(extended_ontology)
        result = extractor.extract(medical_text, source_name="Medical Research")
        
        print(f"Extracted {len(result.triples)} triples:\n")
        for i, triple in enumerate(result.triples, 1):
            print(f"  {i}. ({triple.subject}) --[{triple.predicate}]--> ({triple.object})")
        print()
    
    # Step 6: Demonstrate the analyze_and_extend convenience method
    print("Step 6: Using Convenience Method (analyze_and_extend)")
    print("-" * 80)
    
    tech_policy_text = """
    The Federal Trade Commission, led by Commissioner Lisa Park, announced new
    regulations regarding data privacy for social media platforms. The regulation
    requires companies to obtain explicit consent before collecting user data.
    Meta and Google have six months to comply with the new requirements.
    """
    
    print(f"Text:\n{tech_policy_text}\n")
    
    # Use convenience method with auto_apply
    extension3, maybe_new_ontology = recommender.analyze_and_extend(
        text=tech_policy_text,
        current_ontology=initial_ontology,
        scope="balanced",
        auto_apply=True
    )
    
    print(f"Extension Needed: {extension3.needs_extension}")
    
    if maybe_new_ontology:
        print(f"\nAutomatically Extended Ontology:")
        print(f"  {len(maybe_new_ontology.entity_types)} entity types, "
              f"{len(maybe_new_ontology.relation_types)} relation types")
        print(f"\nNew Types Added:")
        print(f"  Entities: {[et.name for et in extension3.new_entity_types]}")
        print(f"  Relations: {[rt.name for rt in extension3.new_relation_types]}")
        
        # Extract with new ontology
        extractor = SpindleExtractor(maybe_new_ontology)
        result = extractor.extract(tech_policy_text, source_name="Policy News")
        print(f"\nExtracted {len(result.triples)} triples with extended ontology")
    else:
        print(f"\nNo extension needed - using original ontology")
        print(f"Reasoning: {extension3.reasoning}")
    
    print()
    
    # Step 7: Serialize extension analysis
    print("Step 7: Serialize Extension Analysis")
    print("-" * 80)
    
    if extension2.needs_extension:
        extension_dict = extension_to_dict(extension2)
        print("Extension Analysis as JSON:")
        print(json.dumps(extension_dict, indent=2))
    
    print()
    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print()
    print("Key Principles Demonstrated:")
    print("  ✓ Conservative extension - only when critical information at risk")
    print("  ✓ Analysis explains WHY extension is/isn't needed")
    print("  ✓ Extensions are backward-compatible additions")
    print("  ✓ Original ontology unchanged - new ontology is created")
    print("  ✓ Existing types preferred over creating new ones")
    print("  ✓ Ontology evolves deliberately, not with every new text")
    print("=" * 80)


if __name__ == "__main__":
    main()

