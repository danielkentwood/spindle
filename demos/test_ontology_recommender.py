"""
Quick test script for the OntologyRecommender service.
This is a minimal test to verify the service works correctly.
"""

import os
from dotenv import load_dotenv
from spindle import OntologyRecommender, recommendation_to_dict
import json

# Load environment variables
load_dotenv()

# Check if API key is set
if not os.getenv("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY environment variable not set.")
    print("Please set it in a .env file or as an environment variable.")
    exit(1)


def test_basic_recommendation():
    """Test basic ontology recommendation."""
    print("Testing OntologyRecommender...")
    print("-" * 50)
    
    # Simple test text
    test_text = """
    John Smith is a software engineer at Google. He works on machine learning
    projects and specializes in natural language processing. Google is headquartered
    in Mountain View, California.
    """
    
    # Create recommender
    recommender = OntologyRecommender()
    
    # Get recommendation
    print("Analyzing text and recommending ontology...")
    recommendation = recommender.recommend(
        text=test_text,
        scope="minimal"  # Using minimal scope for this simple test
    )
    
    # Display results
    print("\n✓ Recommendation received!")
    print(f"\nText Purpose:\n  {recommendation.text_purpose}")
    
    print(f"\nEntity Types ({len(recommendation.ontology.entity_types)}):")
    for et in recommendation.ontology.entity_types:
        print(f"  - {et.name}: {et.description}")
    
    print(f"\nRelation Types ({len(recommendation.ontology.relation_types)}):")
    for rt in recommendation.ontology.relation_types:
        print(f"  - {rt.name}: {rt.description}")
        print(f"    ({rt.domain} → {rt.range})")
    
    print(f"\nReasoning:\n  {recommendation.reasoning}")
    
    # Test serialization
    print("\n✓ Testing serialization...")
    recommendation_dict = recommendation_to_dict(recommendation)
    print(f"  Serialized to dict with {len(recommendation_dict)} keys")
    
    print("\n" + "=" * 50)
    print("✓ All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    test_basic_recommendation()

