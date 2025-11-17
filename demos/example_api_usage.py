"""Example demonstrating Spindle REST API usage.

This script shows how to use the Spindle API for knowledge graph extraction.
Run the API server first: spindle-api
"""

import json
import time

import requests


class SpindleAPIClient:
    """Simple client for the Spindle REST API."""

    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None

    def health_check(self):
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def create_session(self, name=None):
        """Create a new session."""
        response = requests.post(
            f"{self.base_url}/api/sessions",
            json={"name": name}
        )
        response.raise_for_status()
        self.session_id = response.json()["session_id"]
        print(f"✓ Created session: {self.session_id}")
        return self.session_id

    def get_session_info(self):
        """Get current session info."""
        if not self.session_id:
            raise ValueError("No active session")
        response = requests.get(f"{self.base_url}/api/sessions/{self.session_id}")
        response.raise_for_status()
        return response.json()

    def recommend_ontology(self, text, scope="balanced"):
        """Recommend an ontology based on text."""
        response = requests.post(
            f"{self.base_url}/api/ontology/recommend",
            json={"text": text, "scope": scope}
        )
        response.raise_for_status()
        return response.json()

    def update_session_ontology(self, ontology):
        """Update session ontology."""
        if not self.session_id:
            raise ValueError("No active session")
        response = requests.put(
            f"{self.base_url}/api/sessions/{self.session_id}/ontology",
            json={"ontology": ontology}
        )
        response.raise_for_status()
        print("✓ Updated session ontology")
        return response.json()

    def extract_triples(self, text, source_name, use_session=False):
        """Extract triples from text."""
        if use_session:
            if not self.session_id:
                raise ValueError("No active session")
            url = f"{self.base_url}/api/extraction/session/{self.session_id}/extract"
        else:
            url = f"{self.base_url}/api/extraction/extract"

        response = requests.post(
            url,
            json={
                "text": text,
                "source_name": source_name
            }
        )
        response.raise_for_status()
        return response.json()

    def extract_batch(self, texts, ontology=None):
        """Extract triples from multiple texts."""
        payload = {"texts": texts}
        if ontology:
            payload["ontology"] = ontology

        response = requests.post(
            f"{self.base_url}/api/extraction/extract/batch",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def extract_stream(self, texts, ontology=None):
        """Extract triples with streaming (yields results as they complete)."""
        payload = {"texts": texts}
        if ontology:
            payload["ontology"] = ontology

        response = requests.post(
            f"{self.base_url}/api/extraction/extract/stream",
            json=payload,
            stream=True
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line.startswith(b'data: '):
                yield json.loads(line[6:])

    def delete_session(self):
        """Delete current session."""
        if not self.session_id:
            return
        response = requests.delete(f"{self.base_url}/api/sessions/{self.session_id}")
        response.raise_for_status()
        print(f"✓ Deleted session: {self.session_id}")
        self.session_id = None


def example_stateless_extraction():
    """Example 1: Stateless triple extraction with auto-recommendation."""
    print("\n" + "="*60)
    print("Example 1: Stateless Extraction (Auto-recommend Ontology)")
    print("="*60)

    client = SpindleAPIClient()

    # Check health
    health = client.health_check()
    print(f"✓ API health: {health['status']}")

    # Extract triples (ontology will be auto-recommended)
    text = "Alice works at TechCorp as a senior engineer. She reports to Bob, the engineering manager."
    result = client.extract_triples(text, "example-doc")

    print(f"\n✓ Extracted {result['triple_count']} triples:")
    for i, triple in enumerate(result['triples'][:3], 1):
        subj = triple['subject']['name']
        pred = triple['predicate']
        obj = triple['object']['name']
        print(f"  {i}. ({subj}) --[{pred}]--> ({obj})")

    if result['triple_count'] > 3:
        print(f"  ... and {result['triple_count'] - 3} more triples")


def example_session_based_extraction():
    """Example 2: Session-based extraction with ontology management."""
    print("\n" + "="*60)
    print("Example 2: Session-based Extraction")
    print("="*60)

    client = SpindleAPIClient()

    # Create session
    client.create_session("example-project")

    # Recommend and set ontology
    sample_text = "Alice works at TechCorp. Bob manages the engineering team."
    ontology_result = client.recommend_ontology(sample_text, scope="minimal")
    print(f"✓ Recommended ontology with {ontology_result['entity_type_count']} entity types")
    print(f"  Purpose: {ontology_result['text_purpose']}")

    client.update_session_ontology(ontology_result['ontology'])

    # Extract triples using session context
    texts = [
        "Alice works at TechCorp as an engineer.",
        "Bob is the manager of the engineering team.",
        "Charlie joined TechCorp last month."
    ]

    for i, text in enumerate(texts, 1):
        result = client.extract_triples(text, f"doc{i}", use_session=True)
        print(f"✓ Document {i}: Extracted {result['triple_count']} triples")

    # Check session state
    session_info = client.get_session_info()
    print(f"\n✓ Session accumulated {session_info['triple_count']} total triples")

    # Clean up
    client.delete_session()


def example_batch_extraction():
    """Example 3: Batch extraction."""
    print("\n" + "="*60)
    print("Example 3: Batch Extraction")
    print("="*60)

    client = SpindleAPIClient()

    # Prepare texts
    texts = [
        {
            "text": "Alice founded StartupCo in 2020.",
            "source_name": "bio-alice.txt"
        },
        {
            "text": "Bob invested $1M in StartupCo.",
            "source_name": "funding-round.txt"
        },
        {
            "text": "StartupCo launched their product in San Francisco.",
            "source_name": "press-release.txt"
        }
    ]

    # Batch extract
    result = client.extract_batch(texts)

    print(f"✓ Processed {len(result['results'])} documents")
    print(f"✓ Extracted {result['total_triples']} total triples\n")

    for doc_result in result['results']:
        print(f"  {doc_result['source_name']}: {doc_result['triple_count']} triples")


def example_streaming_extraction():
    """Example 4: Streaming extraction."""
    print("\n" + "="*60)
    print("Example 4: Streaming Extraction (Real-time Results)")
    print("="*60)

    client = SpindleAPIClient()

    # Prepare texts
    texts = [
        {
            "text": "The company hired 50 new employees this quarter.",
            "source_name": "q1-report.txt"
        },
        {
            "text": "Revenue increased by 30% year over year.",
            "source_name": "q1-report.txt"
        },
        {
            "text": "The product team launched three new features.",
            "source_name": "product-update.txt"
        }
    ]

    print("Streaming results as they complete...\n")

    # Stream extractions
    for i, result in enumerate(client.extract_stream(texts), 1):
        print(f"✓ Result {i}: {result['triple_count']} triples from {result['source_name']}")
        time.sleep(0.1)  # Simulate processing time


def example_ontology_extension():
    """Example 5: Ontology extension analysis."""
    print("\n" + "="*60)
    print("Example 5: Ontology Extension")
    print("="*60)

    client = SpindleAPIClient()

    # Start with a basic ontology
    initial_text = "Alice works at TechCorp."
    ontology_result = client.recommend_ontology(initial_text, scope="minimal")
    ontology = ontology_result['ontology']

    print(f"✓ Initial ontology: {len(ontology['entity_types'])} entity types")

    # New text from different domain
    new_text = "The hospital treated 100 patients. Dr. Smith performed 5 surgeries."

    # Analyze extension needs
    response = requests.post(
        f"{client.base_url}/api/ontology/extend/analyze",
        json={
            "text": new_text,
            "current_ontology": ontology,
            "scope": "balanced"
        }
    )
    extension = response.json()

    if extension['needs_extension']:
        print(f"✓ Extension recommended: {len(extension['new_entity_types'])} new entity types")
        print(f"  Reason: {extension['critical_information_at_risk']}")

        # Apply extension
        response = requests.post(
            f"{client.base_url}/api/ontology/extend/apply",
            json={
                "current_ontology": ontology,
                "extension": extension
            }
        )
        extended = response.json()
        print(f"✓ Extended ontology: {extended['entity_type_count']} entity types")
    else:
        print("✓ No extension needed")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Spindle REST API Examples")
    print("="*60)
    print("\nMake sure the API server is running:")
    print("  $ spindle-api\n")

    try:
        # Run examples
        example_stateless_extraction()
        example_session_based_extraction()
        example_batch_extraction()
        example_streaming_extraction()
        example_ontology_extension()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server")
        print("Please start the server with: spindle-api")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

