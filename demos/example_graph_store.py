"""
Example usage of GraphStore for persistent knowledge graph storage.

This script demonstrates:
1. Creating a GraphStore with environment configuration
2. Extracting triples with SpindleExtractor
3. Storing triples in Kùzu database
4. Querying by pattern (e.g., all "works_at" relationships)
5. Querying by source
6. Querying by date range
7. Updating node/edge metadata
8. Deleting specific nodes/edges
9. Graph statistics
10. Direct Cypher queries for advanced use cases
"""

import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from spindle import (
    SpindleExtractor,
    create_ontology,
    GraphStore
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
    print("Spindle GraphStore Example")
    print("=" * 70)
    print()
    print("NOTE: GraphStore automatically converts all node names and edge predicates")
    print("      to UPPERCASE for consistency. Queries are case-insensitive.")
    print()
    
    # Use a temporary graph name for this example
    graph_name = f"example_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Creating graph: {graph_name}")
    print(f"Graph files will be stored in: /graphs/{graph_name}/")
    print()
    
    # Step 1: Create GraphStore
    print("Step 1: Creating GraphStore...")
    print("-" * 70)
    
    with GraphStore(db_path=graph_name) as store:
        print("✓ GraphStore initialized")
        print()
        
        # Step 2: Define ontology and create extractor
        print("Step 2: Creating SpindleExtractor with ontology...")
        print("-" * 70)
        
        entity_types = [
            {
                "name": "Person", 
                "description": "A human being",
                "attributes": [
                    {"name": "title", "type": "string", "description": "Job title"},
                    {"name": "years_experience", "type": "int", "description": "Years at company"}
                ]
            },
            {
                "name": "Organization", 
                "description": "A company or institution",
                "attributes": [
                    {"name": "founded_year", "type": "int", "description": "Year founded"},
                    {"name": "industry", "type": "string", "description": "Industry sector"}
                ]
            },
            {
                "name": "Location", 
                "description": "A geographic place",
                "attributes": []
            },
            {
                "name": "Technology", 
                "description": "A programming language or tool",
                "attributes": []
            }
        ]
        
        relation_types = [
            {
                "name": "works_at",
                "description": "Employment relationship",
                "domain": "Person",
                "range": "Organization"
            },
            {
                "name": "located_in",
                "description": "Physical location",
                "domain": "Organization",
                "range": "Location"
            },
            {
                "name": "uses",
                "description": "Technology usage",
                "domain": "Person",
                "range": "Technology"
            }
        ]
        
        ontology = create_ontology(entity_types, relation_types)
        extractor = SpindleExtractor(ontology)
        print("✓ Extractor initialized")
        print()
        
        # Step 3: Extract triples from first text
        print("Step 3: Extracting triples from first source...")
        print("-" * 70)
        
        text1 = """
        Alice Johnson works at TechCorp in San Francisco. She primarily uses Python
        for backend development and has been with the company for 3 years.
        """
        
        result1 = extractor.extract(
            text=text1,
            source_name="Employee Database",
            source_url="https://example.com/employees"
        )
        
        print(f"Extracted {len(result1.triples)} triples from first source")
        for triple in result1.triples:
            print(f"  - {triple.subject.name} ({triple.subject.type}) --[{triple.predicate}]--> {triple.object.name} ({triple.object.type})")
            if triple.subject.custom_atts:
                print(f"    Subject attributes: {triple.subject.custom_atts}")
        print()
        
        # Step 4: Store triples in graph database
        print("Step 4: Storing triples in GraphStore...")
        print("-" * 70)
        
        count = store.add_triples(result1.triples)
        print(f"✓ Added {count} triples to graph database")
        print()
        
        # Step 5: Extract from second source
        print("Step 5: Extracting triples from second source...")
        print("-" * 70)
        
        text2 = """
        TechCorp is located in San Francisco and has recently expanded to New York.
        Bob Smith, who also works at TechCorp, specializes in TypeScript development.
        """
        
        result2 = extractor.extract(
            text=text2,
            source_name="Company Profile",
            source_url="https://example.com/company",
            existing_triples=result1.triples
        )
        
        print(f"Extracted {len(result2.triples)} triples from second source")
        for triple in result2.triples:
            print(f"  - {triple.subject.name} ({triple.subject.type}) --[{triple.predicate}]--> {triple.object.name} ({triple.object.type})")
        print()
        
        # Store second batch
        count = store.add_triples(result2.triples)
        print(f"✓ Added {count} more triples to graph database")
        print()
        
        # Step 6: Query by pattern
        print("Step 6: Querying by pattern...")
        print("-" * 70)
        
        # Find all "works_at" relationships
        works_at = store.query_by_pattern(predicate="works_at")
        print(f"All 'works_at' relationships ({len(works_at)}):")
        for edge in works_at:
            print(f"  - {edge['subject']} works at {edge['object']}")
            print(f"    Source: {edge['source']}")
        print()
        
        # Find all relationships involving Alice Johnson
        alice_rels = store.query_by_pattern(subject="Alice Johnson")
        print(f"All relationships with Alice Johnson as subject ({len(alice_rels)}):")
        for edge in alice_rels:
            print(f"  - {edge['subject']} --[{edge['predicate']}]--> {edge['object']}")
        print()
        
        # Find what TechCorp is related to
        techcorp_rels = store.query_by_pattern(subject="TechCorp")
        print(f"All relationships with TechCorp as subject ({len(techcorp_rels)}):")
        for edge in techcorp_rels:
            print(f"  - {edge['subject']} --[{edge['predicate']}]--> {edge['object']}")
        print()
        
        # Step 7: Query by source
        print("Step 7: Querying by source...")
        print("-" * 70)
        
        employee_db = store.query_by_source("Employee Database")
        print(f"Triples from 'Employee Database' ({len(employee_db)}):")
        for edge in employee_db:
            print(f"  - {edge['subject']} --[{edge['predicate']}]--> {edge['object']}")
        print()
        
        company_profile = store.query_by_source("Company Profile")
        print(f"Triples from 'Company Profile' ({len(company_profile)}):")
        for edge in company_profile:
            print(f"  - {edge['subject']} --[{edge['predicate']}]--> {edge['object']}")
        print()
        
        # Step 8: Query by date range
        print("Step 8: Querying by date range...")
        print("-" * 70)
        
        # All triples from the last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent = store.query_by_date_range(start=one_hour_ago)
        print(f"Triples extracted in the last hour ({len(recent)}):")
        for edge in recent:
            print(f"  - {edge['subject']} --[{edge['predicate']}]--> {edge['object']}")
            print(f"    Extracted: {edge['extraction_datetime']}")
        print()
        
        # Step 9: Get node information
        print("Step 9: Retrieving node information...")
        print("-" * 70)
        
        alice = store.get_node("Alice Johnson")
        if alice:
            print(f"Node: {alice['name']}")
            print(f"Type: {alice['type']}")
            print(f"Metadata: {json.dumps(alice['metadata'], indent=2)}")
        print()
        
        # Step 10: Update node metadata
        print("Step 10: Updating node metadata...")
        print("-" * 70)
        
        success = store.update_node(
            "Alice Johnson",
            updates={
                "metadata": {
                    "sources": ["Employee Database"],
                    "first_seen": result1.triples[0].extraction_datetime,
                    "verified": True,
                    "employee_id": "E12345"
                }
            }
        )
        
        if success:
            print("✓ Updated Alice Johnson's metadata")
            alice = store.get_node("Alice Johnson")
            print(f"Updated metadata: {json.dumps(alice['metadata'], indent=2)}")
        print()
        
        # Step 11: Get graph statistics
        print("Step 11: Graph statistics...")
        print("-" * 70)
        
        stats = store.get_statistics()
        print(f"Nodes: {stats['node_count']}")
        print(f"Edges: {stats['edge_count']}")
        print(f"Sources: {', '.join(stats['sources'])}")
        print(f"Predicates: {', '.join(stats['predicates'])}")
        if stats['date_range']:
            print(f"Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
        print()
        
        # Step 12: Direct Cypher query
        print("Step 12: Direct Cypher query (advanced)...")
        print("-" * 70)
        
        # Find all people and what organizations they work at
        cypher_query = """
        MATCH (p:Entity)-[r:Relationship {predicate: 'works_at'}]->(o:Entity)
        RETURN p.name AS person, o.name AS organization
        """
        
        results = store.query_cypher(cypher_query)
        print(f"People and their organizations (via Cypher):")
        for row in results:
            print(f"  - {row['person']} works at {row['organization']}")
        print()
        
        # Step 13: Export triples back to Triple objects
        print("Step 13: Exporting triples...")
        print("-" * 70)
        
        all_triples = store.get_triples()
        print(f"Exported {len(all_triples)} triples from database")
        print("These can be used with any Spindle functions that accept Triple objects")
        print()
        
        # Step 14: Delete specific edge
        print("Step 14: Deleting an edge...")
        print("-" * 70)
        
        # Find an edge to delete
        if works_at:
            edge_to_delete = works_at[0]
            success = store.delete_edge(
                edge_to_delete['subject'],
                edge_to_delete['predicate'],
                edge_to_delete['object']
            )
            if success:
                print(f"✓ Deleted edge: {edge_to_delete['subject']} --[{edge_to_delete['predicate']}]--> {edge_to_delete['object']}")
            
            # Verify deletion
            remaining = store.query_by_pattern(predicate="works_at")
            print(f"Remaining 'works_at' edges: {len(remaining)}")
        print()
        
        # Step 15: Add individual nodes and edges
        print("Step 15: Adding individual nodes and edges...")
        print("-" * 70)
        
        # Add a new person node
        store.add_node(
            name="Carol Davis",
            entity_type="Person",
            metadata={"title": "Data Scientist", "verified": True},
            description="A data scientist at TechCorp",
            custom_atts={}
        )
        print("✓ Added Carol Davis node")
        
        # Add a new edge
        store.add_edge(
            subject="Carol Davis",
            predicate="works_at",
            obj="TechCorp",
            metadata={
                "source": "Manual Entry",
                "extraction_datetime": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
        )
        print("✓ Added edge: Carol Davis works_at TechCorp")
        
        # Show updated statistics
        stats = store.get_statistics()
        print(f"Updated node count: {stats['node_count']}")
        print(f"Updated edge count: {stats['edge_count']}")
        print()
    
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print()
    print("Key features demonstrated:")
    print("  ✓ GraphStore initialization with custom path")
    print("  ✓ Storing extracted triples in Kùzu database")
    print("  ✓ Pattern-based querying with wildcards")
    print("  ✓ Source-based filtering")
    print("  ✓ Date range filtering")
    print("  ✓ Node and edge metadata management")
    print("  ✓ Graph statistics")
    print("  ✓ Direct Cypher queries")
    print("  ✓ Triple export/import")
    print("  ✓ CRUD operations on nodes and edges")
    print("=" * 70)
    print()
    print(f"Note: Graph '{graph_name}' was created in /graphs/{graph_name}/")
    print("You can use GraphStore.delete_graph() to remove it when done.")


if __name__ == "__main__":
    main()

