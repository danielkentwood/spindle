"""
Example usage of the Spindle knowledge graph extraction tool.

This script demonstrates:
1. Defining a custom ontology
2. Extracting triples from text
3. Incremental extraction with entity consistency
"""

import os
import json
from dotenv import load_dotenv
from spindle import (
    SpindleExtractor,
    create_ontology,
    triples_to_dict,
    get_supporting_text,
    filter_triples_by_source,
    parse_extraction_datetime
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
    print("Spindle Knowledge Graph Extraction Example")
    print("=" * 70)
    print()
    
    # Step 1: Define the ontology
    print("Step 1: Defining ontology...")
    print("-" * 70)
    
    entity_types = [
        {
            "name": "Person",
            "description": "A human being, identified by their name"
        },
        {
            "name": "Organization",
            "description": "A company, institution, or organized group"
        },
        {
            "name": "Location",
            "description": "A geographic place, city, or address"
        },
        {
            "name": "Technology",
            "description": "A programming language, framework, or technical tool"
        }
    ]
    
    relation_types = [
        {
            "name": "works_at",
            "description": "Employment or work relationship",
            "domain": "Person",
            "range": "Organization"
        },
        {
            "name": "located_in",
            "description": "Physical location relationship",
            "domain": "Organization",
            "range": "Location"
        },
        {
            "name": "uses",
            "description": "Technology usage relationship",
            "domain": "Person",
            "range": "Technology"
        },
        {
            "name": "founded",
            "description": "Founding relationship",
            "domain": "Person",
            "range": "Organization"
        }
    ]
    
    ontology = create_ontology(entity_types, relation_types)
    
    print(f"Created ontology with {len(entity_types)} entity types "
          f"and {len(relation_types)} relation types")
    print()
    
    # Step 2: Create the extractor
    print("Step 2: Creating SpindleExtractor...")
    print("-" * 70)
    extractor = SpindleExtractor(ontology)
    print("Extractor initialized")
    print()
    
    # Step 3: First extraction
    print("Step 3: Extracting triples from first text...")
    print("-" * 70)
    
    text1 = """
    Alice Johnson is a senior software engineer who works at TechCorp, a rapidly 
    growing technology company that specializes in cloud-based solutions. TechCorp 
    is headquartered in San Francisco, where it occupies a modern office building 
    in the Financial District. The company was founded in 2015 by Bob Smith, a 
    serial entrepreneur with a background in distributed systems and enterprise 
    software.
    
    Alice has been with TechCorp since its early days and has become one of the 
    company's most valued engineers. She is particularly skilled in Python 
    development, having worked with the language for over a decade. In her current 
    role, she leads the backend infrastructure team and is responsible for 
    architecting scalable microservices. However, Alice is also well-versed in 
    frontend technologies and frequently uses React to build user interfaces for 
    internal tools and customer-facing applications.
    
    TechCorp's San Francisco office has grown significantly over the years, now 
    housing over 200 employees across engineering, product, sales, and operations 
    teams. The company's location in San Francisco has been strategic, allowing 
    it to tap into the Bay Area's rich talent pool and maintain close relationships 
    with venture capital firms and other technology companies in Silicon Valley.
    
    Bob Smith continues to serve as TechCorp's CEO and remains deeply involved in 
    the company's technical direction. Before founding TechCorp, Bob spent several 
    years working at major technology firms, where he identified gaps in the market 
    that TechCorp now fills with its innovative cloud platform solutions.
    """
    
    print(f"Input text:\n{text1}")
    print()
    
    result1 = extractor.extract(
        text=text1,
        source_name="TechCorp Company Profile 2024",
        source_url="https://example.com/techcorp/profile"
    )
    
    print(f"Extracted {len(result1.triples)} triples:")
    for i, triple in enumerate(result1.triples, 1):
        print(f"  {i}. ({triple.subject}) --[{triple.predicate}]--> ({triple.object})")
        print(f"     Source: {triple.source.source_name}")
        if triple.source.source_url:
            print(f"     URL: {triple.source.source_url}")
        print(f"     Extracted: {triple.extraction_datetime}")
        print(f"     Supporting evidence ({len(triple.supporting_spans)} span(s)):")
        for j, span in enumerate(triple.supporting_spans, 1):
            snippet = span.text[:80] + "..." if len(span.text) > 80 else span.text
            if span.start is not None and span.start >= 0:
                # Verify the indices are correct by extracting from source
                import re
                extracted = text1[span.start:span.end] if span.end else ""
                # Normalize whitespace for comparison (LLM may clean up formatting)
                norm_extracted = re.sub(r'\s+', ' ', extracted.strip())
                norm_span = re.sub(r'\s+', ' ', span.text.strip())
                match_indicator = "✓" if norm_extracted == norm_span else "✗"
                print(f"       {j}. [{span.start}:{span.end}] {match_indicator} \"{snippet}\"")
            else:
                print(f"       {j}. [NOT FOUND] \"{snippet}\"")
    print()
    print(f"Reasoning: {result1.reasoning}")
    print()
    
    # Step 4: Second extraction with existing triples
    print("Step 4: Extracting from second text with entity consistency...")
    print("-" * 70)
    
    text2 = """
    In recent months, Alice Johnson has been expanding her technical expertise 
    beyond her core competencies. She recently started incorporating TypeScript 
    into her development workflow, recognizing the value of static typing for 
    large-scale applications. Alice has been particularly impressed with how 
    TypeScript enhances code maintainability and reduces runtime errors in 
    TechCorp's growing codebase. She's now advocating for TypeScript adoption 
    across the entire engineering organization and has been conducting internal 
    workshops to help other developers make the transition.
    
    Meanwhile, TechCorp has been experiencing remarkable growth and recently 
    announced plans to expand its operations beyond the West Coast. The company 
    revealed that it will be opening a new office in New York City, specifically 
    in Manhattan's Flatiron District. This new location will serve as TechCorp's 
    East Coast headquarters and is expected to house sales, business development, 
    and a growing engineering team. The New York office is scheduled to open in 
    the next quarter and will initially accommodate 50 employees, with plans to 
    scale to over 150 within the first year.
    
    Bob Smith, who founded TechCorp nearly a decade ago, continues to lead the 
    company as its Chief Executive Officer. Under his leadership, TechCorp has 
    grown from a small startup to a company with over 250 employees and a 
    valuation exceeding $500 million. Bob remains hands-on in the company's 
    day-to-day operations and is known for his open-door policy and regular 
    engagement with employees at all levels. He's been particularly focused on 
    maintaining TechCorp's startup culture even as the company scales, emphasizing 
    innovation, collaboration, and technical excellence.
    
    The expansion to New York represents a significant milestone for TechCorp 
    and reflects the company's ambitions to compete more effectively on the East 
    Coast market. Bob has stated that the New York office will enable TechCorp 
    to be closer to major financial services clients and to tap into New York's 
    diverse talent pool. Alice has been asked to help set up the engineering 
    practices for the New York team and will be traveling between San Francisco 
    and New York regularly during the initial setup phase.
    """
    
    print(f"Input text:\n{text2}")
    print()
    
    # Pass previous triples to maintain entity consistency
    # Note: Duplicate triples are now allowed since this is a different source
    result2 = extractor.extract(
        text=text2,
        source_name="TechCorp Expansion Announcement",
        source_url="https://example.com/techcorp/news/expansion-2024",
        existing_triples=result1.triples
    )
    
    print(f"Extracted {len(result2.triples)} new triples:")
    for i, triple in enumerate(result2.triples, 1):
        print(f"  {i}. ({triple.subject}) --[{triple.predicate}]--> ({triple.object})")
        print(f"     Source: {triple.source.source_name}")
        if triple.source.source_url:
            print(f"     URL: {triple.source.source_url}")
        print(f"     Extracted: {triple.extraction_datetime}")
        print(f"     Supporting evidence ({len(triple.supporting_spans)} span(s)):")
        for j, span in enumerate(triple.supporting_spans, 1):
            snippet = span.text[:80] + "..." if len(span.text) > 80 else span.text
            if span.start is not None and span.start >= 0:
                # Verify the indices are correct by extracting from source
                import re
                extracted = text2[span.start:span.end] if span.end else ""
                # Normalize whitespace for comparison (LLM may clean up formatting)
                norm_extracted = re.sub(r'\s+', ' ', extracted.strip())
                norm_span = re.sub(r'\s+', ' ', span.text.strip())
                match_indicator = "✓" if norm_extracted == norm_span else "✗"
                print(f"       {j}. [{span.start}:{span.end}] {match_indicator} \"{snippet}\"")
            else:
                print(f"       {j}. [NOT FOUND] \"{snippet}\"")
    print()
    print(f"Reasoning: {result2.reasoning}")
    print()
    
    # Step 5: Combine all triples and analyze
    print("Step 5: Complete knowledge graph...")
    print("-" * 70)
    
    all_triples = result1.triples + result2.triples
    print(f"Total triples in knowledge graph: {len(all_triples)}")
    print()
    
    # Identify duplicate facts from different sources
    print("Checking for duplicate facts from different sources...")
    fact_to_sources = {}
    for triple in all_triples:
        fact = (triple.subject, triple.predicate, triple.object)
        if fact not in fact_to_sources:
            fact_to_sources[fact] = []
        fact_to_sources[fact].append(triple.source.source_name)
    
    duplicates = {fact: sources for fact, sources in fact_to_sources.items() if len(sources) > 1}
    if duplicates:
        print(f"Found {len(duplicates)} facts confirmed by multiple sources:")
        for i, (fact, sources) in enumerate(duplicates.items(), 1):
            print(f"  {i}. {fact[0]} --[{fact[1]}]--> {fact[2]}")
            print(f"     Confirmed by: {', '.join(sources)}")
    else:
        print("No duplicate facts found (all facts are unique to their sources)")
    print()
    
    # Step 6: Filter triples by source
    print("Step 6: Filtering triples by source...")
    print("-" * 70)
    
    source1_triples = filter_triples_by_source(all_triples, "TechCorp Company Profile 2024")
    source2_triples = filter_triples_by_source(all_triples, "TechCorp Expansion Announcement")
    
    print(f"Triples from 'TechCorp Company Profile 2024': {len(source1_triples)}")
    print(f"Triples from 'TechCorp Expansion Announcement': {len(source2_triples)}")
    print()
    
    # Step 7: Demonstrate serialization
    print("Step 7: Serialization example...")
    print("-" * 70)
    
    triples_dict = triples_to_dict(all_triples)
    print(f"Triples as dictionaries (showing first triple in detail):")
    if triples_dict:
        print(json.dumps(triples_dict[0], indent=2))
    print()
    
    # Step 8: Demonstrate datetime features
    print("Step 8: Extraction datetime analysis...")
    print("-" * 70)
    
    print("Extraction times:")
    for i, triple in enumerate(all_triples[:3], 1):  # Show first 3
        dt = parse_extraction_datetime(triple)
        if dt:
            print(f"  {i}. {triple.subject} --[{triple.predicate}]--> {triple.object}")
            print(f"     Raw: {triple.extraction_datetime}")
            print(f"     Parsed: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        else:
            print(f"  {i}. {triple.subject} --[{triple.predicate}]--> {triple.object}")
            print(f"     Could not parse: {triple.extraction_datetime}")
    print()
    
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print()
    print("Key features demonstrated:")
    print("  ✓ Source metadata tracking for each triple")
    print("  ✓ Character spans showing supporting evidence")
    print("  ✓ Extraction datetime for temporal tracking")
    print("  ✓ Entity consistency across multiple sources")
    print("  ✓ Duplicate triples from different sources allowed")
    print("  ✓ Source-based filtering")
    print("  ✓ Datetime parsing and analysis")
    print("=" * 70)


if __name__ == "__main__":
    main()

