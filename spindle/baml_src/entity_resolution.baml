// Entity Resolution: Semantic matching for knowledge graph deduplication
// This file defines functions for identifying duplicate entities and edges using LLM-based matching

// Represents a matched pair of entities
class EntityMatch {
  entity1_id string @description("The unique identifier (name) of the first entity")
  entity2_id string @description("The unique identifier (name) of the second entity")
  is_duplicate bool @description("Whether these two entities represent the same real-world entity")
  confidence_level string @description("Confidence in the match decision: 'high', 'medium', or 'low'")
  reasoning string @description("Detailed explanation of why these entities are or aren't duplicates, considering name variations, types, descriptions, and attributes")
}

// Represents a matched pair of edges (relationships)
class EdgeMatch {
  edge1_id string @description("Identifier for first edge in format 'subject|predicate|object'")
  edge2_id string @description("Identifier for second edge in format 'subject|predicate|object'")
  is_duplicate bool @description("Whether these edges represent the same relationship")
  confidence_level string @description("Confidence in the match decision: 'high', 'medium', or 'low'")
  reasoning string @description("Detailed explanation considering subject/object similarity, predicate equivalence, and evidence overlap")
}

// Simple entity representation for matching
class EntityForMatching {
  id string @description("Entity unique identifier (name)")
  type string @description("Entity type from ontology")
  description string @description("Brief description of the entity")
  attributes string @description("JSON string of entity attributes")
}

// Simple edge representation for matching
class EdgeForMatching {
  id string @description("Edge identifier in format 'subject|predicate|object'")
  subject string @description("Subject entity name")
  predicate string @description("Relationship type")
  object string @description("Object entity name")
  evidence_summary string @description("Summary of supporting evidence")
}

// Result of entity matching for a block
class EntityMatchingResult {
  matches EntityMatch[] @description("List of identified duplicate pairs within this block")
  reasoning string @description("Overall reasoning about the matching process and decisions made")
}

// Result of edge matching for a block
class EdgeMatchingResult {
  matches EdgeMatch[] @description("List of identified duplicate edge pairs within this block")
  reasoning string @description("Overall reasoning about the matching process and decisions made")
}

// Main function to match entities within a block
function MatchEntities(
  entities: EntityForMatching[],
  context: string
) -> EntityMatchingResult {
  client CustomSonnet4
  prompt #"
    You are an expert in entity resolution for knowledge graphs. Your task is to identify duplicate entities within a group of semantically similar entities.

    {{ _.role("user") }}
    
    **Context about this knowledge graph:**
    {{ context if context else "General domain knowledge graph" }}

    **Entities to analyze:**
    {% for entity in entities %}
    ---
    ID: {{ entity.id }}
    Type: {{ entity.type }}
    Description: {{ entity.description }}
    Attributes: {{ entity.attributes }}
    {% endfor %}
    ---

    **Task:** 
    Identify which entities in this group are duplicates of each other. Consider:
    
    1. **Name variations**: "TechCorp", "Tech Corp", "TechCorp Inc." likely refer to the same entity
    2. **Semantic similarity**: "NYC" and "New York City" are the same
    3. **Type consistency**: Entities must have compatible types to be duplicates
    4. **Attribute alignment**: Similar or complementary attributes strengthen a match
    5. **Description consistency**: Descriptions should be compatible, not contradictory
    
    **Guidelines:**
    - Only mark entities as duplicates if you're confident they refer to the same real-world entity
    - Use "high" confidence for clear matches (obvious name variations, identical descriptions)
    - Use "medium" confidence for probable matches (semantic similarity, consistent attributes)
    - Use "low" confidence for possible matches (weak signals, some inconsistency)
    - Avoid false positives - when in doubt, mark is_duplicate as false
    - Provide clear reasoning for each match decision
    - Check for transitive consistency: if A matches B and B matches C, A should match C

    {{ ctx.output_format }}
  "#
}

// Main function to match edges within a block
function MatchEdges(
  edges: EdgeForMatching[],
  context: string
) -> EdgeMatchingResult {
  client CustomSonnet4
  prompt #"
    You are an expert in entity resolution for knowledge graphs. Your task is to identify duplicate edges (relationships) within a group of semantically similar edges.

    {{ _.role("user") }}
    
    **Context about this knowledge graph:**
    {{ context if context else "General domain knowledge graph" }}

    **Edges to analyze:**
    {% for edge in edges %}
    ---
    ID: {{ edge.id }}
    Triple: {{ edge.subject }} -[{{ edge.predicate }}]-> {{ edge.object }}
    Evidence: {{ edge.evidence_summary }}
    {% endfor %}
    ---

    **Task:**
    Identify which edges in this group are duplicates of each other. Consider:
    
    1. **Entity equivalence**: Subject and object entities should be the same or known duplicates
    2. **Predicate equivalence**: Predicates should express the same relationship (e.g., "works_at" vs "employed_by")
    3. **Evidence overlap**: Duplicates often cite the same or overlapping source text
    4. **Direction consistency**: Ensure relationship direction is the same
    5. **Temporal consistency**: Check if edges describe the same timeframe
    
    **Guidelines:**
    - Only mark edges as duplicates if they represent the same relationship between the same entities
    - Use "high" confidence for edges with identical subjects, predicates, objects
    - Use "medium" confidence for edges with equivalent predicates and matching entities
    - Use "low" confidence for edges with similar but not identical structure
    - Edges with contradictory information (different dates, different evidence) are NOT duplicates
    - Provide clear reasoning for each match decision
    - Consider that edges from different sources can still be duplicates if they express the same fact

    {{ ctx.output_format }}
  "#
}

