// Spindle: Knowledge Graph Triple Extraction
// This file defines the data structures and extraction function for building knowledge graphs from text

// Source metadata for tracking where a triple came from
class SourceMetadata {
  source_name string @description("The name or identifier of the source document")
  source_url string? @description("Optional URL of the source document")
}

// Character span indicating text evidence for a triple
class CharacterSpan {
  text string @description("The exact text from the source that supports this triple")
  start int? @description("Starting character index in the source text (0-based, computed in post-processing)")
  end int? @description("Ending character index in the source text (exclusive, computed in post-processing)")
}

// A triple represents a subject-predicate-object relationship with supporting evidence
class Triple {
  subject string @description("The subject entity of the triple")
  predicate string @description("The relationship/predicate connecting subject and object")
  object string @description("The object entity of the triple")
  source SourceMetadata @description("Metadata about the source of this triple")
  supporting_spans CharacterSpan[] @description("Character spans in the source text that support this triple")
  extraction_datetime string? @description("ISO 8601 datetime when this triple was extracted (set automatically in post-processing)")
}

// Defines a type of entity that can appear in the knowledge graph
class EntityType {
  name string @description("The name of the entity type (e.g., 'Person', 'Organization')")
  description string @description("A description of what this entity type represents")
}

// Defines a type of relationship between entities
class RelationType {
  name string @description("The name of the relation type (e.g., 'works_at', 'located_in')")
  description string @description("A description of what this relation represents")
  domain string @description("The entity type that can be the subject of this relation")
  range string @description("The entity type that can be the object of this relation")
}

// An ontology defines the structure of the knowledge graph
class Ontology {
  entity_types EntityType[] @description("List of valid entity types")
  relation_types RelationType[] @description("List of valid relation types")
}

// The result of triple extraction
class ExtractionResult {
  triples Triple[] @description("List of extracted triples from the text")
  reasoning string @description("Explanation of the extraction decisions and entity consistency choices")
}

// Main function to extract knowledge graph triples from text
function ExtractTriples(
  text: string,
  ontology: Ontology,
  source_metadata: SourceMetadata,
  existing_triples: Triple[]
) -> ExtractionResult {
  client CustomSonnet4
  prompt #"
    You are a knowledge graph extraction expert. Your task is to extract structured triples (subject-predicate-object) from the provided text, along with supporting evidence.

    {{ _.role("user") }}
    
    ONTOLOGY:
    You must extract triples that conform to the following ontology:
    
    Valid Entity Types:
    {% for entity_type in ontology.entity_types %}
    - {{ entity_type.name }}: {{ entity_type.description }}
    {% endfor %}
    
    Valid Relation Types:
    {% for relation_type in ontology.relation_types %}
    - {{ relation_type.name }}: {{ relation_type.description }}
      (Domain: {{ relation_type.domain }}, Range: {{ relation_type.range }})
    {% endfor %}
    
    SOURCE METADATA:
    Source Name: {{ source_metadata.source_name }}
    {% if source_metadata.source_url %}
    Source URL: {{ source_metadata.source_url }}
    {% endif %}
    
    EXISTING TRIPLES:
    {% if existing_triples|length > 0 %}
    The following triples have already been extracted from OTHER sources. You MUST:
    1. Use consistent entity names - if an entity appears in existing triples, use the EXACT same name
    2. Maintain entity identity - recognize when entities in the new text are the same as entities in existing triples
    3. Duplicate triples ARE ALLOWED if they come from different sources (different source names)
    4. If you extract the same fact that exists in existing triples, it's okay as long as it's from this new source
    
    {% for triple in existing_triples %}
    - {{ triple.subject }} -> {{ triple.predicate }} -> {{ triple.object }} (from: {{ triple.source.source_name }})
    {% endfor %}
    {% else %}
    This is the first extraction, so there are no existing triples to consider.
    {% endif %}
    
    TEXT TO ANALYZE:
    {{ text }}
    
    INSTRUCTIONS:
    1. Extract all meaningful triples from the text that conform to the ontology
    2. Only use entity types and relation types defined in the ontology
    3. Use clear, consistent entity names (e.g., "John Smith" not "John" or "Smith")
    4. If an entity appears in existing triples, use the exact same name for consistency
    5. For each triple, identify the TEXT SPANS that provide evidence for the triple
       - Copy the EXACT text from the source that supports the triple
       - You can include multiple text spans if the evidence is spread across different parts of the text
       - Be precise - copy the text exactly as it appears, including punctuation and spacing
       - Include enough context to make the evidence clear
    6. Set the source metadata for each triple to the provided source information
    7. Provide reasoning explaining your extraction decisions, entity consistency choices, and how you identified supporting spans
    
    EXAMPLE of text spans:
    If the text is "Alice works at TechCorp in San Francisco." and you extract the triple (Alice, works_at, TechCorp),
    the supporting span text would be: "Alice works at TechCorp"
    
    Note: You only need to provide the text content. Character indices will be computed automatically in post-processing.
    
    {{ ctx.output_format }}
  "#
}

