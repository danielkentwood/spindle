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

// Attribute definition for entity types in the ontology
class AttributeDefinition {
  name string @description("The name of the attribute (e.g., 'campaign_launch_dt', 'founded_date')")
  type string @description("Data type: 'string', 'int', 'float', 'bool', 'date', or a nested object schema definition")
  description string @description("A description of what this attribute represents")
}

// Attribute value in extracted entities (includes type metadata)
class AttributeValue {
  value string? @description("The extracted value as a string, or null if not found in the text")
  type string @description("The data type of this attribute (matches the type from AttributeDefinition)")
}

// Structured entity in a triple
class Entity {
  name string @description("The entity's name or identifier extracted from the text")
  type string @description("The entity type from the ontology (must match an EntityType.name exactly)")
  description string @description("A brief description of the entity generated from the text context")
  custom_atts map<string, AttributeValue> @description("Type-specific attributes as defined in the entity type's AttributeDefinition list")
}

// A triple represents a subject-predicate-object relationship with supporting evidence
class Triple {
  subject Entity @description("The subject entity with full metadata and attributes")
  predicate string @description("The relationship/predicate connecting subject and object")
  object Entity @description("The object entity with full metadata and attributes")
  source SourceMetadata @description("Metadata about the source of this triple")
  supporting_spans CharacterSpan[] @description("Character spans in the source text that support this triple")
  extraction_datetime string? @description("ISO 8601 datetime when this triple was extracted (set automatically in post-processing)")
}

// Defines a type of entity that can appear in the knowledge graph
class EntityType {
  name string @description("The name of the entity type (e.g., 'Person', 'Organization', 'Campaign')")
  description string @description("A description of what this entity type represents")
  attributes AttributeDefinition[] @description("List of custom attributes that should be extracted for entities of this type")
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
  client CustomFast
  prompt #"
    You are a knowledge graph extraction expert. Your task is to extract structured triples (subject-predicate-object) from the provided text, with rich entity metadata, custom attributes, and supporting evidence.

    {{ _.role("user") }}
    
    ONTOLOGY:
    You must extract triples that conform to the following ontology:
    
    Valid Entity Types:
    {% for entity_type in ontology.entity_types %}
    - {{ entity_type.name }}: {{ entity_type.description }}
      {% if entity_type.attributes|length > 0 %}
      Custom Attributes:
      {% for attr in entity_type.attributes %}
        * {{ attr.name }} ({{ attr.type }}): {{ attr.description }}
      {% endfor %}
      {% endif %}
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
    - {{ triple.subject.name }} ({{ triple.subject.type }}) -> {{ triple.predicate }} -> {{ triple.object.name }} ({{ triple.object.type }}) [from: {{ triple.source.source_name }}]
    {% endfor %}
    {% else %}
    This is the first extraction, so there are no existing triples to consider.
    {% endif %}
    
    TEXT TO ANALYZE:
    {{ text }}
    
    ENTITY EXTRACTION INSTRUCTIONS:
    
    For each entity in a triple (both subject and object), you MUST provide:
    
    1. **name**: The entity's name or identifier from the text
       - Use clear, consistent names (e.g., "John Smith" not "John" or "Smith")
       - If the entity appears in existing triples, use the EXACT same name
    
    2. **type**: The entity type from the ontology
       - Must match exactly one of the EntityType names defined above
       - Check domain/range constraints in relation types
    
    3. **description**: A brief description of the entity
       - Generate this from the text context (1-2 sentences)
       - Include relevant details that help identify or characterize the entity
       - Be concise but informative
    
    4. **custom_atts**: Extract ALL attributes defined for this entity type
       - Format: {"attribute_name": {"value": "extracted_value", "type": "attribute_type"}}
       - Set "value" to null if the attribute is not mentioned in the text
       - "type" must match the type from the AttributeDefinition
       - Extract dates in ISO format (YYYY-MM-DD) when possible
       - For nested objects, use JSON string representation
    
    TRIPLE EXTRACTION INSTRUCTIONS:
    
    1. Extract all meaningful triples from the text that conform to the ontology
    2. Only use entity types and relation types defined in the ontology
    3. For each triple, identify TEXT SPANS that provide evidence:
       - Copy the EXACT text from the source that supports the triple
       - You can include multiple text spans if evidence is spread across different parts
       - Be precise - copy text exactly as it appears, including punctuation and spacing
       - Include enough context to make the evidence clear
    4. Provide reasoning explaining your extraction decisions
    
    EXAMPLE:
    
    Text: "The Check Your A1C campaign launched on August 10, 2025, targeting diabetic members to remind them to get their A1C checked."
    
    Ontology has EntityType "Campaign" with attributes:
    - campaign_launch_dt (date): Launch date
    - campaign_completion_dt (date): Completion date
    
    Extracted entity for subject:
    {
      "name": "Check Your A1C campaign",
      "type": "Campaign",
      "description": "A campaign targeting diabetic members to remind them to get their A1C checked",
      "custom_atts": {
        "campaign_launch_dt": {"value": "2025-08-10", "type": "date"},
        "campaign_completion_dt": {"value": null, "type": "date"}
      }
    }
    
    Note: Character indices for supporting spans will be computed automatically in post-processing.
    
    {{ ctx.output_format }}
  "#
}


