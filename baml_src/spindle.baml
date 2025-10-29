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

// Result of ontology recommendation
class OntologyRecommendation {
  ontology Ontology @description("The recommended ontology with entity and relation types")
  text_purpose string @description("Analysis of the text's overarching purpose or goal")
  reasoning string @description("Explanation of why these entity and relation types were recommended")
}

// Result of ontology extension analysis
class OntologyExtension {
  needs_extension bool @description("Whether the ontology needs to be extended for this text")
  new_entity_types EntityType[] @description("New entity types to add (empty if none needed)")
  new_relation_types RelationType[] @description("New relation types to add (empty if none needed)")
  critical_information_at_risk string @description("Description of what critical information would be lost without extension")
  reasoning string @description("Detailed explanation of why extension is or isn't needed")
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

// Function to recommend an ontology based on text analysis
function RecommendOntology(
  text: string,
  scope: string
) -> OntologyRecommendation {
  client CustomSonnet4
  prompt #"
    You are a knowledge graph ontology design expert. Your task is to analyze the provided text, understand its overarching purpose and domain, and recommend an appropriate ontology (entity types and relation types) that would be suitable for extracting knowledge from this and similar texts.

    {{ _.role("user") }}
    
    TEXT TO ANALYZE:
    {{ text }}
    
    ONTOLOGY SCOPE: {{ scope }}
    
    GRANULARITY GUIDELINES:
    
    Your goal is to design an ontology with {{ scope }} scope. Use these principles to guide your decisions about how many types to create and how granular to make them.
    
    ENTITY TYPE GRANULARITY PRINCIPLES:
    
    1. **Abstraction Level**: Choose the right level of abstraction
       - Too broad: "Thing", "Entity" (not useful)
       - Too narrow: "SoftwareEngineerAtGoogle", "CEOOfTechStartup" (overfitting)
       - Just right: "Person", "Organization", "JobRole" (reusable across instances)
    
    2. **Domain Relevance**: Include entity types that are central to THIS domain
       - For medical text: Patient, Medication, Condition, Procedure, Hospital
       - For business text: Company, Person, Investment, Product, Market
       - Don't include types that rarely appear or aren't relevant to this specific text
    
    3. **Distinctiveness Test**: Only create separate types if they have:
       - Different attributes or characteristics
       - Different relationship patterns
       - Different analytical value
       - Example: "Professor" and "Student" are distinct if their relationships differ
       - Example: "CEO" and "CTO" might both be "ExecutiveRole" if relationships are similar
    
    4. **Generalization**: Combine similar concepts under broader types when appropriate
       - Instead of: "Dog", "Cat", "Bird" → use "Animal" (if distinctions don't matter)
       - Instead of: "Laptop", "Desktop", "Tablet" → use "Device" or "Computer"
       - BUT: Keep separate if the distinction is analytically important for this domain
    
    5. **Coverage**: Ensure types cover the main concepts in the text
       - Don't create types for concepts that appear only once
       - DO create types for recurring or structurally important concepts
       - Focus on concepts that are central to understanding this domain
    
    RELATION TYPE GRANULARITY PRINCIPLES:
    
    1. **Semantic Precision**: Relations should capture meaningful distinctions
       - Too vague: "related_to", "associated_with" (not informative)
       - Too specific: "works_at_as_senior_engineer_since_2020" (overfitting)
       - Just right: "works_at", "employed_by", "manages" (clear semantics)
    
    2. **Directionality Matters**: Choose direction that reflects real-world semantics
       - "Person works_at Organization" (natural)
       - Not "Organization employs Person" (unless that's the focus)
       - Be consistent in your directional choices
    
    3. **Relationship Patterns**: Include relations that appear or could appear multiple times
       - Medical: "treats", "prescribes", "diagnoses", "hospitalized_at"
       - Business: "invests_in", "founded", "acquired_by", "partners_with"
       - Focus on the relationships that matter for this domain
    
    4. **Avoid Redundancy**: Don't create near-synonyms unless there's semantic distinction
       - Don't have both "works_at" and "employed_by" (pick one)
       - Don't have both "located_in" and "based_in" (pick one)
       - DO have "founded" and "acquired" (different events with different meanings)
    
    5. **Completeness**: Capture the key relationship types in the domain
       - Think: "What questions would someone ask about this data?"
       - Include relations that support those analytical questions
       - Every relation should serve a clear analytical purpose
    
    SCOPE-SPECIFIC GUIDANCE:
    
    **If scope is "minimal":**
    - Create only the most essential entity types (core concepts only)
    - Focus on the most frequent and important relationships
    - Err on the side of broader categories that combine similar concepts
    - Typical result: 3-8 entity types, 4-10 relation types
    - Use case: Quick extraction, simple queries, identifying broad patterns
    - Example: For a business article → Person, Organization, Location, works_at, located_in
    
    **If scope is "balanced":**
    - Include entity types for all significant concepts in the domain
    - Capture the main relationship patterns without over-specifying
    - Balance specificity with reusability across similar texts
    - Typical result: 6-12 entity types, 8-15 relation types
    - Use case: Standard analysis, general-purpose extraction, most common scenarios
    - Example: For a business article → Person, Organization, Location, Product, Investment, works_at, located_in, founded, invests_in, develops
    
    **If scope is "comprehensive":**
    - Include entity types for all distinct and meaningful concepts
    - Capture nuanced relationship types that reflect domain expertise
    - Allow for more domain-specific and specialized types
    - Typical result: 10-20 entity types, 12-25 relation types
    - Use case: Detailed analysis, domain expertise, research, specialized queries
    - Example: For a business article → Person, Organization, Location, Product, Investment, Technology, Market, Role, works_at, located_in, founded, invests_in, develops, competes_with, targets, acquired_by, partners_with
    
    QUALITY OVER QUANTITY:
    - It's better to have fewer, well-defined types than many poorly-defined ones
    - Each type should have a clear, distinct purpose and detailed description
    - Each type should be distinguishable from others in meaningful ways
    - Every type you create should be useful for analysis and querying
    - The numbers above are guidelines, not requirements - use your judgment
    
    SELF-CHECK QUESTIONS:
    Before finalizing your ontology, ask yourself:
    1. Can I clearly explain when to use each entity type vs. others?
    2. Would two different people classify entities consistently with these types?
    3. Are my relation types capturing distinct semantic relationships?
    4. Can I extract useful, actionable information with this ontology?
    5. Is this ontology reusable for similar texts in this domain?
    6. Does the granularity match the requested scope level?
    
    ONTOLOGY DESIGN PRINCIPLES:
    - Entity types should be nouns representing concrete or abstract concepts
    - Relation types should be verbs or verb phrases that connect entities
    - Each relation type must specify its domain (subject entity type) and range (object entity type)
    - Prefer commonly understood terms over jargon unless domain-specific terms are essential
    - Think about what would be useful for querying and analyzing this type of text
    - Ensure descriptions are clear and detailed enough for accurate extraction
    
    Your task is to analyze THIS specific text and recommend the most appropriate ontology for it,
    following the principles above and designing for the requested "{{ scope }}" scope level.
    
    {{ ctx.output_format }}
  "#
}

// Function to analyze if an existing ontology needs extension for new text
function AnalyzeOntologyExtension(
  text: string,
  current_ontology: Ontology,
  scope: string
) -> OntologyExtension {
  client CustomSonnet4
  prompt #"
    You are a knowledge graph ontology expert. Your task is to analyze whether an existing ontology needs to be extended to properly extract knowledge from new text.
    
    {{ _.role("user") }}
    
    CURRENT ONTOLOGY:
    
    Current Entity Types:
    {% for entity_type in current_ontology.entity_types %}
    - {{ entity_type.name }}: {{ entity_type.description }}
    {% endfor %}
    
    Current Relation Types:
    {% for relation_type in current_ontology.relation_types %}
    - {{ relation_type.name }}: {{ relation_type.description }}
      (Domain: {{ relation_type.domain }}, Range: {{ relation_type.range }})
    {% endfor %}
    
    NEW TEXT TO ANALYZE:
    {{ text }}
    
    ONTOLOGY SCOPE: {{ scope }}
    
    YOUR TASK:
    
    Analyze whether the current ontology is sufficient to extract critical information from this new text,
    or whether it needs to be conservatively extended.
    
    CONSERVATIVE EXTENSION PRINCIPLES:
    
    1. **Default to NO Extension**
       - Start with the assumption that the existing ontology is sufficient
       - Only recommend extension if absolutely necessary
       - Prefer using existing types creatively over adding new ones
    
    2. **Critical Information Test**
       - Extension is ONLY justified if failing to extend would result in losing CRITICAL information
       - Ask: "Can the essential facts be captured with existing types?"
       - If yes → No extension needed
       - If no → Consider extension
    
    3. **Definition of Critical Information**
       Critical information is information that:
       - Is central to understanding the text's main points
       - Cannot be reasonably represented with existing entity/relation types
       - Would create significant semantic loss if omitted or forced into existing types
       - Is not a minor detail or edge case
    
    4. **Examples of When Extension IS Needed**
       - New text discusses "Chemical Compounds" but ontology only has "Substance"
         → If chemical-specific relationships matter, extend
       - New text has "Software" entities but ontology only has "Product"
         → If software-specific attributes/relations are critical, extend
       - New text introduces entirely new domain (e.g., legal domain in a medical ontology)
         → If domain is central to the text, extend
    
    5. **Examples of When Extension is NOT Needed**
       - New text mentions "tablets" but ontology has "Device" → Use Device
       - New text has "acquired_by" but ontology has "purchased_by" → Use existing similar relation
       - New text mentions a rare entity type appearing once → Don't extend for one-offs
       - Differences are purely syntactic (naming) not semantic → Don't extend
    
    6. **Backward Compatibility**
       Any new types you recommend must:
       - Be distinct from existing types (not redundant or overlapping)
       - Follow the same naming and description conventions
       - Work alongside existing types without conflicts
       - Be at the appropriate granularity level for the scope
    
    7. **Scope Awareness**
       Consider the current scope ({{ scope }}):
       - For "minimal" scope: Be VERY conservative, only extend for major gaps
       - For "balanced" scope: Extend if missing significant concepts
       - For "comprehensive" scope: More willing to add nuanced types
    
    ANALYSIS PROCESS:
    
    Step 1: Identify concepts in the new text
    Step 2: For each concept, try to map it to existing entity types
    Step 3: For each relationship, try to map it to existing relation types
    Step 4: Identify any critical information that CANNOT be captured
    Step 5: If critical gaps exist, propose minimal extensions
    Step 6: Justify why each extension is necessary
    
    DECISION CRITERIA:
    
    Set needs_extension = true ONLY if:
    ✓ There are concepts in the text that are critically important
    ✓ These concepts cannot be reasonably mapped to existing types
    ✓ Omitting them would cause significant information loss
    ✓ The extensions align with the current scope level
    
    Set needs_extension = false if:
    ✓ All critical information can be captured with existing types
    ✓ Minor details might be lost but core facts are preserved
    ✓ Existing types can be used flexibly to cover new concepts
    
    OUTPUT REQUIREMENTS:
    
    - If needs_extension = false:
      * Set new_entity_types and new_relation_types to empty arrays
      * Explain in reasoning how existing types cover the text
      * Be specific about which existing types map to which concepts
    
    - If needs_extension = true:
      * Provide ONLY the new types needed (don't repeat existing ones)
      * Describe in critical_information_at_risk what would be lost
      * Explain in reasoning why each new type is essential
      * Keep extensions minimal - only add what's absolutely necessary
    
    QUALITY CHECKS:
    
    Before recommending extension, verify:
    1. Can I extract the main facts with existing types? (If yes → no extension)
    2. Is the "missing" information truly critical? (If no → no extension)
    3. Are my new types distinct from existing ones? (If not → no extension)
    4. Am I being conservative? (If not → reconsider)
    
    Remember: It's better to slightly stretch existing types than to proliferate unnecessary new types.
    The ontology should evolve slowly and deliberately, not expand with every new text.
    
    {{ ctx.output_format }}
  "#
}

