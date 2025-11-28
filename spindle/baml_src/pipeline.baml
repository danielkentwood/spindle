// Ontology Pipeline: BAML definitions for semantic knowledge extraction
// Implements the six-stage pipeline: Vocabulary, Metadata, Taxonomy, Thesaurus, Ontology, KG

// =============================================================================
// Stage 1: Controlled Vocabulary Types and Functions
// =============================================================================

class VocabularyTerm {
  term_id string @description("Unique identifier for this term")
  preferred_label string @description("The canonical/preferred form of this term")
  definition string @description("Clear definition explaining what this term means in context")
  synonyms string[] @description("Alternative terms or phrases that mean the same thing")
  domain string? @description("The subject domain this term belongs to (e.g., 'medical', 'legal', 'technology')")
}

class VocabularyExtractionResult {
  terms VocabularyTerm[] @description("Extracted vocabulary terms with definitions")
  domain_summary string @description("Brief summary of the domains/topics covered in the text")
  reasoning string @description("Explanation of term selection and definition choices")
}

function ExtractControlledVocabulary(
  text: string,
  existing_terms: VocabularyTerm[],
  document_id: string
) -> VocabularyExtractionResult {
  client CustomFast
  prompt #"
    You are a knowledge organization expert specializing in controlled vocabulary development.
    Your task is to extract key terms from the text and create clean, disambiguated vocabulary entries.

    {{ _.role("user") }}

    TEXT TO ANALYZE:
    {{ text }}

    DOCUMENT ID: {{ document_id }}

    {% if existing_terms|length > 0 %}
    EXISTING VOCABULARY:
    The following terms have already been extracted. You should:
    1. Use consistent naming - if a term exists, use the SAME preferred_label
    2. Add new synonyms to existing terms rather than creating duplicates
    3. Only create new terms for genuinely different concepts
    
    {% for term in existing_terms %}
    - {{ term.preferred_label }}: {{ term.definition }}
      Synonyms: {{ term.synonyms | join(", ") }}
    {% endfor %}
    {% endif %}

    CONTROLLED VOCABULARY PRINCIPLES:

    1. **Term Selection**: Identify key concepts, entities, and domain-specific terms
       - Focus on nouns and noun phrases
       - Include technical/domain terms that need definition
       - Skip common words that don't need clarification

    2. **Preferred Label**: Choose the most standard/canonical form
       - Use singular form for countable nouns
       - Use the most commonly accepted spelling
       - Prefer full terms over abbreviations (but list abbreviations as synonyms)

    3. **Definition**: Write clear, concise definitions
       - Start with the category: "A [category] that..."
       - Include distinguishing characteristics
       - Keep to 1-2 sentences

    4. **Synonyms**: List alternative forms
       - Abbreviations and acronyms
       - Variant spellings
       - Related terms that are used interchangeably in this context

    5. **Domain**: Identify the subject area
       - Be specific but not overly narrow
       - Examples: "healthcare", "finance", "technology", "legal"

    DEDUPLICATION:
    - If a concept already exists in the existing vocabulary, do NOT create a duplicate
    - Instead, you may suggest additional synonyms in your reasoning
    - Generate unique term_ids using format: "term_[document_id]_[index]"

    {{ ctx.output_format }}
  "#
}

function ConsolidateVocabulary(
  term_sets: VocabularyTerm[][]
) -> VocabularyExtractionResult {
  client CustomFast
  prompt #"
    You are a knowledge organization expert. Your task is to merge and deduplicate
    vocabulary terms from multiple extraction runs into a single, coherent vocabulary.

    {{ _.role("user") }}

    TERM SETS TO CONSOLIDATE:
    {% for term_set in term_sets %}
    --- Set {{ loop.index }} ---
    {% for term in term_set %}
    - {{ term.preferred_label }}: {{ term.definition }}
      Synonyms: {{ term.synonyms | join(", ") }}
      Domain: {{ term.domain or "unspecified" }}
    {% endfor %}
    {% endfor %}

    CONSOLIDATION RULES:

    1. **Merge Duplicates**: Combine terms that refer to the same concept
       - Keep the best/most complete definition
       - Combine all synonyms from duplicate entries
       - Use the most specific domain classification

    2. **Resolve Conflicts**: When definitions differ slightly
       - Prefer more precise definitions
       - Incorporate unique aspects from each
       - Note any genuine ambiguity

    3. **Maintain Consistency**: Ensure the final vocabulary is coherent
       - No duplicate preferred_labels
       - Synonyms should point to only one preferred term
       - Domains should be consistently applied

    4. **Generate New IDs**: Create new term_ids using format: "term_consolidated_[index]"

    {{ ctx.output_format }}
  "#
}

// =============================================================================
// Stage 2: Metadata Standards Types and Functions
// =============================================================================

class MetadataElement {
  element_id string @description("Unique identifier for this metadata element")
  name string @description("Name of the metadata field (e.g., 'author', 'publication_date')")
  element_type string @description("Type: 'structural', 'descriptive', or 'administrative'")
  description string @description("What this metadata element captures")
  data_type string @description("Expected data type: 'string', 'int', 'float', 'bool', 'date', 'list'")
  required bool @description("Whether this element is required for all documents")
  examples string[] @description("Example values this element might contain")
}

class MetadataExtractionResult {
  elements MetadataElement[] @description("Discovered metadata schema elements")
  reasoning string @description("Explanation of the metadata schema design")
}

function ExtractMetadataSchema(
  text: string,
  document_metadata: string,
  existing_elements: MetadataElement[]
) -> MetadataExtractionResult {
  client CustomFast
  prompt #"
    You are an information architect specializing in metadata standards.
    Analyze the document to identify metadata elements that describe its structure and content.

    {{ _.role("user") }}

    DOCUMENT TEXT (sample):
    {{ text }}

    DOCUMENT METADATA (if available):
    {{ document_metadata }}

    {% if existing_elements|length > 0 %}
    EXISTING METADATA SCHEMA:
    {% for elem in existing_elements %}
    - {{ elem.name }} ({{ elem.element_type }}): {{ elem.description }}
    {% endfor %}
    {% endif %}

    METADATA ELEMENT TYPES:

    1. **Structural Metadata** - For machine readability
       - Format, encoding, file type
       - Structure markers (sections, chapters)
       - Technical specifications

    2. **Descriptive Metadata** - For context and discovery
       - Title, author, subject
       - Keywords, abstract, summary
       - Topic classification

    3. **Administrative Metadata** - For maintenance and lineage
       - Creation date, modification date
       - Source, provenance
       - Rights, access restrictions
       - Version information

    GUIDELINES:
    - Identify patterns in the document that suggest metadata fields
    - Consider what metadata would be useful for search and organization
    - Be specific about data types and whether fields are required
    - Generate unique element_ids using format: "meta_[index]"

    {{ ctx.output_format }}
  "#
}

// =============================================================================
// Stage 3: Taxonomy Types and Functions
// =============================================================================

class TaxonomyRelation {
  parent_term string @description("The broader/parent concept label")
  child_term string @description("The narrower/child concept label")
  relation_type string @description("Relationship type: 'broader' or 'narrower'")
  confidence string @description("Confidence level: 'high', 'medium', or 'low'")
}

class TaxonomyExtractionResult {
  relations TaxonomyRelation[] @description("Hierarchical relationships between terms")
  root_terms string[] @description("Top-level terms with no parents")
  reasoning string @description("Explanation of the taxonomic structure")
}

function ExtractTaxonomy(
  terms: VocabularyTerm[],
  text: string,
  existing_relations: TaxonomyRelation[]
) -> TaxonomyExtractionResult {
  client CustomFast
  prompt #"
    You are a taxonomist expert in hierarchical classification systems.
    Build a taxonomy by identifying parent-child relationships between vocabulary terms.

    {{ _.role("user") }}

    VOCABULARY TERMS:
    {% for term in terms %}
    - {{ term.preferred_label }}: {{ term.definition }}
      Domain: {{ term.domain or "unspecified" }}
    {% endfor %}

    CONTEXT TEXT:
    {{ text }}

    {% if existing_relations|length > 0 %}
    EXISTING TAXONOMY RELATIONS:
    {% for rel in existing_relations %}
    - {{ rel.parent_term }} > {{ rel.child_term }}
    {% endfor %}
    {% endif %}

    TAXONOMY PRINCIPLES:

    1. **Hierarchical Structure**
       - Each term can have only ONE parent (strict hierarchy)
       - Terms without parents become root terms
       - Deeper levels = more specific concepts

    2. **IS-A Relationships**
       - Child "is a type of" parent
       - Example: "Dog" is-a "Animal"
       - Avoid part-of relationships (those go in thesaurus)

    3. **Consistency**
       - Siblings should be at the same level of abstraction
       - Use the vocabulary's preferred_label exactly
       - Don't create relations for terms not in the vocabulary

    4. **Confidence Levels**
       - high: Clear hierarchical relationship from text/domain knowledge
       - medium: Reasonable inference but not explicitly stated
       - low: Plausible but uncertain

    Identify root terms (concepts with no broader parent in this vocabulary).

    {{ ctx.output_format }}
  "#
}

// =============================================================================
// Stage 4: Thesaurus Types and Functions
// =============================================================================

class ThesaurusEntry {
  entry_id string @description("Unique identifier for this entry")
  term_id string @description("Reference to the vocabulary term")
  preferred_label string @description("The preferred term label")
  use_for string[] @description("Non-preferred synonyms (USE_FOR/UF relationships)")
  broader_terms string[] @description("Broader/parent terms (BT relationships)")
  narrower_terms string[] @description("Narrower/child terms (NT relationships)")
  related_terms string[] @description("Associated terms (RT relationships)")
  scope_note string? @description("Definition or scope clarification (SN)")
}

class ThesaurusExtractionResult {
  entries ThesaurusEntry[] @description("Thesaurus entries with semantic relationships")
  reasoning string @description("Explanation of relationship assignments")
}

function ExtractThesaurus(
  terms: VocabularyTerm[],
  taxonomy_relations: TaxonomyRelation[],
  text: string
) -> ThesaurusExtractionResult {
  client CustomFast
  prompt #"
    You are a thesaurus construction expert following ISO 25964 standards.
    Build thesaurus entries by adding semantic relationships to vocabulary terms.

    {{ _.role("user") }}

    VOCABULARY TERMS:
    {% for term in terms %}
    - {{ term.preferred_label }}: {{ term.definition }}
      Synonyms: {{ term.synonyms | join(", ") }}
    {% endfor %}

    EXISTING TAXONOMY (BT/NT relationships):
    {% for rel in taxonomy_relations %}
    - {{ rel.parent_term }} BT {{ rel.child_term }}
    {% endfor %}

    CONTEXT TEXT:
    {{ text }}

    THESAURUS RELATIONSHIP TYPES (ISO 25964):

    1. **USE / USE_FOR (UF)**: Synonym control
       - Preferred term USE non-preferred
       - Non-preferred term USE_FOR preferred
       - From vocabulary synonyms

    2. **Broader Term (BT)**: Hierarchical - genus
       - From taxonomy parent relationships
       - "Concept A is a type of Concept B"

    3. **Narrower Term (NT)**: Hierarchical - species
       - From taxonomy child relationships
       - Inverse of BT

    4. **Related Term (RT)**: Associative
       - NOT hierarchical (no is-a relationship)
       - Concepts often discussed together
       - Same domain, complementary concepts
       - Avoid if BT/NT already connects them

    5. **Scope Note (SN)**: Definition/clarification
       - From vocabulary definition
       - Additional context if needed

    GUIDELINES:
    - BT/NT come from taxonomy_relations
    - UF comes from vocabulary synonyms  
    - RT is for non-hierarchical associations found in text
    - Generate entry_ids using format: "thes_[index]"

    {{ ctx.output_format }}
  "#
}

// =============================================================================
// Stage 5: Ontology Enhancement (uses existing RecommendOntology but with context)
// =============================================================================

class OntologyEnhancementResult {
  entity_types EntityType[] @description("Recommended entity types informed by vocabulary")
  relation_types RelationType[] @description("Recommended relation types informed by thesaurus")
  reasoning string @description("How vocabulary/taxonomy informed the ontology design")
}

function EnhanceOntologyFromPipeline(
  text: string,
  vocabulary: VocabularyTerm[],
  thesaurus_entries: ThesaurusEntry[],
  scope: string
) -> OntologyEnhancementResult {
  client CustomFast
  prompt #"
    You are an ontology engineer. Design an ontology for knowledge graph extraction,
    informed by the controlled vocabulary and thesaurus already built for this corpus.

    {{ _.role("user") }}

    CONTROLLED VOCABULARY:
    {% for term in vocabulary %}
    - {{ term.preferred_label }}: {{ term.definition }}
      Domain: {{ term.domain or "general" }}
    {% endfor %}

    THESAURUS RELATIONSHIPS:
    {% for entry in thesaurus_entries %}
    - {{ entry.preferred_label }}
      {% if entry.broader_terms|length > 0 %}BT: {{ entry.broader_terms | join(", ") }}{% endif %}
      {% if entry.related_terms|length > 0 %}RT: {{ entry.related_terms | join(", ") }}{% endif %}
    {% endfor %}

    SAMPLE TEXT:
    {{ text }}

    SCOPE: {{ scope }}

    ONTOLOGY DESIGN PRINCIPLES:

    1. **Entity Types from Vocabulary**
       - High-level vocabulary terms often become entity types
       - Root taxonomy terms are good candidates
       - Group related terms under entity types

    2. **Relation Types from Thesaurus**
       - RT (related term) relationships suggest relation types
       - Consider the semantic nature of associations
       - Ensure domain/range constraints match entity types

    3. **Scope Guidelines**
       {% if scope == "minimal" %}
       - Essential types only (3-8 entities, 4-10 relations)
       - Broad categories, minimal attributes
       {% elif scope == "balanced" %}
       - Core types plus important distinctions (6-12 entities, 8-15 relations)
       - Standard attributes for each type
       {% else %}
       - Comprehensive coverage (10-20 entities, 12-25 relations)
       - Detailed attributes and nuanced relations
       {% endif %}

    4. **Attribute Design**
       - Use vocabulary terms to inform attributes
       - Consider what properties distinguish entities

    {{ ctx.output_format }}
  "#
}

// =============================================================================
// Utility: Import EntityType and RelationType from spindle.baml
// =============================================================================

// Note: EntityType and RelationType are already defined in spindle.baml
// They are automatically available in the BAML client

