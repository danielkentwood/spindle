# Spindle Redesign: Knowledge Governance Platform Architecture

## Executive Summary

This document outlines a first-principles redesign of Spindle, transforming it from a knowledge extraction tool into a comprehensive **Knowledge Governance Platform** optimized for enterprise knowledge systems. The redesign addresses three critical dimensions:

1. **Extraction**: Multi-perspective, quality-gated extraction with confidence scoring and tacit knowledge capture
2. **Maintenance**: Version control, change detection, drift analysis, and continuous learning
3. **Governance**: Validation workflows, quality metrics, conflict resolution, and expert review

A key innovation in this redesign is the integration of **LinkML** ([linkml.io](https://linkml.io/)) as the modeling language for governance schemas, lifecycle management, and knowledge structures. LinkML provides a unified way to define models that can be automatically converted to JSON-Schema, RDF/OWL, Python dataclasses, and other formats, ensuring consistency across the entire knowledge lifecycle.

The redesign maintains **BAML** ([baml.dev](https://baml.dev/)) for LLM prompt engineering and extraction-specific schemas, creating a clear separation of concerns: BAML handles LLM interaction and extraction outputs, while LinkML handles knowledge governance and lifecycle management.

---

## Table of Contents

1. [Design Principles](#design-principles)
2. [Architecture Overview](#architecture-overview)
3. [Template-Driven Extraction System](#template-driven-extraction-system)
4. [LinkML Integration](#linkml-integration)
5. [BAML-LinkML Integration Strategy](#baml-linkml-integration-strategy)
6. [Core Layers](#core-layers)
7. [Implementation Architecture](#implementation-architecture)
8. [Migration Path](#migration-path)
9. [Benefits](#benefits)

---

## Design Principles

### 1. Knowledge as a Living System
Knowledge artifacts are versioned, validated, and continuously evolving. They have lifecycle states, version history, and change tracking.

### 2. Multi-Perspective Support
The system captures and reconciles conflicting views from different stakeholders, preserving source attribution and enabling consensus mechanisms.

### 3. Governance-First Architecture
Validation, quality metrics, and change management are built into every layer, not bolted on as afterthoughts.

### 4. Unified Semantic Representation
Processes, entities, and facts exist in a single unified semantic model, enabling queries across all knowledge types.

### 5. Continuous Learning
Feedback loops from usage, validation, and corrections drive automated refinement and knowledge improvement.

### 6. Schema-Driven Architecture
Governance schemas, lifecycle models, and knowledge structures are defined in LinkML, enabling automatic code generation, validation, and interoperability. LLM extraction schemas remain in BAML for optimal prompt engineering and type-safe LLM interactions.

---

## Architecture Overview

The redesigned Spindle consists of six interconnected layers:

```
┌─────────────────────────────────────────────────────────┐
│ Layer 6: Unified Knowledge Representation                │
│ (Processes + Entities + Facts in one semantic model)   │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│ Layer 5: Maintenance & Continuous Learning             │
│ (Feedback loops, process mining, automated refinement) │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│ Layer 4: Version Control & Change Management           │
│ (Git-like versioning, drift detection, change tracking) │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│ Layer 3: Governance & Validation                       │
│ (Validation workflows, quality metrics, expert review)  │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│ Layer 2: Extraction & Acquisition                       │
│ (Multi-perspective extraction, quality gates)          │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│ Layer 1: Knowledge Foundation                          │
│ (Lifecycle states, abstraction levels, registry)         │
└─────────────────────────────────────────────────────────┘
```

---

## Template-Driven Extraction System

### Overview

Spindle uses a **unified template-driven architecture** where extraction templates specify goals, constraints, and extraction strategies that influence all pipeline stages from vocabulary building through final graph extraction. This approach unifies knowledge graph extraction, process graph extraction, and future graph types under a single, configurable framework.

### Template Architecture

Templates are defined in YAML/JSON and specify:

1. **Graph Type**: The type of graph to extract (knowledge graph, process graph, workflow graph, etc.)
2. **Extraction Goals**: What to extract and how to structure it
3. **Stage Instructions**: How each pipeline stage should adapt to template requirements
4. **Constraints**: Validation rules, relationship types, quality criteria
5. **Output Configuration**: How to structure and store extracted graphs

### Template Structure

```yaml
# Example: spindle/templates/process_graph.yaml
name: process_graph
description: Extract process DAGs with workflow semantics
version: 1.0.0

graph_type: process_graph
output_format: ProcessGraph

# Stage-specific instructions
stages:
  vocabulary:
    focus: ["action_verbs", "temporal_terms", "decision_points"]
    exclude: ["static_entities"]
    instructions: |
      Prioritize terms that describe actions, sequences, and decisions.
      Focus on procedural language rather than entity descriptions.
  
  taxonomy:
    relationship_types: ["precedes", "enables", "blocks", "requires"]
    instructions: |
      Build hierarchical relationships between process steps.
      Identify parent-child relationships in nested processes.
  
  thesaurus:
    semantic_relations: ["synonym", "antonym", "temporal", "causal"]
    instructions: |
      Map action synonyms (e.g., "start" = "begin" = "initiate").
      Identify temporal relationships between actions.
  
  ontology:
    entity_types:
      - name: ProcessStep
        description: A step in a process workflow
        attributes:
          - name: step_type
            type: enum
            values: [ACTIVITY, DECISION, EVENT, PARALLEL_GATEWAY, SUBPROCESS]
      - name: Actor
        description: Person or system that performs a step
      - name: Resource
        description: Input or output of a process step
    relation_types:
      - name: precedes
        domain: ProcessStep
        range: ProcessStep
        description: Step A must complete before Step B starts
      - name: requires
        domain: ProcessStep
        range: Resource
        description: Step requires this resource as input
      - name: produces
        domain: ProcessStep
        range: Resource
        description: Step produces this resource as output
    instructions: |
      Build ontology focused on process semantics.
      Entity types should emphasize actions and workflows.
      Relation types should capture temporal and causal dependencies.
  
  extraction:
    baml_function: ExtractProcessGraph
    output_type: ProcessGraph
    constraints:
      - no_circular_dependencies
      - valid_start_end_steps
      - consistent_step_ids
    quality_gates:
      - completeness_score > 0.7
      - dependency_coverage > 0.8
    instructions: |
      Extract process DAG with steps and dependencies.
      Capture actors, resources, and temporal relationships.
      Include evidence spans and confidence scores.

# Template metadata
metadata:
  domain: process_management
  use_cases: ["workflow_documentation", "process_analysis", "compliance"]
  tags: ["process", "workflow", "dag"]
```

### Default Template

A default template provides simple, naive extraction for users who don't need customization:

```yaml
# spindle/templates/default_knowledge_graph.yaml
name: default_knowledge_graph
description: Default knowledge graph extraction (simple/naive)
version: 1.0.0

graph_type: knowledge_graph
output_format: Triple[]

stages:
  vocabulary:
    instructions: Extract all significant terms with definitions
  
  taxonomy:
    instructions: Build hierarchical relationships between concepts
  
  thesaurus:
    instructions: Identify synonyms and related terms
  
  ontology:
    instructions: Auto-recommend ontology based on text analysis
    scope: balanced
  
  extraction:
    baml_function: ExtractTriples
    output_type: ExtractionResult
    instructions: Extract triples conforming to recommended ontology
```

### Template Influence on Pipeline Stages

Templates influence each stage through context propagation:

```python
# Template context flows through all stages
template_context = {
    "template": template_spec,
    "graph_type": template.graph_type,
    "stage_instructions": template.stages,
    "constraints": template.constraints
}

# Each stage receives template context
vocabulary_stage.extract_from_text(
    text=text,
    context={
        **template_context,
        "vocabulary_instructions": template.stages.vocabulary.instructions
    }
)

ontology_stage.extract_from_text(
    text=text,
    context={
        **template_context,
        "ontology_config": template.stages.ontology,
        "vocabulary": vocabulary_artifacts,
        "thesaurus": thesaurus_artifacts
    }
)

extraction_stage.extract_from_text(
    text=text,
    context={
        **template_context,
        "extraction_config": template.stages.extraction,
        "ontology": ontology_artifact
    }
)
```

### Template Registry

Templates are registered and resolved based on user selection:

```python
# Template registry
class ExtractionTemplate:
    name: str
    graph_type: str  # "knowledge_graph", "process_graph", etc.
    stages: Dict[str, StageConfig]
    constraints: List[Constraint]
    metadata: TemplateMetadata

class TemplateRegistry:
    def register(self, template: ExtractionTemplate) -> None:
        """Register a template."""
    
    def get(self, name: str) -> ExtractionTemplate:
        """Get template by name."""
    
    def list_by_graph_type(self, graph_type: str) -> List[ExtractionTemplate]:
        """List templates for a graph type."""
    
    def resolve_default(self, graph_type: str) -> ExtractionTemplate:
        """Get default template for graph type."""
```

### Benefits of Template-Driven Approach

1. **Unified Architecture**: Single pipeline for all graph types
2. **Flexibility**: Easy to add new graph types via templates
3. **Consistency**: Same semantic foundation (vocabulary/ontology) for all extractions
4. **Customization**: Domain-specific templates without code changes
5. **Reusability**: Templates can be shared and versioned
6. **Simplicity**: Default templates for common use cases
7. **Incremental Enhancement**: Templates guide early stages for better final extraction

### Template Examples

**Knowledge Graph Template**:
- Focuses on entities and relationships
- Builds rich ontology with entity types
- Extracts subject-predicate-object triples

**Process Graph Template**:
- Focuses on actions and sequences
- Builds ontology with process semantics
- Extracts DAGs with steps and dependencies

**Workflow Template** (future):
- Combines process and knowledge graph elements
- Links processes to entities
- Extracts both procedural and factual knowledge

---

## LinkML Integration

### Why LinkML for Governance?

[LinkML](https://linkml.io/) is a general-purpose modeling language that provides:

- **YAML-based modeling**: Human-readable schema definitions
- **Multi-format code generation**: JSON-Schema, RDF/OWL, Python dataclasses, GraphQL, SQL DDL
- **Semantic web ready**: Automatic JSON-LD context generation
- **Flexible inheritance**: Support for complex class hierarchies
- **Semantic enumerations**: Enum binding to ontologies
- **Stealth semantics**: Everything has a URI, but developers work with YAML/JSON

LinkML is ideal for governance schemas because it provides the semantic richness, multi-format support, and interoperability needed for enterprise knowledge management systems. It complements BAML, which remains the optimal choice for LLM prompt engineering and extraction-specific schemas.

### LinkML Schema Structure

Governance and lifecycle schemas will be defined in LinkML YAML files:

```yaml
# Example: spindle/schemas/knowledge_lifecycle.yaml
id: https://spindle.ai/schemas/knowledge_lifecycle
name: knowledge_lifecycle
title: Knowledge Lifecycle Schema
description: Schema for knowledge artifact lifecycle management
version: 1.0.0

prefixes:
  spindle: https://spindle.ai/schemas/
  prov: http://www.w3.org/ns/prov#
  schema: https://schema.org/

imports:
  - linkml:types

classes:
  KnowledgeArtifact:
    description: Base class for all knowledge artifacts
    is_a: NamedThing
    slots:
      - artifact_id
      - lifecycle_state
      - abstraction_level
      - created_at
      - updated_at
      - version
      - provenance
    
  KnowledgeState:
    description: Lifecycle state of a knowledge artifact
    enum:
      - DRAFT
      - CANDIDATE
      - VALIDATED
      - DEPRECATED
      - ARCHIVED
  
  AbstractionLevel:
    description: Level of abstraction for knowledge
    enum:
      - SCHEMA
      - INSTANCE
      - PATTERN

slots:
  artifact_id:
    description: Unique identifier for the artifact
    range: string
    identifier: true
  
  lifecycle_state:
    description: Current lifecycle state
    range: KnowledgeState
    required: true
  
  abstraction_level:
    description: Level of abstraction
    range: AbstractionLevel
    required: true
  
  version:
    description: Version identifier
    range: string
    required: true
  
  provenance:
    description: Provenance information (PROV-O)
    range: ProvenanceRecord
    multivalued: false
```

### LinkML Schema Organization

```
spindle/schemas/                   # LinkML schemas for governance
├── knowledge_lifecycle.yaml      # Lifecycle states and abstraction levels
├── governance.yaml                # Validation, quality, governance
├── versioning.yaml                # Version control, changes, drift
├── feedback.yaml                  # Feedback, usage tracking, learning
├── unified_knowledge.yaml         # Unified entity-process-fact model
└── knowledge_artifacts.yaml       # Knowledge artifact base schemas

spindle/baml_src/                  # BAML schemas for LLM extraction
├── spindle.baml                   # Triple extraction (Entity, Triple, Ontology)
├── process.baml                   # Process extraction
├── entity_resolution.baml         # Entity resolution prompts
└── pipeline.baml                  # Pipeline stage extraction
```

### Code Generation from LinkML

LinkML automatically generates:

1. **Python Dataclasses**: Type-safe Python classes for all models
2. **JSON-Schema**: Validation schemas for JSON data
3. **RDF/OWL**: Semantic web ontologies
4. **JSON-LD Contexts**: Linked data contexts
5. **GraphQL Schemas**: API schemas
6. **SQL DDL**: Database schemas

Example usage:

```python
# Auto-generated from LinkML schemas
from spindle.schemas.generated import (
    KnowledgeArtifact,
    KnowledgeState,
    ExtractionResult,
    QualityMetrics,
    ValidationWorkflow
)

# Type-safe creation
artifact = KnowledgeArtifact(
    artifact_id="kg:triple:12345",
    lifecycle_state=KnowledgeState.DRAFT,
    abstraction_level=AbstractionLevel.INSTANCE,
    version="1.0.0"
)

# Automatic serialization
json_data = artifact.to_json()
rdf_data = artifact.to_rdf()
```

---

## BAML-LinkML Integration Strategy

### Overview

Spindle uses a **dual-schema architecture** that leverages the strengths of both BAML and LinkML:

- **BAML**: LLM prompt engineering and extraction-specific schemas (`Entity`, `Triple`, `Ontology`, `ExtractionResult`)
- **LinkML**: Knowledge governance, lifecycle management, and system architecture schemas (`KnowledgeArtifact`, `ValidationWorkflow`, `QualityMetrics`, `Version`)

This separation provides clear boundaries: BAML handles the LLM interaction layer, while LinkML handles the knowledge governance layer.

### Integration Pattern

The integration follows a clear data flow pattern:

```
┌─────────────────────────────────────────────────────────┐
│ LLM Extraction Layer (BAML)                            │
│                                                         │
│  Text → BAML ExtractTriples() → BAML ExtractionResult │
│         - Entity (BAML)                                │
│         - Triple (BAML)                                │
│         - Ontology (BAML)                              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Conversion Bridge Layer                                 │
│                                                         │
│  BAML → LinkML Converter                               │
│  - Converts BAML ExtractionResult to LinkML artifacts  │
│  - Adds governance metadata                            │
│  - Creates KnowledgeArtifact instances                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Knowledge Governance Layer (LinkML)                    │
│                                                         │
│  LinkML KnowledgeArtifact → Governance Pipeline        │
│  - Validation workflows                                │
│  - Version control                                     │
│  - Quality metrics                                     │
│  - Lifecycle management                                │
└─────────────────────────────────────────────────────────┘
```

### Conversion Bridge

The conversion bridge transforms BAML extraction results into LinkML-governed artifacts:

```python
# spindle/extraction/bridge.py

from spindle.baml_client.types import (
    ExtractionResult as BAMLExtractionResult,
    Triple as BAMLTriple,
    Entity as BAMLEntity,
    Ontology as BAMLOntology
)
from spindle.schemas.generated import (
    KnowledgeArtifact,
    KnowledgeState,
    AbstractionLevel,
    ProvenanceRecord
)

def convert_baml_to_linkml_artifact(
    baml_result: BAMLExtractionResult,
    source_metadata: dict
) -> KnowledgeArtifact:
    """
    Convert BAML extraction result to LinkML KnowledgeArtifact.
    
    This bridge function:
    1. Extracts triples from BAML ExtractionResult
    2. Creates KnowledgeArtifact with governance metadata
    3. Sets lifecycle state to DRAFT (requires validation)
    4. Preserves provenance information
    """
    # Create provenance record from BAML source metadata
    provenance = ProvenanceRecord(
        source_name=source_metadata.get("source_name"),
        source_url=source_metadata.get("source_url"),
        extraction_datetime=datetime.utcnow().isoformat(),
        extraction_method="BAML_LLM_EXTRACTION"
    )
    
    # Create knowledge artifact
    artifact = KnowledgeArtifact(
        artifact_id=f"kg:extraction:{uuid.uuid4()}",
        lifecycle_state=KnowledgeState.DRAFT,
        abstraction_level=AbstractionLevel.INSTANCE,
        version="1.0.0",
        provenance=provenance,
        # Store BAML extraction result as raw data
        raw_extraction_data=baml_result.model_dump(),
        # Extract quality metrics from BAML result
        quality_metrics=extract_quality_metrics(baml_result)
    )
    
    return artifact
```

### Schema Responsibilities

#### BAML Schemas (LLM Extraction)

**Purpose**: Define structures for LLM prompt engineering and extraction outputs.

**Scope**:
- `Entity`: Entity structure for extraction (name, type, description, custom attributes)
- `Triple`: Subject-predicate-object relationships with supporting evidence
- `Ontology`: Entity types and relation types for extraction
- `ExtractionResult`: Complete extraction output with triples and reasoning
- `Process`: Process extraction structures
- `VocabularyTerm`, `MetadataElement`, etc.: Pipeline stage extraction structures

**Characteristics**:
- Optimized for LLM prompt engineering
- Includes extraction-specific metadata (supporting spans, extraction datetime)
- Type-safe Python clients auto-generated by BAML
- Focused on extraction accuracy and LLM interaction

#### LinkML Schemas (Knowledge Governance)

**Purpose**: Define structures for knowledge lifecycle, governance, and system architecture.

**Scope**:
- `KnowledgeArtifact`: Base class for all governed knowledge artifacts
- `ValidationWorkflow`: Validation and approval workflows
- `QualityMetrics`: Comprehensive quality assessment
- `Version`: Version control and change tracking
- `ChangeSet`: Change detection and drift analysis
- `Feedback`: Usage tracking and continuous learning
- `GovernancePolicy`: Domain-specific governance rules

**Characteristics**:
- Multi-format code generation (JSON-Schema, RDF/OWL, Python, GraphQL, SQL)
- Semantic web ready (JSON-LD contexts, RDF/OWL ontologies)
- Lifecycle and governance metadata
- Enterprise knowledge management features

### Benefits of This Approach

1. **Separation of Concerns**
   - BAML focuses on LLM interaction and extraction accuracy
   - LinkML focuses on knowledge governance and lifecycle management
   - Clear boundaries prevent schema conflicts

2. **No Duplication**
   - BAML schemas remain focused on extraction needs
   - LinkML schemas focus on governance needs
   - No redundant schema definitions

3. **Type Safety Throughout**
   - BAML provides type-safe LLM client generation
   - LinkML provides type-safe governance model generation
   - Conversion bridge ensures type-safe transformations

4. **Independent Evolution**
   - Extraction schemas can evolve based on LLM capabilities
   - Governance schemas can evolve based on enterprise needs
   - Changes in one layer don't require changes in the other

5. **Optimal Tool Selection**
   - BAML excels at LLM prompt engineering and structured outputs
   - LinkML excels at semantic modeling and multi-format interoperability
   - Each tool is used for its strengths

### Migration Considerations

During migration:

1. **BAML schemas remain unchanged**: Existing BAML extraction code continues to work
2. **LinkML schemas are additive**: New governance features use LinkML
3. **Conversion bridge is introduced**: Gradual migration path for existing artifacts
4. **Dual support period**: Both schemas coexist during transition
5. **Gradual adoption**: Teams can adopt governance features incrementally

---

## Core Layers

### Layer 1: Knowledge Foundation Layer

**Purpose**: Organize knowledge by abstraction level and lifecycle state.

#### Key Concepts

```yaml
# spindle/schemas/knowledge_foundation.yaml
classes:
  KnowledgeRegistry:
    description: Central catalog of all knowledge artifacts
    slots:
      - artifacts
      - schemas
      - instances
      - patterns
  
  SchemaStore:
    description: Storage for abstract schemas (ontologies, types)
    slots:
      - entity_types
      - relation_types
      - process_definitions
      - ontology_versions
  
  InstanceStore:
    description: Storage for concrete instances (triples, executions)
    slots:
      - triples
      - process_executions
      - entity_instances
  
  PatternLibrary:
    description: Extracted patterns from instance data
    slots:
      - patterns
      - pattern_frequency
      - pattern_confidence
```

#### Implementation

- **Knowledge Registry**: Central catalog with lifecycle state tracking
- **Schema Store**: Separate storage for ontologies and type definitions
- **Instance Store**: GraphStore for actual triples and process executions
- **Pattern Library**: Extracted patterns that inform schema evolution

---

### Layer 2: Extraction & Acquisition Layer

**Purpose**: Template-driven multi-stage extraction pipeline with quality gates and multi-perspective support.

**Note**: This layer uses **BAML** for LLM extraction schemas (`Entity`, `Triple`, `Ontology`, `ExtractionResult`, `ProcessGraph`) and **LinkML** for governance schemas that wrap extraction results. All extraction is driven by **templates** that specify graph type, extraction goals, and stage-specific instructions.

#### BAML Extraction Schemas

Extraction-specific schemas remain in BAML for optimal LLM interaction:

```baml
// spindle/baml_src/spindle.baml
class Entity {
  name string
  type string
  description string
  custom_atts map<string, AttributeValue>
}

class Triple {
  subject Entity
  predicate string
  object Entity
  source SourceMetadata
  supporting_spans CharacterSpan[]
  extraction_datetime string?
}

class ExtractionResult {
  triples Triple[]
  reasoning string
}

// spindle/baml_src/process.baml
class ProcessGraph {
  process_name string?
  scope string?
  primary_goal string
  start_step_ids string[]
  end_step_ids string[]
  steps ProcessStep[]
  dependencies ProcessDependency[]
  notes string[]
}

class ProcessStep {
  step_id string
  title string
  summary string
  step_type ProcessStepType  // ACTIVITY, DECISION, EVENT, etc.
  actors string[]
  inputs string[]
  outputs string[]
  duration string?
  prerequisites string[]
  evidence EvidenceSpan[]
}
```

#### LinkML Governance Schema

```yaml
# spindle/schemas/extraction.yaml
classes:
  ExtractionResult:
    description: Result of knowledge extraction
    slots:
      - triples
      - processes
      - quality_metrics
      - conflicts
      - confidence_scores
      - extraction_metadata
      - validation_requirements
  
  QualityMetrics:
    description: Quality assessment metrics
    slots:
      - completeness_score
      - confidence_score
      - consistency_score
      - coverage_score
      - validation_status
  
  KnowledgeConflict:
    description: Detected conflict in knowledge
    slots:
      - conflict_type
      - conflicting_artifacts
      - severity
      - resolution_strategy
  
  MultiPerspectiveExtraction:
    description: Extraction from multiple sources
    slots:
      - perspectives
      - agreement_areas
      - disagreement_areas
      - consensus_score
```

#### Process Knowledge Extraction Enhancements

Based on process knowledge management principles, the extraction layer includes specialized capabilities for capturing both explicit and tacit process knowledge:

**Enhanced Process Extraction**:
- **PKO-Inspired Modeling**: Distinguishes between procedures (abstract specifications) and executions (concrete instances)
- **Semantic Relationship Types**: Beyond simple "precedes", includes `requires`, `produces`, `governed_by`, `enables`, `validates`
- **Tacit Knowledge Capture**: Extracts exceptions, variations, judgment calls, and contextual factors
- **Multi-Level Abstraction**: Supports hierarchical process modeling (strategic, tactical, operational)
- **Evidence and Confidence**: Captures supporting text spans with confidence scores and ambiguity markers
- **Exception Handling**: Captures alternative flows, conditional variations, and workarounds

**Process-Specific Quality Gates**:
- **Completeness Checks**: Validates that all documented steps are captured
- **Dependency Validation**: Ensures no circular dependencies or invalid structures
- **Actor Expertise Tracking**: Captures required skill levels and training requirements
- **Contextual Sensitivity**: Identifies process variations based on context

#### Template-Driven Pipeline Flow

```
Template Selection
       ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 1: Vocabulary                                      │
│ - Template influences term focus and priorities        │
│ - Process templates focus on action verbs               │
│ - Knowledge graph templates focus on entities           │
└─────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 2: Metadata                                        │
│ - Template may specify metadata requirements            │
│ - Process templates capture workflow metadata           │
└─────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 3: Taxonomy                                        │
│ - Template specifies relationship types                 │
│ - Process templates: precedes, enables, blocks         │
│ - Knowledge graph templates: broader semantic types    │
└─────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 4: Thesaurus                                      │
│ - Template guides semantic relationship extraction      │
│ - Process templates emphasize temporal/causal links    │
└─────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 5: Ontology                                       │
│ - Template specifies entity/relation types             │
│ - Process templates: ProcessStep, Actor, Resource     │
│ - Knowledge graph templates: domain-specific entities  │
└─────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 6: Graph Extraction                               │
│ - Template selects BAML function (ExtractTriples, etc.)│
│ - Template specifies output format and constraints     │
│ - Quality gates enforced per template                  │
└─────────────────────────────────────────────────────────┘
```

#### Key Components

1. **Template Registry**: Manages extraction templates, resolves templates by name/type
2. **Template-Aware Pipeline Stages**: All stages receive and adapt to template instructions
3. **BAML Extraction Functions**: LLM-based extraction using BAML schemas, selected by template
4. **Conversion Bridge**: Transforms BAML extraction results to LinkML `KnowledgeArtifact`
5. **Multi-Perspective Extractor**: Extract from multiple sources, flag agreements/disagreements
6. **Quality Assessment Engine**: Confidence scoring, completeness checks, consistency validation (template-specific)
7. **Conflict Detection System**: Compare against existing knowledge, detect contradictions
8. **Tacit Knowledge Capture**: Extract exceptions, variations, judgment calls (enhanced by template)
9. **Graph Type Handlers**: Specialized handlers for each graph type (knowledge graph, process graph, etc.)

---

### Layer 3: Governance & Validation Layer

**Purpose**: Comprehensive governance framework with validation workflows and quality metrics.

#### LinkML Schema

```yaml
# spindle/schemas/governance.yaml
classes:
  ValidationWorkflow:
    description: Workflow for validating knowledge artifacts
    slots:
      - workflow_id
      - artifact
      - validators
      - validation_criteria
      - current_stage
      - history
  
  ValidationDecision:
    description: Decision from a validator
    enum:
      - APPROVE
      - REJECT
      - REQUEST_CHANGES
      - DEFER
      - CONFLICT
  
  ValidationResult:
    description: Result of validation
    slots:
      - decision
      - validator_id
      - comments
      - timestamp
      - confidence
  
  QualityMetrics:
    description: Comprehensive quality metrics
    slots:
      - completeness_score
      - confidence_score
      - consistency_score
      - coverage_score
      - validation_status
      - expert_approval_count
      - conflict_count
      - last_validated_at
  
  GovernancePolicy:
    description: Governance policy rules
    slots:
      - policy_id
      - domain
      - validation_rules
      - required_validators
      - approval_thresholds
      - escalation_paths
```

#### Key Components

1. **Validation Engine**: Rule-based auto-validation, expert review workflows
2. **Quality Dashboard**: Real-time metrics, validation queue, conflict resolution
3. **Governance Policies**: Domain-specific rules, approval thresholds, escalation

---

### Layer 4: Version Control & Change Management

**Purpose**: Git-like version control for knowledge artifacts.

#### LinkML Schema

```yaml
# spindle/schemas/versioning.yaml
classes:
  KnowledgeVersionControl:
    description: Version control system for knowledge
    slots:
      - versions
      - branches
      - change_history
  
  Version:
    description: Version of a knowledge artifact
    slots:
      - version_id
      - artifact_id
      - change_type
      - author
      - reason
      - timestamp
      - parent_version
  
  ChangeSet:
    description: Set of changes between versions
    slots:
      - added
      - modified
      - deleted
      - conflicts
      - drift_indicators
  
  ChangeType:
    enum:
      - CREATED
      - UPDATED
      - DELETED
      - DEPRECATED
      - MERGED
      - SPLIT
  
  DriftReport:
    description: Report of schema-instance drift
    slots:
      - schema_version
      - instance_data
      - drift_indicators
      - severity
      - recommendations
```

#### Key Components

1. **Version Store**: Immutable version history, branching, diff visualization
2. **Change Detection Engine**: Compare versions, detect drift, flag inconsistencies
3. **Drift Analysis**: Schema-instance alignment, process drift, trend analysis

---

### Layer 5: Maintenance & Continuous Learning

**Purpose**: Feedback loops and continuous improvement, including process mining and drift detection.

#### LinkML Schema

```yaml
# spindle/schemas/feedback.yaml
classes:
  FeedbackSystem:
    description: System for collecting and applying feedback
    slots:
      - feedback_records
      - usage_tracking
      - improvement_suggestions
  
  Feedback:
    description: Individual feedback record
    slots:
      - feedback_id
      - artifact_id
      - feedback_type
      - user_id
      - usage_context
      - outcome
      - correction
      - timestamp
  
  UsageContext:
    description: Context of knowledge usage
    slots:
      - user_id
      - use_case
      - query
      - timestamp
  
  UsageOutcome:
    description: Outcome of knowledge usage
    slots:
      - success
      - quality_rating
      - issues_encountered
      - corrections_made
  
  ImprovementSuggestion:
    description: Suggested improvement to knowledge
    slots:
      - suggestion_id
      - artifact_id
      - suggestion_type
      - rationale
      - confidence
      - priority

# spindle/schemas/process_mining.yaml
classes:
  ProcessMiningResult:
    description: Result of process mining from event logs
    slots:
      - mined_process_model
      - event_log_source
      - mining_algorithm
      - confidence_metrics
      - discovered_patterns
  
  ProcessDrift:
    description: Detected drift between documented and actual processes
    slots:
      - process_id
      - documented_version
      - actual_execution_pattern
      - drift_type
      - severity
      - affected_steps
      - recommendations
  
  ExecutionTrace:
    description: Concrete execution trace of a process
    slots:
      - execution_id
      - process_definition_id
      - actual_steps
      - step_durations
      - deviations
      - outcomes
      - timestamp
```

#### Key Components

1. **Usage Analytics**: Track usage patterns, identify unused knowledge
2. **Automated Refinement**: Learn from feedback, update confidence scores
3. **Process Mining Engine**: Extract process models from event logs, compare documented vs. actual
4. **Process Drift Detection**: Identify when actual executions deviate from documented processes
5. **Execution Trace Analysis**: Analyze concrete process executions to identify patterns and bottlenecks
6. **Hybrid Process Knowledge**: Combine text-based extraction with event log analysis

---

### Layer 6: Unified Knowledge Representation

**Purpose**: Unified semantic model for processes, entities, and facts, inspired by PKO (Procedural Knowledge Ontology).

#### LinkML Schema

```yaml
# spindle/schemas/unified_knowledge.yaml
classes:
  UnifiedKnowledgeModel:
    description: Unified representation of all knowledge types
    slots:
      - entities
      - processes
      - facts
      - relationships
  
  ProcessEntity:
    description: Process as first-class entity (PKO-inspired)
    is_a: Entity
    slots:
      - process_id
      - process_type
      - abstraction_level        # SCHEMA (procedure) or INSTANCE (execution)
      - steps
      - actors
      - resources
      - outcomes
      - provenance              # PROV-O integration
  
  ProcessProcedure:
    description: Abstract specification of a process (PKO procedure concept)
    is_a: ProcessEntity
    slots:
      - procedure_id
      - intended_steps
      - required_resources
      - success_criteria
      - regulatory_constraints
      - version
  
  ProcessExecution:
    description: Concrete execution instance (PKO execution concept)
    is_a: ProcessEntity
    slots:
      - execution_id
      - procedure_reference     # Links to ProcessProcedure
      - actual_steps
      - start_time
      - end_time
      - outcomes
      - deviations
      - actor_performance
      - resource_usage
  
  ProcessStep:
    description: Step in a process (PKO step concept)
    is_a: Entity
    slots:
      - step_id
      - step_type               # ACTIVITY, DECISION, EVENT, etc.
      - semantic_relations     # requires, produces, governed_by, enables, validates
      - dependencies
      - actors
      - resources
      - conditions
      - exceptions             # Tacit knowledge: exception paths
      - contextual_variants   # Context-dependent variations
  
  ProcessEntityLink:
    description: Links processes to entities in knowledge graph
    slots:
      - process_id
      - entity_id
      - relationship_type       # participates_in, produces, consumes, etc.
      - role                   # actor, resource, input, output
  
  UnifiedExtractionResult:
    description: Result of unified extraction
    slots:
      - entities
      - processes
      - relationships
      - process_entity_links
      - extraction_metadata
```

#### PKO Integration Principles

The unified model aligns with PKO (Procedural Knowledge Ontology) principles:

1. **Procedure vs. Execution Distinction**: Clear separation between abstract procedures and concrete executions
2. **PROV-O Provenance**: Full provenance tracking using PROV-O standards
3. **Semantic Relationships**: Rich relationship types beyond simple dependencies
4. **Tacit Knowledge Support**: Captures exceptions, variations, and contextual factors
5. **Multi-Level Abstraction**: Supports hierarchical process modeling
6. **Process-Entity Integration**: Processes are first-class entities in the knowledge graph

#### Key Components

1. **Unified Extractor**: Extract triples and processes together, link them semantically
2. **Process-Aware Query Engine**: SPARQL-like queries across unified model
3. **Process-Entity Linker**: Automatically links process steps to entities in knowledge graph
4. **Hierarchical Process Modeler**: Supports nested processes and multi-level abstraction
5. **PKO Converter**: Converts BAML ProcessGraph to PKO-aligned LinkML structures

---

## Implementation Architecture

### New Core Modules

```
spindle/
├── templates/                    # NEW: Extraction templates
│   ├── __init__.py
│   ├── registry.py              # Template registry and resolution
│   ├── default_knowledge_graph.yaml
│   ├── process_graph.yaml
│   ├── workflow_graph.yaml      # Future: Combined process + knowledge
│   └── custom/                   # User-defined templates
│
├── schemas/                      # LinkML schema definitions (governance)
│   ├── knowledge_lifecycle.yaml
│   ├── governance.yaml
│   ├── versioning.yaml
│   ├── feedback.yaml
│   ├── process_mining.yaml      # Process mining and drift detection
│   ├── unified_knowledge.yaml
│   ├── process_knowledge.yaml   # PKO-inspired process schemas
│   ├── extraction_templates.yaml # Template schema definitions
│   └── knowledge_artifacts.yaml
│
├── schemas/generated/            # Auto-generated from LinkML
│   ├── __init__.py
│   ├── knowledge_lifecycle.py    # Python dataclasses
│   ├── governance.py
│   └── ...
│
├── baml_src/                     # BAML schema definitions (extraction)
│   ├── spindle.baml              # Triple extraction
│   ├── process.baml             # Process extraction
│   ├── entity_resolution.baml
│   └── pipeline.baml
│
├── baml_client/                  # Auto-generated BAML client (do not edit)
│   ├── types.py                  # BAML extraction types
│   └── ...
│
├── extraction/                   # ENHANCED: Template-driven extraction
│   ├── extractor.py              # Template-aware extraction orchestrator
│   ├── bridge.py                 # NEW: BAML → LinkML conversion
│   ├── process_bridge.py         # NEW: BAML ProcessGraph → LinkML ProcessProcedure
│   ├── graph_handlers/           # NEW: Graph type-specific handlers
│   │   ├── knowledge_graph.py    # Knowledge graph extraction handler
│   │   ├── process_graph.py     # Process graph extraction handler
│   │   └── base.py               # Base handler interface
│   ├── multi_perspective.py
│   ├── quality_gates.py         # Template-aware quality gates
│   ├── conflict_detection.py
│   └── tacit_knowledge.py        # Enhanced for process knowledge
│
├── pipeline/                     # ENHANCED: Template-aware pipeline stages
│   ├── base.py                   # BasePipelineStage (template-aware)
│   ├── vocabulary.py            # Template-influenced vocabulary extraction
│   ├── metadata.py
│   ├── taxonomy.py              # Template-influenced taxonomy
│   ├── thesaurus.py
│   ├── ontology_stage.py        # Template-influenced ontology generation
│   ├── graph_extraction_stage.py # NEW: Template-driven unified extraction stage
│   ├── orchestrator.py          # Template-aware orchestrator
│   └── types.py
│
├── governance/                   # NEW: Governance layer
│   ├── validation/
│   │   ├── engine.py
│   │   ├── workflows.py
│   │   ├── rules.py
│   │   └── consensus.py
│   ├── quality/
│   │   ├── metrics.py
│   │   ├── assessment.py
│   │   └── dashboard.py
│   └── policies.py
│
├── versioning/                   # NEW: Version control
│   ├── version_store.py
│   ├── change_detection.py
│   ├── drift_analysis.py
│   └── merge.py
│
├── feedback/                     # NEW: Continuous learning
│   ├── usage_tracking.py
│   ├── feedback_loop.py
│   ├── refinement.py
│   └── process_mining/           # NEW: Process mining capabilities
│       ├── event_log_analyzer.py
│       ├── drift_detector.py
│       ├── execution_trace.py
│       └── hybrid_extraction.py   # Combine text + event logs
│
├── knowledge_model/              # NEW: Unified knowledge model
│   ├── lifecycle.py
│   ├── abstraction.py
│   ├── unified_extractor.py
│   ├── process_entity_linker.py  # NEW: Link processes to entities
│   ├── hierarchical_modeler.py   # NEW: Multi-level process abstraction
│   └── query_engine.py           # Enhanced: Process-aware queries
│
└── registry/                     # NEW: Knowledge registry
    ├── catalog.py
    ├── schema_store.py
    └── instance_store.py
```

### Build Process

#### LinkML Code Generation

```makefile
# Makefile for LinkML code generation
.PHONY: generate-linkml-schemas
generate-linkml-schemas:
	linkml-gen python spindle/schemas/knowledge_lifecycle.yaml -o spindle/schemas/generated/knowledge_lifecycle.py
	linkml-gen python spindle/schemas/governance.yaml -o spindle/schemas/generated/governance.py
	linkml-gen python spindle/schemas/versioning.yaml -o spindle/schemas/generated/versioning.py
	# ... generate all LinkML schemas
	
	linkml-gen json-schema spindle/schemas/*.yaml -d spindle/schemas/json-schema/
	linkml-gen rdf spindle/schemas/*.yaml -d spindle/schemas/rdf/
	linkml-gen json-ld-context spindle/schemas/*.yaml -d spindle/schemas/json-ld/
```

#### BAML Code Generation

BAML client generation is handled automatically by the BAML toolchain:

```bash
# BAML automatically generates Python clients from .baml files
# No manual build step required - happens on import or via baml-cli
```

#### Combined Build

```makefile
.PHONY: generate-all-schemas
generate-all-schemas: generate-linkml-schemas
	@echo "LinkML schemas generated"
	@echo "BAML clients auto-generated on import"
```

---

## Migration Path

The migration path is organized into incremental phases that build upon each other, with process knowledge management enhancements integrated throughout.

### Phase 1: Foundation (3-6 months)

**Core Infrastructure**

1. **Template System Foundation**
   - Design template schema (YAML/JSON structure)
   - Implement template registry (`templates/registry.py`)
   - Create default knowledge graph template
   - Create process graph template
   - Template validation and error handling

2. **Template-Aware Pipeline**
   - Update `BasePipelineStage` to accept template context
   - Modify all pipeline stages to use template instructions
   - Implement template context propagation through stages
   - Update orchestrator to pass template to stages

3. **Unified Graph Extraction Stage**
   - Replace separate `KnowledgeGraphStage` with unified `GraphExtractionStage`
   - Implement graph type handlers (knowledge graph, process graph)
   - Template-driven BAML function selection
   - Template-driven output format handling

4. **LinkML Schema Definition**
   - Define governance schemas in LinkML YAML
   - Define template schema in LinkML (`extraction_templates.yaml`)
   - Set up LinkML code generation pipeline
   - Generate Python dataclasses, JSON-Schema, RDF
   - **Note**: BAML schemas remain unchanged

5. **BAML-LinkML Bridge**
   - Implement conversion bridge (`extraction/bridge.py`)
   - Convert BAML `ExtractionResult` to LinkML `KnowledgeArtifact`
   - Convert BAML `ProcessGraph` to LinkML `ProcessProcedure`
   - Preserve extraction metadata during conversion

6. **Knowledge Registry**
   - Implement Knowledge Registry with lifecycle states
   - Separate schema and instance stores
   - Basic version control

7. **Basic Validation**
   - Simple validation workflows
   - Quality metrics calculation (template-aware)
   - Basic governance policies

**Process Knowledge Foundation (Phase 1.5)**

8. **Process Template Enhancement**
   - Enhance process graph template with semantic relationship types
   - Add `requires`, `produces`, `governed_by`, `enables`, `validates` to template
   - Improve evidence span capture with confidence scores in template

9. **Basic Process-Entity Linking**
   - Link process steps to entities in knowledge graph
   - Identify actors, resources, inputs, outputs as entities
   - Store process-entity relationships

### Phase 2: Quality & Governance (3-6 months)

**Quality and Multi-Perspective**

8. **Quality Assessment**
   - Implement quality assessment engine
   - Multi-perspective extraction
   - Conflict detection system

9. **Advanced Validation**
   - Expert review workflows
   - Consensus mechanisms
   - Governance dashboard

10. **Change Management**
    - Full version control system
    - Change detection and drift analysis
    - Branch merging capabilities

**Process Knowledge Quality (Phase 2.5)**

11. **Process Quality Metrics**
    - Completeness checks for process extraction
    - Dependency validation (no circular dependencies)
    - Step coverage analysis
    - Actor expertise tracking

12. **Tacit Knowledge Capture**
    - Enhance extraction prompts to capture exceptions and variations
    - Extract contextual factors that influence execution
    - Capture judgment calls and decision points
    - Identify success and failure indicators

13. **Process Version Control**
    - Version process definitions
    - Track process changes over time
    - Compare process versions
    - Detect process drift from documentation

### Phase 3: Advanced Features (6-12 months)

**Continuous Learning and Process Mining**

14. **Continuous Learning**
    - Feedback collection system
    - Usage analytics
    - Automated refinement

15. **Process Mining Foundation**
    - Event log ingestion and parsing
    - Basic process discovery from event logs
    - Execution trace storage

16. **Process Drift Detection**
    - Compare documented processes with execution traces
    - Identify deviations and variations
    - Flag process drift patterns
    - Generate drift reports

**Unified Model and Advanced Process Features**

17. **Unified Model**
    - Unified extractor (triples + processes together)
    - Process-aware query engine
    - Cross-domain queries

18. **PKO-Aligned Process Modeling**
    - Implement procedure vs. execution distinction
    - PROV-O provenance integration
    - Hierarchical process modeling (nested processes)
    - Multi-level abstraction support

19. **Process Mining Advanced**
    - Hybrid extraction (text + event logs)
    - Bottleneck identification
    - Process optimization recommendations
    - Execution pattern analysis

20. **AI Workflow Integration**
    - Process-aware AI agents
    - RAG enhancement with process knowledge
    - Process-compliant recommendation generation
    - Query interface: "How to do X?"

### Phase 4: Process Knowledge Excellence (6-12 months)

**Advanced Process Knowledge Management**

21. **Multi-Perspective Process Extraction**
    - Extract processes from multiple sources
    - Merge conflicting process views
    - Flag areas of agreement/disagreement
    - Consensus mechanisms for process knowledge

22. **Contextual Process Variants**
    - Named graph strategies for multiple perspectives
    - Context-dependent process variations
    - Organizational context support
    - Resource availability variants

23. **Process Knowledge Evaluation**
    - Comprehensive quality metrics
    - Accuracy validation against expert models
    - Semantic quality assessment
    - Process pattern recognition

24. **Process Knowledge Governance**
    - Process-specific validation workflows
    - Domain expert review for processes
    - Process compliance checking
    - Regulatory constraint validation

### Incremental Adoption Strategy

**Quick Wins (Can start immediately)**:
- Enhance `ProcessDependency` relationship types
- Improve evidence span capture
- Add confidence scoring to process extraction

**Medium-Term (3-6 months)**:
- Process-entity linking
- Process version control
- Basic process mining

**Long-Term (6-12 months)**:
- PKO-aligned modeling
- Hierarchical processes
- Advanced process mining
- AI workflow integration

---

## Benefits

### 1. Extraction Improvements

- **Template-driven architecture**: Unified pipeline for all graph types, easy to extend
- **Semantic foundation**: All extractions benefit from vocabulary/ontology building
- **Multi-perspective support**: Capture conflicting views, preserve attribution
- **Quality gates**: Confidence scoring, completeness checks at every stage (template-specific)
- **Tacit knowledge**: Capture exceptions, variations, judgment calls
- **Conflict detection**: Automatic detection of contradictions and updates
- **Flexibility**: Domain-specific templates without code changes

### 2. Maintenance Capabilities

- **Version control**: Git-like versioning for all knowledge artifacts
- **Change detection**: Automatic detection of changes and drift
- **Continuous learning**: Feedback loops drive improvement
- **Process mining**: Compare documented vs. actual processes

### 3. Governance Features

- **Validation workflows**: Built-in expert review and approval
- **Quality metrics**: Comprehensive quality assessment
- **Conflict resolution**: Multi-stakeholder consensus mechanisms
- **Governance policies**: Domain-specific rules and thresholds

### 4. Unified Representation

- **Template-driven extraction**: Single pipeline architecture for all graph types
- **Single semantic model**: Processes, entities, and facts together
- **Cross-domain queries**: Query across all knowledge types
- **Process-aware**: Processes are first-class entities
- **PKO-aligned**: Procedure vs. execution distinction, PROV-O provenance
- **Hierarchical modeling**: Multi-level process abstraction
- **Process-entity integration**: Processes linked to entities in knowledge graph
- **Consistent foundation**: All graph types share vocabulary/ontology building stages

### 5. BAML-LinkML Integration Benefits

- **Optimal tool selection**: BAML for LLM extraction, LinkML for governance
- **Separation of concerns**: Clear boundaries between extraction and governance
- **No schema conflicts**: Each tool handles its domain without overlap
- **Type safety throughout**: Both BAML and LinkML provide type-safe Python classes
- **Independent evolution**: Extraction and governance schemas can evolve separately
- **Multi-format support**: LinkML provides JSON-Schema, RDF, Python, GraphQL, SQL
- **Semantic web ready**: LinkML automatic JSON-LD and RDF generation
- **Developer friendly**: Work with YAML/JSON, get semantic web for free

### 6. Enterprise-Ready

- **Scalable**: Designed for large-scale, multi-stakeholder knowledge management
- **Auditable**: Complete version history and change tracking
- **Governable**: Built-in governance and validation
- **Maintainable**: Continuous learning and automated refinement

### 7. Process Knowledge Management Excellence

- **PKO-inspired**: Aligned with Procedural Knowledge Ontology principles
- **Tacit knowledge capture**: Exceptions, variations, judgment calls
- **Process mining**: Compare documented vs. actual processes
- **Process drift detection**: Identify when processes evolve
- **Multi-level abstraction**: Strategic, tactical, and operational views
- **Contextual sensitivity**: Process variants based on context
- **Process-entity integration**: Unified queries across processes and entities
- **AI workflow ready**: Process knowledge available for AI agents and RAG systems

---

## Conclusion

This redesign transforms Spindle from an extraction tool into a comprehensive **Knowledge Governance Platform**. The integration of LinkML provides a solid foundation for schema-driven development, ensuring consistency, type safety, and interoperability across all components.

The six-layer architecture addresses the full lifecycle of enterprise knowledge management:
- **Foundation**: Lifecycle states and abstraction levels
- **Extraction**: Template-driven, multi-perspective, quality-gated extraction with process knowledge specialization
- **Governance**: Validation, quality metrics, expert review
- **Versioning**: Change tracking and drift detection (including process drift)
- **Learning**: Continuous improvement from feedback and process mining
- **Unification**: Single semantic model for all knowledge (processes, entities, facts)

The template-driven extraction system provides:
- **Unified architecture**: Single pipeline for knowledge graphs, process graphs, and future graph types
- **Semantic consistency**: All extractions benefit from shared vocabulary/ontology building
- **Easy extensibility**: New graph types added via templates without code changes
- **Domain customization**: Templates encode domain-specific extraction strategies
- **Default simplicity**: Default templates for common use cases

Process knowledge management is integrated throughout all layers, with specialized capabilities for:
- Capturing both explicit and tacit process knowledge
- Distinguishing procedures (abstract) from executions (concrete)
- Linking processes to entities in the knowledge graph
- Detecting process drift and variations
- Supporting multi-level abstraction and contextual variants

By adopting LinkML for governance and maintaining BAML for extraction, Spindle gains:
- **Optimal LLM interaction**: BAML's specialized prompt engineering capabilities
- **Governance excellence**: LinkML's semantic modeling and multi-format support
- **Type safety**: Both tools provide type-safe Python classes
- **Semantic web interoperability**: LinkML's automatic RDF/OWL generation
- **Developer-friendly interfaces**: YAML/JSON for both tools
- **Clear separation**: Extraction schemas (BAML) vs. governance schemas (LinkML)
- **Independent evolution**: Each layer can evolve based on its specific needs

This architecture positions Spindle as the foundation for trustworthy, maintainable, and governable enterprise knowledge systems.

---

## Process Knowledge Management Integration

This redesign incorporates principles and recommendations from process knowledge management theory and practice, particularly:

- **PKO (Procedural Knowledge Ontology)**: Distinction between procedures and executions, PROV-O integration
- **Tacit Knowledge Capture**: Exceptions, variations, judgment calls, contextual factors
- **Multi-Level Abstraction**: Hierarchical process modeling for different stakeholder needs
- **Process Mining**: Comparison of documented vs. actual processes from event logs
- **Process-Entity Integration**: Unified semantic model linking processes to knowledge graph entities
- **Contextual Sensitivity**: Process variants based on organizational context and resource availability

Process knowledge management is integrated through the template system, where process graph templates specify:
- Process-focused vocabulary extraction (action verbs, temporal terms)
- Process-specific ontology building (ProcessStep, Actor, Resource entity types)
- Process extraction constraints (no circular dependencies, valid start/end steps)
- Process quality gates (completeness, dependency coverage)

For detailed recommendations and implementation guidance, see [Process Knowledge Management Consolidated Overview](./PROCESS_KNOWLEDGE_MANAGEMENT_CONSOLIDATED.md).

---

## References

- [LinkML Documentation](https://linkml.io/)
- [Process Knowledge Management Consolidated Overview](./PROCESS_KNOWLEDGE_MANAGEMENT_CONSOLIDATED.md)
- [Procedural Knowledge Ontology (PKO)](https://w3id.org/pko)
- [PROV-O: Provenance Ontology](https://www.w3.org/TR/prov-o/)
- [BAML Documentation](https://baml.dev/)

