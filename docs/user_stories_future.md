I want to start over and rebuild the codebase.
  * simplify and reduce interdependencies as much as possible
  * everything should be built to be accessible to an agent that will help users build and maintain their knowledge systems. Each component will be accessible via an agentic Skill (in the Anthropic sense) that the agent can use.
  * each separate "project" is a Knowledge Organization System (KOS).

# Core functionality

* Capable of building a full KOS from a corpus of documents
  * For our purposes, the KOS comprises the following components:
    1. **Controlled Vocabulary** - Extract and define key terms
    2. **Metadata Standards** - Define metadata schema
    3. **Taxonomy** - Build hierarchical structure
    4. **Thesaurus** - Add semantic relationships (BT, NT, RT)
    5. **Ontology** - Generate entity and relation types
  * Each KOS artifact is extracted using a different BAML function. 
  * Each step of the KOS needs to identify sources, whether manually defined by user or extracted from a document. If the latter, then specific spans within the document(s) are cited.
  * The ontology is the backbone of the KOS. It is used to extract different types of graphs from a corpus of documents. It embodies the semantics and logic of the KOS.
  * Within a domain, a corpus may contain canonical document types that have standardized formats. Spindle should have a system for incorporating these into the KOS as metadata so they can be treated differently (e.g., custom instructions) during graph extraction.
  * The KOS extraction process is guided and orchestrated by a KOS YAML config file. Users can use pre-defined templates (for common KOS use cases), create their own template(s), or create their own single-purpose KOS config file from scratch. The KOS YAML has the following sections:
    * owner (name, email, organization)
    * project (name, description, path to project)
    * corpus (name, description, path to corpus)
    * general prompts:
      * project description: description of the scope, domain, purpose, etc. of the project
      * cold start: entities, relations, attributes, etc. that the user already knows and wants to ensure are included in the KOS
    * vocabulary
    * metadata
    * taxonomy
    * thesaurus
    * ontology
  * A CLI is provided to help users create a new KOS config. It will walk them through the key decisions they need to make.
* Capable of extracting different types of graphs from a corpus of documents, adhering to an ontology (default template if not provided).
  * Supports the following graph types:
    * knowledge graph
    * temporal knowledge graph
    * process dag
    * reasoning/decision map (for agentic workflows)
  * Each graph type is extracted using a different BAML function.
  * If canonical corpus document types are identified, then the graph extraction process should be guided by custom instructions for each document type (with fallback to general instructions).
  * The graph extraction process is guided by a graph extraction YAML config file. Users can use pre-defined templates (for common graph extraction use cases), create their own template(s), or create their own single-purpose graph extraction config file from scratch. The graph extraction YAML has the following sections:
    * graph type: the type of graph to extract
    * graph name: the name of the graph
    * graph description: a description of the graph
    * ontology path
    * kos path
* Given a diff in a document or a new document, capable of updating the KOS and/or graphs when necessary


# Other (optional)functionality
* Document preprocessing (potentially as a separate module)
  * if the corpus has a hierarchy of documents, then the KOS should be able to extract the hierarchy and create a graph of the hierarchy.
  * if the corpus has some documents that are more reliable/informative than others, these should be consumed first to establish the foundation of the KOS. 
  * After an initial KOS is established, document preprocessing should include an entity recognition step where  entities are "tagged" with the entity class. If the KG is already established, entities should be linked to any existing entities in the KG. This provides a more reliable foundation for the triple extraction step (since entity recognition step has already been partially completed).
  * Coreference resolution must precede extraction
* Fine-tuning flywheel: tools for easily fine-tuning an open-source LLM to improve triple extraction. Maybe even fine-tuning on a specific domain if the economics make sense.
