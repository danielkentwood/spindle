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
  * The ontology is the backbone of the KOS. It is used to extract different types of graphs from a corpus of documents. It embodies the semantics and logic of the KOS
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
* Capable of extracting different types of graphs from a corpus of documents, adhering to an ontology.
  * Supports the following graph types:
    * knowledge graph
    * temporal knowledge graph (similar to grafiti)
    * process dag
    * reasoning/decision map (for agentic workflows)
  * Each graph type is extracted using a different BAML function.
* Given a diff in a document or a new document, capable of updating the KOS and/or graphs when necessary
