
# Spindle Workflow
## Config
1. Project configuration
2. (optional) Select upper ontology from a set of options
3. (optional) Select middle ontology based on use case (i.e., process mining, static knowledge graph, event-centric knowledge graph, causal graph)

## Corpus: Doc Ingestion & Preprocessing
### Ingestion & Version control
1. Mirror corpus files (capable of accepting an external source of original documents)
2. Check for diffs (using deepdiff; https://github.com/seperman/deepdiff) in mirror (for initial run, diffs = entire documents)
3. For all diffs, extraction of content into JSON format (using Docling: https://github.com/docling-project/docling), save versioned JSON to doc storage
4. Updating of Document objects with metadata (title, author, date, source, etc.) in SQL db.

### Preprocessing
1. Semantic chunking: we do not save the chunks in separate files at this point. Rather, we edit the JSON by adding chunk fields at the top level. 
2. Coreference resolution: we do a fast pass of coref resolution, adding a tag next to each reference of an entity (containing the resolved entity's name)
3. NER (vocabulary/taxonomy guidance optional): fast pass of inferring the entity type for all entities. This adds another tag to the text, containing the entity type. 
4. (optional) vocabulary validation/recommendation: tag entities according to whether or not they match an existing entity in a vocabulary. For the case where they don't, the LLM is used to recommend a new entity name and type.
5. (optional) taxonomy validation: tag entities according to whether or not they match an existing entity type in a taxonomy

## KOS Development

NOTES:
* For v1, we'll do all of this in-memory and save artifacts in a dedicated file storage. In the future, we might allow KOS and KG development to interact directly with a REST API wrapping a dedicated triple store. 
* For the different kinds of graphs mentioned in the config step, we'll probably need some templated prompts aside from the upper/middle ontology info. This is likely true of both the KOS and KG development pipelines. 

PROCESS:
1. Load vocabulary into memory (or initialize a new RDF-star instance; NOTE: we are using RDF-star because we intend on tracking provenance)
    1. Cycle through corpus and extract vocabulary (using a python script that extracts the tags from the preprocessing step). Provenance is maintained in the vocabulary artifact by referencing a provenance object ID (the provenance object is specific to that node or edge). The provenance object, saved elsewhere, contains the detailed provenance information for that node or edge.
    2. After completing extraction, do entity resolution over the full vocabulary. 
    3. User review and validation of vocabulary.
    4. Save RDF-star vocab as TTL in KOS artifact storage.
2. Use LLM to create a first draft of a taxonomy based on the vocabulary. 
    1. User review and validation of taxonomy. 
3. Cycle through corpus and use LLM to extract ontology. 
    1. User review and validation of ontology.
    2. Save as OWL in KOS artifact storage.
    2. (optional) Create SHACL validation (see [here](https://share.google/aimode/hS2pRdk7y6KS6n7Tb) for some thoughts on best practices for triple validation with OWL and SHACL)

## Knowledge Graph Development



# Spindle Modules



## Entity Resolution

We'll have multiple levels of entity resolution. 
* Fast pass: fuzzy search over vocabulary/taxonomy.
* Medium pass: semantic search over vocabulary/taxonomy.
* Slow pass: semantic search over blocked entities from knowledge graph.

## Embedding Service

* document chunks
* vocabulary entities
* knowledge graph edges and nodes
    * For nodes, embeddings are computed by aggregating information up from the immediate neighbors. 
        1. Start from leaf nodes (out-degree = 0). No aggregation happens here.
        2. Cycle through nodes ranked by out-degree, aggregating information (i.e., type:name:description) up from the outgoing edges (and the nodes connected to them)
        3. Use a small, fast LLM to compress the aggregated information. 
        4. Compute the embedding from the compressed information.
    * For edges, no aggregation happens. The embedding is computed from the edge's type:name:description.


# NOTES
* re: KNOME (specifically, Hypatia): Ideally, we can implement a true triple-store solution (e.g., StarDog or [GraphDB by ontotext](https://share.google/aimode/W4BIDgFjVYFpiF2cz), with the latter having a free self-hosted option) in the future. But for now, the approach is to store the artifacts in GCS (or equivalent) and pull them into memory with [Oxigraph](https://github.com/oxigraph/oxigraph) (for vocab + taxonomy). Either way, we can have an API. For example, [it is possible](https://share.google/aimode/tij0B4P0yOrIzzO1i) to serve the Oxigraph implementation as a CRUD REST API behind FastAPI and still have concurrent asynch Update requests.
* I want to make the package very user-friendly. There should be pipeline classes that can be imported separately (e.g., Corpus, KOS, KnowledgeGraph)
    * Each class should store a client for loading and saving the artifact(s) related to the class. 
    * The client should allow interacting with the current state (e.g., `Corpus.client().metadata.read()`)
    * The client should have a set of pipeline steps for building/modifying the artifact(s) (e.g., `Corpus.client().pipe_steps.get_diffs()`)

