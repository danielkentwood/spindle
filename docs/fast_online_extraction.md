
# Fast online extraction

## 3 stage process
1. Fast streaming entity extraction and relation extraction.
    - Current view of the available tools is that GLiNER2 and spaCy are the best options for RE and NER. GLiNER2 is a more robust, accurate, unified (single forward pass for RE/NER) and flexible solution, but spaCy is much faster. It's worth exploring the tradeoffs.
    - Outputs:
        - final: a version of the document with the entities labeled (this will be crucial for training later)
        - streaming: JSON containing extracted entities and relations
    - UI:
        - preliminary results are displayed in 2 ways:
            - the text in the conversation shows the extracted entities as pills — when you hover over them, you see the “resolved” entity
            - you see a “preliminary” graph in the graph viewer. Just nodes and edges with simple lables; no node types or extended functionality.
    - For reference, here are some other options to keep track of.
        - entity extraction: Gliner2, Flair
        - Document & Web specialized extractors: Apache Tika (toolkit for extracting metadata and text from over 1K filetypes) and Firecrawl (ideal for web, converts to .md or .json and isolates key entities)
        - Relation extraction: OpenNRE, Relik, DeepDive
        - General Purpose NLP (both entity/relation extraction): spaCy, Stanford CoreNLP (Stanza is much slower than spaCy but has higher precision on academic benchmarks), HuggingFace Transformers
        - LLM based (could be paired with small, specialized models for fast entity/relation extraction): LangExtract, unstract
2. Slower, more accurate NER and RE
    - Using LLMs
    - Uses the results of the first stage (preprocessed text + NER + RE)
    - ontology-constrained triple extraction with provenance (keeps precise span indices containing the triple)
        - After first forward pass, there is a entity resolution step to ensure comportment with current entities. Solutions:
            - this can be a slower process if necessary. Accuracy is more important than speed for this step.
        - Provenance is a hard problem to solve fully. Spindle's current solution is pretty good. The LLM extracts, along with the triple, the span containing the original text. Then a post-processing step attempts to match the span in the text. Would be good to come up with a way of quantifying how accurate the current implementation is (a simple, scalable approach would be to just calculate exact match precision).
        - Other solutions to look at:
            - **Top Open-Source Frameworks for Provenance**
                - **ContextGem**: A framework designed specifically to eliminate boilerplate in LLM data extraction. It features **precise reference mapping** to source content and built-in justifications for every extraction, ensuring each triple is tied to its originating text.
                - **LLMGraphTransformer (via LangChain)**: This tool allows for the extraction of nodes and edges by prompting the LLM to process documents. It can output a **GraphDocument** that includes metadata useful for verifying extraction accuracy against the original source.
                - **Instructor** and **Outlines**: These libraries specialize in **structured output** using Pydantic schemas. To capture provenance, you can define a schema that explicitly requires a "source_span" or "justification" field alongside each triple.
            - **Hybrid and Specialized Methods**
                - **Model Collaboration (LLM + Small Model)**: This approach uses a smaller, high-precision evaluation model to identify potential entity pairs and spans first. These identified spans are then embedded into the LLM's prompt to guide the triple extraction, significantly improving accuracy in complex sentences.
                - **REBEL (Relation Extraction By End-to-end Language generation)**: An open-source seq2seq model that extracts entities and relationships simultaneously. It is often used as a baseline for high-precision triple extraction because it is trained specifically for this task rather than being a general-purpose model.
                - **ProvSEEK**: An agentic framework that uses **verification-first design**. It generates not just an answer but also supporting evidence (system provenance) to minimize hallucinations and ensure every claim is grounded in verifiable data.
            - **Proven Strategies for Accurate Attribution**
                - **Chain-of-Thought (CoT) Prompting**: Encourages the LLM to articulate its reasoning steps before providing the final triple, which often includes quoting the relevant text span.
                - **PROV-DM Alignment**: Prompting the LLM to adapt its output to the PROV-DM (Provenance Data Model) standard. This forces the model to categorize entities, agents, and activities, creating a more traceable extraction path.
                - **Evaluation-Filtering Frameworks**: Implementing a pipeline where extracted triples are filtered by a second model to check for "claim-evidence correlation," ensuring the triple was actually derived from the provided source sentence.
    - Outputs:
        - Finalized document with labeled entities
        - JSON (or other structured format) containing extracted knowledge graph
    - UI:
        - During the time between the first pass and second pass, have some kind of communication with the user about what is happening.
        - Finalization of text labeling (using resolved entities from extracted triples)
        - Finalization of graph visualization (containing full interactive functionality and  node type differentiation)
3. Offline training of spaCy model with labeled data
    - This ensures our extraction process is self-improving. The fast step will continue to improve as we fine-tune a spacy model for the project/domain.
    - this step is optional; not all projects will want to devote the resources for fine-tuning.


