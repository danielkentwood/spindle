* For the prompt(s) to extend the KOS, we need to have a section where we define the domain/topic and the primary use cases. How will people use the KOS? This is what the LLM needs to know in order to capture meaningful new entities and relations.
* templates: 
    * these are just config files pre-populated for specific use cases. 
    * templates define the following: 
        * where data artifacts will be stored
        * graph type (temporal knowledge graph, process graph, causal graph, etc) 
        * KOS stages (vocabulary, taxonomy, thesaurus, ontology)
        * use cases for the KOS. For example:
            * logistics and supply chain management for company X
            * Rx research and development for company Y
            * healthcare policy and regulations for government Z
            