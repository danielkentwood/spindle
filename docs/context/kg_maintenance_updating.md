# LLM-driven knowledge graph updates from evolving corpora

**The most effective approach to updating knowledge graphs with LLMs combines semantic document chunking, schema-guided multi-pass extraction, and bitemporal soft-delete versioning — a pipeline that recent research shows can achieve 89–99% precision at scale.** This matters because knowledge graphs are increasingly the backbone of enterprise AI systems and RAG architectures, yet most existing pipelines assume one-time construction rather than continuous evolution. The field has advanced rapidly since 2023, with frameworks like Microsoft GraphRAG, LightRAG, and KGGen establishing practical baselines, while papers at ACL, EMNLP, and NeurIPS have introduced techniques for incremental updates, conflict resolution, and provenance tracking. This report synthesizes the state of the art across three critical dimensions: preprocessing pipelines, triple extraction prompting, and deprecated-information management.

---

## Document preprocessing determines extraction quality more than model choice

The preprocessing pipeline — how raw documents become LLM-ready inputs — has an outsized impact on knowledge graph quality. Research consistently shows that **chunking strategy, coreference resolution, and change detection** are the three highest-leverage preprocessing decisions.

**Chunking** has moved well beyond fixed-size token windows. Microsoft GraphRAG defaults to **1,200-token chunks** (called "TextUnits"), but the SC-LKM method (published in *Electronics*, 2025) demonstrates that hierarchical semantic chunking — respecting native document structure (sections, paragraphs) and refining boundaries using topic similarity and named-entity continuity — produces statistically significant improvements over fixed-size approaches. The SLIDE technique (arXiv:2503.17952, March 2025) introduces overlapping sliding windows that yielded **24% higher entity extraction and 39% higher relationship extraction** versus standard chunking in English, with even larger gains for low-resource languages. For incremental KG updates specifically, the "atomic fact" approach — decomposing text into self-contained factual statements rather than arbitrary text blocks — enables surgical updates to individual facts without re-processing entire documents.

Practical chunk size recommendations depend on corpus scale. For fewer than 100 documents, larger chunks (1,000–1,200 tokens) with GPT-4-class models work well. For 100–1,500 documents, 300–600 tokens with 10% overlap balances fidelity against cost. Beyond 1,500 documents, fine-tuning smaller models or using hybrid NLP/LLM approaches becomes cost-effective. The critical insight from Cognee's research is that chunking must be **invertible** — every character of input preserved in exactly one chunk — if you plan to reconstruct context from graph-based retrieval.

**Coreference resolution** is the most underappreciated preprocessing step. Without it, the same entity appears as multiple disconnected nodes ("John Smith," "Mr. Smith," "he"). The LINK-KG framework (arXiv:2510.26486) introduces a three-stage LLM-guided coreference pipeline with type-specific prompt caches that persist across chunks, resolving long-range references. The simpler CORE-KG approach (arXiv:2510.26512) performs type-wise sequential resolution — processing Person entities before Location before Organization — and reduces **node duplication by 33% and noise by 38%** compared to GraphRAG baselines.

**Change detection** for incremental updates follows three patterns: hash-based comparison of document/chunk digests (simplest), timestamp-based tracking of last-modified dates (most common), and delta-based snapshot comparison (most thorough). The IncRML framework (*Semantic Web Journal*, 2024) combines RML mapping rules with change detection functions, achieving up to **315× less storage and 4.4× faster KG construction** versus full reprocessing. CocoIndex implements a declarative Python framework where data lineage is tracked automatically, ensuring re-runs never duplicate nodes or edges.

**Entity recognition** approaches span a spectrum from fully LLM-based to hybrid. LangChain's LLMGraphTransformer and LlamaIndex's SchemaLLMPathExtractor represent the pure-LLM end, supporting GPT-4o, Gemini, Claude, and open-source models. REBEL (Babelscape, EMNLP 2021) provides a faster, cheaper seq2seq alternative handling 200+ relation types at 512-token limits. The EDC framework (Zhang & Soh, EMNLP 2024) offers perhaps the best balance: a three-stage Extract → Define → Canonicalize pipeline where open extraction is followed by semantic definition of types and then schema normalization via a trained retrieval component.

---

## What the best triple extraction prompts look like

Triple extraction prompt design has converged on a clear set of principles, though with some surprising findings about which techniques actually help.

**The optimal prompt structure** contains six components in order: (1) a system role establishing the LLM as an extraction specialist with strict rules ("You are an AI expert specialized in knowledge graph extraction. Only extract explicitly stated facts"), (2) a schema/ontology block defining allowed entity types, relation types, and their valid pairings, (3) one to three relevant few-shot examples, (4) clearly delimited input text, (5) strict output format specification (JSON with explicit keys), and (6) constraint rules (e.g., "predicates must be 1–3 words," "output null if unsure").

The finding on **few-shot examples** is striking. Polat et al. (*Semantic Web Journal*, 2024), testing on RED-FM with GPT-4, found that **a single example improves performance 2–3× over zero-shot**, with diminishing returns beyond that. The quality of examples matters more than quantity — examples retrieved via Maximal Marginal Relevance (balancing relevance to the input text with diversity) consistently outperform fixed canonical examples. Papaluca et al. (ACL 2024 KaLLM Workshop) found that even a "0.5-shot" approach — providing just related triples from an existing KB as context, without full input-output examples — significantly boosts extraction quality.

**Chain-of-thought reasoning** produced mixed results for triple extraction, which is an important nuance. Polat et al. found that CoT, self-consistency, and ReAct prompting **did not outperform simpler methods** for knowledge extraction — their interpretation is that triple extraction is fundamentally a "format conversion" task rather than a "reasoning" task. However, Nie et al. (VLDB 2024 LLM+KG Workshop) showed CoT helps when combined with ontology alignment, with their CoT+Ontology approach outperforming baselines in 8 of 10 domains. The reconciliation: **CoT is most valuable for validation and filtering of extracted triples** (catching incorrect relations, resolving ambiguous references), not for initial extraction itself. A practical CoT pattern from Yu (2025) has the model first identify entities, then draft candidate triples, then self-verify each triple before outputting final results.

**Schema guidance** is the single most impactful technique. Apple's ODKE+ system (SIGMOD), which processes millions of facts across 195 predicates in production, dynamically generates "ontology snippets" — curated schema fragments included in the prompt — that **reduced hallucinated extractions by 35% and improved factual precision from 91% to 98.8%**. TextMine (2024) found that ontology-guided prompts improve accuracy by up to **44.2%** and reduce hallucinations by **22.5%**. The OntoLogX system uses SHACL constraints for schema validation and feeds previously generated KGs as few-shot examples for new extractions. The practical lesson: always provide the LLM with explicit entity types, relation types, and domain/range constraints when they exist.

**Multi-pass extraction** — extracting entities first, then relations in a second call — consistently outperforms single-pass approaches. KGGen (arXiv:2502.09956) and iText2KG (arXiv:2409.03284) both confirm that this two-step approach produces better entity consistency because relations are grounded in already-identified entities. The SAC-KG framework (Chen et al., ACL 2024) extends this further with a Generator → Verifier → Pruner pipeline that achieves **89.3% precision at million-node scale**, a 20% improvement over prior state of the art.

For **entity normalization**, the best approaches operate as a post-processing step rather than within the extraction prompt. KGGen uses iterative LLM-based clustering to merge references like "New York City," "NYC," and "New York" into canonical forms. The EDC framework generates natural-language definitions for each entity and then uses vector similarity of these definitions to detect and merge duplicates. iText2KG applies cosine similarity thresholds (default 0.6) on entity embeddings, replacing hallucinated entities with the most similar real ones from the input.

A production-ready prompt template synthesized from these findings:

```
SYSTEM: You are a knowledge extraction specialist. Extract only 
explicitly stated facts. Output valid JSON only.

SCHEMA: Entity types: [list]. Relation types: [list with 
domain/range constraints]. 

EXAMPLE: [1 relevant retrieved example showing input→output]

RULES: Predicates ≤3 words. Use canonical entity names. If unsure, 
output null. No information not present in the text.

TEXT: ```{chunk}```

OUTPUT: [{"subject": "", "subject_type": "", "predicate": "", 
"object": "", "object_type": ""}]
```

---

## Deprecated information should be preserved, not deleted

The evidence strongly favors **retaining deprecated and contradictory information** in the knowledge graph, with structured metadata indicating validity status — a practice known as soft delete with tiered archival.

Polleres et al.'s landmark survey "How Does Knowledge Evolve in Open Knowledge Graphs?" (TGDK, 2023) identifies three complementary perspectives on KG evolution: **temporal KGs** (time as data, annotating when facts hold), **versioned KGs** (time as metadata, discrete graph snapshots), and **dynamic KGs** (continuous streams of insertions/deletions). These are not mutually exclusive, and the best systems combine all three.

**Bitemporal modeling** — tracking both *valid time* (when a fact was true in the real world) and *transaction time* (when it was recorded in the system) — is the gold standard. RDF-star has emerged as the preferred representation format, supported by GraphDB, Apache Jena, and Stardog, offering compact annotation without the verbosity of traditional reification. YAGO 4.5 and the OpenAI Temporal Agents Cookbook (July 2025) both adopt this approach. In RDF-star, a temporal annotation looks like:

```
<<:Obama :holdsOffice :USPresident>> :validFrom "2009-01-20" ;
                                      :validUntil "2017-01-20" .
```

For **versioning**, a hybrid delta-based approach with periodic full snapshots is optimal. Pure snapshot-based versioning (as DBpedia uses with quarterly releases) is storage-intensive and misses fine-grained changes. Pure delta-based versioning (as OSTRICH implements) is storage-efficient but requires replaying deltas to reconstruct any state. The hybrid approach, periodically materializing full snapshots while storing deltas between them, balances **storage efficiency against query performance**. The ConVer-G system (arXiv:2409.04499) advances this with concurrent versioned SPARQL queries across multiple graph versions.

**Conflict resolution** when new information contradicts existing triples follows a detect-then-resolve pattern. The CRDL framework (*Mathematics*, 2024) first classifies relations by cardinality (1-to-1 vs. many-to-many) and applies different resolution strategies: for 1-to-1 relations, it selects the triple with the lowest LLM perplexity as truth; for many-to-many relations, it uses perplexity thresholds combined with LLM evaluation. This approach achieved a **56% improvement in recall and 68% increase in F1** over prior methods. TeCre (*Information*, 2023) handles temporal conflicts specifically, using LSTM-based temporal KG embeddings to enforce constraints like time disjointness and precedence. TruthfulRAG (arXiv:2511.10375) introduces entropy-based filtering for RAG-integrated KGs: if a new reasoning path causes significant entropy increase relative to existing knowledge, it signals a factual conflict requiring resolution.

An underappreciated insight from AutoBioKG (2025) is that many apparent contradictions are actually **context-dependent truths**. The statement "Junctin activates RyR" and "Junctin inhibits RyR" are both correct under different conditions (luminal calcium concentration). Rather than choosing one, the system adds conditional attributes to triples, preserving both facts with their qualifying contexts. This models real-world knowledge more faithfully than forced binary resolution.

**Provenance tracking** should be implemented from day one using the W3C PROV standard, which provides three core classes — Entity, Activity, and Agent — with relationships like `wasGeneratedBy`, `wasDerivedFrom`, and `wasAttributedTo`. The PROV-STAR extension (Dibowski, FOIS 2024) enhances this with RDF-star for triple-level change tracking, introducing `TripleGenerationSet` and `TripleInvalidationSet` concepts that enable full history retrieval via SPARQL-star queries. At minimum, each triple should carry three confidence scores: **extraction confidence** (from the LLM/algorithm), **source confidence** (reliability of the document), and **temporal decay** (recency weighting).

Wikidata's production model validates this approach at scale. With **1.5 billion+ triples**, it maintains a complete edit history per entity, allows multiple claims per property with ranks (preferred/normal/deprecated), and attaches source references to every statement. The Wikidated 1.0 dataset reveals that entities see a mean of ~70 days between consecutive revisions, meaning most facts are periodically reviewed and updated rather than written once.

The practical architecture should implement **three tiers**: an active tier of current valid triples optimized for query performance; a deprecated tier of recently invalidated triples with full provenance, queryable but excluded from default results; and an archive tier of historical triples in cold storage for audit and compliance. The OpenAI Temporal Agents Cookbook recommends computing a relevance score as `recency × trust × query_frequency` and automatically migrating low-scoring facts to the archive tier.

---

## Key frameworks and tools for implementation

Several mature, open-source frameworks implement these patterns end-to-end:

- **Microsoft GraphRAG** (github.com/microsoft/graphrag): The most comprehensive pipeline for initial KG construction with community detection and hierarchical summarization, but requires full reindexing for updates — best for batch-oriented workflows
- **LightRAG** (github.com/HKUDS/LightRAG, EMNLP 2025): Supports true incremental insertion via `rag.ainsert()` with automatic deduplication, achieving **10× token reduction** versus GraphRAG
- **KGGen** (`pip install kg-gen`, arXiv:2502.09956): The simplest entry point — two-step extraction plus iterative LLM clustering for entity resolution, with the MINE benchmark for evaluation
- **iText2KG** (arXiv:2409.03284): Zero-shot incremental construction maintaining semantic uniqueness constraints across additions, with built-in hallucination handling
- **EDC** (github.com/clear-nus/edc, EMNLP 2024): Best for schema-intensive domains with its Extract-Define-Canonicalize pipeline handling large schemas through RAG-style schema retrieval
- **Neo4j LLM Graph Builder** (llm-graph-builder.neo4jlabs.com): Production-ready web application supporting multiple LLMs, with community detection and entity merging built in

For consistency checking, **SHACL** (W3C Recommendation) with incremental validation (supported by GraphDB and RDF4J) enables pre-validation of updates before they're applied — critical for preventing invalid states in production graphs. The recent work on SHACL Validation Under Graph Updates (ISWC 2025) allows verifying whether an update will break constraints *before* applying it.

---

## Conclusion

Three insights emerge from this synthesis that practitioners should internalize. First, **preprocessing is not overhead — it is the primary determinant of KG quality**. Semantic chunking with coreference resolution and incremental change detection can double extraction yields compared to naive fixed-size chunking. Second, **prompt simplicity beats prompt cleverness** for extraction: one good schema-guided example outperforms elaborate chain-of-thought reasoning for basic triple extraction, though CoT adds value in the validation and filtering stage. Third, the field has decisively settled the deprecation question: **always soft-delete, always track provenance, and allow multiple ranked claims per property** rather than forcing premature truth resolution. The most sophisticated systems recognize that contradictions often reflect context-dependent truths rather than errors, and model them accordingly with conditional qualifiers. The combination of bitemporal RDF-star annotations, PROV-based provenance, and SHACL-validated incremental updates provides a production-ready architecture for continuously evolving knowledge graphs.