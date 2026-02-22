This is a great question — it sits at the intersection of NLP coreference resolution and entity linking/disambiguation at scale. Here are the most effective approaches:

## Blocking / Candidate Generation

The core challenge is that you can't compare every mention against hundreds of thousands of entities. So every practical system uses a **candidate retrieval** step first:

- **Sparse retrieval (BM25/TF-IDF):** Index your entity names and aliases, then retrieve a short candidate list (top-k) for each mention. Fast, simple, and surprisingly effective as a first pass.
- **Dense retrieval:** Encode entities and mentions into a shared embedding space (e.g., using a bi-encoder like BLINK from Facebook Research). At query time, use approximate nearest neighbor search (FAISS, ScaNN, HNSW) to pull top-k candidates in sub-linear time. This handles synonyms and paraphrases far better than sparse methods.
- **Alias tables / string normalization:** Maintain a dictionary mapping surface forms, acronyms, abbreviations, and common misspellings to canonical entity IDs. This catches the easy cases cheaply.

## Re-ranking / Disambiguation

Once you have a short candidate list (say 10–50), you apply a more expensive model:

- **Cross-encoders:** Pass `[mention + context, candidate entity description]` through a transformer to score compatibility. Much more accurate than bi-encoders but too slow to run against the full entity set — hence the two-stage pipeline.
- **Graph-based coherence:** If you're resolving multiple mentions in a document, jointly disambiguate them by favoring entity assignments that are coherent in your knowledge graph (e.g., entities that are neighbors or share relations). This is the key insight behind systems like AIDA and REL.

## Practical Architectural Patterns

For your specific scenario (preprocessing documents for triple extraction against a large existing graph):

1. **Bi-encoder + FAISS pipeline:** Embed all your graph entities offline. At inference, encode each mention, do ANN lookup, then re-rank with a cross-encoder. This is the standard modern approach (BLINK, GENRE, ReFinED all follow variants of this).

2. **Hierarchical resolution:** First resolve "easy" mentions via exact/fuzzy string match against your alias table. Only route ambiguous or novel mentions to the neural pipeline. This saves enormous compute.

3. **Type-constrained resolution:** If your graph has entity types (Person, Organization, Location, etc.), run NER first to get the mention's type, then only search within that type partition. This reduces the candidate space by an order of magnitude or more.

4. **Incremental index updates:** As your graph grows, you need to add new entity embeddings without re-encoding everything. Use an index that supports incremental adds (FAISS IndexIVF with `add()`, or a vector database like Milvus/Qdrant/Weaviate).

## Handling True Coreference (Pronouns, Nominals)

If you also need to resolve pronouns and nominal mentions ("the company", "he", "the agreement") before linking to graph entities, layer a **within-document coreference model** first (e.g., a SpanBERT-based coref model, or an LLM-based approach), cluster the mentions into chains, then link each chain's most informative mention (usually the longest named mention) to your entity graph using the pipeline above.

## Key Recommendations

- **Don't skip the blocking step.** Naive all-pairs comparison is O(mentions × entities) and will not scale.
- **Invest in alias table quality.** A good alias table with abbreviations, alternate names, and common references resolves 60–80% of mentions before any ML is needed.
- **Use context windows.** When encoding mentions, include surrounding sentences — a mention of "Apple" in a tech article vs. a recipe is trivially disambiguated with context.
- **Benchmark on your domain.** Off-the-shelf entity linkers are trained on Wikipedia; if your graph is domain-specific (biomedical, legal, financial), you'll likely need to fine-tune the bi-encoder and cross-encoder on your own data.

The state of the art for this general problem is well-represented by systems like **ReFinED** (Amazon), **BLINK** (Meta), **mGENRE** (for multilingual), and **REL** (Radboud). All follow the two-stage retrieve-then-rerank pattern and scale to millions of entities.