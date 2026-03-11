## KOS Data Model

Spindle's KOS (Knowledge Organization System) runtime is implemented by `KOSService`.
It loads SKOS/OWL/SHACL artifacts into an in-process Oxigraph store and builds
derived indices for low-latency lookup and resolution.

## On-disk inputs

Expected under `kos/`:

- `kos.ttls`
- `ontology.owl`
- `shapes.ttl`
- `config/scheme.ttl`
- optional `blacklist.txt`

## Derived runtime structures

- concept cache (`uri -> ConceptRecord`)
- label map (`normalized label -> concept URIs`)
- Aho-Corasick automaton for lexical mention detection
- optional ANN index for semantic lookup (when embedding function is available)

## Core operations

- `search_ahocorasick(text, longest_match_only=False)`
- `search_ann(query, top_k=10)`
- `resolve_multistep(mentions, threshold=0.7)`
- concept CRUD (`list_concepts`, `get_concept`, `create_concept`, `delete_concept`)
- hierarchy helpers (`get_hierarchy`, `get_ancestors`, `get_descendants`)
- validation (`validate_skos`, `validate_triples`)
- `sparql(query)`
- `reload()`

## API surface

FastAPI routes are exposed under `/kos`, including:

- `/kos/search/*`
- `/kos/concepts*`
- `/kos/hierarchy/*`
- `/kos/validate*`
- `/kos/sparql`
- `/kos/reload`

## Notes

- `resolve_multistep` is the current mention-resolution method.
- ANN search returns empty results if no embedding function is configured.
# KOS Data Model & Serving Architecture

This document specifies how Spindle stores, loads, and serves its Knowledge Organization System (KOS) artifacts: the consolidated SKOS thesaurus (vocabulary + taxonomy + associative relationships), the OWL ontology, and optional SHACL shapes. It also describes the `KOSService` runtime, the derived indices for low-latency operations, and the FastAPI endpoints.

---

## Table of Contents

1. [File Layout](#1-file-layout)
2. [Data Model: SKOS + RDF-star](#2-data-model-skos--rdf-star)
3. [Data Model: OWL + PROV-O Ontology](#3-data-model-owl--prov-o-ontology)
4. [Data Model: SHACL Shapes](#4-data-model-shacl-shapes)
5. [Provenance Integration](#5-provenance-integration)
6. [Blacklist](#6-blacklist)
7. [KOSService Runtime](#7-kosservice-runtime)
8. [Derived Indices](#8-derived-indices)
9. [FastAPI Endpoints](#9-fastapi-endpoints)
10. [Pipeline Integration](#10-pipeline-integration)
11. [Scaling & Migration Path](#11-scaling--migration-path)

---

## 1. File Layout

```
kos/
├── kos.ttls                      # Consolidated SKOS thesaurus (source of truth)
├── ontology.owl                  # OWL classes, properties, restrictions + PROV-O
├── shapes.ttl                    # SHACL validation constraints
├── blacklist.txt                 # Terms to ignore during extraction (one per line)
├── rejections.db                 # SQLite: rejected staging candidates (see §6.1)
├── staging/                      # Pipeline intermediates (not loaded by KOSService)
│   ├── vocabulary.ttls           # Stage output: concepts, labels, definitions
│   ├── taxonomy.ttls             # Stage output: adds skos:broader/narrower
│   └── thesaurus.ttls            # Stage output: adds skos:related, scope notes
├── provenance/
│   └── (managed by SQLite — see §5)
└── config/
    └── scheme.ttl                # skos:ConceptScheme + namespace prefix declarations
```

### File Formats

| File | Format | Specification | Purpose |
|------|--------|---------------|---------|
| `kos.ttls` | Turtle-Star | [RDF-star in Turtle](https://w3c.github.io/rdf-star/cg-spec/editors_draft.html) | Consolidated SKOS thesaurus with RDF-star provenance references |
| `ontology.owl` | OWL 2 / Turtle | [OWL 2 Web Ontology Language](https://www.w3.org/TR/owl2-overview/) | Formal ontology with PROV-O annotations |
| `shapes.ttl` | Turtle | [SHACL](https://www.w3.org/TR/shacl/) | Data validation constraints |
| `scheme.ttl` | Turtle | [SKOS](https://www.w3.org/TR/skos-reference/) | ConceptScheme metadata, namespace prefixes |
| `staging/*.ttls` | Turtle-Star | Same as `kos.ttls` | Per-stage pipeline intermediates |
| `blacklist.txt` | Plain text | One term per line | Terms excluded from NER and extraction |
| `rejections.db` | SQLite | — | Rejected staging candidates (see §6.1) |

### Two-File Model Rationale

The consolidated `kos.ttls` merges what the pipeline produces in three stages (vocabulary, taxonomy, thesaurus) into a single SKOS thesaurus. This is the file that `KOSService` loads and that downstream consumers query.

- **Why not three files?** SKOS relationships are cross-cutting. `skos:broader`, `skos:narrower`, and `skos:related` all connect the same `skos:Concept` resources. Splitting by relationship type forces artificial file boundaries, complicates Oxigraph loading, and fragments provenance queries.
- **Why a staging area?** The pipeline writes incrementally — vocabulary first, then taxonomy, then thesaurus. Staging files preserve per-stage outputs for user review, debugging, and selective re-runs. After validation, they are merged into `kos.ttls`.
- **The ontology is separate** because it uses OWL (a different formalism), has a different lifecycle (it evolves with domain modeling, not with term extraction), and is consumed by different tools (reasoners vs. SKOS validators).

---

## 2. Data Model: SKOS + RDF-star

All vocabulary terms, hierarchical relationships, and associative relationships live in `kos.ttls` as SKOS Concepts. RDF-star annotations attach provenance references (see §5).

### Namespace Prefixes

```turtle
@prefix skos:  <http://www.w3.org/2004/02/skos/core#> .
@prefix dct:   <http://purl.org/dc/terms/> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
@prefix spndl: <http://spindle.dev/ns/> .
```

### ConceptScheme (in `config/scheme.ttl`)

```turtle
spndl:scheme/main a skos:ConceptScheme ;
    dct:title "Spindle KOS"@en ;
    dct:created "2025-01-15"^^xsd:date ;
    dct:description "Auto-generated controlled vocabulary for the project corpus"@en .
```

### Concept with All Layers

A single concept in `kos.ttls` carries vocabulary, taxonomy, and thesaurus properties together:

```turtle
# --- Vocabulary layer ---
spndl:concept/pump a skos:Concept ;
    skos:inScheme spndl:scheme/main ;
    dct:identifier "pump-001" ;
    skos:prefLabel "Pump"@en ;
    skos:altLabel "Fluid pump"@en , "Mechanical pump"@en ;
    skos:definition "A mechanical device used to move fluids"@en ;
    skos:scopeNote "In the context of process engineering"@en ;

    # --- Taxonomy layer ---
    skos:broader spndl:concept/equipment ;

    # --- Thesaurus layer ---
    skos:related spndl:concept/valve ,
                 spndl:concept/motor .

# --- RDF-star: provenance object references ---
<<spndl:concept/pump a skos:Concept>>
    spndl:hasProvenance "pump-001" .

<<spndl:concept/pump skos:altLabel "Fluid pump"@en>>
    spndl:hasProvenance "pump-001" .

<<spndl:concept/pump skos:broader spndl:concept/equipment>>
    spndl:hasProvenance "pump-001" ;
    spndl:confidence "high" .

<<spndl:concept/pump skos:related spndl:concept/valve>>
    spndl:hasProvenance "pump-001" ;
    spndl:confidence "medium" .
```

### Top Concepts

Root concepts in the taxonomy hierarchy use `skos:topConceptOf`:

```turtle
spndl:concept/equipment a skos:Concept ;
    skos:topConceptOf spndl:scheme/main ;
    skos:prefLabel "Equipment"@en ;
    skos:definition "Physical apparatus used in industrial processes"@en .
```

### SKOS Properties Used

| Property | Layer | Cardinality | Description |
|----------|-------|-------------|-------------|
| `skos:prefLabel` | Vocabulary | 1 per language | Canonical label |
| `skos:altLabel` | Vocabulary | 0..n | Synonyms (maps to ISO 25964 UF) |
| `skos:hiddenLabel` | Vocabulary | 0..n | Indexing-only labels (misspellings, abbreviations) |
| `skos:definition` | Vocabulary | 0..1 | Natural-language definition |
| `skos:scopeNote` | Vocabulary | 0..1 | Usage context clarification |
| `skos:broader` | Taxonomy | 0..n | Parent concept(s) — polyhierarchy allowed |
| `skos:narrower` | Taxonomy | 0..n | Child concept(s) |
| `skos:topConceptOf` | Taxonomy | 0..1 | Marks root concepts |
| `skos:related` | Thesaurus | 0..n | Associative (non-hierarchical) relationships |
| `skos:historyNote` | Thesaurus | 0..1 | Historical context or change notes |
| `skos:exactMatch` | Thesaurus | 0..n | Equivalence to concepts in external schemes |
| `skos:closeMatch` | Thesaurus | 0..n | Near-equivalence to external concepts |
| `dct:identifier` | Vocabulary | 1 | Stable term ID (used as provenance `object_id`) |
| `skos:inScheme` | Vocabulary | 1 | Links concept to its ConceptScheme |

### `dct:identifier` Generation Strategy

Every `skos:Concept` must have exactly one `dct:identifier`. This value is the primary key that links the RDF-star `spndl:hasProvenance` annotation to the SQLite `provenance_objects.object_id`. Identifiers are generated as follows:

1. **Slug from `skos:prefLabel`**: Lowercase the preferred label, replace whitespace and non-alphanumeric characters with hyphens, collapse consecutive hyphens, and strip leading/trailing hyphens. Example: `"Centrifugal Pump"` → `centrifugal-pump`.
2. **Append a numeric suffix** starting at `-001` if the slug already exists in the current KOS (to handle label collisions). Example: `centrifugal-pump-001`.
3. **Identifiers are immutable** — once assigned, a concept's `dct:identifier` never changes, even if `skos:prefLabel` is updated. This preserves provenance linkage integrity.

For OWL entities in `ontology.owl`, the `spndl:hasProvenance` value follows the pattern `ontology-{slug}` (e.g., `"ontology-pump"`), using the same slugification rules.

---

## 3. Data Model: OWL + PROV-O Ontology

The ontology lives in `ontology.owl` and defines the formal class hierarchy and object/data properties that govern knowledge graph extraction. PROV-O annotations track which pipeline run generated each axiom.

### Namespace Prefixes

```turtle
@prefix owl:   <http://www.w3.org/2002/07/owl#> .
@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
@prefix prov:  <http://www.w3.org/ns/prov#> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
@prefix spndl: <http://spindle.dev/ns/> .
```

### Ontology Header

```turtle
spndl:ontology a owl:Ontology ;
    rdfs:label "Spindle Domain Ontology"@en ;
    owl:versionIRI spndl:ontology/v1 ;
    prov:wasGeneratedBy spndl:activity/kos-pipeline-run-001 ;
    prov:generatedAtTime "2025-06-01T12:00:00Z"^^xsd:dateTime .
```

### Class Definitions

```turtle
spndl:Equipment a owl:Class ;
    rdfs:label "Equipment"@en ;
    rdfs:comment "Physical apparatus used in industrial processes"@en ;
    prov:wasDerivedFrom spndl:activity/ontology-synthesis ;
    spndl:hasProvenance "ontology-equipment" .

spndl:Pump a owl:Class ;
    rdfs:subClassOf spndl:Equipment ;
    rdfs:label "Pump"@en ;
    prov:wasDerivedFrom spndl:activity/ontology-synthesis ;
    spndl:hasProvenance "ontology-pump" .
```

### Object Properties

```turtle
spndl:operatedBy a owl:ObjectProperty ;
    rdfs:domain spndl:Equipment ;
    rdfs:range spndl:Operator ;
    rdfs:label "operated by"@en ;
    prov:wasDerivedFrom spndl:activity/ontology-synthesis ;
    spndl:hasProvenance "ontology-operatedBy" .
```

### Data Properties (Attributes)

```turtle
spndl:flowRate a owl:DatatypeProperty ;
    rdfs:domain spndl:Pump ;
    rdfs:range xsd:float ;
    rdfs:label "flow rate"@en ;
    rdfs:comment "Operating flow rate in liters per minute"@en ;
    spndl:hasProvenance "ontology-flowRate" .
```

### PROV-O Activity Records

Pipeline run metadata is embedded in the ontology file:

```turtle
spndl:activity/kos-pipeline-run-001 a prov:Activity ;
    prov:startedAtTime "2025-06-01T11:30:00Z"^^xsd:dateTime ;
    prov:endedAtTime "2025-06-01T12:00:00Z"^^xsd:dateTime ;
    prov:wasAssociatedWith spndl:agent/spindle-v2 ;
    prov:used spndl:corpus/project-docs-v3 .

spndl:agent/spindle-v2 a prov:SoftwareAgent ;
    rdfs:label "Spindle v2 KOS Pipeline" .

spndl:activity/ontology-synthesis a prov:Activity ;
    prov:wasInformedBy spndl:activity/kos-pipeline-run-001 ;
    rdfs:label "Ontology synthesis from accumulated KOS artifacts" .
```

---

## 4. Data Model: SHACL Shapes

SHACL shapes in `shapes.ttl` validate data entering the knowledge graph against the ontology. They are kept in a separate file because they have a different lifecycle (data quality rules evolve independently of the domain model) and are consumed by different tools (`pyshacl` vs. OWL reasoners).

```turtle
@prefix sh:    <http://www.w3.org/ns/shacl#> .
@prefix spndl: <http://spindle.dev/ns/> .
@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .

spndl:EquipmentShape a sh:NodeShape ;
    sh:targetClass spndl:Equipment ;
    sh:property [
        sh:path rdfs:label ;
        sh:minCount 1 ;
        sh:datatype xsd:string ;
        sh:message "Every Equipment instance must have a label" ;
    ] ;
    sh:property [
        sh:path spndl:operatedBy ;
        sh:class spndl:Operator ;
        sh:message "operatedBy must point to an Operator" ;
    ] .

spndl:PumpShape a sh:NodeShape ;
    sh:targetClass spndl:Pump ;
    sh:property [
        sh:path spndl:flowRate ;
        sh:datatype xsd:float ;
        sh:minInclusive 0.0 ;
        sh:message "flowRate must be a non-negative float" ;
    ] .
```

SHACL shapes can be auto-generated from OWL axioms as a starting point, then manually tightened. Validation runs before triples enter the knowledge graph:

```bash
uv run pyshacl -s kos/shapes.ttl -e kos/ontology.owl -df turtle data.ttl
```

---

## 5. Provenance Integration

Provenance follows the model described in [`provenance_model.md`](provenance_model.md). The key design decision: **RDF-star annotations hold only a provenance `object_id` reference; the full provenance data (source documents, evidence spans) lives in the SQLite provenance store.**

### Why Not Materialize Provenance as RDF?

A single concept with 5 source documents and 20 evidence spans would produce 50+ RDF triples just for provenance. Multiplied across thousands of concepts, this would bloat `kos.ttls` significantly and make SPARQL provenance queries slower than indexed SQLite lookups.

### How It Works

1. **In `kos.ttls`** — RDF-star annotations carry the `object_id`:

    ```turtle
    <<spndl:concept/pump a skos:Concept>>
        spndl:hasProvenance "pump-001" .
    ```

2. **In SQLite** — The `provenance_objects` table (with `object_type = 'vocab_entry'`) holds the detailed provenance:

    ```sql
    -- Point lookup: concept → source docs + spans + section context
    SELECT pd.doc_id, es.text, es.start_offset, es.end_offset, es.section_path
    FROM provenance_objects po
    JOIN provenance_docs pd ON pd.provenance_object_id = po.object_id
    JOIN evidence_spans es ON es.provenance_doc_id = pd.id
    WHERE po.object_id = 'pump-001';

    -- Reverse lookup: doc changed → affected concepts
    SELECT po.object_type, po.object_id
    FROM provenance_docs pd
    JOIN provenance_objects po ON po.object_id = pd.provenance_object_id
    WHERE pd.doc_id = 'manual-v2';
    ```

3. **For the ontology** — Same pattern. Each OWL class/property carries `spndl:hasProvenance` directly (see §3 examples), and the SQLite store holds the detail with `object_type = 'owl_entity'`. SHACL shapes (§4) inherit provenance transitively through the OWL entity they target — they do not carry independent provenance.

### `object_type` Values

| `object_type` | Description | `object_id` Example |
|--------------|-------------|---------------------|
| `vocab_entry` | SKOS concept in `kos.ttls` | `pump-001` (= `dct:identifier`) |
| `owl_entity` | OWL class or property in `ontology.owl` | `ontology-pump` (= `spndl:hasProvenance` value) |
| `kg_edge` | Triple in the knowledge graph (Kùzu) | `edge-abc123` |

---

## 6. Blacklist

`blacklist.txt` contains terms that should be excluded from NER recognition and vocabulary extraction. One term per line, case-insensitive matching:

```
the
a
an
it
this
```

The `KOSService` loads this file on startup and filters blacklisted terms from the Aho-Corasick automaton and from vocabulary extraction results.

### 6.1 Rejection Log

`rejections.db` is a SQLite database that records staging candidates rejected during human review (see [kos_extraction.md](kos_extraction.md) for the review workflow). Terms that appear frequently in the rejection log are candidates for the blacklist.

```sql
CREATE TABLE rejections (
    id INTEGER PRIMARY KEY,
    rejected_term TEXT NOT NULL,
    source_doc_id TEXT NOT NULL,
    chunk_index INTEGER,
    rejection_reason TEXT,
    rejected_at TEXT NOT NULL,   -- ISO 8601
    rejected_by TEXT             -- user identifier
);
```

---

## 7. KOSService Runtime

`KOSService` is the in-process runtime that loads KOS artifacts into memory and exposes query methods. It is instantiated once at application startup and shared across FastAPI request handlers.

### Architecture

```
                          ┌────────────────────────────────┐
                          │        KOS Files (disk)        │
                          │                                │
                          │  kos.ttls    ontology.owl      │
                          │  shapes.ttl  blacklist.txt     │
                          │  config/scheme.ttl             │
                          └───────────────┬────────────────┘
                                          │ load on startup
                                          │ reload on POST /kos/reload
                                          ▼
              ┌────────────────────────────────────────────────────┐
              │                    KOSService                      │
              │                                                    │
              │  ┌──────────────────┐   ┌───────────────────────┐  │
              │  │ Oxigraph Store   │   │ Derived Indices       │  │
              │  │ (in-process)     │   │                       │  │
              │  │                  │   │ • NER Automaton       │  │
              │  │ • SPARQL 1.1     │   │   (Aho-Corasick)      │  │
              │  │ • SPARQL-star    │   │                       │  │
              │  │ • Graph CRUD     │   │ • Search Index        │  │
              │  │                  │   │   (hnswlib or FAISS)  │  │
              │  │ Loads:           │   │                       │  │
              │  │  kos.ttls        │   │ • Label→URI Map       │  │
              │  │  ontology.owl    │   │   (dict)              │  │
              │  │  shapes.ttl      │   │                       │  │
              │  │  scheme.ttl      │   │ • URI→Concept Cache   │  │
              │  │                  │   │   (dict)              │  │
              │  └──────────────────┘   └───────────────────────┘  │
              │                                                    │
              │  ┌──────────────────┐   ┌───────────────────────┐  │
              │  │ Provenance Store │   │ Blacklist             │  │
              │  │ (SQLite — shared │   │ (set[str])            │  │
              │  │  with KG prov)   │   │                       │  │
              │  └──────────────────┘   └───────────────────────┘  │
              └────────────────────────────────────────────────────┘
```

### Oxigraph Store

The Oxigraph store is the queryable in-process representation of the KOS. It loads all Turtle-Star and OWL files and serves SPARQL/SPARQL-star queries.

**What gets loaded:**

| File | Named Graph URI | Purpose |
|------|-----------------|---------|
| `kos.ttls` | `spndl:graph/kos` | Consolidated SKOS thesaurus |
| `ontology.owl` | `spndl:graph/ontology` | OWL class/property definitions |
| `shapes.ttl` | `spndl:graph/shapes` | SHACL constraints (for validation queries) |
| `config/scheme.ttl` | `spndl:graph/scheme` | ConceptScheme metadata |

Using named graphs allows SPARQL queries to target a specific layer:

```sparql
SELECT ?concept ?label WHERE {
    GRAPH spndl:graph/kos {
        ?concept a skos:Concept ;
                 skos:prefLabel ?label .
    }
}
```

Or query across all graphs by omitting the `GRAPH` clause.

**Concurrency model:** Oxigraph supports concurrent reads. Writes (which only happen during `reload()` or CRUD mutations) acquire a write lock. Since KOS updates are infrequent batch operations (pipeline runs, user edits), this is not a bottleneck.

### Key Methods

In addition to the query methods exposed via FastAPI endpoints (§9), `KOSService` provides internal methods for in-process consumers:

```python
def get_label_set(self, include_alt: bool = True) -> list[str]:
    """Return deduplicated list of prefLabels and optionally altLabels for NER seeding.

    Reads from the already-built Label-to-URI map (§8.3), so effectively free.
    Used by the GLiNER2 discovery pass (see kos_extraction.md) to seed the
    neural NER label set with known vocabulary terms.
    """
```

### Lifecycle

1. **Startup:** `KOSService.__init__()` loads files into Oxigraph, builds derived indices, loads blacklist.
2. **Serving:** All reads (NER, search, SPARQL, provenance lookups) are concurrent and lock-free.
3. **Reload:** `POST /kos/reload` triggers a full reload — new Oxigraph store, rebuilt indices. The swap is atomic (build new, then replace reference).
4. **CRUD writes:** Concept mutations go through Oxigraph first, then derived indices are incrementally updated or fully rebuilt.

---

## 8. Derived Indices

Derived indices are purpose-built data structures optimized for low-latency operations. They are built from the Oxigraph store contents and rebuilt on reload.

### 8.1 NER Automaton (Aho-Corasick)

**Purpose:** Find all vocabulary terms mentioned in a text string in a single linear scan.

**Data source:** All `skos:prefLabel`, `skos:altLabel`, and `skos:hiddenLabel` values from `kos.ttls`, minus blacklisted terms.

**Complexity:** O(n) where n = length of input text, regardless of vocabulary size.

**Build process:**
1. SPARQL query extracts all (label, concept URI) pairs.
2. Blacklisted labels are filtered out.
3. Labels are lowercased and added to a `pyahocorasick.Automaton`.
4. The automaton is finalized with `.make_automaton()`.

**Query behavior:**
- Input: text string (typically coref-resolved chunk text from doc_service — see [kos_extraction.md](kos_extraction.md) for the extraction pipeline context).
- Output: list of `EntityMention` objects with `text`, `start`, `end`, `concept_uri`, `matched_label`.
- Overlapping matches are returned (caller can apply longest-match or other disambiguation).

### 8.2 Search Index (Vector ANN)

**Purpose:** Fuzzy semantic search over concept labels and definitions.

**Data source:** For each concept, embed a text string composed of `"{prefLabel}: {definition}"`. Uses the same embedding model as the project's `ChromaVectorStore`.

**Implementation options (in order of preference):**
1. **hnswlib** — lightweight, fast, no external server. Good for up to ~1M vectors.
2. **FAISS** — more features (IVF, PQ), heavier dependency.
3. **ChromaVectorStore** — reuse existing Spindle infrastructure, but adds Chroma dependency to the KOS service.

**Query behavior:**
- Input: query string + `top_k`.
- Output: ranked list of `(concept_uri, score)` pairs.
- Used for fuzzy search, vocabulary validation, and the "medium pass" of entity resolution (semantic search over vocabulary).

### 8.3 Label-to-URI Map

**Purpose:** O(1) exact label lookup for the "fast pass" of entity resolution.

**Data source:** All `skos:prefLabel` and `skos:altLabel` values, lowercased and normalized (whitespace collapsed, punctuation stripped).

**Data structure:** `dict[str, list[str]]` mapping normalized label → list of concept URIs (multiple concepts may share an altLabel).

**Query behavior:**
- Input: normalized entity mention string.
- Output: list of candidate concept URIs, or empty list.

### 8.4 URI-to-Concept Cache

**Purpose:** Avoid SPARQL round-trips for concept detail lookups on hot paths.

**Data source:** SPARQL query that materializes all concept properties into Python dataclass instances.

**Data structure:** `dict[str, ConceptRecord]` where `ConceptRecord` holds `uri`, `pref_label`, `alt_labels`, `definition`, `broader`, `narrower`, `related`, `provenance_object_id`.

---

## 9. FastAPI Endpoints

The KOS API is a new FastAPI router mounted at `/kos`. It wraps `KOSService` methods.

### 9.1 NER — Named Entity Recognition

```
POST /kos/search/ahocorasick
```

Scans input text for vocabulary term mentions using the Aho-Corasick automaton.

**Request body:**
```json
{
    "text": "The centrifugal pump is connected to the main valve.",
    "longest_match_only": true
}
```

**Response:**
```json
{
    "mentions": [
        {
            "text": "centrifugal pump",
            "start": 4,
            "end": 20,
            "concept_uri": "http://spindle.dev/ns/concept/centrifugal-pump",
            "matched_label": "centrifugal pump",
            "pref_label": "Centrifugal Pump"
        },
        {
            "text": "valve",
            "start": 45,
            "end": 50,
            "concept_uri": "http://spindle.dev/ns/concept/valve",
            "matched_label": "valve",
            "pref_label": "Valve"
        }
    ]
}
```

### 9.2 Search — Semantic Concept Search

```
GET /kos/search/ann?q={query}&top_k={k}
```

Fuzzy semantic search over concept labels and definitions.

**Response:**
```json
{
    "results": [
        {
            "concept_uri": "http://spindle.dev/ns/concept/pump",
            "pref_label": "Pump",
            "definition": "A mechanical device used to move fluids",
            "score": 0.92
        }
    ]
}
```

### 9.3 Resolve — Entity Resolution (Fast + Medium)

```
POST /kos/search/multistep
```

Given a list of entity mentions, resolve each to a vocabulary concept (or flag as unresolved). Runs the fast pass (exact label match) first, then the medium pass (semantic search) for unresolved mentions.

**Request body:**
```json
{
    "mentions": ["fluid pump", "centrifugal pump", "XYZ-9000"],
    "threshold": 0.7
}
```

**Response:**
```json
{
    "resolutions": [
        {
            "mention": "fluid pump",
            "resolved": true,
            "method": "exact_label",
            "concept_uri": "http://spindle.dev/ns/concept/pump",
            "pref_label": "Pump",
            "score": 1.0
        },
        {
            "mention": "centrifugal pump",
            "resolved": true,
            "method": "exact_label",
            "concept_uri": "http://spindle.dev/ns/concept/centrifugal-pump",
            "pref_label": "Centrifugal Pump",
            "score": 1.0
        },
        {
            "mention": "XYZ-9000",
            "resolved": false,
            "method": "semantic_search",
            "candidates": [],
            "score": 0.0
        }
    ]
}
```

### 9.4 Concept CRUD

```
GET    /kos/concepts                        # List concepts (paginated)
GET    /kos/concepts/{concept_id}           # Get concept detail
POST   /kos/concepts                        # Create concept
PUT    /kos/concepts/{concept_id}           # Update concept
DELETE /kos/concepts/{concept_id}           # Soft-delete concept
```

**GET /kos/concepts/{concept_id} response:**
```json
{
    "uri": "http://spindle.dev/ns/concept/pump",
    "concept_id": "pump-001",
    "pref_label": "Pump",
    "alt_labels": ["Fluid pump", "Mechanical pump"],
    "definition": "A mechanical device used to move fluids",
    "scope_note": "In the context of process engineering",
    "broader": ["http://spindle.dev/ns/concept/equipment"],
    "narrower": [
        "http://spindle.dev/ns/concept/centrifugal-pump",
        "http://spindle.dev/ns/concept/positive-displacement-pump"
    ],
    "related": [
        "http://spindle.dev/ns/concept/valve",
        "http://spindle.dev/ns/concept/motor"
    ],
    "provenance_object_id": "pump-001"
}
```

Write operations (POST, PUT, DELETE) mutate the Oxigraph store and trigger an incremental rebuild of affected derived indices.

### 9.5 Provenance — Concept Provenance Lookup

```
GET /kos/concepts/{concept_id}/provenance
```

Fetches the full provenance chain for a concept: source documents and evidence spans from the SQLite provenance store.

**Response:**
```json
{
    "concept_id": "pump-001",
    "docs": [
        {
            "doc_id": "manual-v2",
            "evidence_spans": [
                {"text": "The pump is used to move fluid", "start": 142, "end": 172, "section_path": ["Equipment", "Pumps"]},
                {"text": "centrifugal and positive displacement pumps", "start": 305, "end": 348, "section_path": ["Equipment", "Pumps", "Types"]}
            ]
        },
        {
            "doc_id": "spec-3.1",
            "evidence_spans": [
                {"text": "fluid pump specifications", "start": 12, "end": 37, "section_path": ["Specifications"]}
            ]
        }
    ]
}
```

### 9.6 Hierarchy — Taxonomy Traversal

```
GET /kos/hierarchy?root={concept_id}&depth={max_depth}
GET /kos/hierarchy/roots
GET /kos/hierarchy/{concept_id}/ancestors
GET /kos/hierarchy/{concept_id}/descendants
```

Uses SPARQL property paths (`skos:broader+`, `skos:narrower+`) for transitive traversal.

### 9.7 Ontology — OWL Queries

```
GET /kos/ontology/classes                   # List OWL classes
GET /kos/ontology/classes/{class_name}      # Class detail (properties, restrictions)
GET /kos/ontology/properties                # List object + data properties
GET /kos/ontology/properties/{prop_name}    # Property detail (domain, range)
```

### 9.8 Validation — SHACL

```
POST /kos/validate
```

Validates a set of triples against the SHACL shapes.

**Request body:**
```json
{
    "triples": [
        {
            "subject": "Pump-A1",
            "subject_type": "Pump",
            "predicate": "operatedBy",
            "object": "John Smith",
            "object_type": "Operator"
        }
    ]
}
```

**Response:**
```json
{
    "conforms": true,
    "violations": []
}
```

### 9.9 SPARQL — Ad-Hoc Queries

```
POST /kos/sparql
```

Executes arbitrary SPARQL or SPARQL-star queries against the Oxigraph store. Intended for advanced users, dashboards, and debugging.

**Request body:**
```json
{
    "query": "PREFIX skos: <http://www.w3.org/2004/02/skos/core#> SELECT ?c ?label WHERE { ?c a skos:Concept ; skos:prefLabel ?label } LIMIT 10"
}
```

### 9.10 Admin — Reload

```
POST /kos/reload
```

Triggers a full reload of KOS files from disk and rebuilds all derived indices. Returns the reload status and index statistics.

**Response:**
```json
{
    "status": "ok",
    "concepts_loaded": 1247,
    "ontology_classes": 42,
    "ontology_properties": 38,
    "ner_automaton_patterns": 3891,
    "search_index_vectors": 1247,
    "reload_time_ms": 340
}
```

### 9.11 Rejections — Query Rejection Log

```
GET /kos/rejections?term={term}&doc_id={doc_id}
```

Queries the rejection log (`rejections.db`) for rejected staging candidates. Both query parameters are optional and act as filters. See §6.1 for the schema.

**Response:**
```json
{
    "rejections": [
        {
            "id": 1,
            "rejected_term": "widget assembly",
            "source_doc_id": "manual-v2",
            "chunk_index": 14,
            "rejection_reason": "Too generic, not a domain concept",
            "rejected_at": "2025-07-01T14:30:00Z",
            "rejected_by": "dwood"
        }
    ]
}
```

### Endpoint Summary

| Method | Path | Index Used | Latency Target |
|--------|------|-----------|----------------|
| POST | `/kos/search/ahocorasick` | Aho-Corasick | < 1ms per KB of text |
| GET | `/kos/search/ann` | Vector ANN | < 5ms |
| POST | `/kos/search/multistep` | Label map + Vector ANN | < 10ms per mention |
| GET | `/kos/concepts/{id}` | URI cache (or Oxigraph) | < 1ms |
| GET | `/kos/concepts/{id}/provenance` | SQLite | < 1ms |
| GET | `/kos/hierarchy/*` | Oxigraph SPARQL | < 10ms |
| GET | `/kos/ontology/*` | Oxigraph SPARQL | < 5ms |
| POST | `/kos/validate` | Oxigraph + SHACL | < 50ms |
| POST | `/kos/sparql` | Oxigraph SPARQL | varies |
| POST | `/kos/reload` | (rebuilds all) | < 5s |
| CRUD | `/kos/concepts` | Oxigraph + index rebuild | < 50ms |
| GET | `/kos/rejections` | SQLite | < 5ms |

---

## 10. Pipeline Integration

### Stage Outputs → Staging Files

Each KOS pipeline stage writes its output to the staging directory:

| Pipeline Stage | Staging File | Content |
|----------------|-------------|---------|
| Vocabulary Extraction | `staging/vocabulary.ttls` | `skos:Concept` with `prefLabel`, `altLabel`, `definition`, `scopeNote` + RDF-star provenance refs |
| Taxonomy Extraction | `staging/taxonomy.ttls` | `skos:broader` / `skos:narrower` relationships + provenance refs |
| Thesaurus Extraction | `staging/thesaurus.ttls` | `skos:related` relationships, additional `scopeNote`s + provenance refs |

### Merge Process

After the user reviews and validates each stage, the staging files are merged into `kos.ttls`:

1. Load all three staging files into a temporary Oxigraph store.
2. Run SKOS integrity checks (no orphaned concepts, no cycles in `skos:broader`, all `skos:related` are symmetric).
3. Serialize the merged graph as Turtle-Star to `kos.ttls`.
4. Trigger `POST /kos/reload` to refresh the live service.

The merge is a **graph union** — concepts that appear in multiple staging files (which they will, since taxonomy and thesaurus reference vocabulary concepts) are unified by their URI. The merge adds properties; it does not replace them.

### GLiNER2 Discovery Pass

The KOS vocabulary also serves as a label seed for the GLiNER2-based discovery pass during incremental extraction. GLiNER2 uses `prefLabel` and `altLabel` values (obtained via `KOSService.get_label_set()`) to bias neural NER toward known concepts, while detecting novel entities not yet in the KOS. See [kos_extraction.md](kos_extraction.md) for the full three-pass cascade (Aho-Corasick → multistep → GLiNER2).

### Ontology Stage

Ontology Synthesis writes directly to `ontology.owl`. It consumes the consolidated SKOS thesaurus (via `KOSService`) plus the corpus to generate OWL axioms. SHACL shapes can be auto-generated from the OWL output and saved to `shapes.ttl`.

---

## 11. Scaling & Migration Path

### Current Target (v1)

- **Corpus size:** < 1,500 documents
- **Vocabulary size:** < 50K concepts
- **Oxigraph in-process:** handles this comfortably
- **Single-user writes:** pipeline runs and user edits are serialized

### When to Upgrade

| Signal | Action |
|--------|--------|
| Vocabulary exceeds 100K concepts | Consider domain-sharded TriG-star files with named graphs |
| Multiple concurrent writers needed | Move to Oxigraph Server (HTTP SPARQL endpoint) or GraphDB Free |
| Sub-millisecond SPARQL needed | Add materialized views or pre-computed query caches |
| NER vocabulary exceeds 1M terms | Switch from pyahocorasick to a compiled Rust automaton (e.g., `aho-corasick` crate via PyO3) |

### Migration Path to a Dedicated Triple Store

The `KOSService` interface stays the same. The internal change is:
1. Replace `pyoxigraph.Store()` with an HTTP SPARQL client pointing at GraphDB / Stardog / Oxigraph Server.
2. Derived indices continue to be built by querying the SPARQL endpoint at startup.
3. File-based `kos.ttls` becomes the import format for the triple store rather than the runtime store.
