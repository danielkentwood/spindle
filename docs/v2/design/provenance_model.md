# Provenance Model

## Two Access Patterns

1. **Edge/Entity → Provenance** (point lookup): User clicks an edge → fetch its `ProvenanceObject` → display all source docs and evidence spans.
2. **Doc → Affected Provenance** (reverse lookup): A source document changes → find *all* `ProvenanceObject`s referencing that `doc_id` → cascade updates to the graph.

## Current State

The existing Kùzu schema stores provenance inline as JSON on edges:

```23:34:spindle/graph_store/backends/kuzu.py
// (illustrative — the supporting_evidence JSON blob)
```

This works for pattern 1 (it's already on the edge), but pattern 2 is brutal: you'd have to scan every edge's JSON blob to find which ones reference a given `doc_id`. No indexing, no efficiency.

## Recommendation: SQLite Side-Table with Indexed Foreign Keys

The cleanest approach is a **dedicated SQLite provenance store** with Kùzu edges/entities holding only a lightweight `provenance_object_id` reference.

### Schema

```python
# Three normalized tables in SQLite

# 1. Links provenance objects to their graph owners
CREATE TABLE provenance_objects (
    object_id TEXT PRIMARY KEY,       -- same as the edge/entity id in Kùzu
    object_type TEXT NOT NULL          -- 'kg_edge' | 'owl_entity' | 'vocab_entry'
);
CREATE INDEX idx_prov_type ON provenance_objects(object_type);

# 2. Which docs contribute to each provenance object
CREATE TABLE provenance_docs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provenance_object_id TEXT NOT NULL REFERENCES provenance_objects(object_id) ON DELETE CASCADE,
    doc_id TEXT NOT NULL
);
CREATE INDEX idx_provdoc_object ON provenance_docs(provenance_object_id);
CREATE INDEX idx_provdoc_docid ON provenance_docs(doc_id);  -- critical for pattern 2

# 3. Evidence spans within each doc
CREATE TABLE evidence_spans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provenance_doc_id INTEGER NOT NULL REFERENCES provenance_docs(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    start_offset INTEGER,
    end_offset INTEGER,
    section_path TEXT              -- JSON array of section headings, e.g. '["Introduction", "Background"]'
);
CREATE INDEX idx_spans_provdoc ON evidence_spans(provenance_doc_id);
CREATE INDEX idx_spans_section ON evidence_spans(section_path);
```

### Why This Works Well for Both Patterns

**Pattern 1 — Edge click → show sources:**
```sql
SELECT pd.doc_id, es.text, es.start_offset, es.end_offset, es.section_path
FROM provenance_objects po
JOIN provenance_docs pd ON pd.provenance_object_id = po.object_id
JOIN evidence_spans es ON es.provenance_doc_id = pd.id
WHERE po.object_id = ?;  -- PK lookup, O(1)
```
Single indexed lookup + two joins. Microsecond-level on SQLite. The `section_path` column (a JSON array like `["Introduction", "Background"]`) tells the caller which document section the evidence came from, enabling UI features like "show in context" without re-parsing the source document.

**Pattern 2 — Doc changed → find all affected graph elements:**
```sql
SELECT po.object_type, po.object_id
FROM provenance_docs pd
JOIN provenance_objects po ON po.object_id = pd.provenance_object_id
WHERE pd.doc_id = ?;  -- B-tree index on doc_id
```
This gives you every edge/entity/vocab entry that references the changed document, directly via the `idx_provdoc_docid` index. Then you can:
- Delete/update the affected evidence spans
- Re-extract from the updated document
- Merge new evidence into existing provenance objects

## Why Not Other Options

| Approach | Pattern 1 | Pattern 2 | Drawback |
|----------|-----------|-----------|----------|
| **JSON in Kùzu edges** (current) | Fast (inline) | Scan all edges | No reverse index; O(n) for doc lookup |
| **ProvenanceObject as Kùzu nodes** | 1-hop traversal | Index on doc_id node | Kùzu can't create relationships FROM relationships (no hyper-edges), so you'd need a clunky FK-on-node pattern |
| **Redis/document store** | Fast | Need secondary index | Extra infra; you already have SQLite |
| **SQLite (recommended)** | PK lookup | B-tree indexed | Two-store coordination (manageable) |

## Integration with RDF-Star Vocabulary

For your vocabulary in RDF-star (Oxigraph), provenance annotation is a first-class concept. You could do something like:

```turtle
<< spndl:concept/machine-learning a skos:Concept >>
    spndl:hasProvenance "machine-learning-001" .
```

The RDF-star annotation holds the `object_id` as a **string literal** matching `provenance_objects.object_id` in SQLite (= the concept's `dct:identifier`). The actual `ProvenanceObject` data (which docs, which spans) lives in SQLite rather than being materialized as dozens of RDF triples.

## Practical Considerations

1. **`object_id` = graph element ID?** Since each edge/entity gets exactly one ProvenanceObject (per your workflow doc), `object_id` is the Kùzu edge `id` (or SKOS `dct:identifier`, SHACL shape name, etc.). No extra indirection needed.

2. **Cascade deletes**: SQLite's `ON DELETE CASCADE` on the foreign keys means deleting a provenance object automatically cleans up its docs and spans.

3. **Batch updates**: When a document changes, you can do the reverse lookup, delete all affected `provenance_docs` rows for that `doc_id` (cascading to spans), then re-insert after re-extraction — all in a single transaction.

4. **Consistency**: Wrap Kùzu graph mutations and SQLite provenance mutations in application-level transactions (since they're separate stores). Your existing `EventRecorder` pattern could help here — record the intent, then apply to both stores.

5. **Scale**: SQLite handles millions of rows with proper indexing trivially. Unless you're dealing with hundreds of millions of evidence spans, this will be more than sufficient.