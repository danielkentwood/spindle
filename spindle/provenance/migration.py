"""One-time migration utility: Kùzu JSON provenance blobs → SQLite ProvenanceStore.

Run once after upgrading to Spindle v2::

    python -m spindle.provenance.migration \\
        --graph-db path/to/kuzu_db \\
        --prov-db path/to/provenance.db

The migration reads ``supporting_evidence`` JSON blobs from all edges in the
Kùzu graph database and imports them into the ProvenanceStore's three-table
schema.  Existing ProvenanceStore rows are preserved; duplicates are skipped.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def migrate_provenance(
    graph_db_path: str | Path,
    prov_db_path: str | Path,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Migrate inline JSON provenance from Kùzu to SQLite ProvenanceStore.

    Args:
        graph_db_path: Path to the Kùzu database directory.
        prov_db_path: Path to the target SQLite provenance database.
        dry_run: If True, scan and report counts without writing.

    Returns:
        Dict with ``edges_scanned``, ``objects_migrated``, ``spans_migrated``,
        ``skipped``.
    """
    try:
        import kuzu
    except ImportError:
        raise ImportError(
            "kuzu is required for provenance migration. "
            "Install it with: pip install kuzu"
        )

    from spindle.provenance.models import EvidenceSpan, ProvenanceDoc
    from spindle.provenance.store import ProvenanceStore

    graph_db_path = Path(graph_db_path)
    prov_db_path = Path(prov_db_path)

    db = kuzu.Database(str(graph_db_path))
    conn = kuzu.Connection(db)

    stats = {"edges_scanned": 0, "objects_migrated": 0, "spans_migrated": 0, "skipped": 0}

    # Query all edges that have a supporting_evidence column
    try:
        result = conn.execute(
            "MATCH ()-[e:TRIPLE]->() RETURN e.predicate, e.subject, e.object, "
            "e.supporting_evidence, e.provenance_object_id LIMIT 100000"
        )
        rows = result.get_as_df() if hasattr(result, "get_as_df") else []
    except Exception as exc:
        logger.warning("Could not query TRIPLE edges: %s", exc)
        return stats

    with ProvenanceStore(prov_db_path) as store:
        for _, row in (rows.iterrows() if hasattr(rows, "iterrows") else []):
            stats["edges_scanned"] += 1
            evidence_json = row.get("e.supporting_evidence")
            object_id = row.get("e.provenance_object_id")

            if not evidence_json or not object_id:
                stats["skipped"] += 1
                continue

            try:
                evidence = json.loads(evidence_json) if isinstance(evidence_json, str) else evidence_json
            except (json.JSONDecodeError, TypeError):
                stats["skipped"] += 1
                continue

            # Check if already migrated
            existing = store.get_provenance(object_id)
            if existing is not None:
                stats["skipped"] += 1
                continue

            if dry_run:
                stats["objects_migrated"] += 1
                continue

            # Build ProvenanceDoc list from legacy evidence structure
            docs = _evidence_to_docs(evidence)
            span_count = sum(len(d.spans) for d in docs)

            try:
                store.create_provenance(
                    object_id=object_id,
                    object_type="kg_edge",
                    docs=docs,
                )
                stats["objects_migrated"] += 1
                stats["spans_migrated"] += span_count
            except Exception as exc:
                logger.debug("Migration failed for %s: %s", object_id, exc)
                stats["skipped"] += 1

    logger.info(
        "Migration complete: %d edges scanned, %d objects migrated, "
        "%d spans migrated, %d skipped",
        stats["edges_scanned"],
        stats["objects_migrated"],
        stats["spans_migrated"],
        stats["skipped"],
    )
    return stats


def _evidence_to_docs(evidence: Any) -> List[Any]:
    """Convert legacy supporting_evidence JSON to ProvenanceDoc list."""
    from spindle.provenance.models import EvidenceSpan, ProvenanceDoc

    docs = []
    if isinstance(evidence, list):
        items = evidence
    elif isinstance(evidence, dict):
        items = [evidence]
    else:
        return docs

    for item in items:
        if not isinstance(item, dict):
            continue
        doc_id = item.get("source_id") or item.get("doc_id") or item.get("source_name", "unknown")
        text = item.get("text") or item.get("supporting_text", "")
        start = item.get("start_offset") or item.get("start")
        end = item.get("end_offset") or item.get("end")

        span = EvidenceSpan(
            text=str(text),
            start_offset=int(start) if start is not None else None,
            end_offset=int(end) if end is not None else None,
        )
        docs.append(ProvenanceDoc(doc_id=str(doc_id), spans=[span]))

    return docs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Migrate Spindle v1 JSON provenance blobs to SQLite ProvenanceStore"
    )
    parser.add_argument("--graph-db", required=True, help="Path to Kùzu database directory")
    parser.add_argument("--prov-db", required=True, help="Path to target provenance.db")
    parser.add_argument("--dry-run", action="store_true", help="Scan without writing")
    args = parser.parse_args()

    stats = migrate_provenance(
        graph_db_path=args.graph_db,
        prov_db_path=args.prov_db,
        dry_run=args.dry_run,
    )
    print(stats)


if __name__ == "__main__":
    _main()
