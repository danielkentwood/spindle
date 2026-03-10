"""Staging and review workflow for KOS candidates.

Candidates discovered by the extraction pipeline (cold start or discovery
pass) are written to staging Turtle files, reviewed by a human, and then
merged into the canonical kos.ttls.

Staging layout
~~~~~~~~~~~~~~
kos/staging/
    vocabulary.ttls   — skos:Concept triples
    taxonomy.ttls     — skos:broader / skos:narrower triples
    thesaurus.ttls    — skos:related triples
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from spindle.kos.service import KOSService


_SPINDLE_NS = "http://spindle.dev/ns/"
_SKOS_NS = "http://www.w3.org/2004/02/skos/core#"
_DCT_NS = "http://purl.org/dc/terms/"


def write_staging(
    candidates: List[Dict[str, Any]],
    stage_dir: Path,
) -> None:
    """Append new concept candidates to the staging vocabulary file.

    Each candidate dict should have at minimum:
      - text / pref_label  — preferred label string
      - confidence         — float (optional)
      - definition         — string (optional)

    Args:
        candidates: List of candidate dicts from the extraction pipeline.
        stage_dir: Path to ``kos/staging/`` directory.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = stage_dir / "vocabulary.ttls"

    lines: List[str] = []
    if not vocab_path.exists() or vocab_path.stat().st_size == 0:
        lines += [
            "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
            "@prefix dct:  <http://purl.org/dc/terms/> .",
            f"@prefix spndl: <{_SPINDLE_NS}> .",
            "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .",
            "",
        ]

    for candidate in candidates:
        label = _safe_label(candidate.get("text") or candidate.get("pref_label", ""))
        if not label:
            continue
        slug = _slugify(label)
        uri = f"{_SPINDLE_NS}staging/concept/{slug}"
        concept_id = slug
        confidence = candidate.get("confidence", 0.5)
        definition = _escape_turtle_string(candidate.get("definition", ""))
        now = datetime.now(timezone.utc).isoformat()

        lines.append(f"<{uri}> a skos:Concept ;")
        lines.append(f'    skos:prefLabel "{_escape_turtle_string(label)}"@en ;')
        lines.append(f'    dct:identifier "{concept_id}" ;')
        if definition:
            lines.append(f'    skos:definition "{definition}"@en ;')
        lines.append(f'    <{_SPINDLE_NS}extractionConfidence> "{confidence}"^^xsd:float ;')
        lines.append(f'    <{_SPINDLE_NS}stagedAt> "{now}"^^xsd:dateTime .')
        lines.append("")

    if lines:
        with vocab_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def merge_staging(
    stage_dir: Path,
    kos_service: "KOSService",
) -> Dict[str, Any]:
    """Atomically merge accepted staging entries into kos.ttls.

    Reads the staging vocabulary file, calls ``KOSService.create_concept``
    for each entry, runs SKOS validation, and clears the staging file if
    all entries are valid.

    Args:
        stage_dir: Path to ``kos/staging/`` directory.
        kos_service: Live KOSService to update.

    Returns:
        Summary dict with counts.
    """
    vocab_path = stage_dir / "vocabulary.ttls"
    if not vocab_path.exists():
        return {"merged": 0, "skipped": 0, "violations": []}

    concepts = _parse_staging_vocabulary(vocab_path)
    merged = 0
    skipped = 0

    for concept in concepts:
        try:
            result = kos_service.create_concept(
                pref_label=concept["pref_label"],
                definition=concept.get("definition"),
                alt_labels=concept.get("alt_labels"),
            )
            if result is not None:
                merged += 1
            else:
                skipped += 1
        except Exception:
            skipped += 1

    violations = kos_service.validate_skos()
    return {"merged": merged, "skipped": skipped, "violations": violations}


def reject_candidate(
    term: str,
    doc_id: str,
    kos_service: "KOSService",
    reason: Optional[str] = None,
    rejected_by: Optional[str] = None,
    chunk_index: Optional[int] = None,
) -> None:
    """Log a rejected staging candidate to the rejection log.

    Args:
        term: The rejected term text.
        doc_id: Source document identifier.
        kos_service: KOSService to access the rejection log.
        reason: Human-readable rejection reason.
        rejected_by: Reviewer identifier.
        chunk_index: Chunk index within the document.
    """
    kos_service._rejection_log.add(
        term=term,
        source_doc_id=doc_id,
        chunk_index=chunk_index,
        rejection_reason=reason,
        rejected_by=rejected_by,
    )


def clear_staging_file(stage_dir: Path, filename: str = "vocabulary.ttls") -> None:
    """Truncate a staging file after successful merge."""
    path = stage_dir / filename
    if path.exists():
        path.write_text("", encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(label: str) -> str:
    label = label.lower().strip()
    label = re.sub(r"[^\w\s-]", "", label)
    label = re.sub(r"[\s_]+", "-", label)
    label = re.sub(r"-+", "-", label).strip("-")
    return label or "concept"


def _safe_label(text: str) -> str:
    return text.strip()


def _escape_turtle_string(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _parse_staging_vocabulary(path: Path) -> List[Dict[str, Any]]:
    """Minimal parser for the staging vocabulary file.

    Extracts prefLabel and optional definition from the Turtle-like file.
    """
    text = path.read_text(encoding="utf-8")
    concepts: List[Dict[str, Any]] = []

    # Match skos:prefLabel lines
    for m in re.finditer(r'skos:prefLabel\s+"([^"]+)"', text):
        label = m.group(1)
        # Try to find paired definition
        defn_match = re.search(
            r'dct:identifier\s+"' + re.escape(_slugify(label)) + r'"[^.]*skos:definition\s+"([^"]+)"',
            text,
        )
        concepts.append({
            "pref_label": label,
            "definition": defn_match.group(1) if defn_match else None,
        })

    return concepts
