"""KOS Extraction Pipeline orchestrator.

Two paths:
- **Cold start**: no kos.ttls exists yet — runs LLM-based vocab/taxonomy/
  thesaurus extraction via the existing spindle.pipeline stages, then writes
  to staging for human review.
- **Incremental**: existing KOS is loaded — runs three-pass NER cascade
  (Aho-Corasick → multistep resolution → GLiNER2 discovery) and writes
  novel candidates to staging.

Both paths write provenance records to the ProvenanceStore.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from spindle.kos.service import KOSService
    from spindle.preprocessing.models import Chunk
    from spindle.provenance.store import ProvenanceStore

from spindle.kos.ner import discovery_pass, fast_pass, medium_pass
from spindle.kos.staging import write_staging

logger = logging.getLogger(__name__)


class KOSExtractionPipeline:
    """Orchestrate KOS extraction from preprocessed document chunks.

    Args:
        kos_service: Loaded KOSService (incremental) or an empty one
                     (cold start, no kos.ttls present).
        stage_dir: Path to ``kos/staging/`` directory for new candidates.
        provenance_store: Optional ProvenanceStore for evidence recording.
        tracker: Optional tracker for metrics emission.
    """

    def __init__(
        self,
        kos_service: "KOSService",
        stage_dir: Optional[Path] = None,
        provenance_store: Optional["ProvenanceStore"] = None,
        tracker: Optional[Any] = None,
    ) -> None:
        self._kos = kos_service
        self._stage_dir = stage_dir or (kos_service._kos_dir / "staging")
        self._prov = provenance_store
        self._tracker = tracker or _noop_tracker()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        chunks: List["Chunk"],
        mode: str = "auto",
    ) -> Dict[str, Any]:
        """Run the extraction pipeline on a list of preprocessed chunks.

        Args:
            chunks: Chunks from ``SpindlePreprocessor``.
            mode: ``"cold_start"`` | ``"incremental"`` | ``"auto"``.
                  ``"auto"`` selects cold_start when the KOS is empty.

        Returns:
            Summary dict with counts per pass.
        """
        if mode == "auto":
            mode = "cold_start" if self._kos.stats()["concepts"] == 0 else "incremental"

        if mode == "cold_start":
            return self._cold_start(chunks)
        return self._incremental(chunks)

    # ------------------------------------------------------------------
    # Cold start
    # ------------------------------------------------------------------

    def _cold_start(self, chunks: List["Chunk"]) -> Dict[str, Any]:
        """LLM-based extraction for new KOS bootstrapping."""
        logger.info("KOS cold start: %d chunks", len(chunks))

        candidates: List[Dict[str, Any]] = []

        # Try to use the existing pipeline vocabulary/taxonomy stages
        try:
            from spindle.kos.extraction_llm import extract_candidates_via_llm
            candidates = extract_candidates_via_llm(chunks)
        except (ImportError, Exception) as exc:
            logger.warning("LLM extraction unavailable (%s); using heuristic fallback", exc)
            candidates = _heuristic_candidates(chunks)

        write_staging(candidates, self._stage_dir)
        self._emit("kos/cold_start.complete", {"candidates_staged": len(candidates)})

        return {
            "mode": "cold_start",
            "candidates_staged": len(candidates),
            "chunks_processed": len(chunks),
        }

    # ------------------------------------------------------------------
    # Incremental (three-pass cascade)
    # ------------------------------------------------------------------

    def _incremental(self, chunks: List["Chunk"]) -> Dict[str, Any]:
        """Three-pass NER cascade for incremental KOS maintenance."""
        logger.info("KOS incremental: %d chunks", len(chunks))

        # Pass 1 — Aho-Corasick
        matched_fast, unmatched = fast_pass(chunks, self._kos)
        self._write_provenance(matched_fast)
        self._emit("kos/pass1.complete", {"matched": len(matched_fast), "unmatched": len(unmatched)})

        # Pass 2 — multistep resolution
        matched_medium, still_unmatched = medium_pass(unmatched, self._kos)
        self._write_provenance(matched_medium)
        self._emit("kos/pass2.complete", {"matched": len(matched_medium), "unmatched": len(still_unmatched)})

        # Pass 3 — GLiNER2 discovery
        novel, gliner_matched = discovery_pass(still_unmatched, self._kos)
        self._write_provenance(gliner_matched)
        if novel:
            write_staging(novel, self._stage_dir)
        self._emit("kos/pass3.complete", {"novel": len(novel), "matched": len(gliner_matched)})

        return {
            "mode": "incremental",
            "pass1_matched": len(matched_fast),
            "pass2_matched": len(matched_medium),
            "pass3_novel": len(novel),
            "pass3_matched": len(gliner_matched),
            "chunks_processed": len(chunks),
        }

    # ------------------------------------------------------------------
    # Provenance helper
    # ------------------------------------------------------------------

    def _write_provenance(self, resolved: List[Dict[str, Any]]) -> None:
        """Write evidence spans for resolved mentions to ProvenanceStore."""
        if self._prov is None or not resolved:
            return

        from spindle.preprocessing.offsets import to_document_offset

        for item in resolved:
            chunk = item.get("chunk")
            if chunk is None or item.get("concept_uri") is None:
                continue
            concept_uri = item["concept_uri"]
            span_start = item.get("start") or 0
            span_end = item.get("end") or len(item.get("text", ""))
            doc_offset_start, doc_offset_end = to_document_offset(chunk, span_start, span_end)

            from spindle.provenance.models import ProvenanceDoc, EvidenceSpan
            try:
                self._prov.create_provenance(
                    object_id=concept_uri,
                    object_type="vocab_entry",
                    docs=[
                        ProvenanceDoc(
                            doc_id=chunk.source_id,
                            spans=[
                                EvidenceSpan(
                                    text=item.get("text", ""),
                                    start_offset=doc_offset_start,
                                    end_offset=doc_offset_end,
                                )
                            ],
                        )
                    ],
                )
            except Exception as exc:
                logger.debug("Provenance write failed for %s: %s", concept_uri, exc)

    # ------------------------------------------------------------------
    # Tracker helper
    # ------------------------------------------------------------------

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        try:
            self._tracker.log_event(event, data)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _noop_tracker() -> Any:
    try:
        from spindle.tracking import NoOpTracker
        return NoOpTracker()
    except ImportError:
        class _Noop:
            def log_event(self, *a: Any, **kw: Any) -> None:
                pass
        return _Noop()


def _heuristic_candidates(chunks: List["Chunk"]) -> List[Dict[str, Any]]:
    """Simple fallback: extract capitalised multi-word phrases as candidates."""
    import re
    seen: set = set()
    candidates: List[Dict[str, Any]] = []
    for chunk in chunks:
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", chunk.text):
            phrase = m.group(1)
            if phrase not in seen:
                seen.add(phrase)
                candidates.append({"text": phrase, "confidence": 0.5})
    return candidates
