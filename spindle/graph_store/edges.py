"""
Edge operation utilities for graph stores.

This module provides backend-agnostic utilities for edge operations,
including evidence merging logic.
"""

from typing import Any, Dict, List, Tuple


def merge_evidence(
    existing_evidence: List[Dict[str, Any]],
    new_source_nm: str,
    new_source_url: str,
    new_spans: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Merge new evidence into existing evidence with deduplication.
    
    Deduplication Rules:
    - Same source + same span text → Skip, return message
    - Same source + different span → Add span to existing source
    - New source → Add new source entry
    
    Args:
        existing_evidence: List of existing source evidence dicts
        new_source_nm: New source name
        new_source_url: New source URL
        new_spans: List of new span dicts with 'text', 'start', 'end', 'extraction_datetime'
    
    Returns:
        Tuple of (merged_evidence_list, status_message)
    """
    # Find if source already exists
    source_idx = None
    for idx, evidence in enumerate(existing_evidence):
        if evidence.get("source_nm") == new_source_nm:
            source_idx = idx
            break
    
    # If source doesn't exist, add it as new
    if source_idx is None:
        existing_evidence.append({
            "source_nm": new_source_nm,
            "source_url": new_source_url,
            "spans": new_spans
        })
        return (existing_evidence, f"Added new source: {new_source_nm}")
    
    # Source exists - check spans for duplicates
    existing_spans = existing_evidence[source_idx]["spans"]
    existing_span_texts = {span["text"] for span in existing_spans}
    
    added_count = 0
    duplicate_count = 0
    
    for new_span in new_spans:
        if new_span["text"] in existing_span_texts:
            duplicate_count += 1
        else:
            existing_spans.append(new_span)
            existing_span_texts.add(new_span["text"])
            added_count += 1
    
    # Update the source's spans
    existing_evidence[source_idx]["spans"] = existing_spans
    
    # Create status message
    if duplicate_count > 0 and added_count == 0:
        return (existing_evidence, f"All spans already exist for source: {new_source_nm}")
    elif added_count > 0 and duplicate_count > 0:
        return (existing_evidence, f"Added {added_count} new span(s) to source: {new_source_nm} ({duplicate_count} duplicate(s) skipped)")
    else:
        return (existing_evidence, f"Added {added_count} new span(s) to source: {new_source_nm}")

