"""
Internal helper functions for extraction operations.

This module contains internal utilities for:
- Event recording
- Span processing and index computation
- String utilities
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import baml_py
from spindle.baml_client.types import (
    CharacterSpan,
)
from spindle.observability import get_event_recorder

EXTRACTOR_RECORDER = get_event_recorder("extractor")


def _record_extractor_event(name: str, payload: Dict[str, Any]) -> None:
    EXTRACTOR_RECORDER.record(name=name, payload=payload)


def _extract_model_from_collector(collector: baml_py.baml_py.Collector) -> Optional[str]:
    """Extract actual model name from BAML collector logs.

    Extracts the real provider model (e.g., 'gpt-5-mini-2025-08-07') from the
    HTTP response, not the BAML client name (e.g., 'CustomFast').

    Returns:
        Model identifier string, or None if not found
    """
    import json

    if not hasattr(collector, 'logs') or not collector.logs:
        return None

    for log in collector.logs:
        if not hasattr(log, 'selected_call'):
            continue

        selected_call = log.selected_call

        # Extract model from HTTP response body (provider's actual response)
        if hasattr(selected_call, 'http_response') and hasattr(selected_call.http_response, 'body'):
            body = selected_call.http_response.body

            # Try body.text() method (most common)
            if hasattr(body, 'text'):
                text_val = body.text
                try:
                    text_content = text_val() if callable(text_val) else text_val
                    if isinstance(text_content, str):
                        body_dict = json.loads(text_content)
                        if 'model' in body_dict:
                            model = body_dict['model']
                            if model and not str(model).startswith('Custom'):
                                return model
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass

            # Try body.json() method as fallback
            if hasattr(body, 'json'):
                json_val = body.json
                if callable(json_val):
                    try:
                        parsed = json_val()
                        if isinstance(parsed, dict) and 'model' in parsed:
                            model = parsed['model']
                            if model and not str(model).startswith('Custom'):
                                return model
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        pass

    return None


def _extract_metrics_from_collector(collector: baml_py.baml_py.Collector) -> Dict[str, Any]:
    """Extract token and cost metrics from BAML collector.

    Args:
        collector: BAML collector instance

    Returns:
        Dictionary with:
        - total_tokens: int (input + output tokens)
        - input_tokens: int | None
        - output_tokens: int | None
        - total_cost: float (always 0.0; computed by pricing fallback if needed)
    """
    agg_input_tokens = 0
    agg_output_tokens = 0
    agg_total_cost = 0.0

    if hasattr(collector, 'logs') and collector.logs:
        for log in collector.logs:
            # Extract from log.usage (BAML's aggregated usage object)
            if hasattr(log, 'usage'):
                usage = log.usage
                if hasattr(usage, 'input_tokens'):
                    agg_input_tokens += getattr(usage, 'input_tokens', 0) or 0
                if hasattr(usage, 'output_tokens'):
                    agg_output_tokens += getattr(usage, 'output_tokens', 0) or 0

            # Check for cost (rarely provided by provider SDKs)
            if hasattr(log, 'total_cost'):
                cost = getattr(log, 'total_cost', None)
                if cost is not None:
                    agg_total_cost += cost

    # Compute total from input + output
    total_tokens = agg_input_tokens + agg_output_tokens

    return {
        "total_tokens": total_tokens,
        "input_tokens": agg_input_tokens if agg_input_tokens > 0 else None,
        "output_tokens": agg_output_tokens if agg_output_tokens > 0 else None,
        "total_cost": agg_total_cost,
    }


def _normalize_ws(text: str) -> str:
    """Normalize whitespace: collapse all whitespace to single spaces."""
    return re.sub(r'\s+', ' ', text.strip())


def _normalize_loose_with_map(text: str) -> Tuple[str, List[int]]:
    """
    Build a punctuation-tolerant normalized string and index map.

    The normalized form:
    - lowercases all characters
    - keeps alphanumeric chars
    - treats punctuation/whitespace as separators collapsed to one space

    Returns:
        A tuple of (normalized_text, index_map) where index_map[i] is the original
        source index for normalized_text[i].
    """
    normalized_chars: List[str] = []
    index_map: List[int] = []
    previous_was_space = True

    for idx, char in enumerate(text):
        if char.isalnum():
            normalized_chars.append(char.lower())
            index_map.append(idx)
            previous_was_space = False
            continue

        if not previous_was_space:
            normalized_chars.append(" ")
            index_map.append(idx)
            previous_was_space = True

    while normalized_chars and normalized_chars[-1] == " ":
        normalized_chars.pop()
        index_map.pop()

    return "".join(normalized_chars), index_map


def _find_span_indices_loose(source_text: str, span_text: str) -> Optional[Tuple[int, int]]:
    """Find span indices using punctuation-tolerant matching."""
    normalized_source, source_map = _normalize_loose_with_map(source_text)
    normalized_span, _ = _normalize_loose_with_map(span_text)

    if not normalized_span or not normalized_source:
        return None

    start = normalized_source.find(normalized_span)
    if start == -1:
        return None

    end = start + len(normalized_span) - 1
    if start >= len(source_map) or end >= len(source_map):
        return None

    return source_map[start], source_map[end] + 1


def _find_span_indices(source_text: str, span_text: str, normalized_source: Optional[str] = None) -> Optional[Tuple[int, int]]:
    """
    Find the start and end indices of span_text within source_text.

    Uses multiple strategies in order:
    1. Exact substring match
    2. Whitespace-normalized fuzzy match (handles newlines, extra spaces)
    3. Case-insensitive fuzzy match
    4. Case-insensitive exact match
    5. Punctuation-tolerant normalized match

    Args:
        source_text: The full source text
        span_text: The text span to find
        normalized_source: Pre-computed normalized source text (optional, for performance)

    Returns:
        Tuple of (start, end) indices, or None if not found
    """
    # Strategy 1: Try exact match first (fast path)
    start = source_text.find(span_text)
    if start != -1:
        return (start, start + len(span_text))

    # Strategy 2: Fuzzy match with whitespace normalization
    words = span_text.split()
    if len(words) > 0:
        pattern_parts = [re.escape(word) for word in words]
        pattern = r'\s+'.join(pattern_parts)

        try:
            match = re.search(pattern, source_text, re.DOTALL)
            if match:
                return (match.start(), match.end())
        except re.error:
            pass

    # Strategy 3: Case-insensitive fuzzy match
    if len(words) > 0:
        pattern_parts = [re.escape(word) for word in words]
        pattern = r'\s+'.join(pattern_parts)
        try:
            match = re.search(pattern, source_text, re.DOTALL | re.IGNORECASE)
            if match:
                return (match.start(), match.end())
        except re.error:
            pass

    # Strategy 4: Case-insensitive exact match
    lower_source = source_text.lower()
    lower_span = span_text.lower()
    start = lower_source.find(lower_span)
    if start != -1:
        return (start, start + len(span_text))

    # Strategy 5: Punctuation-tolerant fallback
    loose_match = _find_span_indices_loose(source_text, span_text)
    if loose_match:
        return loose_match

    # Could not find the span
    return None


def _compute_all_span_indices(source_text: str, spans: List[CharacterSpan]) -> List[CharacterSpan]:
    """
    Compute character indices for all spans efficiently using batch processing.

    Args:
        source_text: The full source text
        spans: List of CharacterSpan objects that need index computation

    Returns:
        List of CharacterSpan objects with computed indices
    """
    # Pre-compute normalized source text once for all spans
    normalized_source = _normalize_ws(source_text)

    updated_spans = []
    for span in spans:
        if span.start is not None and span.end is not None:
            # Span already has indices, keep it as-is
            updated_spans.append(span)
        else:
            # Find the span text in the source text
            indices = _find_span_indices(source_text, span.text, normalized_source)
            if indices:
                updated_span = CharacterSpan(
                    text=span.text,
                    start=indices[0],
                    end=indices[1]
                )
                updated_spans.append(updated_span)
            else:
                updated_span = CharacterSpan(
                    text=span.text,
                    start=-1,
                    end=-1
                )
                updated_spans.append(updated_span)

    return updated_spans
