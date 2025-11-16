"""Dynamic LLM pricing lookup and cost estimation.

This module fetches pricing information from official provider pages at runtime
and exposes a helper to estimate request costs when provider SDKs do not report
them. We avoid hard-coding numbers; if a model is unknown or prices cannot be
retrieved, the estimator returns None.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import requests


_LOCK = threading.Lock()


@dataclass(frozen=True)
class ModelPrice:
    """Price per 1M tokens, separately for input and output."""

    input_per_million: float
    output_per_million: float


def _fetch_anthropic_prices() -> Dict[str, ModelPrice]:
    """Fetch Anthropic pricing from the official pricing page.

    Returns a mapping with normalized keys like:
    - 'claude-3-5-haiku'
    - 'claude-3-5-sonnet'
    - 'claude-3-5-opus'
    Key format is lowercase with spaces/dots replaced by hyphens.
    """
    url = "https://www.anthropic.com/pricing"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception:
        return {}

    text = resp.text
    # Very loose regex to capture "Claude 3.5 Haiku" style headers and two prices
    # around "per 1M tokens" (or "per million tokens") for input and output.
    #
    # We search in sections to reduce false positives.
    model_patterns = [
        ("claude-3-5-haiku", r"Claude\s*3\.5\s*Haiku"),
        ("claude-3-5-sonnet", r"Claude\s*3\.5\s*Sonnet"),
        ("claude-3-5-opus", r"Claude\s*3\.5\s*Opus"),
        ("claude-3-haiku", r"Claude\s*3\s*Haiku"),
        ("claude-3-sonnet", r"Claude\s*3\s*Sonnet"),
        ("claude-3-opus", r"Claude\s*3\s*Opus"),
    ]

    prices: Dict[str, ModelPrice] = {}
    for key, pat in model_patterns:
        # Find a slice of the text around the model header
        for match in re.finditer(pat, text, flags=re.IGNORECASE):
            start = max(0, match.start() - 1000)
            end = min(len(text), match.end() + 2000)
            blob = text[start:end]
            # Find two dollar amounts close to "per 1M" or "per million"
            # Try to capture Input first then Output. If ambiguous, we pick the
            # first two occurrences.
            dollar_re = re.compile(
                r"\$([0-9]+(?:\.[0-9]+)?)\s*/?\s*(?:per\s*(?:1M|million)\s*tokens|/M)", re.IGNORECASE
            )
            dollars = [float(m.group(1)) for m in dollar_re.finditer(blob)]
            if len(dollars) >= 2:
                prices[key] = ModelPrice(input_per_million=dollars[0], output_per_million=dollars[1])
                break
    return prices


def _fetch_openai_prices() -> Dict[str, ModelPrice]:
    """Fetch OpenAI pricing from the official pricing page.

    We only record models we can confidently parse. Unknown/future models will
    not be included in the map.
    """
    url = "https://openai.com/api/pricing"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception:
        return {}

    text = resp.text
    # Capture known 4o/mini style; 5-series may not be listed (leave absent).
    # Examples we try to detect:
    # "gpt-4o mini" or "GPT‑4o mini" with two $X per 1M tokens values near it.
    candidates = [
        ("gpt-4o-mini", r"GPT[\-\u2011\u2013\u2014]?4o\s*mini|gpt[\-\u2011\u2013\u2014]?4o\s*mini"),
        ("gpt-4o", r"GPT[\-\u2011\u2013\u2014]?4o(?!\s*mini)|gpt[\-\u2011\u2013\u2014]?4o(?!\s*mini)"),
    ]
    prices: Dict[str, ModelPrice] = {}
    for key, pat in candidates:
        for match in re.finditer(pat, text, flags=re.IGNORECASE):
            start = max(0, match.start() - 1000)
            end = min(len(text), match.end() + 2000)
            blob = text[start:end]
            dollar_re = re.compile(
                r"\$([0-9]+(?:\.[0-9]+)?)\s*/?\s*(?:per\s*(?:1M|million)\s*tokens|/M)", re.IGNORECASE
            )
            dollars = [float(m.group(1)) for m in dollar_re.finditer(blob)]
            if len(dollars) >= 2:
                prices[key] = ModelPrice(input_per_million=dollars[0], output_per_million=dollars[1])
                break
    return prices


def _normalize_model_id(model: str) -> str:
    """Normalize model identifiers to aid matching.
    
    Strips provider-specific date suffixes and version strings to map
    to base model pricing.
    """
    m = model.strip().lower()
    # Replace spaces with hyphens
    m = m.replace(" ", "-")
    
    # Strip Anthropic date suffixes like "-20241022"
    m = re.sub(r"-20\d{6}$", "", m)
    
    # Strip OpenAI date suffixes like "-2025-08-07"
    m = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", m)
    
    return m


_CACHED: Optional[Dict[str, ModelPrice]] = None

# Fallback pricing for common models (used if web scraping fails)
# Prices per 1M tokens, based on official provider pricing as of Nov 2024
_FALLBACK_PRICES: Dict[str, ModelPrice] = {
    # OpenAI models
    "gpt-4o": ModelPrice(input_per_million=2.5, output_per_million=10.0),
    "gpt-4o-mini": ModelPrice(input_per_million=0.15, output_per_million=0.60),
    "gpt-5": ModelPrice(input_per_million=1.25, output_per_million=10.0),
    "gpt-5-mini": ModelPrice(input_per_million=0.25, output_per_million=2.0),
    "gpt-5-nano": ModelPrice(input_per_million=0.05, output_per_million=0.40),
    
    # Anthropic models
    "claude-3-5-haiku": ModelPrice(input_per_million=0.80, output_per_million=4.0),
    "claude-3-5-sonnet": ModelPrice(input_per_million=3.0, output_per_million=15.0),
    "claude-3-haiku": ModelPrice(input_per_million=0.25, output_per_million=1.25),
    "claude-3-sonnet": ModelPrice(input_per_million=3.0, output_per_million=15.0),
    "claude-3-opus": ModelPrice(input_per_million=15.0, output_per_million=75.0),
    "claude-sonnet-4": ModelPrice(input_per_million=3.0, output_per_million=15.0),
    "claude-opus-4": ModelPrice(input_per_million=15.0, output_per_million=75.0),
}


def _load_prices() -> Dict[str, ModelPrice]:
    global _CACHED
    with _LOCK:
        if _CACHED is not None:
            return _CACHED
        
        # Start with fallback prices
        merged: Dict[str, ModelPrice] = dict(_FALLBACK_PRICES)
        
        # Try to fetch from web (will override fallbacks if successful)
        try:
            merged.update(_fetch_anthropic_prices())
        except Exception:
            pass
        try:
            merged.update(_fetch_openai_prices())
        except Exception:
            pass
        
        _CACHED = merged
        return merged


def get_model_price(model: Optional[str]) -> Optional[ModelPrice]:
    """Return ModelPrice for a model id, or None if unknown.

    The lookup tries the normalized id and some simple aliases.
    """
    if not model:
        return None
    prices = _load_prices()
    key = _normalize_model_id(model)
    if key in prices:
        return prices[key]
    # Simple aliasing for Anthropic names with/without dots/hyphens
    alias = key.replace(".", "-")
    return prices.get(alias)


def estimate_cost_usd(model: Optional[str], input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
    """Estimate total USD cost given a model and token counts, using fetched prices.

    Returns None if no pricing is available.
    """
    price = get_model_price(model)
    if price is None:
        return None
    it = (input_tokens or 0)
    ot = (output_tokens or 0)
    total = (it / 1_000_000.0) * price.input_per_million + (ot / 1_000_000.0) * price.output_per_million
    # Keep a sensible precision
    return round(total, 6)


__all__ = ["ModelPrice", "get_model_price", "estimate_cost_usd"]


