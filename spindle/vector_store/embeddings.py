"""
Embedding function factories for various providers.

This module provides factory functions for creating embedding functions from
different providers (OpenAI, Hugging Face, Google Gemini) and a convenience
function to get the best available embedding function.
"""

import os
from typing import Callable, List, Optional

# Try to import API clients for embeddings
try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

try:
    from huggingface_hub import InferenceClient
    _HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    _HUGGINGFACE_HUB_AVAILABLE = False

try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

from spindle.observability import get_event_recorder

VECTOR_STORE_RECORDER = get_event_recorder("vector_store")


def _record_vector_event(name: str, payload: dict) -> None:
    """Record a vector store event."""
    VECTOR_STORE_RECORDER.record(name=name, payload=payload)


def create_openai_embedding_function(
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None
) -> Callable[[str], List[float]]:
    """
    Create an embedding function using OpenAI's API.
    
    Args:
        model: OpenAI embedding model name (default: "text-embedding-3-small")
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
    
    Returns:
        Callable function that takes text and returns embedding vector
    
    Raises:
        ImportError: If openai package is not installed
        ValueError: If API key is not provided
    """
    if not _OPENAI_AVAILABLE:
        raise ImportError(
            "OpenAI package is required for API embeddings. "
            "Install it with: pip install openai"
        )
    
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key required. Set OPENAI_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    client = openai.OpenAI(api_key=api_key)
    
    def embed(text: str) -> List[float]:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    
    return embed


def create_huggingface_embedding_function(
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    api_key: Optional[str] = None
) -> Callable[[str], List[float]]:
    """
    Create an embedding function using Hugging Face Inference API.
    
    Prefers InferenceClient from huggingface_hub if available, falls back to direct API.
    
    Args:
        model: Hugging Face model name
        api_key: Hugging Face API key (defaults to HF_API_KEY env var)
    
    Returns:
        Callable function that takes text and returns embedding vector
    
    Raises:
        ImportError: If neither huggingface_hub nor requests package is installed
        ValueError: If API key is not provided
    """
    api_key = api_key or os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise ValueError(
            "Hugging Face API key required. Set HF_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    # Try InferenceClient first (preferred method)
    if _HUGGINGFACE_HUB_AVAILABLE:
        client = InferenceClient(token=api_key)
        
        def embed(text: str) -> List[float]:
            result = client.feature_extraction(text, model=model)
            # Handle different response formats
            if isinstance(result, list):
                embedding = result[0] if len(result) > 0 else result
            elif hasattr(result, 'tolist'):
                embedding = result.tolist()
            else:
                embedding = result
            # Ensure we return a list of floats
            if isinstance(embedding, (list, tuple)):
                return list(embedding)
            return embedding
        
        return embed
    
    # Fallback to direct API if huggingface_hub not available
    if not _REQUESTS_AVAILABLE:
        raise ImportError(
            "For Hugging Face API embeddings, either huggingface_hub or requests package is required. "
            "Install with: pip install huggingface_hub (preferred) or pip install requests"
        )
    
    # Use the new Hugging Face Inference API endpoint
    # Old endpoint (deprecated): https://api-inference.huggingface.co/pipeline/feature-extraction/{model}
    # New endpoint: https://api-inference.huggingface.co/models/{model}
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    def embed(text: str) -> List[float]:
        response = requests.post(api_url, headers=headers, json={"inputs": text}, timeout=30)
        response.raise_for_status()
        result = response.json()
        # Handle different response formats
        if isinstance(result, list):
            return result[0] if len(result) > 0 else result
        elif isinstance(result, dict) and "embeddings" in result:
            return result["embeddings"][0] if result["embeddings"] else []
        else:
            return result
    
    return embed


def create_gemini_embedding_function(
    model: str = "models/embedding-001",
    api_key: Optional[str] = None,
    task_type: str = "retrieval_document"
) -> Callable[[str], List[float]]:
    """
    Create an embedding function using Google Gemini's API.
    
    Args:
        model: Gemini embedding model name (default: "models/embedding-001")
        api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
        task_type: Task type for embedding (default: "retrieval_document")
                  Options: "retrieval_document", "retrieval_query", "semantic_similarity", etc.
    
    Returns:
        Callable function that takes text and returns embedding vector
    
    Raises:
        ImportError: If google-generativeai package is not installed
        ValueError: If API key is not provided
    """
    if not _GEMINI_AVAILABLE:
        raise ImportError(
            "google-generativeai package is required for Gemini API embeddings. "
            "Install it with: pip install google-generativeai"
        )
    
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key required. Set GEMINI_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    genai.configure(api_key=api_key)
    
    def embed(text: str) -> List[float]:
        embedding = genai.embed_content(
            model=model,
            content=text,
            task_type=task_type
        )
        return embedding['embedding']
    
    return embed


def get_default_embedding_function(
    prefer_local: bool = True,
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    hf_api_key: Optional[str] = None
) -> Optional[Callable[[str], List[float]]]:
    """
    Get the best available embedding function, with fallback priority:
    1. sentence-transformers (if available and prefer_local=True)
    2. OpenAI API (if API key available)
    3. Gemini API (if API key available)
    4. Hugging Face API (if API key available)
    5. None (if nothing available)
    
    Args:
        prefer_local: If True, prefer local sentence-transformers over API
        openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        gemini_api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
        hf_api_key: Hugging Face API key (defaults to HF_API_KEY env var)
    
    Returns:
        Embedding function or None if no options available
    """
    # Try local first if preferred
    if prefer_local and _SENTENCE_TRANSFORMERS_AVAILABLE:
        return None  # Will use SentenceTransformer in ChromaVectorStore
    
    # Try OpenAI API
    openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if openai_key and _OPENAI_AVAILABLE:
        try:
            return create_openai_embedding_function(api_key=openai_key)
        except Exception:
            pass
    
    # Try Gemini API
    gemini_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
    if gemini_key and _GEMINI_AVAILABLE:
        try:
            return create_gemini_embedding_function(api_key=gemini_key)
        except Exception:
            pass
    
    # Try Hugging Face API
    hf_key = hf_api_key or os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
    if hf_key and (_HUGGINGFACE_HUB_AVAILABLE or _REQUESTS_AVAILABLE):
        try:
            return create_huggingface_embedding_function(api_key=hf_key)
        except Exception:
            pass
    
    return None

