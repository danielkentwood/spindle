"""Utility functions for the API."""

import json
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Union

from fastapi import UploadFile


async def save_uploaded_file(upload: UploadFile) -> Path:
    """Save an uploaded file to a temporary location.
    
    Args:
        upload: FastAPI UploadFile object
        
    Returns:
        Path to saved temporary file
    """
    suffix = Path(upload.filename).suffix if upload.filename else ""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    
    try:
        content = await upload.read()
        temp_file.write(content)
        temp_file.flush()
        return Path(temp_file.name)
    finally:
        temp_file.close()


def convert_baml_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert BAML Pydantic models to JSON-serializable dictionaries.
    
    Args:
        obj: BAML Pydantic model instance
        
    Returns:
        Dictionary representation
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif hasattr(obj, "dict"):
        return obj.dict()
    elif isinstance(obj, dict):
        return obj
    else:
        # Fallback to JSON serialization
        return json.loads(json.dumps(obj, default=str))


def serialize_extraction_result(result: Any) -> Dict[str, Any]:
    """Serialize an extraction result to a dictionary.
    
    Args:
        result: ExtractionResult from BAML
        
    Returns:
        Serialized dictionary
    """
    data = convert_baml_to_dict(result)
    
    # Process triples to ensure they're serializable
    if "triples" in data:
        serialized_triples = []
        for triple in data["triples"]:
            if hasattr(triple, "model_dump"):
                serialized_triples.append(triple.model_dump())
            elif hasattr(triple, "dict"):
                serialized_triples.append(triple.dict())
            else:
                serialized_triples.append(triple)
        data["triples"] = serialized_triples
    
    return data


def serialize_ontology(ontology: Any) -> Dict[str, Any]:
    """Serialize an ontology to a dictionary.
    
    Args:
        ontology: Ontology from BAML
        
    Returns:
        Serialized dictionary
    """
    return convert_baml_to_dict(ontology)


def serialize_process_graph(graph: Any) -> Dict[str, Any]:
    """Serialize a process graph to a dictionary.
    
    Args:
        graph: ProcessGraph from BAML
        
    Returns:
        Serialized dictionary
    """
    return convert_baml_to_dict(graph)


async def sse_generator(iterator: AsyncIterator[Any]) -> AsyncIterator[str]:
    """Generate Server-Sent Events from an async iterator.
    
    Args:
        iterator: Async iterator yielding data objects
        
    Yields:
        Formatted SSE strings
    """
    async for item in iterator:
        if hasattr(item, "model_dump"):
            data = item.model_dump()
        elif hasattr(item, "dict"):
            data = item.dict()
        else:
            data = item
        
        # Format as SSE
        json_data = json.dumps(data, default=str)
        yield f"data: {json_data}\n\n"


def create_temp_directory() -> Path:
    """Create a temporary directory for file operations.
    
    Returns:
        Path to temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix="spindle_api_")
    return Path(temp_dir)


def ensure_path_exists(path: Union[str, Path]) -> Path:
    """Ensure a path exists and return it as a Path object.
    
    Args:
        path: File or directory path
        
    Returns:
        Path object
        
    Raises:
        FileNotFoundError: If path doesn't exist
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return p


def create_parent_directories(path: Union[str, Path]) -> Path:
    """Create parent directories for a path if they don't exist.
    
    Args:
        path: File path
        
    Returns:
        Path object
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

