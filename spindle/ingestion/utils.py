"""Utility helpers shared across ingestion modules."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, TypeVar


T = TypeVar("T")


def import_from_string(path: str) -> Any:
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid import path '{path}'")
    module = importlib.import_module(module_path)
    try:
        return getattr(module, attr)
    except AttributeError as exc:  # pragma: no cover - defensive programming
        raise ImportError(f"Cannot import '{attr}' from '{module_path}'") from exc


def ensure_callable(obj: str | Callable[..., T]) -> Callable[..., T]:
    if callable(obj):
        return obj
    if not isinstance(obj, str):
        raise TypeError(f"Expected callable or dotted path, got {type(obj)!r}")
    return import_from_string(obj)

